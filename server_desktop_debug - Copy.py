# server_desktop_debug_fixed.py
import argparse
import asyncio
import json
import os
import ssl
import logging
import fractions
import time
from aiortc import RTCRtpTransceiver

import av
import numpy as np
import mss
import cv2 
import dxcam

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.rtcrtpsender import RTCRtpSender

from upscaler import ESRGANUpscaler

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
for name in ["aiortc", "aioice", "av", "ice"]:
    logging.getLogger(name).setLevel(logging.DEBUG)
log = logging.getLogger("server")

def sdp_summary(sdp: str, label: str):
    lines = [l for l in sdp.splitlines() if l.startswith(("m=", "a=send", "a=recv", "a=mid", "a=rtpmap"))]
    log.debug("----- %s SDP summary -----\n%s\n---------------------------", label, "\n".join(lines[:120]))

class DesktopCaptureTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, monitor_index=0, fps=30):
        super().__init__()
        self.fps = fps
        self.frame_time = 1 / fps

        # تنظیم dxcam
        self.camera = dxcam.create(output_idx=monitor_index, max_buffer_len=8)
        if self.camera is None:
            raise RuntimeError("Unable to create DXCAM camera. Is DirectX 11/12 available?")

        # شروع کپچر
        self.camera.start(target_fps=fps, video_mode=True)
        self.counter = 0

        log.info("DXCAM DesktopCapture initialized for monitor %d at %d FPS", monitor_index, fps)

    async def recv(self):
        await asyncio.sleep(self.frame_time)

        # گرفتن فریم از GPU (BGR)
        frame = self.camera.get_latest_frame()
        if frame is None:
            log.warning("DXCAM: No frame received, using blank")
            frame = np.zeros((720, 1024, 3), dtype=np.uint8)

        h, w = frame.shape[:2]

        # تبدیل به av.VideoFrame
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame.pts = self.counter
        av_frame.time_base = fractions.Fraction(1, self.fps)
        self.counter += 1

        if self.counter % (self.fps * 5) == 0:
            log.debug("DXCAM Frame #%d size=%dx%d", self.counter, w, h)

        return av_frame

    def stop(self):
        if self.camera:
            self.camera.stop()
        super().stop()
class DownscaleTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack, target_width=640, target_height=360):
        super().__init__()
        self.track = track
        self.tw = target_width
        self.th = target_height
        self.counter = 0

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # ↓↓ Downscale روی سرور
        down = cv2.resize(img, (self.tw, self.th), interpolation=cv2.INTER_AREA)

        new_frame = av.VideoFrame.from_ndarray(down, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        self.counter += 1
        if self.counter % 60 == 0:
            log.debug("Downscaled frame #%d: %dx%d -> %dx%d",
                      self.counter, img.shape[1], img.shape[0], self.tw, self.th)

        return new_frame
class VideoUpscaleTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, track: MediaStreamTrack, upscaler: ESRGANUpscaler):
        super().__init__()
        self.track = track
        self.upscaler = upscaler
        self.counter = 0

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # 🔎 فقط اگر upscaler فعال باشه upscale کن
        if self.upscaler and getattr(self.upscaler, "enabled", False):
            # بهتره upscale رو توی thread جدا انجام بدیم تا event-loop قفل نشه
            up = await asyncio.to_thread(self.upscaler.upscale, img)
        else:
            # بدون تغییر (passthrough)
            up = img  

        new_frame = av.VideoFrame.from_ndarray(up, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        self.counter += 1
        if self.counter % 60 == 0:
            log.debug(
                "Video frame #%d: %dx%d -> %dx%d (upscaler %s)",
                self.counter,
                img.shape[1], img.shape[0],
                up.shape[1], up.shape[0],
                "enabled" if getattr(self.upscaler, "enabled", False) else "disabled"
            )
        return new_frame


ROOT = os.path.join(os.path.dirname(__file__), "client")
pcs = set()

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r", encoding="utf-8").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    remote_ip = request.remote
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    log.info("Offer from %s, len=%d", remote_ip, len(offer.sdp))
    sdp_summary(offer.sdp, "REMOTE (offer)")

    config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)
    log.info("Created PC; total PCs: %d", len(pcs))

    scale = int(request.query.get("scale", "2"))
    onnx_path = request.query.get("model", r"C:\\Users\\AmirAli\\Documents\\streaming\\files\\webrtc_ai_upscale_demo\\models\\Real-ESRGAN-x4plusx4plus.onnx")
    provider = request.query.get("provider", "CUDA")
    up = ESRGANUpscaler(onnx_path=onnx_path, scale=scale, provider=provider)
    log.info("Upscaler enabled=%s, model_exists=%s, provider=%s",
             getattr(up, "enabled", False), os.path.isfile(onnx_path), provider)

    @pc.on("icegatheringstatechange")
    def on_icegatheringstatechange():
        log.debug("ICE gathering state: %s", pc.iceGatheringState)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log.info("ICE connection state: %s", pc.iceConnectionState)
        if pc.iceConnectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log.info("PC connection state: %s", pc.connectionState)
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)

    # ✅ فقط یک بار setRemoteDescription
    await pc.setRemoteDescription(offer)

    # ساخت track
    desktop = DesktopCaptureTrack(monitor_index=0, fps=60)  # ⚠️ fps=150 خیلی زیاده!
    local_video = VideoUpscaleTrack(desktop, up)

    # ✅ اضافه کردن track (sender ایجاد می‌شه)
    video_sender = pc.addTrack(local_video)
    log.info("Track added via addTrack(); sender created")

    # ✅ تنظیم کُدِک بعد از setRemoteDescription و قبل از createAnswer
    for transceiver in pc.getTransceivers():
        if transceiver.sender == video_sender:

            log.info("Found video transceiver, setting codec preferences...")
            caps = RTCRtpSender.getCapabilities("video")
            log.debug("Available codecs: %s", caps.codecs)
            h264_codecs = [
                codec for codec in caps.codecs
                if codec.mimeType.lower() == "video/h264"
                and codec.parameters.get("packetization-mode") == "1"
                and codec.parameters.get("profile-level-id") in ["42e01f", "42001f"]
            ]
            if h264_codecs:
                transceiver.setCodecPreferences(h264_codecs)
                log.info("✅ H.264 selected as preferred codec")
            else:
                log.warning("❌ H.264 NOT available in capabilities. Using default (likely VP8)")
            break

    # ✅ ساخت پاسخ
    answer = await pc.createAnswer()
    sdp_summary(answer.sdp, "LOCAL (answer)")
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )
async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def main():
    parser = argparse.ArgumentParser(description="WebRTC Desktop + AI Upscale (DEBUG, fixed)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cert")
    parser.add_argument("--key")
    args = parser.parse_args()

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_static("/client", ROOT)

    ssl_context = None
    if args.cert and args.key:
        ssl_context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.cert, args.key)

    log.info("Starting server on %s:%d", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)

if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
