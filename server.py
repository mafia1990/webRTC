import argparse
import asyncio
import json
import os
import ssl
import uuid

import av
import numpy as np
import mss

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole

from upscaler import ESRGANUpscaler


# ---- Desktop Capture Track ----
class DesktopCaptureTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, monitor_index=1, fps=30):
        super().__init__()  
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]  # monitor 1 = full desktop
        self.fps = fps
        self.frame_time = 1 / fps

    async def recv(self):
        await asyncio.sleep(self.frame_time)
        img = np.array(self.sct.grab(self.monitor))  # BGRA
        frame = av.VideoFrame.from_ndarray(img[..., :3], format="bgr24")
        frame.pts, frame.time_base = None, None
        return frame


# ---- Video Upscale Track ----
class VideoUpscaleTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track: MediaStreamTrack, upscaler: ESRGANUpscaler):
        super().__init__()
        self.track = track
        self.upscaler = upscaler

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        up = self.upscaler.upscale(img)
        new_frame = av.VideoFrame.from_ndarray(up, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


# ---- Web server ----
ROOT = os.path.join(os.path.dirname(__file__), "client")
pcs = set()


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r", encoding="utf-8").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaBlackhole()

    # Upscaler setup
    scale = int(request.query.get("scale", "2"))
    onnx_path = request.query.get("model", None)
    provider = request.query.get("provider", None)
    up = ESRGANUpscaler(onnx_path=onnx_path, scale=scale, provider=provider)

    # Desktop track
    desktop = DesktopCaptureTrack(fps=20)
    local_video = VideoUpscaleTrack(desktop, up)
    pc.addTrack(local_video)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE state:", pc.iceConnectionState)
        if pc.iceConnectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def main():
    parser = argparse.ArgumentParser(description="WebRTC Desktop + AI Upscale")
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

    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
