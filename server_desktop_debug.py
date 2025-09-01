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
from wincam import DXCamera
import av
import numpy as np
import cv2 
import win32api
from pynput.keyboard import Controller as KeyController, Key
from pynput.mouse import Controller as MouseController, Button
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.rtcrtpsender import RTCRtpSender
import sounddevice as sd
import av
from aiortc import MediaStreamTrack
from upscaler import ESRGANUpscaler
import ctypes
try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom as cupy_zoom
    _CUPY_OK = True
except Exception:
    _CUPY_OK = False
# ساختار برای mouse_event
SendInput = ctypes.windll.user32.mouse_event

MOUSEEVENTF_MOVE = 0x0001

def move_mouse_relative(dx, dy):
    SendInput(MOUSEEVENTF_MOVE, dx, dy, 0, 0)


keyboard = KeyController()
mouse = MouseController()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
# for name in ["aiortc", "aioice", "av", "ice"]:
    # logging.getLogger(name).setLevel(logging.DEBUG)
log = logging.getLogger("server")
from pynput.keyboard import Key
 
key_map = {
    # اعداد بالا
    "Digit0": "0",
    "Digit1": "1",
    "Digit2": "2",
    "Digit3": "3",
    "Digit4": "4",
    "Digit5": "5",
    "Digit6": "6",
    "Digit7": "7",
    "Digit8": "8",
    "Digit9": "9",

    # کلیدهای حروف (KeyA → a)
    # این‌ها جدا هندل میشن (پایین)

    # کنترل‌ها
    "Space": Key.space,
    "Enter": Key.enter,
    "Tab": Key.tab,
    "Backspace": Key.backspace,
    "Escape": Key.esc,

    "ShiftLeft": Key.shift,
    "ShiftRight": Key.shift,
    "ControlLeft": Key.ctrl,
    "ControlRight": Key.ctrl,
    "AltLeft": Key.alt,
    "AltRight": Key.alt,
    "MetaLeft": Key.cmd,   # ویندوز/⌘
    "MetaRight": Key.cmd,

    # کلیدهای جهت
    "ArrowUp": Key.up,
    "ArrowDown": Key.down,
    "ArrowLeft": Key.left,
    "ArrowRight": Key.right,

    # فانکشن
    "F1": Key.f1, "F2": Key.f2, "F3": Key.f3, "F4": Key.f4,
    "F5": Key.f5, "F6": Key.f6, "F7": Key.f7, "F8": Key.f8,
    "F9": Key.f9, "F10": Key.f10, "F11": Key.f11, "F12": Key.f12,

    # بقیه کلیدهای خاص
    "Insert": Key.insert
}
def translate_key(code: str):
    # اگر توی map بود (کلید خاص)
    if code in key_map:
        return key_map[code]

    # حروف: KeyA → "a"
    if code.startswith("Key") and len(code) == 4:
        return code[-1].lower()

    # اعداد: Digit1 → "1"
    if code.startswith("Digit") and len(code) == 6:
        return code[-1]

    # Numpad: Numpad1 → "1" یا کلید numpad جدا
    if code.startswith("Numpad") and code[6:].isdigit():
        return code[-1]

    return None

def _resize_bilinear_numpy(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Pure NumPy bilinear resize for HxWxC uint8 images (no cv2/PIL).
    """
    in_h, in_w, channels = img.shape
    if (in_w, in_h) == (out_w, out_h):
        return img

    # Create normalized grid in input space
    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)
    x_grid, y_grid = np.meshgrid(x, y)

    x0 = np.floor(x_grid).astype(np.int32)
    y0 = np.floor(y_grid).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)

    wa = (x1 - x_grid) * (y1 - y_grid)
    wb = (x_grid - x0) * (y1 - y_grid)
    wc = (x1 - x_grid) * (y_grid - y0)
    wd = (x_grid - x0) * (y_grid - y0)

    out = np.empty((out_h, out_w, channels), dtype=np.float32)
    for c in range(channels):
        Ia = img[y0, x0, c]
        Ib = img[y0, x1, c]
        Ic = img[y1, x0, c]
        Id = img[y1, x1, c]
        out[..., c] = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return np.clip(out, 0, 255).astype(np.uint8)


def sdp_summary(sdp: str, label: str):
    lines = [l for l in sdp.splitlines() if l.startswith(("m=", "a=send", "a=recv", "a=mid", "a=rtpmap"))]
    log.debug("----- %s SDP summary -----\n%s\n---------------------------", label, "\n".join(lines[:120]))
def handle_input(data):
    t = data.get("type")
    if t in ("keydown", "keyup"):
        key_code = data["key"]
        key = translate_key(key_code)

        if key is None:
            log.warning(f"Unknown key: {key_code}")
            return

        if t == "keydown":
            keyboard.press(key)
        else:
            keyboard.release(key)

    elif t == "mousemove":
        dx, dy = data["x"], data["y"]
        move_mouse_relative(dx, dy)

    elif t == "mousedown":
        if data["button"] == 0: mouse.press(Button.left)
        elif data["button"] == 1: mouse.press(Button.middle)
        elif data["button"] == 2: mouse.press(Button.right)

    elif t == "mouseup":
        if data["button"] == 0: mouse.release(Button.left)
        elif data["button"] == 1: mouse.release(Button.middle)
        elif data["button"] == 2: mouse.release(Button.right)

    elif t == "wheel":
        mouse.scroll(0, -1 if data["delta"]>0 else 1)
class MicrophoneAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, samplerate=48000, channels=1):
        super().__init__()
        device_index = None
        for i, dev in enumerate(sd.query_devices()):
            if "CABLE Output" in dev["name"]:  # یا Stereo Mix
                device_index = i
                break
        print(sd.query_devices())
        self.stream = sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=samplerate,
            dtype="int16",
            blocksize=960
        )
        self.stream.start()
        self.counter = 0
        self.samplerate = samplerate
        self.channels = channels

    async def recv(self):
        frames, _ = self.stream.read(960)   # (960, channels)
        arr = np.array(frames, dtype=np.int16).T  # → (channels, 960)

        layout = "mono" if self.channels == 1 else "stereo"

        frame = av.AudioFrame.from_ndarray(arr, format="s16", layout=layout)
        frame.sample_rate = self.samplerate
        frame.pts = self.counter
        frame.time_base = fractions.Fraction(1, self.samplerate)
        self.counter += frame.samples
        return frame


class DesktopCaptureTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, monitor_index=0, fps=30, out_w=960, out_h=540):
        super().__init__()
        self.fps = fps
        self.frame_interval = 1 / fps
        # صف کوچک برای کم‌کردن لگ
        self.camera = dxcam.create(output_idx=monitor_index, max_buffer_len=1)
        if self.camera is None:
            raise RuntimeError("Unable to create DXCAM camera.")
        self.camera.start(target_fps=fps,include_cursor=True, video_mode=True)
        self.counter = 0
        self.out_w, self.out_h = out_w, out_h
        self._t0 = time.perf_counter()

    async def recv(self):
        # t_expected = self.counter * self.frame_interval
        # t_now = time.perf_counter() - self._t0
        # sleep = t_expected - t_now
        # if sleep > 0:
            # await asyncio.sleep(sleep)

        frame = await asyncio.to_thread(self.camera.get_latest_frame)
        if frame is None:
            frame = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)

        # if (frame.shape[1], frame.shape[0]) != (self.out_w, self.out_h):
            # frame = cv2.resize(frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            # gpu_frame = cv2.cuda_GpuMat()
            # gpu_frame.upload(frame)
            # resized_gpu = cv2.cuda.resize(gpu_frame, (self.out_w, self.out_h))
            # frame = resized_gpu.download()
            # frame = await asyncio.to_thread(cv2.resize, frame, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        # ⬅️ بدون تغییر رنگ، مستقیم BGR
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        av_frame.pts = self.counter
        av_frame.time_base = fractions.Fraction(1, self.fps)

        self.counter += 1
        return av_frame


        



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

    # scale = int(request.query.get("scale", "2"))
    # onnx_path = request.query.get("model", r"C:\\Users\\AmirAli\\Documents\\streaming\\files\\webrtc_ai_upscale_demo\\models\\Real-ESRGAN-x4plusx4plus.onnx")
    # provider = request.query.get("provider", "CUDA")
    # # up = ESRGANUpscaler(onnx_path=onnx_path, scale=scale, provider=provider)
    # log.info("Upscaler enabled=%s, model_exists=%s, provider=%s",
             # getattr(up, "enabled", False), os.path.isfile(onnx_path), provider)

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
    @pc.on("datachannel")
    def on_datachannel(channel):
        log.info("DataChannel created: %s", channel.label)

        @channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                handle_input(data)
            except Exception as e:
                log.error("Input parse error: %s", e)

    # ✅ فقط یک بار setRemoteDescription
    await pc.setRemoteDescription(offer)

    # ساخت track
    desktop = DesktopCaptureTrack( fps=30, out_w=960, out_h=540)  # ⚠️ fps=150 خیلی زیاده!

    # ✅ اضافه کردن track (sender ایجاد می‌شه)
    video_sender = pc.addTrack(desktop)
    audio = MicrophoneAudioTrack()
    pc.addTrack(audio)
    log.info("Track added via addTrack(); sender created")
    
    # ✅ تنظیم کُدِک بعد از setRemoteDescription و قبل از createAnswer
    for transceiver in pc.getTransceivers():
        if transceiver.sender == video_sender:

            log.info("Found video transceiver, setting codec preferences...")
            caps = RTCRtpSender.getCapabilities("video")
            log.info("Available codecs: %s", caps.codecs)
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


    answer = await pc.createAnswer()

    # SDP munging: محدود کردن bitrate به 1500 kbps
    sdp = answer.sdp.replace(
        "a=mid:0",
        "a=mid:0\r\nb=AS:2000\r\nx-google-start-bitrate:100\r\nx-google-max-bitrate:3500"
    )
    answer = RTCSessionDescription(sdp=sdp, type=answer.type)

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
