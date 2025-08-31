import os
import cv2   # 👈 اینجا global import
import numpy as np

try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

class ESRGANUpscaler:
    def __init__(self, onnx_path: str | None = None, scale: int = 2, provider: str | None = None):
        self.scale = int(scale)
        self.session = None
        self.input_name = None
        self.output_name = None
        self.enabled = False

        if onnx_path and os.path.isfile(onnx_path) and _HAS_ORT:
            providers = [provider] if provider else None
            try:
                self.session = ort.InferenceSession(
                    onnx_path,
                    providers=providers or ort.get_available_providers()
                )
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.enabled = True
            except Exception as e:
                print(f"[Upscaler] Failed to load ONNX model: {e}. Falling back to bicubic.")
        else:
            if not _HAS_ORT:
                print("[Upscaler] onnxruntime not available. Using bicubic fallback.")
            else:
                print("[Upscaler] ONNX model path missing. Using bicubic fallback.")

    def upscale(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return frame_bgr

        # اگر مدل فعال نباشه → fallback
        if not self.enabled or self.session is None:
            h, w = frame_bgr.shape[:2]
            return cv2.resize(frame_bgr, (w * self.scale, h * self.scale),
                              interpolation=cv2.INTER_CUBIC)

        # اگر مدل فعال باشه → ESRGAN اجرا می‌شه
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out = self.session.run([self.output_name], {self.input_name: inp})[0]
        out = np.clip(out[0].transpose(1, 2, 0), 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
