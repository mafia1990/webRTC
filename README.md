# WebRTC Desktop Capture + AI Upscale

This demo captures the Windows desktop server-side using `mss`, upscales frames with ESRGAN ONNX model, and streams them via WebRTC to the browser.

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python server_desktop.py --port 8080
```
Then open: http://localhost:8080/
