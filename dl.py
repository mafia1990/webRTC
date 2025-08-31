from huggingface_hub import hf_hub_download

# مثال: دانلود مدل x4plus
path = hf_hub_download(
    repo_id="qualcomm/Real-ESRGAN-x4plus",
    filename="Real-ESRGAN-x4plus.onnx"
)
print("Downloaded to", path)
