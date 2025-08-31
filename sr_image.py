import os
import re
import cv2
import argparse
import numpy as np
from pathlib import Path

def mild_artifact_reduction(img_bgr):
    # کاهش نویز/بلوکینگ ملایم
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 15)

def mild_unsharp(img_bgr):
    # شارپن ملایم بدون هاله
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.2)
    sharp = cv2.addWeighted(img_bgr, 1.15, blur, -0.15, 0)
    return sharp

def detect_scale_from_filename(model_path, fallback_scale):
    m = re.search(r"[xX]\s*(\d+)", os.path.basename(model_path))
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return fallback_scale

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"❌ نتوانستم تصویر را باز کنم: {path}")
    # اگر آلفا داشت، به BGR تبدیل می‌کنیم (کانال آلفا را نگه نمی‌داریم تا SR ساده بماند)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # تصاویر خاکستری را به BGR ببریم برای سازگاری با SR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def ensure_even(img):
    # بعضی مدل‌ها/فیلترها با عرض/ارتفاع فرد مشکل دارند؛ برش یک پیکسل از لبه اگر فرد بود
    h, w = img.shape[:2]
    nh = h - (h % 2)
    nw = w - (w % 2)
    if (nh, nw) != (h, w):
        img = img[:nh, :nw]
    return img

def make_side_by_side(items, pad=12):
    # items: لیستی از تصاویر هم‌ارتفاع؛ کنار هم می‌چینیم
    heights = [im.shape[0] for im in items]
    target_h = min(heights)
    resized = [cv2.resize(im, (int(im.shape[1]*target_h/im.shape[0]), target_h)) for im in items]
    # پَد بین ستون‌ها
    pads = [np.full((target_h, pad, 3), 32, dtype=np.uint8) for _ in range(len(resized)-1)]
    # ترکیب
    out = resized[0]
    for i in range(1, len(resized)):
        out = np.hstack([out, pads[i-1], resized[i]])
    return out

def add_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return out

def main():
    parser = argparse.ArgumentParser(
        description="سوپررزولوشن عکس با OpenCV DNN SuperRes (ESPCN/FSRCNN) + خروجی مقایسه‌ای",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", required=True, help="مسیر تصویر ورودی")
    parser.add_argument("--output-dir", "-o", default="outputs", help="پوشه‌ی ذخیره خروجی‌ها")
    parser.add_argument("--model-path", default="ESPCN_x2.pb", help="مسیر مدل .pb (ESPCN/FSRCNN)")
    parser.add_argument("--model-name", default="espcn", choices=["espcn", "fsrcnn"], help="نام مدل")
    parser.add_argument("--scale", type=int, default=2, help="مقیاس آپ‌اسکیل اگر از نام مدل تشخیص نشود")
    parser.add_argument("--pre-denoise", action="store_true", help="کاهش نویز ملایم قبل از SR")
    parser.add_argument("--post-sharpen", action="store_true", help="شارپن ملایم بعد از SR")
    parser.add_argument("--prefer-opencl", action="store_true", help="تلاش برای OpenCL (اگر در دسترس باشد)")
    parser.add_argument("--no-side", action="store_true", help="عدم ساخت خروجی ساید-بای-ساید")
    args = parser.parse_args()

    inp_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # خواندن تصویر
    img = load_image(str(inp_path))
    img = ensure_even(img)

    # تعیین مقیاس
    SCALE = detect_scale_from_filename(args.model_path, args.scale)
    print(f"[INFO] SCALE = x{SCALE}")

    # پیش‌پردازش اختیاری
    lr = img.copy()
    if args.pre_denoise:
        lr = mild_artifact_reduction(lr)
        print("[INFO] pre-denoise فعال است.")

    # مسیر‌ها
    stem = inp_path.stem
    out_sr_path = out_dir / f"{stem}_SR_x{SCALE}.png"
    out_bi_path = out_dir / f"{stem}_Bicubic_x{SCALE}.png"
    out_side_path = out_dir / f"{stem}_COMPARE_x{SCALE}.jpg"

    # Bicubic
    bi = cv2.resize(lr, (lr.shape[1]*SCALE, lr.shape[0]*SCALE), interpolation=cv2.INTER_CUBIC)

    # SR
    use_sr = os.path.exists(args.model_path)
    backend_name = "CPU"
    sr_img = None

    if use_sr:
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(args.model_path)
            sr.setModel(args.model_name, SCALE)

            if args.prefer_opencl:
                try:
                    cv2.ocl.setUseOpenCL(True)
                    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                    backend_name = "OpenCL"
                except Exception:
                    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    backend_name = "CPU"
            else:
                sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            print(f"[INFO] مدل {args.model_name.upper()} بارگذاری شد. Backend: {backend_name}")
            sr_img = sr.upsample(lr)
        except Exception as e:
            print(f"[WARN] خطا در SR: {e}\n[WARN] از Bicubic استفاده می‌شود.")
            use_sr = False

    if sr_img is None:
        sr_img = bi.copy()  # fallback

    # شارپن پس‌پردازش اختیاری
    if args.post_sharpen:
        sr_img = mild_unsharp(sr_img)
        print("[INFO] post-sharpen فعال است.")

    # ذخیره خروجی‌ها
    cv2.imwrite(str(out_bi_path), bi)
    cv2.imwrite(str(out_sr_path), sr_img)

    print(f"[OK] ذخیره شد: {out_bi_path}")
    print(f"[OK] ذخیره شد: {out_sr_path}")

    # خروجی مقایسه‌ای
    if not args.no_side:
        lr_lab = add_label(lr, "Input (LR)")
        bi_lab = add_label(bi, f"Bicubic x{SCALE}")
        sr_lab = add_label(sr_img, f"{args.model_name.upper()} x{SCALE}" if use_sr else f"SR (Fallback=Bicubic) x{SCALE}")
        side = make_side_by_side([lr_lab, bi_lab, sr_lab], pad=16)
        cv2.imwrite(str(out_side_path), side, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"[OK] ذخیره شد: {out_side_path}")

    print("🎉 تمام شد.")

if __name__ == "__main__":
    main()
