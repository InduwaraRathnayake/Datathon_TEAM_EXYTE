#!/usr/bin/env python3
"""
Anonymize images: faces (YOLOv8), plates (YOLOv8, optional), and text (optional via EasyOCR).

Defaults look for your .pt models in your OS "Documents/models" folder:
  - Face:  any *.pt with "face" in the filename (first match)
  - Plate: any *.pt with "plate" in the filename (first match)

You can override with --face_model and --plate_model.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
import numpy as np
from tqdm import tqdm

# ---------- helpers ----------
def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

def expand_box(x1, y1, x2, y2, w, h, margin=0.08):  # tighter default margin
    bw, bh = (x2 - x1), (y2 - y1)
    mx, my = int(bw * margin), int(bh * margin)
    return clamp_box(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)

def blur_region(img, box, ksize=17):  # smaller blur kernel
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    ksize = int(ksize) | 1  # make odd
    img[y1:y2, x1:x2] = cv.GaussianBlur(roi, (ksize, ksize), 0)
    return img

def pixelate_region(img, box, downscale=10):
    x1, y1, x2, y2 = box
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img
    h, w = roi.shape[:2]
    small = cv.resize(roi, (max(1, w // downscale), max(1, h // downscale)), interpolation=cv.INTER_LINEAR)
    pix = cv.resize(small, (w, h), interpolation=cv.INTER_NEAREST)
    img[y1:y2, x1:x2] = pix
    return img

def redact(img, boxes, mode="blur", margin=0.08, blur_ksize=17, pixel_downscale=10):
    H, W = img.shape[:2]
    out = img.copy()
    for (x1, y1, x2, y2, _) in boxes:
        X1, Y1, X2, Y2 = expand_box(x1, y1, x2, y2, W, H, margin)
        if mode == "blur":
            out = blur_region(out, (X1, Y1, X2, Y2), ksize=blur_ksize)
        else:
            out = pixelate_region(out, (X1, Y1, X2, Y2), downscale=pixel_downscale)
    return out

# ---------- detectors ----------
class YOLOBoxDetector:
    """Generic YOLOv8 detector wrapper (faces or plates)."""
    def __init__(self, weights_path: str, conf=0.25, iou=0.5, imgsz=768, device=None):
        from ultralytics import YOLO
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device  # 0, 'cuda:0', or 'cpu'/None

    def detect(self, img: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        res = self.model.predict(source=img, conf=self.conf, iou=self.iou,
                                 imgsz=self.imgsz, device=self.device, verbose=False)[0]
        boxes = []
        H, W = img.shape[:2]
        if getattr(res, "boxes", None) is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf[0].cpu().numpy())
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2, conf))
        return boxes

# ---------- text (optional) ----------
class TextDetectorEasyOCR:
    def __init__(self, lang_list=None, gpu=False, min_conf=0.3):
        import easyocr
        self.reader = easyocr.Reader(lang_list or ["en"], gpu=gpu)
        self.min_conf = float(min_conf)
    def detect(self, img: np.ndarray, min_box=12) -> List[Tuple[int, int, int, int, float]]:
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb, detail=1, paragraph=False)
        out = []
        H, W = img.shape[:2]
        for (box, text, conf) in results:
            c = float(conf) if conf is not None else 0.0
            if c < self.min_conf:
                continue
            xs = [int(p[0]) for p in box]; ys = [int(p[1]) for p in box]
            x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(W-1, max(xs)), min(H-1, max(ys))
            if (x2 - x1) >= min_box and (y2 - y1) >= min_box:
                out.append((x1, y1, x2, y2, c))
        return out

# ---------- utils ----------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def default_documents_models():
    """Try to find face/plate models inside ~/Documents/models (Windows/macOS/Linux)."""
    docs = Path.home() / "Documents" / "models"
    face = plate = None
    if docs.exists():
        pts = list(docs.rglob("*.pt"))
        # pick the first matching by name hint, otherwise first .pt found
        for p in pts:
            n = p.name.lower()
            if "face" in n and face is None:
                face = str(p)
            if "plate" in n and plate is None:
                plate = str(p)
        if face is None and pts:
            face = str(pts[0])
        if plate is None and len(pts) > 1:
            plate = str([p for p in pts if str(p) != face][0])
    return face, plate

# ---------- main ----------
def process_dir(
    in_dir: str,
    out_dir: str,
    # face
    use_face: int = 1,
    face_model: str = "",
    face_conf: float = 0.25,
    face_iou: float = 0.5,
    face_imgsz: int = 768,
    face_mode: str = "blur",
    face_margin: float = 0.08,
    # plate
    use_plate: int = 0,
    plate_model: str = "",
    plate_conf: float = 0.25,
    plate_iou: float = 0.5,
    plate_imgsz: int = 768,
    plate_mode: str = "pixel",
    plate_margin: float = 0.08,
    # text
    use_text: int = 0,
    text_langs: str = "en",
    text_minbox: int = 12,
    text_mode: str = "pixel",
    text_margin: float = 0.08,
    text_min_conf: float = 0.3,
    # strengths
    blur_ksize: int = 17,
    pixel_downscale: int = 10,
    # device
    device: str = None
) -> None:

    in_dir = Path(in_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # auto-discover models in Documents if not provided
    auto_face, auto_plate = default_documents_models()
    if use_face and not face_model:
        if auto_face:
            face_model = auto_face
            print(f"[info] Using face model: {face_model}")
        else:
            raise ValueError("--face_model not provided and no face model found in ~/Documents/models")
    if use_plate and not plate_model:
        if auto_plate:
            plate_model = auto_plate
            print(f"[info] Using plate model: {plate_model}")
        else:
            raise ValueError("--plate_model not provided and no plate model found in ~/Documents/models")

    face = YOLOBoxDetector(face_model, conf=face_conf, iou=face_iou, imgsz=face_imgsz, device=device) if use_face else None
    plate = YOLOBoxDetector(plate_model, conf=plate_conf, iou=plate_iou, imgsz=plate_imgsz, device=device) if use_plate else None
    text = TextDetectorEasyOCR(lang_list=[s.strip() for s in text_langs.split(",")], gpu=(device not in (None, "cpu")), min_conf=text_min_conf) if use_text else None

    images = [p for p in in_dir.rglob("*") if p.is_file() and is_image(p)]
    if not images:
        print(f"[warn] No images found under: {in_dir}")
        return

    for src in tqdm(images, desc="Anonymizing"):
        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = cv.imread(str(src))
        if img is None:
            print(f"[warn] Could not read: {src}")
            continue

        face_boxes = face.detect(img) if face else []
        text_boxes = text.detect(img, min_box=text_minbox) if text else []
        plate_boxes = plate.detect(img) if plate else []

        work = img.copy()
        if face:
            work = redact(work, face_boxes, mode=face_mode, margin=face_margin,
                          blur_ksize=blur_ksize, pixel_downscale=pixel_downscale)
        if text:
            work = redact(work, text_boxes, mode=text_mode, margin=text_margin,
                          blur_ksize=blur_ksize, pixel_downscale=pixel_downscale)
        if plate:
            work = redact(work, plate_boxes, mode=plate_mode, margin=plate_margin,
                          blur_ksize=blur_ksize, pixel_downscale=pixel_downscale)

        ok = cv.imwrite(str(dst), work)
        if not ok:
            print(f"[warn] Failed to save: {dst}")

def parse_args():
    ap = argparse.ArgumentParser(description="Anonymize faces/plates/text in images using YOLOv8 detectors.")
    ap.add_argument("--in_dir", required=True, help="Input folder with images")
    ap.add_argument("--out_dir", required=True, help="Output folder for anonymized images")

    ap.add_argument("--use_face", type=int, default=1)
    ap.add_argument("--face_model", type=str, default="", help=".pt path for face model (YOLOv8)")
    ap.add_argument("--face_conf", type=float, default=0.25)
    ap.add_argument("--face_iou", type=float, default=0.5)
    ap.add_argument("--face_imgsz", type=int, default=768)
    ap.add_argument("--face_mode", choices=["blur", "pixel"], default="blur")
    ap.add_argument("--face_margin", type=float, default=0.08)

    ap.add_argument("--use_plate", type=int, default=0)
    ap.add_argument("--plate_model", type=str, default="", help=".pt path for plate model (YOLOv8)")
    ap.add_argument("--plate_conf", type=float, default=0.25)
    ap.add_argument("--plate_iou", type=float, default=0.5)
    ap.add_argument("--plate_imgsz", type=int, default=768)
    ap.add_argument("--plate_mode", choices=["blur", "pixel"], default="pixel")
    ap.add_argument("--plate_margin", type=float, default=0.08)

    ap.add_argument("--use_text", type=int, default=0)
    ap.add_argument("--text_langs", type=str, default="en")
    ap.add_argument("--text_minbox", type=int, default=12)
    ap.add_argument("--text_mode", choices=["blur", "pixel"], default="pixel")
    ap.add_argument("--text_margin", type=float, default=0.08)
    ap.add_argument("--text_min_conf", type=float, default=0.3)

    ap.add_argument("--blur_ksize", type=int, default=17)
    ap.add_argument("--pixel_downscale", type=int, default=10)

    ap.add_argument("--device", type=str, default=None, help="0 / 'cuda:0' for GPU, 'cpu' for CPU")

    return ap.parse_args()

def main():
    args = parse_args()
    process_dir(
        in_dir=args.in_dir, out_dir=args.out_dir,
        use_face=args.use_face, face_model=args.face_model, face_conf=args.face_conf,
        face_iou=args.face_iou, face_imgsz=args.face_imgsz, face_mode=args.face_mode,
        face_margin=args.face_margin,
        use_plate=args.use_plate, plate_model=args.plate_model, plate_conf=args.plate_conf,
        plate_iou=args.plate_iou, plate_imgsz=args.plate_imgsz, plate_mode=args.plate_mode,
        plate_margin=args.plate_margin,
        use_text=args.use_text, text_langs=args.text_langs, text_minbox=args.text_minbox,
        text_mode=args.text_mode, text_margin=args.text_margin, text_min_conf=args.text_min_conf,
        blur_ksize=args.blur_ksize, pixel_downscale=args.pixel_downscale,
        device=args.device
    )

if __name__ == "__main__":
    main()
