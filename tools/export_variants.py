"""Step 3: export lighter variants to test what buys back latency on the phones.

Written from the Ultralytics docs; NOT run by the author. Run it in the folder that holds
labels.txt (the 207-class vocabulary from export_models.py) and the .pt weights.

Produces (copy the .tflite files into android/app/src/main/assets/):
  yoloe-26s-det.tflite       detection-only YOLOE-26s (mask branch removed), same 207 classes
  yoloe-26n-det.tflite       detection-only YOLOE-26n
  yolo26s-depth-640.tflite   depth-s at 640 instead of 768 (~0.69x the pixels)
  yolo26n-depth-768.tflite   depth-n at 768 (skipped with a message if the weights don't exist)

Detection-only recipe is from the YOLOE docs (Export Usage): build the model from the .yaml,
load the seg weights, load the saved prompt embeddings.
"""
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO, YOLOE

IMG = "bus.jpg"
vocab = [ln.strip() for ln in Path("labels.txt").read_text().splitlines() if ln.strip()]
print(f"vocabulary: {len(vocab)} classes")


def export_yoloe_detect(scale: str) -> None:
    npz = f"vocab-{scale}.npz"
    seg = YOLOE(f"yoloe-{scale}-seg.pt")
    seg.set_classes(vocab)
    seg.save_prompt_embeddings(npz)

    det = YOLOE(f"yoloe-{scale}.yaml").load(f"yoloe-{scale}-seg.pt")
    det.load_prompt_embeddings(npz)
    path = det.export(format="litert", imgsz=640, nms=None)
    out = f"yoloe-{scale}-det.tflite"
    shutil.copy(path, out)
    print(f"-> {out}")


def export_depth(weights: str, imgsz: int, out: str) -> None:
    path = YOLO(weights).export(format="litert", imgsz=imgsz)
    shutil.copy(path, out)
    print(f"-> {out}")


def depth_sensitivity() -> None:
    """How much does running depth-s at 640 change the answer vs 768? (Not accuracy vs ground truth.)"""
    m = YOLO("yolo26s-depth.pt")
    d768 = m(IMG, imgsz=768, verbose=False)[0].depth.data.cpu().numpy()
    d640 = m(IMG, imgsz=640, verbose=False)[0].depth.data.cpu().numpy()
    rel = np.abs(d640 - d768) / np.maximum(d768, 1e-3)
    print(f"depth-s 640 vs 768 on {IMG}: median relative difference {np.median(rel):.2%}, "
          f"90th percentile {np.percentile(rel, 90):.2%}")


for scale in ("26s", "26n"):
    try:
        export_yoloe_detect(scale)
    except Exception as e:
        print(f"YOLOE {scale} detect-only export failed: {type(e).__name__}: {e}")

try:
    export_depth("yolo26s-depth.pt", 640, "yolo26s-depth-640.tflite")
    depth_sensitivity()
except Exception as e:
    print(f"depth-s 640 failed: {type(e).__name__}: {e}")

try:
    export_depth("yolo26n-depth.pt", 768, "yolo26n-depth-768.tflite")
except Exception as e:
    print(f"depth-n 768 skipped: {type(e).__name__}: {e}")
