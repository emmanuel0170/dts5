"""Step 1: export both models to static LiteRT and verify them.

Written from the Ultralytics docs. NOT run by the author -- run it and share the output,
because the Kotlin decoder depends on the printed tensor shapes.

Environment (per the ultralytics Flutter plugin docs): Linux x86 or macOS, Python >= 3.10,
ultralytics >= 8.4.142.
    pip install -U ultralytics ai-edge-litert
Network needed once: set_classes() downloads a ~254 MB text encoder (mobileclip2_b.ts)
into the current directory.

Put one class name per line in vocab.txt to override the starter list. The order of the
lines is the class-index order of the exported model.
"""
from pathlib import Path

import numpy as np
from ultralytics import YOLO, YOLOE

YOLOE_WEIGHTS = "yoloe-26s-seg.pt"
DEPTH_WEIGHTS = "yolo26s-depth.pt"
YOLOE_IMGSZ = 640
DEPTH_IMGSZ = 768  # released depth weights are trained at 768 on stretched (not letterboxed) input
TEST_IMAGE = "https://ultralytics.com/images/bus.jpg"

STARTER_VOCAB = [
    "person", "door", "stairs", "chair", "table", "couch", "bed", "toilet", "sink",
    "bottle", "cup", "laptop", "backpack", "handbag", "umbrella", "keys", "phone",
    "car", "bus", "truck", "bicycle", "motorcycle", "traffic light", "stop sign",
    "bench", "trash can", "pole", "fire hydrant", "dog", "elevator",
]


def load_vocab() -> list[str]:
    f = Path("vocab.txt")
    if f.exists():
        names = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
        if names:
            return names
    return STARTER_VOCAB


def describe(path: str) -> None:
    """Print the real input/output tensors of an exported .tflite."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("  (pip install ai-edge-litert to inspect tensors)")
        return
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    print(f"\n== {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")
    for kind, details in (("in ", it.get_input_details()), ("out", it.get_output_details())):
        for d in details:
            print(f"  {kind} {d['name']}  shape={list(d['shape'])}  "
                  f"dtype={d['dtype'].__name__}  quant={d['quantization']}")


def export_yoloe(vocab: list[str]) -> str:
    model = YOLOE(YOLOE_WEIGHTS)
    model.set_classes(vocab)  # baked into the export; cannot change on device
    path = model.export(format="litert", imgsz=YOLOE_IMGSZ, nms=None)  # nms=None: raw outputs (plugin's setting)
    Path("labels.txt").write_text("\n".join(vocab) + "\n")
    print(f"labels.txt written ({len(vocab)} classes). Index i == line i; confirm below.")
    return str(path)


def export_depth() -> str:
    return str(YOLO(DEPTH_WEIGHTS).export(format="litert", imgsz=DEPTH_IMGSZ))


def check_yoloe(tflite: str, vocab: list[str]) -> None:
    """Compare top detections of the .pt model and the export on one image."""
    try:
        pt = YOLOE(YOLOE_WEIGHTS)
        pt.set_classes(vocab)
        ref = pt.predict(TEST_IMAGE, imgsz=YOLOE_IMGSZ, verbose=False)[0]
        out = YOLO(tflite, task="segment").predict(TEST_IMAGE, imgsz=YOLOE_IMGSZ, verbose=False)[0]
        for tag, r in (("pt    ", ref), ("tflite", out)):
            rows = sorted(zip(r.boxes.conf.tolist(), r.boxes.cls.tolist(), r.boxes.xyxy.tolist()), reverse=True)[:5]
            print(f"  {tag}: " + "; ".join(f"{vocab[int(c)]} {s:.2f} {[round(v) for v in b]}" for s, c, b in rows))
    except Exception as e:  # report, don't crash: the failure itself is information
        print(f"  YOLOE parity check failed: {type(e).__name__}: {e}")


def check_depth(tflite: str) -> None:
    """Does the export keep the metric calibration? Compare depth maps in meters."""
    try:
        ref = YOLO(DEPTH_WEIGHTS)(TEST_IMAGE, imgsz=DEPTH_IMGSZ, verbose=False)[0].depth.data.cpu().numpy()
        out = YOLO(tflite, task="depth")(TEST_IMAGE, imgsz=DEPTH_IMGSZ, verbose=False)[0].depth.data.cpu().numpy()
        print(f"  pt     shape={ref.shape} min/median/max = {ref.min():.2f}/{np.median(ref):.2f}/{ref.max():.2f} m")
        print(f"  tflite shape={out.shape} min/median/max = {out.min():.2f}/{np.median(out):.2f}/{out.max():.2f} m")
        if ref.shape == out.shape:
            rel = np.abs(out - ref) / np.maximum(ref, 1e-3)
            print(f"  median relative difference = {np.median(rel):.3%}")
    except Exception as e:
        print(f"  depth parity check failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    vocab = load_vocab()
    yoloe_path = export_yoloe(vocab)
    depth_path = export_depth()
    describe(yoloe_path)
    describe(depth_path)
    print("\n-- YOLOE parity (class names come from vocab.txt order) --")
    check_yoloe(yoloe_path, vocab)
    print("\n-- Depth parity (meters) --")
    check_depth(depth_path)
