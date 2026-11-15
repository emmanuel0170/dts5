"""Step 2: pin down the exact numerics the Kotlin engine must reproduce.

Re-implements preprocessing, decoding and NMS in plain numpy directly on the exported
.tflite files, then compares the result with Ultralytics' own pipeline on bus.jpg.
Whatever matches here is what the Kotlin code will port 1:1.

NOT run by the author. Run it from the folder holding the .tflite files, labels.txt and bus.jpg.
"""
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ultralytics import YOLO

IMG = "bus.jpg"
YOLOE_TFLITE = "yoloe-26s-seg.tflite"
DEPTH_TFLITE = "yolo26s-depth.tflite"
CONF, IOU = 0.25, 0.7
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def run(path, x):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    return [it.get_tensor(d["index"]) for d in it.get_output_details()]


def stats(name, a):
    print(f"  {name}: shape={a.shape} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")


def to_input(img_bgr, mean=None, std=None):
    """BGR uint8 HWC -> RGB float32 NCHW in [0,1] (optionally mean/std normalised)."""
    x = img_bgr[..., ::-1].astype(np.float32) / 255.0
    if mean is not None:
        x = (x - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, left, top


def iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def fmt(d):
    return f"{d[0]} {d[1]:.2f} {[round(v) for v in d[2]]}"


def yoloe_numpy(img, labels):
    h, w = img.shape[:2]
    lb, r, left, top = letterbox(img, 640)
    outs = run(YOLOE_TFLITE, to_input(lb))
    pred = next(o for o in outs if o.ndim == 3)[0].T  # (8400, C)
    nc = pred.shape[1] - 4 - 32
    print(f"  channels={pred.shape[1]} -> nc={nc} IF layout is [box4, classes, coef32]; "
          f"labels.txt has {len(labels)} lines")
    if nc != len(labels):
        print("  !! MISMATCH: the layout assumption is wrong. Stop and report this.")
    box, cls, coef = pred[:, :4], pred[:, 4:4 + nc], pred[:, 4 + nc:]
    stats("box", box)
    stats("class scores", cls)
    stats("mask coefs", coef)
    scale = 640.0 if box.max() <= 2.0 else 1.0
    print(f"  box units look like: {'NORMALISED (multiply by 640)' if scale == 640.0 else 'pixels of the 640 input'}")
    box = box * scale
    conf, cid = cls.max(1), cls.argmax(1)
    keep = conf > CONF
    box, conf, cid = box[keep], conf[keep], cid[keep]
    if len(conf) == 0:
        return []
    xyxy = np.stack([box[:, 0] - box[:, 2] / 2, box[:, 1] - box[:, 3] / 2,
                     box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1)
    xywh = np.stack([xyxy[:, 0], xyxy[:, 1], box[:, 2], box[:, 3]], 1)
    idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), CONF, IOU)).reshape(-1)  # class-agnostic
    dets = []
    for i in idx:
        x1, y1, x2, y2 = xyxy[i]
        b = [np.clip((x1 - left) / r, 0, w), np.clip((y1 - top) / r, 0, h),
             np.clip((x2 - left) / r, 0, w), np.clip((y2 - top) / r, 0, h)]
        dets.append((labels[int(cid[i])], float(conf[i]), b))
    return sorted(dets, key=lambda d: -d[1])


def yoloe_ultralytics(labels):
    r = YOLO(YOLOE_TFLITE, task="segment").predict(
        IMG, imgsz=640, conf=CONF, iou=IOU, agnostic_nms=True, verbose=False)[0]
    dets = [(labels[int(c)], float(s), b) for s, c, b in
            zip(r.boxes.conf.tolist(), r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    return sorted(dets, key=lambda d: -d[1])


def compare(mine, ref):
    def has(d, pool):
        return any(d[0] == p[0] and iou_xyxy(d[2], p[2]) > 0.9 for p in pool)
    print(f"  numpy: {len(mine)} detections   ultralytics(tflite): {len(ref)} detections")
    for d in [d for d in mine if not has(d, ref)]:
        print(f"   only in numpy        : {fmt(d)}")
    for d in [d for d in ref if not has(d, mine)]:
        print(f"   only in ultralytics  : {fmt(d)}")
    print("  (no 'only in' lines => the numpy decode reproduces Ultralytics' pipeline)")


def depth_check(img):
    h, w = img.shape[:2]
    ref = YOLO(DEPTH_TFLITE, task="depth")(IMG, imgsz=768, verbose=False)[0].depth.data.cpu().numpy()
    rs = cv2.resize(img, (768, 768), interpolation=cv2.INTER_LINEAR)  # stretch, no letterbox
    variants = {"rgb/255": to_input(rs), "imagenet mean/std": to_input(rs, IMAGENET_MEAN, IMAGENET_STD)}
    for name, x in variants.items():
        raw = run(DEPTH_TFLITE, x)[0][0, 0]
        stats(f"[{name}] raw output (metres?)", raw)
        up = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        rel = np.abs(up - ref) / np.maximum(ref, 1e-3)
        print(f"    median relative difference vs Ultralytics: {np.median(rel):.3%}")
    print("  (the correct normalisation is the one near 0%; the other should be far off)")


if __name__ == "__main__":
    img = cv2.imread(IMG)
    assert img is not None, f"{IMG} not found in the current folder"
    labels = [ln.strip() for ln in Path("labels.txt").read_text().splitlines() if ln.strip()]
    print("-- YOLOE: numpy decode of the raw tflite output --")
    mine = yoloe_numpy(img, labels)
    compare(mine, yoloe_ultralytics(labels))
    print("\n-- Depth: preprocessing hypotheses --")
    depth_check(img)
