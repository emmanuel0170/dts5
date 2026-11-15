"""Render the guidance phrase bank with Kokoro on your PC. The app plays these clips: no on-device
synthesis latency, no CPU contention with the vision models, and the voice is Kokoro's.

Run from the project root:
    pip install kokoro soundfile numpy      # also needs espeak-ng installed (apt install espeak-ng / brew install espeak-ng)
    python tools/render_phrase_bank.py labels.txt

Reads  assets/phrase_bank/phrases.json  (fixed phrases)  and  labels.txt  (one vocabulary word per line).
Writes assets/phrase_bank/clips/*.wav (24 kHz mono 16-bit) and assets/phrase_bank/manifest.json.

NOT run by the author: written from the Kokoro package's documented usage (KPipeline). If the call shape
differs in your installed version, the error will point at the `pipe(...)` line.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

VOICE = "af_heart"
SPEED = 1.0
SR = 24000
ROOT = Path("assets/phrase_bank")

labels_file = Path(sys.argv[1] if len(sys.argv) > 1 else "labels.txt")
phrases = json.loads((ROOT / "phrases.json").read_text())
labels = [ln.strip() for ln in labels_file.read_text().splitlines() if ln.strip()]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


items = dict(phrases)
items.update({f"label:{slug(label)}": label for label in labels})


def trim(a: np.ndarray, thr: float = 0.01, pad: int = int(0.03 * SR)) -> np.ndarray:
    idx = np.flatnonzero(np.abs(a) > thr)
    return a[max(idx[0] - pad, 0): idx[-1] + pad] if idx.size else a


pipe = KPipeline(lang_code="a")
clips = ROOT / "clips"
clips.mkdir(parents=True, exist_ok=True)
manifest = {}
for key, text in items.items():
    chunks = [np.asarray(audio, dtype=np.float32).reshape(-1) for _, _, audio in pipe(text, voice=VOICE, speed=SPEED)]
    if not chunks:
        print(f"!! no audio for {key!r}")
        continue
    a = trim(np.concatenate(chunks))
    a = a / max(1e-6, float(np.abs(a).max())) * 0.7  # same peak level for every clip
    name = key.replace(":", "__") + ".wav"
    sf.write(clips / name, a, SR, subtype="PCM_16")
    manifest[key] = {"file": f"clips/{name}", "ms": int(len(a) / SR * 1000)}

(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
total_ms = sum(v["ms"] for v in manifest.values())
print(f"{len(manifest)} clips, {total_ms / 1000:.0f} s of audio, ~{total_ms * 48 / 1e6:.0f} MB")
