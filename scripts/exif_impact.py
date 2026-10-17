"""RESULT: there is no EXIF orientation problem. Kept as the proof.

I suspected the indexer read rotated photographs sideways: 1096 of 2465 indexed images
carry EXIF orientation 6/8/3, and viewing one raw showed it on its side.

Measured over 40 affected photographs: 301 faces found either way, identical mean
det score, zero photographs changed. The reason is that **cv2.imread and cv2.imdecode
apply EXIF orientation by default** - IMREAD_IGNORE_ORIENTATION is the opt-out, and
images_agent does not pass it. cv2 output is pixel-identical to exif-corrected PIL.

What actually looked sideways was the image VIEWER showing raw pixels, not the archive
and not the pipeline. Re-run this before ever believing the theory again.

⚠ NARROWER THAN IT FIRST SOUNDS. images_agent._read_image_bgr calls cv2.imread(str(path)),
which returns None for a NON-ASCII path, and its PIL fallback does NOT call exif_transpose.
So a rotated photograph under a non-ASCII (e.g. Cyrillic) directory WOULD be indexed
sideways. Check whether any indexed photo at a non-ASCII path carries an orientation tag.
One-line fix in images-agent: wrap the PIL fallback in ImageOps.exif_transpose.

    .venv\\Scripts\\python.exe scripts\\exif_impact.py [sample_size]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from _paths import IMAGES_AGENT  # noqa: E402
sys.path.insert(0, str(IMAGES_AGENT))
from images_agent.faces import _load_face_app, FACE_MODEL_NAME

from _paths import DB  # noqa: E402
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def raw_bgr(p: Path):
    """What the indexer currently sees."""
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def upright_bgr(p: Path):
    """What it should see."""
    im = ImageOps.exif_transpose(Image.open(p).convert("RGB"))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main() -> None:
    conn = sqlite3.connect(DB)
    rows = conn.execute("select path, count(*) from face group by path").fetchall()
    affected = []
    for p, n in rows:
        try:
            o = (Image.open(p)._getexif() or {}).get(274, 1) or 1
        except Exception:
            continue
        if o != 1:
            affected.append((p, n))
    print(f"{len(affected)} affected photographs; sampling {N}")

    rng = np.random.default_rng(0)
    pick = [affected[i] for i in rng.choice(len(affected), min(N, len(affected)), replace=False)]

    app = _load_face_app(1024, -1)
    same = gained = lost = 0
    d_raw = d_up = 0.0
    n_raw = n_up = 0
    worst = []

    for p, indexed_n in pick:
        pth = Path(p)
        a, b = raw_bgr(pth), upright_bgr(pth)
        if a is None or b is None:
            continue
        fa, fb = app.get(a), app.get(b)
        n_raw += len(fa); n_up += len(fb)
        d_raw += sum(float(f.det_score) for f in fa)
        d_up += sum(float(f.det_score) for f in fb)
        if len(fb) > len(fa):
            gained += len(fb) - len(fa)
            worst.append((len(fb) - len(fa), len(fa), len(fb), pth.name))
        elif len(fb) < len(fa):
            lost += len(fa) - len(fb)
        else:
            same += 1

    print(f"\nfaces found sideways (as indexed): {n_raw}")
    print(f"faces found upright              : {n_up}")
    if n_raw:
        print(f"change: {n_up - n_raw:+d} faces  ({(n_up - n_raw) / n_raw * 100:+.0f}%)")
    if n_raw and n_up:
        print(f"mean det score  sideways {d_raw / max(n_raw,1):.3f}   upright {d_up / max(n_up,1):.3f}")
    print(f"\nphotographs unchanged: {same} of {len(pick)}")
    print(f"faces gained by rotating: {gained}   lost: {lost}")
    if worst:
        print("\nbiggest gains:")
        for g, a_, b_, nm in sorted(worst, reverse=True)[:10]:
            print(f"   +{g}  ({a_} -> {b_} faces)  {nm}")


if __name__ == "__main__":
    main()
