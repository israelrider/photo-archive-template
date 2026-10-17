"""MiVOLO ages for face ids, via a subprocess in MiVOLO's own venv.

WHY THIS EXISTS

Every `ageNN_` prefix in `by-person\\` should come from MiVOLO. The `face.age` column in the database is `buffalo_l`'s
estimate, and `images-agent\\docs\\AGE_ESTIMATION.md` opens by calling it **unusable**: it
tagged a ten-year-old 47 and 60. Anything that names a file by age must use this module,
never that column.

The cost of getting it wrong is visible in an album: a distributor that reads `face.age`
files a woman in her twenties as `age55_` and `age79_`, and sorts a ten-year-old's school
portrait in among the same person's seventies.

HOW

MiVOLO pins a `timm` dev build that conflicts with the main venv's torch, so it cannot be
imported here. It lives in its own venv and runs as a subprocess; both are named in
archive.toml (`mivolo_python`, `mivolo_repo`). This module only reads that venv and the
HuggingFace cache, and sets `HF_HUB_OFFLINE=1` so a missing model fails loudly rather than
downloading.

`ages()` NEVER falls back to `face.age`. If MiVOLO cannot run it returns `{}`, and the
caller is expected to write `noage` - a filename that admits it does not know beats one
that states a wrong number with confidence.

    from _mivolo import ages
    a = ages([7941, 7797])          # {7941: 2.8, 7797: 30.5}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from _paths import MIVOLO_PYTHON, MIVOLO_REPO

REPO = Path(__file__).resolve().parent.parent
VENV = MIVOLO_PYTHON
MIVOLO_CLONE = MIVOLO_REPO
WORKER = REPO / "scripts" / "_mivolo_worker.py"

# The crop the review pages draw and the crops these ages were first measured on: the
# detector box grown by 40% of its longer side. Changing this changes every age it
# returns, so it is fixed here rather than passed in.
MARGIN = 0.4


def available() -> bool:
    return bool(VENV and MIVOLO_CLONE) and VENV.is_file() and MIVOLO_CLONE.is_dir()


def _crop(path: str, bbox) -> "np.ndarray | None":
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in bbox)
    m = int(MARGIN * max(x2 - x1, y2 - y1))
    h, w = img.shape[:2]
    c = img[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
    return c if c.size else None


def ages(face_ids, conn=None) -> dict[int, float]:
    """{face_id: mivolo_age}. Missing ids simply do not appear; {} if MiVOLO cannot run."""
    ids = sorted({int(i) for i in face_ids})
    if not ids or not available():
        if not available():
            print("MiVOLO unavailable - ages will be 'noage', NOT face.age", file=sys.stderr)
        return {}

    close = conn is None
    if conn is None:
        import sqlite3
        import _archive as A
        conn = sqlite3.connect(A.DB)
    try:
        q = "select id, path, bbox from face where id in (%s)" % ",".join("?" * len(ids))
        rows = list(conn.execute(q, ids))
    finally:
        if close:
            conn.close()

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for fid, path, bbox in rows:
            c = _crop(path, json.loads(bbox))
            if c is not None:
                cv2.imencode(".png", c)[1].tofile(str(d / f"{fid}.png"))
        env = dict(os.environ, PYTHONPATH=str(MIVOLO_CLONE), HF_HUB_OFFLINE="1",
                   PYTHONIOENCODING="utf-8")
        r = subprocess.run([str(VENV), str(WORKER), str(d)],
                           capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(f"MiVOLO worker failed ({r.returncode}); ages will be 'noage':\n"
              f"{r.stderr[-2000:]}", file=sys.stderr)
        return {}
    try:
        return {int(k): float(v) for k, v in json.loads(r.stdout.strip().splitlines()[-1]).items()}
    except Exception as e:                                  # noqa: BLE001
        print(f"MiVOLO output unreadable ({e}); ages will be 'noage'", file=sys.stderr)
        return {}
