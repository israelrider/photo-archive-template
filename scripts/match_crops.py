"""Match hand-saved crop images back to the faces they came from.

The family annotates by cropping a face out of a print and saving it into the folder
with the person's name (`Анна Смирнова.png`, `Мария сестра Анны.png`). That is a real
identification channel, but it carries no face id. This detects the face in each crop,
embeds it, and reports which archive face it is - so a filename becomes a face id.

    .venv\\Scripts\\python.exe scripts\\match_crops.py <folder-or-image> [...]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np

import _archive as A

from _paths import IMAGES_AGENT  # noqa: E402
sys.path.insert(0, str(IMAGES_AGENT))
from images_agent.faces import _load_face_app, _read_image_bgr   # noqa: E402


def main() -> None:
    targets: list[Path] = []
    for a in sys.argv[1:]:
        p = Path(a)
        targets += sorted(x for x in p.iterdir() if x.suffix.lower() == ".png") if p.is_dir() else [p]
    if not targets:
        raise SystemExit("give a folder or image paths")

    named = A.named_clusters()
    cl, paths, bbs, dets, M = A.load_faces()
    conn = sqlite3.connect(A.DB)
    ids = [r[0] for r in conn.execute(
        "select id from face where model=? and cluster_id is not null and cluster_id!=-1",
        (A.MODEL,))]

    app = _load_face_app(1024, -1)
    for t in targets:
        img = _read_image_bgr(t)
        if img is None:
            print(f"{t.name}: unreadable")
            continue
        faces = sorted(app.get(img), key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        print(f"\n=== {t.name}   ({len(faces)} face(s) detected)")
        for k, f in enumerate(faces):
            q = np.asarray(f.normed_embedding, dtype=np.float32)
            q /= np.linalg.norm(q)
            s = M @ q
            order = np.argsort(-s)[:4]
            w = int(f.bbox[2] - f.bbox[0])
            print(f"  face {k} ({w}px, det {f.det_score:.2f}):")
            for i in order:
                who = named.get(int(cl[i]), "— unnamed —")
                print(f"      {s[i]:.3f}  face {ids[i]:<5} c{int(cl[i]):<5} {who:<28} "
                      f"{Path(paths[i]).name}")


if __name__ == "__main__":
    main()
