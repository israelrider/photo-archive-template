"""Render every detected face in the given prints, side by side, with its cluster.

Useful when one face in a photograph is identified and the others are not -- naming a
companion often names the whole print.

    .venv\\Scripts\\python.exe scripts\\show_photo_faces.py k_01.jpg a_23.jpg
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
from _paths import BP, LOGS
from _paths import DB  # noqa: E402
SIZE = 260


def names() -> dict[int, str]:
    out: dict[int, str] = {}
    v: dict[int, Counter] = defaultdict(Counter)
    for r in csv.DictReader((LOGS / "main_import_log.csv").open(encoding="utf-8-sig")):
        c, w = (r.get("cluster") or "").strip(), (r.get("person") or "").strip()
        if c.lstrip("-").isdigit() and w and not w.startswith("Person_"):
            v[int(c)][w] += 1
    for c, cc in v.items():
        out[c] = cc.most_common(1)[0][0]
    sl = REPO / "data" / "split_import_log.csv"
    if sl.exists():
        for r in csv.DictReader(sl.open(encoding="utf-8-sig")):
            if (r["person"] or "").strip():
                out[int(r["cluster"])] = r["person"]
    return out


def main() -> None:
    wanted = sys.argv[1:]
    if not wanted:
        raise SystemExit("give one or more print filenames, e.g. k_01.jpg")
    named = names()
    conn = sqlite3.connect(DB)
    strips = []
    for nm in wanted:
        rows = conn.execute(
            "select path, cluster_id, bbox, det_score from face where path like ? order by bbox",
            ("%" + nm,)).fetchall()
        if not rows:
            print(f"{nm}: no faces in the index")
            continue
        print(f"\n{nm}: {len(rows)} faces")
        tiles = []
        for path, cid, bb, ds in rows:
            p = Path(path)
            if not p.exists():
                continue
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            x1, y1, x2, y2 = [float(x) for x in bb.strip("[]").split(",")]
            pad = 0.45 * max(x2 - x1, y2 - y1)
            h, w = img.shape[:2]
            xa, ya = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            xb, yb = min(w, int(x2 + pad)), min(h, int(y2 + pad))
            if xb <= xa or yb <= ya:
                continue
            who = named.get(cid, "unnamed")
            tag = f"c{cid}" if cid != -1 else "unclustered"
            print(f"   {tag:<13} {who:<24} det={ds:.2f}")
            tile = cv2.resize(img[ya:yb, xa:xb], (SIZE, SIZE))
            lab = np.full((24, SIZE, 3), 255, np.uint8)
            cv2.putText(lab, f"{nm[:8]} {tag}", (5, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([lab, tile]))
        if tiles:
            strips.append(np.hstack(tiles))

    if strips:
        w = max(s.shape[1] for s in strips)
        canvas = np.vstack([
            np.hstack([s, np.full((s.shape[0], w - s.shape[1], 3), 245, np.uint8)])
            if s.shape[1] < w else s for s in strips])
        out = REPO / "artifacts" / "checks" / ("faces_" + "_".join(Path(x).stem for x in wanted) + ".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", canvas)[1].tofile(str(out))
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
