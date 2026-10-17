"""Inspect two people whose clusters sit suspiciously close, and render the faces side by side.

Low margin in CONFIDENCE.csv means "these two could be mixed up". It does not say which
way, or whether anything is actually wrong. This pulls the offending clusters, reports the
cross-similarities, and writes a contact sheet so a human can look.

    .venv\\Scripts\\python.exe scripts\\check_pair.py "<person A>" "<person B>"
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
ARCHIVE = Path.home() / "Pictures" / "family-archive"
from _paths import BP
from _paths import DB  # noqa: E402
MODEL = "buffalo_l"
THUMB = 96
PER_CLUSTER = 8


def load():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "select cluster_id, path, bbox, vector, age from face "
        "where model=? and cluster_id is not null and cluster_id!=-1", (MODEL,)).fetchall()
    cl = np.array([r[0] for r in rows])
    paths = [r[1] for r in rows]
    bboxes = [r[2] for r in rows]
    ages = [r[4] for r in rows]
    M = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return cl, paths, bboxes, ages, M


def owner_map() -> dict[str, set[str]]:
    o: dict[str, set[str]] = {}
    for r in csv.DictReader((REPO / "data" / "MANIFEST.csv").open(encoding="utf-8-sig")):
        if r["origin_path"]:
            o.setdefault(str(Path(r["origin_path"]).resolve()), set()).add(r["person"])
    return o


def clusters_of(person: str, cl, paths, owner) -> list[int]:
    folder_n = sum(1 for v in owner.values() if person in v)
    out = []
    for c in set(cl.tolist()):
        idx = np.where(cl == c)[0]
        ph = {str(Path(paths[i]).resolve()) for i in idx}
        n = sum(1 for p in ph if person in owner.get(p, ()))
        if n / len(ph) >= 0.5 and n >= max(2, 0.05 * folder_n):
            out.append(c)
    return sorted(out)


def crop(path: str, bbox_str: str):
    p = Path(path)
    if not p.exists():
        return None
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_str.strip("[]").split(",")]
    h, w = img.shape[:2]
    pad = 0.25 * max(x2 - x1, y2 - y1)
    x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    x2, y2 = min(w, int(x2 + pad)), min(h, int(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (THUMB, THUMB))


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    cl, paths, bboxes, ages, M = load()
    owner = owner_map()
    ca, cb = clusters_of(a, cl, paths, owner), clusters_of(b, cl, paths, owner)
    print(f"{a}: clusters {ca}")
    print(f"{b}: clusters {cb}")

    cent = {}
    for c in set(ca + cb):
        cent[c] = M[cl == c].mean(axis=0)
        cent[c] /= np.linalg.norm(cent[c])

    print("\ncross-similarity (own cohesion in brackets):")
    pairs = []
    for x in ca:
        coh_x = float(np.mean(M[cl == x] @ cent[x]))
        for y in cb:
            s = float(cent[x] @ cent[y])
            pairs.append((s, x, y))
            if s >= 0.45:
                print(f"   c{x} [{coh_x:.2f}, n={int((cl==x).sum())}]  vs  "
                      f"c{y} [n={int((cl==y).sum())}]   sim={s:.3f}"
                      f"{'   <-- SAME PERSON?' if s >= 0.6 else ''}")
    pairs.sort(reverse=True)

    rows_img = []
    for who, cls in ((a, ca), (b, cb)):
        for c in cls:
            idx = list(np.where(cl == c)[0])[:PER_CLUSTER]
            tiles = [t for t in (crop(paths[i], bboxes[i]) for i in idx) if t is not None]
            if not tiles:
                continue
            strip = np.hstack(tiles)
            lab = np.full((22, strip.shape[1], 3), 255, np.uint8)
            cv2.putText(lab, f"{who[:22]}  c{c}  n={int((cl==c).sum())}", (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            rows_img.append(np.vstack([lab, strip]))

    if rows_img:
        w = max(r.shape[1] for r in rows_img)
        canvas = np.vstack([
            np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)])
            if r.shape[1] < w else r for r in rows_img])
        out = REPO / "artifacts" / "checks" / f"pair_{a.split()[0]}_{b.split()[0]}.jpg"
        cv2.imencode(".jpg", canvas)[1].tofile(str(out))
        print(f"\ncontact sheet -> {out}")


if __name__ == "__main__":
    main()
