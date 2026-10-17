"""Render the split clusters nobody has identified, with the context needed to name them.

For each unresolved cluster: every face crop, the source prints, the estimated age, and
which already-named people appear in the same photographs. Co-occurrence is usually what
cracks these -- knowing a face sits beside a known grandmother in a 1948 print narrows it
a long way.

Labels are ASCII: OpenCV's built-in font cannot draw Cyrillic, so named relatives are
listed in the console output rather than burned into the image.

    .venv\\Scripts\\python.exe scripts\\show_unresolved.py
"""
from __future__ import annotations

import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
from _paths import BP, LOGS
from _paths import DB  # noqa: E402
OUT = REPO / "artifacts" / "checks" / "unresolved_split_clusters.jpg"
MODEL = "buffalo_l"
THUMB = 120


def main() -> None:
    unresolved = [int(r["cluster"]) for r in
                  csv.DictReader((REPO / "data" / "split_import_log.csv").open(encoding="utf-8-sig"))
                  if not (r["person"] or "").strip() and r["method"] != "not-a-person"]
    print(f"{len(unresolved)} unresolved clusters: {unresolved}\n")

    # cluster -> person for everything that IS named, to report co-occurrence
    named: dict[int, str] = {}
    votes: dict[int, Counter] = defaultdict(Counter)
    for r in csv.DictReader((LOGS / "main_import_log.csv").open(encoding="utf-8-sig")):
        c, who = (r.get("cluster") or "").strip(), (r.get("person") or "").strip()
        if c.lstrip("-").isdigit() and who and not who.startswith("Person_"):
            votes[int(c)][who] += 1
    for c, v in votes.items():
        named[c] = v.most_common(1)[0][0]
    for r in csv.DictReader((REPO / "data" / "split_import_log.csv").open(encoding="utf-8-sig")):
        if (r["person"] or "").strip():
            named[int(r["cluster"])] = r["person"]

    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "select cluster_id, path, bbox, age, sex from face "
        "where model=? and cluster_id is not null and cluster_id!=-1", (MODEL,)).fetchall()
    by_cluster: dict[int, list] = defaultdict(list)
    by_photo: dict[str, set[int]] = defaultdict(set)
    for cid, p, bb, age, sex in rows:
        by_cluster[cid].append((p, bb, age, sex))
        by_photo[p].add(cid)

    strips = []
    for c in unresolved:
        faces = by_cluster.get(c, [])
        if not faces:
            continue
        photos = sorted({Path(p).name for p, _, _, _ in faces})
        others = sorted({named[o] for p, _, _, _ in faces
                         for o in by_photo[p] if o in named})
        ages = [a for _, _, a, _ in faces if a and a > 0]
        sexes = Counter(s for _, _, _, s in faces if s)
        age_txt = f"~{int(np.median(ages))}" if ages else "?"
        sex_txt = sexes.most_common(1)[0][0] if sexes else "?"

        print(f"c{c}  {len(faces)} faces  age{age_txt} {sex_txt}")
        print(f"     prints : {', '.join(photos)}")
        print(f"     with   : {', '.join(others) if others else '(nobody identified)'}\n")

        tiles = []
        for p, bb, _, _ in faces:
            fp = Path(p)
            if not fp.exists():
                continue
            img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb.strip("[]").split(",")]
            pad = 0.3 * max(x2 - x1, y2 - y1)
            h, w = img.shape[:2]
            xa, ya = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            xb, yb = min(w, int(x2 + pad)), min(h, int(y2 + pad))
            if xb > xa and yb > ya:
                tiles.append(cv2.resize(img[ya:yb, xa:xb], (THUMB, THUMB)))
        if not tiles:
            continue
        strip = np.hstack(tiles)
        lab = np.full((26, strip.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"c{c}  n={len(tiles)}  age{age_txt} {sex_txt}  [{photos[0]}]",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        strips.append(np.vstack([lab, strip]))

    if strips:
        w = max(s.shape[1] for s in strips)
        canvas = np.vstack([
            np.hstack([s, np.full((s.shape[0], w - s.shape[1], 3), 245, np.uint8)])
            if s.shape[1] < w else s for s in strips])
        OUT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", canvas)[1].tofile(str(OUT))
        print(f"contact sheet -> {OUT}")


if __name__ == "__main__":
    main()
