"""Clusters where the software is confident it is ONE person, but not WHICH person.

Distinct from rematch: there the top candidate is clear and the question is whether to
accept it. Here the top two candidates are within a hair of each other, so accepting the
leader would be a coin toss. These are the ones most likely to be filed under the wrong
relative -- a mother and daughter, say -- an error that can take a day to find later.

    .venv\\Scripts\\python.exe scripts\\make_borderline_review.py [gap] [min_score]
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import base64

import cv2
import numpy as np

import _archive as A

_cache: dict = {}


def _img(p):
    im = _cache.get(p)
    if im is None:
        im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if len(_cache) > 10:
            _cache.clear()
        _cache[p] = im
    return im


def _enc(img, px, q=74):
    h, w = img.shape[:2]
    if max(h, w) > px:
        s = px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""

GAP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30
REF, OWN = 3, 5


def main() -> None:
    named = A.named_clusters()
    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)

    by_person = defaultdict(list)
    for i, c in enumerate(cl):
        who = named.get(int(c))
        if who:
            by_person[who].append(i)

    rows = []
    for c in sorted(set(cl.tolist())):
        if int(c) in named:
            continue
        idx = np.where(cl == c)[0]
        s = P @ A.cluster_centroid(cl, M, c)
        o = np.argsort(-s)
        t1, t2 = float(s[o[0]]), float(s[o[1]])
        if t1 >= MIN and (t1 - t2) < GAP:
            rows.append((t1 - t2, t1, c, [(people[k], float(s[k])) for k in o[:3]], idx))
    rows.sort(key=lambda r: r[0])          # tightest contest first
    print(f"{len(rows)} contested clusters (top two within {GAP}, leader >= {MIN})")

    body = []
    for gap, t1, c, cand, idx in rows:
        step = max(1, len(idx) // OWN)
        shots = []
        for pth in sorted({paths[i] for i in idx})[:3]:
            im = _img(pth)
            if im is None:
                continue
            box = im.copy()
            for i in idx:
                if paths[i] == pth:
                    x1, y1, x2, y2 = [int(v) for v in bbs[i]]
                    cv2.rectangle(box, (x1, y1), (x2, y2), (0, 220, 255),
                                  max(3, int(im.shape[1] / 260)))
            shots.append(f"<figure><img src='data:image/jpeg;base64,{_enc(box, 600)}'>"
                         f"<figcaption>{A.esc(Path(pth).name)}</figcaption></figure>")
        own = "".join(shots)
        refs = "".join(
            f"<div class=grp><h4>{A.esc(n)} — {v:.2f}</h4>" +
            "".join(f"<img src='data:image/jpeg;base64,{A.crop(paths[i], bbs[i])}'>"
                    for i in by_person[n][:REF]) + "</div>"
            for n, v in cand[:2])
        sc = " · ".join(f"{A.esc(n)} <b>{v:.2f}</b>" for n, v in cand)
        body.append(
            f"<div class=card data-id='c{c}'>"
            f"<div class=hd><b>кластер {c}</b><span class=n>{len(idx)} лиц</span>"
            f"<span class=cl>разрыв {gap:.3f}</span></div>"
            f"<div class=row>{refs}</div><div class=prints>{own}</div>"
            f"<div class=sc>{sc}</div>"
            f"<input class=nm placeholder='кто это на самом деле…' list=people>"
            f"<button class=sk>пропустить</button></div>")

    lede = (f"{len(rows)} кластеров, где программа уверена, что это <b>один человек</b>, "
            f"но не уверена, <b>какой именно</b>: два лучших кандидата отличаются меньше "
            f"чем на {GAP:.2f}.\n"
            "Принять лидера здесь — всё равно что бросить монету, а ошибка такого рода "
            "потом расходится по всему архиву. Слева — лица кластера, справа — эталоны "
            "двух ближайших кандидатов.")
    out = A.REVIEWS / "borderline-review.html"
    out.write_text(A.page("Спорные лица — borderline review", "Спорные лица",
                          lede, "".join(body), "borderline-review-v2",
                          "".join(f"<option value='{A.esc(p)}'>" for p in people)),
                   encoding="utf-8")
    print(f"-> {out}  {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
