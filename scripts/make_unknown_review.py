"""The unnamed Person_NN folders, shown as PHOTOGRAPHS rather than face crops.

`images-agent albums` leaves one by-person\\Person_NN\\ folder per cluster (an archive may
also have gathered them under by-person\\Unknown\\; both are read). The family names most of
them by renaming the folder (import_named_folders.py records that); this page asks about
the rest. Crops alone are not enough -- you cannot recognise anyone from a crop:
what places a face is the room, the clothes, the occasion, and who is standing next to
them. This shows the whole print with the face boxed, plus who else in the frame is
already identified.

A thousand photographs across hundreds of folders will not fit in one page, so it paginates.
Most tractable first: folders that share a photograph with someone already named.

    .venv\\Scripts\\python.exe scripts\\make_unknown_review.py [per_page] [max_photos]
"""
from __future__ import annotations

import base64
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import _archive as A

PER_PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MAX_PHOTOS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PRINT_PX, FACE_PX = 620, 84
UNK = A.BP / "Unknown"
PERSON_DIR = re.compile(r"Person_(\d+)")


def unknown_folders() -> dict[str, Path]:
    """Person_NN folder name -> its path, from by-person\\ and by-person\\Unknown\\."""
    out = {}
    for root in (A.BP, UNK):
        if root.is_dir():
            for d in root.iterdir():
                if d.is_dir() and PERSON_DIR.fullmatch(d.name):
                    out.setdefault(d.name, d)
    return out


def enc(img, px, q=74) -> str:
    h, w = img.shape[:2]
    if max(h, w) > px:
        s = px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def main() -> None:
    named = A.named_clusters()
    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)

    # which clusters belong to each Person_NN folder: the import log if it says, else the
    # folder name itself (images-agent names a cluster's folder Person_<cluster>)
    log = A.LOGS / "main_import_log.csv"
    rows = list(csv.DictReader(log.open(encoding="utf-8-sig"))) if log.exists() else []
    folder_clusters: dict[str, set[int]] = defaultdict(set)
    for f in unknown_folders():
        folder_clusters[f].add(int(PERSON_DIR.fullmatch(f).group(1)))
    for r in rows:
        p, c = (r.get("person") or "").strip(), (r.get("cluster") or "").strip()
        if re.fullmatch(r"Person_\d+", p) and c.lstrip("-").isdigit():
            folder_clusters[p].add(int(c))

    photo_clusters: dict[str, set[int]] = defaultdict(set)
    cluster_faces: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cl):
        photo_clusters[paths[i]].add(int(c))
        cluster_faces[int(c)].append(i)

    # A folder whose clusters have all been settled elsewhere is not an open question. The
    # rematch and borderline reviews answer and close clusters, and this page enumerates
    # folders, so without this it happily asks about them again.
    declined: set[int] = set()
    for s in A.REVIEW_SETS.values():
        p = A.REPO / "data" / s["declined"]
        if p.exists():
            declined |= {int(r["cluster"]) for r in csv.DictReader(p.open(encoding="utf-8-sig"))}
    settled = set(named) | declined
    # Folders the family looked at and skipped, page by page. `пропустить` is an answer.
    _ud = A.REPO / "data" / "unknown_declined.csv"
    done_folders = {r["folder"] for r in csv.DictReader(_ud.open(encoding="utf-8-sig"))} \
        if _ud.exists() else set()

    folders = sorted(f for f in unknown_folders() if f not in done_folders)
    cards, skipped = [], 0
    for f in folders:
        cls = {c for c in folder_clusters.get(f, set()) if c not in settled}
        if folder_clusters.get(f) and not cls:
            skipped += 1
            continue
        idx = [i for c in cls for i in cluster_faces.get(c, [])]
        if not idx:
            continue
        photos = sorted({paths[i] for i in idx})
        with_who = Counter()
        for p in photos:
            for c in photo_clusters[p]:
                w = named.get(c)
                if w:
                    with_who[w] += 1
        v = M[idx].mean(axis=0)
        v /= np.linalg.norm(v)
        s = P @ v
        cand = [(people[k], float(s[k])) for k in np.argsort(-s)[:3]]
        cards.append(dict(f=f, cls=sorted(cls), idx=idx, photos=photos,
                          with_who=with_who, cand=cand))

    # tractable first: known company, then more photographs, then better candidate
    cards.sort(key=lambda c: (-len(c["with_who"]), -len(c["photos"]), -c["cand"][0][1]))
    pages = [cards[i:i + PER_PAGE] for i in range(0, len(cards), PER_PAGE)]
    print(f"{len(cards)} folders -> {len(pages)} pages of {PER_PAGE} "
          f"({skipped} skipped: already answered or already declined elsewhere)")

    datalist = "".join(f"<option value='{A.esc(p)}'>" for p in people)
    for pi, chunk in enumerate(pages, 1):
        body = []
        for k in chunk:
            face_i = k["idx"][0]
            face = enc(_crop(paths[face_i], bbs[face_i]), FACE_PX)
            shots = []
            for p in k["photos"][:MAX_PHOTOS]:
                im = _img(p)
                if im is None:
                    continue
                box = im.copy()
                for i in k["idx"]:
                    if paths[i] == p:
                        x1, y1, x2, y2 = [int(v) for v in bbs[i]]
                        cv2.rectangle(box, (x1, y1), (x2, y2), (0, 220, 255),
                                      max(3, int(im.shape[1] / 260)))
                shots.append(f"<figure><img src='data:image/jpeg;base64,{enc(box, PRINT_PX)}'>"
                             f"<figcaption>{A.esc(Path(p).name)}</figcaption></figure>")
            co = (", ".join(f"{A.esc(n)} <span>×{c}</span>" for n, c in k["with_who"].most_common(5))
                  or "<i>никого опознанного рядом</i>")
            sc = " · ".join(f"{A.esc(n)} <b>{v:.2f}</b>" for n, v in k["cand"])
            hi = " hi" if k["cand"][0][1] >= 0.35 else ""
            body.append(
                f"<div class=card data-id='{A.esc(k['f'])}'>"
                f"<div class=hd><b>{A.esc(k['f'])}</b>"
                f"<span class=n>{len(k['photos'])} фото · {len(k['idx'])} лиц</span>"
                f"<span class=cl>c{',c'.join(map(str, k['cls']))}</span></div>"
                f"<div class=co><span>в кадре с:</span> {co}</div>"
                f"<div class=row><div class=grp><h4>это лицо</h4>"
                f"<img src='data:image/jpeg;base64,{face}'></div></div>"
                f"<div class=prints>{''.join(shots)}</div>"
                f"<div class='sc{hi}'>{sc}</div>"
                f"<input class=nm placeholder='кто это…' list=people>"
                f"<button class=sk>пропустить</button></div>")

        nav = " · ".join(
            f"<b>{i}</b>" if i == pi else
            f"<a href='unknown-review-{i:02d}.html'>{i}</a>" for i in range(1, len(pages) + 1))
        lede = (f"Страница {pi} из {len(pages)}. Показан <b>весь снимок</b>, а не только лицо — "
                f"жёлтой рамкой отмечено, о ком речь. Узнать человека по одному вырезанному "
                f"лицу почти невозможно: помогает комната, одежда, повод и то, кто стоит рядом.\n"
                f"Сначала идут те, кто попал в кадр с кем-то уже опознанным. "
                f"Страницы: {nav}")
        out = A.REVIEWS / f"unknown-review-{pi:02d}.html"
        out.write_text(A.page(f"Неопознанные — стр. {pi}", "Неопознанные лица",
                              lede, "".join(body), f"unknown-review-v3-p{pi}", datalist),
                       encoding="utf-8")
        print(f"   {out.name}  {out.stat().st_size // 1024} KB  {len(chunk)} cards")


_cache: dict[str, object] = {}


def _img(p: str):
    im = _cache.get(p)
    if im is None:
        im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if len(_cache) > 12:
            _cache.clear()
        _cache[p] = im
    return im


def _crop(p: str, bb):
    im = _img(p)
    if im is None:
        return np.zeros((10, 10, 3), np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bb]
    m = int(0.4 * max(x2 - x1, y2 - y1))
    h, w = im.shape[:2]
    c = im[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
    return c if c.size else np.zeros((10, 10, 3), np.uint8)


if __name__ == "__main__":
    main()
