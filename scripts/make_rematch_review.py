"""Unassigned clusters that resemble somebody already named — shown as PHOTOGRAPHS.

Reference faces may be crops: you already know what your grandmother looks like. The unknown
cluster may not. Its prints are shown whole, with the face boxed, because the room,
the clothes, the occasion and the company are what decide whether the candidate is
right - the score only says two faces are similar.

Paginates; most confident first.

    .venv\\Scripts\\python.exe scripts\\make_rematch_review.py [min_score] [per_page]
"""
from __future__ import annotations

import base64
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import _archive as A

_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
MIN = float(_pos[0]) if _pos else 0.35
PER_PAGE = int(_pos[1]) if len(_pos) > 1 else 30
# --only 204,655 builds one page for named clusters regardless of score or closure, under
# its own filename. The numbered pages are the record of a question already asked; asking
# a follow-up must not overwrite them.
ONLY = next((set(int(x) for x in a.split("=", 1)[1].replace("c", "").split(","))
             for a in sys.argv[1:] if a.startswith("--only=")), None)
OUT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
STRONG, REF, MAX_PRINTS = 0.50, 3, 3
PRINT_PX = 600


def enc(img, px, q=74) -> str:
    h, w = img.shape[:2]
    if max(h, w) > px:
        s = px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


_cache: dict[str, object] = {}


def img_of(p: str):
    im = _cache.get(p)
    if im is None:
        im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if len(_cache) > 10:
            _cache.clear()
        _cache[p] = im
    return im


def declined() -> set[int]:
    """Clusters the family examined and did not recognise. They are a
    finding, not a backlog - rebuilding them into open cards asks the same people the
    same question. See close_rematch_review.py.

    Every review's declined list counts, not just this one's: "не знаю" said on the
    borderline page is the same answer as "не знаю" said here, and re-asking it under a
    different heading is still re-asking it.
    """
    out: set[int] = set()
    for s in A.REVIEW_SETS.values():
        p = A.REPO / "data" / s["declined"]
        if p.exists():
            out |= {int(r["cluster"]) for r in csv.DictReader(p.open(encoding="utf-8-sig"))}
    return out


def main() -> None:
    named = A.named_clusters()
    closed = declined()
    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)

    by_person = defaultdict(list)
    photo_clusters = defaultdict(set)
    for i, c in enumerate(cl):
        photo_clusters[paths[i]].add(int(c))
        w = named.get(int(c))
        if w:
            by_person[w].append(i)

    rows = []
    for c in sorted(set(cl.tolist())):
        if ONLY is not None and int(c) not in ONLY:
            continue
        if ONLY is None and (int(c) in named or int(c) in closed):
            continue
        idx = np.where(cl == c)[0]
        s = P @ A.cluster_centroid(cl, M, c)
        o = np.argsort(-s)
        if ONLY is None and float(s[o[0]]) < MIN:
            continue
        rows.append((float(s[o[0]]), int(c), [(people[k], float(s[k])) for k in o[:3]], idx))
    rows.sort(reverse=True, key=lambda r: r[0])
    pages = [rows[i:i + PER_PAGE] for i in range(0, len(rows), PER_PAGE)]
    print(f"{len(rows)} clusters >= {MIN} -> {len(pages)} page(s)")

    datalist = "".join(f"<option value='{A.esc(p)}'>" for p in people)
    for pi, chunk in enumerate(pages, 1):
        body = []
        for top, c, cand, idx in chunk:
            best = cand[0][0]
            # A follow-up card has to settle a specific question, usually between two people
            # who look alike, so it shows every candidate's reference and every print.
            shown = [n for n, _ in cand] if ONLY else [best]
            ref = "".join(
                f"<div class=grp><h4>эталон — {A.esc(n)}</h4>" +
                "".join(f"<img src='data:image/jpeg;base64,{A.crop(paths[i], bbs[i], px=78)}'>"
                        for i in by_person[n][:REF]) + "</div>" for n in shown)
            photos, shots = sorted({paths[i] for i in idx}), []
            for p in (photos if ONLY else photos[:MAX_PRINTS]):
                im = img_of(p)
                if im is None:
                    continue
                box = im.copy()
                for i in idx:
                    if paths[i] == p:
                        x1, y1, x2, y2 = [int(v) for v in bbs[i]]
                        cv2.rectangle(box, (x1, y1), (x2, y2), (0, 220, 255),
                                      max(3, int(im.shape[1] / 260)))
                cap = f"{A.esc(Path(p).parent.name)}\\{A.esc(Path(p).name)}"
                if ONLY:
                    zoom = A.crop(p, next(bbs[i] for i in idx if paths[i] == p), px=150, pad=1.1)
                    shots.append(
                        "<figure style='margin:0 0 12px'>"
                        "<div style='display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap'>"
                        f"<img style='width:150px;height:150px;border-radius:5px' "
                        f"src='data:image/jpeg;base64,{zoom}'>"
                        f"<img style='max-width:100%;border-radius:5px' "
                        f"src='data:image/jpeg;base64,{enc(box, 1100)}'></div>"
                        f"<figcaption>{cap}</figcaption></figure>")
                else:
                    shots.append(f"<figure><img src='data:image/jpeg;base64,{enc(box, PRINT_PX)}'>"
                                 f"<figcaption>{cap}</figcaption></figure>")
            co = Counter()
            for p in photos:
                for o in photo_clusters[p]:
                    w = named.get(o)
                    if w:
                        co[w] += 1
            colin = (", ".join(f"{A.esc(n)} <span>×{k}</span>" for n, k in co.most_common(5))
                     or "<i>никого опознанного рядом</i>")
            sc = " · ".join(f"{A.esc(n)} <b>{v:.2f}</b>" for n, v in cand)
            body.append(
                f"<div class=card data-id='c{c}'>"
                f"<div class=hd><b>кластер {c}</b>"
                f"<span class=n>{len(photos)} фото · {len(idx)} лиц</span>"
                f"<span class=cl>{'≥0.50' if top >= STRONG else ''}</span></div>"
                f"<div class=co><span>в кадре с:</span> {colin}</div>"
                f"<div class=row>{ref}</div>"
                f"<div class=prints>{''.join(shots)}</div>"
                f"<div class='sc{' hi' if top >= STRONG else ''}'>{sc}</div>"
                f"<input class=nm placeholder='имя, если согласны…' list=people>"
                f"<button class=sk>пропустить</button></div>")

        if ONLY:
            # Not "a question that is still open": the page outlives the answer, and a
            # lede that asserts a status goes stale the moment somebody answers it.
            # stamp_reviews.py writes the status, from the logs, on every run.
            lede = ("Отдельная карточка — вопрос задан вне нумерованных страниц.\n"
                    "Эталоны — по трём кандидатам сразу; дальше <b>все</b> снимки кластера "
                    "целиком, лицо обведено жёлтым, слева от снимка — оно же крупно.\n"
                    "Это групповые фотографии: решает не столько лицо, сколько кто стоит рядом.")
            out = A.REVIEWS / (OUT or f"cluster-{sorted(ONLY)[0]}-check.html")
            out.write_text(A.page("Проверка кластера", "Проверка кластера", lede,
                                  "".join(body), f"cluster-check-{'-'.join(map(str, sorted(ONLY)))}",
                                  datalist), encoding="utf-8")
            print(f"   {out.name}  {out.stat().st_size // 1024} KB  {len(chunk)} card(s)")
            continue

        nav = " · ".join(f"<b>{i}</b>" if i == pi else
                         f"<a href='rematch-review-{i:02d}.html'>{i}</a>"
                         for i in range(1, len(pages) + 1))
        lede = (f"Страница {pi} из {len(pages)}. {len(rows)} кластеров, похожих на кого-то из "
                f"{len(people)} названных людей ({MIN:.2f} и выше; {sum(1 for r in rows if r[0] >= STRONG)} "
                f"из них ≥{STRONG:.2f}).\n"
                "Слева — эталонные лица, дальше — <b>целые снимки</b> кластера, "
                "лицо обведено жёлтым. Совпадение говорит только о похожести; решает "
                "контекст — комната, повод и кто рядом.\n"
                f"Страницы: {nav}")
        out = A.REVIEWS / f"rematch-review-{pi:02d}.html"
        # A page that has been answered is the record of the question asked, and
        # close_rematch_review.py reads it back to know what was shown. Overwriting it
        # rewrites history - so a rebuild over an existing page has to be deliberate.
        if out.exists() and "--force" not in sys.argv:
            raise SystemExit(
                f"{out.name} already exists and has been answered.\n"
                "Use --only=<ids> --out=<name> to ask a follow-up under its own filename, "
                "or --force to deliberately rebuild the numbered pages.")
        out.write_text(A.page(f"Повторный подбор — стр. {pi}", "Повторный подбор лиц",
                              lede, "".join(body), f"rematch-review-v3-p{pi}", datalist),
                       encoding="utf-8")
        print(f"   {out.name}  {out.stat().st_size // 1024} KB  {len(chunk)} cards")


if __name__ == "__main__":
    main()
