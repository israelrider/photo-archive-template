"""Check one identification against everything that can independently contradict it.

A review answer is a judgement made from one card. This assembles the evidence that card
did not show, for a cluster already named:

  * the family's own labelled crops for that person (`data\\annotations\\*.png`) scored
    against every face in the cluster. These are the strongest reference the archive has -
    somebody cropped that face out and wrote the name on it - and they are independent of
    any clustering.
  * the person's OTHER clusters, cluster-to-cluster. A person centroid averaged over
    several clusters can sit closer to a stranger than any single cluster of theirs does,
    so the honest comparison is against the clusters, one at a time.
  * whether another cluster of the same person appears in the SAME print. One person is
    rarely twice in one photograph, and that alone has decided identifications here.
  * the runner-up candidates, with their references.

    .venv\\Scripts\\python.exe scripts\\verify_cluster.py 676 [--out=name.html]
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import cv2
import numpy as np

import _archive as A

from _paths import IMAGES_AGENT  # noqa: E402
sys.path.insert(0, str(IMAGES_AGENT))
from images_agent.faces import _load_face_app, _read_image_bgr   # noqa: E402

REF, PRINT_PX = 4, 1000


def enc(img, px, q=74) -> str:
    h, w = img.shape[:2]
    if max(h, w) > px:
        s = px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not pos:
        raise SystemExit("usage: verify_cluster.py <cluster> [--out=name.html]")
    C = int(pos[0].lstrip("c"))
    out_name = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                    f"verify-c{C}.html")

    named = A.named_clusters()
    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)
    # --person checks a claim that is not (or no longer) in the logs. A withdrawn answer
    # must be re-rendered as a claim: a page whose heading still read "recorded as X"
    # would outlive the record it describes.
    claim = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--person=")), None)
    person = named.get(C) or claim
    if not person:
        raise SystemExit(f"c{C} is not named - pass --person=\"Имя\" to check a claim, "
                         f"or use make_rematch_review.py --only={C}")
    status = (f"записан как {person}" if named.get(C) == person
              else f"проверялся как {person}; в журнале не значится")
    idx = np.where(cl == C)[0]
    print(f"c{C} = {person}: {len(idx)} faces, {len({paths[i] for i in idx})} prints")

    parts = []

    # 1. the family's own labelled crops
    ann = A.REPO / "data" / "annotations"
    # `<folder> — <who is in the crop>.png`: the half AFTER the dash names the person
    # depicted, not the folder it was found in. Most crops saved in one person's folder
    # can show somebody else, so filtering on the folder half would score this cluster
    # against the wrong relatives and call it evidence.
    def toks(s: str) -> set[str]:
        return set(re.sub(r"[()\d]", " ", s).lower().split())

    crops = []
    if ann.is_dir():
        for p in sorted(ann.glob("*.png")):
            depicted = p.stem.split(" — ", 1)[1] if " — " in p.stem else p.stem
            if toks(person) <= toks(depicted) or toks(depicted) <= toks(person):
                crops.append(p)
    if crops:
        app = _load_face_app(1024, -1)
        rows = []
        for cp in crops:
            img = _read_image_bgr(cp)
            if img is None:
                print(f"   unreadable crop: {cp.name}")
                continue
            for f in sorted(app.get(img), key=lambda f: -(f.bbox[2] - f.bbox[0])):
                q = np.asarray(f.normed_embedding, dtype=np.float32)
                q /= np.linalg.norm(q)
                s = M @ q
                b = int(np.argmax(s))
                # A crop can hold two people. Only the face the archive already reads as
                # this person is a reference for them; the other is a bystander.
                if named.get(int(cl[b])) != person:
                    continue
                here = sorted((float(M[i] @ q) for i in idx), reverse=True)
                thumb = A.crop(str(cp), f.bbox.tolist(), px=110)
                rows.append(
                    f"<tr><td><img style='width:80px;border-radius:4px' "
                    f"src='data:image/jpeg;base64,{thumb}'></td>"
                    f"<td>{A.esc(cp.name)}</td>"
                    f"<td><b>{here[0]:.3f}</b></td>"
                    f"<td>{float(s[b]):.3f} — c{int(cl[b])}</td></tr>")
                print(f"   crop {cp.name[:44]:<44} anchored to c{int(cl[b])} at "
                      f"{float(s[b]):.3f}   vs c{C}: {here[0]:.3f}")
        if rows:
            parts.append(
                "<div class=card><div class=hd><b>Подписанные семьёй вырезки</b>"
                "<span class=n>независимая проверка</span></div>"
                "<div class=co>Столбец «в кластере» — насколько вырезка похожа на лучшее лицо "
                f"кластера c{C}. Столбец справа — на что она похожа больше всего во всём архиве."
                "</div><table style='border-collapse:collapse;font-size:12.5px'>"
                "<tr><th></th><th style='text-align:left'>вырезка</th>"
                f"<th>в кластере c{C}</th><th style='text-align:left'>лучшее совпадение</th></tr>"
                + "".join(rows) + "</table></div>")

    # 2. the person's other clusters, one at a time
    mine = sorted(c for c, w in named.items() if w == person and c != C)
    if mine:
        cen = A.cluster_centroid(cl, M, C)
        groups = []
        for c in mine:
            sc = float(cen @ A.cluster_centroid(cl, M, c))
            j = np.where(cl == c)[0]
            imgs = "".join(f"<img src='data:image/jpeg;base64,{A.crop(paths[i], bbs[i], px=78)}'>"
                           for i in j[:REF])
            groups.append(f"<div class=grp><h4>c{c} — {sc:.3f} · {len(j)} лиц</h4>{imgs}</div>")
            print(f"   c{C} vs c{c} (same person, other cluster): {sc:.3f}")
        parts.append(f"<div class=card><div class=hd><b>Другие кластеры «{A.esc(person)}»</b>"
                     f"<span class=n>сравнение кластер-с-кластером</span></div>"
                     f"<div class=row>{''.join(groups)}</div></div>")

    # 3. same person already in the same print?
    twice = []
    for c in mine:
        shared = {paths[i] for i in np.where(cl == c)[0]} & {paths[i] for i in idx}
        for p in sorted(shared):
            twice.append(f"<li>c{c} тоже на <code>{A.esc(Path(p).parent.name)}\\"
                         f"{A.esc(Path(p).name)}</code></li>")
            print(f"   WARNING: c{c} is also in {Path(p).name}")
    if twice:
        parts.append("<div class=card><div class=hd><b>Тот же человек уже в этом кадре</b>"
                     "<span class=n>один человек редко дважды на одном снимке</span></div>"
                     f"<ul>{''.join(twice)}</ul></div>")

    # 4. the prints themselves, and the runner-ups
    s = P @ A.cluster_centroid(cl, M, C)
    o = np.argsort(-s)
    line = " · ".join(f"{A.esc(people[k])} <b>{float(s[k]):.3f}</b>" for k in o[:4])
    by_person = {}
    for k in o[:3]:
        j = [i for i in range(len(cl)) if named.get(int(cl[i])) == people[k]][:REF]
        by_person[people[k]] = "".join(
            f"<img src='data:image/jpeg;base64,{A.crop(paths[i], bbs[i], px=78)}'>" for i in j)
    refs = "".join(f"<div class=grp><h4>{A.esc(n)}</h4>{v}</div>" for n, v in by_person.items())

    shots = []
    for p in sorted({paths[i] for i in idx}):
        im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        box = im.copy()
        for i in idx:
            if paths[i] == p:
                x1, y1, x2, y2 = [int(v) for v in bbs[i]]
                cv2.rectangle(box, (x1, y1), (x2, y2), (0, 220, 255),
                              max(3, int(im.shape[1] / 260)))
        zoom = A.crop(p, next(bbs[i] for i in idx if paths[i] == p), px=150, pad=1.1)
        shots.append(
            "<figure style='margin:0 0 12px'>"
            "<div style='display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap'>"
            f"<img style='width:150px;height:150px;border-radius:5px' "
            f"src='data:image/jpeg;base64,{zoom}'>"
            f"<img style='max-width:100%;border-radius:5px' "
            f"src='data:image/jpeg;base64,{enc(box, PRINT_PX)}'></div>"
            f"<figcaption>{A.esc(Path(p).parent.name)}\\{A.esc(Path(p).name)}</figcaption></figure>")

    parts.append(
        f"<div class=card data-id='c{C}'><div class=hd><b>кластер {C} — {A.esc(status)}"
        f"</b><span class=n>{len({paths[i] for i in idx})} фото · "
        f"{len(idx)} лиц</span></div>"
        f"<div class=row>{refs}</div><div class=prints>{''.join(shots)}</div>"
        f"<div class=sc>{line}</div>"
        "<input class=nm placeholder='имя, если менять…' list=people>"
        "<button class=sk>оставить как есть</button></div>")

    lede = (f"Проверка одной записи: c{C} — {A.esc(status)}.\n"
            "Сверху — то, что карточка опроса не показывала: подписанные семьёй вырезки, "
            "другие кластеры того же человека и совпадения в одном кадре. "
            "Внизу — сам кластер целиком.")
    out = A.REVIEWS / out_name
    out.write_text(A.page(f"Проверка c{C}", f"Проверка c{C}", lede, "".join(parts),
                          f"verify-c{C}",
                          "".join(f"<option value='{A.esc(p)}'>" for p in people)),
                   encoding="utf-8")
    print(f"-> {out}  {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
