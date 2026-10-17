r"""The whole remaining duplicate question, as one page, in the order it has to be asked.

`build_photographs.py` reduced "a lot of duplications" to two bounded piles, and they are
not the same question:

  A. the dHash pairs at d<=5 that would join negatives now held apart. The question is
     whether they are one photograph at all. Nothing gets deleted by answering it.
  B. the pairs inside a negative already agreed to be one photograph but never judged
     print-vs-rescan. The question is whether a file can go. Something does get deleted.

B is the destructive one, so B is shown first. Counts move whenever the manifest is
rebuilt, so they are printed at run time rather than written down here.

Every pair is put through the classifier that already exists -- `split_rescan_vs_print`'s
blob-damage measure, which counts localised differences AFTER alignment and sharpness
matching, because damage happens to a print and grain belongs to the negative. The machine
reading is printed on each card, but it is a reading, not a verdict: `blobs>=4` was set on
three known pairs and a synthetic control, and the archive has since produced pairs where
the family overruled it in both directions.

Most pairs are answered by the measurements, and a page that asks all of them anyway buys
the same fatigue that let duplicates through before. So each pair is bucketed: the ones a
person has to see are open and ordered hardest first, the rest are folded shut for
spot-checking. Note that `damage()` still returns a blob count for two scans that do not
align at all, and that number means nothing -- those cards say so instead.

Measurements are cached in `data/duplicate_review_machine.csv` and reused per pair, so a
manifest rebuild that opens three new pairs measures three, not a hundred. `--remeasure`
forces the lot.

    .venv\Scripts\python.exe scripts\make_duplicate_review.py [--remeasure]
"""
from __future__ import annotations

import base64
import csv
import html
import itertools
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import _archive as A
import split_rescan_vs_print as S
from find_same_negative_album import registers_arrays

sys.stdout.reconfigure(encoding="utf-8")

REMEASURE = "--remeasure" in sys.argv
REPO = A.REPO
CACHE = REPO / "data" / "duplicate_review_machine.csv"
OUT = REPO / "artifacts" / "reviews" / "duplicate-review.html"
SHOW_PX = 660          # display width per scan; big enough to see a crease, small enough to send
CHECK_PX = 300         # folded sections are spot-checked, not judged: a quarter of the bytes

STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")


def key(name: str) -> str:
    return STRIP.match(os.path.basename(name)).group(1)


def rows(name: str) -> list[dict]:
    p = REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))) if p.exists() else []


# ---------------------------------------------------------------- the open pairs

def open_pairs() -> list[dict]:
    P = rows("photographs.csv")
    scan = {r["scan"]: r for r in P}
    neg: dict[str, list[dict]] = defaultdict(list)
    for r in P:
        neg[r["negative"]].append(r)

    decided = {tuple(sorted((key(r["reference"]), key(r["other"]))))
               for r in rows("album_print_vs_rescan.csv")}

    out = []
    for n, members in neg.items():
        if not any(r["status"] == "plates-undecided" for r in members):
            continue
        for a, b in itertools.combinations(sorted(members, key=lambda r: r["scan"]), 2):
            if a["plate"] == b["plate"]:
                continue                      # already one print
            if tuple(sorted((a["scan"], b["scan"]))) in decided:
                continue                      # already judged
            out.append({"section": "B", "negative": n, "hamming": "",
                        "scan_a": a["scan"], "scan_b": b["scan"]})

    for r in rows("photograph_proposals.csv"):
        if int(r["hamming"]) <= 5 and r["already_one_negative"] == "no":
            out.append({"section": "A", "negative": f"{r['negative_a']} + {r['negative_b']}",
                        "hamming": r["hamming"],
                        "scan_a": r["scan_a"], "scan_b": r["scan_b"]})

    for p in out:
        for s in ("a", "b"):
            p[f"origin_{s}"] = scan[p[f"scan_{s}"]]["origin_path"]
            p[f"persons_{s}"] = scan[p[f"scan_{s}"]]["persons"]
    return out


# ---------------------------------------------------------------- measuring

def measure(pairs: list[dict]) -> list[dict]:
    # Incremental: rebuilding the manifest changes which pairs are open, and re-measuring
    # a hundred pairs to learn about three is the kind of cost that stops a check being run.
    have: dict[tuple[str, str], dict] = {}
    if CACHE.exists() and not REMEASURE:
        have = {(r["scan_a"], r["scan_b"]): r for r in rows(CACHE.name)}
    todo = [p for p in pairs if (p["scan_a"], p["scan_b"]) not in have]
    if not todo:
        print(f"cache hit: {CACHE.name} covers all {len(pairs)} pairs")
        return [have[(p["scan_a"], p["scan_b"])] for p in pairs]
    print(f"measuring {len(todo)} pairs ({len(pairs) - len(todo)} already cached)")

    cache: dict[str, np.ndarray] = {}

    def grey(path: str) -> np.ndarray:
        if path not in cache:
            if len(cache) > 8:               # some of these are 44 MB decoded
                cache.clear()
            cache[path] = S.load(path)
        return cache[path]

    done = []
    for i, p in enumerate(todo, 1):
        Ai, Bi = grey(p["origin_a"]), grey(p["origin_b"])
        res, inl = registers_arrays(Ai, Bi)
        d = S.damage(Ai, Bi)
        blobs, area = d if d else ("", "")
        if d:
            kind = "separate-print" if (blobs >= S.BLOB_MIN or area >= S.BLOB_AREA_MIN) else "rescan"
            conf = S.confidence(blobs)
        else:
            kind, conf = "undecided", "no-alignment"
        r = dict(p, blobs=blobs, blob_area=("" if area == "" else round(area, 1)),
                 kind=kind, confidence=conf,
                 residual=("" if res is None else round(res, 3)), inliers=inl,
                 mp_a=round(Image.open(p["origin_a"]).size[0]
                            * Image.open(p["origin_a"]).size[1] / 1e6, 1),
                 mp_b=round(Image.open(p["origin_b"]).size[0]
                            * Image.open(p["origin_b"]).size[1] / 1e6, 1))
        done.append(r)
        print(f"  {i:3d}/{len(todo)}  {p['section']}  {kind:15s} {conf:22s} "
              f"blobs={blobs} res={r['residual']}  {p['scan_a'][:34]} | {p['scan_b'][:34]}")

    for r in done:
        have[(r["scan_a"], r["scan_b"])] = r
    out = [have[(p["scan_a"], p["scan_b"])] for p in pairs]
    with CACHE.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else [])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {CACHE}  ({len(out)} pairs)")
    return out


# ---------------------------------------------------------------- the page

def thumb(path: str, px: int = SHOW_PX, q: int = 78) -> str:
    im = Image.open(path)
    im.draft("RGB", (px * 2, px * 2))
    im = im.convert("RGB")
    if im.width > px:
        im = im.resize((px, max(1, int(im.height * px / im.width))), Image.LANCZOS)
    buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, q])[1]
    return base64.b64encode(buf.tobytes()).decode()


def esc(s) -> str:
    return html.escape(str(s))


RESIDUAL_MAX = 0.45          # find_same_negative_album's own cut for "these register at all"


def registers(r: dict) -> bool:
    return r["residual"] != "" and float(r["residual"]) <= RESIDUAL_MAX


CHOICES = [
    ("keep-a", "оставить ЛЕВОЕ", "один отпечаток, пересканированный: правый файл лишний"),
    ("keep-b", "оставить ПРАВОЕ", "один отпечаток, пересканированный: левый файл лишний"),
    ("both", "оба (разные отпечатки)", "один негатив, но два разных предмета — ничего не удалять"),
    ("not-same", "разные снимки", "не один негатив: группу разорвать"),
]


def card(i: int, r: dict) -> str:
    pid = f"{r['scan_a']}||{r['scan_b']}"
    if not registers(r):
        # damage() still returns blobs on a failed alignment, and they mean nothing:
        # it is counting two different pictures, not damage to one print.
        reading, lean = "не совмещаются — счёт пятен здесь ничего не значит", ""
    elif r["kind"] == "undecided":
        reading, lean = "не удалось совместить — судить только глазом", ""
    else:
        reading = f"{r['kind']} ({r['confidence']})"
        lean = "hi" if r["kind"] == "separate-print" else ""
    bits = [f"blobs {r['blobs']}", f"площадь {r['blob_area']}"]
    if r["residual"] != "":
        bits.append(f"невязка {r['residual']}, {r['inliers']} inliers")
    if r["hamming"]:
        bits.append(f"dHash d={r['hamming']}")
    btns = "".join(
        f"<button class=ch data-v='{v}' title='{esc(t)}'>{esc(lbl)}</button>"
        for v, lbl, t in CHOICES)
    return f"""
<div class=card data-id="{esc(pid)}" data-sec="{r['section']}">
  <div class=hd><b>{i}</b> <span class=n>{esc(r['negative'])}</span>
    <span class="rd {lean}">{esc(reading)}</span>
    <span class=cl>{esc(' · '.join(bits))}</span></div>
  <div class=row>
    <figure><img src="data:image/jpeg;base64,{r['img_a']}">
      <figcaption>{esc(r['scan_a'])}<br><span class=n>{r['mp_a']} MP · {esc(r['persons_a'])}</span></figcaption></figure>
    <figure><img src="data:image/jpeg;base64,{r['img_b']}">
      <figcaption>{esc(r['scan_b'])}<br><span class=n>{r['mp_b']} MP · {esc(r['persons_b'])}</span></figcaption></figure>
  </div>
  <div class=ans>{btns}<button class=sk>пропустить</button>
    <input class=note placeholder="почему (необязательно)"></div>
</div>"""


def main() -> None:
    pairs = open_pairs()
    print(f"{len(pairs)} open pairs "
          f"({sum(1 for p in pairs if p['section']=='B')} in section B, "
          f"{sum(1 for p in pairs if p['section']=='A')} in section A)")
    meas = measure(pairs)

    # Triage. 110 pairs is not 110 questions: the measurements answer most of them, and a
    # page that asks all 110 anyway buys the same fatigue that let duplicates through
    # before. Each pair lands in exactly one bucket, hardest question first.
    def bucket(r: dict) -> str:
        if not registers(r):
            # dHash brought them together on gross tonal layout; nothing else did, and
            # they do not align. In section B this instead contradicts the group.
            return "no-register-B" if r["section"] == "B" else "no-register-A"
        if r["section"] == "A":
            return "A-real"
        if r["kind"] == "rescan":
            return "B-rescan"
        return "B-weak" if r["confidence"] == "separate-print-weak" else "B-strong"

    for r in meas:
        r["bucket"] = bucket(r)
    buckets = defaultdict(list)
    for r in meas:
        buckets[r["bucket"]].append(r)

    SECTIONS = [
        ("B-rescan", "1. Один отпечаток, отсканированный дважды — что оставить",
         "Здесь и только здесь ответ удаляет файл. Машина считает, что это один и тот же "
         "предмет: после совмещения между ними почти нет локальных различий. Смотреть надо на "
         "то, какой скан лучше — резкость и шум, а не размер в мегапикселях: один раз выбор "
         "уже был сделан по размеру и оказался неверным.", True),
        ("B-weak", "2. Пограничные: два отпечатка, но уверенности мало",
         "От 4 до 9 пятен различия — чуть выше порога. Порог поставлен по трём известным парам "
         "и одному синтетическому контролю, так что здесь он говорит тише всего.", True),
        ("no-register-B", "3. Противоречие: группа говорит «один снимок», совмещение — нет",
         "Эти пары лежат в группе, признанной одним негативом, но не совмещаются между собой "
         "(невязка выше 0.45). Либо группу держит вместе третий скан, либо в неё попало лишнее.", True),
        ("A-real", "4. Новые пары, которые действительно совмещаются",
         "Перцептивный хэш нашёл их, и совмещение подтверждает: это один снимок, до сих пор "
         "числившийся двумя. Здесь ничего не удаляется — ответ только соединяет группы.", True),
        ("B-strong", "5. Уверенно два разных отпечатка — подтвердить и забыть",
         "10 и больше пятен различия: это два физических предмета с одного негатива, оба "
         "остаются. Просмотр не обязателен; секция здесь для проверки, а не для работы.", False),
        ("no-register-A", "6. Ложные срабатывания хэша — уже отброшены",
         "Хэш сблизил их по общей раскладке светлого и тёмного, но они не совмещаются вовсе "
         "(невязка до 1.45 при пороге 0.45). Это разные снимки. Оставлено для проверки.", False),
    ]

    # A folded section is there to be spot-checked, not worked through, so it gets small
    # images. The four open sections get the size a crease is actually visible at.
    print("rendering thumbnails...")
    px = {k: (SHOW_PX if ask else CHECK_PX) for k, _, _, ask in SECTIONS}
    for r in meas:
        p = px[r["bucket"]]
        r["img_a"], r["img_b"] = thumb(r["origin_a"], p), thumb(r["origin_b"], p)
        r["show_px"] = p

    need = sum(len(buckets[k]) for k, _, _, ask in SECTIONS if ask)
    body, n = [], 0
    for k, title, blurb, ask in SECTIONS:
        rs = sorted(buckets[k], key=lambda r: (float(r["blobs"] or 0), r["scan_a"]))
        if not rs:
            continue
        cards = []
        for r in rs:
            n += 1
            cards.append(card(n, r))
        head = f"{esc(title)} <span class=cnt>{len(rs)}</span>"
        if ask:
            body.append(f"<h2>{head}</h2><p class=lede>{esc(blurb)}</p>{''.join(cards)}")
        else:
            # Folded shut: these are shown so the machine can be checked, not worked through.
            body.append(f"<details><summary><b>{head}</b></summary>"
                        f"<p class=lede>{esc(blurb)}</p>{''.join(cards)}</details>")
    print(f"  {need} pairs put to a person, {n - need} folded shut as already settled")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("{{BODY}}", "".join(body))
                       .replace("{{NEED}}", str(need))
                       .replace("{{REST}}", str(n - need)), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {n} pairs)")


PAGE = r"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Дубликаты: что осталось решить</title>
<style>
:root{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--ink:#1a1a18;--mut:#666;--bd:#ddd;--acc:#33517A}
@media(prefers-color-scheme:dark){:root{--bg:#17181A;--card:#212327;--ink:#E4E3DE;--mut:#A2A49F;--bd:#3A3D42;--acc:#7FA3D6}}
body{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--ink)}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:26px 0 4px}
.lede{color:var(--mut);max-width:70em;margin:0 0 14px}
.cnt{font-weight:400;color:var(--mut);font-size:13px}
details{margin:26px 0 0;border-top:1px solid var(--bd);padding-top:10px}
summary{cursor:pointer;font-size:16px;padding:4px 0}
#bar{position:sticky;top:0;background:var(--bg);padding:9px 0;z-index:9;border-bottom:1px solid var(--bd);margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:9px;padding:10px 12px;margin-bottom:14px}
.card.done{opacity:.4}
.hd{display:flex;gap:10px;align-items:baseline;margin-bottom:8px;flex-wrap:wrap}
.n{color:var(--mut);font-size:12px}
.rd{font-size:12px;color:var(--mut)} .rd.hi{color:var(--acc);font-weight:600}
.cl{margin-left:auto;color:var(--mut);font-size:11px;font-family:ui-monospace,monospace}
.row{display:flex;gap:12px;flex-wrap:wrap}
figure{margin:0} figure img{max-width:100%;border-radius:5px;display:block}
figcaption{font-size:11.5px;color:var(--ink);margin-top:4px;word-break:break-all}
figure{max-width:100%}
.ans{margin-top:9px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
button{padding:5px 12px;border:1px solid var(--bd);background:var(--bg);color:var(--ink);border-radius:5px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
input.note{flex:1;min-width:220px;padding:5px 7px;border:1px solid var(--bd);border-radius:5px;font:inherit;background:var(--bg);color:var(--ink)}
#out{white-space:pre;font-family:ui-monospace,monospace;font-size:12px;background:var(--card);
border:1px solid var(--bd);border-radius:6px;padding:12px;margin-top:12px;display:none;overflow-x:auto}
</style>
<h1>Дубликаты: что осталось решить</h1>
<p class=lede>2869 файлов в by-person — это 1828 снимков. Из всего, что оставалось спорного, машина
разобрала большую часть сама: <b>{{NEED}} пар</b> нужно решить глазом, остальные {{REST}} сложены внизу
и открываются только для проверки. Разделы идут от самого дорогого решения к самому дешёвому:
первый удаляет файл, последний не меняет ничего. Ответы сохраняются в браузере;
кнопка внизу собирает их для вставки.</p>
<div id=bar><span id=cnt></span><button onclick=exp()>показать заполненное</button></div>
{{BODY}}
<pre id=out></pre>
<script>
const K='dup-review-v1';
const st=JSON.parse(localStorage.getItem('dup-review-v1')||'{}');
const cards=[...document.querySelectorAll('.card')];
function save(){localStorage.setItem(K,JSON.stringify(st));cnt();}
cards.forEach(c=>{
  const id=c.dataset.id, note=c.querySelector('.note');
  const paint=()=>{const v=(st[id]||{}).v;
    c.querySelectorAll('button.ch').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
    c.classList.toggle('done',!!v);
    if((st[id]||{}).note!==undefined) note.value=st[id].note;};
  c.querySelectorAll('button.ch').forEach(b=>b.onclick=()=>{
    st[id]=Object.assign(st[id]||{},{v:b.dataset.v});save();paint();});
  c.querySelector('.sk').onclick=()=>{st[id]=Object.assign(st[id]||{},{v:'skip'});save();paint();};
  note.addEventListener('input',()=>{st[id]=Object.assign(st[id]||{},{note:note.value});save();});
  paint();
});
function cnt(){const v=Object.values(st).filter(x=>x.v&&x.v!=='skip').length;
  document.getElementById('cnt').textContent=v+' решено · '+
  Object.values(st).filter(x=>x.v==='skip').length+' пропущено · из '+cards.length+'  ';}
function exp(){const o=document.getElementById('out');
  const L=Object.entries(st).filter(([k,x])=>x.v&&x.v!=='skip')
    .map(([k,x])=>k.split('||')[0]+'\t'+k.split('||')[1]+'\t'+x.v+'\t'+(x.note||''));
  o.textContent=L.length?'scan_a\tscan_b\tverdict\tnote\n'+L.join('\n'):'ничего не заполнено';
  o.style.display='block';o.scrollIntoView({behavior:'smooth'});}
cnt();
</script>"""


if __name__ == "__main__":
    main()
