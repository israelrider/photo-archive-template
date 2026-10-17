r"""Ask the unresolved negatives the question their shape actually poses.

`duplicate-review.html` asks pairwise, and inside a negative holding five scans that is
the wrong question: answers can come back as a cycle -- A beats B, B beats C, C beats A --
so no survivor exists and nothing can be applied (apply_duplicate_review.py holds such
negatives back). Pairwise buttons cannot elect one winner out of five.

So this asks once per negative, over all its scans at once: **keep or drop, each**. Not
"which single one survives" -- a group of five can hold two real prints and three rescans
of one of them, and forcing a single winner would delete a physical object to satisfy the
form of the question.

What makes the judgment possible is not the thumbnails, it is the measurement, and it is
lifted from `compare_album_pair.py`:

  * every scan warped into one geometry, so the numbers compare like with like;
  * masks built from a blurred average of all of them -- flat, edge and shadow regions --
    so no single scan's own grain picks the region it is then judged on;
  * NOISE on the flat mask, DETAIL on the edge mask, and their ratio;
  * that noise split by WHERE it sits -- on the lit subject, or in the dark ground. The
    single figure hides the difference: the scan a family keeps can have the worst
    detail-per-noise in its group, with 80% of its grain in a dark background it renders
    deeper than its rivals. Grain behind a subject is cheap; grain
    on a face is not. The share does not say a scan is clean -- it says whether the ratio
    is being dragged down by the background, and the absolute subject figure is the one
    to compare;
  * DETAIL split into three scale bands. This matters more than the single figure: a scan
    that leads only in the fine band is winning on GRAIN, not on structure, and that
    distinction can overturn a verdict.

And two matched crops at the highest-gradient points of the picture, at native resolution,
side by side in scan order -- the thing that actually decides it.

    .venv\Scripts\python.exe scripts\make_group_review.py [N0123 ...]
"""
from __future__ import annotations

import base64
import csv
import html
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps

import _archive as A

sys.stdout.reconfigure(encoding="utf-8")

REPO = A.REPO
OUT = REPO / "artifacts" / "reviews" / "group-review.html"
SHOW_PX = 460          # each scan, whole frame
CROP = 300             # half-size of a native-resolution detail crop (3 scans or fewer)
CROP_MANY = 210        # five panels at 600 px wide cannot be seen at once; these can
LABEL_H = 22           # the name is drawn INTO the strip, so it survives sideways scrolling
REG_W = 1100           # width the homography is estimated at

Image.MAX_IMAGE_PIXELS = None      # one of these scans is 115.9 MP and is not a bomb


def rows(name: str) -> list[dict]:
    p = REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))) if p.exists() else []


def esc(s) -> str:
    return html.escape(str(s))


# `age46_m05_... .jpg` in a corrections row is the same scan as `m05_... .jpg` here.
STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")

# EXIF orientation -> the PIL rotation that applies it (PIL turns counter-clockwise)
_APPLY = {3: 180, 6: -90, 8: 90}


def corrections() -> dict[str, int]:
    """scan key -> the orientation its album copy was corrected to.

    This page reads ORIGINS, and origins are deliberately never modified, so a scan whose
    album copy has been fixed would otherwise keep showing up here on its side forever.
    The recorded correction is applied to the origin in memory instead.
    """
    out: dict[str, int] = {}
    for r in rows("orientation_corrections.csv"):
        v = str(r.get("new_orientation") or "").strip()
        if v.isdigit():
            out[STRIP.match(os.path.basename(r["file"])).group(1)] = int(v)
    return out


def full(path: str, orient: int | None = None) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path))
    if orient and im.getexif().get(0x0112) is None and orient in _APPLY:
        im = im.rotate(_APPLY[orient], expand=True)
    return np.asarray(im.convert("L"))


def small(a: np.ndarray, w: int = REG_W) -> np.ndarray:
    return cv2.resize(a, (w, max(1, int(a.shape[0] * w / a.shape[1]))),
                      interpolation=cv2.INTER_AREA)


def picture_area(img: np.ndarray, frac: float = 0.62) -> np.ndarray:
    """The paper margin is the sharpest edge in a scan; judging must stay inside the print."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    dy, dx = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    m[dy:h - dy, dx:w - dx] = 255
    return m


def energy(img: np.ndarray, mask: np.ndarray, s: float = 1.2) -> float:
    z = img.astype(np.float32)
    return float(((z - cv2.GaussianBlur(z, (0, 0), s)) ** 2)[mask].mean())


def band(img: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> float:
    z = img.astype(np.float32)
    return float(((cv2.GaussianBlur(z, (0, 0), lo)
                   - cv2.GaussianBlur(z, (0, 0), hi)) ** 2)[mask].mean())


def enc(img: np.ndarray, q: int = 88) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img,
                                         [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


def thumb(a: np.ndarray, px: int = SHOW_PX) -> str:
    if a.shape[1] > px:
        a = cv2.resize(a, (px, max(1, int(a.shape[0] * px / a.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    return enc(a, 80)


def measure_group(paths: list[str], names: list[str],
                  fixed: dict[str, int] | None = None) -> dict:
    """Warp every scan into the LARGEST one's frame, then measure on one set of masks.

    Two things here are not cosmetic. The reference is the highest-resolution scan, not
    the first: a group can hold a 9104x12736 scan beside a 3986x5952 one, and measuring in
    the small frame throws away exactly the advantage being judged. And any warp that shrinks
    its source is pre-blurred first -- resampling 9104 px down to 3986 without a filter
    aliases, and aliasing reads as detail: unfixed, it scores that scan 100 times its
    group, which is not sharpness, it is the same grain mistake in a new place.
    """
    fixed = fixed or {}
    I = [full(p, fixed.get(n)) for p, n in zip(paths, names)]
    ref = max(range(len(I)), key=lambda k: I[k].size)
    Sm = [small(a) for a in I]
    orb = cv2.ORB_create(6000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    K = [orb.detectAndCompute(s, picture_area(s)) for s in Sm]
    sc = [I[k].shape[1] / Sm[k].shape[1] for k in range(len(I))]

    W: list[np.ndarray | None] = [None] * len(I)
    inl = [0] * len(I)
    rot = [0.0] * len(I)        # how far each scan had to be turned to meet the reference
    W[ref], inl[ref] = I[ref], len(K[ref][0] or [])
    for j in range(len(I)):
        if j == ref:
            continue
        (ka, da), (kb, db) = K[ref], K[j]
        if da is None or db is None:
            continue
        m = sorted(bf.match(da, db), key=lambda x: x.distance)[:900]
        inl[j] = len(m)
        if len(m) < 15:
            continue
        s = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        d = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(s, d, cv2.RANSAC, 4.0)
        if H is None:
            inl[j] = 0
            continue
        Hf = np.diag([sc[j], sc[j], 1.0]) @ H @ np.diag([1 / sc[ref], 1 / sc[ref], 1.0])
        Hi = np.linalg.inv(Hf)
        rot[j] = float(np.degrees(np.arctan2(Hi[1, 0], Hi[0, 0])))
        src = I[j]
        scale = float(np.sqrt(abs(np.linalg.det(Hi[:2, :2]))))   # source px -> ref px
        if scale < 0.95:
            src = cv2.GaussianBlur(src, (0, 0), 0.5 / max(scale, 1e-3))
        W[j] = cv2.warpPerspective(src, Hi, (I[ref].shape[1], I[ref].shape[0]))
        inl[j] = int(mask.sum())

    good = [k for k, w in enumerate(W) if w is not None]
    pa = picture_area(small(I[ref], 1400))
    pa = cv2.resize(pa, (I[ref].shape[1], I[ref].shape[0]), interpolation=cv2.INTER_NEAREST)
    pa = cv2.erode(pa, np.ones((81, 81), np.uint8))
    valid = pa > 0
    for k in good[1:]:
        valid &= W[k] > 0

    out = {"names": names, "inliers": inl, "aligned": good, "reference": names[ref],
           "rot": rot, "native_mp": [a.size / 1e6 for a in I],
           "corrected": {n: fixed[n] for n in names if n in fixed},
           "coverage": float(valid.mean()), "rows": [], "crops": []}
    if valid.sum() < 50000:
        out["failed"] = "the scans do not share enough picture area to be measured together"
        out["thumbs"] = [thumb(a) for a in I]
        return out

    avg = cv2.GaussianBlur(sum(W[k].astype(np.float32) for k in good) / len(good), (0, 0), 4.0)
    gx = cv2.Sobel(avg, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(avg, cv2.CV_32F, 0, 1, ksize=5)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 6.0)
    flat = (grad < np.percentile(grad[valid], 20)) & valid
    edge = (grad > np.percentile(grad[valid], 95)) & valid
    dark = (avg < np.percentile(avg[valid], 25)) & valid
    flat_dark, flat_light = flat & dark, flat & ~dark

    for k in range(len(I)):
        if W[k] is None:
            out["rows"].append({"name": names[k], "aligned": False})
            continue
        n, dt = energy(W[k], flat), energy(W[k], edge)
        # WHERE the grain sits, which the single noise figure cannot say: a scan with five
        # times the noise can be carrying it in a dark background, and grain behind a subject costs far less than grain on a face.
        nd = energy(W[k], flat_dark) if flat_dark.any() else 0.0
        nl = energy(W[k], flat_light) if flat_light.any() else 0.0
        out["rows"].append({
            "name": names[k], "aligned": True,
            "noise": n, "detail": dt, "ratio": dt / max(n, 1e-6),
            "noise_dark": nd, "noise_light": nl,
            "in_shadow": nd / max(nd + nl, 1e-9),
            "shadow": energy(W[k], dark, 2.5),
            "fine": band(W[k], edge, 1.2, 3),
            "mid": band(W[k], edge, 3, 6),
            "coarse": band(W[k], edge, 6, 12),
            "mp": I[k].shape[0] * I[k].shape[1] / 1e6,
            "resample": I[ref].shape[1] / I[k].shape[1],
        })

    # two matched crops, at the two strongest structures far enough apart to differ
    cr = CROP if len(good) <= 3 else CROP_MANY
    g2 = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 40.0) * valid
    picks = []
    for _ in range(2):
        cy, cx = np.unravel_index(np.argmax(g2), g2.shape)
        cy = int(np.clip(cy, cr, I[ref].shape[0] - cr))
        cx = int(np.clip(cx, cr, I[ref].shape[1] - cr))
        picks.append((cy, cx))
        cv2.circle(g2, (cx, cy), int(0.22 * max(g2.shape)), 0, -1)
    for cy, cx in picks:
        strip = np.full((2 * cr + LABEL_H, len(good) * 2 * cr + 10 * (len(good) - 1)),
                        255, np.uint8)
        x = 0
        for k in good:
            c = W[k][cy - cr:cy + cr, cx - cr:cx + cr]
            strip[LABEL_H:LABEL_H + c.shape[0], x:x + c.shape[1]] = c
            cv2.putText(strip, names[k][:34], (x + 3, LABEL_H - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv2.LINE_AA)
            x += c.shape[1] + 10
        out["crops"].append(enc(strip, 92))
    out["thumbs"] = [thumb(W[k] if W[k] is not None else I[k]) for k in range(len(I))]
    return out


def main() -> None:
    P = rows("photographs.csv")
    neg = defaultdict(list)
    for r in P:
        neg[r["negative"]].append(r)
    # A named negative can be rendered on demand. Ids follow the scans that hold them
    # (stable_ids.py), but a retired id resolves to nothing: when in doubt, look the scan up
    # in data\scan_ids.csv -- it is the scan names that are stable.
    named = [a for a in sys.argv[1:] if a.startswith("N") and a[1:].isdigit()]
    open_negs = set(named) or {n for n, rs in neg.items()
                               if any(r["status"] == "plates-undecided" for r in rs)}
    if not open_negs:
        print("no negative holds an unjudged pair -- nothing to review.\n"
              "Pass a negative id (e.g. N0123) to render one anyway.")
        return
    print(f"{len(open_negs)} negatives still open: "
          f"{sum(len(neg[n]) for n in open_negs)} scans")

    fixed = corrections()
    M = {}
    for r in rows("duplicate_review_machine.csv"):
        M[tuple(sorted((r["scan_a"], r["scan_b"])))] = r

    cards = []
    for i, n in enumerate(sorted(open_negs), 1):
        members = sorted(neg[n], key=lambda r: r["scan"])
        names = [r["scan"] for r in members]
        print(f"  {n}: {len(names)} scans -- measuring")
        if named:
            print(f"      (ids are positional; {n} today means: {', '.join(names)})")
        g = measure_group([r["origin_path"] for r in members], names, fixed)
        cards.append((n, i, members, g))

    body = []
    for n, i, members, g in cards:
        # The thumbnails are the ALIGNED versions, so the eye compares like with like.
        # Where a scan had to be turned more than 45 degrees to meet the others, the file
        # itself is stored on its side -- that is a defect in the file, not in this page,
        # and it is said out loud rather than quietly corrected away.
        thumbs = []
        for j, (nm, t) in enumerate(zip(g["names"], g["thumbs"])):
            turned = abs(g["rot"][j]) > 45
            if nm in g.get("corrected", {}):
                tag = (f"<br><span class=ok>ориентация исправлена в альбоме "
                       f"(EXIF {g['corrected'][nm]}); оригинал лежит на боку и не тронут</span>")
            elif turned:
                tag = (f"<br><span class=warn>лежит на боку в файле "
                       f"(повёрнут на {g['rot'][j]:.0f}° при совмещении)</span>")
            else:
                tag = ""
            thumbs.append(f"<figure><img src='data:image/jpeg;base64,{t}'>"
                          f"<figcaption>{esc(nm)}<br>"
                          f"<span class=n>{g['native_mp'][j]:.1f} MP</span>{tag}</figcaption></figure>")
        thumbs = "".join(thumbs)

        if g.get("failed"):
            table = f"<p class=warn>{esc(g['failed'])}</p>"
        else:
            best_ratio = max((r["ratio"] for r in g["rows"] if r["aligned"]), default=0)
            low_noise = min((r["noise"] for r in g["rows"] if r["aligned"]), default=0)
            tr = []
            for r in g["rows"]:
                if not r["aligned"]:
                    tr.append(f"<tr><td>{esc(r['name'])}</td>"
                              f"<td colspan=10 class=warn>не совместился с остальными</td></tr>")
                    continue
                mark = lambda cond: " class=win" if cond else ""
                tr.append(
                    f"<tr><td>{esc(r['name'])}</td><td>{r['mp']:.1f}</td>"
                    f"<td{mark(r['noise'] == low_noise)}>{r['noise']:.2f}</td>"
                    f"<td>{r['detail']:.2f}</td>"
                    f"<td{mark(r['ratio'] == best_ratio)}>{r['ratio']:.2f}</td>"
                    f"<td>{r['noise_light']:.2f}</td>"
                    f"<td{' class=dim' if r['in_shadow'] > 0.75 else ''}>"
                    f"{r['in_shadow']*100:.0f}%</td>"
                    f"<td>{r['shadow']:.2f}</td>"
                    f"<td>{r['fine']:.2f}</td><td>{r['mid']:.2f}</td><td>{r['coarse']:.2f}</td>"
                    f"<td{' class=warn' if r['resample'] > 1.3 else ''}>"
                    f"{r['resample']:.2f}x</td></tr>")
            spread = max((r["resample"] for r in g["rows"] if r["aligned"]), default=1)
            caveat = ("" if spread <= 1.3 else
                      "<p class=warn>Эталон здесь крупнее остальных в %.1f раза. "
                      "Другие сканы растянуты до его сетки, и мелкие детали у них "
                      "подавлены самим растяжением — числа в столбцах «детали» и «мелк.» "
                      "занижены не потому, что скан хуже. Решать здесь надо по вырезкам "
                      "внизу, а не по таблице.</p>" % spread)
            table = (
                "<table><thead><tr><th>скан</th><th>MP</th><th>шум</th><th>детали</th>"
                "<th>детали/шум</th><th>шум в светах</th><th>зерно в тенях</th>"
                "<th>тени</th><th>мелк. 1.2-3</th><th>средн. 3-6</th>"
                "<th>крупн. 6-12</th><th>растянут</th></tr></thead><tbody>" + "".join(tr) + "</tbody></table>"
                "<p class=hint>Меньше шум — лучше. Выше детали/шум — лучше. "
                "Но если скан ведёт только в столбце «мелк.», он выигрывает на ЗЕРНЕ, "
                "а не на структуре: зерно живёт в негативе и одинаково у всех отпечатков.</p>"
                "<p class=hint>Столбец «шум» складывает всё зерно вместе, а важно, ГДЕ оно. "
                "<b>«Шум в светах»</b> — зерно на лице и одежде: вот оно и портит снимок, "
                "и сравнивать надо именно эти числа. "
                "<b>«Зерно в тенях»</b> — какая доля всего зерна приходится на тёмный фон. "
                "Высокая доля НЕ значит, что скан чистый: она значит только, что столбец "
                "«детали/шум» у него занижен фоном, а не лицом, и решать по нему нельзя. "
                "Порог 75% поставлен по одному случаю (80% против 61%) и держится "
                "на нём одном — это подсказка, а не правило.</p>"
                + caveat)

        crops = "".join(
            f"<div class=crop><img src='data:image/jpeg;base64,{c}'></div>"
            for c in g["crops"])
        pairs = []
        for a in range(len(g["names"])):
            for b in range(a + 1, len(g["names"])):
                m = M.get(tuple(sorted((g["names"][a], g["names"][b]))))
                if m:
                    pairs.append(f"{g['names'][a][:26]} / {g['names'][b][:26]}: "
                                 f"{m['kind']} {m['blobs']} пятен, невязка {m['residual']}")
        toggles = "".join(
            f"<label class=tg data-scan=\"{esc(nm)}\"><input type=checkbox checked>"
            f"<span>{esc(nm)}</span></label>" for nm in g["names"])

        body.append(f"""
<div class=card data-id="{esc(n)}">
  <div class=hd><b>{i}</b> <span class=n>{esc(n)} — {len(g['names'])} сканов</span>
    <span class=cl>совмещено {len(g['aligned'])}/{len(g['names'])} · эталон {esc(g.get('reference',''))} · площадь {g['coverage']*100:.0f}%</span></div>
  <p class=hint>Снимки показаны СОВМЕЩЁННЫМИ — приведены к одной геометрии, чтобы
  сравнивать одно с одним. Чёрные поля по краям — там, где скан не покрывает эталон.</p>
  <div class=row>{thumbs}</div>
  {table}
  <h4>Одно и то же место снимка, в натуральном разрешении. Имя подписано на каждой вырезке</h4>
  {crops}
  <details><summary>попарные измерения</summary><pre>{esc(chr(10).join(pairs))}</pre></details>
  <div class=ans><span class=lbl>оставить:</span>{toggles}
    <button class=keepall>оставить все</button>
    <span class=mark>не отмечено</span></div>
  <div class=ans2><input class=note placeholder="почему (необязательно)"></div>
</div>""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("{{BODY}}", "".join(body))
                       .replace("{{N}}", str(len(cards))), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


PAGE = r"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Группы снимков: что оставить</title>
<style>
:root{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--ink:#1a1a18;--mut:#666;--bd:#ddd;--acc:#33517A;--win:#2a7}
@media(prefers-color-scheme:dark){:root{--bg:#17181A;--card:#212327;--ink:#E4E3DE;--mut:#A2A49F;--bd:#3A3D42;--acc:#7FA3D6;--win:#5c9}}
body{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--ink)}
h1{font-size:20px;margin:0 0 4px} h4{margin:16px 0 6px;font-size:12px;color:var(--mut);font-weight:600}
.lede{color:var(--mut);max-width:72em;margin:0 0 14px}
#bar{position:sticky;top:0;background:var(--bg);padding:9px 0;z-index:9;border-bottom:1px solid var(--bd);margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:9px;padding:12px 14px;margin-bottom:18px}
.card.done{border-color:var(--acc)}
.hd{display:flex;gap:10px;align-items:baseline;margin-bottom:10px;flex-wrap:wrap}
.n{color:var(--mut);font-size:12px}
.cl{margin-left:auto;color:var(--mut);font-size:11px;font-family:ui-monospace,monospace}
.row{display:flex;gap:10px;flex-wrap:wrap}
figure{margin:0} figure img{max-width:100%;border-radius:4px;display:block}
figcaption{font-size:11px;color:var(--mut);margin-top:3px;max-width:460px;word-break:break-all}
table{border-collapse:collapse;margin-top:12px;font-size:12px;font-family:ui-monospace,monospace}
th,td{border:1px solid var(--bd);padding:3px 7px;text-align:right}
th:first-child,td:first-child{text-align:left;font-family:system-ui,sans-serif}
th{color:var(--mut);font-weight:600}
td.win{color:var(--win);font-weight:700}
.warn{color:#b45}
td.dim{color:var(--mut)}
.ok{color:var(--win)}
.hint{font-size:11.5px;color:var(--mut);max-width:64em;margin:6px 0 0}
.crop{overflow-x:auto;margin-bottom:8px;border:1px solid var(--bd);border-radius:4px}
.crop img{display:block;max-width:none}
details{margin-top:10px} summary{cursor:pointer;color:var(--mut);font-size:12px}
pre{font-size:11.5px;color:var(--mut);white-space:pre-wrap}
.ans{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;
border-top:1px solid var(--bd);padding-top:10px}
.lbl{color:var(--mut);font-size:12px}
.tg{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--bd);border-radius:5px;
padding:4px 9px;cursor:pointer;font-size:12px}
.tg.off{opacity:.5;text-decoration:line-through;border-color:#b45;color:#b45}
.tg:hover{border-color:var(--acc)}
.mark{margin-left:auto;font-size:12px;color:var(--mut)}
.card.done .mark{color:var(--win);font-weight:600}
.ans2{display:flex;margin-top:8px}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}
button.primary:disabled{background:var(--card);color:var(--mut);border-color:var(--bd);font-weight:400;cursor:default}
#bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input.note{flex:1;min-width:200px;padding:5px 7px;border:1px solid var(--bd);border-radius:5px;font:inherit;background:var(--bg);color:var(--ink)}
button{padding:5px 12px;border:1px solid var(--bd);background:var(--card);color:var(--ink);border-radius:5px;cursor:pointer}
#out{white-space:pre;font-family:ui-monospace,monospace;font-size:12px;background:var(--card);
border:1px solid var(--bd);border-radius:6px;padding:12px;margin-top:12px;display:none;overflow-x:auto}
</style>
<h1>Группы снимков: что оставить</h1>
<p class=lede>Если спрашивать парами, внутри группы из нескольких сканов это неверный вопрос:
ответы образовали цикл и применить их было нельзя. Здесь вопрос задаётся один раз на всю группу —
<b>у каждого скана галочка «оставить»</b>. Снимите её с тех, что лишние — снятая галочка
станет красной и зачёркнутой. Если в группе всё остаётся, нажмите «оставить все»:
группа тоже считается отвеченной. Когда закончите — кнопка
<b>«Скопировать ответы»</b> наверху, и вставьте их мне в чат. Можно оставить и два, и три:
группа из пяти вполне может быть двумя настоящими отпечатками и тремя пересканами одного из них.
Ничего не удаляется по галочке — она только записывается.</p>
<div id=bar><span id=cnt></span>
<button id=copy onclick=copyAll() class=primary>Скопировать ответы</button>
<button onclick=show()>показать текстом</button>
<span id=nostore hidden class=warn>Браузер не даёт этой странице сохранять ответы
(так бывает при открытии файлом). Отвечать можно, всё работает — но не закрывайте
вкладку, пока не скопируете ответы.</span></div>
{{BODY}}
<pre id=out></pre>
<script>
const K='group-review-v1';
// Opened as a file:// page, Chrome may refuse storage outright and THROW here. Unguarded,
// that killed the whole script on the first line: every checkbox went dead, the counter
// never moved, and the button did nothing. The answers must not depend on storage.
let store=true, st={};
try{ st=JSON.parse(localStorage.getItem(K)||'{}'); }catch(e){ store=false; st={}; }
function save(){ if(store){ try{ localStorage.setItem(K,JSON.stringify(st)); }catch(e){ store=false; warn(); } } updateCount(); }
function warn(){ const b=document.getElementById('nostore'); if(b) b.hidden=false; }
if(!store) warn();

const cards=[...document.querySelectorAll('.card')];
cards.forEach(c=>{
  const id=c.dataset.id, note=c.querySelector('.note');
  st[id]=st[id]||{drop:[],note:'',touched:false};
  const paint=()=>{
    c.querySelectorAll('.tg').forEach(t=>{
      const off=st[id].drop.includes(t.dataset.scan);
      t.classList.toggle('off',off); t.querySelector('input').checked=!off;});
    c.classList.toggle('done',st[id].touched);
    const m=c.querySelector('.mark');
    if(m) m.textContent = st[id].touched
      ? (st[id].drop.length? 'убрать: '+st[id].drop.length+' · оставить: '+(c.querySelectorAll('.tg').length-st[id].drop.length)
                           : 'оставить все')
      : 'не отмечено';
    if(note.value!==(st[id].note||'')) note.value=st[id].note||'';};
  c.querySelectorAll('.tg').forEach(t=>{
    t.querySelector('input').onchange=e=>{
      const sc=t.dataset.scan, d=st[id].drop;
      if(e.target.checked){st[id].drop=d.filter(x=>x!==sc);}
      else if(!d.includes(sc)){d.push(sc);}
      st[id].touched=true;save();paint();};});
  const keepAll=c.querySelector('.keepall');
  if(keepAll) keepAll.onclick=()=>{st[id].drop=[];st[id].touched=true;save();paint();};
  note.addEventListener('input',()=>{st[id].note=note.value;st[id].touched=true;save();});
  paint();
});

const TAB=String.fromCharCode(9), NL=String.fromCharCode(10);
function tsv(){
  const L=[];
  cards.forEach(c=>{const id=c.dataset.id,s=st[id];if(!s||!s.touched)return;
    c.querySelectorAll('.tg').forEach(t=>{
      L.push([id,t.dataset.scan,s.drop.includes(t.dataset.scan)?'drop':'keep',s.note||''].join(TAB));});});
  return L.length? ['negative','scan','verdict','note'].join(TAB)+NL+L.join(NL) : '';
}
function updateCount(){
  const done=Object.values(st).filter(x=>x.touched).length;
  document.getElementById('cnt').textContent=done+' из '+cards.length+' групп отмечено';
  const b=document.getElementById('copy');
  b.disabled=!done; b.textContent=done? 'Скопировать ответы ('+done+')' : 'Ещё нечего копировать';
}
function show(){const o=document.getElementById('out');
  o.textContent=tsv()||'ничего не отмечено'; o.style.display='block'; o.scrollIntoView({behavior:'smooth'});}
// file:// is not a secure context, so navigator.clipboard is unavailable there and the
// old flow -- scroll up, click, select the text by hand, copy -- was the actual problem.
function copyAll(){
  const t=tsv(); if(!t) return;
  const done=()=>{const b=document.getElementById('copy');const o=b.textContent;
    b.textContent='Скопировано ✓'; setTimeout(()=>{b.textContent=o;},1600);};
  if(navigator.clipboard && window.isSecureContext){
    navigator.clipboard.writeText(t).then(done,()=>fallback(t,done));
  } else fallback(t,done);
}
function fallback(t,done){
  const ta=document.createElement('textarea');
  ta.value=t; ta.style.position='fixed'; ta.style.top='-1000px';
  document.body.appendChild(ta); ta.select();
  let ok=false; try{ ok=document.execCommand('copy'); }catch(e){}
  document.body.removeChild(ta);
  if(ok) done(); else show();
}
updateCount();</script>"""


if __name__ == "__main__":
    main()
