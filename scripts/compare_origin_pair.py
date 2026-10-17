r"""Judge two ORIGIN scans of what looks like one photograph, and render it for a person.

`compare_album_pair.py` answers this question for two files inside `by-person\`, resolved
through MANIFEST.csv. Two ORIGINS that were never filed are not in the manifest, so the
same measurements need a path-addressed entry point.
The measurements themselves are the archive's existing ones, not new ones:

  * RESCAN OR SEPARATE PRINT -- split_rescan_vs_print.damage(). Grain lives in the negative,
    so shared texture proves nothing; what separates two physical prints is damage they took
    AFTER printing (foxing, dust, scratches). Align, normalise out density, difference, count
    localised blobs. >= 4 blobs, or >= 3.0 area per 1e4 px, means two objects.
  * WHICH IS THE KEEPER -- in this order:
      1. a scan that CROPS the other's content away cannot be the keeper, whatever else it
         leads on. Containment below 0.90 is a crop.
      2. among scans that do not crop, the COARSE band decides. A file leading only the fine
         band is winning on grain, not on structure.
      3. settled only where detail-per-noise agrees with the coarse band. Where those two
         disagree the answer needs a person, so the page says so rather than picking.

Resolution is reported and never weighted: the family's own choices often delete the LARGER
scan, sometimes one 58% larger and leading nothing. A pixel count is not detail.

NOTHING IS DELETED, MOVED OR WRITTEN except the page itself. This decides nothing; it shows.

    .venv\Scripts\python.exe scripts\compare_origin_pair.py A.jpg B.jpg

A bare filename is looked up under every origin root.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import os
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import split_rescan_vs_print as S

REPO = pathlib.Path(__file__).resolve().parent.parent
from _paths import ORIGIN_ROOTS  # noqa: E402

# NAMED AFTER THE PAIR, because a fixed filename silently overwrites the last comparison --
# including a committed page that a decision rested on.
OUTDIR = REPO / "artifacts" / "reviews"
Image.MAX_IMAGE_PIXELS = None

CONTAIN_MIN = 0.90      # below this, a scan crops the other's content away


def find(name: str) -> pathlib.Path:
    p = pathlib.Path(name)
    if p.exists():
        return p
    for root in ORIGIN_ROOTS:
        for d, _, fs in os.walk(root):
            if p.name in fs:
                return pathlib.Path(d) / p.name
    raise SystemExit(f"not found: {name}")


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def gray(p: pathlib.Path) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("L"))


def rgb(p: pathlib.Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def small(a: np.ndarray, w: int = 1000) -> np.ndarray:
    return cv2.resize(a, (w, int(a.shape[0] * w / a.shape[1])), interpolation=cv2.INTER_AREA)


def b64(im: Image.Image, px: int, q: int = 86) -> str:
    im = im.copy()
    im.thumbnail((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def b64cv(a: np.ndarray, px: int) -> str:
    return b64(Image.fromarray(a), px)


def align(A_: np.ndarray, B_: np.ndarray):
    """B warped into A's frame, plus the homography and inlier count."""
    orb = cv2.ORB_create(6000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    sa, sb = small(A_), small(B_)
    (ka, da), (kb, db) = orb.detectAndCompute(sa, None), orb.detectAndCompute(sb, None)
    m = sorted(bf.match(da, db), key=lambda x: x.distance)[:900]
    src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    ca, cb = A_.shape[1] / sa.shape[1], B_.shape[1] / sb.shape[1]
    Hf = np.diag([cb, cb, 1.0]) @ H @ np.diag([1 / ca, 1 / ca, 1.0])
    W = cv2.warpPerspective(B_, np.linalg.inv(Hf), (A_.shape[1], A_.shape[0]))
    return W, Hf, int(mask.sum())


def containment(A_: np.ndarray, B_: np.ndarray, Hf: np.ndarray) -> tuple[float, float]:
    """How much of each scan's content the other one still holds. Rule 1's measure."""
    ha, wa = A_.shape
    hb, wb = B_.shape
    ca = np.float32([[0, 0], [wa, 0], [wa, ha], [0, ha]]).reshape(-1, 1, 2)
    cb = np.float32([[0, 0], [wb, 0], [wb, hb], [0, hb]]).reshape(-1, 1, 2)
    b_in_a = cv2.perspectiveTransform(cb, np.linalg.inv(Hf))
    a_in_b = cv2.perspectiveTransform(ca, Hf)

    def frac(poly, w, h):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.int32(poly).reshape(-1, 2)], 1)
        return float(m.mean())
    return frac(b_in_a, wa, ha), frac(a_in_b, wb, hb)


def main() -> None:
    args = sys.argv[1:]
    names = [a for a in args if not a.startswith("--")]
    if len(names) != 2:
        raise SystemExit("give exactly two files")
    pa, pb = (find(n) for n in names)
    out = OUTDIR / f"pair-{pa.stem}-vs-{pb.stem}.html"
    print(f"A  {pa}\nB  {pb}\n")

    ims = [rgb(pa), rgb(pb)]
    meta = []
    for p, im in zip((pa, pb), ims):
        meta.append(dict(name=p.name, path=str(p), w=im.width, h=im.height,
                         mp=im.width * im.height / 1e6, bytes=p.stat().st_size, sha=sha(p)))
        print(f"{p.name:24} {im.width}x{im.height} = {meta[-1]['mp']:.1f} MP   "
              f"{meta[-1]['bytes']/1e6:.2f} MB   sha {meta[-1]['sha'][:16]}")
    identical = meta[0]["sha"] == meta[1]["sha"]
    print(f"byte-identical: {identical}\n")

    # ---- same negative, and one object or two? -------------------------------------------
    La, Lb = S.load(pa), S.load(pb)
    dmg = S.damage(La, Lb)
    if dmg is None:
        blobs, area, kind, conf = None, None, "undecided", "could not align"
    else:
        blobs, area = dmg
        kind = ("separate-print" if blobs >= S.BLOB_MIN or area >= S.BLOB_AREA_MIN
                else "rescan")
        conf = S.confidence(blobs)
    print(f"damage blobs {blobs}, area {area} -> {kind} ({conf})")

    # ---- align at full size, then measure only inside the picture ------------------------
    A_, B_ = gray(pa), gray(pb)
    W, Hf, inliers = align(A_, B_)
    print(f"aligned: {inliers} inliers")
    c_a, c_b = containment(A_, B_, Hf)
    print(f"containment: {meta[0]['name']} holds {c_a:.3f} of the other, "
          f"{meta[1]['name']} holds {c_b:.3f}")

    pic = S.picture_area(small(A_, 1400))
    pic = cv2.resize(pic.astype(np.uint8), (A_.shape[1], A_.shape[0]),
                     interpolation=cv2.INTER_NEAREST)
    pic = cv2.erode(pic, np.ones((81, 81), np.uint8))
    valid = (pic > 0) & (W > 0)
    print(f"picture area judged: {100*valid.mean():.1f}% of the frame")

    avg = cv2.GaussianBlur((A_.astype(np.float32) + W.astype(np.float32)) / 2, (0, 0), 4.0)
    gx = cv2.Sobel(avg, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(avg, cv2.CV_32F, 0, 1, ksize=5)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 6.0)
    flat = (grad < np.percentile(grad[valid], 20)) & valid
    edge = (grad > np.percentile(grad[valid], 95)) & valid

    def energy(img, mk, s=1.2):
        z = img.astype(np.float32)
        return float(((z - cv2.GaussianBlur(z, (0, 0), s)) ** 2)[mk].mean())

    def band(img, mk, lo, hi):
        z = img.astype(np.float32)
        return float(((cv2.GaussianBlur(z, (0, 0), lo)
                       - cv2.GaussianBlur(z, (0, 0), hi)) ** 2)[mk].mean())

    frames = [A_, W]
    for i, m in enumerate(meta):
        m["noise"] = energy(frames[i], flat)
        m["detail"] = energy(frames[i], edge)
        m["ratio"] = m["detail"] / max(m["noise"], 1e-6)
        m["fine"] = band(frames[i], edge, 1.2, 3)
        m["mid"] = band(frames[i], edge, 3, 6)
        m["coarse"] = band(frames[i], edge, 6, 12)
        m["contain"] = (c_a, c_b)[i]
    print(f"\n{'file':26} {'NOISE':>9} {'DETAIL':>9} {'ratio':>7} "
          f"{'fine':>9} {'mid':>9} {'coarse':>9}  contain")
    for m in meta:
        print(f"{m['name']:26} {m['noise']:9.2f} {m['detail']:9.2f} {m['ratio']:7.2f} "
              f"{m['fine']:9.2f} {m['mid']:9.2f} {m['coarse']:9.2f}  {m['contain']:.3f}")

    # ---- the keeper rule, in its order ---------------------------------------------------
    crops = [m for m in meta if m["contain"] < CONTAIN_MIN]
    coarse_win = max(meta, key=lambda m: m["coarse"])
    ratio_win = max(meta, key=lambda m: m["ratio"])
    if identical:
        verdict, why = "identical", "The two files are byte-identical. Either one may go."
    elif len(crops) == 1:
        keep = [m for m in meta if m is not crops[0]][0]
        verdict, why = keep["name"], (
            f"Rule 1: {crops[0]['name']} holds only {crops[0]['contain']:.3f} of the other's "
            f"content — it crops the frame, so it cannot be the keeper whatever else it leads on.")
    elif coarse_win is ratio_win:
        verdict, why = coarse_win["name"], (
            f"Rule 2 and rule 3 agree: {coarse_win['name']} leads the coarse band "
            f"({coarse_win['coarse']:.2f} against {min(m['coarse'] for m in meta):.2f}) and also "
            f"leads detail-per-noise ({coarse_win['ratio']:.2f}). Settled.")
    else:
        verdict, why = "needs a person", (
            f"The coarse band picks {coarse_win['name']} and detail-per-noise picks "
            f"{ratio_win['name']}. When those two disagree the answer needs "
            f"a person, so this one is not settled by measurement.")
    print(f"\nKEEPER: {verdict}\n  {why}")

    # ---- pictures for the page -----------------------------------------------------------
    # where they differ, at a size that can be seen
    n1, n2 = S.local_norm(La.astype(np.float32)), S.local_norm(S.load(pb).astype(np.float32))
    # the damage map is computed by S at its own scale; show the plain aligned difference
    d = cv2.absdiff(cv2.GaussianBlur(A_, (0, 0), 1.5), cv2.GaussianBlur(W, (0, 0), 1.5))
    d = np.where(valid, d, 0)
    dv = np.clip(d.astype(np.float32) * (200.0 / max(np.percentile(d[valid], 99.5), 1)), 0, 255)
    diff_img = Image.fromarray(dv.astype(np.uint8))

    # a 100% crop where the picture has the most structure, same place in both
    g2 = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 40.0) * valid
    cy, cx = np.unravel_index(np.argmax(g2), g2.shape)
    R = 420
    cy = int(np.clip(cy, R, A_.shape[0] - R))
    cx = int(np.clip(cx, R, A_.shape[1] - R))
    detail = [Image.fromarray(f[cy - R:cy + R, cx - R:cx + R]) for f in frames]

    rows = "".join(
        f"<tr><th>{html.escape(m['name'])}</th>"
        f"<td>{m['w']}×{m['h']}</td><td>{m['mp']:.1f} MP</td><td>{m['bytes']/1e6:.2f} MB</td>"
        f"<td>{m['contain']:.3f}</td><td>{m['noise']:.2f}</td><td>{m['detail']:.2f}</td>"
        f"<td class='{'win' if m is ratio_win else ''}'>{m['ratio']:.2f}</td>"
        f"<td>{m['fine']:.2f}</td><td>{m['mid']:.2f}</td>"
        f"<td class='{'win' if m is coarse_win else ''}'>{m['coarse']:.2f}</td></tr>"
        for m in meta)

    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(meta[0]['name'])} vs {html.escape(meta[1]['name'])}</title><style>
:root{{--bg:#f7f6f3;--card:#fff;--ink:#1b1b1a;--dim:#6b6862;--line:#e0ddd6;--accent:#2f5d8a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif}}
header{{padding:26px 22px 18px;border-bottom:1px solid var(--line);background:var(--card)}}
h1{{margin:0 0 6px;font-size:22px}}
.sub{{color:var(--dim);max-width:80ch}}
main{{padding:20px 22px 60px;max-width:1700px}}
h2{{font-size:15px;margin:30px 0 10px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}}
.verdict{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:7px;padding:15px 18px;margin:16px 0 4px}}
.verdict b{{font-size:18px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:13px}}
.fn{{font:12px/1.4 ui-monospace,Consolas,monospace;color:#514d47;word-break:break-all;margin:0 0 9px}}
img{{display:block;width:100%;height:auto;border-radius:5px;border:1px solid var(--line);
background:#eee;cursor:zoom-in}}
img.zoom{{position:fixed;inset:12px;z-index:60;width:auto;height:auto;
max-width:calc(100vw - 24px);max-height:calc(100vh - 24px);margin:auto;object-fit:contain;
background:#000;cursor:zoom-out;box-shadow:0 0 0 100vmax rgba(0,0,0,.85)}}
table{{border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:8px;overflow:hidden;font-variant-numeric:tabular-nums;width:100%}}
th,td{{padding:8px 11px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
thead th{{background:#fbfaf7;font-size:12px;text-transform:uppercase;letter-spacing:.05em;
color:var(--dim)}}
tbody th{{text-align:left;font:12px ui-monospace,Consolas,monospace}}
td.win{{background:#e8f1e9;font-weight:700}}
.note{{color:var(--dim);max-width:80ch;margin:9px 0 0}}
@media (max-width:900px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>{html.escape(meta[0]['name'])} &nbsp;vs&nbsp; {html.escape(meta[1]['name'])}</h1>
<div class="sub"><p>Two origin scans that look like one photograph. The question is whether they are one print scanned twice or two prints from one
negative — and if one is redundant, which scan to keep. <b>Nothing has been deleted, moved or
written.</b> Click any picture to fill the screen.</p></div></header><main>

<div class="verdict"><b>{html.escape(kind)}</b> — {html.escape(conf)}.
{blobs if blobs is not None else '?'} localised damage blobs, area {area if area is not None else '?'}
per 10,000 px of overlap. The cut is {S.BLOB_MIN} blobs or {S.BLOB_AREA_MIN} area.
<p class="note">Grain lives in the negative, so shared texture proves nothing — two prints from
one negative share it. What separates two physical objects is damage taken <i>after</i> printing:
foxing, dust, scratches. That is what is counted here.</p></div>

<div class="verdict"><b>Keeper: {html.escape(verdict)}</b>
<p class="note">{html.escape(why)}</p></div>

<h2>The two scans</h2>
<div class="two">
 <div class="card"><p class="fn">{html.escape(meta[0]['name'])}</p>
  <img src="data:image/jpeg;base64,{b64(ims[0], 1500)}" alt=""></div>
 <div class="card"><p class="fn">{html.escape(meta[1]['name'])}</p>
  <img src="data:image/jpeg;base64,{b64(ims[1], 1500)}" alt=""></div>
</div>

<h2>Where they differ, aligned</h2>
<div class="card"><p class="note" style="margin:0 0 9px">The second scan warped onto the first
and subtracted, inside the picture area only, then stretched so anything non-black is visible.
A rescan shows edges and glass dust. Two prints show blobs that sit in one and not the other.</p>
<img src="data:image/jpeg;base64,{b64(diff_img, 1400)}" alt=""></div>

<h2>Same detail, both scans, at full resolution</h2>
<div class="two">
 <div class="card"><p class="fn">{html.escape(meta[0]['name'])}</p>
  <img src="data:image/jpeg;base64,{b64(detail[0], 840)}" alt=""></div>
 <div class="card"><p class="fn">{html.escape(meta[1]['name'])} (aligned)</p>
  <img src="data:image/jpeg;base64,{b64(detail[1], 840)}" alt=""></div>
</div>

<h2>Measurements</h2>
<table><thead><tr><th style="text-align:left">file</th><th>pixels</th><th>size</th><th>bytes</th>
<th>contains</th><th>noise</th><th>detail</th><th>detail/noise</th>
<th>fine 1.2–3</th><th>mid 3–6</th><th>coarse 6–12</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note"><b>Read it in the rule's order.</b> <i>contains</i> first: below
{CONTAIN_MIN:.2f} a scan crops the other's content away and cannot be the keeper, whatever else
it leads. Then the <i>coarse</i> band — a file leading only the <i>fine</i> band is winning on
grain, not on structure. <i>detail/noise</i> confirms; where it disagrees with coarse, the
answer needs a person. Resolution is shown and never weighted: a pixel count is not detail.</p>
<p class="note">Alignment: {inliers} inliers. Picture area judged: {100*valid.mean():.1f}% of the
frame — the white paper margin is the sharpest edge in any scan, so every measurement above
excludes it.</p>
</main>
<script>document.addEventListener('click',e=>{{
 if(e.target.tagName==='IMG') e.target.classList.toggle('zoom');}});</script>
</body></html>"""
    out.write_bytes(doc.encode("utf-8"))
    print(f"\nwrote {out.relative_to(REPO)}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
