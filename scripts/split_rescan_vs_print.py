r"""Split the album's same-negative groups into rescans and separate physical prints.

`album_same_negative.csv` lists the photographs that exist in `by-person\` more than once. That
bundles two different things, and they are not the same decision:

  * **a rescan** -- one physical print digitised twice, once per pass. Genuinely redundant:
    the second file adds no object and no information.
  * **a separate print** -- somebody ordered two copies from one negative, decades ago. Two
    different objects, aged differently. `Unknown\` kept `(2) 0006` over `(2) 0007` precisely
    because the prints were not interchangeable.

**Two discriminators were tried. The obvious one is wrong.** Correlating *fine detail* after
alignment looks right -- surely two scans of one print share texture that two prints do not --
and it fails: known different prints score 0.54 to 0.93, because photographic grain lives in
the NEGATIVE and every print from it carries the same grain. The measure is dominated by the
picture.

What actually distinguishes two prints is damage they took **after** they were printed --
foxing, dust, scratches, creases. That is sparse and localised, a fraction of a percent of
the pixels, which is exactly why a global correlation cannot see it. So: align, remove
density by local contrast normalisation (a darkroom difference must not register as damage),
difference, and count connected blobs.

    known different prints      10, 10, 28 blobs
    synthetic rescan             0,  0,  0 blobs   (same file resized, rotated 0.4deg, JPEG q88)

The synthetic control is idealised -- a real second pass adds its own noise and its own dust
on the glass -- so the cut sits at 4 blobs, well clear of zero and well below ten.

**`--apply` MERGES; it used to overwrite, and that was a data-loss bug.** The output file is
not a pure derivation: three of its rows are `family-verdict`, entered by hand from what
people could tell about their own prints, and it accumulates verdicts across passes over
pairs a single run may not judge -- an unreadable scan is skipped, and a group whose
membership changed yields a different pair set. A rerun once replaced 313 judged rows with
239, turned 102 decided pairs back into `undecided`, and deleted all three family verdicts;
only the commit saved them. The write now keeps a family verdict over any machine verdict,
keeps a decided row over a fresh `undecided` (a run that could not judge a pair has learned
nothing about it), and carries through rows it did not produce. Pairs are matched on the
scan key with the age stripped, so an age rename does not duplicate a row.

    .venv\Scripts\python.exe scripts\split_rescan_vs_print.py [--apply]
"""
from __future__ import annotations

import csv
import re
import os
import sys
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image

import _archive as A
from find_same_negative_album import picture_area

sys.stdout.reconfigure(encoding="utf-8")

APPLY = "--apply" in sys.argv
SRC = A.REPO / "data" / "album_same_negative.csv"
OUT = A.REPO / "data" / "album_print_vs_rescan.csv"

# Pair identity that survives an age rename. `age40_m03_... .jpg` becoming
# `age30_m03_... .jpg` must not read as a different pair, or a merge would duplicate
# every row a rename touched. Same rule as build_photographs.py: strip the age, keep
# the scan key.
STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")


def pair_key(r: dict) -> tuple:
    a = STRIP.match(os.path.basename(r["reference"])).group(1)
    b = STRIP.match(os.path.basename(r["other"])).group(1)
    return (r["group"], *sorted((a, b)))

W = 1400
BLOB_MIN = 4          # >= this many localised differences => two physical objects
BLOB_AREA_MIN = 3.0   # or this much blob area per 1e4 pixels of overlap


def load(p) -> np.ndarray:
    im = Image.open(p)
    im.draft("L", (W * 2, W * 2))
    im = im.convert("L")
    return np.asarray(im.resize((W, max(1, int(im.height * W / im.width))), Image.LANCZOS))


def local_norm(z: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Remove local density. A print made darker in the darkroom must not look like damage."""
    z = z.astype(np.float32)
    mu = cv2.GaussianBlur(z, (0, 0), sigma)
    sd = np.sqrt(cv2.GaussianBlur((z - mu) ** 2, (0, 0), sigma)) + 3.0
    return (z - mu) / sd


def hf_energy(z: np.ndarray) -> float:
    return float(np.mean((z.astype(np.float32) -
                          cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.2)) ** 2))


def match_sharpness(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blur the sharper copy until both carry the same high-frequency energy.

    Two passes rarely scan at the same effective resolution, and local contrast
    normalisation turns the sharper one's grain into dozens of fake "damage" blobs -- one
    checked pair scored 19 on nothing but a sharpness mismatch. Equalise first, then any
    blob that survives is a mark on one object and not the other."""
    ex, ey = hf_energy(x), hf_energy(y)
    if max(ex, ey) < 1e-6:
        return x, y
    sharper, target = (x, ey) if ex > ey else (y, ex)
    lo, hi = 0.0, 4.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if hf_energy(cv2.GaussianBlur(sharper, (0, 0), max(mid, 1e-3))) > target:
            lo = mid
        else:
            hi = mid
    out = cv2.GaussianBlur(sharper, (0, 0), max((lo + hi) / 2, 1e-3))
    return (out, y) if ex > ey else (x, out)


def damage(Ai: np.ndarray, Bi: np.ndarray) -> tuple[int, float] | None:
    orb = cv2.ORB_create(8000)
    ka, da = orb.detectAndCompute(Ai, picture_area(Ai))
    kb, db = orb.detectAndCompute(Bi, picture_area(Bi))
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db),
               key=lambda x: x.distance)[:1200]
    if len(m) < 15:
        return None
    src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None
    warp = cv2.warpPerspective(Ai, H, (Bi.shape[1], Bi.shape[0]), borderValue=0)
    ok = ((warp > 0) & (picture_area(Bi) > 0)).astype(np.uint8)
    if ok.sum() < 20000:
        return None
    w2, b2 = match_sharpness(warp, Bi)
    d = cv2.GaussianBlur(np.abs(local_norm(w2) - local_norm(b2)), (0, 0), 1.5) * ok
    thr = max(float(np.percentile(d[ok > 0], 99.7)), 0.9)
    bw = cv2.morphologyEx(((d > thr) & (ok > 0)).astype(np.uint8),
                          cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    big = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 25]
    area = sum(s[cv2.CC_STAT_AREA] for s in big) / max(int(ok.sum()), 1) * 1e4
    return len(big), float(area)


def confidence(blobs: int | str) -> str:
    """How much the blob count is worth.

    Checked by eye: the 10+ band shows real marks (a scratch in one copy, foxing absent from
    the other), and 0 means the same scan filed twice. The 4-9 band is NOT reliable -- of
    three sampled, one showed genuine damage, one was scanner noise and one was ambiguous --
    so it is labelled weak rather than counted as a finding."""
    if blobs == "":
        return "undecided"
    n = int(blobs)
    return ("rescan-certain" if n == 0 else "rescan-probable" if n < 4 else
            "separate-print-weak" if n < 10 else "separate-print-strong")


def pass_of(name: str) -> str:
    """Which digitisation pass a file came from, from the album's naming convention."""
    m = re.match(r"^age\d+_([a-z]\d{0,2})_", name)
    return m.group(1) if m else "?"


def main() -> None:
    rd = csv.DictReader(SRC.open(encoding="utf-8-sig"))
    if "verdict" not in (rd.fieldnames or []):
        raise SystemExit("album_same_negative.csv has no verdicts yet - run "
                         "review_album_same_negative.py --apply first")
    rows = [r for r in rd if r["verdict"] == "same-negative"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    groups = {g: m for g, m in groups.items() if len(m) > 1}
    pairs = [(g, m[0], other) for g, m in groups.items() for other in m[1:]]
    print(f"{len(groups)} groups, {len(pairs)} pairs to judge")

    need = sorted({r["path"] for m in groups.values() for r in m})
    print(f"decoding {len(need)} files at {W}px")
    img = {}
    for n, p in enumerate(need, 1):
        try:
            img[p] = load(p)
        except Exception as e:
            print(f"  unreadable: {p}: {e}")
            img[p] = None
        if n % 100 == 0:
            print(f"  {n}/{len(need)}", flush=True)

    out = []
    for n, (g, ref, other) in enumerate(pairs, 1):
        a, b = img.get(ref["path"]), img.get(other["path"])
        res = damage(a, b) if a is not None and b is not None else None
        if res is None:
            kind, blobs, area = "undecided", "", ""
        else:
            blobs, area = res
            kind = ("separate-print" if blobs >= BLOB_MIN or area >= BLOB_AREA_MIN
                    else "rescan")
        pa, pb = pass_of(Pathname(ref)), pass_of(Pathname(other))
        out.append({"group": g, "kind": kind, "confidence": confidence(blobs), "blobs": blobs, "blob_area_per_1e4": (
            f"{area:.1f}" if area != "" else ""), "folder": other["folder"],
            "reference": Pathname(ref), "other": Pathname(other),
            "pass_reference": pa, "pass_other": pb,
            "cross_pass": "yes" if pa != pb else "no",
            "residual": other["residual_to_first"], "inliers": other["inliers_to_first"]})
        if n % 40 == 0:
            print(f"  judged {n}/{len(pairs)}", flush=True)

    k = defaultdict(int)
    for r in out:
        k[r["kind"]] += 1
    print("\npairs:")
    for key in sorted(k):
        print(f"  {k[key]:4}  {key}")

    gk: dict[str, set[str]] = defaultdict(set)
    for r in out:
        gk[r["group"]].add(r["kind"])
    gsum = defaultdict(int)
    for g, kinds in gk.items():
        gsum["mixed" if len(kinds - {"undecided"}) > 1
             else next(iter(kinds - {"undecided"}), "undecided")] += 1
    print("\ngroups:")
    for key in sorted(gsum):
        print(f"  {gsum[key]:4}  {key}")

    xt = defaultdict(int)
    for r in out:
        xt[(r["kind"], r["cross_pass"])] += 1
    print("\nkind x cross-pass (independent check -- provenance was never an input):")
    for key in sorted(xt):
        print(f"  {xt[key]:4}  {key[0]:16} cross_pass={key[1]}")

    nz = sorted(r["blobs"] for r in out if r["blobs"] != "")
    if nz:
        print(f"\nblob counts: min {nz[0]}, median {nz[len(nz)//2]}, max {nz[-1]}, "
              f"{sum(1 for x in nz if x == 0)} pairs at exactly 0")

    if not APPLY:
        print("\ndry run -- nothing written. Re-run with --apply.")
        return

    # ---------- merge, never overwrite ----------
    # This file holds decisions a rerun CANNOT reproduce. Three rows are `family-verdict`:
    # entered by hand, from people, about prints only they can tell apart. It also holds
    # verdicts for pairs a given run does not judge at all -- a scan it cannot open is
    # skipped, and a group whose membership changed produces a different pair set.
    #
    # Overwriting cost exactly that: a rerun replaced 313 judged rows with 239, turned 102
    # decided pairs back into `undecided`, and deleted all three family verdicts. It was
    # recoverable only because the file was committed. So the write merges instead:
    #
    #   * a `family-verdict` row is never replaced by a machine verdict
    #   * a decided row is never replaced by `undecided` -- a run that cannot judge a pair
    #     has learned nothing about it, and silence must not overwrite knowledge
    #   * a row this run did not produce is carried through untouched
    #
    # Everything else takes the fresh measurement, which is the point of re-running.
    fields = list(out[0].keys())
    old: dict[tuple, dict] = {}
    if OUT.exists():
        prev_rows = list(csv.DictReader(OUT.open(encoding="utf-8-sig")))
        if prev_rows and list(prev_rows[0]) != fields:
            raise SystemExit(f"REFUSED: {OUT.name} has a different header; merge would "
                             f"misalign columns.\n  on disk: {list(prev_rows[0])}\n  fresh:   {fields}")
        old = {pair_key(r): r for r in prev_rows}

    order = {k: i for i, k in enumerate(old)}
    merged, seen = [], set()
    kept_human = kept_decided = 0
    for r in out:
        k = pair_key(r)
        seen.add(k)
        was = old.get(k)
        if was is not None and was.get("confidence") == "family-verdict":
            merged.append(was)
            kept_human += 1
        elif was is not None and r["kind"] == "undecided" and was["kind"] != "undecided":
            merged.append(was)
            kept_decided += 1
        else:
            merged.append(r)
    carried = [r for k, r in old.items() if k not in seen]
    merged.extend(carried)
    merged.sort(key=lambda r: order.get(pair_key(r), len(order)))

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)
    print(f"\nmerged into {OUT.relative_to(A.REPO)}: {len(merged)} rows "
          f"({len(old)} before, {len(out)} judged this run)")
    print(f"  {kept_human:4} family verdicts left untouched")
    print(f"  {kept_decided:4} decided rows kept where this run could not judge the pair")
    print(f"  {len(carried):4} rows carried through -- this run did not produce them")


def Pathname(r: dict) -> str:
    from pathlib import Path
    return Path(r["path"]).name


if __name__ == "__main__":
    main()
