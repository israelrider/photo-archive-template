r"""Find photographs that exist in `by-person\` as more than one print.

A sha256 finds copies of one scan, but two prints of one negative are different objects with
different bytes, different darkroom density and different damage, and no hash will ever pair
them. This asks the question of the whole album.

Pairwise registration does not scale: 3000 files is 4.5M pairs and about eleven days. Three
stages instead.

  A. decode once, cache. PIL's `draft` mode scales during JPEG decode (about twice as fast),
     so a 60 GB album becomes a 256x256 grayscale array per file in ~10 minutes, cached to
     an .npz so later stages are free to re-run.

  B. collapse exact copies. The album deliberately holds one photograph in 2-9 person folders
     -- that duplication is how people get identified and is not what we are looking for.
     Identical thumbnails mean identical content; one representative each goes forward.

  C. prefilter, then register. **Not by intensity correlation** -- that misses two prints of
     one negative that differ so much in density they fall below any usable cutoff. Gradient magnitude survives what density does to a
     print, so the descriptor is a masked, contrast-normalised gradient thumbnail, compared
     by one matmul. Each image keeps its top-K neighbours (a rank, not a threshold, so an
     unusually dense print cannot fall off the list), and only those pairs are registered.

Registration (`registers_arrays`): ORB inside the middle 62% of the frame --
**the mask is not optional**, a studio mount card registers two different children at 784
inliers -- RANSAC homography, then mean absolute difference over the aligned overlap.

Recall is checked, not assumed: pairs already proved to be one negative can be listed in
`data\same_negative_controls.csv` (columns a,b: origin paths, absolute or relative to the
archive). They are carried in as positive controls, and the run reports whether the
prefilter found each one and with how much margin -- and refuses if it missed one. Add a
few as soon as the family has confirmed some; until then recall is unchecked.

    .venv\Scripts\python.exe scripts\find_same_negative_album.py [--stage a|b|c] [--topk N]
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import _archive as A

sys.stdout.reconfigure(encoding="utf-8")

SCRATCH = Path(os.environ.get("CLAUDE_SCRATCH", Path(__file__).parent.parent / ".cache"))
SCRATCH.mkdir(parents=True, exist_ok=True)
CACHE = SCRATCH / "album_thumbs.npz"
OUT = A.REPO / "data" / "album_same_negative.csv"

THUMB = 256
TOPK = int(sys.argv[sys.argv.index("--topk") + 1]) if "--topk" in sys.argv else 24
STAGE = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"

# negatives already proved, as origin files -- positive controls for recall
def _controls() -> list[tuple[Path, Path]]:
    f = A.LOGS / "same_negative_controls.csv"
    if not f.exists():
        return []
    return [(A.ARCHIVE / r["a"], A.ARCHIVE / r["b"])
            for r in csv.DictReader(f.open(encoding="utf-8-sig"))]


CONTROL_PATHS = _controls()
CONTROLS = [(a.name, b.name) for a, b in CONTROL_PATHS]
CONTROL_FILES = sorted({p for pair in CONTROL_PATHS for p in pair})


def album_files() -> list[Path]:
    fs = [p for p in sorted(A.BP.rglob("*")) if p.is_file()
          and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
    return fs


# EXIF orientation -> quarter turns for np.rot90. Scanned albums often have half their JPEGs
# carrying a rotating tag, so ignoring it compares half the album against the other half at
# 90 degrees: two prints of one negative score about zero similarity, and the positive
# controls catch it. Verified against PIL's own exif_transpose.
EXIF_ROT = {3: 2, 6: 3, 8: 1}


def thumb(p: Path) -> np.ndarray | None:
    try:
        im = Image.open(p)
        k = EXIF_ROT.get(im.getexif().get(274), 0)
        im.draft("L", (THUMB * 2, THUMB * 2))       # decode straight to a small size
        a = np.asarray(im.convert("L").resize((THUMB, THUMB), Image.LANCZOS), dtype=np.uint8)
        # a square resize scales each axis independently, so rotating after it is exact
        return np.ascontiguousarray(np.rot90(a, k)) if k else a
    except Exception as e:
        print(f"  unreadable, skipped: {p.name}: {e}")
        return None


def stage_fix() -> None:
    """Rotate an existing cache into display orientation, without decoding 66 GB again."""
    paths, T = load()
    n = 0
    for i, p in enumerate(paths):
        try:
            k = EXIF_ROT.get(Image.open(p).getexif().get(274), 0)
        except Exception:
            k = 0
        if k:
            T[i] = np.rot90(T[i], k)
            n += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    np.savez_compressed(CACHE, names=np.array([str(p) for p in paths]), thumbs=T)
    print(f"rotated {n} of {len(paths)} thumbnails into display orientation")


def stage_a() -> None:
    fs = album_files() + CONTROL_FILES
    print(f"stage A: {len(fs)} files (album + {len(fs) - len(album_files())} controls)")
    names, arrs = [], []
    t0 = time.time()
    for i, f in enumerate(fs, 1):
        a = thumb(f)
        if a is None:
            continue
        names.append(str(f))
        arrs.append(a)
        if i % 250 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(fs)}  {el/60:.1f} min elapsed, "
                  f"{(len(fs)-i)*el/i/60:.1f} min left", flush=True)
    np.savez_compressed(CACHE, names=np.array(names), thumbs=np.stack(arrs))
    print(f"cached {len(names)} thumbnails -> {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)")


def load() -> tuple[list[Path], np.ndarray]:
    z = np.load(CACHE, allow_pickle=False)
    return [Path(s) for s in z["names"]], z["thumbs"]


def descriptors(T: np.ndarray) -> np.ndarray:
    """Masked, contrast-normalised gradient thumbnails, L2-normalised for a matmul.

    Gradient magnitude rather than intensity: two prints of one negative can differ hugely in
    density -- that is what a darkroom does -- while their edges stay put. The centre mask
    keeps a studio mount card's identical border out of the comparison."""
    n = T.shape[0]
    m = THUMB // 2
    lo, hi = (THUMB - m) // 2, (THUMB + m) // 2
    D = np.empty((n, m * m // 16), dtype=np.float32)
    for i in range(n):
        g = T[i].astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.resize(np.hypot(gx, gy)[lo:hi, lo:hi], (m // 4, m // 4),
                         interpolation=cv2.INTER_AREA).ravel()
        mag -= mag.mean()
        D[i] = mag / (np.linalg.norm(mag) + 1e-6)
    return D


def picture_area(img: np.ndarray, frac: float = 0.62) -> np.ndarray:
    h, w = img.shape
    dy, dx = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    mk = np.zeros_like(img, dtype=np.uint8)
    mk[dy:h - dy, dx:w - dx] = 255
    return mk


def gray(p: Path, w: int = 900) -> np.ndarray:
    im = Image.open(p)
    im.draft("L", (w * 2, w * 2))
    im = im.convert("L")
    return np.asarray(im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS))


_ORB = cv2.ORB_create(6000)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def registers(pa: Path, pb: Path) -> tuple[float | None, int]:
    try:
        return registers_arrays(gray(pa), gray(pb))
    except Exception:
        return None, 0


def registers_arrays(A_: np.ndarray | None, B_: np.ndarray | None) -> tuple[float | None, int]:
    if A_ is None or B_ is None:
        return None, 0
    ka, da = _ORB.detectAndCompute(A_, picture_area(A_))
    kb, db = _ORB.detectAndCompute(B_, picture_area(B_))
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, 0
    m = sorted(_BF.match(da, db), key=lambda x: x.distance)[:800]
    if len(m) < 15:
        return None, len(m)
    src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
    dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None, len(m)
    inl = int(mask.sum())
    warp = cv2.warpPerspective(A_, H, (B_.shape[1], B_.shape[0]), borderValue=0)
    ok = (warp > 0) & (picture_area(B_) > 0)
    if ok.sum() < 5000:
        return None, inl
    x = warp[ok].astype(np.float32)
    y = B_[ok].astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return float(np.abs(x - y).mean()), inl


RESIDUAL_MAX = 0.45
INLIERS_MIN = 40
REG_W = 900            # working width for registration; every representative is decoded once
SCREEN_FEATURES = 6000 # NOT a smaller number: ORB builds the same pyramid either way and
                       # then pays to rank and retain the best N, so 900 features measured
                       # 172 ms/image against 29 ms for 6000. Asking for fewer is 6x slower.
SCREEN_INLIERS = 25    # deliberately below INLIERS_MIN so the screen never decides a pair


def main() -> None:
    paths, T = load()
    ctrl = set(CONTROL_FILES)
    is_ctrl = np.array([p in ctrl for p in paths])
    print(f"stage B: {len(paths)} thumbnails ({int(is_ctrl.sum())} controls)")

    # identical content -> one representative. This is the album's intentional duplication:
    # one photograph filed in 2-9 person folders, which is not what we are looking for.
    key = {}
    content = np.empty(len(paths), dtype=np.int64)
    members: list[list[int]] = []
    for i in range(len(paths)):
        k = hashlib.sha256(T[i].tobytes()).digest()
        if k not in key:
            key[k] = len(members)
            members.append([])
        content[i] = key[k]
        members[content[i]].append(i)
    print(f"  {len(members)} distinct contents "
          f"({len(paths) - len(members)} exact copies collapsed)")

    # a representative per content, preferring an album path over a control origin
    rep = []
    for ms in members:
        alb = [i for i in ms if not is_ctrl[i]]
        rep.append(alb[0] if alb else ms[0])
    rep = np.array(rep)

    print(f"stage C: descriptors for {len(rep)} contents")
    # An EXIF tag fixes the tagged rotations; this archive also holds at least one scan that
    # is sideways with NO tag ((4) 0028). Score every pair at all four quarter turns and keep
    # the best, so an untagged rotation cannot hide a pair from the prefilter either.
    base = T[rep]
    D0 = descriptors(base)
    S = D0 @ D0.T
    for k in (1, 2, 3):
        Dk = descriptors(np.ascontiguousarray(np.rot90(base, k, axes=(1, 2))))
        np.maximum(S, D0 @ Dk.T, out=S)
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, -2.0)

    k = min(TOPK, len(rep) - 1)
    nbr = np.argpartition(-S, k - 1, axis=1)[:, :k]
    cand = set()
    for i in range(len(rep)):
        for j in nbr[i]:
            cand.add((min(i, int(j)), max(i, int(j))))
    print(f"  {len(cand)} candidate pairs from top-{k} neighbours "
          f"({len(rep)*(len(rep)-1)//2} pairs avoided)")

    # recall on the negatives already proved (data\same_negative_controls.csv)
    print(f"\n  positive controls ({len(CONTROLS)} proved negatives):")
    idx_of = {}
    for i, p in enumerate(paths):
        if is_ctrl[i]:
            idx_of[p.name] = content[i]
    rank_of = {}
    for i in range(len(rep)):
        order = np.argsort(-S[i])
        for r, j in enumerate(order[:200]):
            rank_of[(i, int(j))] = r
    missed = 0
    for a, b in CONTROLS:
        ca, cb = idx_of.get(a), idx_of.get(b)
        if ca is None or cb is None:
            print(f"    {a[-13:]} ~ {b[-13:]}   control file not cached")
            continue
        pair = (min(ca, cb), max(ca, cb))
        r = rank_of.get((ca, cb), rank_of.get((cb, ca)))
        hit = pair in cand
        if not hit:
            missed += 1
        print(f"    {a[-13:]} ~ {b[-13:]}   similarity {S[ca, cb]:+.3f}  "
              f"rank {r if r is not None else '>200'}  "
              f"{'FOUND by prefilter' if hit else 'MISSED -- raise --topk'}")
    if missed:
        raise SystemExit(f"\nREFUSING: prefilter missed {missed} known positives; "
                         f"re-run with a larger --topk")

    # Decode each representative ONCE. Registering 40k pairs by re-opening both files would
    # decode 80k times and take four hours; the same work off a decoded cache takes minutes.
    print(f"\n  decoding {len(rep)} representatives once")
    imgs: list[np.ndarray | None] = []
    t0 = time.time()
    for n, i in enumerate(rep, 1):
        try:
            imgs.append(gray(paths[i], REG_W))
        except Exception:
            imgs.append(None)
        if n % 400 == 0:
            el = time.time() - t0
            print(f"    {n}/{len(rep)}  {el/60:.1f} min, "
                  f"{(len(rep)-n)*el/n/60:.1f} min left", flush=True)

    # Cheap screen: few features, match + homography only, no warp and no residual.
    screen = cv2.ORB_create(SCREEN_FEATURES)
    KD = []
    for a in imgs:
        KD.append((None, None) if a is None else screen.detectAndCompute(a, picture_area(a)))

    print(f"  screening {len(cand)} pairs on {SCREEN_FEATURES} features")
    survivors, t0 = [], time.time()
    for n, (i, j) in enumerate(sorted(cand), 1):
        (ka, da), (kb, db) = KD[i], KD[j]
        if da is None or db is None or len(ka) < 20 or len(kb) < 20:
            continue
        m = _BF.match(da, db)
        if len(m) < 15:
            continue
        m = sorted(m, key=lambda x: x.distance)[:400]
        src = np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        dst = np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or int(mask.sum()) < SCREEN_INLIERS:
            continue
        survivors.append((i, j))
        if n % 5000 == 0:
            el = time.time() - t0
            print(f"    {n}/{len(cand)}  {el/60:.1f} min, "
                  f"{(len(cand)-n)*el/n/60:.1f} min left, {len(survivors)} through",
                  flush=True)
    print(f"  {len(survivors)} pairs survive the screen -> full registration")

    hits = []
    for i, j in survivors:
        res, inl = registers_arrays(imgs[i], imgs[j])
        if res is not None and res < RESIDUAL_MAX and inl >= INLIERS_MIN:
            hits.append((res, inl, i, j))

    parent = list(range(len(rep)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, _, i, j in hits:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(len(rep)):
        groups.setdefault(find(i), []).append(i)
    groups = {g: v for g, v in groups.items() if len(v) > 1}

    # the control origins were injected on purpose; they are not an album finding
    is_ctrl_rep = {i for i in range(len(rep)) if is_ctrl[rep[i]]}
    ctrl_groups = {g: v for g, v in groups.items() if any(i in is_ctrl_rep for i in v)}
    groups = {g: v for g, v in groups.items() if g not in ctrl_groups}
    print(f"\n  {len(ctrl_groups)} groups are the injected controls, excluded below")

    import csv
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "path", "folder", "residual_to_first", "inliers_to_first",
                    "other_folders_with_this_content"])
        for gi, (_, v) in enumerate(sorted(groups.items()), 1):
            v = sorted(v)
            for i in v:
                p = paths[rep[i]]
                pair = next(((r, n) for r, n, a, b in hits
                             if {a, b} == {v[0], i}), ("", ""))
                also = sorted({paths[m].parent.name for m in members[content[rep[i]]]}
                              - {p.parent.name})
                w.writerow([gi, str(p), p.parent.name,
                            f"{pair[0]:.3f}" if pair[0] != "" else "",
                            pair[1], "; ".join(also)])

    print(f"\n{len(hits)} same-negative pairs, {len(groups)} groups, "
          f"{sum(len(v) for v in groups.values())} distinct contents involved")
    for gi, (_, v) in enumerate(sorted(groups.items()), 1):
        v = sorted(v)
        print(f"\n  group {gi}: {len(v)} prints")
        for i in v:
            p = paths[rep[i]]
            also = sorted({paths[m].parent.name for m in members[content[rep[i]]]}
                          - {p.parent.name})
            print(f"      {p.parent.name}\\{p.name}"
                  + (f"   also in: {', '.join(also)}" if also else ""))
        for r, n, a, b in sorted(hits):
            if a in v and b in v:
                print(f"        {r:.3f} residual, {n} inliers  "
                      f"{paths[rep[a]].name[-13:]} ~ {paths[rep[b]].name[-13:]}")
    print(f"\nwrote {OUT.relative_to(A.REPO)}")


if __name__ == "__main__":
    if STAGE in ("a", "all") or not CACHE.exists():
        stage_a()
    if STAGE == "a":
        sys.exit(0)
    if STAGE == "fix":
        stage_fix()
        sys.exit(0)
    main()
