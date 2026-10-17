r"""Put the unreviewed dHash proposals through the test that actually decides them.

The hash is a screen, not a verdict, and the d<=5 batch already showed how weak a screen:
of 22 pairs that would have joined negatives, 18 did not register at all -- residuals up to
1.45 against a 0.45 cut. The hash had matched them on the gross layout of light and dark.
Only 4 survived alignment.

The d=6 tail is 59 more of the same, plus one d=4 pair never reached, and reviewing 60
pairs by eye to find the handful that are real is the kind of task that gets abandoned
half-done. So registration is run first: it is one measurement per pair and it is the thing
that separates a real match from a coincidence of tone. Only what survives it is worth a
person's time, and `damage()` is only run on those -- its blob count means nothing on two
pictures that were never aligned.

Nothing here decides anything on its own. It writes `data/dhash_tail_machine.csv` and says
which pairs, if any, deserve a look.

    .venv\Scripts\python.exe scripts\check_dhash_tail.py [--remeasure] [--apply]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import date

import numpy as np

import _archive as A
import split_rescan_vs_print as S
from find_same_negative_album import registers_arrays

sys.stdout.reconfigure(encoding="utf-8")

REMEASURE = "--remeasure" in sys.argv
APPLY = "--apply" in sys.argv
REPO = A.REPO
CACHE = REPO / "data" / "dhash_tail_machine.csv"
RESIDUAL_MAX = 0.45          # find_same_negative_album's own cut
TODAY = str(date.today())


def rows(name: str) -> list[dict]:
    p = REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))) if p.exists() else []


def main() -> None:
    P = {r["scan"]: r for r in rows("photographs.csv")}
    todo = [r for r in rows("photograph_proposals.csv")
            if r["already_one_negative"] == "no" and not r["reviewed"]
            and r["scan_a"] in P and r["scan_b"] in P]
    todo.sort(key=lambda r: (int(r["hamming"]), r["scan_a"]))
    print(f"{len(todo)} unreviewed proposals that would join negatives now held apart")

    have = {}
    if CACHE.exists() and not REMEASURE:
        have = {(r["scan_a"], r["scan_b"]): r for r in rows(CACHE.name)}
    pending = [r for r in todo if (r["scan_a"], r["scan_b"]) not in have]
    print(f"{len(pending)} to measure ({len(todo) - len(pending)} cached)")

    cache: dict[str, np.ndarray] = {}

    def grey(path: str) -> np.ndarray:
        if path not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[path] = S.load(path)
        return cache[path]

    for i, r in enumerate(pending, 1):
        a, b = P[r["scan_a"]]["origin_path"], P[r["scan_b"]]["origin_path"]
        Ai, Bi = grey(a), grey(b)
        res, inl = registers_arrays(Ai, Bi)
        registers = res is not None and res <= RESIDUAL_MAX
        blobs = area = ""
        if registers:
            d = S.damage(Ai, Bi)
            if d:
                blobs, area = d[0], round(d[1], 1)
        out = {"hamming": r["hamming"], "scan_a": r["scan_a"], "scan_b": r["scan_b"],
               "negative_a": r["negative_a"], "negative_b": r["negative_b"],
               "residual": "" if res is None else round(res, 3), "inliers": inl,
               "registers": "yes" if registers else "no",
               "blobs": blobs, "blob_area": area}
        have[(r["scan_a"], r["scan_b"])] = out
        print(f"  {i:3d}/{len(pending)}  d={r['hamming']}  res="
              f"{out['residual']:<6} inl={inl:<4} {'REGISTERS' if registers else '-':10s} "
              f"{r['scan_a'][:30]} | {r['scan_b'][:30]}")

    result = [have[(r["scan_a"], r["scan_b"])] for r in todo]
    if pending:
        with CACHE.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result[0]) if result else [])
            w.writeheader()
            w.writerows(result)
        print(f"wrote {CACHE}")

    real = [r for r in result if r["registers"] == "yes"]
    print("\n" + "=" * 74)
    print(f"{len(result)} pairs tested, {len(real)} register, {len(result)-len(real)} do not")
    by_d = defaultdict(lambda: [0, 0])
    for r in result:
        by_d[int(r["hamming"])][r["registers"] == "yes"] += 1
    for d in sorted(by_d):
        no, yes = by_d[d]
        print(f"  d={d}  {yes+no:3d} pairs   {yes:2d} register   {no:2d} do not")

    if real:
        print("\nWORTH A LOOK -- these align, so they may be one photograph filed as two:")
        for r in sorted(real, key=lambda x: float(x["residual"])):
            print(f"  res={r['residual']:<6} inliers={r['inliers']:<4} blobs={r['blobs']:<4} "
                  f"{r['negative_a']}+{r['negative_b']}  "
                  f"{r['scan_a']}  |  {r['scan_b']}")

    # a scan that the hash pairs with everything is telling you about the hash, not the archive
    hub = defaultdict(lambda: [0, 0])
    for r in result:
        for s in (r["scan_a"], r["scan_b"]):
            hub[s][r["registers"] == "yes"] += 1
    worst = sorted(hub.items(), key=lambda kv: -sum(kv[1]))[:6]
    print("\nScans the hash proposed most often (misses / real):")
    for s, (no, yes) in worst:
        print(f"  {no+yes:3d} proposals  {yes:2d} real   {s}")

    if not APPLY:
        print("\n(pass --apply to record these verdicts)")
        return

    # Both survivors were looked at full size and each is one negative printed twice.
    # The second scored only 66 inliers because the two prints are toned so differently --
    # one faded almost to yellow, one printed very dark -- which this archive has already
    # recorded once as a reason for a low inlier count, and is not a weak match.
    LOG = REPO / "data" / "duplicate_review_log.csv"
    existing, fields = A.read_ledger(LOG.name)
    seen = {tuple(sorted((r["scan_a"], r["scan_b"]))) for r in existing}
    added = 0
    for r in result:
        pair = tuple(sorted((r["scan_a"], r["scan_b"])))
        if pair in seen:
            continue
        real = r["registers"] == "yes"
        row = {k: "" for k in fields}
        row.update({
            "date": TODAY,
            "negative": r["negative_a"] + " + " + r["negative_b"],
            "section": "dhash-tail",
            "scan_a": r["scan_a"], "scan_b": r["scan_b"],
            "answer": "both" if real else "not-same",
            "machine": ("separate-print, " + str(r["blobs"]) + " blobs" if real
                        else "does not register"),
            "residual": r["residual"],
            "resolution": ("same-negative, separate prints" if real else
                           "different-photograph (does not register; the hash matched on tone)"),
        })
        existing.append(row)
        seen.add(pair)
        added += 1
    with LOG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing)
    print("\nwrote " + str(LOG) + "  (+" + str(added) + " rows)")
    print("Now rerun:  build_photographs.py")


if __name__ == "__main__":
    main()
