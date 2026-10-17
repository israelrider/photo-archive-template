r"""Judge the cross-pairs that a negative MERGE creates, which nothing else was watching.

Joining two negatives is not free. When both already hold scans, merging them on the
strength of one dHash pair creates cross-pairs no verdict file has ever seen. The build
catches it -- `plates-undecided` goes up -- but nothing generates the missing verdicts, and
the negative sits open until someone notices why. This measures those pairs; the ones that
read as separate prints are recorded, because every one of those physical prints stays.

A pair reading as a RESCAN is deliberately not recorded here. That verdict decides which
file gets deleted, and deletions in this archive are the family's to make, not a script's.
Those are printed and left open instead.

    .venv\Scripts\python.exe scripts\close_merged_negative.py [N0123] [--apply]
"""
from __future__ import annotations

import csv
import itertools
import os
import re
import sys

import _archive as A
import split_rescan_vs_print as S
from find_same_negative_album import registers_arrays

sys.stdout.reconfigure(encoding="utf-8")

APPLY = "--apply" in sys.argv
NEG = next((a for a in sys.argv[1:] if a.startswith("N") and a[1:].isdigit()), None)
PVR = A.REPO / "data" / "album_print_vs_rescan.csv"

STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")


def key(name: str) -> str:
    return STRIP.match(os.path.basename(name)).group(1)


def rows(name: str) -> list[dict]:
    p = A.REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline="")))


def main() -> None:
    P = {r["scan"]: r for r in rows("photographs.csv")}
    pvr = rows("album_print_vs_rescan.csv")
    judged = {tuple(sorted((key(r["reference"]), key(r["other"])))) for r in pvr}

    # every negative that still holds an unjudged pair, not just the one named
    targets = [NEG] if NEG else sorted({r["negative"] for r in P.values()
                                        if r["status"] == "plates-undecided"})
    if not targets:
        print("no negative holds an unjudged pair")
        return

    cache: dict[str, object] = {}

    def grey(p: str):
        if p not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[p] = S.load(p)
        return cache[p]

    add, open_pairs = [], []
    for neg in targets:
        members = sorted(s for s in P if P[s]["negative"] == neg)
        missing = [(a, b) for a, b in itertools.combinations(members, 2)
                   if tuple(sorted((a, b))) not in judged]
        print(f"{neg}: {len(members)} scans, {len(missing)} of "
              f"{len(members)*(len(members)-1)//2} pairs never judged")
        for a, b in missing:
            d = S.damage(grey(P[a]["origin_path"]), grey(P[b]["origin_path"]))
            res, inl = registers_arrays(grey(P[a]["origin_path"]), grey(P[b]["origin_path"]))
            blobs, area = d if d else ("", "")
            kind = ("separate-print" if d and (blobs >= S.BLOB_MIN or area >= S.BLOB_AREA_MIN)
                    else "rescan" if d else "undecided")
            conf = S.confidence(blobs) if d else "no-alignment"
            row = {"group": neg, "kind": kind, "confidence": conf, "blobs": blobs,
                   "blob_area_per_1e4": "" if not d else round(area, 1),
                   "reference": a, "other": b,
                   "residual": "" if res is None else round(res, 3), "inliers": inl}
            print(f"  {kind:15s} {conf:22s} blobs={blobs:<4} res={row['residual']:<6} "
                  f"{a[:30]} | {b[:30]}")
            (add if kind == "separate-print" else open_pairs).append(row)

    print(f"\n{len(add)} separate prints to record, {len(open_pairs)} needing a person")
    for r in open_pairs:
        print(f"  STILL OPEN: {r['reference']} | {r['other']}  ({r['confidence']}) "
              f"-- reads as a rescan, so which file goes is the family's call")
    if not add:
        return
    if not APPLY:
        print("(pass --apply to record the separate prints)")
        return

    fields = list(pvr[0]) if pvr else A.LEDGER_FIELDS["album_print_vs_rescan.csv"]
    for r in add:
        row = {k: "" for k in fields}
        row.update(r)
        pvr.append(row)
    with PVR.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pvr)
    print(f"wrote {PVR}  (+{len(add)} rows)")
    print("Now rerun:  build_photographs.py")


if __name__ == "__main__":
    main()
