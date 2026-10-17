r"""Apply the per-group verdicts, and close the negatives they were asked about.

A pairwise page can produce a cycle inside one negative, and then nothing can be applied.
Asked once per group instead -- keep or drop, each scan -- every group must keep at least one
scan and every scan must get a verdict. A group that keeps several is the discrimination
that says these were judged rather than clicked through.

Two things this does beyond deleting.

**It closes the negatives.** A group is only settled if the scans that REMAIN are on record
as separate prints; otherwise the same negative comes back as `plates-undecided` in the next
build and the review was for nothing. For every group that keeps more than one scan, the
surviving pairs are written to `album_print_vs_rescan.csv` as a family verdict.

**It tells the truth about what a deletion costs.** A dropped scan that is byte-identical
to an origin that survives is recoverable by copying the origin back. A crop out of an
uncut sheet is not: the sheet is what survives. Restoring it means cropping again, not copying, and its row
says so rather than implying a copy would do.

    .venv\Scripts\python.exe scripts\apply_group_review.py [--apply]
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import os
import re
import sys
from collections import defaultdict
from datetime import date

import _archive as A

sys.stdout.reconfigure(encoding="utf-8")

APPLY = "--apply" in sys.argv
REPO = A.REPO
BP = A.BP
RAW = REPO / "data" / "group_review_answers.raw.txt"
LOG = REPO / "data" / "group_review_log.csv"
THINNED = REPO / "data" / "album_thinned.csv"
PVR = REPO / "data" / "album_print_vs_rescan.csv"
TODAY = str(date.today())

STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")


def scan_key(name: str) -> str:
    return STRIP.match(os.path.basename(name)).group(1)


def rows(name: str) -> list[dict]:
    p = REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))) if p.exists() else []


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    P = {r["scan"]: r for r in rows("photographs.csv")}
    M = {tuple(sorted((r["scan_a"], r["scan_b"]))): r
         for r in rows("duplicate_review_machine.csv")}
    album: dict[str, list[str]] = defaultdict(list)
    for r in rows("MANIFEST.csv"):
        album[scan_key(r["album_file"])].append(r["album_file"])

    ans: dict[str, dict[str, str]] = defaultdict(dict)
    for line in RAW.open(encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] != "negative":
            ans[p[0]][p[1]] = p[2]

    # ---- refuse to act on a group that does not hold together ----
    for neg, d in sorted(ans.items()):
        keeps = [s for s, v in d.items() if v == "keep"]
        held = {s for s in P if P[s]["negative"] == neg}
        if not keeps:
            sys.exit(f"{neg}: every scan dropped -- refusing, that deletes the photograph")
        if held - set(d):
            sys.exit(f"{neg}: no verdict for {sorted(held - set(d))} -- refusing")

    log, plan, pairs = [], [], []
    for neg, d in sorted(ans.items()):
        keeps = sorted(s for s, v in d.items() if v == "keep")
        for s in sorted(x for x, v in d.items() if v == "drop"):
            origin = P[s]["origin_path"]
            identical = os.path.exists(origin) and any(
                sha(BP / af.replace("/", os.sep)) == sha(origin)
                for af in album.get(s, [])
                if (BP / af.replace("/", os.sep)).exists())
            for af in sorted(album.get(s, [])):
                p = BP / af.replace("/", os.sep)
                if p.exists():
                    plan.append((p, s, neg, keeps, origin, identical))
            log.append({"date": TODAY, "negative": neg, "scan": s, "verdict": "drop",
                        "kept_instead": "; ".join(keeps),
                        "origin": origin,
                        "recoverable": "copy the origin back" if identical
                                       else "origin is an uncut sheet: re-crop, do not copy"})
        for s in keeps:
            log.append({"date": TODAY, "negative": neg, "scan": s, "verdict": "keep",
                        "kept_instead": "", "origin": P[s]["origin_path"], "recoverable": ""})
        # what remains in the negative is, by this answer, separate physical prints
        for a, b in itertools.combinations(keeps, 2):
            pairs.append((neg, a, b))

    print(f"{sum(1 for r in log if r['verdict']=='drop')} scans dropped, "
          f"{sum(1 for r in log if r['verdict']=='keep')} kept, across {len(ans)} negatives")
    total = 0
    for p, s, neg, keeps, origin, identical in plan:
        total += p.stat().st_size
        print(f"  {p.relative_to(BP)}")
        print(f"      {neg}  keeping {', '.join(k[:34] for k in keeps)}"
              f"{'' if identical else '   <-- CROP of a sheet, not a copy'}")
    print(f"\n{len(plan)} album files, {total/1e6:.0f} MB")
    print(f"{len(pairs)} surviving pairs to record as separate prints "
          f"({', '.join(sorted({n for n, _, _ in pairs})) or 'none'})")

    if not APPLY:
        print("\n(dry run -- pass --apply to write)")
        return

    with LOG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]) if log else [])
        w.writeheader()
        w.writerows(log)
    print(f"wrote {LOG}")

    pvr, fields = A.read_ledger("album_print_vs_rescan.csv")
    seen = {tuple(sorted((scan_key(r["reference"]), scan_key(r["other"])))) for r in pvr}
    added = 0
    for neg, a, b in pairs:
        if tuple(sorted((a, b))) in seen:
            continue
        m = M.get(tuple(sorted((a, b))), {})
        row = {k: "" for k in fields}
        row.update({"group": neg, "kind": "separate-print", "confidence": "family-verdict",
                    "blobs": m.get("blobs", ""), "blob_area_per_1e4": m.get("blob_area", ""),
                    "reference": a, "other": b,
                    "residual": m.get("residual", ""), "inliers": m.get("inliers", "")})
        pvr.append(row)
        seen.add(tuple(sorted((a, b))))
        added += 1
    with PVR.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pvr)
    print(f"wrote {PVR}  (+{added} rows)")

    th, fields = A.read_ledger("album_thinned.csv")
    for p, s, neg, keeps, origin, identical in plan:
        row = {k: "" for k in fields}
        row.update({
            "date": TODAY, "deleted_path": str(p), "bytes": p.stat().st_size,
            "sha256": sha(p), "origin": origin, "group": neg,
            "reason": (f"per-group review of {neg}: the family kept "
                       f"{', '.join(keeps)} and dropped this one. "
                       + ("Byte-identical to its origin, which survives."
                          if identical else
                          "NOT a copy -- this is a crop out of the uncut sheet at the origin, "
                          "which survives; restoring it means cropping again, not copying.")),
        })
        th.append(row)
        p.unlink()
        print(f"  deleted {p.relative_to(BP)}")
    with THINNED.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(th)
    print(f"wrote {THINNED}")
    print("\nNow rerun:  rebuild_manifest.py && patch_derived.py && build_photographs.py")


if __name__ == "__main__":
    main()
