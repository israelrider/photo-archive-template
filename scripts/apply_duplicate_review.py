r"""Turn the answers from `duplicate-review.html` into records, and only then into deletions.

Three kinds of answer are NOT applied as given, each for a stated reason, because a review
page is evidence and not an instruction:

**A delete verdict in section 5 is discarded.** Section 5 holds pairs measured 10 or more
damage blobs apart -- the population `split_rescan_vs_print` exists to protect, where two
prints from one negative are two objects that both stay. A delete click there is far more
likely a slip than a judgement, so those pairs are written to `album_print_vs_rescan.csv` as
what they already were: machine verdicts. That closes them rather than leaving them to
surface in the next review.

**Section 6 becomes a rejection, not a merge.** A "both" answer there would assert one
negative with two prints for a pair that does not register at all (residual above the 0.45
cut). They are recorded as different photographs -- a proposal examined and turned down is
an answer and has to leave a trace, or the next dHash pass proposes it again.

**A negative whose answers form a cycle is left alone.** With five scans, A beats B, B beats
C and C beats A is easy to answer and leaves no survivor. That is a fault in the question,
not in the answers -- pairwise buttons cannot elect one winner out of five -- so such a
negative is re-asked per group (make_group_review.py) rather than resolved by guessing.

What IS applied: scans that a section 1 or 2 answer marks redundant inside a negative whose
answers do agree. Deleting the album copy never touches the origin.

    .venv\Scripts\python.exe scripts\apply_duplicate_review.py [--apply]
"""
from __future__ import annotations

import csv
import hashlib
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
RAW = REPO / "data" / "duplicate_review_answers.raw.txt"
LOG = REPO / "data" / "duplicate_review_log.csv"
THINNED = REPO / "data" / "album_thinned.csv"
PVR = REPO / "data" / "album_print_vs_rescan.csv"
TODAY = str(date.today())
RESIDUAL_MAX = 0.45

STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")

THINNED_FIELDS = A.LEDGER_FIELDS["album_thinned.csv"]
PVR_FIELDS = A.LEDGER_FIELDS["album_print_vs_rescan.csv"]


def cyclic_negatives(drops: dict) -> set[str]:
    """Negatives whose keep/drop answers form a cycle, so no scan survives them all."""
    out = set()
    for s, (keeper, _, neg) in drops.items():
        seen = {s}
        while keeper in drops:
            if keeper in seen:
                out.add(neg)
                break
            seen.add(keeper)
            keeper = drops[keeper][0]
    return out


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


def bucket(r: dict) -> str:
    registers = r["residual"] != "" and float(r["residual"]) <= RESIDUAL_MAX
    if not registers:
        return "3" if r["section"] == "B" else "6"
    if r["section"] == "A":
        return "4"
    if r["kind"] == "rescan":
        return "1"
    return "2" if r["confidence"] == "separate-print-weak" else "5"


def main() -> None:
    M = {(r["scan_a"], r["scan_b"]): r for r in rows("duplicate_review_machine.csv")}
    P = {r["scan"]: r for r in rows("photographs.csv")}

    album: dict[str, list[str]] = defaultdict(list)
    for r in rows("MANIFEST.csv"):
        album[scan_key(r["album_file"])].append(r["album_file"])

    ans: dict[tuple[str, str], str] = {}
    for line in RAW.open(encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] != "scan_a":
            ans[(p[0], p[1])] = p[2]

    # ---------------- resolve every answered pair ----------------
    log, drops = [], {}          # drops: scan key -> (keeper, confidence, negative)
    for k, v in sorted(ans.items()):
        m = M[k]
        b = bucket(m)
        neg = P.get(k[0], {}).get("negative", m["negative"])
        keeper = loser = ""
        if b in ("1", "2") and v in ("keep-a", "keep-b"):
            keeper, loser = (k[0], k[1]) if v == "keep-a" else (k[1], k[0])
            resolution = "rescan"
        elif b == "6":
            resolution = "different-photograph"
        elif b in ("4", "5") and v in ("keep-a", "keep-b"):
            # Same ruling as section 5: a delete verdict on a pair measured far apart was
            # not deliberate. Section 4's own purpose still stands -- the negatives merge.
            resolution = ("separate-print (machine); delete verdict not deliberate"
                          if b == "5" else "same-negative, separate prints")
        elif v == "both":
            resolution = "same-negative, separate prints"
        else:
            resolution = "recorded, not acted on"
        log.append({
            "date": TODAY, "negative": neg, "section": b,
            "scan_a": k[0], "scan_b": k[1], "answer": v,
            "machine": f"{m['kind']}/{m['confidence']}", "residual": m["residual"],
            "resolution": resolution, "keeper": keeper, "dropped": loser,
        })
        if loser:
            drops[loser] = (keeper, m["confidence"], neg)

    # A negative whose answers disagree elects no one; it is re-asked as a group.
    HELD_BACK = cyclic_negatives(drops)
    for s, (keeper, conf, neg) in list(drops.items()):
        if neg in HELD_BACK:
            del drops[s]

    # Answers can form a chain: A loses to B, which loses to C. Recording the immediate winner would point the record at a file about to go, so the
    # keeper is followed through to the scan that actually survives.
    for s in list(drops):
        keeper, conf, neg = drops[s]
        seen_chain = {s}
        while keeper in drops and keeper not in seen_chain:
            seen_chain.add(keeper)
            keeper = drops[keeper][0]
        drops[s] = (keeper, conf, neg)

    # ---------------- what the deletions touch ----------------
    plan = []
    for s, (keeper, conf, neg) in sorted(drops.items()):
        for af in sorted(album.get(s, [])):
            p = BP / af.replace("/", os.sep)
            plan.append((p, s, keeper, conf, neg))

    print(f"answered pairs logged: {len(log)}")
    by_res = defaultdict(int)
    for r in log:
        by_res[r["resolution"]] += 1
    for r, n in sorted(by_res.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {r}")
    print(f"\nscans to remove from by-person: {len(drops)}  "
          f"(held back in {sorted(HELD_BACK)}: "
          f"{sum(1 for k, v in ans.items() if bucket(M[k]) in '12' and v.startswith('keep') and P.get(k[0], {}).get('negative') in HELD_BACK)} answers)")
    total = 0
    for p, s, keeper, conf, neg in plan:
        ok = p.exists()
        total += p.stat().st_size if ok else 0
        print(f"  {'' if ok else 'MISSING '}{p.relative_to(BP)}")
        print(f"        {neg}  redundant with {keeper}  [{conf}]")
    print(f"\n{len(plan)} album files, {total/1e6:.0f} MB")

    # section 5 pairs, written where the existing machine verdicts live
    sec5 = [r for r in log if r["section"] == "5"]
    print(f"\n{len(sec5)} section-5 pairs to record in album_print_vs_rescan.csv as machine verdicts")

    if not APPLY:
        print("\n(dry run -- pass --apply to write)")
        return

    # ---------------- write ----------------
    with LOG.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]) if log else [])
        w.writeheader()
        w.writerows(log)
    print(f"wrote {LOG}")

    pvr = rows("album_print_vs_rescan.csv")
    seen = {tuple(sorted((scan_key(r["reference"]), scan_key(r["other"])))) for r in pvr}
    fields = list(pvr[0]) if pvr else PVR_FIELDS
    added = 0
    for r in sec5:
        pair = tuple(sorted((r["scan_a"], r["scan_b"])))
        if pair in seen:
            continue
        m = M[(r["scan_a"], r["scan_b"])]
        row = {k: "" for k in fields}
        row.update({"group": r["negative"], "kind": "separate-print",
                    "confidence": m["confidence"], "blobs": m["blobs"],
                    "blob_area_per_1e4": m["blob_area"], "folder": "",
                    "reference": r["scan_a"], "other": r["scan_b"],
                    "residual": m["residual"], "inliers": m["inliers"]})
        pvr.append(row)
        seen.add(pair)
        added += 1
    with PVR.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pvr)
    print(f"wrote {PVR}  (+{added} rows)")

    th = rows("album_thinned.csv")
    fields = list(th[0]) if th else THINNED_FIELDS
    for p, s, keeper, conf, neg in plan:
        if not p.exists():
            continue
        origin = P.get(s, {}).get("origin_path", "")
        row = {k: "" for k in fields}
        row.update({"date": TODAY, "deleted_path": str(p), "bytes": p.stat().st_size,
                    "sha256": sha(p), "origin": origin, "group": neg,
                    "reason": f"the family judged this one print scanned twice and kept "
                              f"{keeper}; the machine read the pair as {conf}. "
                              f"Origin untouched."})
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
