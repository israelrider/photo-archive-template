r"""Write the human verdicts into `data\album_same_negative.csv`.

`find_same_negative_album.py` proposes groups. Every group holding a pair in the
**0.30-0.45 residual band** must be looked at, full size, one by one, and the verdicts are
recorded in `data\same_negative_verdicts.csv`:

    group,verdict,file,note,reviewed
    51,reject,,same man bareheaded and in a fur hat -- two sittings,2026-01-31
    62,split,IMG_0009.jpg,same wall, different photograph,2026-01-31
    49,confirm,,,2026-01-31

    reject    the whole group is not one negative (different-photograph)
    split     `file` does not belong to an otherwise sound group (split-off)
    confirm   the group was checked by eye and is one negative

The typical false positive is not a threshold problem: **same scene, seconds apart**. One
man bareheaded and then in a fur hat; five people against a patterned wall and then three.
The background registers perfectly, which is exactly why the residual stays low, and the
people have moved, which is the only thing that matters.

**No cut through the band separates them**: true matches and false ones interleave across
0.30-0.45, so tightening `RESIDUAL_MAX` drops genuine groups with the false ones. The band
is settled by looking, and this script records what the looking decided -- rejected groups
stay in the file, because a group examined and refused is an answer and has to leave a trace.

It also repairs the source file. `residual_to_first` can be blank on rows whose only link
runs through a *non-first* member of the group, and a split changes which file the group's
first is. Every non-reference row is re-registered against its group's surviving reference,
so the column means one thing throughout.

    .venv\Scripts\python.exe scripts\review_album_same_negative.py [--apply]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import _archive as A
from find_same_negative_album import gray, registers_arrays

sys.stdout.reconfigure(encoding="utf-8")

APPLY = "--apply" in sys.argv
CSV = A.REPO / "data" / "album_same_negative.csv"
VERDICTS = A.REPO / "data" / "same_negative_verdicts.csv"


def load_verdicts():
    r"""(REJECT, SPLIT, CONFIRMED, REVIEWED) from data\same_negative_verdicts.csv."""
    reject, split, confirmed, reviewed = {}, {}, set(), {}
    if VERDICTS.exists():
        for r in csv.DictReader(VERDICTS.open(encoding="utf-8-sig")):
            g, v = r["group"].strip(), r["verdict"].strip()
            reviewed[g] = (r.get("reviewed") or "").strip()
            if v == "reject":
                reject[g] = r.get("note") or "examined: not one negative"
            elif v == "split":
                split[g] = (r["file"].strip(), r.get("note") or "examined: does not belong")
            elif v == "confirm":
                confirmed.add(g)
            else:
                raise SystemExit(f"{VERDICTS.name}: group {g}: unknown verdict {v!r}")
    return reject, split, confirmed, reviewed


REJECT, SPLIT, CONFIRMED, REVIEWED = load_verdicts()


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    print(f"{len(rows)} rows, {len(groups)} groups")

    # 1. verdicts
    for g, members in groups.items():
        if g in REJECT:
            for r in members:
                r["verdict"] = "different-photograph"
                r["review_note"] = REJECT[g]
                r["reviewed"] = REVIEWED[g]
            continue
        odd = SPLIT.get(g)
        for r in members:
            name = Path(r["path"]).name
            if odd and name == odd[0]:
                r["verdict"] = "split-off"
                r["review_note"] = odd[1]
                r["reviewed"] = REVIEWED[g]
            elif g in CONFIRMED:
                r["verdict"] = "same-negative"
                r["review_note"] = "confirmed by eye in the 0.30-0.45 band"
                r["reviewed"] = REVIEWED[g]
            else:
                r["verdict"] = "same-negative"
                r["review_note"] = "machine only; residual below the band that needed review"
                r["reviewed"] = ""
        if odd and not any(Path(r["path"]).name == odd[0] for r in members):
            raise SystemExit(f"group {g}: {odd[0]} not found")

    # 2. re-register every kept row against its group's surviving reference
    kept = {g: [r for r in m if r["verdict"] == "same-negative"] for g, m in groups.items()}
    # every row, not just the kept ones: a rejected or split-off file keeps the score that
    # flagged it, otherwise the file records a refusal with no evidence behind it
    need = sorted({r["path"] for r in rows})
    print(f"decoding {len(need)} files to recompute residual_to_first")
    img = {}
    for n, p in enumerate(need, 1):
        try:
            img[p] = gray(Path(p), 900)
        except Exception as e:
            print(f"  unreadable: {p}: {e}")
            img[p] = None
        if n % 100 == 0:
            print(f"  {n}/{len(need)}", flush=True)

    changed = 0
    for g, members in groups.items():
        ref = kept[g][0] if kept.get(g) else members[0]
        for r in members:
            if r is ref:
                new_res, new_inl = "", ""
            else:
                res, inl = registers_arrays(img.get(ref["path"]), img.get(r["path"]))
                new_res = "" if res is None else f"{res:.3f}"
                new_inl = "" if res is None else inl
            if (r["residual_to_first"], str(r["inliers_to_first"])) != (new_res, str(new_inl)):
                changed += 1
            r["residual_to_first"], r["inliers_to_first"] = new_res, new_inl

    print(f"{changed} rows got a corrected residual/inlier figure")
    v = defaultdict(int)
    for r in rows:
        v[r["verdict"]] += 1
    for k in sorted(v):
        print(f"  {v[k]:4}  {k}")
    good = {g for g, m in kept.items() if len(m) > 1}
    print(f"\n{len(good)} groups survive review, "
          f"{sum(len(kept[g]) for g in good)} files, "
          f"{sum(len(kept[g]) - 1 for g in good)} redundant copies")

    if not APPLY:
        print("\ndry run -- nothing written. Re-run with --apply.")
        return

    fields = ["group", "path", "folder", "residual_to_first", "inliers_to_first",
              "other_folders_with_this_content", "verdict", "review_note", "reviewed"]
    with CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {CSV.relative_to(A.REPO)}")


if __name__ == "__main__":
    main()
