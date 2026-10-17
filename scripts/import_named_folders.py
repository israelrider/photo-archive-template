"""Turn the family's renamed Person_NN folders into data\\main_import_log.csv.

`images-agent albums` builds by-person\\Person_NN\\ -- one folder per face cluster -- and
writes album_manifest.csv beside it (path, num_people, person_clusters, ...). The family
then looks at each folder and RENAMES it to the person's name. That rename is the first and
largest identification the archive gets, and it lives only in a folder name until this
records it as (person, cluster) rows that named_clusters() and every review can read.

A renamed folder no longer says which cluster it was, so this asks the files: every file
in the folder is a copy of an origin listed in album_manifest.csv with its clusters, and the
cluster most of the folder's files share is the folder's cluster. A folder whose vote is
not clear (the winner holds under --min-share of its files) is reported and NOT recorded:
two people merged into one folder is a question for the family, not a tie to break.

Folders still called Person_NN, Unsorted_Faces and No_People are left alone. Re-running
replaces the rows for every folder it records and keeps everything else.

    .venv\\Scripts\\python.exe scripts\\import_named_folders.py [--min-share=0.6] [--apply]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _paths import BP, LOGS

sys.stdout.reconfigure(encoding="utf-8")

APPLY = "--apply" in sys.argv
MIN_SHARE = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--min-share=")), 0.6)
MANIFEST = BP / "album_manifest.csv"
OUT = LOGS / "main_import_log.csv"
FIELDS = ["person", "file", "source", "cluster", "age", "status", "note"]
SKIP = ("Person_", "Unsorted_Faces", "No_People", "Unknown", ".")
IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic"}


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found - run `images-agent albums` first")
    by_name: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for r in csv.DictReader(MANIFEST.open(encoding="utf-8-sig")):
        cl = [int(c) for c in (r.get("person_clusters") or "").split(";") if c.lstrip("-").isdigit()]
        by_name[Path(r["path"]).name.lower()].append((r["path"], cl))

    rows, unclear = [], []
    for d in sorted(p for p in BP.iterdir() if p.is_dir() and not p.name.startswith(SKIP)):
        files = [f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMG]
        votes, src_of = Counter(), {}
        for f in files:
            for path, cl in by_name.get(f.name.lower(), []):
                for c in cl:
                    if c >= 0:
                        votes[c] += 1
                src_of.setdefault(f, path)
        if not votes:
            unclear.append((d.name, "no file matches album_manifest.csv"))
            continue
        c, n = votes.most_common(1)[0]
        share = n / max(1, len(files))
        if share < MIN_SHARE:
            unclear.append((d.name, f"best cluster c{c} holds only {n}/{len(files)} files; "
                                    f"runner-up {votes.most_common(2)[1:] or 'none'}"))
            continue
        for f in files:
            rows.append({"person": d.name, "file": f.name, "source": src_of.get(f, ""),
                         "cluster": c, "age": "", "status": "ok",
                         "note": "renamed Person_ folder (import_named_folders.py)"})
        print(f"   c{c:<5} {d.name:<32} {len(files):4d} files  ({share:.0%} vote)")

    if unclear:
        print(f"\n{len(unclear)} folders NOT recorded -- ask the family, or split the folder:")
        for name, why in unclear:
            print(f"   {name}: {why}")

    if not APPLY:
        print(f"\n{len(rows)} rows. DRY RUN - pass --apply to write {OUT.name}.")
        return
    recorded = {r["person"] for r in rows}
    prior = list(csv.DictReader(OUT.open(encoding="utf-8-sig"))) if OUT.exists() else []
    keep = [r for r in prior if r["person"] not in recorded]
    LOGS.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(keep + rows)
    print(f"\nwrote {len(keep) + len(rows)} rows -> {OUT} ({len(keep)} kept)")


if __name__ == "__main__":
    main()
