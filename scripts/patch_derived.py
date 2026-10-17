"""Resolve the UNRESOLVED manifest rows: edited versions (rotated/cropped/re-saved) whose
bytes differ from their origin. Resolve them by name -- but only when the name is
unambiguous -- and mark relation='derived' so they are never mistaken for exact copies.
Then diff the content-based manifest against the last basename-based backup, if any.

STAGE TWO OF TWO. `rebuild_manifest.py` runs first and marks these rows UNRESOLVED with an
empty origin; this script fills them in. Running the first WITHOUT the second is a silent
regression - it wipes every `derived` origin that only this script can restore. Always run
the pair, in order.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _paths import BP, EXTERNAL_ORIGINS, LOGS, ORIGIN_ROOTS

# Non-ASCII person folders in the report kill a run on the console's cp1252 default,
# AFTER the manifest has been written.
sys.stdout.reconfigure(encoding="utf-8")

MAN = LOGS / "MANIFEST.csv"
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

by_name: dict[str, list[Path]] = defaultdict(list)
for root in [*ORIGIN_ROOTS, *EXTERNAL_ORIGINS]:
    if root.is_dir():
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in EXT:
                by_name[f.stem.lower()].append(f)

# An album name is `<age>_<source tag>_<origin stem>`, the tag being any letter and digits
# (m01, a02, ...). A pattern that knows only one letter leaves every file from the other
# roots UNRESOLVED.
AGE = re.compile(r"^(age\d+|noage)_([a-z]\d+_)?", re.I)
DATESUF = re.compile(r"_((18|19|20)\d\d|\d{8}|\d\d)$")

rows = list(csv.DictReader(MAN.open(encoding="utf-8-sig")))
unresolved_before = sum(1 for r in rows if r["relation"] == "UNRESOLVED")
fixed = 0
for r in rows:
    if r["relation"] != "UNRESOLVED":
        continue
    stem = Path(r["album_file"]).stem
    for cand in (stem, AGE.sub("", stem), DATESUF.sub("", AGE.sub("", stem))):
        hits = by_name.get(cand.lower(), [])
        if len(hits) == 1:
            r["origin_path"] = str(hits[0])
            r["relation"] = "derived"
            fixed += 1
            break

with MAN.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["album_file", "person", "relation", "origin_path"])
    w.writeheader(); w.writerows(rows)

# The count comes from the data: every orientation fix adds derived files.
print(f"resolved {fixed} of {unresolved_before} as 'derived'")
print("relations:", dict(Counter(r["relation"] for r in rows)))
print("still unresolved:", [r["album_file"] for r in rows if r["relation"] == "UNRESOLVED"])
print("origin paths missing on disk:", sum(1 for r in rows if r["origin_path"] and not Path(r["origin_path"]).exists()))

# ---- diff against the old basename-built manifest ----
baks = sorted(BP.glob("MANIFEST.csv.bak-*"))
if baks:
    old = {r["album_file"]: r["origin_path"] for r in csv.DictReader(baks[-1].open(encoding="utf-8-sig"))}
    changed = [(k, old.get(k, ""), r["origin_path"]) for k, r in
               ((r["album_file"], r) for r in rows) if old.get(k) != r["origin_path"]]
    print(f"\nvs old basename manifest ({baks[-1].name}): {len(changed)} rows changed origin_path")
    for k, o, nw in changed[:8]:
        print(f"   {k}")
        print(f"      was: {o}")
        print(f"      now: {nw}")
