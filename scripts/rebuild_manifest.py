r"""Rebuild data\MANIFEST.csv resolving origins by CONTENT, not basename.

A scanner restarts its numbering in every job, so one basename can name several different
origins (`0001.jpg` in every folder). Name matching binds album files to the wrong origin.
Strategy: match on file size first (cheap), disambiguate collisions with sha256. Falls back
to the hand-made mapping in data\crop_origins.csv for crops and versos, which are
genuinely different bytes.

Origin roots come from archive.toml ([origins]). A quarantine of removed duplicates is
deliberately NOT a root: resolving an album file to a quarantined origin would record
provenance that is about to be deleted.

STAGE ONE OF TWO. Album files that are edited versions - rotated, cropped, re-saved - match
no origin by bytes, so this script reports them UNRESOLVED with an empty origin. That is
not the finished manifest: `patch_derived.py` resolves them by name and marks them
`derived`. Running this script alone WIPES those origins. Always run the pair:

    .venv\Scripts\python.exe scripts\rebuild_manifest.py
    .venv\Scripts\python.exe scripts\patch_derived.py

When a path or a root moves, everything naming it has to move too, and nothing here fails
loudly enough to say it did not. Change archive.toml, never a script.
"""
from __future__ import annotations

import csv
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from _paths import ARCHIVE, BORN_DIGITAL, BP, EXTERNAL_ORIGINS, LOGS, ORIGIN_ROOTS

# Non-ASCII person folders in the UNRESOLVED report kill a run on the console's cp1252
# default, AFTER the manifest has been written.
sys.stdout.reconfigure(encoding="utf-8")

EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAN = LOGS / "MANIFEST.csv"
# album_file name -> (origin path relative to the archive, relation). For album files that
# are a crop or the back of a print: their bytes match nothing, so they are mapped by hand.
CROPS = LOGS / "crop_origins.csv"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- catalogue every candidate origin, indexed by size ----
by_size: dict[int, list[Path]] = defaultdict(list)
roots = [*ORIGIN_ROOTS, *EXTERNAL_ORIGINS]
n = 0
for root in roots:
    if not root.is_dir():
        continue
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in EXT:
            by_size[f.stat().st_size].append(f)
            n += 1
print(f"catalogued {n} candidate origin files across {len([r for r in roots if r.is_dir()])} roots")

DERIVED: dict[str, tuple[Path, str]] = {}
if CROPS.exists():
    for r in csv.DictReader(CROPS.open(encoding="utf-8-sig")):
        DERIVED[r["album_file"]] = (ARCHIVE / r["origin"], r.get("relation") or "crop")


def relation_for(origin: Path) -> str:
    for root in ORIGIN_ROOTS:
        if root.name in BORN_DIGITAL and origin.is_relative_to(root):
            return "born-digital"
    if any(origin.is_relative_to(e) for e in EXTERNAL_ORIGINS):
        return "external"
    return "copy"


rows, unresolved, ambiguous_fixed = [], [], 0
hash_cache: dict[Path, str] = {}

for d in sorted(p for p in BP.iterdir() if p.is_dir()):
    for f in sorted(d.glob("*")):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        origin = rel = None
        if f.name in DERIVED:
            origin, rel = DERIVED[f.name]
        else:
            cands = by_size.get(f.stat().st_size, [])
            if len(cands) == 1:
                origin = cands[0]
            elif len(cands) > 1:
                want = sha(f)
                for c in cands:
                    if hash_cache.setdefault(c, sha(c)) == want:
                        origin = c
                        break
                if origin and len({c.name for c in cands}) > 1 or (origin and len(cands) > 1):
                    ambiguous_fixed += 1
            if origin is not None:
                rel = relation_for(origin)
        if origin is None:
            unresolved.append(f"{d.name}/{f.name}")
            rel = "UNRESOLVED"
        rows.append({"album_file": f"{d.name}/{f.name}", "person": d.name,
                     "relation": rel, "origin_path": str(origin) if origin else ""})

with MAN.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=["album_file", "person", "relation", "origin_path"])
    w.writeheader(); w.writerows(rows)

print(f"\nMANIFEST rebuilt: {len(rows)} rows")
print("relations:", dict(Counter(r["relation"] for r in rows)))
print(f"resolved via content hash (size collision): {ambiguous_fixed}")
print("origin paths missing on disk:", sum(1 for r in rows if r['origin_path'] and not Path(r['origin_path']).exists()))
if unresolved:
    print(f"\nUNRESOLVED ({len(unresolved)}):")
    for u in unresolved[:20]:
        print("   ", u)
