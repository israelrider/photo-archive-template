"""Audit: for every file in by-person, can it be traced back to an origins root?

Every top-level folder under the configured `archive` path is an origin root (one per
delivery: a scanning job, a phone export, a CD).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from _paths import ARCHIVE, BP

ROOTS = {d.name: d for d in sorted(ARCHIVE.iterdir()) if d.is_dir()} if ARCHIVE.is_dir() else {}

# index every origin file by basename and by stem
origin_by_name: dict[str, str] = {}
origin_by_stem: dict[str, str] = {}
counts = Counter()
for label, root in ROOTS.items():
    if not root.is_dir():
        continue
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            counts[label] += 1
            origin_by_name.setdefault(f.name.lower(), label)
            origin_by_stem.setdefault(f.stem.lower(), label)
print("origin roots:", dict(counts), "| total", sum(counts.values()))

AGE = re.compile(r"^(age\d+|noage)_(m\d+_|d_|)?", re.I)
YEAR = re.compile(r"_(18|19|20)\d\d$")

def candidates(name: str):
    """Yield plausible origin basenames for a by-person filename."""
    stem, suf = Path(name).stem, Path(name).suffix
    yield name
    s = AGE.sub("", stem)                      # drop age/method prefix
    yield s + suf
    yield YEAR.sub("", s) + suf                # drop trailing _1948 date
    parts = stem.split("_", 2)
    if len(parts) == 3:
        yield parts[2] + suf
        yield YEAR.sub("", parts[2]) + suf
    parts2 = stem.split("_")
    if len(parts2) >= 2:
        yield "_".join(parts2[1:]) + suf

orphans: dict[str, list[str]] = defaultdict(list)
traced = Counter()
total = 0
for d in sorted(p for p in BP.iterdir() if p.is_dir()):
    for f in d.glob("*"):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        total += 1
        hit = None
        for c in candidates(f.name):
            hit = origin_by_name.get(c.lower()) or origin_by_stem.get(Path(c).stem.lower())
            if hit:
                break
        if hit:
            traced[hit] += 1
        else:
            orphans[d.name].append(f.name)

print(f"\nby-person files: {total}")
for k, v in traced.most_common():
    print(f"  traced to {k}: {v}")
n_orph = sum(len(v) for v in orphans.values())
print(f"  ORPHANS (no origin found): {n_orph}")
print()
for folder, files in sorted(orphans.items()):
    print(f"  {folder} ({len(files)}):")
    for f in sorted(files)[:12]:
        print(f"      {f}")
    if len(files) > 12:
        print(f"      ... +{len(files)-12} more")
