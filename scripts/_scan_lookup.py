r"""Resolve an album file to its `photographs.csv` row. Import this; do not write it again.

**An origin file is not a scan.** An uncut sheet is one origin holding several photographs, so
`photographs.csv` can carry two rows with the same `origin_path` on different negatives:

    scans\sheet_06.png  ->  sheet_06_crew_b.png             N0120
                            sheet_06_sailors_a.png          N0138
    scans\j_07.jpg      ->  j_07_1948.jpg                   N0264
                            back of j_07_1948.jpg           N1816   (the BACK)

A dict keyed on `origin_path` silently keeps whichever row came last. Few origins do this,
which is exactly why the mistake survives every test until it doesn't: it puts the reverse
side of a print into a group of prints to choose between, reports false drift, and makes a
thinning script resolve the keeper to the back of the print and refuse a legitimate deletion.
So it lives here and the callers import it.

**Match the scan name, and prefer an exact hit.** `back of j_07_1948.jpg` ENDS WITH
`j_07_1948.jpg`, so a suffix test alone matches both rows of that sheet and resolves neither.
"""
from __future__ import annotations

import collections
import csv
import io
import pathlib
import re

# an album filename is the scan name behind a person's age prefix
STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")


def load(repo: pathlib.Path):
    """-> (album_file -> origin_path, origin_path -> [photographs rows])."""
    origin = {r["album_file"]: r["origin_path"] for r in
              csv.DictReader(io.open(repo / "data" / "MANIFEST.csv",
                                     encoding="utf-8-sig", newline=""))}
    by_origin = collections.defaultdict(list)
    for r in csv.DictReader(io.open(repo / "data" / "photographs.csv",
                                    encoding="utf-8-sig", newline="")):
        by_origin[r["origin_path"]].append(r)
    return origin, by_origin


def photo_of(origin: dict, by_origin: dict, person: str, filename: str):
    """The one `photographs.csv` row for `person/filename`, or None if it cannot be resolved.

    Returning None rather than guessing is deliberate: a caller about to delete a file should
    refuse, not proceed on the wrong row."""
    cands = by_origin.get(origin.get(f"{person}/{filename}", ""), [])
    if len(cands) == 1:
        return cands[0]
    if not cands:
        return None
    stem = STRIP.match(filename).group(1)
    exact = [c for c in cands if c["scan"] == stem]
    if len(exact) == 1:
        return exact[0]
    hits = [c for c in cands if stem.endswith(c["scan"])]
    return hits[0] if len(hits) == 1 else None


def negative_of(origin: dict, by_origin: dict, person: str, filename: str):
    r = photo_of(origin, by_origin, person, filename)
    return r["negative"] if r else None
