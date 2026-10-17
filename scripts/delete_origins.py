r"""Erase the origins from this PC, but only the ones an external copy provably still holds.

Deletes everything under the configured `archive` folder and nothing else. The album roots
must be peers of it on disk, not children, so nothing here can reach them -- preflight
refuses to start otherwise.

The safety rule is the one every deletion in the archive uses: a file goes only after a
byte-identical twin has been read back somewhere else. The external backup is that
somewhere else, so this script does not take "I made the copy" as an answer -- it is
pointed at the copy and hashes it. A file whose twin is missing, or differs, stops
the whole run; nothing is deleted while any file is unaccounted for.

This is why it must not be run before the backup exists. It cannot be: with no external
copy to point at, it refuses to start.

    python scripts\delete_origins.py --against E:\family-archive           # dry run
    python scripts\delete_origins.py --against E:\family-archive --apply   # deletes

The dry run is the real output. `--apply` only repeats it and then unlinks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import stat
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ALBUM_ROOTS, ARCHIVE, LOGS  # noqa: E402
LEDGER = LOGS / "origins_erased.csv"
LEDGER_COLS = ["date", "deleted_path", "bytes", "sha256", "verified_twin"]
FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse(msg: str) -> int:
    print(f"REFUSING: {msg}")
    return 2


def preflight(external: Path) -> int:
    """Everything that must be true before a single byte is hashed."""
    if not ARCHIVE.is_dir():
        return refuse(f"{ARCHIVE} does not exist -- already erased, or the wrong machine")
    for root in ALBUM_ROOTS:
        if not root.is_dir():
            return refuse(f"album root {root} is missing; a run that cannot see an album "
                          "has no business deleting anything")
        if root.resolve() == ARCHIVE.resolve() or ARCHIVE.resolve() in root.resolve().parents:
            return refuse(f"album root {root} lies inside {ARCHIVE}")
    if not external.is_dir():
        return refuse(f"--against {external} does not exist; make the step 2 copy first")
    if external.resolve().drive.upper() == ARCHIVE.resolve().drive.upper():
        return refuse(f"--against {external} is on the same drive as the originals; the "
                      "copy this checks against has to survive this drive")
    # A reparse point under the origins would make a delete walk somewhere else entirely.
    for d, dirs, files in os.walk(ARCHIVE):
        for name in dirs + files:
            attrs = getattr((Path(d) / name).lstat(), "st_file_attributes", 0)
            if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
                return refuse(f"{Path(d) / name} is a link, not a real file")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", required=True, type=Path,
                    help="the step 2 external copy of family-archive")
    ap.add_argument("--apply", action="store_true", help="delete; without it, nothing moves")
    args = ap.parse_args()

    external = args.against
    bad = preflight(external)
    if bad:
        return bad

    local = sorted(p for p in ARCHIVE.rglob("*") if p.is_file())
    print(f"{len(local)} files under {ARCHIVE}")
    print(f"checking each against {external}\n")

    covered, missing, differs = [], [], []
    done_bytes = 0
    for i, src in enumerate(local, 1):
        rel = src.relative_to(ARCHIVE)
        twin = external / rel
        if not twin.is_file():
            missing.append(rel)
        else:
            digest = sha256(src)
            if sha256(twin) != digest:
                differs.append(rel)
            else:
                covered.append((src, twin, src.stat().st_size, digest))
                done_bytes += src.stat().st_size
        if i % 250 == 0 or i == len(local):
            print(f"  {i}/{len(local)}  {done_bytes / 2**30:.1f} GiB verified")

    print(f"\nverified byte-identical on the copy: {len(covered)}")
    print(f"missing from the copy:               {len(missing)}")
    print(f"present but DIFFERENT:               {len(differs)}")
    for rel in missing[:40]:
        print(f"  MISSING  {ARCHIVE / rel}")
    for rel in differs[:40]:
        print(f"  DIFFERS  {ARCHIVE / rel}")
    if missing or differs:
        return refuse(f"{len(missing) + len(differs)} files are not safely on the copy. "
                      "Nothing deleted. Re-run step 2, then run this again.")

    if not args.apply:
        print(f"\nDRY RUN. {len(covered)} files, {done_bytes / 2**30:.1f} GiB would be erased.")
        print("Re-run with --apply to delete them.")
        return 0

    new = not LEDGER.exists()
    with LEDGER.open("ab") as fh:
        def row(vals):
            fh.write((",".join('"' + str(v).replace('"', '""') + '"'
                               for v in vals) + "\n").encode("utf-8"))
        if new:
            row(LEDGER_COLS)
        today = date.today().isoformat()
        for src, twin, size, digest in covered:
            src.chmod(stat.S_IWRITE)
            src.unlink()
            row([today, src, size, digest, twin])
            fh.flush()

    for d, dirs, files in os.walk(ARCHIVE, topdown=False):
        if not os.listdir(d):
            os.rmdir(d)
    print(f"\nerased {len(covered)} files, {done_bytes / 2**30:.1f} GiB")
    print(f"ledger: {LEDGER}")
    print(f"{ARCHIVE} {'is gone' if not ARCHIVE.exists() else 'still holds something'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
