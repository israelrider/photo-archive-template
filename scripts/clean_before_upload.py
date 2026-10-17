r"""Take the service files and the withdrawal trees out of the albums before uploading.

What uploads to Drive is what the family opens, so the albums should hold photographs and
the things that explain them -- nothing else. Two kinds of thing are removed here.

The service files: desktop.ini, Thumbs.db, the dated *.csv.bak-* backups, and thumbnail
caches some album programs leave beside the photographs (*.jpg_320x320 etc.). No
photograph is among them and none is verified against anything, because there is nothing
to lose.

The withdrawal trees, <album>\.thinned and <album>\.misassigned: real photographs, pulled
out of the album on purpose during review. They are verified byte-identical against the
backup copy first, because a withdrawn photograph is still a photograph and this may be
the last copy leaving the PC.

Deliberately NOT removed, though a junk filter written by extension would take them:
  * .htm pages saved from Word (poems, letters, notes) and the .xml / .thmx files inside
    their _files\ folders, which are what makes them open;
  * _manifest.csv / album_manifest.csv -- they tell whoever opens the folder what is in it.

--against is the backup copy of the folder that holds the album roots, laid out the same
way (e.g. the albums live under C:\Users\me\Pictures and the backup is D:\Pictures).

    python scripts\clean_before_upload.py --against D:\Pictures           # dry run
    python scripts\clean_before_upload.py --against D:\Pictures --apply
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import ALBUM_ROOTS, LOGS  # noqa: E402

# The folder the album roots sit in; --against mirrors it.
PICTURES = Path(os.path.commonpath([str(r.parent) for r in ALBUM_ROOTS]))
SERVICE_ROOTS = list(ALBUM_ROOTS)
SERVICE = ["desktop.ini", "Thumbs.db", "*.csv.bak-*", "*.jpg_56x42", "*.jpg_320x320",
           "*.jpg_170x128"]
TREES = [root / name for root in ALBUM_ROOTS for name in (".thinned", ".misassigned")]
LEDGER = LOGS / "cleaned_before_upload.csv"
COLS = ["date", "deleted_path", "bytes", "kind", "sha256", "verified_twin"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", required=True, type=Path,
                    help=f"the backup copy of {PICTURES}")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.against.is_dir():
        print(f"REFUSING: {args.against} does not exist")
        return 2

    # The two lists must not overlap. .misassigned carries its own desktop.ini, and it
    # matched both sweeps: the service pass unlinked it, then the photograph pass tried to
    # stat a file that was already gone. A service file is service wherever it sits, so the
    # service sweep keeps the whole album; it is the photograph pass that yields.
    service = sorted({p for root in SERVICE_ROOTS for pat in SERVICE
                      for p in root.rglob(pat) if p.is_file()})
    photos = sorted(p for tree in TREES for p in tree.rglob("*")
                    if p.is_file() and not any(p.match(pat) for pat in SERVICE))

    print(f"service files: {len(service)}, "
          f"{sum(p.stat().st_size for p in service) / 2**20:.1f} MiB")
    print(f"withdrawn photographs: {len(photos)}, "
          f"{sum(p.stat().st_size for p in photos) / 2**20:.1f} MiB\n")

    verified, unsafe = [], []
    for src in photos:
        twin = args.against / src.relative_to(PICTURES)
        if not twin.is_file():
            unsafe.append((src, "missing from the copy"))
            continue
        digest = sha256(src)
        if sha256(twin) != digest:
            unsafe.append((src, "DIFFERS on the copy"))
        else:
            verified.append((src, twin, digest))
    for src, why in unsafe:
        print(f"  {why}: {src}")
    if unsafe:
        print(f"\nREFUSING: {len(unsafe)} withdrawn photographs are not safely on the copy. "
              "Nothing deleted.")
        return 2
    print(f"all {len(verified)} withdrawn photographs verified byte-identical on the copy")

    if not args.apply:
        print("\nDRY RUN. Re-run with --apply.")
        return 0

    new = not LEDGER.exists()
    with LEDGER.open("ab") as fh:
        def row(vals):
            fh.write((",".join('"' + str(v).replace('"', '""') + '"'
                               for v in vals) + "\n").encode("utf-8"))
        if new:
            row(COLS)
        today = date.today().isoformat()
        for p in service:
            size = p.stat().st_size
            p.unlink()
            row([today, p, size, "service", "", ""])
        for src, twin, digest in verified:
            size = src.stat().st_size
            src.unlink()
            row([today, src, size, "withdrawn-photograph", digest, twin])
    for tree in TREES:
        shutil.rmtree(tree, ignore_errors=True)
    # A thumbnail-cache folder holds nothing but what it cached. Prune only the folders THIS
    # run emptied: an empty person folder that was already empty is somebody's decision.
    for d in sorted({p.parent for p in service}, key=lambda p: -len(p.parts)):
        while d not in SERVICE_ROOTS and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            d = d.parent
    print(f"\nremoved {len(service)} service files and {len(verified)} withdrawn photographs")
    print(f"ledger: {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
