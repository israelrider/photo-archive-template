"""Build a share folder as a derived copy of the albums. Dry run unless --apply.

The share is not a curated subset: every album root (archive.toml: by_person and
other_albums) is copied, and the audience decision happens later, per folder, per person,
e.g. in Google Drive's sharing UI. So this script excludes nothing on grounds of audience.
It excludes only three kinds of thing:

  1. service files the family should never see  -- desktop.ini, Thumbs.db, CSVs, vectors
  2. working state that is not the album        -- .thinned\\, .misassigned\\
  3. nothing else

It also renames, which is the one place it changes what a folder says. `unknown-persons\\`
becomes the [share] unknown_folder name and each filename loses its `unknown-person-NNN__`
prefix, because every uploaded folder must be safe to hand to a person ON ITS OWN, and
`unknown-person-043` is a machine's name for somebody's relative. Per-cluster
`unknown-person-NNN\\` folders are dropped: `unknown-persons\\` must be a byte-identical
superset of them (check before relying on it).

What the rename costs is the cluster id that lived in the filename. That is the same
`age39_...jpg` problem the plan warns about: an album filename names the face that claimed
the photograph, not the adult in it, and the prefix named a machine's cluster, not a person.
It is recoverable from MANIFEST.csv, which belongs on the backup disks beside the photographs.
The metadata is excluded from the share, never destroyed.

Every copy is verified by sha256 against its source before the run is called clean, and the
run appends one row per file to data\\share_build_log.csv. Nothing is ever deleted from an
album root: this script only reads them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths as P  # noqa: E402

# --- what ships -------------------------------------------------------------
# Each entry: (source root, destination name, directory names to skip)
#
# by-person\Unknown SHIPS by default. Unknown faces can be a large share of everything filed;
# as an opt-in flag they would be absent from the share, and nobody would notice until the
# originals were gone. A default that has to be remembered at the right moment is not a
# default. --exclude-unknown is there for a share that deliberately leaves it out.
WORKING_STATE = {".thinned", ".misassigned"}
ROOTS = [(root, root.name, set(WORKING_STATE)) for root in P.ALBUM_ROOTS]

IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO = {".mp4", ".avi", ".mov", ".mts", ".m4v", ".wmv", ".3gp", ".mpg", ".mpeg"}
SHIPS = IMAGE | VIDEO

# Service files, by exact name and by extension. Excluded from the share, not destroyed.
JUNK_NAMES = {"desktop.ini", "thumbs.db", "manifest.csv", "album_manifest.csv"}
JUNK_EXT = {".ini", ".db", ".csv", ".npy", ".npz", ".pkl", ".py", ".json", ".sqlite",
            ".bak", ".txt", ".log", ".md", ".lnk", ".url"}

# --- the one rename ---------------------------------------------------------
UNKNOWN_FLAT = "unknown-persons"
UNKNOWN_DEST = P.CONFIG.get("share", {}).get("unknown_folder", "Неопознанные")
UNKNOWN_PREFIX_DROP = "unknown-person-"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_unknown_prefix(name: str) -> str:
    """`unknown-person-043__age16_v01_X.jpg` -> `age16_v01_X.jpg`.

    Also handles the two-cluster form `unknown-person-827+850__X.jpg`. A name that does not
    carry the prefix is returned untouched, so this is safe to run over the whole folder.
    """
    if not name.startswith(UNKNOWN_PREFIX_DROP):
        return name
    _, sep, rest = name.partition("__")
    return rest if sep and rest else name


def is_shippable(path: Path) -> bool:
    if path.name.lower() in JUNK_NAMES:
        return False
    ext = path.suffix.lower()
    if ext in JUNK_EXT:
        return False
    return ext in SHIPS


def plan_root(src_root: Path, dest_name: str, skip_dirs: set[str], dest_base: Path):
    """Yield (source, destination, note) for one album root. Reads only."""
    if not src_root.exists():
        raise SystemExit(f"source root is missing: {src_root}")

    for dirpath, dirnames, filenames in os.walk(src_root):
        here = Path(dirpath)
        # Drop the per-person unknown folders: unknown-persons is a byte-identical superset.
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in skip_dirs
            and not (d.startswith(UNKNOWN_PREFIX_DROP) and d != UNKNOWN_FLAT)
        )
        rel = here.relative_to(src_root)
        parts = list(rel.parts)
        note = ""
        if parts and parts[0] == UNKNOWN_FLAT:
            parts[0] = UNKNOWN_DEST
            note = "renamed folder"
        out_dir = dest_base / dest_name / Path(*parts) if parts else dest_base / dest_name

        for fn in sorted(filenames):
            src = here / fn
            if not is_shippable(src):
                continue
            out_name = fn
            if note:
                out_name = strip_unknown_prefix(fn)
                if out_name != fn:
                    note = "renamed folder + stripped prefix"
            yield src, out_dir / out_name, note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=P.SHARE,
                    help=r"where to build the share, e.g. D:\archive-share (default: paths.share)")
    ap.add_argument("--apply", action="store_true",
                    help="actually copy; without it the run only reports")
    ap.add_argument("--exclude-unknown", action="store_true",
                    help=r"leave by-person\Unknown\ out; it ships by default")
    args = ap.parse_args()
    if args.dest is None:
        raise SystemExit("no destination: pass --dest or set paths.share in archive.toml")

    roots = ROOTS
    if args.exclude_unknown:
        roots = [(s, d, sk | {"Unknown"}) if s == P.BP else (s, d, sk) for s, d, sk in ROOTS]

    dest_base: Path = args.dest.resolve()
    if dest_base.exists() and any(dest_base.iterdir()) and args.apply:
        raise SystemExit(f"destination is not empty, refusing to merge into it: {dest_base}")

    items: list[tuple[Path, Path, str]] = []
    per_root: dict[str, tuple[int, int]] = {}
    for src_root, dest_name, skip in roots:
        got = list(plan_root(src_root, dest_name, skip, dest_base))
        items.extend(got)
        per_root[dest_name] = (len(got), sum(p.stat().st_size for p, _, _ in got))

    total_files = len(items)
    total_bytes = sum(n for _, n in per_root.values())

    print(f"{'root':<16}{'files':>9}{'GB':>9}")
    for name, (n, b) in per_root.items():
        print(f"{name:<16}{n:>9}{b / 2**30:>9.1f}")
    print(f"{'TOTAL':<16}{total_files:>9}{total_bytes / 2**30:>9.1f}")

    renamed = [(s, d) for s, d, note in items if note]
    print(f"\nrenamed: {len(renamed)} files under {UNKNOWN_DEST}/")
    for s, d in renamed[:3]:
        print(f"  {s.name}\n    -> {d.name}")

    collisions = {}
    for _, d, _ in items:
        collisions[d] = collisions.get(d, 0) + 1
    clash = {d: n for d, n in collisions.items() if n > 1}
    if clash:
        print(f"\nREFUSING: {len(clash)} destination paths are claimed twice")
        for d in list(clash)[:10]:
            print(f"  {d}")
        return 2

    free = shutil.disk_usage(dest_base.anchor).free
    print(f"\nfree at {dest_base.anchor}: {free / 2**30:.1f} GB "
          f"/ need {total_bytes / 2**30:.1f} GB")
    if args.apply and free < total_bytes * 1.02:
        raise SystemExit("not enough free space, refusing to start")

    if not args.apply:
        print("\ndry run: nothing was copied. Re-run with --apply.")
        return 0

    log = P.LOGS / "share_build_log.csv"
    new = not log.exists()
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    copied = failed = 0
    with open(log, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "source", "dest", "sha256", "bytes", "note", "verified"])
        for i, (src, dst, note) in enumerate(items, 1):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            a, b = sha256(src), sha256(dst)
            ok = a == b
            copied += ok
            failed += not ok
            w.writerow([stamp, str(src), str(dst), a, src.stat().st_size, note,
                        "yes" if ok else "SHA256 MISMATCH"])
            if not ok:
                print(f"  MISMATCH {src}")
            if i % 250 == 0:
                print(f"  {i}/{total_files}", flush=True)

    print(f"\ncopied {copied} verified byte-identical, {failed} mismatched")
    print(f"log: {log}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
