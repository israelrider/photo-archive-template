r"""Give a JPEG an EXIF Orientation it never had, without touching a single pixel.

`fix_orientation_value.py` rewrites a WRONG orientation value in an existing APP1 segment:
two bytes change and the file length is unchanged. It cannot help a file that has no EXIF
block at all, and scanner software often writes such files -- `FFD8 FFE0`, a JFIF APP0 and
nothing else, so there is no tag to correct.

Typical case: a portrait photograph stored in a 4944x3662 landscape frame, so every viewer
lays it on its side. It needs orientation 8, "rotate 90 degrees counter-clockwise to
display" (6 = clockwise, 3 = upside down, 1 = as stored).

**Rotating the pixels is not an option.** Re-encoding a JPEG loses quality, and this archive's
orientation work has always proved pixels identical before and after. A lossless transform is
not available either: lossless JPEG rotation needs both dimensions to be multiples of the MCU,
and 3662 is not a multiple of 16 (the MCU, checked per file).

So a minimal APP1 segment is built by hand and inserted after the existing APP0, carrying one
IFD0 entry and nothing else:

    FFE1 0022  "Exif\0\0"  II 2A00 08000000  0100  0112 0003 00000001 0008 0000  00000000

36 bytes: SOI and every other byte of the original stream are copied through untouched, so the
entropy-coded scan is bit-for-bit what it was. The file grows by 36 bytes and the decoded
pixels are compared exactly before and after; the write is rejected if a single value differs.

The album copy therefore stops being byte-identical to its origin, which is a **recorded**
category rather than damage: the thinning scripts skip such files and `album_thinned.csv`
carries restore instructions for them. Origins are never modified -- the origin here has no
EXIF block either, and keeps none.

    .venv\Scripts\python.exe scripts\add_orientation_tag.py <person> <file> <1|3|6|8> [--apply]
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import pathlib
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")

REPO = pathlib.Path(__file__).resolve().parent.parent
from _paths import BP
from _paths import read_ledger
CSVP = REPO / "data" / "orientation_corrections.csv"
MAN = REPO / "data" / "MANIFEST.csv"
TODAY = datetime.date.today().isoformat()
APPLY = "--apply" in sys.argv


def note(w: int, h: int, value: int) -> str:
    """Compose the log note from THIS file's measurements.

    A module-level constant describing one file -- a portrait in a 4944x3662 landscape
    frame -- Reused verbatim it records false measurements against
    every later file, and orientation_corrections.csv is the archive's permanent account of
    what was changed and why. So the two facts that vary are measured per file: which way the
    frame sits versus the photograph inside it, and which dimension blocks a lossless rotate.
    """
    frame = "landscape" if w > h else "portrait"
    shown = "portrait" if (value in (6, 8)) == (w > h) else "landscape"
    blockers = [str(d) for d in (w, h) if d % 16]
    mcu = (f"which {' and '.join(blockers)} {'is' if len(blockers) == 1 else 'are'} not"
           if blockers else "which both dimensions satisfy, but the tag is the smaller edit")
    return ("no EXIF block at all -- the file was FFD8 FFE0, a JFIF APP0 and nothing else, so "
            "fix_orientation_value.py had no tag to correct. The photograph inside is a "
            f"{shown} stored in a {w}x{h} {frame} frame, so every viewer laid it on its side. "
            "A minimal 36-byte APP1 carrying only IFD0 Orientation was inserted after the "
            "APP0; the entropy-coded scan is copied through bit-for-bit and the decoded "
            "pixels were compared exactly before and after. Rotating the pixels was rejected: "
            "re-encoding loses quality and lossless JPEG rotation needs both dimensions to be "
            f"multiples of the MCU, {mcu}. The ORIGIN has no EXIF block either and was not "
            "modified, so this album copy is deliberately no longer byte-identical to it.")


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def app1(value: int) -> bytes:
    tiff = (b"II" + (42).to_bytes(2, "little") + (8).to_bytes(4, "little")
            + (1).to_bytes(2, "little")
            + (0x0112).to_bytes(2, "little") + (3).to_bytes(2, "little")
            + (1).to_bytes(4, "little") + value.to_bytes(2, "little") + b"\x00\x00"
            + (0).to_bytes(4, "little"))
    body = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body


def insert_after_app0(raw: bytes, seg: bytes) -> bytes:
    if raw[:2] != b"\xff\xd8":
        raise SystemExit("REFUSED: not a JPEG (no SOI)")
    i = 2
    # step over APP0 and any other leading APPn, stop at the first non-APP marker
    while i + 4 <= len(raw) and raw[i] == 0xFF and 0xE0 <= raw[i + 1] <= 0xEF:
        if raw[i + 1] == 0xE1 and raw[i + 4:i + 10] == b"Exif\x00\x00":
            raise SystemExit("REFUSED: the file already has an EXIF APP1 -- use "
                             "fix_orientation_value.py to change its value instead")
        i += 2 + int.from_bytes(raw[i + 2:i + 4], "big")
    return raw[:i] + seg + raw[i:]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    person, name, value = args[0], args[1], int(args[2])
    if value not in (1, 3, 6, 8):
        raise SystemExit("REFUSED: orientation must be 1, 3, 6 or 8")

    p = BP / person / name
    if not p.exists():
        raise SystemExit(f"REFUSED: {p} is not on disk")
    # Every handle must be CLOSED before the replace: Windows refuses to rename over a file
    # PIL still has open, and a half-finished orientation edit is not something to leave on
    # an irreplaceable scan.
    with Image.open(p) as im:
        if im.getexif().get(0x0112) is not None:
            raise SystemExit("REFUSED: this file already carries an orientation tag")
        before = np.asarray(im).copy()

    raw = p.read_bytes()
    out = insert_after_app0(raw, app1(value))
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(out)
    try:
        with Image.open(tmp) as after_im:
            after = np.asarray(after_im).copy()
            tag = after_im.getexif().get(0x0112)
        if tag != value:
            raise SystemExit("REFUSED: the tag did not read back as written")
        if before.shape != after.shape or not np.array_equal(before, after):
            raise SystemExit("REFUSED: decoded pixels changed")
        print(f"  {person}/{name}")
        print(f"    stored {before.shape[1]}x{before.shape[0]}, pixels identical after the edit")
        print(f"    orientation tag added: {value}   file {len(raw)} -> {len(out)} bytes (+{len(out)-len(raw)})")
        if not APPLY:
            print("\ndry run -- pass --apply to write")
            return
        digest_before = sha256(p)
        tmp.replace(p)
    finally:
        if tmp.exists():
            tmp.unlink()

    origin = {r["album_file"]: r["origin_path"]
              for r in csv.DictReader(io.open(MAN, encoding="utf-8-sig", newline=""))}
    rows, fields = read_ledger("orientation_corrections.csv")
    o = origin.get(f"{person}/{name}", "")
    rows.append({"date": TODAY, "file": str(p), "old_orientation": "", "new_orientation": value,
                 "sha256_before": digest_before, "sha256_after": sha256(p), "taken_from": "",
                 "origin": o, "origin_orientation": "",
                 "note": note(before.shape[1], before.shape[0], value)})
    with io.open(CSVP, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\n  orientation_corrections.csv: +1 row")


if __name__ == "__main__":
    main()
