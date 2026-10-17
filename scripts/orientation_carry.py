r"""Carry a recorded orientation correction onto a fresh copy of the same photograph.

THE DEFECT THIS CLOSES

Orientation is corrected album-side; origins are never modified, by standing policy. But
the distributors copy FROM the origin, so when a later run files an already-corrected
photograph to another person it writes a fresh copy carrying the origin's wrong tag, and
the correction is silently gone for that copy.

This is not hypothetical: a photograph corrected one day is sideways again days later, in
a folder the next distribution wrote. Fixing the files again does not fix the mechanism,
which undoes the next run just as reliably. This module is the mechanism -- every
distributor should pass each fresh copy through it.

WHY THE KEY IS A HASH AND NOT A PATH

A photograph reaches an album under a name built from its age, root letter and origin
filename, and the same origin lands under different names for different people -- so a
correction cannot be matched to a future copy by name. What does not change is the bytes
being copied. `orientation_corrections.csv` records `sha256_before`: the copy exactly as it
came off the origin, before the tag was touched. That is the origin's own sha256 whenever
the copy was byte-identical to it, which is what every distributor guarantees. So the key
is the origin's hash and the value is the tag the review settled on.

The log was checked before this was written, and the mapping is a function: 353 rows, 334
distinct `sha256_before`, and NO hash carrying two different answers -- the 19 repeats are
second copies of one photograph corrected the same way in two albums. No row's
`sha256_before` equals another's `sha256_after`, so no correction was ever corrected again
and there is no chain to follow. Should either ever stop being true, `load()` raises rather
than guessing which answer to carry.

WHAT A CARRIED COPY IS

The copy is written byte-identical to its origin and verified that way first; the tag is
then set, and the decoded pixels are compared exactly before and after. So the file is the
origin's image with the answer about which way up it goes, and it is deliberately no longer
byte-identical to the origin -- the same recorded trade every one of the 353 corrections
already made. Callers must say so rather than print that every copy matches its origin,
which is why `straighten()` returns what it did instead of staying quiet.
"""
from __future__ import annotations

import csv
import hashlib
import io
import pathlib
import struct

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = pathlib.Path(__file__).resolve().parent.parent
CSVP = REPO / "data" / "orientation_corrections.csv"

ROTATION_ONLY = {1, 3, 6, 8}


def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(path: pathlib.Path | None = None) -> dict[str, int]:
    """{sha256 of the uncorrected bytes: the Orientation the review settled on}.

    Raises if the log stops being a function -- one hash with two answers, or a correction
    that was itself later corrected. Either would make "which tag does this copy want?"
    ambiguous, and a distributor must not answer that by picking one.
    """
    p = path or CSVP
    if not p.exists():
        return {}                     # a new archive: nothing has been corrected yet
    rows = list(csv.DictReader(io.open(p, encoding="utf-8-sig", newline="")))
    out: dict[str, int] = {}
    afters: dict[str, str] = {}
    for r in rows:
        b, a = r["sha256_before"], r["sha256_after"]
        if not b or not a:
            raise SystemExit(f"orientation_corrections.csv: row for {r['file']} has no sha256")
        new = int(r["new_orientation"])
        if new not in ROTATION_ONLY:
            raise SystemExit(f"orientation_corrections.csv: {r['file']} carries a mirrored "
                             f"or unknown Orientation {new}; this module handles 1/3/6/8 only")
        if b in out and out[b] != new:
            raise SystemExit(f"orientation_corrections.csv: {b[:12]} is recorded as wanting "
                             f"both {out[b]} and {new}; resolve it before distributing")
        out[b] = new
        afters[a] = r["file"]
    chained = sorted(set(out) & set(afters))
    if chained:
        raise SystemExit(f"orientation_corrections.csv: {len(chained)} correction(s) were "
                         f"themselves corrected later, e.g. {afters[chained[0]]}; this module "
                         "does not follow chains")
    return out


def _orientation(buf: bytes) -> tuple[int | None, int | None, str | None]:
    """-> (value, absolute offset of the 2-byte value, byte order) for IFD0 Orientation."""
    i, n = 2, len(buf)
    while i + 4 <= n and buf[i] == 0xFF:
        marker = buf[i + 1]
        if marker == 0xDA:
            break
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        ln = struct.unpack(">H", buf[i + 2:i + 4])[0]
        p0 = i + 4
        if marker == 0xE1 and buf[p0:p0 + 6] == b"Exif\x00\x00":
            tiff = p0 + 6
            bo = buf[tiff:tiff + 2]
            if bo in (b"II", b"MM"):
                e = "<" if bo == b"II" else ">"
                ifd0 = tiff + struct.unpack(e + "I", buf[tiff + 4:tiff + 8])[0]
                for k in range(struct.unpack(e + "H", buf[ifd0:ifd0 + 2])[0]):
                    ent = ifd0 + 2 + k * 12
                    if struct.unpack(e + "H", buf[ent:ent + 2])[0] == 0x0112:
                        return struct.unpack(e + "H", buf[ent + 8:ent + 10])[0], ent + 8, e
        i = i + 2 + ln
    return None, None, None


def read_orientation(p: pathlib.Path) -> int | None:
    return _orientation(p.read_bytes())[0]


def _minimal_app1(value: int) -> bytes:
    """An Exif APP1 holding IFD0 with exactly one entry: Orientation."""
    payload = (b"Exif\x00\x00" + b"II" + struct.pack("<HI", 42, 8)
               + struct.pack("<H", 1)
               + struct.pack("<HHIHH", 0x0112, 3, 1, value, 0)
               + struct.pack("<I", 0))
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def straighten(dst: pathlib.Path, tag: int) -> str | None:
    """Set dst's Orientation to tag without touching a pixel. -> what changed, or None.

    Returns None when the file already carries the tag, so a re-run is quiet and idempotent.
    Any failure restores the original bytes and re-raises: a half-written album copy is
    worse than a sideways one.
    """
    buf = dst.read_bytes()
    cur, off, e = _orientation(buf)
    if cur == tag:
        return None
    px = np.array(Image.open(io.BytesIO(buf)).convert("RGB"))
    if off is not None:
        out = bytearray(buf)
        out[off:off + 2] = struct.pack(e + "H", tag)
        out, how = bytes(out), f"{cur} -> {tag}, two bytes inside the APP1 already present"
    else:
        out = buf[:2] + _minimal_app1(tag) + buf[2:]
        how = f"none -> {tag}, a 36-byte APP1 inserted after the APP0"
    dst.write_bytes(out)
    try:
        if not np.array_equal(np.array(Image.open(dst).convert("RGB")), px):
            raise SystemExit(f"straighten({dst}): pixels changed -- reverted")
        if read_orientation(dst) != tag:
            raise SystemExit(f"straighten({dst}): tag did not take -- reverted")
    except BaseException:
        dst.write_bytes(buf)
        raise
    return how
