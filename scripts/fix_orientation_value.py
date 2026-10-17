r"""Rewrite a wrong EXIF Orientation value in place, changing nothing else.

The usual case in a scanned archive is a portrait-stored file carrying orientation 6
correctly, because the photograph inside it is landscape. This handles the opposite: a
photograph stored upright but TAGGED 6 (or 8), so every EXIF-aware viewer lays it on its
side. Judge each file against its own picture, never against its batch: in one scanning
batch some files carry 6 correctly and their neighbours carry it wrongly.

The value is rewritten in the existing APP1 segment: two bytes change, the file length is
unchanged, and every other byte is copied through. Pixels are decoded before and after and
compared exactly; the file is rejected if they differ by a single value. A file with NO
EXIF block cannot be fixed this way -- use add_orientation_tag.py.

Origins are never modified; only the album copy is, and the change is recorded in
data\orientation_corrections.csv.

    .venv\Scripts\python.exe scripts\fix_orientation_value.py <person> <file> <1|3|6|8>
        [<person> <file> <value> ...] [--note="why"] [--apply]
"""
import csv, datetime, io, os, sys, hashlib
import numpy as np
from PIL import Image

from _paths import BP, LOGS, read_ledger

Image.MAX_IMAGE_PIXELS = None
sys.stdout.reconfigure(encoding="utf-8")

APPLY = "--apply" in sys.argv
CSVP = LOGS / "orientation_corrections.csv"
ORIENT = 0x0112
TODAY = datetime.date.today().isoformat()
NOTE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--note=")),
            "stored upright but tagged with a rotating orientation, so every EXIF-aware "
            "viewer laid it on its side. Two bytes rewritten inside the existing APP1; file "
            "length unchanged, pixels verified identical before and after. Origin NOT modified.")

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not _args or len(_args) % 3:
    raise SystemExit(__doc__)
FIX = {f"{_args[i]}/{_args[i + 1]}": int(_args[i + 2]) for i in range(0, len(_args), 3)}
if any(v not in (1, 3, 6, 8) for v in FIX.values()):
    raise SystemExit("orientation must be 1, 3, 6 or 8")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def orientation_value_offset(raw: bytes):
    """Absolute file offset of the 2-byte Orientation SHORT value, or None."""
    if raw[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    while i < len(raw) - 1:
        if raw[i] != 0xFF:
            return None
        marker = raw[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:          # start of scan; no EXIF beyond here
            return None
        seglen = int.from_bytes(raw[i + 2:i + 4], "big")
        seg = raw[i + 4:i + 2 + seglen]
        if marker == 0xE1 and seg[:6] == b"Exif\x00\x00":
            tiff_at = i + 4 + 6
            bo = raw[tiff_at:tiff_at + 2]
            end = "little" if bo == b"II" else "big"
            if bo not in (b"II", b"MM"):
                return None
            ifd0 = tiff_at + int.from_bytes(raw[tiff_at + 4:tiff_at + 8], end)
            n = int.from_bytes(raw[ifd0:ifd0 + 2], end)
            for k in range(n):
                e = ifd0 + 2 + 12 * k
                tag = int.from_bytes(raw[e:e + 2], end)
                typ = int.from_bytes(raw[e + 2:e + 4], end)
                cnt = int.from_bytes(raw[e + 4:e + 8], end)
                if tag == ORIENT:
                    if typ != 3 or cnt != 1:
                        return None
                    # SHORT, count 1 -> value sits in the first 2 bytes of the value field
                    return (e + 8, end)
            return None
        i += 2 + seglen
    return None




rows, fields = read_ledger("orientation_corrections.csv")
done = {r["file"] for r in rows}
manifest = {r["album_file"].replace("\\", "/"): r
            for r in csv.DictReader(io.open(LOGS / "MANIFEST.csv", encoding="utf-8-sig", newline=""))}

staged = []
for rel, want in FIX.items():
    p = BP / rel.replace("/", os.sep)
    print(f"\n{rel}")
    if not p.exists():
        print("   MISSING -- skipped"); continue
    if str(p) in done:
        print("   already recorded -- skipped"); continue

    before = Image.open(p)
    old = before.getexif().get(ORIENT)
    px_before = np.asarray(before.convert("L"))
    if old == want:
        print(f"   already {want} -- nothing to do"); continue

    raw = p.read_bytes()
    loc = orientation_value_offset(raw)
    if loc is None:
        print("   no in-place Orientation SHORT found -- skipped"); continue
    off, end = loc
    patched = bytearray(raw)
    patched[off:off + 2] = want.to_bytes(2, end)
    patched = bytes(patched)

    chk = Image.open(io.BytesIO(patched))
    got = chk.getexif().get(ORIENT)
    px_after = np.asarray(chk.convert("L"))
    same = px_before.shape == px_after.shape and np.array_equal(px_before, px_after)
    diff = sum(1 for a, b in zip(raw, patched) if a != b)
    print(f"   orientation {old} -> {got}   length {len(raw)} -> {len(patched)}   "
          f"bytes changed: {diff}   pixels identical: {same}")
    if got != want or not same or len(raw) != len(patched) or diff > 2:
        print("   REFUSED: the patch changed more than the tag"); continue

    origin = manifest.get(rel, {}).get("origin_path", "")
    oo = ""
    if origin and os.path.exists(origin):
        oo = Image.open(origin).getexif().get(ORIENT) or ""
    staged.append((p, patched, sha(p), old, want, origin, oo))

if not staged:
    print("\nnothing to do"); raise SystemExit
if not APPLY:
    print(f"\n{len(staged)} files ready -- pass --apply to write"); raise SystemExit

for p, patched, before_sha, old, want, origin, oo in staged:
    p.write_bytes(patched)
    assert Image.open(p).getexif().get(ORIENT) == want, p
    rows.append({"date": TODAY, "file": str(p),
                 "old_orientation": "" if old is None else old, "new_orientation": want,
                 "sha256_before": before_sha, "sha256_after": sha(p),
                 "taken_from": "", "origin": origin, "origin_orientation": oo, "note": NOTE})
    print(f"   wrote {p.relative_to(BP)}")

with io.open(CSVP, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {CSVP}  (+{len(staged)} rows)")
