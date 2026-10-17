r"""Put a thinned album copy back, from the origin that was never touched.

Every row in `album_thinned.csv` records the album path that was deleted, its sha256, and the
origin it was byte-identical to. Thinning is therefore reversible exactly, and this is the
script that does it -- restoring a file byte-for-byte and putting its `MANIFEST.csv` row back,
so `build_photographs.py` sees it again.

**Why this exists.** The keeper in a thinning was chosen on sharpness and grain measured
INSIDE the picture area, and that measure is blind to how much of the negative each print
actually holds. A tighter crop is sharper per pixel and wins, while the print it replaces may
show more of the photograph. A coverage audit over the 68 deletions made on 6-7 September
found 13 where the keeper covers less than 80% of what it deleted, down to 0.40. Density and
grain you can live with; a missing half of the frame you cannot.

Pass the deleted paths, or `--from-audit data\coverage_audit.csv` to take every row the audit
marked `KEEPER SHOWS LESS`. The thinning row is left in place and annotated rather than
removed, because the deletion is history that happened and the ledger should say so.

    .venv\Scripts\python.exe scripts\restore_thinned.py --from-audit [--apply]
    .venv\Scripts\python.exe scripts\restore_thinned.py "<full deleted_path>" [--apply]
"""
from __future__ import annotations

import csv
import hashlib
import io
import pathlib
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")

REPO = pathlib.Path(__file__).resolve().parent.parent
MAN = REPO / "data" / "MANIFEST.csv"
THINNED = REPO / "data" / "album_thinned.csv"
AUDIT = REPO / "data" / "coverage_audit.csv"
APPLY = "--apply" in sys.argv


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def read_crlf(p: pathlib.Path) -> list[str]:
    raw = p.read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n"), f"{p.name} is not uniformly CRLF"
    lines = raw.decode("utf-8-sig").split("\r\n")
    assert lines[-1] == ""
    return lines[:-1]


def write_crlf(p: pathlib.Path, body: list[str]) -> None:
    res = ("﻿" + "\r\n".join(body) + "\r\n").encode("utf-8")
    assert res.count(b"\r\n") == res.count(b"\n") == len(body)
    p.write_bytes(res)


def main() -> None:
    thinned = list(csv.DictReader(io.open(THINNED, encoding="utf-8-sig", newline="")))
    by_path = {r["deleted_path"]: r for r in thinned}

    if "--from-audit" in sys.argv:
        if not AUDIT.exists():
            raise SystemExit(f"REFUSED: {AUDIT} not found")
        want = set()
        for a in csv.DictReader(io.open(AUDIT, encoding="utf-8-sig", newline="")):
            if a["note"] == "KEEPER SHOWS LESS":
                for r in thinned:
                    if pathlib.Path(r["deleted_path"]).name == a["deleted"] \
                            and r["origin"] == a["deleted_origin"]:
                        want.add(r["deleted_path"])
        targets = sorted(want)
    else:
        targets = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not targets:
        raise SystemExit("nothing to restore -- pass paths or --from-audit")

    staged = []
    for t in targets:
        r = by_path.get(t)
        if r is None:
            raise SystemExit(f"REFUSED: {t} is not in album_thinned.csv")
        dst, src = pathlib.Path(r["deleted_path"]), pathlib.Path(r["origin"])
        if dst.exists():
            print(f"  already present, skipping: {dst.name}")
            continue
        if not src.exists():
            raise SystemExit(f"REFUSED: origin missing for {dst.name}\n  {src}")
        if sha256(src) != r["sha256"]:
            raise SystemExit(f"REFUSED: origin no longer matches the recorded sha256 for "
                             f"{dst.name} -- restoring it would not reproduce what was deleted")
        staged.append((src, dst, r))
        print(f"  restore {dst.parent.name}/{dst.name}   ({int(r['bytes'])/1e6:.1f} MB)")

    print(f"\n{len(staged)} files to restore, all verified against their recorded sha256")
    if not APPLY:
        print("dry run -- pass --apply to write")
        return

    lines = read_crlf(MAN)
    header, body = lines[0], lines[1:]
    for src, dst, r in staged:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256(dst) != r["sha256"]:
            raise SystemExit(f"REFUSED: restored {dst.name} does not match its recorded sha256")
        album = f"{dst.parent.name}/{dst.name}"
        body.append(f"{album},{dst.parent.name},copy,{r['origin']}")
        print(f"  restored {album}")
    body.sort()
    write_crlf(MAN, [header] + body)
    print(f"\n  MANIFEST.csv: +{len(staged)} rows")

    # The row must GO, not be annotated. `build_photographs.py` treats every path named in
    # album_thinned.csv as deleted and drops its manifest row, so a restored file that is
    # still listed here becomes invisible: it reads back as "filed under nobody" and its
    # manifest row as stale. Annotating instead of removing was tried and produced exactly
    # that. The history of the deletion survives in git and in data/coverage_audit.csv,
    # which is where the reason for the reversal belongs.
    fields = list(thinned[0]) if thinned else []
    restored = {str(dst) for _, dst, _ in staged}
    remaining = [r for r in thinned if r["deleted_path"] not in restored]
    assert len(thinned) - len(remaining) == len(staged), "ledger rows removed != files restored"
    with io.open(THINNED, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(remaining)
    print(f"  album_thinned.csv: {len(staged)} rows removed -- these files are no longer thinned")
    print("\nnow re-derive:  build_photographs.py")


if __name__ == "__main__":
    main()
