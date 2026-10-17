"""Repoint face rows that cite a by-person COPY to the origin it was copied from.

A face indexed from an album copy is filed against a file that thinning may delete. The
origin is the authority: every row is moved to the byte-identical origin, or nothing is.
Also refreshes mtime_ns/size so path_already_indexed() sees the new path as unchanged
and a future re-index won't re-detect these photos.
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

from _paths import BP, DB, ORIGIN_ROOTS

APPLY = "--apply" in sys.argv
by_name = {f.name: f for root in ORIGIN_ROOTS for f in root.rglob("*") if f.is_file()}
print(f"origin roots hold {len(by_name)} files")

conn = sqlite3.connect(DB)
rows = conn.execute(
    "select path, count(*) from face where path like ? group by path", (f"{BP}%",)
).fetchall()
print(f"index rows citing by-person: {len(rows)} paths, {sum(n for _, n in rows)} faces\n")


def sha(p: Path) -> bytes:
    return hashlib.sha256(p.read_bytes()).digest()


plan, problems = [], []
for old, nfaces in rows:
    src = Path(old)
    dst = by_name.get(src.name)
    if dst is None:
        problems.append(f"no origin twin: {src.name}")
    elif not src.exists():
        problems.append(f"album copy gone: {old}")
    elif sha(src) != sha(dst):
        problems.append(f"CONTENT DIFFERS: {src.name}")
    else:
        plan.append((old, dst, nfaces))

if problems:
    print("ABORT - nothing changed:")
    for p in problems:
        print("   ", p)
    raise SystemExit(1)
if not APPLY:
    print(f"{len(plan)} paths would be repointed. DRY RUN - pass --apply to write.")
    raise SystemExit(0)

for old, dst, nfaces in plan:
    st = dst.stat()
    conn.execute(
        "update face set path=?, mtime_ns=?, size=? where path=?",
        (str(dst.resolve()), st.st_mtime_ns, st.st_size, old),
    )
conn.commit()
print(f"repointed {len(plan)} paths ({sum(n for _, _, n in plan)} face rows)")

left = conn.execute("select count(*) from face where path like ?", (f"{BP}%",)).fetchone()[0]
print(f"\nby-person rows remaining : {left}")
print("total faces :", conn.execute("select count(*) from face").fetchone()[0])
print("total photos:", conn.execute("select count(distinct path) from face").fetchone()[0])
conn.close()
