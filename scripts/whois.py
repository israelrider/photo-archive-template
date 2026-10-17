r"""Reverse lookup: which named by-person folders hold the photos of these clusters?

    .venv\Scripts\python.exe scripts\whois.py 2 4 21
"""
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from _archive import origin_name
from _paths import BP as BY_PERSON, DB, MODEL

conn = sqlite3.connect(DB)

# basename -> set of folders that contain it
owner = {}
for d in BY_PERSON.iterdir():
    if not d.is_dir():
        continue
    for f in d.rglob("*.jp*g"):
        owner.setdefault(origin_name(f.name), set()).add(d.name)

for cl in (int(a) for a in sys.argv[1:]):
    rows = conn.execute(
        "select path, age, sex from face where cluster_id=? and model=?", (cl, MODEL)
    ).fetchall()
    if not rows:
        print(f"\n=== cluster {cl}: no faces")
        continue
    ages = sorted(a for _, a, _ in rows if a and a > 0)
    folders = Counter()
    for p, _, _ in rows:
        for name in owner.get(Path(p).name.lower(), ()):
            folders[name] += 1
    sexes = Counter(s for _, _, s in rows)
    print(f"\n=== cluster {cl}: {len(rows)} faces, {len(set(p for p,_,_ in rows))} photos")
    print(f"    age range {ages[0]:.0f}-{ages[-1]:.0f} (median {ages[len(ages)//2]:.0f})  sex {dict(sexes)}")
    print(f"    appears in folders: {folders.most_common(6)}")
