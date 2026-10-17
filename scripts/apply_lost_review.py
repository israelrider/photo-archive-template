"""Apply the lost-review identifications: single faces named by the family.

These are FACE-level answers, not cluster-level, and most of them sit in cluster -1
where nothing can see them. To make them count without disturbing existing clusters,
each person gets one new cluster in the 3000+ range holding their newly-named faces.
Faces already in a cluster that belongs to the right person are left alone.

<parsed.csv> has columns person,face_id[,note]: the pasted answers, one face per row.
Writes data/lost_review_log.csv, a (person, cluster) record alongside main_import_log.csv.

    .venv\\Scripts\\python.exe scripts\\apply_lost_review.py <parsed.csv> [--date=YYYY-MM-DD] [--apply]
"""
from __future__ import annotations

import csv
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import _archive as A

BASE = 3000
APPLY = "--apply" in sys.argv
# The method column dates the ANSWER, not the run: pass --date=YYYY-MM-DD for the day the
# family answered. Defaults to today.
DATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--date=")), str(date.today()))


def out_path():
    return A.REPO / "data" / "lost_review_log.csv"


def main() -> None:
    src = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    print(f"{len(rows)} identifications from {src.name}")

    named = A.named_clusters()
    conn = sqlite3.connect(A.DB)
    face = {fid: (p, c) for fid, p, c in conn.execute("select id, path, cluster_id from face")}

    # A person's 3000+ cluster is allocated ONCE and reused for ever after. The first
    # version numbered them by enumerating the answer set alphabetically, which is correct
    # exactly once: a second batch with a different set of people renumbers everybody and
    # files new faces into clusters that already belong to somebody else. Read the existing
    # allocation out of the log and only mint ids for people who have none.
    prior = list(csv.DictReader(out_path().open(encoding="utf-8-sig"))) if out_path().exists() else []
    slot = {r["person"]: int(r["cluster"]) for r in prior
            if str(r["cluster"]).lstrip("-").isdigit() and int(r["cluster"]) >= BASE}
    nxt = max(slot.values(), default=BASE - 1) + 1
    for p in sorted({r["person"] for r in rows}):
        if p not in slot:
            slot[p] = nxt
            print(f"   new cluster c{nxt} for {p}")
            nxt += 1
    people = sorted(slot)

    keep, move = 0, defaultdict(list)
    for r in rows:
        fid = int(r["face_id"])
        if fid not in face:
            print("  missing face", fid); continue
        _, cur = face[fid]
        if cur is not None and named.get(cur) == r["person"]:
            keep += 1                      # already correctly attributed
        else:
            move[r["person"]].append(fid)

    n_move = sum(len(v) for v in move.values())
    print(f"already attributed correctly : {keep}")
    print(f"to place in a new cluster    : {n_move}")
    print()
    for p in sorted(move, key=lambda x: -len(move[x])):
        was = Counter(face[f][1] for f in move[p])
        noise = was.get(-1, 0)
        print(f"   c{slot[p]:<5} {p:<30} {len(move[p]):>3} faces  ({noise} were unclustered)")

    if not APPLY:
        print("\nDRY RUN. Re-run with --apply to write.")
        return

    bak = A.DB.with_suffix(f".sqlite.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(A.DB, bak)
    print(f"\nbacked up -> {bak.name}")

    log_rows = []
    for r in rows:
        fid = int(r["face_id"])
        if fid not in face:
            continue
        person = r["person"]
        if fid in move.get(person, []):
            cid = slot[person]
            conn.execute("update face set cluster_id=? where id=?", (cid, fid))
        else:
            cid = face[fid][1]
        log_rows.append({"person": person, "cluster": cid, "face_id": fid,
                         "source": face[fid][0], "note": r.get("note", ""),
                         "method": f"family-identification:{DATE}"})
    conn.commit()

    # Append, never replace. Rewriting the file from the batch in hand is invisible until
    # the second batch, which would delete every answer already there.
    out = out_path()
    # A row may legitimately carry no face_id: a person lying with her face in her hand,
    # whom the detector never found, is still named by the family. Keying the dedup on int() alone made
    # any later run die on that row, after the database had already been committed.
    fresh = {int(x["face_id"]) for x in log_rows}
    keep = [r for r in prior
            if not str(r["face_id"]).strip().isdigit() or int(r["face_id"]) not in fresh]
    allr = keep + log_rows
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["person", "cluster", "face_id", "source", "note", "method"])
        w.writeheader(); w.writerows(allr)
    print(f"wrote {len(allr)} rows -> {out.name} ({len(keep)} kept, {len(log_rows)} from this batch)")
    print("faces  :", conn.execute("select count(*) from face").fetchone()[0])
    print("noise  :", conn.execute("select count(*) from face where cluster_id=-1").fetchone()[0])
    print("clusters >=3000:", conn.execute(
        "select count(distinct cluster_id) from face where cluster_id>=3000").fetchone()[0])
    conn.close()


if __name__ == "__main__":
    main()
