"""Record the rematch-review answers: whole clusters named by the family.

These are CLUSTER-level answers, unlike lost-review's face-level ones, so nothing in the
database changes — the clusters already exist and already hold the right faces. All that
is missing is the name. This writes a (person, cluster) record alongside
main_import_log.csv and lost_review_log.csv.

The score is recorded next to every answer, including where the family disagreed with the
embedding, so a later reader can see which identifications rest on the picture and which
rest on the family's word.

    .venv\\Scripts\\python.exe scripts\\apply_rematch_review.py [--review=rematch|borderline] [--date=YYYY-MM-DD] [--apply]
"""
from __future__ import annotations

import csv
import json
from datetime import date
import sqlite3
import sys
from collections import defaultdict

import numpy as np

import _archive as A

DATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--date=")), str(date.today()))
APPLY = "--apply" in sys.argv
REVIEW = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--review=")), "rematch")
SET = A.REVIEW_SETS[REVIEW]
SRC = A.REPO / "data" / SET["answers"]
OUT = A.REPO / "data" / SET["log"]

# The family answers with a description, not a folder name. Keep the raw text as the note.
ALIAS = A.ALIASES
# People who keep a personal album instead of a by-person folder ([people] in archive.toml).
PERSONAL = A.ELSEWHERE


def key_bbox(bb) -> str:
    return ";".join(str(int(v)) for v in bb)


def main() -> None:
    answers = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cid, raw = line.split("\t")
        answers.append((int(cid.strip().lstrip("c")), raw.strip()))
    print(f"{len(answers)} answers from {SRC.name}")

    # Rows already recorded here, kept verbatim. The log is rebuilt from the whole answer
    # file each run, and named_clusters() reads this log - without this, every cluster
    # recorded on an earlier run would look like a conflict with itself and be dropped.
    # Keeping the old rows also keeps their score: recomputing it against centroids that
    # now include the cluster itself would only flatter the earlier answer.
    prior_rows, prior_who = defaultdict(list), {}
    if OUT.exists():
        for r in csv.DictReader(OUT.open(encoding="utf-8-sig")):
            prior_rows[int(r["cluster"])].append(r)
            prior_who[int(r["cluster"])] = r["person"]

    named = A.named_clusters()
    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)
    folders = {p.name for p in A.BP.iterdir() if p.is_dir()}

    rows, problems, kept = [], [], 0
    for c, raw in answers:
        person = ALIAS.get(raw, raw)
        idx = np.where(cl == c)[0]
        if len(idx) == 0:
            problems.append(f"c{c}: no faces in the index"); continue
        if prior_who.get(c) == person:
            rows.extend(prior_rows[c]); kept += 1
            continue
        confirms = ""
        if c in prior_who:
            print(f"   c{c}: re-answered {prior_who[c]!r} -> {person!r}")
        elif named.get(c) == person:
            # Another review already got this one. Two reviews reaching the same answer
            # independently is worth more than either alone - record it, do not drop it.
            confirms = "confirms an earlier review"
        elif c in named:
            problems.append(f"c{c}: already named {named[c]!r}"); continue
        if person not in folders and person not in PERSONAL:
            problems.append(f"c{c}: {person!r} is neither a by-person folder nor a personal album")
            continue
        s = P @ A.cluster_centroid(cl, M, c)
        order = np.argsort(-s)
        rank = next((r for r, k in enumerate(order, 1) if people[k] == person), None)
        score = float(s[people.index(person)]) if person in people else float("nan")
        best = people[order[0]]
        note = "; ".join(x for x in (raw if raw != person else "", confirms) if x)
        if rank != 1:
            note = (note + "; " if note else "") + f"embedding preferred {best} {s[order[0]]:.3f}"
        for i in idx:
            rows.append({"person": person, "cluster": c, "face_id": "",
                         "source": paths[i], "bbox": key_bbox(bbs[i]),
                         "score": f"{score:.3f}", "rank": rank or "",
                         "note": note,
                         "method": f"family-identification:{DATE} ({REVIEW}-review)"})
        flag = "" if rank == 1 else f"   <- family over embedding ({best} {s[order[0]]:.3f})"
        print(f"   c{c:<5} {person:<28} {len(idx):>2} faces  "
              f"{len({paths[i] for i in idx})} photos  {score:.3f} rank {rank}{flag}")

    if problems:
        print("\nnot recorded:")
        for p in problems:
            print("   " + p)

    # face_id is what the distributor keys on (for the age tag); fill it from the database,
    # matching on (path, bbox) because load_faces() does not carry ids.
    conn = sqlite3.connect(A.DB)
    fid = {(p, key_bbox(json.loads(b))): i for i, p, b in conn.execute(
        "select id, path, bbox from face where model=?", (A.MODEL,))}
    conn.close()
    missing = 0
    for r in rows:
        f = fid.get((r["source"], r["bbox"]))
        if f is None:
            missing += 1
        else:
            r["face_id"] = f
    if missing:
        print(f"\nWARNING: {missing} rows could not be keyed back to a face id")

    print(f"\n{len(rows)} face rows for {len({int(r['cluster']) for r in rows})} clusters "
          f"({kept} carried over unchanged), {len({r['person'] for r in rows})} people")
    if not APPLY:
        print("\nDRY RUN. Re-run with --apply to write.")
        return

    cols = ["person", "cluster", "face_id", "source", "bbox", "score", "rank", "note", "method"]
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT.relative_to(A.REPO)}")


if __name__ == "__main__":
    main()
