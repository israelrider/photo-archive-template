"""Close one page of the unknown review: record the folders examined and not recognised.

`пропустить` on a card is an answer - "I looked, I do not know them" - and it lives only in
the browser's localStorage until somebody writes it down. This records the folders on a
given page as reviewed-unidentified so the next build stops asking.

Deliberately per page. The unknown review runs over several pages and the family works
through it a page at a time; closing all of it because one page came back would
record decisions nobody made.

    .venv\\Scripts\\python.exe scripts\\close_unknown_review.py 06 [--date=YYYY-MM-DD] [--apply]
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import _archive as A

# The day the family answered, not the day this runs.
DATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--date=")), str(date.today()))
APPLY = "--apply" in sys.argv
OUT = A.REPO / "data" / "unknown_declined.csv"


def main() -> None:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not pos:
        raise SystemExit("usage: close_unknown_review.py <page number> [--apply]")
    page = A.REVIEWS / f"unknown-review-{int(pos[0]):02d}.html"
    if not page.exists():
        raise SystemExit(f"{page.name} not found")
    ids = list(dict.fromkeys(re.findall(r"data-id='(Person_\d+)'", page.read_text(encoding="utf-8"))))

    named = A.named_clusters()
    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)
    # The folder's clusters: from the import log if it says, else the folder name itself
    # (images-agent names a cluster's folder Person_<cluster>).
    folder_cl = defaultdict(set)
    for pid in ids:
        m = re.fullmatch(r"Person_(\d+)", pid)
        if m:
            folder_cl[pid].add(int(m.group(1)))
    log = A.LOGS / "main_import_log.csv"
    for r in (csv.DictReader(log.open(encoding="utf-8-sig")) if log.exists() else []):
        if r["person"].startswith("Person_") and r["cluster"].lstrip("-").isdigit():
            folder_cl[r["person"]].add(int(r["cluster"]))

    prior = {r["folder"]: r for r in csv.DictReader(OUT.open(encoding="utf-8-sig"))} \
        if OUT.exists() else {}
    rows = list(prior.values())
    for pid in ids:
        if pid in prior:
            print(f"   {pid}: already recorded"); continue
        cls = sorted(folder_cl.get(pid, ()))
        # A folder whose clusters got named elsewhere is answered, not declined.
        if any(c in named for c in cls):
            print(f"   {pid}: named elsewhere ({[named[c] for c in cls if c in named]}), skipping")
            continue
        idx = [i for c in cls for i in np.where(cl == c)[0]]
        if not idx:
            print(f"   {pid}: no faces in the index"); continue
        v = M[idx].mean(axis=0)
        v /= np.linalg.norm(v)
        s = P @ v
        o = np.argsort(-s)
        rows.append({"folder": pid, "clusters": " ".join(f"c{c}" for c in cls),
                     "faces": len(idx), "photos": len({paths[i] for i in idx}),
                     "best_candidate": people[o[0]], "score": f"{float(s[o[0]]):.3f}",
                     "shown_on": page.name, "method": f"reviewed-unidentified:{DATE}"})
        print(f"   {pid}  {' '.join(f'c{c}' for c in cls):<10} {len(idx)} faces  "
              f"{rows[-1]['photos']} photos  rejected {people[o[0]]} {float(s[o[0]]):.3f}")

    print(f"\n{len(ids)} card(s) on {page.name} · {len(rows)} folder(s) recorded in total")
    if not APPLY:
        print("\nDRY RUN. Re-run with --apply to write.")
        return
    if not rows:
        print("nothing to record")
        return
    rows.sort(key=lambda r: r["folder"])
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT.relative_to(A.REPO)}")


if __name__ == "__main__":
    main()
