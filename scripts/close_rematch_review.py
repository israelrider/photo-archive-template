"""Close the rematch review: record what was shown and left unnamed.

When the family has worked through the pages, the clusters they did not name are not a
backlog. They were looked at, as whole photographs, by the people
who would know - and nobody recognised them. That is a finding, and it has to be written
down, or the next run of make_rematch_review.py puts the same faces in front of the same
people and calls it progress.

Same convention as the split review (`reviewed-unidentified:<date>` in split_import_log.csv),
kept in its own file because rematch_log.csv is face-level and feeds the distributor.

What was shown is read from the review pages themselves rather than recomputed: the pages
are the record of the question actually asked, and the candidate list has since changed.

    .venv\\Scripts\\python.exe scripts\\close_rematch_review.py [--review=rematch|borderline]
        [--date=YYYY-MM-DD] [--also=c12,c34] [--note=c12:text] [--apply]
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import date

import numpy as np

import _archive as A

# The day the family answered, not the day this runs.
DATE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--date=")), str(date.today()))
APPLY = "--apply" in sys.argv
REVIEW = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--review=")), "rematch")
SET = A.REVIEW_SETS[REVIEW]
OUT = A.REPO / "data" / SET["declined"]


def shown() -> dict[int, str]:
    """cluster -> the review page it appeared on."""
    out = {}
    for page in sorted(A.REVIEWS.glob(SET["pages"])):
        for m in re.finditer(r"data-id='c(\d+)'", page.read_text(encoding="utf-8")):
            out.setdefault(int(m.group(1)), page.name)
    return out


def main() -> None:
    cards = shown()
    # Clusters closed by a follow-up card rather than by a numbered page - asked separately
    # and answered "не знаю". They are not in the pages, so without this they
    # would vanish on the next run and be put back in front of the family.
    also = {int(x) for a in sys.argv[1:] if a.startswith("--also=")
            for x in a.split("=", 1)[1].replace("c", "").split(",")}
    notes = {int(a.split("=", 1)[1].split(":", 1)[0].lstrip("c")):
             a.split("=", 1)[1].split(":", 1)[1]
             for a in sys.argv[1:] if a.startswith("--note=")}
    also |= set(notes)
    prior = {int(r["cluster"]): r for r in csv.DictReader(OUT.open(encoding="utf-8-sig"))} \
        if OUT.exists() else {}
    # Named ANYWHERE, not just in this review's own log: a card already settled by another
    # review's pages must not be marked them unidentified here would contradict
    # an answer the family has already given.
    named = A.named_clusters()
    answered = set(named) | {int(r["cluster"]) for r in
                             csv.DictReader((A.REPO / "data" / SET["log"]).open(encoding="utf-8-sig"))}
    declined = sorted((set(cards) | set(prior) | also) - answered)
    print(f"{len(cards)} clusters shown · {len(answered & set(cards))} named · "
          f"{len(declined)} left unnamed ({len(set(declined) - set(cards))} from follow-up cards)")

    cl, paths, bbs, dets, M = A.load_faces()
    people, P = A.person_centroids(cl, M, named)

    rows = []
    for c in declined:
        idx = np.where(cl == c)[0]
        s = P @ A.cluster_centroid(cl, M, c)
        o = np.argsort(-s)
        rows.append({"cluster": c, "faces": len(idx), "photos": len({paths[i] for i in idx}),
                     "best_candidate": people[o[0]], "score": f"{float(s[o[0]]):.3f}",
                     "shown_on": cards.get(c) or (prior[c]["shown_on"] if c in prior
                                                  else f"c{c}-check.html"),
                     "method": f"reviewed-unidentified:{DATE}",
                     # A cluster that was answered and then reversed is not the same as one
                     # nobody ever recognised, and the difference is worth a sentence.
                     "note": notes.get(c, prior.get(c, {}).get("note", ""))})
        print(f"   c{c:<5} {len(idx):>2} faces  {rows[-1]['photos']} photos  "
              f"rejected {people[o[0]]} {float(s[o[0]]):.3f}")

    if not APPLY:
        print("\nDRY RUN. Re-run with --apply to write.")
        return
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT.relative_to(A.REPO)}")


if __name__ == "__main__":
    main()
