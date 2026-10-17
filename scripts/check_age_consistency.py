r"""Find mislabelled ages by making the archive check itself: people age together.

The age in every album filename is a guess. Two models produce those guesses -- buffalo
and MiVOLO -- and on old scans they routinely disagree by more than 15 years, often by 30.
Wherever the model the filename took was the wrong one, the error went into the filename.
Neither model is reliably right: on two frames of the same two people taken minutes apart,
MiVOLO can be steady for one person and noise for the other, and buffalo the exact reverse.

So a check that asks "is this age plausible?" has nothing to appeal to. This one asks a
question the data can answer on its own.

If two photographs share several people, then whatever the gap between the two photographs
is -- a day, thirty years -- EVERY shared person's age must differ by about that same gap.
People do not age at different rates. So take the deltas across the shared people, and the
consensus (median) delta is the gap. Anyone whose delta departs from that consensus is
mislabelled in one of the two photographs. The check needs no ground truth, no dates, and
no trust in either model.

The motivating case: two people, A and B, in two frames taken minutes apart. A reads 33
then 34 (delta +1), B reads 31 then 62 (delta +31). One of B's two labels is wrong -- the
blurred frame, where MiVOLO lost the ageing cues.

WHICH of the two labels is wrong is a separate question, and the deltas cannot answer it:
the pair only says they disagree. So for every flagged face the script pulls buffalo and
MiVOLO back out of `data/renamed_ages.csv` (person, file, buffalo, mivolo) when it
exists. The label to distrust is the one where the two
models were far apart; the label to keep is the one where they agreed. That is corroboration
from an independent measurement, not a third guess.

Two shared people is the weak case. With only two deltas the median sits exactly between
them, so both people score the same residual and a naive report accuses both -- it flags
A's steady 33/34 as hard as B's 31/62. So a
two-person pair produces ONE finding, not two, and the model evidence picks which of the
two faces to distrust; the other is recorded as the alternative. Three or more shared
people gives a real consensus and the outlier is named outright.

The strongest findings need no consensus at all: one photograph, one person, two different
ages, because two scans of the same negative were labelled separately. That is a flat
contradiction rather than an inference, so those are emitted too, as `same-negative`.

Nothing here renames anything. It writes `data/age_consistency_machine.csv` and says which
faces deserve a look.

    .venv\Scripts\python.exe scripts\check_age_consistency.py [--manifest PATH]
                                                                [--min-residual N] [--top N]
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

import _archive as A

sys.stdout.reconfigure(encoding="utf-8")

REPO = A.REPO
OUT = REPO / "data" / "age_consistency_machine.csv"
AGE_RE = re.compile(r"^age(\d+)_")

# Below this, a delta gap is ordinary model jitter rather than a defect. Buffalo and MiVOLO
# routinely sit ~10 years apart on the same face; 15 is outside that and still well inside
# the 30-year errors the check exists to catch.
MIN_RESIDUAL = 15


def rows(name: str) -> list[dict]:
    p = REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))) if p.exists() else []


def arg(flag: str, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def load_album(manifest: Path) -> list[dict]:
    """One record per (person, album file) that carries a parsable age."""
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        out = []
        for r in csv.DictReader(f):
            person, _, fname = r["album_file"].partition("/")
            m = AGE_RE.match(fname)
            if m:
                out.append({"person": person, "file": fname, "age": int(m.group(1)),
                            "origin": r["origin_path"]})
        return out


def photo_key(origin: str, fname: str, neg_of: dict[str, str]) -> str:
    """Identify the PHOTOGRAPH, so two scans of one negative are not read as two photos."""
    if origin and origin in neg_of:
        return neg_of[origin]
    return origin or fname


def model_scores() -> dict[tuple[str, str], tuple[float, float]]:
    """(person, current filename) -> (buffalo, mivolo), from the rename log."""
    out = {}
    for r in rows("renamed_ages.csv"):
        try:
            out[(r["person"], r["new_name"])] = (float(r["buffalo_age"]), float(r["mivolo_age"]))
        except (ValueError, KeyError):
            pass
    return out


def model_gap(rec: dict, scores: dict) -> float | None:
    """How far buffalo and MiVOLO sat apart on this one face. None if it was never scored.

    This is the only independent evidence available: a label both models landed on is
    corroborated, one they split over is a coin flip that the rename settled by always
    taking MiVOLO.
    """
    g = scores.get((rec["person"], rec["file"]))
    return None if g is None else abs(g[0] - g[1])


def describe(rec: dict, scores: dict) -> str:
    g = scores.get((rec["person"], rec["file"]))
    return "unscored" if g is None else f"buffalo/mivolo {g[0]:.0f}/{g[1]:.1f}"


def arbitrate(cands: list[dict], scores: dict) -> tuple[dict, str]:
    """Of several labels that cannot all be right, name the one to distrust.

    Ranked by how far the two models sat apart on that face. An unscored face cannot be
    ranked, so it never wins the accusation by default -- it is reported as ambiguous.
    """
    gaps = [(model_gap(c, scores), c) for c in cands]
    scored = [(g, c) for g, c in gaps if g is not None]
    detail = "; ".join(f"{c['age']} ({describe(c, scores)})" for c in cands)
    if not scored or max(g for g, _ in scored) < MIN_RESIDUAL:
        return {}, f"{detail} -- no model evidence to separate them"
    gap, worst = max(scored, key=lambda t: t[0])
    return worst, f"{detail} -- distrust age{worst['age']} (models {gap:.0f}yr apart)"


def main() -> None:
    manifest = Path(arg("--manifest", str(REPO / "data" / "MANIFEST.csv")))
    min_res = arg("--min-residual", MIN_RESIDUAL)
    top = arg("--top", 25)

    neg_of = {r["origin_path"]: r["negative"] for r in rows("photographs.csv") if r.get("origin_path")}
    album = load_album(manifest)
    print(f"{len(album)} dated album files from {manifest.name}, {len(neg_of)} scans mapped to negatives")

    # photo -> {person: record}. A person twice in one photo means two scans of one negative
    # disagreeing with itself; that is its own finding, collected as we go.
    photo: dict[str, dict[str, dict]] = defaultdict(dict)
    same_neg: list[tuple] = []
    for r in album:
        k = photo_key(r["origin"], r["file"], neg_of)
        seen = photo[k].get(r["person"])
        if seen is None:
            photo[k][r["person"]] = r
        else:
            if seen["age"] != r["age"]:
                same_neg.append((k, r["person"], seen, r))
            if r["age"] > seen["age"]:          # keep one, deterministically
                photo[k][r["person"]] = r

    multi = {k: v for k, v in photo.items() if len(v) >= 2}
    print(f"{len(photo)} distinct photographs, {len(multi)} with 2+ named people")

    # Candidate photo pairs: any two photographs that share at least two people. Indexing by
    # person-PAIR enumerates exactly those, instead of walking every pair of photographs.
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k, people in multi.items():
        for pq in combinations(sorted(people), 2):
            by_pair[pq].append(k)

    candidates = set()
    for photos in by_pair.values():
        if len(photos) >= 2:
            candidates.update(combinations(sorted(photos), 2))
    print(f"{len(candidates)} photograph pairs share 2+ people")

    scores = model_scores()
    findings = []

    def emit(kind, person, residual, consensus, n_shared, a, b, co_present):
        worst, note = arbitrate([a, b], scores)
        findings.append({
            "kind": kind,
            "person": person,
            "residual": round(residual, 1),
            "consensus_delta": consensus,
            "n_shared": n_shared,
            "age_a": a["age"], "file_a": f"{a['person']}/{a['file']}",
            "age_b": b["age"], "file_b": f"{b['person']}/{b['file']}",
            "distrust": f"{worst['person']}/{worst['file']}" if worst else "",
            "evidence": note,
            "co_present": "; ".join(co_present),
        })

    # A photograph that contradicts itself: same negative, same person, two ages.
    for k, p, x, y in same_neg:
        emit("same-negative", p, abs(x["age"] - y["age"]), 0, 1, x, y, [])

    for ka, kb in candidates:
        pa, pb = photo[ka], photo[kb]
        shared = sorted(set(pa) & set(pb))
        deltas = {p: pb[p]["age"] - pa[p]["age"] for p in shared}
        consensus = median(deltas.values())
        off = [(abs(deltas[p] - consensus), p) for p in shared]
        off = [(r, p) for r, p in off if r >= min_res]
        if not off:
            continue
        if len(shared) == 2:
            # The median sits between two deltas, so both residuals are identical and
            # blaming both is blaming neither. One finding; the models pick the culprit.
            residual, _ = max(off)
            cands = [photo[k][p] for p in shared for k in (ka, kb)]
            worst, note = arbitrate(cands, scores)
            person = worst["person"] if worst else " / ".join(shared)
            a, b = (photo[ka][person], photo[kb][person]) if worst else (photo[ka][shared[0]], photo[kb][shared[0]])
            findings.append({
                "kind": "disputed", "person": person,
                "residual": round(residual, 1), "consensus_delta": consensus, "n_shared": 2,
                "age_a": a["age"], "file_a": f"{a['person']}/{a['file']}",
                "age_b": b["age"], "file_b": f"{b['person']}/{b['file']}",
                "distrust": f"{worst['person']}/{worst['file']}" if worst else "",
                "evidence": note,
                "co_present": "; ".join(shared),
            })
        else:
            for residual, p in off:
                emit("outlier", p, residual, consensus, len(shared),
                     photo[ka][p], photo[kb][p], [q for q in shared if q != p])

    findings.sort(key=lambda f: (-f["residual"], f["kind"], f["person"]))
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(findings[0]) if findings else ["person"])
        w.writeheader()
        w.writerows(findings)

    named = sum(1 for f in findings if f["distrust"])
    by_kind = Counter(f["kind"] for f in findings)
    print(f"\n{len(findings)} findings (residual >= {min_res}yr): "
          + ", ".join(f"{n} {k}" for k, n in by_kind.most_common()))
    print(f"{named} name a specific file to distrust -> {OUT.relative_to(REPO)}")

    print(f"\nTop {min(top, len(findings))} by residual:")
    for f in findings[:top]:
        where = ("same photograph" if f["kind"] == "same-negative"
                 else f"consensus {f['consensus_delta']:+g}, {f['n_shared']} shared")
        print(f"  {f['residual']:5.1f}yr  [{f['kind']:<13}] {f['person']:<28} "
              f"{f['age_a']:>3} vs {f['age_b']:<3} ({where})")
        if f["co_present"]:
            print(f"          with: {f['co_present'][:70]}")
        print(f"          {f['evidence']}")
        if f["distrust"]:
            print(f"          -> {f['distrust']}")


if __name__ == "__main__":
    main()
