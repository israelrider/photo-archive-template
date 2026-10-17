"""Score how well each named person is supported by IMAGE EVIDENCE, as opposed to
family testimony.

Every by-person folder looks equally authoritative on disk. In reality some people are
backed by a hundred tight, well-separated faces and others by a single photo somebody
recognised. Publishing both with the same confidence would misrepresent the archive.

Measurement note: cohesion is computed WITHIN each cluster, never against one centroid
per person. Someone photographed from 20 to 80 occupies several clusters; a single
centroid sits between them and matches none, which scores the best-documented people
as the weakest. Separation likewise compares cluster-to-cluster.

Writes data/CONFIDENCE.csv. Re-run after any identification change.

    .venv\\Scripts\\python.exe scripts\\confidence.py
"""
from __future__ import annotations

import csv
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
from _paths import BP, DB, LOGS, MODEL
OUT = REPO / "data" / "CONFIDENCE.csv"

SKIP_EXACT = {"Unknown", "Unsorted_Faces"}
SKIP_RE = re.compile(r"^(Person_\d+|\.tmp\.|desktop\.ini)", re.I)
# folders holding two people share their faces between them; cluster->name voting
# cannot resolve them, so they are reported but never scored as one identity. The scheme
# assumes such a folder cannot exist: split it into one folder per person instead.
JOINT: set[str] = set()


def is_person_folder(name: str) -> bool:
    return name not in SKIP_EXACT and not SKIP_RE.match(name)


def main() -> None:
    conn = sqlite3.connect(DB)
    clusters, paths, vecs = [], [], []
    for cid, p, blob in conn.execute(
        "select cluster_id, path, vector from face where model=? and cluster_id is not null",
        (MODEL,),
    ):
        clusters.append(cid); paths.append(p)
        vecs.append(np.frombuffer(blob, dtype=np.float32))
    M = np.vstack(vecs)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    clusters = np.array(clusters)

    # Album file -> person, keyed by ORIGIN PATH from the manifest, never by filename.
    # Filename keying silently fails wherever a date was appended on copy
    # (split\g_09.png vs age08_g_09_1948.png), which hid every split-scan person.
    people = {d.name for d in BP.iterdir() if d.is_dir() and is_person_folder(d.name)}
    # Someone can be identified without owning an album folder - the lost review names
    # people from photographs that never reached one. Scoring only folder-owners would
    # make them invisible in their own archive.
    for _lg in ("lost_review_log.csv", "split_import_log.csv", "rematch_log.csv",
                "borderline_log.csv"):
        _p = REPO / "data" / _lg
        if _p.exists():
            for _r in csv.DictReader(_p.open(encoding="utf-8-sig")):
                _w = (_r.get("person") or "").strip()
                if _w and is_person_folder(_w):
                    people.add(_w)
    owner: dict[str, set[str]] = {}
    man = REPO / "data" / "MANIFEST.csv"
    if not man.exists():
        raise SystemExit(f"{man} missing - run the manifest builder first")
    for r in csv.DictReader(man.open(encoding="utf-8-sig")):
        who = r["person"]
        if who in people and r["origin_path"]:
            owner.setdefault(str(Path(r["origin_path"]).resolve()), set()).add(who)
    print(f"manifest links {len(owner)} origin files to {len(people)} people")

    # How many photos each person's folder holds (by origin, deduplicated).
    folder_photos = Counter()
    for who_set in owner.values():
        for w in who_set:
            folder_photos[w] += 1

    # cluster -> person.
    #
    # AUTHORITATIVE SOURCE FIRST. main_import_log.csv records the (person, cluster)
    # pair decided at import time. Inferring it back from folder membership is
    # unreliable for group photographs and actively wrong: it hands a mother's cluster to
    # her daughter (whose folder is smaller, so specificity favours her) and gives the
    # daughter a cluster that belongs to a stranger.
    log_person: dict[int, str] = {}
    log = LOGS / "main_import_log.csv"
    if log.exists():
        votes_by_cl: dict[int, Counter] = defaultdict(Counter)
        for r in csv.DictReader(log.open(encoding="utf-8-sig")):
            c = (r.get("cluster") or "").strip()
            who = (r.get("person") or "").strip()
            if c.lstrip("-").isdigit() and who and not who.startswith("Person_"):
                votes_by_cl[int(c)][who] += 1
        for c, v in votes_by_cl.items():
            log_person[c] = v.most_common(1)[0][0]

    # Clusters merged in from a separately indexed batch can be recorded in
    # split_import_log.csv (person, cluster), resolved by embedding match against
    # already-named clusters or by the family's own identification. Rows with a blank
    # person are deliberately unresolved and stay unattributed.
    split_log = REPO / "data" / "split_import_log.csv"
    n_split = 0
    if split_log.exists():
        for r in csv.DictReader(split_log.open(encoding="utf-8-sig")):
            who = (r.get("person") or "").strip()
            c = (r.get("cluster") or "").strip()
            if who and c.isdigit():
                log_person[int(c)] = who
                n_split += 1
    # Clusters named from a crop the family saved into the folder (see match_crops.py).
    crop_log = REPO / "data" / "crop_identifications.csv"
    if crop_log.exists():
        for r in csv.DictReader(crop_log.open(encoding="utf-8-sig")):
            if (r.get("person") or "").strip() and str(r["cluster"]).isdigit():
                log_person[int(r["cluster"])] = r["person"]

    # Named by the family in a review. The list of reviews lives in _archive.REVIEW_SETS so
    # that adding one does not mean remembering to edit this file too - which is exactly the
    # duplication that let confidence.py and the review generators drift apart before.
    from _archive import FACE_LEVEL_LOGS, REVIEW_SETS
    n_review = {}
    for fn in (*FACE_LEVEL_LOGS, *(s["log"] for s in REVIEW_SETS.values())):
        p = REPO / "data" / fn
        if not p.exists():
            continue
        n = 0
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            c = str(r.get("cluster") or "")
            if (r.get("person") or "").strip() and c.lstrip("-").isdigit():
                if int(c) not in log_person:
                    n += 1
                log_person[int(c)] = r["person"]
        n_review[fn.replace("_log.csv", "").replace("_review", "")] = n
    print(f"logs give {len(log_person)} cluster->person pairs ({n_split} split, "
          + ", ".join(f"{v} from {k}" for k, v in n_review.items()) + ")")

    # Anything still absent from both logs falls back to folder inference.
    cl_person: dict[int, str] = {}
    for cl in set(clusters.tolist()):
        if cl == -1:
            continue
        if cl in log_person:
            if log_person[cl] in people and log_person[cl] not in JOINT:
                cl_person[cl] = log_person[cl]
            continue
        idx = np.where(clusters == cl)[0]
        cl_photos = {str(Path(paths[i]).resolve()) for i in idx}
        shared = Counter()
        for p in cl_photos:
            for w in owner.get(p, ()):
                shared[w] += 1
        best, best_f1 = None, 0.0
        for w, n in shared.items():
            if w in JOINT:
                continue
            coverage = n / len(cl_photos)
            specificity = n / max(folder_photos.get(w, 1), 1)
            if coverage < 0.5:
                continue
            f1 = 2 * coverage * specificity / (coverage + specificity)
            if f1 > best_f1:
                best, best_f1 = w, f1
        if best:
            cl_person[cl] = best

    # A cluster owned by P must RECUR across P's folder. Otherwise a stranger who only
    # ever appears beside P is counted as P -- which is how one person ends up with over a
    # hundred "clusters". Require presence in >=2 of the person's photos, and >=5%
    # of them for people with large folders.
    hits: dict[tuple[int, str], int] = Counter()
    for cl, who in cl_person.items():
        for i in np.where(clusters == cl)[0]:
            if who in owner.get(str(Path(paths[i]).resolve()), ()):
                hits[(cl, who)] += 1
    had_any = set(cl_person.values())      # before filtering: any face at all
    # The filter polices INFERENCE, not the logs. A cluster the family named directly
    # can legitimately have no folder presence at all - the lost-review faces are
    # exactly that, photographs that never reached anyone's folder, which is why they
    # were lost. Filtering them would discard 197 human answers.
    cl_person = {
        cl: who for cl, who in cl_person.items()
        if cl in log_person or hits[(cl, who)] >= max(2, 0.05 * folder_photos.get(who, 0))
    }
    scattered = had_any - set(cl_person.values())   # faces, but none that recur

    # per-cluster centroid + cohesion
    cl_idx: dict[int, np.ndarray] = {}
    cl_cent: dict[int, np.ndarray] = {}
    cl_coh: dict[int, float] = {}
    for cl in cl_person:
        idx = np.where(clusters == cl)[0]
        cl_idx[cl] = idx
        c = M[idx].mean(axis=0)
        c /= np.linalg.norm(c)
        cl_cent[cl] = c
        cl_coh[cl] = float(np.mean(M[idx] @ c)) if len(idx) > 1 else float("nan")

    person_clusters: dict[str, list[int]] = defaultdict(list)
    for cl, who in cl_person.items():
        person_clusters[who].append(cl)

    # co-occurrence
    photo_people: dict[str, set[str]] = defaultdict(set)
    for i, cl in enumerate(clusters):
        who = cl_person.get(int(cl))
        if who:
            photo_people[paths[i]].add(who)
    partners: dict[str, set[str]] = defaultdict(set)
    for pp in photo_people.values():
        for a in pp:
            partners[a] |= pp - {a}

    all_cl = sorted(cl_cent)
    CC = np.vstack([cl_cent[c] for c in all_cl]) if all_cl else np.zeros((0, 512), np.float32)
    owner_of = [cl_person[c] for c in all_cl]

    rows = []
    for who in sorted(people):
        cls = person_clusters.get(who, [])
        idx = np.concatenate([cl_idx[c] for c in cls]) if cls else np.array([], int)
        n_faces = len(idx)
        n_photos = len({paths[i] for i in idx})
        multi = [c for c in cls if len(cl_idx[c]) > 1]
        singles = len(cls) - len(multi)

        if multi:
            w = np.array([len(cl_idx[c]) for c in multi], float)
            cohesion = float(np.average([cl_coh[c] for c in multi], weights=w))
        else:
            cohesion = float("nan")

        # worst-case separation: nearest OTHER person's cluster to any of ours
        margin, nearest = float("nan"), ""
        if cls:
            worst = None
            for c in cls:
                sims = CC @ cl_cent[c]
                for k, s in enumerate(sims):
                    if owner_of[k] == who:
                        continue
                    base = cl_coh[c] if len(cl_idx[c]) > 1 else 1.0
                    m = base - float(s)
                    if worst is None or m < worst[0]:
                        worst = (m, owner_of[k], float(s))
            if worst:
                margin, nearest = worst[0], worst[1]

        n_partners = len(partners.get(who, ()))

        if who in JOINT:
            tier, basis = "-", "joint folder: two people share it, not scorable as one identity"
        elif n_faces == 0 and who in scattered:
            # A face exists but never recurs - typically one manually-assigned photograph.
            # Nothing here can confirm the identification; it stands on the family's word.
            tier, basis = "D", ("a single photograph, manually assigned and never recurring - "
                                "no embedding support; rests on family testimony")
        elif n_faces == 0:
            tier, basis = "D", "testimony only: no identified photograph of this person"
        elif not multi:
            # A lone face matches only itself. There is nothing here that could confirm
            # the identification, so it stands on the family's word exactly as if there
            # were no photograph at all.
            tier, basis = "D", (f"{singles} lone face(s), never recurring - nothing can "
                                f"confirm this identification; rests on family testimony")
        elif n_photos <= 3:
            tier, basis = "C", "weak: too few photographs to confirm by image"
        elif cohesion < 0.40 or margin < 0.10:
            tier, basis = "B", "moderate: real support, but loose or close to another person"
        elif n_photos <= 8:
            tier, basis = "B", "moderate: consistent, still a small sample"
        else:
            tier, basis = "A", "strong: many photographs, tight clusters, well separated"

        if tier in ("C", "D") and n_partners == 0:
            basis += "; appears with no identified relative"

        rows.append({
            "person": who, "tier": tier, "photos": n_photos, "faces": n_faces,
            "clusters": len(cls), "lone_face_clusters": singles,
            "cohesion": "" if np.isnan(cohesion) else round(cohesion, 3),
            "margin": "" if np.isnan(margin) else round(margin, 3),
            "nearest_other": nearest, "co_occurs_with": n_partners, "basis": basis,
        })

    order = {"A": 0, "B": 1, "C": 2, "D": 3, "-": 4}
    rows.sort(key=lambda r: (order[r["tier"]], -r["photos"]))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} people -> {OUT}")
    print("tiers:", dict(Counter(r["tier"] for r in rows)))
    print()
    hdr = f"{'person':<30}{'tier':<5}{'phot':>5}{'cl':>4}{'cohes':>7}{'margin':>8}{'co':>4}  nearest"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['person']:<30}{r['tier']:<5}{r['photos']:>5}{r['clusters']:>4}"
              f"{str(r['cohesion']):>7}{str(r['margin']):>8}{r['co_occurs_with']:>4}  {r['nearest_other']}")


if __name__ == "__main__":
    main()
