r"""Collapse the archive's three layers -- scan, negative, print -- into one table.

A duplicate pass over `by-person` runs over a derived view: one file per DETECTED FACE,
so a photograph with three people in it is three album files, and each question about a
pair gets asked once per person the photograph is filed under. The unit of the question
is wrong there; this table fixes it.

Three levels, and only one of them is redundancy:

    scan       one origin file. `by-person` copies of it are filing, not duplication.
    plate      one physical print. Two scans of one plate -> one can go.
    negative   one photograph. Two plates of one negative are two real objects -> keep both.

`album_same_negative.csv` decides the negative, `album_print_vs_rescan.csv` decides the
plate, and this reads both and states what follows.

Negatives are built from REVIEWED evidence only. dHash pairs are unreviewed, so
they are not merged -- they are written out as proposals, with the count of negatives each
would join, because the point of this table is to separate what is settled from what is
still open rather than to bury the difference.

Reads only. Writes `data/photographs.csv`, `data/photograph_proposals.csv`, and a report.

    .venv\Scripts\python.exe scripts\build_photographs.py
"""
from __future__ import annotations

import csv
import datetime
import os
import re
import sys
from collections import defaultdict

import _archive as A
import stable_ids as S

sys.stdout.reconfigure(encoding="utf-8")

REPO = A.REPO
OUT = REPO / "data" / "photographs.csv"
PROPOSALS = REPO / "data" / "photograph_proposals.csv"
REPORT = REPO / "artifacts" / "checks" / "photographs.txt"
REGISTRY = REPO / "data" / "scan_ids.csv"
TODAY = ""          # set from the clock in main()

# `age30_m05_... .jpg`, `noage_m05_... .jpg` and `Person_368__age09_m05_... .jpg` are one
# scan under three labels. The age is a guess that gets revised; the scan key never moves.
STRIP = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+_|noage_)?(.*)$")

# Origins under these folders are uncut sheets: several different photographs were scanned
# together and split afterwards, so many scan keys legitimately share one origin path.
# Origin is therefore not an identity for a scan -- the scan key is. ([origins] split_sheets)
SPLIT_ORIGINS = tuple(str(p) for p in A.SPLIT_SHEETS)


def scan_key(name: str) -> str:
    return STRIP.match(os.path.basename(name)).group(1)


class Union:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # keep the lexicographically smaller root so ids are stable across runs
        lo, hi = sorted((ra, rb))
        self.p[hi] = lo
        return True

    def groups(self) -> dict[str, list[str]]:
        g = defaultdict(list)
        for x in self.p:
            g[self.find(x)].append(x)
        return g


def rows(name: str) -> list[dict]:
    p = REPO / "data" / name
    return list(csv.DictReader(p.open(encoding="utf-8-sig", newline=""))) if p.exists() else []


def main() -> None:
    notes: list[str] = []
    global TODAY
    TODAY = datetime.date.today().isoformat()

    # ---------- the scans, and everything filed on top of each one ----------
    # A thinned album file is deleted from disk but its manifest row survives, so the
    # manifest overstates by-person by exactly the thinning. Drop those rows here rather
    # than counting a file that is not there.
    gone = {os.path.relpath(r["deleted_path"], str(A.BP)).replace(os.sep, "/")
            for r in rows("album_thinned.csv")}

    origin: dict[str, str] = {}                        # scan key -> origin path
    album: dict[str, list[str]] = defaultdict(list)    # scan key -> live album files
    persons: dict[str, set[str]] = defaultdict(set)
    stale = 0
    for r in rows("MANIFEST.csv"):
        k = scan_key(r["album_file"])
        origin[k] = r["origin_path"]                   # the origin outlives the filing
        if r["album_file"].replace(os.sep, "/") in gone:
            stale += 1
            continue
        album[k].append(r["album_file"])
        persons[k].add(r["person"])

    # a scan whose every copy was thinned is no longer filed under anyone
    unfiled = {k for k in origin if not album[k]}

    unmapped: set[str] = set()

    def key_of(path: str) -> str | None:
        """A by-person path -> its scan key, or None if that scan has no manifest row."""
        k = scan_key(path)
        if k in origin:
            return k
        unmapped.add(k)      # by key, so one absent scan is not reported under three labels
        return None

    # ---------- negatives: reviewed same-negative verdicts ----------
    neg = Union()
    for k in origin:
        neg.find(k)

    neg_ev: dict[str, set[str]] = defaultdict(set)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows("album_same_negative.csv"):
        if "verdict" not in r:
            raise SystemExit("album_same_negative.csv has no verdicts yet - run "
                         "review_album_same_negative.py --apply first")
        by_group[r["group"]].append(r)

    rejected = 0
    for gid, grp in by_group.items():
        members = []
        for r in grp:
            if r["verdict"] != "same-negative":
                rejected += 1
                continue
            k = key_of(r["path"])
            if k is not None:
                members.append(k)
        for k in members[1:]:
            neg.join(members[0], k)
        for k in members:
            neg_ev[k].add(f"same-negative g{gid}")

    # ---------- plates: the print-vs-rescan verdicts, inside a negative ----------
    plate = Union()
    for k in origin:
        plate.find(k)
    plate_ev: dict[str, set[str]] = defaultdict(set)
    split: set[tuple[str, str]] = set()      # pairs judged two separate physical prints
    for r in rows("album_print_vs_rescan.csv"):
        a, b = key_of(r["reference"]), key_of(r["other"])
        if a is None or b is None:
            continue
        if r["kind"] == "rescan":
            plate.join(a, b)
            plate_ev[a].add(r["confidence"])
            plate_ev[b].add(r["confidence"])
        else:
            split.add(tuple(sorted((a, b))))

    # ---------- the review's own verdicts ----------
    # `duplicate_review_log.csv` carries what the family settled that no earlier file could
    # express: pairs that ARE one negative shown by two separate prints, and pairs the
    # dHash proposed that they refused. The refusal has to be recorded too, or the next
    # hash pass proposes all fifteen again.
    refused: set[tuple[str, str]] = set()
    for r in rows("duplicate_review_log.csv"):
        pair = tuple(sorted((r["scan_a"], r["scan_b"])))
        if r["resolution"].startswith("different-photograph"):
            refused.add(pair)
        elif r["resolution"].startswith("same-negative"):
            a, b = key_of(r["scan_a"]), key_of(r["scan_b"])
            if a and b:
                neg.join(a, b)
                neg_ev[a].add("review: one negative, two prints")
                neg_ev[b].add("review: one negative, two prints")
                split.add(tuple(sorted((a, b))))

    # ---------- verdicts that exist only as prose in album_thinned.csv ----------
    # A thinning is a decided pair -- "rescan of X", "keeping X", "Kept X" -- but it was
    # written as a sentence, so nothing could read it back. Where the sentence names the
    # keeper by filename the pair is machine-readable; where it names it in prose
    # ("0021 is KEPT") it is counted and reported, never guessed.
    #
    # **Only one of those phrasings is a claim about plates.** "rescan of X" says the two
    # scans are one physical print, so they belong on one plate. "kept X" / "keeping X" says
    # only that somebody chose between two objects -- and the family has been choosing
    # between SEPARATE PRINTS, on quality. Joining those welded two surviving prints onto one
    # plate and undercounted what the archive holds: group 2 was rated `separate-print-strong`
    # by `album_print_vs_rescan.csv` and came back `one-plate` anyway, because its reason
    # sentence -- which opens "separate prints (18 blobs, area 12.0)" -- also contains the
    # word "kept".
    #
    # 43 of the 61 joins this block used to make were quality choices, 36 of them saying
    # "separate prints" in the same sentence that names the keeper. The split is clean in the
    # other direction too: every "rescan of" thinning lost all its copies, and every one of
    # the 12 thinnings whose dropped scan still survives in another folder was a "kept".
    # So the verb is the whole discriminator, and no survivorship guard is needed beside it.
    KEEPER = re.compile(r"(?:rescan of|[Kk]ept|keeping)\s+((?:age\d+|noage)_[^;,.]+?\.(?:jpg|png))")
    RESCAN = re.compile(r"rescan of\s+(?:age\d+|noage)_")
    prose_only, quality_choice = [], []
    for r in rows("album_thinned.csv"):
        dropped = scan_key(r["deleted_path"])
        m = KEEPER.search(r["reason"])
        keeper = scan_key(m.group(1)) if m else None
        if keeper == dropped:
            continue          # a second copy of one scan: filing, not a pair to link
        if keeper is None or keeper not in origin or dropped not in origin:
            # Only a scan that lost every copy needs a link: while other copies of the
            # same scan survive, the deletion removed filing, not a second photograph.
            if dropped in unfiled:
                prose_only.append(os.path.basename(r["deleted_path"]))
            continue
        if not RESCAN.search(r["reason"]):
            # A choice between two prints, not evidence that they are one print. The
            # negative still holds both; `album_print_vs_rescan.csv` decides the plates.
            quality_choice.append(os.path.basename(r["deleted_path"]))
            continue
        plate.join(dropped, keeper)
        plate_ev[dropped].add("thinned")
        plate_ev[keeper].add("thinned")

    # A replacement in quality_improved_log is the same physical print twice over -- a
    # rephotograph put in place of a scan of it - so the pair is one plate wherever both
    # sides still have a row.
    for r in rows("quality_improved_log.csv"):
        a, b = scan_key(r["replaced_album_file"]), scan_key(r["new_album_file"])
        if a in origin and b in origin and a != b:
            plate.join(a, b)
            plate_ev[a].add("quality-replaced")
            plate_ev[b].add("quality-replaced")

    # A plate merge is evidence about the negative too: a rescan of a print is trivially
    # the same photograph, even where no same-negative group ever said so.
    for members in plate.groups().values():
        for k in members[1:]:
            neg.join(members[0], k)

    # ---------- unreviewed dHash proposals, kept out of the table ----------
    proposals = []
    txt = REPO / "near_duplicate_scans.txt"
    if txt.exists():
        for d, a, b in re.findall(r"^d=(\d+)\s+(.+?)\s+<->\s+(.+?)\s*$",
                                  txt.read_text(encoding="utf-8"), re.M):
            ka, kb = key_of(a), key_of(b)
            if ka is None or kb is None:
                continue
            proposals.append({
                "hamming": int(d),
                "scan_a": ka,
                "scan_b": kb,
                "negative_a": neg.find(ka),
                "negative_b": neg.find(kb),
                "already_one_negative": "yes" if neg.find(ka) == neg.find(kb) else "no",
                "judged_separate_prints": "yes" if tuple(sorted((ka, kb))) in split else "no",
                "reviewed": ("refused: different photographs"
                             if tuple(sorted((ka, kb))) in refused else ""),
            })
    proposals.sort(key=lambda p: (p["hamming"], p["scan_a"]))

    # ---------- ids ----------
    # An id belongs to a SET OF SCANS, not to a position in this run. Numbering them
    # positionally meant dropping one negative slid every later id down by one, so an id
    # quoted in a note or a commit message came back naming a different photograph.
    # stable_ids.py hands each group the id its own members already carried.
    neg_groups = neg.groups()
    plate_groups = plate.groups()
    registry = S.Registry(REGISTRY)
    prior_neg, neg_moved = registry.resolve("negative", origin)
    prior_plate, plate_moved = registry.resolve("plate", origin)
    neg_id, neg_notes = S.assign(neg_groups, prior_neg, "N",
                                 registry.max_number("negative"))
    plate_id, plate_notes = S.assign(plate_groups, prior_plate, "P",
                                     registry.max_number("plate"))
    id_notes = neg_moved + neg_notes + plate_moved + plate_notes
    for p in proposals:
        p["negative_a"] = neg_id[p["negative_a"]]
        p["negative_b"] = neg_id[p["negative_b"]]

    plates_in_neg: dict[str, set[str]] = defaultdict(set)
    for k in origin:
        plates_in_neg[neg.find(k)].add(plate.find(k))

    def status(k: str) -> str:
        if k in unfiled:
            return "unfiled"
        n = neg.find(k)
        members = neg_groups[n]
        if len(members) == 1:
            return "single"
        if len(plates_in_neg[n]) == 1:
            return "one-plate"
        pairs = {(a, b) for a in members for b in members if a < b}
        undecided = [p for p in pairs
                     if p not in split and plate.find(p[0]) != plate.find(p[1])]
        return "plates-undecided" if undecided else "separate-prints"

    out = []
    missing = 0
    for k in sorted(origin):
        o = origin[k]
        try:
            size = os.path.getsize(o)
        except OSError:
            size = ""
            missing += 1
        out.append({
            "negative": neg_id[neg.find(k)],
            "plate": plate_id[plate.find(k)],
            "scan": k,
            "status": status(k),
            "album_files": len(album[k]),
            "persons": "; ".join(sorted(persons[k])),
            "origin_bytes": size,
            "origin_path": o,
            "negative_evidence": "; ".join(sorted(neg_ev[k])),
            "plate_evidence": "; ".join(sorted(plate_ev[k])),
        })
    out.sort(key=lambda r: (r["negative"], r["plate"], r["scan"]))

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else [])
        w.writeheader()
        w.writerows(out)
    with PROPOSALS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proposals[0]) if proposals else ["scan_a", "scan_b"])
        w.writeheader()
        w.writerows(proposals)

    # The registry is the reason an id means the same thing next run. Retired rows stay:
    # they are what makes an old citation legible, and a reissued number would be worse
    # than a retired one.
    registry.update("negative", neg_id, neg_groups, origin, TODAY)
    registry.update("plate", plate_id, plate_groups, origin, TODAY)
    retired = registry.retire_absent(set(origin), TODAY)
    registry.save()

    # ---------- the report ----------
    n_album = sum(len(v) for v in album.values())
    multi = [n for n, m in neg_groups.items() if len(m) > 1]
    redundant = sum(len(m) - 1 for m in plate_groups.values())
    st: dict[str, int] = defaultdict(int)
    for r in out:
        st[r["status"]] += 1
    open_negs = {r["negative"] for r in out if r["status"] == "plates-undecided"}
    new_pairs = [p for p in proposals if p["already_one_negative"] == "no"]

    L = notes.append
    L("THE ARCHIVE IN THREE LAYERS")
    L("=" * 78)
    L(f"  {n_album:5d}  album files in by-person")
    L(f"  {len(origin):5d}  scans          (distinct origin files)")
    L(f"  {len(plate_groups):5d}  plates         (distinct physical prints)")
    L(f"  {len(neg_groups):5d}  negatives      (distinct photographs)")
    L("")
    L(f"  {n_album - len(origin):5d}  album files are another person's copy of a scan already here.")
    L("         Not redundancy: it is one file per detected face. It goes away by")
    L("         generating by-person from the manifest, not by deleting anything.")
    L(f"  {redundant:5d}  scans are a second scan of a plate already scanned -- the only")
    L("         population where deleting a file loses nothing.")
    L(f"  {len(plate_groups) - len(neg_groups):5d}  plates beyond one per negative: second physical prints. Keep.")
    L("")
    L("STATUS OF EVERY SCAN")
    L("=" * 78)
    for k in sorted(st, key=lambda x: -st[x]):
        L(f"  {st[k]:5d}  {k}")
    L("")
    L(f"  {len(multi):5d}  negatives hold more than one scan")
    L(f"  {len(open_negs):5d}  of those have a pair never judged print-vs-rescan  <-- open work")
    L("")
    L("UNREVIEWED dHash PROPOSALS (not merged above)")
    L("=" * 78)
    for d in sorted({p["hamming"] for p in proposals}):
        at = [p for p in proposals if p["hamming"] == d]
        same = sum(1 for p in at if p["already_one_negative"] == "yes")
        L(f"  d={d}  {len(at):3d} pairs   {same:3d} already one negative   {len(at)-same:3d} would be new")
    refused_n = sum(1 for p in proposals if p["reviewed"])
    open_props = [p for p in new_pairs if not p["reviewed"]]
    L(f"  {len(new_pairs):5d}  pairs would join negatives currently held apart, of which")
    L(f"  {refused_n:5d}  have been examined and refused -- the hash matched them on tone")
    L(f"  {len(open_props):5d}  are still unexamined  <-- open work")
    L("")
    L("ANOMALIES")
    L("=" * 78)
    L(f"  {stale:5d}  MANIFEST rows for album files thinning already deleted -- the")
    L("         manifest has not been rebuilt since. Excluded from every count above.")
    L(f"  {len(unfiled):5d}  scans no longer filed under anyone (every copy thinned)")
    L(f"  {missing:5d}  origin files named by the manifest that are not on disk")
    # Once the manifest is rebuilt, a removed file has no row at all, so most of these are
    # not anomalies but history: a verdict naming a scan that a later decision took out.
    # Two decisions do that, and both keep a record of it.
    deleted = {scan_key(r["deleted_path"]) for r in rows("album_thinned.csv")}
    deleted |= {scan_key(r["replaced_album_file"]) for r in rows("quality_improved_log.csv")}
    history, adrift = sorted(unmapped & deleted), sorted(unmapped - deleted)
    L(f"  {len(history):5d}  scans a verdict file names that a later decision removed --")
    L("         history, not damage: album_thinned or quality_improved_log says why")
    L(f"  {len(adrift):5d}  scans named by a verdict file and accounted for nowhere:")
    for u in adrift:
        L(f"           {u}")
    L("")
    shared = defaultdict(list)
    for k, o in origin.items():
        shared[o].append(k)
    odd = {o: ks for o, ks in shared.items()
           if len(ks) > 1 and not (SPLIT_ORIGINS and o.startswith(SPLIT_ORIGINS))}
    L(f"  {len(odd):5d}  origins claimed by more than one scan and NOT an uncut sheet --")
    L("         two scan keys pointing at one file, which one of them must be wrong:")
    for o, ks in sorted(odd.items()):
        L(f"           {o}")
        for k in sorted(ks):
            L(f"               {k}")
    L("")
    L(f"  {len(prose_only):5d}  thinnings whose reason names its keeper only in prose, so the")
    L("         pair could not be read back into a plate. Every one is a decided")
    L("         duplicate the table below therefore still shows as two plates:")
    for n in prose_only:
        L(f"           {n}")
    L("")
    L(f"  {len(quality_choice):5d}  thinnings that chose between two SEPARATE PRINTS rather than")
    L("         collapsing a rescan. The keeper is named and readable, but the")
    L("         sentence is not a claim that the two are one print, so the plates")
    L("         are left to album_print_vs_rescan.csv and the negative keeps both:")
    for n in quality_choice:
        L(f"           {n}")
    lonely = [r for r in out if r["status"] == "unfiled"
              and sum(1 for x in out if x["plate"] == r["plate"]) == 1]
    L("")
    L(f"  {len(lonely):5d}  scans thinned out of by-person with no surviving twin on their")
    L("         plate -- the print is now in no one's folder:")
    for r in lonely:
        L(f"           {r['scan']}")
    L("")
    L(f"  {rejected:5d}  same-negative rows rejected at review (different-photograph / split-off)")
    L("")

    # Ids are the part of this table other files quote, so every movement is stated
    # rather than left to be discovered in a diff.
    L(f"  {len(id_notes):5d}  id movements this run "
      f"(new, merged, split, or following a renamed scan):")
    for n in id_notes:
        L(f"           {n}")
    if not id_notes:
        L("           none -- every group kept the id it had")
    L("")
    L(f"  {len(retired):5d}  scans that left the archive since the last run; their ids "
      f"are retired")
    L("         and will NOT be reissued -- see data\\retired_ids.csv")

    text = "\n".join(notes)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT}\nwrote {PROPOSALS}\nwrote {REPORT}")


if __name__ == "__main__":
    main()
