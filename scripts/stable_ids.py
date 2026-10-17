r"""Negative and plate ids that survive the table being rebuilt.

`build_photographs.py` numbered its groups positionally -- `enumerate(sorted(groups), 1)` --
so an id meant "the Nth group in this run", not "this photograph". Removing one negative slid
every later id down by one. That is invisible in a diff of a 2000-row CSV and it silently
invalidates every id ever written down: drop one folder from the album, and ids cited in
commit messages and notes come back pointing at different scans.

An id here belongs to a SET OF SCANS, not to a position. The registry remembers which scans
carried which id, and a rebuild hands each group the id its members already had:

    merged      two groups become one -> it keeps the numerically smaller id, the other retires
    split       one group becomes two -> the part holding more of the old scans keeps the id
    shrunk      a member is deleted    -> the id follows the surviving members
    gone        every member deleted   -> the id retires and is NEVER reissued
    new         no member seen before  -> the next number above every id ever issued

Retired rows are kept forever. They are what makes an old citation legible -- and reissuing
an id to a different photograph later would be worse than retiring it.
"""
from __future__ import annotations

import csv
import io
import pathlib


class Registry:
    """scan -> the negative and plate ids it carried, across every run so far.

    The origin path is stored beside the scan key because the scan key MOVES: dating a
    photograph appends `_19650226` to its album basename, and an id cited before the rename
    no longer matches the scan by name. An origin is the one thing the archive never
    renames, so it is the fallback when a scan key has moved under an id.
    """

    FIELDS = ["scan", "origin", "negative", "plate", "first_seen", "last_seen", "status"]

    RETIRED_FIELDS = ["kind", "id", "retired_on", "reason"]

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            with io.open(path, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    self.rows[r["scan"]] = r
        # Every id ever issued, including ones no live row still mentions. A merge
        # overwrites the absorbed id in every scan row that carried it, so without this
        # ledger the high-water mark would fall back and the number would be reissued to a
        # different photograph -- the exact failure this module exists to prevent.
        self.retired_path = path.with_name("retired_ids.csv")
        self.retired: dict[tuple[str, str], dict] = {}
        if self.retired_path.exists():
            with io.open(self.retired_path, encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    self.retired[(r["kind"], r["id"])] = r

    def retire(self, kind: str, ident: str, today: str, reason: str) -> None:
        self.retired.setdefault((kind, ident),
                                {"kind": kind, "id": ident,
                                 "retired_on": today, "reason": reason})

    def ids(self, kind: str) -> dict[str, str]:
        """scan -> id, for every scan ever seen (retired included)."""
        return {s: r[kind] for s, r in self.rows.items() if r.get(kind)}

    def resolve(self, kind: str, scans: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        """current scan -> the id it carried before: by scan key, else by origin path.

        `scans` is scan -> origin_path for the run being built. An origin shared by more
        than one prior id (several scans split out of one uncut sheet) is not used as
        a fallback: a guess there would move an id rather than hold it still.
        """
        by_scan = {s: r[kind] for s, r in self.rows.items() if r.get(kind)}
        by_origin: dict[str, set[str]] = {}
        for r in self.rows.values():
            if r.get(kind) and r.get("origin"):
                by_origin.setdefault(r["origin"], set()).add(r[kind])

        out: dict[str, str] = {}
        notes: list[str] = []
        for scan, origin in scans.items():
            if scan in by_scan:
                out[scan] = by_scan[scan]
                continue
            cand = by_origin.get(origin, set())
            if len(cand) == 1:
                ident = next(iter(cand))
                out[scan] = ident
                prior = sorted(s for s, r in self.rows.items()
                               if r.get(kind) == ident and r.get("origin") == origin)
                notes.append(f"{ident} followed a renamed scan: {prior[0]!r} -> {scan!r}")
            elif len(cand) > 1:
                notes.append(f"{scan!r}: origin carries {len(cand)} prior {kind} ids "
                             f"({', '.join(sorted(cand))}) -- not used, id left to be reissued")
        return out, notes

    def max_number(self, kind: str) -> int:
        """The highest number ever issued for this kind, retired ids included."""
        nums = [int(r[kind][1:]) for r in self.rows.values()
                if r.get(kind) and r[kind][1:].isdigit()]
        nums += [int(i[1:]) for (k, i) in self.retired
                 if k == kind and i[1:].isdigit()]
        return max(nums, default=0)

    def update(self, kind: str, assigned: dict[str, str], members: dict[str, list[str]],
               origins: dict[str, str], today: str) -> None:
        before = {r[kind] for r in self.rows.values() if r.get(kind)}
        for root, ident in assigned.items():
            for scan in members[root]:
                row = self.rows.get(scan)
                if row is None:
                    row = {"scan": scan, "origin": origins.get(scan, ""),
                           "negative": "", "plate": "",
                           "first_seen": today, "last_seen": today, "status": "live"}
                    self.rows[scan] = row
                row[kind] = ident
                row["origin"] = origins.get(scan, row.get("origin", ""))
                row["last_seen"] = today
                row["status"] = "live"

        # An id that was in use and is now held by no group has gone: absorbed by a merge,
        # or its every scan left the archive. Record it so its number is never reissued.
        for ident in sorted(before - set(assigned.values())):
            self.retire(kind, ident, today,
                        "no group holds it after this run (merged away, or every scan left)")

    def retire_absent(self, live_scans: set[str], today: str) -> list[str]:
        out = []
        for scan, row in self.rows.items():
            if scan not in live_scans and row["status"] != "retired":
                row["status"] = "retired"
                out.append(scan)
        return out

    def save(self) -> None:
        with io.open(self.path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS, lineterminator="\r\n")
            w.writeheader()
            for scan in sorted(self.rows):
                w.writerow({k: self.rows[scan].get(k, "") for k in self.FIELDS})
        if self.retired:
            with io.open(self.retired_path, "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.RETIRED_FIELDS,
                                   lineterminator="\r\n")
                w.writeheader()
                for key in sorted(self.retired, key=lambda k: (k[0], k[1])):
                    w.writerow(self.retired[key])


def assign(groups: dict[str, list[str]], prior: dict[str, str], prefix: str,
           start_above: int) -> tuple[dict[str, str], list[str]]:
    """Give every group an id, preferring the one its members already carried.

    groups  root -> members (the union-find output)
    prior   scan -> id, from the registry
    Returns (root -> id, human-readable notes about every id that moved).
    """
    def num(i: str) -> int:
        return int(i[1:])

    # which groups have a claim on each id, and how strong
    claims: dict[str, list[tuple[int, str, str]]] = {}
    for root, members in groups.items():
        held: dict[str, int] = {}
        for m in members:
            if m in prior:
                held[prior[m]] = held.get(prior[m], 0) + 1
        for ident, n in held.items():
            # more of the id's old scans wins; ties go to the smaller member name
            claims.setdefault(ident, []).append((n, min(members), root))

    assigned: dict[str, str] = {}
    taken: set[str] = set()
    notes: list[str] = []

    # settle ids in numeric order so a merged group keeps the smaller number
    for ident in sorted(claims, key=num):
        # strongest claim wins: most shared scans, then lexicographically smallest member
        best = sorted(claims[ident], key=lambda c: (-c[0], c[1]))[0]
        root = best[2]
        if root in assigned:
            notes.append(f"{ident} merged into {assigned[root]}")
            continue
        assigned[root] = ident
        taken.add(ident)
        losers = [c for c in claims[ident] if c[2] != root]
        if losers:
            notes.append(f"{ident} stayed with the group holding {best[0]} of its scans; "
                         f"{len(losers)} other group(s) split off and are renumbered")

    nxt = start_above + 1
    for root in sorted(groups):
        if root in assigned:
            continue
        while f"{prefix}{nxt:04d}" in taken:
            nxt += 1
        ident = f"{prefix}{nxt:04d}"
        assigned[root] = ident
        taken.add(ident)
        notes.append(f"{ident} new ({len(groups[root])} scan(s), first is {min(groups[root])})")
        nxt += 1

    return assigned, notes
