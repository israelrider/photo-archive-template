"""Copy the photographs of named faces into the person's album folder.

A review names a FACE; the photograph stays in its origin root with no copy in anyone's
folder until this runs. It copies each one to its person, following the album convention:

    age<NN>_<root tag><subfolder>_<original name>      e.g. age27_m05_IMG_0012.jpg

`<root tag>` is the origin root's letter from [origins.tags] in archive.toml (default: the
first letter of the root's folder name). People listed under [people.folders] are filed into
that folder instead of by-person\\<name>\\ (e.g. the archive owner's own album).

Origins are never touched, and MANIFEST.csv gains a row per by-person copy.

A copy whose origin hash appears in `data\\orientation_corrections.csv` gets that recorded
tag written onto it after the byte-identity check, so it is the origin's image the right way
up rather than the origin's bytes on their side. Without it, filing an already-corrected
photograph to another person reinstates the wrong tag. See `orientation_carry.py`.

Any log with (person, face_id, source) columns can be distributed: lost_review_log.csv,
rematch_log.csv, borderline_log.csv. Re-running is safe: a copy that already exists is
skipped, and a copy taken out of a folder on purpose is withheld (see _archive.withdrawn()).

    .venv\\Scripts\\python.exe scripts\\distribute.py [log.csv] [--apply]

THE AGE PREFIX COMES FROM MiVOLO, NEVER FROM `face.age`

`face.age` is `buffalo_l`'s estimate and is unusable on old scans - it tags a ten-year-old
47 and 60. `_mivolo.ages()` supplies the prefix and never falls back: if MiVOLO cannot run
the tag is `noage`. A filename that admits it does not know beats one that states a wrong
number with confidence. Rounded, not truncated (26.8 is `age27`).
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

import _archive as A
import _mivolo
from orientation_carry import load as load_corrections, straighten

APPLY = "--apply" in sys.argv
FOLDERS = A.PERSON_FOLDERS
TAGS = A.CONFIG.get("origins", {}).get("tags", {})


def destination(person: str) -> tuple[Path, str]:
    """Where this person's copy goes, and which kind of root that is."""
    if person in FOLDERS:
        return FOLDERS[person], "personal"
    return A.BP / person, "by-person"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def root_tag(src: Path) -> str:
    """`<root letter><subfolder>`, e.g. `m05`; just the letter for a file at the root's top.

    A tag must never change once files carry it: changing it gives existing copies new
    names, and the next run copies them all again."""
    for root in A.ORIGIN_ROOTS:
        if src.is_relative_to(root):
            letter = TAGS.get(root.name, root.name[:1].lower() or "x")
            return letter + ("" if src.parent == root else src.parent.name.replace("_", "-"))
    return "x"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    log = Path(args[0]) if args else A.LOGS / "lost_review_log.csv"
    print(f"log: {log.name}")
    rows = list(csv.DictReader(log.open(encoding="utf-8-sig")))

    # A ROW WITH NO face_id IS A PHOTOGRAPH-LEVEL ANSWER, AND THIS SCRIPT CANNOT FILE ONE.
    # Which folder it belongs in is a routing question somebody has to answer, not one a
    # distributor may settle in passing, so such rows are held out and printed.
    faceless = [r for r in rows if not (r.get("face_id") or "").strip()]
    rows = [r for r in rows if (r.get("face_id") or "").strip()]
    if faceless:
        print(f"   {len(faceless)} rows name a photograph, not a face -- NOT FILED here:")
        for r in faceless:
            print(f"      {Path(r['source']).name} -> {r['person']} ({r.get('method', '')})")
    conn = sqlite3.connect(A.DB)
    # One MiVOLO subprocess for the whole batch; {} if it cannot run, so every tag is
    # "noage" rather than a wrong year.
    age = _mivolo.ages([int(r["face_id"]) for r in rows], conn)

    plan: list[tuple[Path, Path, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        src = Path(r["source"])
        person = r["person"]
        if (person, str(src)) in seen:
            continue                      # two faces of one person in one print
        seen.add((person, str(src)))
        a = age.get(int(r["face_id"]), -1)
        tag = f"age{round(a):02d}" if a and a > 0 else "noage"
        name = f"{tag}_{root_tag(src)}_{src.name}"
        folder, root = destination(person)
        plan.append((src, folder / name, person, root))

    # A COPY TAKEN OUT OF A FOLDER ON PURPOSE IS NOT A MISSING COPY. Skipping a destination
    # that exists makes this idempotent against the LIVE folder only; a thinned file is in
    # .thinned\, so without this every re-run re-files everything ever thinned.
    wd = A.withdrawn()
    withheld, keep = [], []
    for p in plan:
        hit = A.withheld_from(wd, p[1].parent.name, p[0].name)
        (withheld if hit else keep).append((p, hit))
    plan = [p for p, _ in keep]

    new_folders = sorted({d.parent for _, d, _, root in plan
                          if root != "personal" and not d.parent.is_dir()})
    print(f"{len(rows)} faces -> {len(plan)} copies")
    for root in ("by-person", "personal"):
        print(f"   into {root:<10s}: {sum(1 for *_, r in plan if r == root)}")
    print(f"   withheld, taken out of that folder on purpose: {len(withheld)}")
    for (src, _, person, _), (rec, why) in withheld:
        print(f"      {src.name} -> {person}: {rec} ({why[:60]})")
    if new_folders:
        print("   new folders to create:")
        for f in new_folders:
            print(f"      {f.parent.name}\\{f.name}")
    clash = [d for _, d, _, _ in plan if d.exists()]
    print(f"   destinations that already exist: {len(clash)}")

    if not APPLY:
        print("\nDRY RUN. Re-run with --apply.")
        return

    CARRY = load_corrections()
    done, failed, straightened = 0, [], []
    for src, dst, person, _ in plan:
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha(src) != sha(dst):
            failed.append(str(dst))
        else:
            tag = CARRY.get(sha(dst))
            if tag is not None:
                how = straighten(dst, tag)
                if how:
                    straightened.append((dst, how))
            done += 1
    print(f"\ncopied+verified: {done}")
    if failed:
        print("VERIFY FAILED:", failed[:5])
        raise SystemExit(1)
    if straightened:
        print(f"\nstraightened {len(straightened)} copy(ies) from "
              "orientation_corrections.csv -- these are deliberately NOT "
              "byte-identical to their origins:")
        for d, how in straightened:
            print(f"   {how:<52s} {d}")

    # manifest rows for the by-person copies only (personal folders are outside it)
    man = A.LOGS / "MANIFEST.csv"
    existing = list(csv.DictReader(man.open(encoding="utf-8-sig"))) if man.exists() else []
    have = {r["album_file"] for r in existing}
    added = 0
    for src, dst, person, root in plan:
        if root != "by-person":
            continue
        key = f"{person}/{dst.name}"
        if key in have:
            continue
        existing.append({"album_file": key, "person": person,
                         "relation": "copy", "origin_path": str(src)})
        added += 1
    existing.sort(key=lambda r: r["album_file"])
    with man.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["album_file", "person", "relation", "origin_path"])
        w.writeheader(); w.writerows(existing)
    print(f"manifest: +{added} rows, {len(existing)} total")


if __name__ == "__main__":
    main()
