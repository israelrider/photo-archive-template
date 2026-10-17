"""Where the archive lives. One definition, imported by everything that needs a path.

Every location comes from `archive.toml` at the repository root, so the same scripts serve
any person's archive: copy `archive.example.toml` to `archive.toml` and fill it in.

Deliberately stdlib-only. `_archive` is the natural home for shared constants, but it imports
cv2, numpy and sqlite3 at module level, and a one-shot filing script should not have to load
a vision stack to learn where a folder is.

ARCHIVE holds the origins (the files as received, never edited); BP is the by-person album
the family reads. ALBUM_ROOTS lists BP plus any other album trees that hold copies of
origins, so "has this origin been filed?" checks all of them.
"""
from __future__ import annotations

import csv
import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("ARCHIVE_CONFIG", REPO / "archive.toml"))

if not CONFIG_FILE.exists():
    raise SystemExit(f"{CONFIG_FILE} not found - copy archive.example.toml to archive.toml "
                     "and fill in the paths for this archive")

with CONFIG_FILE.open("rb") as _f:
    CONFIG = tomllib.load(_f)


def _path(v: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(v)))


expand = _path


_p = CONFIG["paths"]
ARCHIVE = _path(_p["archive"])
_o = CONFIG.get("origins", {})


def _origin_roots() -> list[Path]:
    """Origin roots: the configured list, else every top-level folder of ARCHIVE."""
    if _o.get("roots"):
        return [ARCHIVE / r for r in _o["roots"]]
    if not ARCHIVE.is_dir():
        return []
    skip = tuple(_o.get("skip_suffixes", ["-quarantine"]))
    return [d for d in sorted(ARCHIVE.iterdir())
            if d.is_dir() and not d.name.startswith(".") and not d.name.endswith(skip)]


ORIGIN_ROOTS = _origin_roots()
# Roots whose files were born digital (phone, camera, e-mail) rather than scanned.
BORN_DIGITAL = set(_o.get("born_digital", []))
# Folders OUTSIDE the archive whose files may also be the origin of an album copy
# (e.g. a relative's own album). Matched as relation `external`.
EXTERNAL_ORIGINS = [_path(v) for v in _o.get("external", [])]
# Folders (relative to ARCHIVE) holding uncut sheets: one scan of several photographs that
# was split afterwards, so several scan keys legitimately share one origin path.
SPLIT_SHEETS = [ARCHIVE / v for v in _o.get("split_sheets", [])]
BP = _path(_p["by_person"])
ALBUM_ROOTS = [BP, *(_path(v) for v in _p.get("other_albums", []))]
DB = _path(_p["face_db"])
IMAGES_AGENT = _path(_p["images_agent"])
MIVOLO_PYTHON = _path(_p["mivolo_python"]) if _p.get("mivolo_python") else None
MIVOLO_REPO = _path(_p["mivolo_repo"]) if _p.get("mivolo_repo") else None
SHARE = _path(_p["share"]) if _p.get("share") else None
MODEL = CONFIG.get("faces", {}).get("model", "buffalo_l")

ARCHIVE_NAME = CONFIG.get("archive", {}).get("name", "Family archive")
# People filed into a folder of their own choosing instead of by-person\<name>\.
PERSON_FOLDERS = {k: _path(v) for k, v in CONFIG.get("people", {}).get("folders", {}).items()}
ELSEWHERE = set(PERSON_FOLDERS)
# Review answers that are a description rather than a folder name -> the folder name.
ALIASES = dict(CONFIG.get("people", {}).get("aliases", {}))
# The folder that collects faces the family has looked at and could not name.
NO_NAME_FOLDER = CONFIG.get("archive", {}).get("no_name_folder", "Без имени")

# Ledgers describing the album, kept in the repo rather than loose inside by-person\.
# They are version-controlled and append-only; the album holds photographs.
LOGS = REPO / "data"

# The columns of the ledgers several scripts append to. A new archive starts with none of
# these files, so an appender cannot take its header from the first row it reads.
LEDGER_FIELDS = {
    "album_thinned.csv": ["date", "deleted_path", "bytes", "sha256", "origin", "group",
                          "reason"],
    "album_print_vs_rescan.csv": ["group", "kind", "confidence", "blobs",
                                  "blob_area_per_1e4", "folder", "reference", "other",
                                  "pass_reference", "pass_other", "cross_pass", "residual",
                                  "inliers"],
    "orientation_corrections.csv": ["date", "file", "old_orientation", "new_orientation",
                                    "sha256_before", "sha256_after", "taken_from", "origin",
                                    "origin_orientation", "note"],
    "duplicate_review_log.csv": ["date", "negative", "section", "scan_a", "scan_b", "answer",
                                 "machine", "residual", "resolution", "keeper", "dropped"],
}


def read_ledger(name: str) -> tuple[list[dict], list[str]]:
    """(rows, header) of data\\<name>. A missing or empty file gives its canonical header."""
    p = LOGS / name
    if not p.exists():
        return [], list(LEDGER_FIELDS.get(name, []))
    with p.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return rows, list(rd.fieldnames or LEDGER_FIELDS.get(name, []))
