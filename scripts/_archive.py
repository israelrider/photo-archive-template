"""Shared loading for the review generators.

Everything that needs "who is this cluster" comes from here, so the generators cannot drift
apart on how a cluster is resolved to a person: named_clusters() reads every log that can
name a cluster, and nothing infers a name from a folder.
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from _paths import (CONFIG, PERSON_FOLDERS, expand)
from _paths import (ALBUM_ROOTS, ALIASES, ARCHIVE, ARCHIVE_NAME, BORN_DIGITAL, BP, DB,
                    ELSEWHERE, IMAGES_AGENT, EXTERNAL_ORIGINS, LEDGER_FIELDS, LOGS, MODEL, NO_NAME_FOLDER,
                    ORIGIN_ROOTS, REPO, SPLIT_SHEETS, read_ledger)

REVIEWS = REPO / "artifacts" / "reviews"

# Reviews that ask the family about whole clusters. They differ only in which clusters they
# select, so one applier and one closer serve all of them - add a review here rather than
# writing another pair of scripts, and named_clusters() and confidence.py both pick it up.
REVIEW_SETS = {
    "rematch": {
        "pages": "rematch-review-*.html",
        "answers": "rematch_review_answers.raw.txt",
        "log": "rematch_log.csv",
        "declined": "rematch_declined.csv",
    },
    "borderline": {
        "pages": "borderline-review.html",
        "answers": "borderline_review_answers.raw.txt",
        "log": "borderline_log.csv",
        "declined": "borderline_declined.csv",
    },
}

# Reviews that answer at FACE level and park the named faces in fresh 3000+ clusters, because
# the faces they ask about sit at cluster -1, which is shared by every unclustered face in the
# archive and cannot carry a name. They are deliberately NOT in REVIEW_SETS: their declined
# records are keyed by the thing they asked about -- a group of faces -- and not by a cluster,
# so every REVIEW_SETS consumer that reads `declined["cluster"]` would die on them.
FACE_LEVEL_LOGS = ("lost_review_log.csv",)

# A FACE THE FAMILY HAS ALREADY RULED ON IS NOT AN OPEN QUESTION
#
# A face-level review has verdicts that write no cluster id: `unknown` means "a person, and I
# do not know them" and `notaperson` means "that is not a person at all". Both leave the face
# at -1, so a builder that selects on cluster_id=-1 alone re-asks all of them on the next run.
#
# Keyed on FACE ID, never on group id. A group id is the lowest face id in a component, and
# components are recomputed from whatever is at -1 on the day: add a face, and the same
# people can regroup under a different id. The faces themselves are what was ruled on.
#
# A face that turns out to be furniture, a doll or a painting is ruled on, not declined: a
# declined record means a person was looked for and not identified, which would be a false
# statement about a cushion. Each file has a `face_ids` column, space-separated.
FACE_VERDICT_RECORDS = ("unknown_declined.csv", "notaperson.csv")


def ruled_on_faces(filenames=FACE_VERDICT_RECORDS) -> dict[int, str]:
    """face_id -> the record that ruled on it, for reviews whose verdicts write no cluster."""
    out: dict[int, str] = {}
    for fn in filenames:
        p = LOGS / fn
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            for fid in str(r.get("face_ids") or "").split():
                if fid.lstrip("-").isdigit():
                    out[int(fid)] = fn
    return out


def named_clusters() -> dict[int, str]:
    """cluster_id -> person, from the import logs and the reviews. Never inferred from folders."""
    out: dict[int, str] = {}
    votes: dict[int, Counter] = defaultdict(Counter)
    main = LOGS / "main_import_log.csv"
    if main.exists():
        for r in csv.DictReader(main.open(encoding="utf-8-sig")):
            c, who = (r.get("cluster") or "").strip(), (r.get("person") or "").strip()
            if c.lstrip("-").isdigit() and who and not who.startswith("Person_"):
                votes[int(c)][who] += 1
    for c, v in votes.items():
        out[c] = v.most_common(1)[0][0]
    # Clusters identified from a crop the family saved into a folder with a name on it
    # (`<name>.png`). Matched back to a face id by embedding - see match_crops.py.
    cp = LOGS / "crop_identifications.csv"
    if cp.exists():
        for r in csv.DictReader(cp.open(encoding="utf-8-sig")):
            if (r.get("person") or "").strip() and str(r["cluster"]).isdigit():
                out[int(r["cluster"])] = r["person"]

    # Named by the family in a review: the lost review names single faces and parks them in
    # clusters 3000+; the cluster reviews name whole clusters in place.
    for fn in (*FACE_LEVEL_LOGS, *(s["log"] for s in REVIEW_SETS.values())):
        p = LOGS / fn
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            c = str(r.get("cluster") or "")
            if (r.get("person") or "").strip() and c.lstrip("-").isdigit():
                out[int(c)] = r["person"]
    return out


# A FILE TAKEN OUT OF THE ALBUM ON PURPOSE IS NOT AN UNFILED FILE
#
# A review generator that asks "has this origin been filed?" by walking only the LIVE album
# tree sees a photograph that was filed and then thinned exactly like one that was never
# filed, and re-asks the family about it. An answer given to the wrong question becomes an
# instruction to undo a decision the same person made days earlier.
#
# ABSENCE FROM THE ALBUM IS NOT ABSENCE OF A DECISION. It lives here rather than in each
# generator so the next one written inherits it.
#
# (file, the column holding the ORIGIN path). Keyed on the origin, never on the album name,
# so a drifting age prefix cannot walk around it.
REMOVAL_LOGS = [("album_thinned.csv", "origin"), ("held_back.csv", "origin_path")]
# Each album root keeps what was thinned out of it in a hidden `.thinned\<person>\` tree.
WITHDRAWN_TREES = [root / ".thinned" for root in ALBUM_ROOTS]
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# `age18_v01_IMG_1764.jpg` -> `IMG_1764.jpg`. The tag is one run without an underscore;
# the origin filename after it may contain them.
#
# An album may also name by year: `1966_m05_scan 0006.jpg` is `<year>_<tag>_<origin>`, the
# year standing where the age prefix stands elsewhere. A year prefix is stripped first and a
# tag is only taken when it has the shape every tag has, `<letter><digits>`.
ALBUM_NAME = re.compile(r"^(?:Person_[\d+]+__)?(?:age\d+|noage)_[^_]+_(.+)$")
YEAR_NAME = re.compile(r"^\d{4}_(.+)$")
ROOT_TAG = re.compile(r"^[a-z]+\d+_(.+)$")


def origin_name(album_file: str) -> str:
    """The origin filename an album copy was made from, lower-cased."""
    m = ALBUM_NAME.match(album_file)
    if m:
        return m.group(1).lower()
    y = YEAR_NAME.match(album_file)
    if y:
        rest = y.group(1)
        t = ROOT_TAG.match(rest)
        return (t.group(1) if t else rest).lower()
    return album_file.lower()


def filed_or_withdrawn() -> set[str]:
    """Origin filenames the album holds, OR held and gave up on purpose. Lower-cased.

    A generator asking "is this origin unfiled?" must ask this, not the live tree alone.
    """
    out: set[str] = set()
    for root in ALBUM_ROOTS:
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(IMG_EXT):
                    out.add(origin_name(f))
                    out.add(f.lower())
    for fn, col in REMOVAL_LOGS:
        path = LOGS / fn
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            v = (r.get(col) or "").strip()
            if v:
                out.add(PureWindowsPath(v).name.lower())
    for tree in WITHDRAWN_TREES:
        for d, _, fs in os.walk(tree):
            for f in fs:
                if f.lower().endswith(IMG_EXT):
                    out.add(origin_name(f))
                    out.add(f.lower())
    return out


# A COPY REMOVED ON PURPOSE IS FILED, AND A DISTRIBUTOR MUST NOT PUT IT BACK
#
# filed_or_withdrawn() above answers "has this origin been dealt with anywhere?" and is what a
# review BUILDER needs. A distributor needs the narrower question, because one photograph can
# legitimately live in one person's folder and have been thinned out of another's: not "was
# this removed?" but "was this removed FROM THE FOLDER I am about to copy it into?"
#
# Hence the key is (folder, origin filename). The origin, never the album name: an age prefix
# is MiVOLO's answer on the day and can drift between runs, so a check keyed on the album name
# walks straight around itself the moment an age changes.
#
# EVERY DISTRIBUTOR IS IDEMPOTENT ONLY AGAINST THE LIVE FOLDER WITHOUT THIS. They skip a
# destination that already exists -- but a thinned file does not exist there any more, it is in
# .thinned\, so a re-run re-files everything ever thinned, beside the keeper that beat it.
#
# (file, origin column, how to find the folder, the column it is in, keeper column, why column)
REMOVAL_RECORDS = [
    ("album_thinned.csv", "origin", "parent", "deleted_path", "", "reason"),
    # held_back is a family decision that a photograph is NOT FILED AND NOT DELETED, so it is
    # withheld from EVERY folder, not one: its keeper_album_file names the file that WON, which
    # is a different photograph in a folder the held-back one was never in.
    ("held_back.csv", "origin_path", "any", "", "", "note"),
]
ANY_FOLDER = "*"


def withdrawn() -> dict[tuple[str, str], tuple[str, str]]:
    """{(folder, origin filename): (which record, why)} for every copy taken out on purpose.

    Folder ANY_FOLDER means "withheld from every folder". Origin names are lower-cased, so a
    caller must lower-case the name it looks up. Read from the logs AND from the withdrawn
    trees: the logs are the record and carry the reason, the trees are the belt-and-braces so
    a file moved by hand without a row still counts as a decision rather than as an accident.
    """
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for fn, origin_col, how, folder_col, keeper_col, why_col in REMOVAL_RECORDS:
        path = LOGS / fn
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8-sig")):
            origin = (r.get(origin_col) or "").strip()
            if not origin:
                continue
            if how == "any":
                folder = ANY_FOLDER
            else:
                folder = PureWindowsPath((r.get(folder_col) or "").strip()).parent.name
            if not folder:
                continue
            why = (r.get(why_col) or "").strip().replace("\n", " ")
            keeper = (r.get(keeper_col) or "").strip() if keeper_col else ""
            if keeper:
                why = f"keeper {keeper}; {why}"
            out.setdefault((folder, PureWindowsPath(origin).name.lower()),
                           (fn, why or "removed on purpose, no reason recorded"))
    for tree in WITHDRAWN_TREES:
        if not tree.is_dir():
            continue
        for person_dir in tree.iterdir():
            if not person_dir.is_dir():
                continue
            for f in person_dir.iterdir():
                if f.suffix.lower() in IMG_EXT:
                    out.setdefault((person_dir.name, origin_name(f.name)),
                                   (f"{tree.name}\\", "present with no log row"))
    return out


def withheld_from(w: dict, folder: str, origin: str) -> tuple[str, str] | None:
    """The (record, why) withholding this origin from this folder, or None. See withdrawn()."""
    name = PureWindowsPath(origin).name.lower()
    return w.get((folder, name)) or w.get((ANY_FOLDER, name))

def load_faces():
    """(clusters, paths, bboxes, dets, M) with M L2-normalised. Excludes noise."""
    conn = sqlite3.connect(DB)
    cl, paths, bbs, dets, vecs = [], [], [], [], []
    for c, p, bb, det, v in conn.execute(
        "select cluster_id, path, bbox, det_score, vector from face "
        "where model=? and cluster_id is not null and cluster_id!=-1", (MODEL,)):
        cl.append(c); paths.append(p); bbs.append(json.loads(bb)); dets.append(det)
        vecs.append(np.frombuffer(v, dtype=np.float32))
    conn.close()
    M = np.vstack(vecs)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return np.array(cl), paths, bbs, dets, M


def person_centroids(cl, M, named):
    people = sorted(set(named.values()))
    mats = []
    for p in people:
        idx = np.isin(cl, [k for k, v in named.items() if v == p])
        m = M[idx].mean(axis=0)
        mats.append(m / np.linalg.norm(m))
    return people, np.vstack(mats)


def cluster_centroid(cl, M, c):
    m = M[cl == c].mean(axis=0)
    return m / np.linalg.norm(m)


_img_cache: dict[str, object] = {}


def crop(path: str, bbox, px: int = 96, pad: float = 0.4, q: int = 72) -> str:
    """Base64 JPEG of one face. Caches decoded images: a print holding nine faces
    would otherwise be decoded nine times, and some are 30 MB."""
    im = _img_cache.get(path)
    if im is None:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if len(_img_cache) > 24:
            _img_cache.clear()
        _img_cache[path] = im
    if im is None:
        return ""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    m = int(pad * max(x2 - x1, y2 - y1))
    h, w = im.shape[:2]
    cr = im[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
    if cr.size == 0:
        return ""
    ok, buf = cv2.imencode(".jpg", cv2.resize(cr, (px, px)), [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def page(title: str, h1: str, lede: str, body: str, store_key: str, datalist: str = "") -> str:
    """Shared shell: sticky progress bar, per-card name box, localStorage, export."""
    return f"""<!doctype html><html lang=ru><meta charset=utf-8><title>{esc(title)}</title>
<style>
:root{{color-scheme:light dark;--bg:#f6f6f4;--card:#fff;--ink:#1a1a18;--mut:#666;--bd:#ddd;--acc:#33517A}}
@media(prefers-color-scheme:dark){{:root{{--bg:#17181A;--card:#212327;--ink:#E4E3DE;--mut:#A2A49F;--bd:#3A3D42;--acc:#7FA3D6}}}}
body{{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--ink)}}
h1{{font-size:20px;margin:0 0 4px}} .lede{{color:var(--mut);max-width:74em;margin:0 0 14px;white-space:pre-line}}
#bar{{position:sticky;top:0;background:var(--bg);padding:9px 0;z-index:9;border-bottom:1px solid var(--bd);margin-bottom:14px}}
.card{{background:var(--card);border:1px solid var(--bd);border-radius:9px;padding:10px 12px;margin-bottom:12px}}
.card.done{{opacity:.45}}
.hd{{display:flex;gap:10px;align-items:baseline;margin-bottom:7px}}
.n{{color:var(--mut);font-size:12px}} .cl{{margin-left:auto;color:var(--mut);font-size:11px;font-family:ui-monospace,monospace}}
.row{{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}}
.grp{{border:1px solid var(--bd);border-radius:6px;padding:6px 7px}}
.grp h4{{margin:0 0 4px;font-size:11px;font-weight:600;color:var(--mut)}}
.grp img{{width:74px;height:74px;object-fit:cover;border-radius:4px;margin-right:2px}}
.sc{{font-size:12px;color:var(--mut);margin:6px 0 3px}} .sc b{{color:var(--ink)}}
.hi b{{color:#2a7}}
.co{{font-size:11.5px;color:var(--acc);margin-bottom:7px}}
input.nm{{width:60%;max-width:340px;padding:5px 7px;border:1px solid var(--bd);border-radius:5px;font:inherit;background:var(--bg);color:var(--ink)}}
button{{padding:5px 12px;border:1px solid var(--bd);background:var(--card);color:var(--ink);border-radius:5px;cursor:pointer;margin-left:6px}}
#out{{white-space:pre;font-family:ui-monospace,monospace;font-size:12px;background:var(--card);
border:1px solid var(--bd);border-radius:6px;padding:12px;margin-top:12px;display:none}}
</style>
<h1>{esc(h1)}</h1><p class=lede>{lede}</p>
<div id=bar><span id=cnt></span><button onclick=exp()>показать заполненное</button></div>
{body}
<datalist id=people>{datalist}</datalist>
<pre id=out></pre>
<script>
const K='{store_key}';
const st=JSON.parse(localStorage.getItem(K)||'{{}}');
const cards=[...document.querySelectorAll('.card')];
cards.forEach(c=>{{
  const i=c.querySelector('input.nm'), id=c.dataset.id;
  if(!i) return;
  if(st[id]){{i.value=st[id];c.classList.add('done');}}
  i.addEventListener('input',()=>{{st[id]=i.value;localStorage.setItem(K,JSON.stringify(st));
    c.classList.toggle('done',!!i.value.trim());cnt();}});
  const b=c.querySelector('button.sk');
  if(b) b.onclick=()=>{{st[id]='—';localStorage.setItem(K,JSON.stringify(st));c.classList.add('done');cnt();}};
}});
function cnt(){{document.getElementById('cnt').textContent=
  Object.values(st).filter(v=>v&&v.trim()&&v!=='—').length+' назначено · '+
  Object.values(st).filter(v=>v==='—').length+' пропущено · из '+cards.length+'  ';}}
function exp(){{const o=document.getElementById('out');
  const L=Object.entries(st).filter(([k,v])=>v&&v.trim()&&v!=='—').map(([k,v])=>k+'\\t'+v.trim());
  o.textContent=L.length?L.join('\\n'):'ничего не заполнено';o.style.display='block';
  o.scrollIntoView({{behavior:'smooth'}});}}
cnt();
</script>"""
