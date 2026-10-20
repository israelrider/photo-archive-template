# photo-archive-template

Tooling for turning one family's photographs — scans of prints, phone exports, an old CD —
into **an album per person**, with every decision recorded and every deletion reversible.

It holds no photographs and no names. Each archive gets its own copy of this repository,
its own `archive.toml`, and its own `data\` ledgers.

Built on [`images-agent`](https://github.com/israelrider/images-agent), which detects and clusters faces (InsightFace
`buffalo_l`) and builds the CLIP index. This repository is the curation layer on top: it
asks the family who people are, records the answers, files copies, and removes duplicates.

## The ideas everything rests on

**Origins and albums are different things.**

| | |
|---|---|
| **origin** | a file exactly as received, under `paths.archive`, one folder per delivery (`scans-2026-07\`, `phone-export\`). Never edited, never renamed, never deleted by a script except `delete_origins.py` at the very end, and only after it has hashed a backup. |
| **album** | `by-person\<Name>\` — COPIES of origins, one per person in the photograph. A photo of six people is in six folders. That duplication is intentional: co-occurrence in group photos is how people get identified. |

**Every album file can be traced to its origin** through `data\MANIFEST.csv`, resolved by
content (size, then sha256), never by filename — scanners restart their numbering in every
job, so `0001.jpg` names a different photograph in every folder.

**Three levels, and only one is redundancy** (`build_photographs.py`):

| | |
|---|---|
| **scan** | one origin file. Its copies across person folders are filing, not duplication. |
| **plate** | one physical print. Two scans of one plate — a rescan — and one can go. |
| **negative** | one photograph. Two prints of one negative are two real objects. The family decides which to keep. |

**A review page is evidence, not an instruction.** The family answers on a self-contained
HTML page (face crops embedded, answers kept in the browser), presses *Скопировать ответы /
Экспорт*, and pastes the text into `data\<review>_answers.raw.txt`. An `apply_*` script turns
that into records, a dry run first, `--apply` second. "Не знаю / пропустить" is an answer too,
and is recorded so the same question is never asked twice.

**Nothing is deleted without a record that can undo it.** Every removed album copy gets a
row in `data\album_thinned.csv` with its sha256 and its origin; `restore_thinned.py` puts it
back byte-identical.

## Setup

```powershell
cd <parent folder>                      # images-agent should sit beside this repo
git clone <this repo> my-family-archive
cd my-family-archive
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ..\images-agent[faces]      # the face / CLIP engine
copy archive.example.toml archive.toml               # then edit it
```

`archive.toml` holds everything about *this* archive: where the origins and albums live,
the face database, optional MiVOLO (age estimation, in its own venv), the origin roots, and
the people with special routing. Every option is documented in `archive.example.toml`.

Run every script from the repo root:

```powershell
.venv\Scripts\python.exe scripts\<name>.py            # dry run: reports, changes nothing
.venv\Scripts\python.exe scripts\<name>.py --apply    # does it
```

> **Privacy.** `.gitignore` keeps photographs, review pages (they embed face crops),
> `archive.toml` and `data\` out of git. If you want the archive's records under version
> control — and you should — use a **private** repository and remove those lines.

## The pipeline

### 1. Receive and index

```powershell
# zips from a scanning service -> a sibling *-extracted folder, resumable, verified
powershell -ExecutionPolicy Bypass -File scripts\Extract-Origins.ps1 -Src <zips> -Dst <zips>-extracted

# a failing CD/DVD: copy a list of files one by one, keeping only complete JPEGs
powershell -ExecutionPolicy Bypass -File scripts\Rescue-DiscFiles.ps1 -Src D:\ -Dst <folder> -WantList want.txt

# faces + clusters + the Person_NN album, with images-agent (set IMAGE_INDEX_SQLITE to paths.face_db)
images-agent faces --root <archive>
images-agent albums --root <by-person>

scripts\index_clip.py --apply            # CLIP vectors for semantic search, over the origins
```

### 2. Name people

| step | script |
|---|---|
| The family renames `Person_NN` folders to real names; record that as (person, cluster) | `import_named_folders.py` |
| Rebuild the provenance manifest (always run the pair) | `rebuild_manifest.py`, `patch_derived.py` |
| Folders still `Person_NN`: show whole photographs, ask who it is | `make_unknown_review.py` → `close_unknown_review.py <page>` |
| Photos whose faces reached no folder at all | `make_lost_review.py` → `apply_lost_review.py <parsed.csv>` |
| Unnamed clusters that resemble someone named | `make_rematch_review.py` → `apply_rematch_review.py` → `close_rematch_review.py` |
| Clusters torn between two named people | `make_borderline_review.py` → `apply_rematch_review.py --review=borderline` |
| Crops the family saved with a name on them (`Анна.png`) | `match_crops.py` |
| File the newly named faces into albums | `distribute.py [log.csv]` |
| How well is each identity supported by the images? | `confidence.py` → `data\CONFIDENCE.csv` |
| Check one identification against everything else | `verify_cluster.py <cluster>`, `check_pair.py "<A>" "<B>"`, `whois.py <clusters>` |
| Faces with nobody to name them yet | `show_unresolved.py`, `show_photo_faces.py <file>` |
| Ages in filenames that contradict each other | `check_age_consistency.py` |
| Which review pages are still open? | `stamp_reviews.py` |

### 3. Remove duplicate prints

| step | script |
|---|---|
| Find prints of one negative across the album (thumbnails → prefilter → ORB registration) | `find_same_negative_album.py` |
| Record the family's look at the 0.30–0.45 band (`data\same_negative_verdicts.csv`) | `review_album_same_negative.py --apply` |
| Rescan of one print, or two prints? (damage blobs after alignment) | `split_rescan_vs_print.py --apply` |
| Collapse into scans / plates / negatives with stable ids | `build_photographs.py` (`stable_ids.py`) |
| Ask the family pairwise, apply | `make_duplicate_review.py` → `apply_duplicate_review.py` |
| Groups of 3+ scans: ask once per group | `make_group_review.py` → `apply_group_review.py` |
| Hash-only proposals, re-measured | `check_dhash_tail.py` |
| A merge creates new pairs: judge them | `close_merged_negative.py <N…>` |
| Compare two scans by hand | `compare_album_pair.py`, `compare_origin_pair.py`, `compare_matched_scale.py` |
| Undo a thinning | `restore_thinned.py "<deleted_path>"` or `--from-audit` |

### 4. Orientation

| | |
|---|---|
| File has no EXIF block, displays sideways | `add_orientation_tag.py <person> <file> <1\|3\|6\|8>` |
| File has a WRONG orientation value | `fix_orientation_value.py <person> <file> <value>` |
| Re-apply recorded corrections to fresh copies (used by `distribute.py`) | `orientation_carry.py` |
| Does the indexer see rotated photos the right way up? | `exif_impact.py` |

Only album copies are changed — 36 bytes added or 2 bytes rewritten, pixels proved identical
— and every change goes to `data\orientation_corrections.csv`. Origins are never modified.

### 5. Ship

| | |
|---|---|
| Copy every album root to a share folder (e.g. for Google Drive), verified by sha256 | `build_share.py --dest <folder>` |
| Remove service files and withdrawn trees, after verifying them on the backup | `clean_before_upload.py --against <backup>` |
| Erase the origins from this PC — only those a backup on another drive provably holds | `delete_origins.py --against <backup>` |
| Integrity checks | `audit_origins.py`, `repoint.py` |

## Rules learned the hard way

These came out of curating a real archive of ~5,000 origins and ~3,000 album files.

**A hash finds copies; only registration finds photographs.** Two prints of one negative have
different bytes, density and damage. Register the pair (ORB + RANSAC homography, then residual
over the aligned overlap), and **mask to the middle of the frame first**: a studio mount card
is identical on every print that studio sold and will register two different children.

**Resolve origins by content, never by filename.** Name matching once bound 18% of a manifest
to the wrong file.

**Never index an album path — index the origin.** One photo in nine folders is one photo.

**Never re-run DBSCAN to add photos.** It renumbers every cluster and breaks the folder→name
mapping. Assign new faces by nearest neighbour; a ~0.55 similarity gate separates real faces
from statues and bystanders. Cluster ids belong to one database: say which.

**Absence from the album is not absence of a decision.** A photograph thinned out of a folder
looks exactly like one never filed. Every builder and distributor asks `_archive.withdrawn()`,
or it re-asks the family and re-files what they removed.

**Age prefixes come from MiVOLO, never from `face.age`.** buffalo_l's estimate tags a
ten-year-old 47. If MiVOLO cannot run, the file is named `noage_`.

**Never trust the file size or pixel count to pick a keeper.** Families often keep the
smaller scan. Detail-per-noise, measured on the lit subject, in the coarse band, is what
matters; where measurements disagree, the page says a person must decide.

**Some identities rest on testimony alone.** `confidence.py` puts them in tier D. Report
them as the family's word, not as a machine result.

**Check the tooling before the archive.** When a number looks wrong — two people suddenly
close, a cluster that "crossed a threshold" — the cause is usually a script reading the wrong
log or a stale cache, not a change in the photographs.

## Layout

| path | what |
|---|---|
| `archive.toml` | this archive's settings (copy of `archive.example.toml`) |
| `scripts\` | the tooling; `_paths.py` reads the config, `_archive.py` holds the shared logic |
| `data\` | the ledgers: who is who, what was filed, removed, rotated, and why — see `data\README.md` |
| `artifacts\reviews\` | the review pages the family answers |
| `artifacts\checks\` | contact sheets and reports |
