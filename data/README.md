# data\ — the archive's ledgers

Every file here is written by a script and read by others. They are the archive's memory:
the photographs can be regenerated from the origins, these cannot. They start empty; each
is created the first time the script that owns it runs with `--apply`.

Ledgers are **append-only by convention**: a decision is never edited away, it is superseded
by a later row, so the history of how an identification was reached stays legible.

| file | written by | what a row means |
|---|---|---|
| `MANIFEST.csv` | `rebuild_manifest.py`, `patch_derived.py`, `distribute.py` | album file → person, relation (`copy`, `derived`, `born-digital`, `external`), origin path |
| `main_import_log.csv` | `import_named_folders.py` | a file in a folder the family renamed: person, cluster |
| `lost_review_log.csv` | `apply_lost_review.py` | one face named by the family; parked in a 3000+ cluster per person |
| `rematch_log.csv`, `borderline_log.csv` | `apply_rematch_review.py` | a whole cluster named by the family, with the embedding score beside it |
| `*_declined.csv`, `unknown_declined.csv` | `close_*_review.py` | shown to the family, and nobody recognised them — a finding, not a backlog |
| `notaperson.csv` | by hand | face ids that are not people (a doll, a cushion, a painting); column `face_ids` |
| `crop_identifications.csv` | `match_crops.py` | a crop the family saved with a name, matched to a cluster |
| `CONFIDENCE.csv` | `confidence.py` | tier A–D per person: how much image evidence supports the identity |
| `album_same_negative.csv` | `find_same_negative_album.py`, `review_album_same_negative.py` | groups of album files that are one negative |
| `same_negative_verdicts.csv` | by hand | `group,verdict,file,note,reviewed`: reject / split / confirm a proposed group |
| `same_negative_controls.csv` | by hand | `a,b`: origin pairs known to be one negative — recall checks for the finder |
| `album_print_vs_rescan.csv` | `split_rescan_vs_print.py`, appliers | a pair of scans of one negative: same print (rescan) or two prints |
| `photographs.csv`, `scan_ids.csv`, `retired_ids.csv` | `build_photographs.py` | scan → plate → negative, with ids that never move or get reused |
| `duplicate_review_*.csv`, `group_review_*` | duplicate / group reviews | the answers and what was done with them |
| `album_thinned.csv` | appliers | an album copy removed: path, sha256, origin, keeper, why. `restore_thinned.py` reads it |
| `held_back.csv` | by hand | a photograph the family chose not to file anywhere and not to delete |
| `orientation_corrections.csv` | orientation tools | an album copy whose EXIF tag was fixed, with before/after sha256 |
| `crop_origins.csv` | by hand | `album_file,origin,relation`: album files that are crops or backs of a print |
| `renamed_ages.csv` | optional | `person,file,buffalo,mivolo`: both models' ages, for `check_age_consistency.py` |
| `share_build_log.csv`, `cleaned_before_upload.csv`, `origins_erased.csv` | shipping scripts | what was copied, removed, erased, and the twin it was verified against |
| `annotations\` | the family | crops saved with a name (`<folder> — <who>.png`) |
| `notes\` | the family | anything they wrote down about the photographs |

`*_answers.raw.txt` files are the family's pasted answers, kept verbatim: the raw evidence
each applier reads.
