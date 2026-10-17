"""Build the CLIP index over every photograph in the archive.

The face index (`images-agent faces`) and the CLIP index are separate: semantic search
works over nothing until this has run.

Indexes the ORIGIN ROOTS, never by-person: the album is copies, and one photograph in nine
folders must not become nine rows. Uses the model pair images-agent is configured with
(its .env: CLIP_VISION_EMBED_MODEL / CLIP_TEXT_EMBED_MODEL).

Resumable and idempotent: a path already embedded at the same mtime and size is skipped,
so an interrupted run resumes where it stopped. Only the embedding table is touched -
the face table is never read or written.

    .venv\\Scripts\\python.exe scripts\\index_clip.py [--apply] [--limit=N]
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import _archive as A

from _paths import IMAGES_AGENT  # noqa: E402
sys.path.insert(0, str(IMAGES_AGENT))
from images_agent import config                                        # noqa: E402
from images_agent.clip_pipeline import embed_images_batched            # noqa: E402
from images_agent.index_store import init_schema, upsert_embedding     # noqa: E402

ROOTS = A.ORIGIN_ROOTS
EXT = {".jpg", ".jpeg", ".png"}
APPLY = "--apply" in sys.argv
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--limit=")), 0)


def main() -> None:
    pair = config.clip_model_pair_key()
    if "|" not in pair or pair.startswith("|"):
        raise SystemExit("CLIP model env vars are not set - source images-agent\\.env first")
    print(f"model pair: {pair}")

    photos = sorted(p for root in ROOTS for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXT)
    print(f"{len(photos)} photographs across {len(ROOTS)} origin roots")

    conn = sqlite3.connect(A.DB)
    init_schema(conn)
    have = {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
        "select path, mtime_ns, size, model_pair from embedding")}
    todo = []
    for p in photos:
        st = p.stat()
        h = have.get(str(p))
        if h and h[0] == st.st_mtime_ns and h[1] == st.st_size and h[2] == pair:
            continue
        todo.append(p)
    print(f"already embedded: {len(photos) - len(todo)}   to do: {len(todo)}")
    if LIMIT:
        todo = todo[:LIMIT]
        print(f"--limit: {len(todo)} this run")

    if not APPLY:
        print("\nDRY RUN. Re-run with --apply to embed and write.")
        return
    if not todo:
        print("nothing to do")
        return

    if have:
        bak = A.DB.with_suffix(f".sqlite.bak-{datetime.now():%Y%m%d-%H%M%S}")
        shutil.copy2(A.DB, bak)
        print(f"backed up -> {bak.name}")

    ok = fail = 0
    failed: list[Path] = []
    t0 = datetime.now()
    for i, (pth, vec) in enumerate(embed_images_batched(todo, batch_size=8), 1):
        if vec is None:
            print(f"   FAILED {pth}")
            failed.append(pth)
            fail += 1
        else:
            st = pth.stat()
            upsert_embedding(conn, path=str(pth), mtime_ns=st.st_mtime_ns, size=st.st_size,
                             dim=int(vec.shape[0]), model_pair=pair, vector=vec.tobytes())
            ok += 1
        if i % 50 == 0 or i == len(todo):
            conn.commit()
            el = (datetime.now() - t0).total_seconds()
            rate = i / el if el else 0
            left = (len(todo) - i) / rate if rate else 0
            print(f"  {i}/{len(todo)}  ok={ok} fail={fail}  {rate:.1f}/s  "
                  f"~{left/60:.0f} min left", flush=True)
    conn.commit()

    # A large-format scan can be 17866x10320 - 184 megapixels - and PIL refuses it as a
    # decompression bomb. It is not corrupt; OpenCV reads it. CLIP resizes everything to
    # 224x224 anyway, so a downscaled copy carries the same information for this purpose.
    # The row is written under the ORIGINAL path: the vector describes that photograph,
    # and nothing should have to know a temporary file ever existed.
    if failed:
        import cv2
        import numpy as np
        tmp = A.REPO / ".clip-retry"
        tmp.mkdir(exist_ok=True)
        retry, small = [], {}
        for pth in failed:
            im = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                print(f"   unreadable even by OpenCV: {pth}")
                continue
            h, w = im.shape[:2]
            s = 2000 / max(h, w)
            if s < 1:
                im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            q = tmp / (pth.stem + ".jpg")
            cv2.imwrite(str(q), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            small[q] = pth
            retry.append(q)
        print(f"\nretrying {len(retry)} oversized scan(s) downscaled to 2000px")
        for q, vec in embed_images_batched(retry, batch_size=4):
            orig = small[q]
            if vec is None:
                print(f"   STILL FAILED {orig}")
                continue
            st = orig.stat()
            upsert_embedding(conn, path=str(orig), mtime_ns=st.st_mtime_ns, size=st.st_size,
                             dim=int(vec.shape[0]), model_pair=pair, vector=vec.tobytes())
            ok += 1
            fail -= 1
            print(f"   recovered {orig.name}")
        conn.commit()
        shutil.rmtree(tmp, ignore_errors=True)

    n = conn.execute("select count(*) from embedding").fetchone()[0]
    print(f"\nembedded {ok}, failed {fail}. embedding table now holds {n} rows.")
    for root in ROOTS:
        r = conn.execute("select count(*) from embedding where path like ?",
                         (str(ARCHIVE / root) + "%",)).fetchone()[0]
        t = sum(1 for p in (ARCHIVE / root).rglob("*")
                if p.is_file() and p.suffix.lower() in EXT)
        print(f"   {root:<30} {r}/{t}")
    conn.close()


if __name__ == "__main__":
    main()
