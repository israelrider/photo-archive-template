"""Stamp every review page with what is still true about it.

A review page is a snapshot of a question. Once it is answered the file keeps saying
"fill this in" forever, and its lede keeps claiming things that have since stopped being
true - that nothing has been indexed, that a question is open, that a cluster is
recorded as somebody. Anyone opening the directory later has no way to tell a live
question from a finished one.

This reads the logs and writes a banner into each page: how many of its cards are answered,
how many were examined and left unidentified, how many are genuinely still open, and which
individual cards are dead. Idempotent - the banner is stripped and rebuilt each run.

Nothing else in the page is touched. The cards stay exactly as the family saw them, because
close_rematch_review.py reads them back to know what was asked.

    .venv\\Scripts\\python.exe scripts\\stamp_reviews.py
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import _archive as A

MONTHS = {1: "января", 2: "февраля", 3: "марта", 4: "апреля", 5: "мая", 6: "июня",
          7: "июля", 8: "августа", 9: "сентября", 10: "октября", 11: "ноября",
          12: "декабря"}
CSS = """<style>
.rvstatus{border:1px solid var(--bd);border-left:4px solid var(--acc);border-radius:6px;
padding:9px 12px;margin:0 0 14px;font-size:13px;background:var(--card)}
.rvstatus b{font-size:13.5px} .rvstatus .dead{color:var(--mut);font-size:12px}
.rvstatus.closed{border-left-color:#2a7}
</style>"""
MARK = "<!--rvstatus-->"


def ru_date(s: str) -> str:
    d = datetime.strptime(s, "%Y-%m-%d")
    return f"{d.day} {MONTHS[d.month]} {d.year}"


def main() -> None:
    named = A.named_clusters()
    declined: dict[int, str] = {}
    for name, s in A.REVIEW_SETS.items():
        p = A.REPO / "data" / s["declined"]
        if p.exists():
            for r in csv.DictReader(p.open(encoding="utf-8-sig")):
                declined[int(r["cluster"])] = r.get("method", "")

    # which review named a cluster, so a page can say who answered it
    answered_in: dict[int, str] = {}
    for name, s in A.REVIEW_SETS.items():
        p = A.REPO / "data" / s["log"]
        if p.exists():
            for r in csv.DictReader(p.open(encoding="utf-8-sig")):
                if (r.get("person") or "").strip():
                    answered_in[int(r["cluster"])] = name

    # Which folders are still unidentified. When `Unknown\` is flattened its folders live
    # on in the filenames -- `Person_258+259+260__m02_...jpg` is one photograph holding three
    # of them. Reading only directories here would report them all as merged into other
    # albums, which is the opposite of what happened. Directories are still
    # read too, so the check survives if any are ever filed back.
    unk = set()
    if (A.BP / "Unknown").is_dir():
        for p in (A.BP / "Unknown").iterdir():
            if p.is_dir():
                unk.add(p.name)
            elif p.is_file() and "__" in p.name:
                head = p.name.split("__")[0]
                if head.startswith("Person_"):
                    unk |= {f"Person_{n}" for n in head[len("Person_"):].split("+")}
    folder_cl = defaultdict(set)
    for r in csv.DictReader((A.LOGS / "main_import_log.csv").open(encoding="utf-8-sig")):
        if r["person"].startswith("Person_") and r["cluster"].lstrip("-").isdigit():
            folder_cl[r["person"]].add(int(r["cluster"]))

    npl = A.REPO / "data" / "new_photos_log.csv"
    new_photos = {(r["photo"], r["face_index"]): r for r in
                  csv.DictReader(npl.open(encoding="utf-8-sig"))} if npl.exists() else {}

    for f in sorted(A.REVIEWS.glob("*.html")):
        h = f.read_text(encoding="utf-8")
        h = re.sub(rf"{MARK}.*?{MARK}", "", h, flags=re.S)
        h = h.replace(CSS, "")
        ids = re.findall(r"data-id='([^']+)'", h)
        bits, dead, closed = [], [], False

        cs = [int(i[1:]) for i in ids if i.startswith("c") and i[1:].isdigit()]
        if cs:
            nm = [c for c in cs if c in named]
            dc = [c for c in cs if c not in named and c in declined]
            op = [c for c in cs if c not in named and c not in declined]
            closed = not op
            bits.append(f"{len(cs)} карточек: <b>{len(nm)} названо</b>, "
                        f"{len(dc)} рассмотрено и не опознано, "
                        f"<b>{len(op)} открыто</b>")
            dead = [f"c{c}" for c in nm + dc]
            # the page's own suggestion versus what was actually recorded
            wrong = []
            for c in nm:
                m = re.search(rf"data-id='c{c}'.*?эталон — ([^<]+)</h4>", h, re.S)
                if m and m.group(1).strip() != named[c]:
                    wrong.append(f"c{c} → {A.esc(named[c])}")
            if wrong:
                bits.append("подсказка на странице разошлась с ответом: " + ", ".join(wrong))

        ps = [i for i in ids if i.startswith("Person_")]
        if ps:
            gone = [p for p in ps if p not in unk]
            res = [p for p in ps if p not in gone
                   and any(c in named for c in folder_cl.get(p, ()))]
            # folder-level "пропустить", recorded by close_unknown_review.py, plus folders
            # whose clusters were declined in a cluster-level review
            _ud = A.REPO / "data" / "unknown_declined.csv"
            skipped = {r["folder"] for r in csv.DictReader(_ud.open(encoding="utf-8-sig"))} \
                if _ud.exists() else set()
            dec = [p for p in ps if p not in gone and p not in res
                   and (p in skipped or any(c in declined for c in folder_cl.get(p, ())))]
            op = len(ps) - len(gone) - len(res) - len(dec)
            closed = op == 0
            bits.append(f"{len(ps)} папок: <b>{len(gone)} слито в личные альбомы</b>, "
                        f"{len(res)} опознано в другом обзоре, "
                        f"{len(dec)} рассмотрено и не опознано, <b>{op} открыто</b>")
            dead = gone + res + dec

        ph = [i for i in ids if "#" in i]
        if ph:
            done = [i for i in ph if (i.rsplit("#", 1)[0], i.rsplit("#", 1)[1]) in new_photos]
            closed = False
            bits.append(f"{len(ph)} лиц: <b>{len(done)} опознано семьёй</b>, "
                        f"{len(ph) - len(done)} без имени")
            bits.append("снимки уже в архиве и разложены по папкам — "
                        "текст ниже про «ничего не скопировано» устарел")
            dead = done

        # newbatch-review is face-level too, but keys on data-id='f<face id>' - the fourth id
        # form in this directory. Each one that has since been named goes into
        # lost_review_log.csv, because that is where apply_lost_review.py puts face answers.
        face_ids = {int(i[1:]) for i in ids if i.startswith("f") and i[1:].isdigit()}
        if face_ids:
            ll = A.REPO / "data" / "lost_review_log.csv"
            done = {int(r["face_id"]) for r in csv.DictReader(ll.open(encoding="utf-8-sig"))} \
                if ll.exists() else set()
            nd = A.REPO / "data" / "newbatch_declined.csv"
            skipped = {int(r["face_id"]) for r in csv.DictReader(nd.open(encoding="utf-8-sig"))} \
                if nd.exists() else set()
            op = face_ids - done - skipped
            closed = not op
            bits.append(f"{len(face_ids)} лиц: <b>{len(face_ids & done)} названо</b>, "
                        f"{len(face_ids & skipped)} рассмотрено и не опознано, "
                        f"<b>{len(op)} открыто</b>")
            dead = [f"f{i}" for i in sorted(face_ids & (done | skipped))]

        # lost-review predates the shared page shell and keys its inputs on data-f=<face id>,
        # not data-id. Counting only data-id read it as a page with no cards at all and
        # called it a finished record - while it holds 486 unanswered faces. A page that
        # uses a different id attribute is not a page with nothing in it.
        fids = {int(x) for x in re.findall(r"data-f='(\d+)'", h)}
        if fids:
            ll = A.REPO / "data" / "lost_review_log.csv"
            done = {int(r["face_id"]) for r in csv.DictReader(ll.open(encoding="utf-8-sig"))} \
                if ll.exists() else set()
            op = fids - done
            closed = not op
            bits.append(f"{len(fids)} лиц: <b>{len(fids & done)} отвечено</b>, "
                        f"<b>{len(op)} открыто</b>")
            bits.append("страница пересобрана после применения ответов, "
                        "поэтому отвеченные с неё уже ушли")

        if not ids and not fids and not face_ids:
            bits.append("карточек для заполнения нет")
            closed = True

        head = ("<b>Закрыто.</b> " if closed else "<b>Открыто.</b> ")
        body = " · ".join(bits)
        if dead:
            body += (f"<div class=dead>уже отвечено, заполнять не нужно: "
                     f"{A.esc(', '.join(sorted(dead)))}</div>")
        banner = (f"{MARK}{CSS}<div class='rvstatus{' closed' if closed else ''}'>"
                  f"{head}{body}<div class=dead>состояние на "
                  f"{ru_date(datetime.now().strftime('%Y-%m-%d'))}</div></div>{MARK}")
        h = re.sub(r"(<h1>.*?</h1>)", lambda m: m.group(1) + banner, h, count=1, flags=re.S)
        f.write_text(h, encoding="utf-8")
        print(f"{f.name:<28} {'ЗАКРЫТО' if closed else 'открыто':<8} {body[:96]}")


if __name__ == "__main__":
    main()
