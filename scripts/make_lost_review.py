# Builds artifacts\reviews\lost-review.html — a review page over the photos that are
# in the face database but have NO row in data\main_import_log.csv, i.e. the
# ones that never reached any person folder and are therefore invisible to
# unknown-review.html (which enumerates folders, not photos).
#
#   .venv\Scripts\python.exe scripts\make_lost_review.py
#
# Re-runnable: photos that gain a log row simply drop out on the next build.
import os, re, sys, csv, json, base64, shutil, sqlite3, datetime, collections
import numpy as np, cv2

# ROOT = the repo; the photos live elsewhere (archive.toml).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from _paths import ALBUM_ROOTS, BP, LOGS
BASE = str(BP)
from _paths import DB  # noqa: E402
OUT = os.path.join(ROOT, 'artifacts', 'reviews', 'lost-review.html')
THUMB_W = 300          # whole-photo thumbnail
FACE = 74              # face crop
MAX_FACES = 8          # montage sheets have up to 93 faces; show the biggest few

nrm = lambda m: m / np.linalg.norm(m)
imread = lambda p: cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def jpg(img, q):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


# ---------- who is already known ------------------------------------------
log = list(csv.DictReader(open(os.path.join(LOGS, 'main_import_log.csv'), encoding='utf-8-sig')))
in_log = {r['source'] for r in log}

# "No main_import_log row" is not "lost" once more than one record system files photos:
# a born-digital photo filed straight into the album is recorded only in MANIFEST.csv.
_man = os.path.join(ROOT, 'data', 'MANIFEST.csv')
if os.path.exists(_man):
    in_log |= {r['origin_path'] for r in csv.DictReader(open(_man, encoding='utf-8-sig'))
               if r['origin_path']}
# Photos in another album root are curated there by deliberate rule, not lost.
_personal = tuple(str(r) for r in ALBUM_ROOTS[1:])

# Faces the family named directly in the lost-review pass. Without this the page
# re-lists every photograph they just finished answering.
_lost_log = os.path.join(ROOT, 'data', 'lost_review_log.csv')
if os.path.exists(_lost_log):
    in_log |= {r['source'] for r in csv.DictReader(open(_lost_log, encoding='utf-8-sig'))
               if r['source']}

# Photographs the family examined and could not identify. This page selects by the
# ABSENCE of a log row, so "не знаю" is not a state it can otherwise represent: without
# this, every photograph they said no to comes back on every rebuild, for ever.
_lost_declined = os.path.join(ROOT, 'data', 'lost_declined.csv')
if os.path.exists(_lost_declined):
    in_log |= {r['photo'] for r in csv.DictReader(open(_lost_declined, encoding='utf-8-sig'))
               if r['photo']}

_split_log = os.path.join(ROOT, 'data', 'split_import_log.csv')
if os.path.exists(_split_log):
    _recorded = {int(r['cluster']) for r in csv.DictReader(open(_split_log, encoding='utf-8-sig'))
                 if r['cluster'].isdigit()}
    _db = sqlite3.connect(DB)
    for _p, _c in _db.execute('select path, cluster_id from face'):
        if _c in _recorded:
            in_log.add(_p)
    _db.close()
live = [r for r in log if r['status'] not in ('retired', 'statue')]
owners = collections.defaultdict(collections.Counter)
for r in live:
    if r['cluster']:
        owners[int(r['cluster'])][r['person']] += 1
is_stranger = lambda p: bool(re.fullmatch(r'Person_\d+', p))
# cluster -> person from ALL the logs, not just main_import_log: otherwise the page scores
# faces only against the people the first import knew, and misses everyone named since by
# the crop, lost, rematch or borderline reviews. _archive.named_clusters() is the one
# resolution every generator shares.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _archive as _A                                                    # noqa: E402
c2p = {c: p for c, cnt in owners.items()
       for p in [max(cnt.items(), key=lambda kv: (kv[1], not is_stranger(kv[0])))[0]]}
c2p.update(_A.named_clusters())
named = sorted({p for p in c2p.values() if not is_stranger(p)})

# ---------- embeddings ------------------------------------------------------
db = sqlite3.connect(DB)
cvecs = collections.defaultdict(list)
cphotos = collections.defaultdict(set)
rows = list(db.execute('select id,path,cluster_id,bbox,det_score,age,sex,vector from face'))
for fid, path, c, bb, det, age, sex, v in rows:
    a = np.frombuffer(v, dtype=np.float32)
    cvecs[c].append(a / np.linalg.norm(a))
    cphotos[c].add(path)

pvecs = collections.defaultdict(list)
for c, p in c2p.items():
    if not is_stranger(p) and c in cvecs:
        pvecs[p] += cvecs[c]
pcent = {p: nrm(np.mean(np.stack(v), 0)) for p, v in pvecs.items()}
pnames = sorted(pcent)
P = np.stack([pcent[n] for n in pnames])

# ---------- the lost photos -------------------------------------------------
byphoto = collections.defaultdict(list)
for fid, path, c, bb, det, age, sex, v in rows:
    byphoto[path].append((fid, c, json.loads(bb), det, age, sex, v))
lost = sorted(p for p in byphoto
              if p not in in_log and not p.startswith(_personal))
print(f'{len(lost)} photos with no log row, {sum(len(byphoto[p]) for p in lost)} faces')

cards = []
for i, path in enumerate(lost, 1):
    im = imread(path)
    if im is None:
        print('UNREADABLE', path); continue
    h, w = im.shape[:2]
    s = THUMB_W / max(w, h)
    thumb = jpg(cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s)))), 62)

    faces = sorted(byphoto[path], key=lambda t: -(t[2][2] - t[2][0]))[:MAX_FACES]
    fl = []
    for fid, c, bb, det, age, sex, v in faces:
        x1, y1, x2, y2 = [int(x) for x in bb]
        m = int(0.35 * max(x2 - x1, y2 - y1))
        cr = im[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
        crop = jpg(cv2.resize(cr, (FACE, FACE)), 66) if cr.size else ''
        a = nrm(np.frombuffer(v, dtype=np.float32))
        sc = P @ a
        top = [(pnames[k], float(sc[k])) for k in np.argsort(-sc)[:3]]
        # is this cluster known elsewhere?
        hint = ''
        if c != -1:
            who = c2p.get(c)
            hint = (f'c{c} = {who}' if who else
                    f'c{c}, {len(cphotos[c])} ph, unassigned')
        else:
            hint = 'не кластеризовано'
        fl.append(dict(fid=fid, crop=crop, top=top, hint=hint, det=det,
                       px=x2 - x1, age=age, sex=sex))
    cards.append(dict(path=path, sub=os.path.basename(os.path.dirname(path)),
                      name=os.path.basename(path), thumb=thumb, faces=fl,
                      nf=len(byphoto[path]),
                      clustered=any(f[1] != -1 for f in byphoto[path])))
    if i % 40 == 0:
        print(f'  {i}/{len(lost)}', flush=True)

# photos with a clustered face are more tractable — show them first
cards.sort(key=lambda c: (not c['clustered'], -len(c['faces'])))

esc = lambda s: s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
out = []
for c in cards:
    fs = []
    for f in c['faces']:
        guesses = ' · '.join(f'{esc(n)} <b>{v:.2f}</b>' for n, v in f['top'])
        strong = ' hi' if f['top'] and f['top'][0][1] >= 0.35 else ''
        fs.append(
            f"<div class=face><img src='data:image/jpeg;base64,{f['crop']}'>"
            f"<div class=fm><div class=hint>{esc(f['hint'])} · {f['px']}px</div>"
            f"<div class='sc{strong}'>{guesses}</div>"
            f"<input class=nm data-f='{f['fid']}' placeholder='кто это…' list=people></div></div>")
    out.append(
        f"<div class=card data-p=\"{esc(c['path'])}\" data-n='{len(c['faces'])}'"
        f" data-c='{1 if c['clustered'] else 0}'>"
        f"<div class=hd><b>{esc(c['sub'])}\\{esc(c['name'])}</b>"
        f"<span class=n>{c['nf']} лиц{'' if c['nf'] == 1 else ''}</span></div>"
        f"<img class=ph src='data:image/jpeg;base64,{c['thumb']}'>"
        + ''.join(fs) +
        "<button class=sk>пропустить</button></div>")

datalist = ''.join(f"<option value='{esc(n)}'>" for n in named)
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
nclust = sum(1 for c in cards if c['clustered'])

HTML = f"""<!doctype html>
<html lang=ru><meta charset=utf-8><title>Потерянные фотографии — review</title>
<style>
:root{{color-scheme:light dark}}
body{{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:16px;
background:#f6f6f4;color:#1a1a18}}
h1{{font-size:20px;margin:0 0 4px}} .lede{{color:#555;max-width:74em;margin:0 0 12px}}
#bar{{position:sticky;top:0;background:#f6f6f4;padding:10px 0;z-index:9;
border-bottom:1px solid #ddd;margin-bottom:14px}}
#grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:9px;padding:9px 10px}}
.card.done{{opacity:.4}}
.hd{{display:flex;gap:8px;align-items:baseline;margin-bottom:6px;font-size:12px}}
.hd b{{font-family:ui-monospace,monospace;font-size:11.5px}}
.n{{margin-left:auto;color:#888;font-size:11px}}
img.ph{{width:100%;border-radius:5px;display:block;margin-bottom:8px;background:#eee}}
.face{{display:flex;gap:8px;align-items:flex-start;margin-bottom:7px}}
.face img{{width:74px;height:74px;object-fit:cover;border-radius:4px;flex:none}}
.fm{{flex:1;min-width:0}}
.hint{{font-size:10.5px;color:#999;font-family:ui-monospace,monospace}}
.sc{{font-size:11px;color:#666;margin:2px 0 4px}} .sc.hi b{{color:#2a7}}
input.nm{{width:100%;padding:3px 6px;border:1px solid #ccc;border-radius:5px;font:inherit;font-size:12px}}
button.sk{{padding:3px 9px;border:1px solid #ccc;border-radius:5px;background:#fafafa;cursor:pointer;font:inherit;font-size:12px}}
#out{{width:100%;height:160px;font-family:ui-monospace,monospace;font-size:12px;margin-top:8px;display:none}}
@media(prefers-color-scheme:dark){{
 body{{background:#16161a;color:#e8e8e6}} .card{{background:#1f1f24;border-color:#33333a}}
 #bar{{background:#16161a;border-color:#33333a}} .lede,.sc{{color:#aaa}}
 input.nm,button.sk{{background:#2a2a30;color:#e8e8e6;border-color:#44444c}}
}}
</style>
<div id=bar>
 <h1>Потерянные фотографии</h1>
 <span id=prog></span>
 <button class=sk id=exp>Экспорт</button>
 <button class=sk id=rst>Сброс</button>
 <label style="margin-left:12px"><input type=checkbox id=hide> скрыть готовые</label>
 <label style="margin-left:12px"><input type=checkbox id=onlyc> только с кластером</label>
 <textarea id=out readonly></textarea>
</div>
<p class=lede>Собрано <b>{stamp}</b>. <b>{len(cards)}</b> фотографий, которые есть в
индексе лиц, но <b>ни разу не попали в <code>by-person</code></b> — ни в одну папку,
ни в один лог. Поэтому их не видно на странице <code>unknown-review.html</code>,
которая перечисляет папки, а не снимки. У <b>{nclust}</b> из них хотя бы одно лицо
попало в кластер (такие идут первыми — их опознать легче); у остальных все лица
остались вне кластеров. Под каждым снимком — его лица, подсказка по кластеру и три
ближайших из {len(named)} известных людей. Впишите имя под нужным лицом; «Экспорт»
даёт список, который я применю одним проходом.</p>
<datalist id=people>{datalist}</datalist>
<div id=grid>
{chr(10).join(out)}
</div>
<script>
const KEY='lost-review-v1';
let D=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const cards=[...document.querySelectorAll('.card')];
const live=new Set(cards.map(c=>c.dataset.p));
let changed=false;
for(const k of Object.keys(D)){{ const p=k.split('||')[0];
  if(!live.has(p)){{delete D[k];changed=true;}} }}
if(changed) localStorage.setItem(KEY,JSON.stringify(D));
function decided(c){{ return Object.keys(D).some(k=>k.split('||')[0]===c.dataset.p); }}
function prog(){{
 const n=cards.filter(decided).length;
 document.getElementById('prog').textContent=n+' / '+cards.length+' размечено   ';
 const h=document.getElementById('hide').checked, oc=document.getElementById('onlyc').checked;
 cards.forEach(c=>{{const d=decided(c); c.classList.toggle('done',d);
  const hideIt=(h&&d)||(oc&&c.dataset.c==='0');
  c.style.display=hideIt?'none':'';}});
}}
cards.forEach(c=>{{
 c.querySelectorAll('input.nm').forEach(inp=>{{
  const k=c.dataset.p+'||'+inp.dataset.f;
  if(D[k]&&D[k]!=='SKIP') inp.value=D[k];
  inp.onchange=()=>{{ if(inp.value.trim()) D[k]=inp.value.trim(); else delete D[k];
   localStorage.setItem(KEY,JSON.stringify(D)); prog(); }};
 }});
 c.querySelector('.sk').onclick=()=>{{ D[c.dataset.p+'||SKIP']='SKIP';
  localStorage.setItem(KEY,JSON.stringify(D)); prog(); }};
}});
document.getElementById('hide').onchange=prog;
document.getElementById('onlyc').onchange=prog;
document.getElementById('exp').onclick=()=>{{
 const o=document.getElementById('out'); o.style.display='block';
 o.value=Object.entries(D).filter(([k,v])=>v&&v!=='SKIP')
   .map(([k,v])=>{{const[p,f]=k.split('||');return p+'  face '+f+' = '+v;}})
   .sort().join('\\n')||'(пока никого не назвали)';
 o.select();
}};
document.getElementById('rst').onclick=()=>{{
 if(confirm('Очистить все пометки?')){{D={{}};localStorage.removeItem(KEY);
  document.getElementById('out').style.display='none';prog();}}
}};
prog();
</script>
</html>"""

if os.path.exists(OUT):
    # No .bak here: this repo is versioned, and 'git show HEAD:artifacts/reviews/...'
    # recovers any previous build. Backup-on-write is what produced 189 stale .bak
    # files in by-person before there was version control.
    pass
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write(HTML)
print(f'{OUT}  {len(HTML)/1e6:.1f} MB  {len(cards)} photos  {nclust} with a clustered face')
