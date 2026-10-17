r"""Judge two album scans of one photograph: same print or two prints, and which is better.

    .venv\Scripts\python.exe scripts\compare_album_pair.py <person folder> <file a> <file b>

Measures INSIDE THE PICTURE AREA ONLY. The white paper margin is the sharpest edge in a
scan and it dominated the edge mask, which reversed one verdict before this was fixed.
"""
import os,sys,hashlib,csv
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, cv2
from PIL import Image, ImageOps
import split_rescan_vs_print as S
SP=os.environ.get('COMPARE_OUT') or os.path.join(os.environ.get('TEMP','.'),'album-pair-compare')
os.makedirs(SP,exist_ok=True)
from _paths import BP as _BP, LOGS
BP = str(_BP)
MAN=str(LOGS / 'MANIFEST.csv')
person=sys.argv[1]; names=sys.argv[2:]
man={r['album_file'].replace('/',os.sep):r['origin_path'] for r in csv.DictReader(open(MAN,encoding='utf-8-sig',newline=''))}
def h(p):
    x=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): x.update(c)
    return x.hexdigest()
paths=[os.path.join(BP,person,n) for n in names]
for p,n in zip(paths,names):
    if not os.path.exists(p): print('MISSING: %s'%p); sys.exit(1)
    im=Image.open(p); o=man.get(os.path.join(person,n),'?')
    print('%s\n   %dx%d = %.1f MP | %d bytes\n   origin: %s   identical: %s'%(
        p,im.size[0],im.size[1],im.size[0]*im.size[1]/1e6,os.path.getsize(p),o,os.path.exists(o) and h(p)==h(o)))
print()
loaded={p:S.load(p) for p in paths}
import itertools
for a,b in itertools.combinations(paths,2):
    r=S.damage(loaded[a],loaded[b])
    if r is None: print('%s vs %s -> undecided'%(os.path.basename(a),os.path.basename(b))); continue
    blobs,area=r
    kind='separate-print' if blobs>=S.BLOB_MIN or area>=S.BLOB_AREA_MIN else 'rescan'
    print('%-40s vs %-40s  %2d blobs, area %5.1f -> %s (%s)'%(os.path.basename(a)[:40],os.path.basename(b)[:40],blobs,area,kind,S.confidence(blobs)))
def full(p): return np.asarray(ImageOps.exif_transpose(Image.open(p)).convert('L'))
def small(a,W=1000): return cv2.resize(a,(W,int(a.shape[0]*W/a.shape[1])),interpolation=cv2.INTER_AREA)
orb=cv2.ORB_create(6000); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
I=[full(p) for p in paths]; Sm=[small(a) for a in I]
K=[orb.detectAndCompute(s,None) for s in Sm]; sc=[I[k].shape[1]/Sm[k].shape[1] for k in range(len(I))]
W=[I[0]]
for j in range(1,len(I)):
    (ka,da),(kb,db)=K[0],K[j]
    m=sorted(bf.match(da,db),key=lambda x:x.distance)[:900]
    s=np.float32([ka[x.queryIdx].pt for x in m]).reshape(-1,1,2)
    d=np.float32([kb[x.trainIdx].pt for x in m]).reshape(-1,1,2)
    H,mask=cv2.findHomography(s,d,cv2.RANSAC,4.0)
    Hf=np.diag([sc[j],sc[j],1.0])@H@np.diag([1/sc[0],1/sc[0],1.0])
    W.append(cv2.warpPerspective(I[j],np.linalg.inv(Hf),(I[0].shape[1],I[0].shape[0])))
    print('   aligned %s: %d inliers'%(names[j],int(mask.sum())))
pa=S.picture_area(small(I[0],1400))
pa=cv2.resize(pa.astype(np.uint8),(I[0].shape[1],I[0].shape[0]),interpolation=cv2.INTER_NEAREST)
pa=cv2.erode(pa,np.ones((81,81),np.uint8))
valid=(pa>0)
for w in W[1:]: valid&=(w>0)
print('\npicture area judged: %.1f%% of the frame'%(100*valid.mean()))
avg=cv2.GaussianBlur(sum(x.astype(np.float32) for x in W)/len(W),(0,0),4.0)
gx=cv2.Sobel(avg,cv2.CV_32F,1,0,ksize=5); gy=cv2.Sobel(avg,cv2.CV_32F,0,1,ksize=5)
grad=cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),6.0)
flat=(grad<np.percentile(grad[valid],20))&valid
edge=(grad>np.percentile(grad[valid],95))&valid
dark=(avg<np.percentile(avg[valid],25))&valid
def energy(img,mk,s=1.2):
    z=img.astype(np.float32); r=(z-cv2.GaussianBlur(z,(0,0),s))**2
    return float(r[mk].mean())
print('\n%-42s %10s %10s %7s %10s'%('file','NOISE','DETAIL','ratio','SHADOW'))
res=[]
for k,n in enumerate(names):
    a=energy(W[k],flat); b=energy(W[k],edge); sv=energy(W[k],dark,2.5)
    res.append((b/max(a,1e-6),n,a,b,sv))
    print('%-42s %10.2f %10.2f %7.2f %10.2f'%(n,a,b,b/max(a,1e-6),sv))
def band(img,mk,lo,hi):
    z=img.astype(np.float32)
    r=(cv2.GaussianBlur(z,(0,0),lo)-cv2.GaussianBlur(z,(0,0),hi))**2
    return float(r[mk].mean())
print()
print('SCALE BANDS on the edge mask -- a file that leads only in the fine band is winning on GRAIN,')
print('not on structure. This overturned one verdict that the single-scale detail figure got wrong.')
print('%-42s %10s %10s %10s'%('file','fine 1.2-3','mid 3-6','coarse 6-12'))
for k,n in enumerate(names):
    print('%-42s %10.2f %10.2f %10.2f'%(n,band(W[k],edge,1.2,3),band(W[k],edge,3,6),band(W[k],edge,6,12)))
res.sort(reverse=True)
print('\nbest by detail-per-noise: %s'%res[0][1])
g2=cv2.GaussianBlur(np.sqrt(gx*gx+gy*gy),(0,0),40.0)*valid
cy,cx=np.unravel_index(np.argmax(g2),g2.shape); S2=340
cy=int(np.clip(cy,S2,I[0].shape[0]-S2)); cx=int(np.clip(cx,S2,I[0].shape[1]-S2))
row=np.full((2*S2,len(W)*2*S2+12*(len(W)-1)),255,np.uint8); x=0
for w in W:
    c=w[cy-S2:cy+S2,cx-S2:cx+S2]; row[:,x:x+c.shape[1]]=c; x+=c.shape[1]+12
cv2.imwrite(SP+'/pair_cmp.jpg',row,[cv2.IMWRITE_JPEG_QUALITY,95])
print('crop written: %s/pair_cmp.jpg   order: %s'%(SP,' | '.join(names)))
