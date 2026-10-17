"""Compare two scans of one photograph at a MATCHED picture scale.

The scale bands are computed in each file's own pixels, so a file with more pixels per
unit of picture scores higher on structure it does not actually hold. This registers the
second file onto the first, crops the same physical region from both, resamples both to an
identical size, and only then measures. Whatever survives that is real.
"""
import sys, pathlib
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
a_path, b_path = sys.argv[1], sys.argv[2]
out = pathlib.Path(sys.argv[3])

A = np.asarray(Image.open(a_path).convert("L"))
B = np.asarray(Image.open(b_path).convert("L"))

# register on downscaled copies, then lift the homography back to full resolution
def prep(g, target=1600):
    s = target / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s

As, sa = prep(A)
Bs, sb = prep(B)
orb = cv2.SIFT_create(6000)
ka, da = orb.detectAndCompute(As, None)
kb, db = orb.detectAndCompute(Bs, None)
m = cv2.BFMatcher().knnMatch(db, da, k=2)
good = [p for p, q in m if p.distance < 0.75 * q.distance]
src = np.float32([kb[p.queryIdx].pt for p in good]).reshape(-1, 1, 2)
dst = np.float32([ka[p.trainIdx].pt for p in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
print(f"matches {len(good)}  inliers {int(mask.sum())}")

# scale factor the homography implies: how many B pixels per A pixel
scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2])))) * (sa / sb)
print(f"B holds {1/scale:.3f} pixels per A pixel  "
      f"(A {A.shape[1]}x{A.shape[0]}, B {B.shape[1]}x{B.shape[0]})")

Sa = np.diag([1 / sa, 1 / sa, 1.0]).astype(np.float64)
Sb = np.diag([sb, sb, 1.0]).astype(np.float64)
Hf = Sa @ H @ Sb                                    # B full-res -> A full-res
Bw = cv2.warpPerspective(B, Hf, (A.shape[1], A.shape[0]), flags=cv2.INTER_LANCZOS4)

h, w = A.shape
boxes = {"upper": (0.30, 0.08, 0.70, 0.30),
         "middle": (0.25, 0.35, 0.75, 0.60),
         "lower": (0.10, 0.60, 0.60, 0.85)}
print(f"\n{'region':10s} {'lapvar A':>10s} {'lapvar B':>10s} {'hi-freq A':>10s} {'hi-freq B':>10s}")
for name, (x0, y0, x1, y1) in boxes.items():
    sl = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
    ca, cb = A[sl].astype(np.float32), Bw[sl].astype(np.float32)
    if cb.max() == 0:
        print(f"{name:10s} (no overlap)"); continue
    # normalise contrast so tone differences do not masquerade as detail
    ca = (ca - ca.mean()) / (ca.std() + 1e-6)
    cb = (cb - cb.mean()) / (cb.std() + 1e-6)
    la = cv2.Laplacian(ca, cv2.CV_32F).var()
    lb = cv2.Laplacian(cb, cv2.CV_32F).var()
    def hf(x):
        F = np.abs(np.fft.rfft2(x))
        yy = np.fft.fftfreq(x.shape[0])[:, None]
        xx = np.fft.rfftfreq(x.shape[1])[None, :]
        r = np.sqrt(yy ** 2 + xx ** 2) / 0.5
        return float(F[r > 0.5].sum() / F.sum())
    print(f"{name:10s} {la:10.4f} {lb:10.4f} {hf(ca):10.4f} {hf(cb):10.4f}")

Image.fromarray(np.hstack([A[int(.35*h):int(.60*h), int(.25*w):int(.75*w)],
                           Bw[int(.35*h):int(.60*h), int(.25*w):int(.75*w)]])
                ).save(out / "matched.jpg", quality=95)
print(f"\nwrote {out / 'matched.jpg'}  (left = A, right = B warped to A)")
