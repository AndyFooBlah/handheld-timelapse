"""Lock the far crop onto the mountain crest, and flag frames whose crest is hidden by cloud or haze.

After farfit.py the far field agrees to about 1% of the frame, but the band fit mixes the crest with the bay and city
below it, and frames still differ by up to ~15 px (of 1920) at the crest itself. Here the crest of one crisp reference
silhouette (CREST_REF) is the anchor: each refitted frame's signed vertical gradient (sky brighter above ground) is
scored along that crest profile at vertical offsets within +/- --search px, each column normalised by the gradient
noise in the sky just above it. The best offset is the vertical lock; the fraction of columns with a clear step there
is the visibility. The reference's own crest can sit off the others (refraction), so the lock target is the median
offset over frames that show a crest. preview.py applies the lock to visible frames and drops daylight frames with the
crest hidden.

Writes data/crest.csv: dy (reference pixels, applied after farfit's C) and visible (0-1).

    uv run python -I scripts/crest.py [--search 20]
"""
import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import CREST_REF, CROPS, REF_WH

ROOT = pathlib.Path(__file__).resolve().parent.parent


def parse_h(s):
    return np.array([float(v) for v in s.split()]).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farfit", default=str(ROOT / "data" / "farfit.csv"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "farfit"))
    ap.add_argument("--work-px", type=int, default=1920, help="must match farfit.py --work-px")
    ap.add_argument("--search", type=int, default=20, help="vertical search, work px (the shoreline is ~35 px below)")
    ap.add_argument("--z", type=float, default=4.0, help="a column shows the crest if its step is this many sky-noise units")
    ap.add_argument("--out", default=str(ROOT / "data" / "crest.csv"))
    args = ap.parse_args()

    rw, rh = REF_WH
    x0, y0, x1, y1 = CROPS["far"]
    mx, my = 0.1 * (x1 - x0), 0.25 * (y1 - y0)  # farfit.py's matching window
    win = (max(0, x0 - mx) * rw, max(0, y0 - my) * rh, min(1, x1 + mx) * rw, min(1, y1 + my) * rh)
    s = args.work_px / (win[2] - win[0])
    Te = np.array([[s, 0, -win[0] * s], [0, s, -win[1] * s], [0, 0, 1.0]])
    cols = np.arange(round((x0 * rw - win[0]) * s), round((x1 * rw - win[0]) * s))
    fit = {r["file"]: r for r in csv.DictReader(open(args.farfit))}

    def grad(f):
        g = cv2.imread(str(pathlib.Path(args.cache) / f"{pathlib.Path(f).stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        C = Te @ parse_h(fit[f]["C"]) @ np.linalg.inv(Te)
        g = cv2.GaussianBlur(cv2.warpPerspective(g, C, g.shape[::-1]).astype(np.float32), (0, 0), 1.5)
        d = np.zeros_like(g)
        d[3:-3] = g[:-6] - g[6:]
        return d

    # Crest of the reference: per column, the strongest step near its median row, lightly smoothed.
    d = grad(CREST_REF)
    lo, hi = round((0.60 * rh - win[1]) * s), round((0.66 * rh - win[1]) * s)
    rough = np.median(d[lo:hi][:, cols].argmax(0)) + lo
    near = int(rough) - 15
    crest = d[near:near + 31][:, cols].argmax(0) + near
    crest = np.round(cv2.medianBlur(crest.astype(np.float32)[None], 5)[0]).astype(int)
    print(f"crest from {CREST_REF}: mean row {crest.mean():.1f}, range {crest.min()}-{crest.max()}")

    # Score every frame at every offset in a wide range; the reference's own crest may sit off the consensus (dawn
    # refraction can lift it by tens of px), so the lock target is the median offset of the frames that show a crest.
    wide = range(-2 * args.search - 5, 2 * args.search + 6)
    table = {}
    for f, r in fit.items():
        if r["ok"] != "True":
            continue
        d = grad(f)
        row = []
        for dy in wide:
            step = np.max([d[crest + dy + o, cols] for o in range(-2, 3)], axis=0)
            sky = np.stack([d[crest + dy - o, cols] for o in range(14, 50)])
            z = step / (np.median(np.abs(sky), axis=0) * 1.4826 + 0.5)
            vis = np.convolve((z > args.z).astype(float), np.ones(15) / 15, "same") > 0.5
            row.append((float(np.median(z)), dy, float(vis.mean())))
        table[f] = row
    seen = [max(row)[1] for row in table.values() if max(row)[2] >= 0.5]
    target = int(np.median(seen))
    print(f"consensus crest {target:+d} px from {CREST_REF}'s ({len(seen)} frames with a visible crest)")

    with open(args.out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["file", "elevation", "dy", "visible"])
        n_vis = 0
        for f, row in table.items():
            _, dy, vis = max(t for t in row if abs(t[1] - target) <= args.search)
            n_vis += vis >= 0.5
            w.writerow([f, fit[f]["elevation"], round((target - dy) / s, 2), round(vis, 3)])
    print(f"wrote {args.out}; {n_vis} of {len(table)} with the crest visible in >= half the columns")


if __name__ == "__main__":
    main()
