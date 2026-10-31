"""Label aligned frames as oak-framed or open, from canopy silhouette in the top band of the warped frame.

After alignment to the common far-field reference, the oak's canopy sits along the top of oak-standpoint frames;
open-standpoint frames have sky there. Canopy is dark against the sky beside it, by day as well as at dawn. The
measure is per row: the fraction of top-band pixels darker than 0.3 x that row's p90. A twilight sky that darkens
toward the top is uniform along each row, so it scores near 0 (an earlier whole-band threshold called it oak).
Edge density is kept in the CSV for reference only; grass, cloud and selfies made it a poor oak signal.
Writes data/standpoint.csv.

    uv run python -I scripts/standpoint.py [--dark-thr 0.12]
"""
import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import STANDPOINT_REFS

ROOT = pathlib.Path(__file__).resolve().parent.parent


def canopy_score(img_path, mask_path, band=0.35, width=504):
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    s = width / g.shape[1]
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))  # ignore the warp border itself
    # Band = top of this frame's own warped area (zoomed shots cover only the middle of the reference frame).
    rows = np.flatnonzero((m > 0).any(axis=1))
    if not len(rows):
        return 0.0, 0.0
    top = slice(rows[0], rows[0] + max(1, int((rows[-1] - rows[0]) * band)))
    edges = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120)
    valid = m[top] > 0
    if not valid.any():
        return 0.0, 0.0
    n_dark = n = 0
    for y in range(top.start, top.stop):
        px = g[y][m[y] > 0].astype(float)
        if len(px) < 20:
            continue
        n_dark += int((px < 0.3 * np.percentile(px, 90)).sum())
        n += len(px)
    return float((edges[top][valid] > 0).mean()), n_dark / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--aligned", default=str(ROOT / "data" / "aligned"))
    ap.add_argument("--dark-thr", type=float, default=0.12, help="per-row silhouette fraction above this = oak")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.align)) if r.get("ok") == "True"]
    feats = {}
    for r in rows:
        stem = pathlib.Path(r["file"]).stem
        feats[r["file"]] = canopy_score(pathlib.Path(args.aligned) / f"{stem}.jpg",
                                        pathlib.Path(args.aligned) / f"{stem}_mask.png")
    is_oak = lambda e, d: d > args.dark_thr
    for name, refs, want in (("oak", STANDPOINT_REFS["oak"], True), ("open", STANDPOINT_REFS["open"], False)):
        got = {f: (round(feats[f][0], 3), round(feats[f][1], 3), is_oak(*feats[f])) for f in refs if f in feats}
        print(f"{name:4} refs (edge, dark, oak?):", got, "" if all(v[2] == want for v in got.values()) else "MISCLASSIFIED")

    with open(ROOT / "data" / "standpoint.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "edge", "dark", "standpoint"])
        for file, (e, d) in feats.items():
            w.writerow([file, round(e, 4), round(d, 4), "oak" if is_oak(e, d) else "open"])
    n_oak = sum(is_oak(*v) for v in feats.values())
    print(f"oak={n_oak} open={len(feats) - n_oak}")


if __name__ == "__main__":
    main()
