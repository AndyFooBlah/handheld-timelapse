"""Per-frame quality, for choosing the one photo a day (or a bracket) contributes, and for dropping branch intrusions.

For every aligned frame, over the full reference view (aligned/<stem>.jpg and its mask): clipped highlights and
shadows, and sharpness (log variance of the Laplacian; motion blur and haze lower it). For frames the far refit
confirmed, the same over the far crop, rendered from farfit.py's cache through C and the crest lock, plus:

  branch : fraction of the sky above the ridge covered by thin, near-black structure: a strong morphological black-hat
           on L, at under half its row's median sky. Oak twigs and leaves hanging in from the edges sit at a black-hat
           of 120-160 and under 0.35 of the sky; the darkest cloud edges measured 40 and 0.6.

Writes data/quality.csv.

    uv run python -I scripts/quality.py
"""
import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from crest import REF_WH, parse_h
from preview import CROPS

ROOT = pathlib.Path(__file__).resolve().parent.parent
W = 960  # measurement width
SKY = 0.40  # far crop: rows above this fraction of the height are sky (the crest sits near 0.47)


def measures(rgb, valid=None):
    """Clipped highlight and shadow fractions and log Laplacian variance over the valid pixels."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    ok = np.ones(L.shape, bool) if valid is None else valid > 0
    lap = cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 0.8).astype(np.float32), cv2.CV_32F)
    inner = cv2.erode(ok.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # no mask-edge steps
    return float((L[ok] >= 250).mean()), float((L[ok] <= 5).mean()), float(np.log(lap[inner].var() + 1e-6))


def branch_cover(rgb):
    """Fraction of the sky band covered by thin dark structure."""
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    sky = L[: round(SKY * L.shape[0])]
    hat = cv2.morphologyEx(sky, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    row = np.median(sky, axis=1, keepdims=True)
    hit = (hat > 60) & (sky < 0.5 * row)
    hit = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # no single-pixel noise
    return float(hit.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--aligned", default=str(ROOT / "data" / "aligned"))
    ap.add_argument("--farfit", default=str(ROOT / "data" / "farfit.csv"))
    ap.add_argument("--crest", default=str(ROOT / "data" / "crest.csv"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "farfit"))
    ap.add_argument("--work-px", type=int, default=1920, help="must match farfit.py --work-px")
    ap.add_argument("--min-crest", type=float, default=0.25, help="as preview.py: below this the lock isn't applied")
    ap.add_argument("--out", default=str(ROOT / "data" / "quality.csv"))
    args = ap.parse_args()

    rw, rh = REF_WH
    x0, y0, x1, y1 = CROPS["far"]
    mx, my = 0.1 * (x1 - x0), 0.25 * (y1 - y0)  # farfit.py's matching window
    win = (max(0, x0 - mx) * rw, max(0, y0 - my) * rh, min(1, x1 + mx) * rw, min(1, y1 + my) * rh)
    s = args.work_px / (win[2] - win[0])
    Te = np.array([[s, 0, -win[0] * s], [0, s, -win[1] * s], [0, 0, 1.0]])
    k = W / ((x1 - x0) * rw * s)  # work px -> measurement px
    crop = np.array([[k, 0, -(x0 * rw - win[0]) * s * k], [0, k, -(y0 * rh - win[1]) * s * k], [0, 0, 1.0]])
    crop_wh = (W, round((y1 - y0) * rh * s * k))
    fit = {r["file"]: r for r in csv.DictReader(open(args.farfit))}
    crest = {r["file"]: r for r in csv.DictReader(open(args.crest))}
    rows = [r for r in csv.DictReader(open(args.align)) if r["ok"] == "True"]

    fields = ["file", "datetime", "clip_hi", "clip_lo", "sharp", "far_clip_hi", "far_clip_lo", "far_sharp", "branch"]
    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            stem = pathlib.Path(r["file"]).stem
            bgr = cv2.imread(str(pathlib.Path(args.aligned) / f"{stem}.jpg"))
            mask = cv2.imread(str(pathlib.Path(args.aligned) / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
            h = round(bgr.shape[0] * W / bgr.shape[1])
            rgb = cv2.cvtColor(cv2.resize(bgr, (W, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            out = {"file": r["file"], "datetime": r["datetime"]}
            out["clip_hi"], out["clip_lo"], out["sharp"] = measures(rgb, cv2.resize(mask, (W, h), interpolation=cv2.INTER_NEAREST))
            f = fit.get(r["file"])
            cache = pathlib.Path(args.cache) / f"{stem}.jpg"
            if f and f["ok"] == "True" and cache.exists():
                c = crest.get(r["file"])
                dy = float(c["dy"]) if c and float(c["visible"]) >= args.min_crest else 0.0
                lock = np.array([[1, 0, 0], [0, 1, dy], [0, 0, 1.0]])
                M = crop @ Te @ lock @ parse_h(f["C"]) @ np.linalg.inv(Te)
                far = cv2.cvtColor(cv2.warpPerspective(cv2.imread(str(cache)), M, crop_wh, flags=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB)
                out["far_clip_hi"], out["far_clip_lo"], out["far_sharp"] = measures(far)
                out["branch"] = branch_cover(far)
            w.writerow({key: round(v, 5) if isinstance(v, float) else v for key, v in out.items()})
            if i % 100 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {r['file']}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
