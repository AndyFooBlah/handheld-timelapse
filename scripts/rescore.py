"""Re-score saved homographies in data/align.csv and re-warp accepted frames, without re-matching.

Check: the scene is mostly far away and standpoints differ by metres, so a good H should be close to a camera
rotation plus zoom, H ~ K_ref R K_img^-1. For each H we search the image focal (0.6-1.6x the EXIF 35 mm value) for
the most orthonormal R and report the residual ||R^T R - I||. Garbage fits (locked onto foreground, wrong scene)
are far from any rotation.

    uv run python -I scripts/rescore.py [--max-rot-err 0.1] [--min-inliers 80] [--dry-run]
"""
import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import PHOTOS, REF
from align import load_rgb

ROOT = pathlib.Path(__file__).resolve().parent.parent


def K(f35, w, h):
    f = f35 * np.hypot(w, h) / 43.27  # 35 mm equivalent focal -> pixels, via the diagonal
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def rotation_error(H, f_img, wh_img, f_ref, wh_ref):
    """(residual, best focal scale) of H against a pure rotation, searching the image focal."""
    Kr_inv = np.linalg.inv(K(f_ref, *wh_ref))
    best = (np.inf, 1.0)
    for s in np.geomspace(0.6, 1.6, 61):
        R = Kr_inv @ H @ K(f_img * s, *wh_img)
        d = np.linalg.det(R)
        if d <= 0:
            continue
        R = R / np.cbrt(d)
        e = float(np.linalg.norm(R.T @ R - np.eye(3)))
        if e < best[0]:
            best = (e, float(s))
    return best


def oriented_wh(row):
    w, h = int(row["width"]), int(row["height"])
    return (h, w) if row.get("orientation") in ("5", "6", "7", "8") else (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--min-inliers", type=int, default=80)
    ap.add_argument("--max-rot-err", type=float, default=0.1)
    ap.add_argument("--out-scale", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "data" / "aligned"))
    ap.add_argument("--dry-run", action="store_true", help="score only; don't warp or rewrite align.csv")
    args = ap.parse_args()

    scan = {r["file"]: r for r in csv.DictReader(open(args.scan))}
    rows = list(csv.DictReader(open(args.align)))
    ref = scan[args.ref]
    wh_ref, f_ref = oriented_wh(ref), float(ref["focal35"])

    for r in rows:
        r["rot_err"], r["focal_scale"] = "", ""
        if not r["H"]:
            r["ok"] = False
            continue
        H = np.array([float(v) for v in r["H"].split()]).reshape(3, 3)
        s = scan[r["file"]]
        e, k = rotation_error(H, float(s["focal35"] or f_ref), oriented_wh(s), f_ref, wh_ref)
        r["rot_err"], r["focal_scale"] = round(e, 4), round(k, 3)
        r["ok"] = int(r["inliers"]) >= args.min_inliers and e <= args.max_rot_err

    good = sorted(r["rot_err"] for r in rows if r["H"] and int(r["inliers"]) >= 300)
    weak = sorted(r["rot_err"] for r in rows if r["H"] and 80 <= int(r["inliers"]) < 150)
    q = lambda v, p: v[int(p * (len(v) - 1))] if v else float("nan")
    print(f"rot_err, inliers>=300: n={len(good)} p50={q(good, .5):.3f} p90={q(good, .9):.3f} max={q(good, 1):.3f}")
    print(f"rot_err, inliers 80-149: n={len(weak)} p50={q(weak, .5):.3f} p90={q(weak, .9):.3f}")
    ok = [r for r in rows if r["ok"]]
    print(f"accepted {len(ok)}/{len(rows)} (inliers >= {args.min_inliers}, rot_err <= {args.max_rot_err}) "
          f"on {len({r['datetime'][:10] for r in ok})} days")
    if args.dry_run:
        return

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = {pathlib.Path(r["file"]).stem for r in ok}
    for p in out.glob("*"):
        if p.stem.removesuffix("_mask") not in keep:
            p.unlink()
    rw, rh = wh_ref
    ow, oh = round(rw * args.out_scale), round(rh * args.out_scale)
    S_out = np.diag([args.out_scale, args.out_scale, 1.0])
    for i, r in enumerate(ok, 1):
        stem = pathlib.Path(r["file"]).stem
        if (out / f"{stem}.jpg").exists() and (out / f"{stem}_mask.png").exists():
            continue
        rgb = load_rgb(pathlib.Path(args.dir) / r["file"])
        h, w = rgb.shape[:2]
        Ho = S_out @ np.array([float(v) for v in r["H"].split()]).reshape(3, 3)
        Image.fromarray(cv2.warpPerspective(rgb, Ho, (ow, oh), flags=cv2.INTER_AREA)).save(out / f"{stem}.jpg", quality=90)
        valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Ho, (ow, oh), flags=cv2.INTER_NEAREST)
        Image.fromarray(valid).save(out / f"{stem}_mask.png")
        r["coverage"] = round(float((valid > 0).mean()), 3)
        if i % 50 == 0:
            print(f"[{i}/{len(ok)}] warped", flush=True)

    fields = list(rows[0].keys())
    with open(args.align, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.align}; {len(ok)} warped frames in {out}")


if __name__ == "__main__":
    main()
