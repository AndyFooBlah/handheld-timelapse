"""Scene filter + alignment: match each candidate to a reference frame and warp it into the reference's frame.

ALIKED features + LightGlue (kornia) on downscaled copies, MAGSAC homography, then the homography is lifted
to full resolution. Writes data/align.csv and warped JPEGs + validity masks to data/aligned/.

    uv run python -I scripts/align.py [--ref IMG_0001.HEIC] [--out-scale 0.5]

Progress ([i/N], rate, ETA) goes to stdout and data/align.log; follow it with `tail -f data/align.log`.
align.csv is written row by row, so a partial run keeps its results.
"""
import argparse
import csv
import os
import pathlib
import sys
import time
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import kornia.feature as KF
import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import PHOTOS, REF

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FIELDS = ["file", "datetime", "status", "focal35", "dist_m", "matches", "inliers", "inlier_ratio",
          "mag", "keystone", "ok", "coverage", "H"]


def load_rgb(path):
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


class Matcher:
    def __init__(self, work_px, max_kp, clahe):
        self.work_px = work_px
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if clahe else None
        self.extractor = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=max_kp, device=DEV).eval()
        self.matcher = KF.LightGlueMatcher("aliked").to(DEV).eval()

    @torch.inference_mode()
    def features(self, rgb, zoom=1.0):
        """Keypoints in full-resolution pixel coords, descriptors, and the working image size.

        zoom = reference focal / this focal: a wide shot is matched at proportionally higher resolution so the
        distant scene has about the same pixel scale as in the reference.
        """
        long_side = max(rgb.shape[:2])
        s = min(max(self.work_px * zoom, 640), long_side) / long_side
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if self.clahe is not None:
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self.clahe.apply(lab[..., 0])
            small = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        t = torch.from_numpy(small).permute(2, 0, 1)[None].float().div(255).to(DEV)
        f = self.extractor(t)[0]
        return f.keypoints, f.descriptors, small.shape[:2], s

    @torch.inference_mode()
    def match(self, fa, fb):
        (ka, da, hwa, sa), (kb, db, hwb, sb) = fa, fb
        lafa = KF.laf_from_center_scale_ori(ka[None], torch.ones(1, len(ka), 1, 1, device=DEV))
        lafb = KF.laf_from_center_scale_ori(kb[None], torch.ones(1, len(kb), 1, 1, device=DEV))
        _, idx = self.matcher(da, db, lafa, lafb, hw1=hwa, hw2=hwb)
        idx = idx.cpu().numpy()
        pa = ka.cpu().numpy()[idx[:, 0]] / sa
        pb = kb.cpu().numpy()[idx[:, 1]] / sb
        return pa, pb


def magnification(H, w, h):
    """Linear scale factor of H at the centre of the source image (>1 means the source gets stretched)."""
    J = np.zeros((2, 2))
    x, y = w / 2, h / 2
    d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d
    J[0] = [(H[0, 0] - u * H[2, 0]) / d, (H[0, 1] - u * H[2, 1]) / d]
    J[1] = [(H[1, 0] - v * H[2, 0]) / d, (H[1, 1] - v * H[2, 1]) / d]
    return float(np.sqrt(abs(np.linalg.det(J))))


def edge_ratio(H, w, h):
    """Worst ratio of opposite edge lengths of the warped source rectangle (1 = no keystone)."""
    c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
    top, right, bottom, left = (np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4))
    r = [top / bottom, left / right]
    return float(max(max(x, 1 / x) for x in r if x > 0)) if all(x > 0 for x in r) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--status", action="append", default=None, help="scan statuses to try (default candidate, unknown)")
    ap.add_argument("--work-px", type=int, default=1280, help="long side used for matching")
    ap.add_argument("--max-kp", type=int, default=4096)
    ap.add_argument("--no-clahe", action="store_true", help="skip local contrast boost before feature extraction")
    ap.add_argument("--limit-file", help="only process files listed (one per line)")
    ap.add_argument("--min-inliers", type=int, default=80, help="below this, don't write a warped frame")
    ap.add_argument("--max-keystone", type=float, default=1.8, help="max opposite-edge length ratio of the warp")
    ap.add_argument("--out-scale", type=float, default=0.5, help="warped output size relative to the reference")
    ap.add_argument("--out", default=str(ROOT / "data" / "aligned"))
    ap.add_argument("--keep-gpu-cache", action="store_true",
                    help="skip torch.mps.empty_cache() per image (cache grows without bound with varying shapes)")
    ap.add_argument("--csv", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--log", default=str(ROOT / "data" / "align.log"))
    args = ap.parse_args()
    statuses = set(args.status or ["candidate", "unknown"])

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(args.scan)) if r["status"] in statuses or r["file"] == args.ref]
    if args.limit_file:
        keep = set(pathlib.Path(args.limit_file).read_text().split("\n"))
        rows = [r for r in rows if r["file"] in keep]
    focal = {r["file"]: float(r["focal35"] or 0) for r in csv.DictReader(open(args.scan))}
    ref_focal = focal.get(args.ref) or 26.0

    m = Matcher(args.work_px, args.max_kp, not args.no_clahe)
    ref_rgb = load_rgb(pathlib.Path(args.dir) / args.ref)
    rh, rw = ref_rgb.shape[:2]
    ref_f = m.features(ref_rgb)
    ow, oh = round(rw * args.out_scale), round(rh * args.out_scale)
    S_out = np.diag([args.out_scale, args.out_scale, 1.0])
    log_f = open(args.log, "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"device={DEV} ref={args.ref} {rw}x{rh} kp={len(ref_f[0])} -> {len(rows)} images")
    csv_f = open(args.csv, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDS)
    writer.writeheader()

    results, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        rgb = load_rgb(pathlib.Path(args.dir) / r["file"])
        h, w = rgb.shape[:2]
        zoom = ref_focal / (focal.get(r["file"]) or ref_focal)
        pa, pb = m.match(m.features(rgb, zoom), ref_f)
        H, inl = (None, None)
        if len(pa) >= 8:
            H, inl = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 4.0 / ref_f[3], maxIters=10000, confidence=0.999)
        n_in = int(inl.sum()) if inl is not None else 0
        res = {"file": r["file"], "datetime": r["datetime"], "status": r["status"], "focal35": r["focal35"],
               "dist_m": r["dist_m"], "matches": len(pa), "inliers": n_in,
               "inlier_ratio": round(n_in / len(pa), 3) if len(pa) else 0,
               "mag": round(magnification(H, w, h), 3) if H is not None else "",
               "keystone": round(edge_ratio(H, w, h), 3) if H is not None else "",
               "H": " ".join(f"{v:.8g}" for v in H.ravel()) if H is not None else ""}
        res["ok"] = H is not None and n_in >= args.min_inliers and res["keystone"] <= args.max_keystone
        if res["ok"]:
            Ho = S_out @ H
            warped = cv2.warpPerspective(rgb, Ho, (ow, oh), flags=cv2.INTER_AREA)
            valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Ho, (ow, oh), flags=cv2.INTER_NEAREST)
            stem = pathlib.Path(r["file"]).stem
            Image.fromarray(warped).save(out / f"{stem}.jpg", quality=90)
            Image.fromarray(valid).save(out / f"{stem}_mask.png")
            res["coverage"] = round(float((valid > 0).mean()), 3)
        if DEV.type == "mps" and not args.keep_gpu_cache:
            torch.mps.empty_cache()
        results.append(res)
        writer.writerow(res)
        csv_f.flush()
        el = time.time() - t0
        eta = el / i * (len(rows) - i)
        n_ok = sum(x["ok"] for x in results)
        gpu = f", gpu {torch.mps.driver_allocated_memory() / 2**30:.1f}G" if DEV.type == "mps" else ""
        log(f"[{i}/{len(rows)} {el / 60:.1f}m elapsed, ETA {eta / 60:.1f}m, {n_ok} ok{gpu}] "
            f"{r['file']:16} {r['datetime'][:10]} f={r['focal35']:>3} matches={len(pa):4} inliers={n_in:4} "
            f"mag={res['mag']} ks={res['keystone']} ok={res['ok']} cov={res.get('coverage', '')}")

    csv_f.close()
    ok = [r for r in results if r["ok"]]
    log(f"aligned {len(ok)}/{len(results)} (>= {args.min_inliers} inliers, keystone <= {args.max_keystone})")


if __name__ == "__main__":
    main()
