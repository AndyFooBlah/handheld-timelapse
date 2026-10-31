"""Find people in aligned frames: YOLO instance segmentation on the original photo, masks mapped into the reference.

Detection runs on the full-resolution original (so distant walkers and zoomed shots get every pixel), then each person
mask is warped through the frame's homography and measured in reference coordinates. Only what lands inside the
reference view, or inside a crop window, counts: a person at the edge of a 14 mm shot that the crop removes is no
reason to drop the frame. Detection runs at each --imgsz and the masks are merged: 1920 finds distant walkers, but a
person filling the frame scores low there (one portrait scored 0.32 at 1920 and 0.95 at 640), so 640 catches close portraits. Writes data/people.csv (one row per aligned frame) with person count, top confidence and the
fraction of the full reference view and of each crop window covered by people; boxes are in reference pixels.

    uv run python -I scripts/people.py [--model yolo26x-seg.pt] [--conf 0.35] [--imgsz 640 1920]
"""
import argparse
import csv
import os
import pathlib
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede the torch import

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import PHOTOS, REF
from align import DEV, load_rgb
from preview import CROPS
from rescore import oriented_wh

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRID = 8  # reference pixels per mask cell when measuring coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "yolo26x-seg.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1920])
    ap.add_argument("--out", default=str(ROOT / "data" / "people.csv"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    pathlib.Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)  # first use downloads the weights from github.com/ultralytics/assets
    scan = {r["file"]: r for r in csv.DictReader(open(args.scan))}
    rows = [r for r in csv.DictReader(open(args.align)) if r["ok"] == "True"][:args.limit]
    rw, rh = oriented_wh(scan[args.ref])
    gw, gh = rw // GRID, rh // GRID
    G = np.diag([1 / GRID, 1 / GRID, 1.0])
    windows = {name: (round(x0 * gw), round(y0 * gh), round(x1 * gw), round(y1 * gh))
               for name, (x0, y0, x1, y1) in CROPS.items()}
    fields = ["file", "datetime", "n_person", "max_conf", "area_full"] + [f"area_{n}" for n in CROPS] + ["boxes"]

    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        t0 = time.time()
        for i, r in enumerate(rows, 1):
            rgb = load_rgb(pathlib.Path(args.dir) / r["file"])
            found = []
            for sz in args.imgsz:
                res = model.predict(rgb[:, :, ::-1], imgsz=sz, conf=args.conf, classes=[0], device=DEV.type,
                                    retina_masks=False, verbose=False)[0]
                if res.masks is not None:
                    found += zip(res.masks.xy, res.boxes.conf.tolist())
            H = G @ np.array([float(v) for v in r["H"].split()]).reshape(3, 3)
            people = np.zeros((gh, gw), np.uint8)
            boxes, confs = [], []
            for poly, c in found:  # the same person found at two sizes is counted twice; the area is not
                if len(poly) < 3:
                    continue
                q = cv2.perspectiveTransform(poly[None].astype(np.float32), H)[0]
                one = np.zeros_like(people)
                cv2.fillPoly(one, [np.round(q).astype(np.int32)], 1)
                if not one.any():
                    continue  # outside the reference view
                people |= one
                ys, xs = np.nonzero(one)
                boxes.append(f"{xs.min() * GRID}:{ys.min() * GRID}:{(xs.max() + 1) * GRID}:{(ys.max() + 1) * GRID}:{c:.2f}")
                confs.append(c)
            out = {"file": r["file"], "datetime": r["datetime"], "n_person": len(boxes),
                   "max_conf": round(max(confs), 3) if confs else 0, "area_full": round(float(people.mean()), 5),
                   "boxes": " ".join(boxes)}
            for name, (x0, y0, x1, y1) in windows.items():
                out[f"area_{name}"] = round(float(people[y0:y1, x0:x1].mean()), 5)
            w.writerow(out)
            fo.flush()
            if DEV.type == "mps":
                torch.mps.empty_cache()
            if i % 25 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"[{i}/{len(rows)} {el / 60:.1f}m elapsed, ETA {el / i * (len(rows) - i) / 60:.1f}m] "
                      f"{r['file']} persons={len(boxes)} area_full={out['area_full']}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
