"""Second-pass alignment: align frames that failed against the reference by matching them to a similar frame that did.

For each unaligned frame we try up to --anchors already-aligned frames, same day first, then closest in solar phase
and focal length (lighting and zoom matter far more to matching than date). The frame-to-anchor homography is composed
with the anchor's homography to the reference and must pass the same gates as align.py/rescore.py: enough inliers and
a near-rotation composed H. Rounds repeat so newly aligned frames can anchor the rest.

Rows go to data/chain.csv as they finish; at the end align.csv is updated (ok, H, inliers, rot_err, via) and the new
frames are warped into data/aligned/. Progress goes to stdout and data/chain.log.

    uv run python -I scripts/chain.py [--rounds 2] [--anchors 5] [--max-dist 60]
"""
import argparse
import csv
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must precede the torch import (align.py sets it too late here)

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import PHOTOS, REF
from align import DEV, Matcher, load_rgb
from rescore import oriented_wh, rotation_error

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIELDS = ["round", "file", "datetime", "via", "tried", "inliers", "rot_err", "focal_scale", "ok", "H"]


def parse_h(s):
    return np.array([float(v) for v in s.split()]).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--solar", default=str(ROOT / "data" / "solar.csv"))
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--anchors", type=int, default=5, help="aligned frames tried per unaligned frame")
    ap.add_argument("--max-dist", type=float, default=60, help="skip frames with GPS farther than this (m)")
    ap.add_argument("--min-inliers", type=int, default=80)
    ap.add_argument("--max-rot-err", type=float, default=0.7)
    ap.add_argument("--work-px", type=int, default=1280)
    ap.add_argument("--max-kp", type=int, default=4096)
    ap.add_argument("--out-scale", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "data" / "aligned"))
    ap.add_argument("--csv", default=str(ROOT / "data" / "chain.csv"))
    ap.add_argument("--log", default=str(ROOT / "data" / "chain.log"))
    ap.add_argument("--resume", action="store_true", help="keep rows already in --csv and skip frames tried there")
    ap.add_argument("--limit", type=int, help="only try the first N unaligned frames (for testing)")
    args = ap.parse_args()

    scan = {r["file"]: r for r in csv.DictReader(open(args.scan))}
    solar = {r["file"]: r for r in csv.DictReader(open(args.solar))}
    rows = list(csv.DictReader(open(args.align)))
    ref = scan[args.ref]
    wh_ref, f_ref = oriented_wh(ref), float(ref["focal35"])
    focal = lambda f: float(scan[f]["focal35"] or f_ref)
    phase = lambda f: float(solar[f]["phase"]) if f in solar else 90.0

    aligned = {r["file"]: parse_h(r["H"]) for r in rows if r["ok"] == "True"}
    todo = [r["file"] for r in rows if r["ok"] != "True"
            and scan[r["file"]]["front"] != "True"
            and (not r["dist_m"] or float(r["dist_m"]) <= args.max_dist)][:args.limit]

    m = Matcher(args.work_px, args.max_kp, clahe=True)
    feat_cache = {}

    def feats(f, zoom):
        """Features of file f, matched at work_px * zoom; cached (anchors are reused many times)."""
        key = (f, round(zoom, 2))
        if key not in feat_cache:
            feat_cache[key] = m.features(load_rgb(pathlib.Path(args.dir) / f), zoom)
            if len(feat_cache) > 400:  # bound memory; oldest entries go first
                feat_cache.pop(next(iter(feat_cache)))
        return feat_cache[key]

    def anchor_order(f):
        day = scan[f]["datetime"][:10]
        same = sorted((a for a in aligned if scan[a]["datetime"][:10] == day),
                      key=lambda a: abs((np.datetime64(scan[a]["datetime"][:19]) - np.datetime64(scan[f]["datetime"][:19]))
                                        .astype(float)))
        rest = sorted((a for a in aligned if a not in same),
                      key=lambda a: abs(phase(a) - phase(f)) / 15 + abs(math.log2(focal(a) / focal(f))))
        return (same[:2] + rest)[:args.anchors]

    log_f = open(args.log, "a" if args.resume else "w")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    results, done = {}, {}
    if args.resume and pathlib.Path(args.csv).exists():
        for r in csv.DictReader(open(args.csv)):
            done.setdefault(int(r["round"]), set()).add(r["file"])
            if r["ok"] == "True":
                results[r["file"]] = r
                aligned[r["file"]] = parse_h(r["H"])
    csv_f = open(args.csv, "a" if done else "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDS)
    if not done:
        writer.writeheader()
    t0 = time.time()
    log(f"device={DEV} resumed {len(results)} chained from {args.csv}" if done else f"device={DEV}")
    log(f"{len(aligned)} aligned anchors, {len(todo)} frames to try, {args.rounds} rounds")

    for rnd in range(1, args.rounds + 1):
        pending = [f for f in todo if f not in aligned and f not in done.get(rnd, set())]
        new = 0
        for i, f in enumerate(pending, 1):
            best = None
            tried = anchor_order(f)
            for a in tried:
                # Match at the anchor's native scale; the frame is scaled by the focal ratio as in align.py.
                fa = feats(a, 1.0)
                # Cap at 2x: a 14 mm frame against a 120 mm anchor would otherwise run ALIKED near full resolution.
                pa, pb = m.match(feats(f, min(focal(a) / focal(f), 2.0)), fa)
                if len(pa) < 8:
                    continue
                H, inl = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 4.0 / fa[3], maxIters=10000, confidence=0.999)
                n_in = int(inl.sum()) if inl is not None else 0
                if H is None or n_in < args.min_inliers:
                    continue
                Hc = aligned[a] @ H
                Hc /= Hc[2, 2]
                e, k = rotation_error(Hc, focal(f), oriented_wh(scan[f]), f_ref, wh_ref)
                if e <= args.max_rot_err and (best is None or n_in > best["inliers"]):
                    best = {"via": a, "inliers": n_in, "rot_err": round(e, 4), "focal_scale": round(k, 3), "H": Hc}
            if DEV.type == "mps":
                torch.mps.empty_cache()
            res = {"round": rnd, "file": f, "datetime": scan[f]["datetime"], "tried": len(tried), "ok": best is not None}
            if best:
                res.update(best)
                res["H"] = " ".join(f"{v:.8g}" for v in best["H"].ravel())
                results[f] = res
                aligned[f] = best["H"]
                new += 1
            writer.writerow(res)
            csv_f.flush()
            el = time.time() - t0
            eta = el / i * (len(pending) - i)
            log(f"[round {rnd} {i}/{len(pending)} {el / 60:.1f}m elapsed, round ETA {eta / 60:.1f}m, {new} new] "
                f"{f:18} {scan[f]['datetime'][:10]} f={scan[f]['focal35']:>3} "
                + (f"via {best['via']} inliers={best['inliers']} rot={best['rot_err']}" if best else "no match"))
        log(f"round {rnd}: {new} newly aligned")
        if not new:
            break
    csv_f.close()

    # Merge into align.csv and warp the new frames.
    out = pathlib.Path(args.out)
    rw, rh = wh_ref
    ow, oh = round(rw * args.out_scale), round(rh * args.out_scale)
    S_out = np.diag([args.out_scale, args.out_scale, 1.0])
    for r in rows:
        r.setdefault("via", "")
        if r["file"] not in results:
            continue
        c = results[r["file"]]
        r.update(ok=True, H=c["H"], inliers=c["inliers"], rot_err=c["rot_err"], focal_scale=c["focal_scale"], via=c["via"])
        rgb = load_rgb(pathlib.Path(args.dir) / r["file"])
        h, w = rgb.shape[:2]
        Ho = S_out @ parse_h(c["H"])
        stem = pathlib.Path(r["file"]).stem
        Image.fromarray(cv2.warpPerspective(rgb, Ho, (ow, oh), flags=cv2.INTER_AREA)).save(out / f"{stem}.jpg", quality=90)
        valid = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Ho, (ow, oh), flags=cv2.INTER_NEAREST)
        Image.fromarray(valid).save(out / f"{stem}_mask.png")
        r["coverage"] = round(float((valid > 0).mean()), 3)
    with open(args.align, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if r["ok"] in (True, "True")]
    log(f"chained {len(results)} frames; align.csv now {len(ok)} ok on {len({r['datetime'][:10] for r in ok})} days")


if __name__ == "__main__":
    main()
