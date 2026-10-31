"""Refine alignment for the far crop on the far field only (ridgeline, bay, distant city).

The homographies in align.csv are fitted on whatever matched, nearby trees included. From the oak standpoint, a few
metres from the reference, that foreground has parallax and pulls the distant ridge off by up to ~10% of the crop
height. Here every frame is rendered into a window a little larger than the far crop and matched against anchor frames
whose far field is trusted, keeping only matches that land in the ridge-and-bay band of the anchor (no sky, so
clouds can't vote, and no treetops). A similarity transform (shift, rotation, zoom) is fitted per anchor; the
estimates must agree. Rounds repeat with confidently corrected frames joining the anchors.

Anchors to start: open-standpoint frames aligned directly to the reference with many inliers (same spot as the
reference, so no parallax).

An anchor whose own check is inconclusive keeps its original fit. Each estimate is compared at ridge height with the
weighted median of all of them; a frame is confirmed when at least two agree within --max-disagree, and only those
are averaged, so one bad anchor is outvoted instead of failing the frame.

scripts/crest.py then locks the result vertically onto the mountain crest and flags hidden crests.

Writes data/farfit.csv: correction C (reference pixels, applied as C @ H) and how many anchors agreed.

    uv run python -I scripts/farfit.py [--rounds 3] [--anchors 5]
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import FAR_BAND, PHOTOS, REF
from align import DEV, Matcher, load_rgb, magnification
from preview import CROPS, covers
from rescore import oriented_wh

ROOT = pathlib.Path(__file__).resolve().parent.parent
BAND = FAR_BAND  # reference-frame rows (fractions) holding the ridge, bay and distant city
FIELDS = ["file", "datetime", "elevation", "round", "anchors", "agreeing", "inliers", "agree_px", "ok", "C"]


def parse_h(s):
    return np.array([float(v) for v in s.split()]).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--solar", default=str(ROOT / "data" / "solar.csv"))
    ap.add_argument("--standpoint", default=str(ROOT / "data" / "standpoint.csv"))
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--work-px", type=int, default=1920, help="width of the matching window render")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=5)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--max-kp", type=int, default=2048)
    ap.add_argument("--search-px", type=int, default=180, help="frame-side margin around the band (work px)")
    ap.add_argument("--max-disagree", type=float, default=8.0, help="px at --work-px, at ridge height, between anchor estimates")
    ap.add_argument("--cache", default=str(ROOT / "data" / "farfit"))
    ap.add_argument("--out", default=str(ROOT / "data" / "farfit.csv"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    scan = {r["file"]: r for r in csv.DictReader(open(args.scan))}
    solar = {r["file"]: r for r in csv.DictReader(open(args.solar))}
    stand = {r["file"]: r["standpoint"] for r in csv.DictReader(open(args.standpoint))}
    rw, rh = oriented_wh(scan[args.ref])
    x0, y0, x1, y1 = CROPS["far"]
    far = (x0 * rw, y0 * rh, x1 * rw, y1 * rh)
    mx, my = 0.1 * (x1 - x0), 0.25 * (y1 - y0)
    win = (max(0, x0 - mx) * rw, max(0, y0 - my) * rh, min(1, x1 + mx) * rw, min(1, y1 + my) * rh)
    s = args.work_px / (win[2] - win[0])
    size = (args.work_px, round((win[3] - win[1]) * s))
    Te = np.array([[s, 0, -win[0] * s], [0, s, -win[1] * s], [0, 0, 1.0]])
    to_work = lambda yf: round((yf * rh - win[1]) * s)
    band = (to_work(BAND[0]), to_work(BAND[1]))
    far_cols = (round((far[0] - win[0]) * s), round((far[2] - win[0]) * s))

    rows = [r for r in csv.DictReader(open(args.align)) if r["ok"] == "True"]
    rows = [r for r in rows if covers(parse_h(r["H"]), oriented_wh(scan[r["file"]]), far)][:args.limit]
    elev = {r["file"]: float(solar[r["file"]]["elevation"]) if r["file"] in solar else 0.0 for r in rows}
    phase = {r["file"]: float(solar[r["file"]]["phase"]) if r["file"] in solar else 90.0 for r in rows}
    when = {r["file"]: r["datetime"] for r in rows}
    day = {f: t[:10] for f, t in when.items()}
    base = [r["file"] for r in rows if not r.get("via") and stand.get(r["file"]) == "open" and int(r["inliers"]) >= 300]
    print(f"{len(rows)} frames cover the far crop; {len(base)} base anchors; device={DEV}", flush=True)

    # 1. Render each frame's matching window with its current H (cached as JPEG).
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        p = cache / f"{pathlib.Path(r['file']).stem}.jpg"
        if not p.exists():
            src = load_rgb(pathlib.Path(args.dir) / r["file"])
            Hc = Te @ parse_h(r["H"])
            mag = magnification(Hc, src.shape[1], src.shape[0])
            if mag < 0.5:
                k = 2 * mag
                src = cv2.resize(src, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                Hc = Hc @ np.diag([1 / k, 1 / k, 1.0])
            cv2.imwrite(str(p), cv2.warpPerspective(src, Hc, size, flags=cv2.INTER_AREA)[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i % 50 == 0 or i == len(rows):
            print(f"[render {i}/{len(rows)} {(time.time() - t0) / 60:.1f}m]", flush=True)
    img = lambda f: cv2.imread(str(cache / f"{pathlib.Path(f).stem}.jpg"))[:, :, ::-1].copy()

    # 2. Features once per frame, then rounds of far-field matching against the anchor pool.
    # Anchor side: just the band. Frame side: the band plus room for the frame's own error.
    m = Matcher(args.work_px, args.max_kp, clahe=True)
    a_rows = (band[0] - 8, band[1] + 8)
    f_rows = (max(0, band[0] - args.search_px), min(size[1], band[1] + args.search_px))
    feats_a, feats_f = {}, {}

    def strip_feats(rgb, rows):
        k, d, hw, sc = m.features(np.ascontiguousarray(rgb[rows[0]:rows[1]]))
        return k, d, hw, sc

    for i, r in enumerate(rows, 1):
        im = img(r["file"])
        feats_a[r["file"]] = strip_feats(im, a_rows)
        feats_f[r["file"]] = strip_feats(im, f_rows)
        if i % 100 == 0 or i == len(rows):
            print(f"[features {i}/{len(rows)} {(time.time() - t0) / 60:.1f}m]", flush=True)
        if DEV.type == "mps" and i % 20 == 0:
            torch.mps.empty_cache()

    # Agreement is judged where it shows: the two ends and the middle of the ridge across the far crop.
    ridge_y = (band[0] + band[1]) / 2
    probe = np.float32([[far_cols[0], ridge_y], [(far_cols[0] + far_cols[1]) / 2, ridge_y], [far_cols[1], ridge_y]])

    def similarity(ests):
        """Inlier-weighted median of the similarity parameters (log scale, angle, shift) of the estimates."""
        P = np.array([[np.log(np.hypot(C[0, 0], C[1, 0])), np.arctan2(C[1, 0], C[0, 0]), C[0, 2], C[1, 2]]
                      for C, _, _ in ests])
        wts = np.array([n for _, n, _ in ests], float)

        def wmedian(v):
            o = np.argsort(v)
            c = np.cumsum(wts[o])
            return v[o][np.searchsorted(c, c[-1] / 2)]
        ls, th, tx, ty = (wmedian(P[:, j]) for j in range(4))
        sc = np.exp(ls)
        return np.array([[sc * np.cos(th), -sc * np.sin(th), tx], [sc * np.sin(th), sc * np.cos(th), ty], [0, 0, 1]])

    def combine(ests):
        """Median correction from the estimates that agree with it at ridge height (outlier anchors are outvoted).

        Returns (C, number agreeing, their median distance to C)."""
        C = similarity(ests)
        at = cv2.perspectiveTransform(probe[None], C)[0]
        d = np.array([np.linalg.norm(cv2.perspectiveTransform(probe[None], E)[0] - at, axis=1).max() for E, _, _ in ests])
        good = [e for e, di in zip(ests, d) if di <= args.max_disagree]
        if len(good) >= 2:
            C = similarity(good)
        return C, len(good), float(np.median(d[d <= args.max_disagree])) if len(good) else float("nan")

    corr = {f: np.eye(3) for f in base}  # work-window corrections of the anchor pool
    result = {}
    for rnd in range(1, args.rounds + 1):
        pool = dict(corr)
        changed = 0
        for i, r in enumerate(rows, 1):
            f = r["file"]
            if f in result and result[f]["ok"]:
                continue
            cands = [a for a in pool if a != f]
            same = [a for a in cands if day[a] == day[f]][:2]
            rest = sorted((a for a in cands if a not in same), key=lambda a: abs(phase[a] - phase[f]))
            ests = []
            for a in (same + rest)[:args.anchors]:
                pa, pb = m.match(feats_f[f], feats_a[a])
                if len(pa) < args.min_inliers:
                    continue
                pa = pa + [0, f_rows[0]]
                pb = pb + [0, a_rows[0]]
                A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                                                     maxIters=5000, confidence=0.999)
                n = int(inl.sum()) if inl is not None else 0
                if A is None or n < args.min_inliers:
                    continue
                ests.append((pool[a] @ np.vstack([A, [0, 0, 1]]), n, a))
            res = {"file": f, "datetime": r["datetime"], "elevation": round(elev[f], 2), "round": rnd,
                   "anchors": len(ests), "agreeing": 0, "inliers": sum(n for _, n, _ in ests), "agree_px": "", "ok": False}
            if ests:
                C, n_good, agree = combine(ests)
                ok = n_good >= 2 or (len(ests) == 1 and ests[0][1] >= 3 * args.min_inliers)
                res.update(agreeing=n_good, agree_px="" if np.isnan(agree) else round(agree, 2), ok=bool(ok), _C=C)
            result[f] = res
            if res["ok"]:
                corr[f] = res["_C"]
                changed += 1
            if DEV.type == "mps" and i % 20 == 0:
                torch.mps.empty_cache()
            if i % 50 == 0 or i == len(rows):
                print(f"[round {rnd} {i}/{len(rows)} {(time.time() - t0) / 60:.1f}m] ok so far {sum(v['ok'] for v in result.values())}",
                      flush=True)
        print(f"round {rnd}: {changed} newly confident; {len(corr)} in pool", flush=True)
        if not changed:
            break

    for f in base:  # an anchor whose own refit was inconclusive keeps its original fit
        if not result.get(f, {}).get("ok"):
            result.setdefault(f, {"file": f, "datetime": when[f], "elevation": round(elev[f], 2), "anchors": 0,
                                  "agreeing": 0, "inliers": 0, "agree_px": ""})
            result[f].update(round=0, ok=True)
            corr[f] = np.eye(3)

    with open(args.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            f = r["file"]
            res = result.get(f) or {"file": f, "datetime": r["datetime"], "elevation": round(elev[f], 2), "round": "",
                                    "anchors": 0, "inliers": 0, "agree_px": "", "ok": False}
            C = corr.get(f, np.eye(3)) if res["ok"] else np.eye(3)
            res["C"] = " ".join(f"{v:.8g}" for v in (np.linalg.inv(Te) @ C @ Te).ravel())
            res.pop("_C", None)
            w.writerow(res)
    print(f"wrote {args.out}: {sum(1 for v in result.values() if v['ok'])} of {len(rows)} frames confirmed", flush=True)


if __name__ == "__main__":
    main()
