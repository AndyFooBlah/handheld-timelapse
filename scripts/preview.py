"""Quick-look preview of aligned frames as GIF + MP4, date-stamped.

--order date  : one frame per day, calendar order.
--order solar : frames sorted by solar phase (scripts/solar.py), dates mixed; bracket duplicates
                (shots within --bracket-s of each other) collapse to one.

--pick score : which frame a day (or a bracket) keeps, from scripts/quality.py and scripts/crest.py:
                 2 x crest visibility (0.5 where unmeasured)
               - 4 x the fraction of clipped highlights and shadows
               + 0.3 x sharpness, log Laplacian variance relative to the day's median
               - --phase-weight per 90 degrees of solar phase from the median of the whole sequence (date order only,
                 so a day with a dawn shot and a noon shot keeps the one lit like its neighbours).
               Measured over the far crop with --crop far, else over the full view. --pick inliers is the old rule.

--crop x0,y0,x1,y1 : a window of the reference frame (fractions), warped straight from the originals at --width, so
                     zoomed shots keep their detail. Only frames whose footprint covers the whole window are used.
                     --crop far is the distant bay and ridgeline.

--people without : drop frames where people (scripts/people.py) cover more than --people-thr of the shown view.

--deflicker N : match each frame's brightness, contrast and colour cast (Lab mean/std) toward the median of the N
                frames around it in playback order that are lit alike (solar phase within --phase-tol degrees), with
                each correction capped. Slow change survives (sunrise in solar order, seasons in date order); jumps
                between neighbours shrink. 0 turns it off.

--crop far also applies the far-field refit (scripts/farfit.py) and the crest lock (scripts/crest.py), and drops
           frames the refit couldn't confirm plus daylight frames whose crest is hidden (visibility below --min-crest),
           and frames with oak branches hanging into the sky (quality.py branch above --max-branch).

--source originals : full view warped straight from the original photos (for finals) instead of the half-size
                     aligned/ JPEGs. Crops always come from the originals.

exclude.txt (repo root) lists photos never to use, one file name per line.

--hold S --fade F : each photo shows for S seconds, the last F of them a crossfade into the next (MP4 at --out-fps).
                    The GIF steps once per hold, without fades.

    uv run python -I scripts/preview.py [--order date|solar] [--standpoint oak|open|all] [--crop far]
                                        [--people with|without] [--deflicker 9] [--hold 0.75] [--fade 0.35]
"""
import argparse
import csv
import datetime as dt
import pathlib
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import CROPS, PHOTOS, REF
from align import load_rgb, magnification
from rescore import oriented_wh

ROOT = pathlib.Path(__file__).resolve().parent.parent


def covers(H, src_wh, rect):
    """True if the warped source rectangle contains every corner of rect (reference pixels)."""
    w, h = src_wh
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None], H)[0]
    x0, y0, x1, y1 = rect
    return all(cv2.pointPolygonTest(quad, (float(x), float(y)), False) >= 0
               for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def lab_stats(rgb, valid=None):
    """Lab L mean, L std, a mean, b mean (OpenCV 8-bit Lab scale) over the valid pixels of a downscaled copy."""
    k = 320 / rgb.shape[1]
    lab = cv2.cvtColor(cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
    px = lab.reshape(-1, 3).astype(float)
    if valid is not None:
        px = px[cv2.resize(valid, lab.shape[1::-1], interpolation=cv2.INTER_NEAREST).ravel() > 0]
    return np.array([px[:, 0].mean(), px[:, 0].std() + 1e-3, px[:, 1].mean(), px[:, 2].mean()])


def deflicker(frames, stats, phases, window, phase_tol, max_shift=30.0, max_gain=1.25, max_cast=10.0):
    """Pull each frame's Lab stats toward the median of its similarly lit neighbours; returns corrected frames."""
    out = []
    for i, (rgb, valid) in enumerate(frames):
        lo, hi = max(0, i - window // 2), min(len(frames), i + window // 2 + 1)
        near = [j for j in range(lo, hi) if abs(phases[j] - phases[i]) <= phase_tol]
        L, sd, a, b = stats[i]
        Lt, sdt, at, bt = np.median(stats[near], axis=0)
        g = np.clip(sdt / sd, 1 / max_gain, max_gain)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] = (lab[..., 0] - L) * g + L + np.clip(Lt - L, -max_shift, max_shift)
        lab[..., 1] += np.clip(at - a, -max_cast, max_cast)
        lab[..., 2] += np.clip(bt - b, -max_cast, max_cast)
        fixed = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        if valid is not None:
            fixed[valid == 0] = 0
        out.append(fixed)
    return out


def load(path, key="file"):
    p = pathlib.Path(path)
    return {r[key]: r for r in csv.DictReader(open(p))} if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", default=str(ROOT / "data" / "align.csv"))
    ap.add_argument("--aligned", default=str(ROOT / "data" / "aligned"))
    ap.add_argument("--order", choices=["date", "solar"], default="date")
    ap.add_argument("--standpoint", choices=["oak", "open", "all"], default="all")
    ap.add_argument("--bracket-s", type=float, default=90, help="solar order: shots closer than this are one moment")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--hold", type=float, default=0.75, help="seconds per photo, fade included")
    ap.add_argument("--fade", type=float, default=0.35, help="crossfade seconds at the end of each hold; 0 = cut")
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--farfit", default=str(ROOT / "data" / "farfit.csv"))
    ap.add_argument("--crest", default=str(ROOT / "data" / "crest.csv"))
    ap.add_argument("--min-crest", type=float, default=0.25, help="far crop: daylight frames need this crest visibility")
    ap.add_argument("--night-elev", type=float, default=-10.0, help="far crop: below this sun elevation, no crest check")
    ap.add_argument("--crop", help="'far' or x0,y0,x1,y1 as fractions of the reference frame")
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--people", choices=["with", "without"], default="with")
    ap.add_argument("--people-thr", type=float, default=0.001, help="without: max fraction of the view covered by people")
    ap.add_argument("--deflicker", type=int, default=9, help="neighbour window in frames; 0 = off")
    ap.add_argument("--phase-tol", type=float, default=25, help="deflicker: only neighbours this close in solar phase")
    ap.add_argument("--quality", default=str(ROOT / "data" / "quality.csv"))
    ap.add_argument("--pick", choices=["score", "inliers"], default="score")
    ap.add_argument("--phase-weight", type=float, default=1.0, help="score: penalty per 90 deg off the median phase")
    ap.add_argument("--max-branch", type=float, default=0.001, help="far crop: max sky fraction covered by branches")
    ap.add_argument("--source", choices=["aligned", "originals"], default="aligned", help="full view only")
    ap.add_argument("--exclude", default=str(ROOT / "exclude.txt"))
    ap.add_argument("--jpeg-q", type=int, default=88, help="quality of the saved frames the MP4 is built from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = f"_{args.crop}" if args.crop in CROPS else "_crop" if args.crop else ""
    tag += "_nopeople" if args.people == "without" else ""
    tag += "" if args.deflicker else "_raw"
    out = args.out or str(ROOT / "data" / f"preview_{args.standpoint}_{args.order}{tag}")

    solar = load(ROOT / "data" / "solar.csv")
    stand = load(ROOT / "data" / "standpoint.csv")
    rows = [r for r in csv.DictReader(open(args.align))
            if r.get("ok") == "True" and (args.standpoint == "all" or stand.get(r["file"], {}).get("standpoint") == args.standpoint)]
    ex = pathlib.Path(args.exclude)
    excluded = {ln.split("#")[0].strip() for ln in ex.read_text().splitlines()} - {""} if ex.exists() else set()
    rows = [r for r in rows if r["file"] not in excluded]
    if args.crop or args.source == "originals":
        scan = load(args.scan)
    if args.crop:
        rw, rh = oriented_wh(scan[args.ref])
        fx0, fy0, fx1, fy1 = CROPS.get(args.crop) or [float(v) for v in args.crop.split(",")]
        rect = (fx0 * rw, fy0 * rh, fx1 * rw, fy1 * rh)
        Hof = lambda r: np.array([float(v) for v in r["H"].split()]).reshape(3, 3)
        rows = [r for r in rows if covers(Hof(r), oriented_wh(scan[r["file"]]), rect)]
        if args.crop == "far":
            fit, crest = load(args.farfit), load(args.crest)
            if not fit or not crest:
                raise SystemExit("--crop far needs data/farfit.csv and data/crest.csv; run scripts/farfit.py, scripts/crest.py")
            n0 = len(rows)
            rows = [r for r in rows if fit.get(r["file"], {}).get("ok") == "True"]
            n1 = len(rows)
            seen = lambda f: float(crest[f]["visible"]) >= args.min_crest
            rows = [r for r in rows if float(fit[r["file"]]["elevation"]) < args.night_elev or seen(r["file"])]
            print(f"far: {n0} cover the window, {n1} refit on the far field, {len(rows)} after the hidden-crest check")
            # Far-field refit, then the vertical lock onto the crest where the crest shows.
            lock = lambda f: np.array([[1, 0, 0], [0, 1, float(crest[f]["dy"]) if seen(f) else 0.0], [0, 0, 1]])
            Hof = lambda r: lock(r["file"]) @ np.array([float(v) for v in fit[r["file"]]["C"].split()]).reshape(3, 3) @ \
                np.array([float(v) for v in r["H"].split()]).reshape(3, 3)
            rows = [r for r in rows if covers(Hof(r), oriented_wh(scan[r["file"]]), rect)]  # shifts can expose an edge
        s_out = args.width / (rect[2] - rect[0])
        crop_size = (args.width, round((rect[3] - rect[1]) * s_out))
        T = np.array([[s_out, 0, -rect[0] * s_out], [0, s_out, -rect[1] * s_out], [0, 0, 1.0]])

    if args.people == "without":
        people = load(ROOT / "data" / "people.csv")
        col = f"area_{args.crop}" if args.crop in CROPS else "area_full"
        if args.crop and args.crop not in CROPS:
            raise SystemExit("--people without needs a named crop (people.csv measures those windows)")
        missing = [r["file"] for r in rows if r["file"] not in people]
        if missing:
            raise SystemExit(f"{len(missing)} frames have no people.csv row; run scripts/people.py")
        rows = [r for r in rows if float(people[r["file"]][col]) <= args.people_thr]

    qual, crest = load(args.quality), load(args.crest)
    far = args.crop == "far"
    if (args.pick == "score" or far) and any(r["file"] not in qual for r in rows):
        raise SystemExit("frames missing from data/quality.csv; run scripts/quality.py")
    if far:
        n0 = len(rows)
        rows = [r for r in rows if float(qual[r["file"]]["branch"]) <= args.max_branch]
        print(f"far: {n0 - len(rows)} dropped for branches in the sky, {len(rows)} left")
    q = lambda r, k: float(qual[r["file"]][("far_" if far and k != "branch" else "") + k])
    sharp_med = {}
    for r in rows:
        sharp_med.setdefault(r["datetime"][:10], []).append(q(r, "sharp"))
    sharp_med = {d: float(np.median(v)) for d, v in sharp_med.items()}
    phase = lambda r: float(solar[r["file"]]["phase"]) if r["file"] in solar else None
    mid = float(np.median([p for p in map(phase, rows) if p is not None])) if rows else 0.0

    def score(r, use_phase):
        if args.pick == "inliers":
            return int(r["inliers"])
        vis = float(crest[r["file"]]["visible"]) if r["file"] in crest else 0.5
        s = 2 * vis - 4 * (q(r, "clip_hi") + q(r, "clip_lo")) + 0.3 * (q(r, "sharp") - sharp_med[r["datetime"][:10]])
        if use_phase and phase(r) is not None:
            d = abs(phase(r) - mid) % 360
            s -= args.phase_weight * min(d, 360 - d) / 90
        return s

    if args.order == "date":
        best = {}
        for r in rows:
            day = r["datetime"][:10]
            if day not in best or score(r, True) > score(best[day], True):
                best[day] = r
        seq = [best[d] for d in sorted(best)]
    else:
        rows.sort(key=lambda r: r["datetime"])
        moments, last_t = [], None
        for r in rows:
            t = dt.datetime.fromisoformat(r["datetime"])
            if last_t is not None and (t - last_t).total_seconds() <= args.bracket_s:
                if score(r, False) > score(moments[-1], False):
                    moments[-1] = r
            else:
                moments.append(r)
            last_t = t
        seq = sorted((r for r in moments if r["file"] in solar), key=lambda r: float(solar[r["file"]]["phase"]))

    with open(out + ".csv", "w", newline="") as fo:  # what went in, for checking picks
        w = csv.writer(fo)
        w.writerow(["frame", "file", "datetime", "score"])
        w.writerows([i, r["file"], r["datetime"], round(score(r, args.order == "date"), 3)] for i, r in enumerate(seq))

    frames_dir = pathlib.Path(out + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    font = ImageFont.load_default(size=max(16, args.width // 30))
    rendered = []
    for i, r in enumerate(seq):
        if args.crop:
            src = load_rgb(pathlib.Path(args.dir) / r["file"])
            Hc = T @ Hof(r)
            mag = magnification(Hc, src.shape[1], src.shape[0])
            if mag < 0.5:  # zoomed shot: shrink first so the warp doesn't alias
                k = 2 * mag
                src = cv2.resize(src, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                Hc = Hc @ np.diag([1 / k, 1 / k, 1.0])
            rendered.append((cv2.warpPerspective(src, Hc, crop_size, flags=cv2.INTER_LANCZOS4), None))
        elif args.source == "originals":
            src = load_rgb(pathlib.Path(args.dir) / r["file"])
            rw, rh = oriented_wh(scan[args.ref])
            size = (args.width, round(rh * args.width / rw))
            Hc = np.diag([args.width / rw, args.width / rw, 1.0]) @ np.array([float(v) for v in r["H"].split()]).reshape(3, 3)
            mag = magnification(Hc, src.shape[1], src.shape[0])
            if mag < 0.5:  # zoomed shot: shrink first so the warp doesn't alias
                k = 2 * mag
                src = cv2.resize(src, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                Hc = Hc @ np.diag([1 / k, 1 / k, 1.0])
            valid = cv2.warpPerspective(np.full(src.shape[:2], 255, np.uint8), Hc, size, flags=cv2.INTER_NEAREST)
            rendered.append((cv2.warpPerspective(src, Hc, size, flags=cv2.INTER_LANCZOS4), valid))
        else:
            stem = pathlib.Path(r["file"]).stem
            im = Image.open(pathlib.Path(args.aligned) / f"{stem}.jpg")
            size = (args.width, round(im.height * args.width / im.width))
            valid = Image.open(pathlib.Path(args.aligned) / f"{stem}_mask.png").resize(size, Image.NEAREST)
            rendered.append((np.asarray(im.resize(size, Image.LANCZOS)), np.asarray(valid)))
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(seq)}] rendered", flush=True)

    if args.deflicker:
        stats = np.array([lab_stats(rgb, valid) for rgb, valid in rendered])
        phases = [float(solar[r["file"]]["phase"]) if r["file"] in solar else 90.0 for r in seq]
        images = deflicker(rendered, stats, phases, args.deflicker, args.phase_tol)
        after = np.array([lab_stats(rgb, valid) for rgb, (_, valid) in zip(images, rendered)])
        jump = lambda st: np.median(np.abs(np.diff(st[:, 0])))
        print(f"deflicker: median frame-to-frame L jump {jump(stats):.1f} -> {jump(after):.1f} (of 255)")
    else:
        images = [rgb for rgb, _ in rendered]

    frames = []
    for i, (r, rgb) in enumerate(zip(seq, images)):
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        label = f"{r['datetime'][:16].replace('T', ' ')}"
        if args.order == "solar":
            s = solar[r["file"]]
            label += f"  sun {float(s['elevation']):+.1f}°  sunrise {float(s['minutes_from_sunrise']):+.0f} min"
        d.text((12, im.height - font.size - 14), label, fill="white", font=font, stroke_width=2, stroke_fill="black")
        im.save(frames_dir / f"{i:05d}.jpg", quality=args.jpeg_q)
        frames.append(im.resize((640, round(im.height * 640 / im.width))) if args.width > 960 else im)

    frames[0].save(out + ".gif", save_all=True, append_images=frames[1:], duration=int(1000 * args.hold), loop=0)
    # MP4: hold, then crossfade into the next photo; raw RGB piped to ffmpeg.
    w, h = (v + v % 2 for v in Image.open(frames_dir / "00000.jpg").size)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(args.out_fps), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                           out + ".mp4"], stdin=subprocess.PIPE)
    n_fade = round(min(args.fade, args.hold) * args.out_fps)
    n_still = max(1, round(args.hold * args.out_fps) - n_fade)
    nxt = lambda i: np.asarray(Image.open(frames_dir / f"{i:05d}.jpg").convert("RGB").crop((0, 0, w, h)), np.float32)
    cur = nxt(0)
    for i in range(len(seq)):
        for _ in range(n_still):
            ff.stdin.write(cur.astype(np.uint8).tobytes())
        if i + 1 < len(seq):
            new = nxt(i + 1)
            for k in range(1, n_fade + 1):
                a = k / (n_fade + 1)
                ff.stdin.write((cur * (1 - a) + new * a).astype(np.uint8).tobytes())
            cur = new
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    print(f"{args.standpoint}/{args.order}: {len(seq)} frames -> {out}.mp4")


if __name__ == "__main__":
    main()
