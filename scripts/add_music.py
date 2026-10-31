"""Put a music track under a rendered timelapse, retiming the photos to fit the music.

Reads the frames and manifest preview.py saved (<name>_frames/, <name>.csv) and re-encodes them over the track:

- every photo's hold is stretched or squeezed (within --min-hold..--max-hold) so the photos end --tail seconds before
  the music does, with the last photo held while the music finishes;
- solar order is split at an anchor photo (default: the first with the sun above the horizon) so that it lands on the
  music's arrival, --anchor-at seconds, or by default the first moment the track's smoothed loudness comes within
  --arrival-db of its peak. Photos before the anchor share the time before the arrival, photos after it the rest.
  --even skips the anchor, for a track with no clear arrival (one that is loud from the first bars).

If the track is too long for the holds allowed, it is cut and faded at the end of the video; if it's too short, the
video runs past it at --min-hold and the music fades out early. Writes <out>.mp4 (H.264 CRF 18, AAC 256k).

    uv run python -I scripts/add_music.py data/final/bay_solar_4k data/music/bay_solar.wav [--anchor-at 41.5]
"""
import argparse
import csv
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def loudness(path, sr=8000, win=0.5):
    """Short-term loudness in dB, one value per win seconds."""
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, np.float32)
    n = int(sr * win)
    x = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-6), win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="preview.py --out path, e.g. data/final/bay_solar_4k")
    ap.add_argument("track")
    ap.add_argument("--anchor-file", help="solar order: photo to land on the arrival (default: first after sunrise)")
    ap.add_argument("--anchor-at", type=float, help="seconds into the track; default: detected from loudness")
    ap.add_argument("--arrival-db", type=float, default=3.0, help="arrival = loudness within this of its peak")
    ap.add_argument("--even", action="store_true", help="solar order too: spread the photos evenly, no anchor")
    ap.add_argument("--min-hold", type=float, default=0.45)
    ap.add_argument("--max-hold", type=float, default=1.25)
    ap.add_argument("--fade", type=float, default=0.35, help="crossfade between photos, seconds")
    ap.add_argument("--tail", type=float, default=2.5, help="seconds the last photo holds while the music ends")
    ap.add_argument("--out-fps", type=int, default=30)
    ap.add_argument("--width", type=int, help="scale the output down, for a quick test")
    ap.add_argument("--out")
    args = ap.parse_args()

    base = pathlib.Path(args.video)
    seq = list(csv.DictReader(open(f"{base}.csv")))
    frames = sorted((base.parent / f"{base.name}_frames").glob("*.jpg"))
    if len(frames) != len(seq):
        raise SystemExit(f"{len(frames)} frames but {len(seq)} manifest rows; re-render {base}")
    n, T = len(seq), duration(args.track)
    solar = {r["file"]: r for r in csv.DictReader(open(ROOT / "data" / "solar.csv"))}
    clamp = lambda h: float(np.clip(h, args.min_hold, args.max_hold))

    is_solar = "solar" in base.name and not args.even
    if is_solar:
        if args.anchor_file:
            a = next(i for i, r in enumerate(seq) if r["file"] == args.anchor_file)
        else:
            a = next(i for i, r in enumerate(seq) if float(solar[r["file"]]["elevation"]) > 0)
        if args.anchor_at is None:
            db, win = loudness(args.track)
            smooth = np.convolve(np.pad(db, (3, 2), mode="edge"), np.ones(6) / 6, "valid")  # 3 s; zero padding would be loud
            args.anchor_at = float(np.argmax(smooth >= smooth.max() - args.arrival_db) * win)
        holds = [clamp(args.anchor_at / a)] * a + [clamp((T - args.tail - args.anchor_at) / (n - a))] * (n - a)
        print(f"anchor {seq[a]['file']} ({seq[a]['datetime'][:16]}), photo {a + 1} of {n}, at {args.anchor_at:.1f} s "
              f"of {T:.1f} s; holds {holds[0]:.2f} s before, {holds[-1]:.2f} s after")
    else:
        holds = [clamp((T - args.tail) / n)] * n
        print(f"{n} photos over {T:.1f} s of music: hold {holds[0]:.2f} s")
    L = sum(holds) + args.tail
    if abs(L - T) > 1:
        print(f"note: video {L:.1f} s, track {T:.1f} s; the music {'is cut' if T > L else 'ends early'} with a fade")

    w, h = Image.open(frames[0]).size
    if args.width:
        w, h = args.width, round(h * args.width / w)
    w, h = w + w % 2, h + h % 2
    out = args.out or f"{base}_music"
    fade_out = min(4.0, min(L, T) / 4) if T > L + 1 else 0.5  # a track played to its end keeps its own ending
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(args.out_fps), "-i", "-", "-i", args.track, "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
         "-af", f"afade=t=out:st={min(L, T) - fade_out:.2f}:d={fade_out:.2f}", "-t", f"{L:.2f}",
         "-movflags", "+faststart", out + ".mp4"], stdin=subprocess.PIPE)
    load = lambda i: np.asarray(Image.open(frames[i]).convert("RGB").resize((w, h), Image.LANCZOS), np.float32)
    n_fade = round(args.fade * args.out_fps)
    cur, t_written, t_end = load(0), 0, 0.0  # frames written so far; where the current photo's hold ends, seconds
    for i in range(n):
        t_end += holds[i] + (args.tail if i == n - 1 else 0)
        last = i == n - 1
        stop = round(t_end * args.out_fps) - (0 if last else n_fade)  # cumulative, so rounding never drifts
        while t_written < stop:
            ff.stdin.write(cur.astype(np.uint8).tobytes())
            t_written += 1
        if not last:
            new = load(i + 1)
            for k in range(1, n_fade + 1):
                ff.stdin.write((cur * (1 - k / (n_fade + 1)) + new * (k / (n_fade + 1))).astype(np.uint8).tobytes())
                t_written += 1
            cur = new
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{n}] encoded", flush=True)
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out}.mp4 ({L:.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
