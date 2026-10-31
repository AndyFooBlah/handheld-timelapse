"""Incrementally read EXIF from exported originals and score them against the spot.

Safe to run while an export is still writing: files modified in the last --settle seconds are skipped
and picked up on a later run. EXIF is cached in data/exif.jsonl, results go to data/scan.csv.

    python3 scripts/scan.py [--dir data/photos] [--radius 150] [--heading-tol 35]
"""
import argparse
import collections
import csv
import json
import math
import pathlib
import sys
import subprocess
import time

IMAGE_EXT = {".heic", ".jpg", ".jpeg", ".png", ".dng", ".tif", ".tiff"}
TAGS = [
    "-DateTimeOriginal", "-OffsetTimeOriginal", "-SubSecTimeOriginal",
    "-Composite:GPSLatitude", "-Composite:GPSLongitude",
    "-GPSImgDirection", "-GPSImgDirectionRef", "-GPSHPositioningError",
    "-ImageWidth", "-ImageHeight", "-Model", "-LensModel", "-FocalLengthIn35mmFormat",
    "-ExposureTime", "-FNumber", "-ISO", "-ExposureCompensation", "-Orientation",
]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import HEADING, MAG_DECLINATION, PHOTOS, SPOT

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "exif.jsonl"
OUT = ROOT / "data" / "scan.csv"


def haversine_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def load_cache():
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["SourceFile"]] = rec
    return cache


def read_exif(paths):
    """exiftool in batches; returns records keyed by absolute path."""
    out = []
    for i in range(0, len(paths), 200):
        batch = [str(p) for p in paths[i:i + 200]]
        res = subprocess.run(["exiftool", "-j", "-n", *TAGS, *batch], capture_output=True, text=True)
        if res.stdout.strip():
            out.extend(json.loads(res.stdout))
    return out


def score(rec, args):
    lat, lon = rec.get("GPSLatitude"), rec.get("GPSLongitude")
    heading = rec.get("GPSImgDirection")
    if heading is not None and rec.get("GPSImgDirectionRef") == "M":
        heading = (heading + MAG_DECLINATION) % 360
    # Front-camera headings are inconsistent (selfies and vista shots at one spot record the same ~275°),
    # so leave them unknown for the scene filter.
    front = "front" in (rec.get("LensModel") or "").lower()
    if front:
        heading = None
    w, h = rec.get("ImageWidth") or 0, rec.get("ImageHeight") or 0
    pano = min(w, h) > 0 and max(w, h) / min(w, h) > 2.2
    dist = haversine_m(SPOT, (lat, lon)) if lat is not None else None
    hdiff = angle_diff(heading, HEADING) if heading is not None else None
    near = dist is not None and dist <= args.radius
    facing = hdiff is not None and hdiff <= args.heading_tol
    # No-GPS or no-heading files stay "unknown" for the scene filter to decide.
    status = ("reject" if pano else
              "candidate" if near and facing else
              "unknown" if (dist is None or (near and hdiff is None)) else
              "reject")
    dt = rec.get("DateTimeOriginal", "")
    return {
        "file": pathlib.Path(rec["SourceFile"]).name,
        "status": status,
        "datetime": dt.replace(":", "-", 2).replace(" ", "T") + (rec.get("OffsetTimeOriginal") or ""),
        "dist_m": None if dist is None else round(dist, 1),
        "heading": None if heading is None else round(heading, 1),
        "heading_diff": None if hdiff is None else round(hdiff, 1),
        "gps_err_m": rec.get("GPSHPositioningError"),
        "width": rec.get("ImageWidth"),
        "height": rec.get("ImageHeight"),
        "orientation": rec.get("Orientation"),
        "model": rec.get("Model"),
        "lens": rec.get("LensModel"),
        "front": front,
        "pano": pano,
        "focal35": rec.get("FocalLengthIn35mmFormat"),
        "exposure": rec.get("ExposureTime"),
        "fnumber": rec.get("FNumber"),
        "iso": rec.get("ISO"),
        "ev_comp": rec.get("ExposureCompensation"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--radius", type=float, default=150, help="metres from SPOT")
    ap.add_argument("--heading-tol", type=float, default=35, help="degrees either side of ESE")
    ap.add_argument("--settle", type=float, default=60, help="skip files modified this recently (s)")
    args = ap.parse_args()

    now = time.time()
    files = [p.resolve() for p in pathlib.Path(args.dir).iterdir() if p.suffix.lower() in IMAGE_EXT]
    settled = [p for p in files if now - p.stat().st_mtime > args.settle]
    cache = load_cache()
    new = [p for p in settled if str(p) not in cache]
    if new:
        recs = read_exif(new)
        with open(CACHE, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
                cache[r["SourceFile"]] = r

    rows = sorted((score(cache[str(p)], args) for p in settled if str(p) in cache), key=lambda r: r["datetime"])
    if rows:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    by = collections.Counter(r["status"] for r in rows)
    cands = [r for r in rows if r["status"] == "candidate"]
    days = collections.Counter(r["datetime"][:10] for r in cands)
    print(f"images seen={len(files)} settled={len(settled)} newly read={len(new)} scored={len(rows)}")
    print(f"status: {dict(by)}  (radius {args.radius:.0f} m, heading {HEADING}±{args.heading_tol:.0f}°)")
    if cands:
        print(f"candidates span {cands[0]['datetime'][:10]} .. {cands[-1]['datetime'][:10]} on {len(days)} days;"
              f" {sum(n > 1 for n in days.values())} days have >1 shot")
        print("per year:", dict(sorted(collections.Counter(r["datetime"][:4] for r in cands).items())))
        print("models:", dict(collections.Counter(r["model"] for r in cands).most_common()))
        print("35mm focal:", dict(sorted(collections.Counter(r["focal35"] for r in cands).items(), key=lambda kv: kv[0] or 0)))


if __name__ == "__main__":
    main()
