"""Find Apple Photos images taken near the timelapse spot; write data/candidates.csv and a UUID list.

Metadata only: reads Photos.sqlite via osxphotos and downloads nothing.
Run with the osxphotos tool's Python:
    ~/.local/share/uv/tools/osxphotos/bin/python -I scripts/candidates.py [--radius 300]
"""
import argparse
import collections
import csv
import math
import pathlib
import sys

import osxphotos

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import SPOT

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data"


def haversine_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=300, help="metres from SPOT")
    args = ap.parse_args()

    photos = osxphotos.PhotosDB().photos(images=True, movies=False)
    rows, no_gps = [], 0
    for p in photos:
        lat, lon = p.location
        if lat is None:
            no_gps += 1
            continue
        d = haversine_m(SPOT, (lat, lon))
        if d > args.radius:
            continue
        rows.append({
            "uuid": p.uuid,
            "date": p.date.isoformat(),
            "dist_m": round(d, 1),
            "lat": lat,
            "lon": lon,
            "width": p.width,
            "height": p.height,
            "camera": p.exif_info.camera_model if p.exif_info else "",
            "burst": p.burst,
            "hdr": p.hdr,
            "local": not p.ismissing,
            "filename": p.original_filename,
        })
    rows.sort(key=lambda r: r["date"])

    OUT.mkdir(exist_ok=True)
    if rows:
        with open(OUT / "candidates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / "candidate_uuids.txt").write_text("".join(r["uuid"] + "\n" for r in rows))

    days = collections.Counter(r["date"][:10] for r in rows)
    years = collections.Counter(r["date"][:4] for r in rows)
    print(f"library images={len(photos)} no_gps={no_gps}")
    print(f"within {args.radius:.0f} m: {len(rows)} images on {len(days)} distinct days")
    if rows:
        print(f"span {rows[0]['date'][:10]} .. {rows[-1]['date'][:10]}")
        print("per year:", dict(sorted(years.items())))
        print(f"not downloaded locally: {sum(not r['local'] for r in rows)}")
        print("cameras:", dict(collections.Counter(r["camera"] for r in rows).most_common()))
        print("days with >1 shot:", sum(1 for n in days.values() if n > 1))


if __name__ == "__main__":
    main()
