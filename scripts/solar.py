"""Sun position and timing for every scanned image; writes data/solar.csv.

phase: sun elevation on the morning branch, 180 - elevation in the afternoon, so sorting by phase runs
night -> dawn -> sunrise -> noon -> sunset -> dusk regardless of date (see docs/variants.md).

    uv run python -I scripts/solar.py
"""
import csv
import datetime as dt
import pathlib
import sys

from astral import Observer
from astral.sun import azimuth, elevation, sun

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
import config

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPOT = Observer(latitude=config.SPOT[0], longitude=config.SPOT[1])  # sea-level horizon, matches published sunrise tables
TZ = config.TZ


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "scan.csv")))
    out = []
    for r in rows:
        if not r["datetime"]:
            continue
        t = dt.datetime.fromisoformat(r["datetime"])
        t = t.replace(tzinfo=TZ) if t.tzinfo is None else t
        s = sun(SPOT, date=t.astimezone(TZ).date(), tzinfo=TZ)
        el = elevation(SPOT, t)
        morning = t < s["noon"]
        out.append({
            "file": r["file"],
            "datetime": r["datetime"],
            "status": r["status"],
            "elevation": round(el, 2),
            "azimuth": round(azimuth(SPOT, t), 1),
            "morning": morning,
            "phase": round(el if morning else 180 - el, 2),
            "minutes_from_sunrise": round((t - s["sunrise"]).total_seconds() / 60, 1),
            "minutes_to_sunset": round((s["sunset"] - t).total_seconds() / 60, 1),
            "sunrise": s["sunrise"].isoformat(timespec="minutes"),
            "sunset": s["sunset"].isoformat(timespec="minutes"),
        })
    with open(ROOT / "data" / "solar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    cand = [o for o in out if o["status"] in ("candidate", "unknown")]
    bins = [(-90, -12, "night/astro+nautical twilight"), (-12, -6, "nautical twilight"), (-6, 0, "civil twilight"),
            (0, 10, "sun 0-10°"), (10, 90, "sun >10°")]
    print(f"{len(out)} images, {len(cand)} candidates/unknown")
    for half in (True, False):
        counts = {label: sum(lo <= o["elevation"] < hi for o in cand if o["morning"] == half) for lo, hi, label in bins}
        print("morning  " if half else "afternoon", counts)


if __name__ == "__main__":
    main()
