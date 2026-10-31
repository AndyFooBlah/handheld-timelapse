"""Settings for one view, read from timelapse.toml at the repo root.

Copy timelapse.example.toml to timelapse.toml and fill it in. timelapse.toml is gitignored, since it holds where the
view is and the names of your own photos.
"""
import pathlib
import sys
import tomllib
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parent.parent
_path = ROOT / "timelapse.toml"
if not _path.exists():
    sys.exit(f"{_path} not found: copy timelapse.example.toml to timelapse.toml and fill it in")
_c = tomllib.loads(_path.read_text())

SPOT = (_c["spot"]["lat"], _c["spot"]["lon"])
HEADING = _c["spot"]["heading"]  # degrees true, the direction the view faces
MAG_DECLINATION = _c["spot"].get("mag_declination", 0.0)  # degrees east; only for files that report a magnetic heading
TZ = ZoneInfo(_c["spot"]["timezone"])
PHOTOS = ROOT / _c["photos"]["dir"]
REF = _c["reference"]["image"]  # every frame is aligned to this one
REF_WH = tuple(_c["reference"]["size"])  # its width and height in pixels, upright
CREST_REF = _c["reference"]["crest"]  # a crisp skyline silhouette, for locking the far crop
FAR_BAND = tuple(_c["reference"]["far_band"])  # rows (fractions) holding the skyline, for refitting the far crop
STANDPOINT_REFS = {"oak": _c["standpoints"]["oak"], "open": _c["standpoints"]["open"]}
CROPS = {k: tuple(v) for k, v in _c["crops"].items()}  # name -> (x0, y0, x1, y1) as fractions of the reference
