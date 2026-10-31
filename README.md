# handheld-timelapse

Turn years of casual, handheld phone photos of one view into a timelapse. The photos can come from different phones
and lenses, from a few metres to either side, and in any light. The pipeline finds the photos of the view, aligns
them to a reference shot, removes people, picks one per day, evens out the exposure, and renders date-stamped videos
in date order or ordered by the height of the sun. It can also retime a video to a music track.

No photos are included. Bring your own.

## Requirements

- macOS on Apple Silicon is what it was built on (feature matching runs on MPS; it falls back to CPU, slowly)
- [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/) and [exiftool](https://exiftool.org/)
- Optional: [osxphotos](https://github.com/RhetTbull/osxphotos), to list candidates straight from an Apple Photos
  library (`candidates.py`). It needs Full Disk Access; exporting from Photos by hand works just as well.

## Setup

1. Export the originals of everything you shot near the view into `data/photos/` (in Photos: select them on the map,
   File → Export → Export Unmodified Originals).
2. `cp timelapse.example.toml timelapse.toml` and fill it in: where you stand, which way you face, your reference
   shot, and the crop of the distant view.
3. `uv sync`

`data/`, `timelapse.toml` and `exclude.txt` are gitignored: they hold your photos, their EXIF (location, phone model)
and where the view is.

## Pipeline

Run each step with `uv run python -I scripts/<name>.py`. All of them read and write `data/`.

1. `scan.py`: reads EXIF from `data/photos/` and scores each photo by distance and heading → `scan.csv`
2. `align.py`: matches each candidate to the reference (ALIKED + LightGlue) and keeps those whose homography is
   plausible → `align.csv`, warped frames in `aligned/`
3. `rescore.py --max-rot-err 0.7`: re-gates the saved homographies by rotation consistency and re-warps
4. `chain.py --resume`: aligns the misses (dim dawn shots, odd zooms) through similar frames that did align
5. `solar.py`, `standpoint.py`: sun elevation and solar phase per frame; which standpoint it was taken from
6. `people.py`: YOLO person masks, mapped into the reference → `people.csv`
7. `farfit.py`: refits the far crop on the distant skyline band → `farfit.csv`
8. `crest.py`: locks the far crop to the skyline crest and flags frames where cloud or haze hides it → `crest.csv`
9. `quality.py`: clipping, sharpness and branch intrusion per frame → `quality.csv`
10. `preview.py [--standpoint oak|open|all] [--order date|solar] [--crop far] [--people with|without]`:
    GIF + MP4 previews, one scored pick per day, deflickered, crossfaded, date-stamped. `<out>.csv` lists the
    photos used. Photos listed in `exclude.txt` (see `exclude.example.txt`) are never used.
11. `zsh scripts/final.sh`: the 4K renders → `data/final/`
12. `zsh scripts/final_music.sh`: the same videos retimed to tracks in `data/music/` (via `add_music.py`)

`contact_sheet.py` makes thumbnail grids for checking a step by eye. [docs/variants.md](docs/variants.md) explains
the variants and the solar ordering; [docs/music.md](docs/music.md) has the Suno prompts and workflow used for the
example videos.

## Licence

Apache-2.0, see [LICENSE](LICENSE). Built with Claude Code.
