"""Contact sheet of scanned images, labelled with status / heading / distance / date / focal length.

    uv run --with pillow --with pillow-heif python -I scripts/contact_sheet.py [--max-dist 150] [--status candidate]
"""
import argparse
import csv
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # python -I doesn't add the script dir
from config import PHOTOS

ROOT = pathlib.Path(__file__).resolve().parent.parent
THUMB = 320


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PHOTOS))
    ap.add_argument("--scan", default=str(ROOT / "data" / "scan.csv"))
    ap.add_argument("--max-dist", type=float, default=150)
    ap.add_argument("--status", action="append", help="limit to these statuses (repeatable)")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "data" / "contact_sheet.jpg"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.scan))
            if (not r["dist_m"] or float(r["dist_m"]) <= args.max_dist)
            and (not args.status or r["status"] in args.status)]
    font = ImageFont.load_default(size=14)
    cell_h = THUMB + 40
    rows_n = (len(rows) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * THUMB, max(rows_n, 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    colour = {"candidate": (0, 120, 0), "reject": (180, 0, 0), "unknown": (160, 110, 0)}
    for i, r in enumerate(rows):
        im = ImageOps.exif_transpose(Image.open(pathlib.Path(args.dir) / r["file"]))
        im.thumbnail((THUMB, THUMB))
        x, y = (i % args.cols) * THUMB, (i // args.cols) * cell_h
        sheet.paste(im, (x + (THUMB - im.width) // 2, y))
        draw.text((x + 4, y + THUMB + 2), f"{r['status']} h={r['heading']} d={r['dist_m']}m f={r['focal35']}",
                  fill=colour.get(r["status"], "black"), font=font)
        draw.text((x + 4, y + THUMB + 20), f"{r['datetime'][:16]} {r['file']}", fill="black", font=font)
    sheet.save(args.out, quality=85)
    print(f"{len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
