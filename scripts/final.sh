#!/bin/zsh
# Final 4K renders, without people: far crop 3840x2160 (16:9), open full view 2880x2160 (4:3 at 4K height), both from
# the original photos. Writes data/final/<name>.mp4 plus a 1280 px phone copy, <name>_small.mp4.
#
#   zsh scripts/final.sh
set -e
cd "${0:A:h}/.."
mkdir -p data/final
render() {  # name, then preview.py args
  local name=$1; shift
  echo "== $name $(date -u +%H:%MZ)"
  uv run python -I scripts/preview.py --people without --jpeg-q 95 --out data/final/$name "$@"
  ffmpeg -y -loglevel error -i data/final/$name.mp4 -vf scale=1280:-2 -c:v libx264 -crf 27 -pix_fmt yuv420p \
    data/final/${name}_small.mp4
  rm -f data/final/$name.gif
}
render bay_date_4k   --standpoint all  --crop far --order date  --width 3840
render bay_solar_4k  --standpoint all  --crop far --order solar --width 3840
render open_date_4k  --standpoint open --order date  --source originals --width 2880
render open_solar_4k --standpoint open --order solar --source originals --width 2880
ls -la data/final/*.mp4
echo ALLDONE
