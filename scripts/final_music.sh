#!/bin/zsh
# Music versions of the 4K finals: data/final/<name>_music.mp4 over data/music/<track>.wav, plus a 1280 px phone copy.
# These tracks were all ~2:15 and loud from the first bars, so photos are spread evenly (--even) rather than
# anchoring sunrise to an arrival, and the bay videos (97-99 photos) may hold up to 1.4 s so the whole track plays.
#
#   zsh scripts/final_music.sh
set -e
cd "${0:A:h}/.."
music() {  # final name, track name, then add_music.py args
  local name=$1 track=$2; shift 2
  echo "== $name $(date -u +%H:%MZ)"
  uv run python -I scripts/add_music.py data/final/$name data/music/$track.wav --even "$@"
  ffmpeg -y -loglevel error -i data/final/${name}_music.mp4 -vf scale=1280:-2 -c:v libx264 -crf 27 -pix_fmt yuv420p \
    -c:a aac -b:a 192k -movflags +faststart data/final/${name}_music_small.mp4
}
music bay_solar_4k  bay_solar  --max-hold 1.4
music bay_date_4k   bay_date   --max-hold 1.4
music open_solar_4k open_solar
music open_date_4k  open_date
ls -la data/final/*_music*.mp4
echo ALLDONE
