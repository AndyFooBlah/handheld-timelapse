# Variants

All variants draw from one pool of aligned, exposure-normalised frames (each one deflickered against similarly lit
neighbours). A variant is one choice along each axis below.

| Axis | Options | Notes |
|---|---|---|
| Standpoint | `oak` (framed by foreground trees), `open` (a few metres away, clear view), `all` | Each standpoint keeps its own foreground, so mixing them makes the foreground jump. `all` suits the far crop, where the foreground is cut away. |
| Crop | full reference frame, `far` (the distant skyline only), or `x0,y0,x1,y1` | `far` needs only the distant view to align, so both standpoints and zoomed shots feed it. Its alignment is refitted on the skyline band (`farfit.py`) and locked to the crest (`crest.py`). Frames where cloud hides the crest, or branches hang into the sky, are dropped. |
| Order | `date`, `solar` | Solar order runs pre-dawn dark → sunrise → midday → sunset, with dates mixed. |
| People | `with`, `without` | `without` drops frames where people cover more than 0.1% of the shown view. YOLO masks at two input sizes are mapped into the reference, so people outside the crop don't count. |

## Solar ordering key

Minutes after sunrise works for the morning, but evening frames would land at different points, because day length
varies by hours over the year. So the key is a solar phase built from sun elevation, with separate branches for
morning and afternoon:

- morning (sun rising): `phase = elevation`, from about −18° (astronomical twilight) up to the noon maximum
- afternoon (sun setting): `phase = 180 − elevation`, so sunset frames follow midday and the evening dims toward night

## Frame choice

- Date order: one frame per day, the best by a quality score (`quality.py`, `preview.py --pick score`). The score
  rewards a visible crest, few clipped pixels and sharpness relative to the day's other shots. It also prefers a
  solar phase close to the sequence's median, so a day with both a dawn shot and a noon shot keeps the one lit like
  its neighbours.
- Solar order: several shots from one morning are distinct points in solar time, so a day can contribute more than
  one. Exposure brackets (same minute, different exposure) still collapse to one, by the same score without the
  phase term.
