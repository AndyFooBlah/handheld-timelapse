# Music with Suno

These are the prompts used for the example videos: classical throughout, with the sun-height versions rising from
quiet to awake and exuberant in the manner of a Vivaldi concerto, and the date versions flowing and changing. The bay
(tight crop) tracks are scored for a chamber group, the full-view tracks for orchestra.

Suno blocks real artist and composer names in prompts, so the Vivaldi feel is described instead: Italian Baroque
concerto, solo violin, harpsichord continuo, ritornello, rushing semiquavers.

## How to make each track

1. Create, Instrumental on.
2. Paste the **Prompt** into the prompt box, and the **Title**. Suno no longer has a separate structure box, so each
   prompt describes the shape of the piece in prose, section by section, instead of using `[Section]` tags.
3. Generate a couple of takes, pick the one whose arc fits best, download it (WAV if your plan offers it, else MP3).
4. Save it as `data/music/<name>.wav` (or `.mp3`), using the names below.

Length doesn't need to be exact: `scripts/add_music.py` retimes the photos (0.45–1.25 s each) so the video ends with
the music, and in the solar videos it lands the sunrise photo on the music's arrival.

| Video | Photos | Fits music of | Sunrise photo |
|---|---|---|---|
| `bay_solar` | 99 | 1:15–2:00 | 54th, ~55% in |
| `open_solar` | 242 | 2:00–4:30 | 95th, ~40% in |
| `bay_date` | 97 | 1:15–2:00 | — |
| `open_date` | 237 | 2:00–4:30 | — |

Optional, to make each pair truly variants of one theme: make the wide (`open_`) track first, then use Suno's Cover on
it with the bay prompt, which keeps the melody and changes the scoring.

## Sun-height order: quiet → awake → exuberant

### `bay_solar`: dawn concerto, chamber scale

Title: `First Light (chamber)`

Prompt:
```
Instrumental Italian Baroque violin concerto for chamber ensemble: solo violin, string quartet, harpsichord continuo, theorbo. Period instruments, gut strings, natural acoustic room. Opens almost in silence: pianissimo sustained low strings and soft harpsichord, slow and hushed like the hour before dawn. The solo violin enters softly and the music slowly brightens, adding voices and tempo in a long crescendo. Just past halfway it breaks into a sparkling major-key Allegro: rapid violin runs, driving repeated-note bass, bold unison ritornello, joyful and exuberant, rising to a climax. Settles into a warm final cadence. No drums, no synths, no vocals, no choir.
```

### `open_solar`: dawn concerto, full orchestra

Title: `First Light`

Prompt:
```
Instrumental Baroque concerto grosso for full baroque orchestra: strings, harpsichord, oboes, recorders, bassoon, natural trumpets and timpani. Period instruments, concert hall acoustic. A three-stage awakening. First a hushed, slow minor-key Largo: sustained strings and soft continuo, pre-dawn stillness. Then an Andante that warms and brightens as woodwinds join, with birdsong-like recorder trills. About a third of the way in, trumpets and timpani enter for a jubilant major-key Allegro: rushing violin semiquavers, call and response between soloists and full ensemble, radiant and exuberant, building to a climax. Closes with a broad, golden cadence that relaxes into calm. No drum kit, no synths, no vocals, no choir.
```

## Date order: flowing and changing

### `bay_date`: piano and cello, intimate

Title: `Seasons at the Fence (duo)`

Prompt:
```
Instrumental flowing classical duo for grand piano and cello. Close-miked piano, natural room. Continuous rippling piano arpeggios at a moderate tempo under a long, lyrical cello melody that keeps evolving, like a theme and variations as the seasons turn. It opens with the flowing theme, then a shadowed minor-key variation, a brighter lifting one, a faster rippling one, and finally the theme returns for a gentle close. The harmony drifts between major and minor with gentle swells and ebbs, always moving, never repetitive. Intimate, reflective, warm. No drums, no synths, no vocals.
```

### `open_date`: theme and variations, chamber orchestra

Title: `Seasons at the Fence`

Prompt:
```
Instrumental classical theme and variations for chamber orchestra. Natural concert hall acoustic. A simple, singing theme in a lilting compound meter, legato, in constant flowing motion as the years pass. Strings and harp present the theme; the first variation passes it to flute, clarinet and oboe; then come a wistful minor variation, a playful pizzicato one, a broad sunlit one for the full ensemble, and a quiet wintry one, before a finale where the theme returns warm and full-hearted. Seamless transitions and gradual key changes, no abrupt stops. No drums, no synths, no vocals, no choir.
```

## Temple variants: Japanese colours with the Baroque

An alternative set that adds a Japanese Buddhist colour. For the example videos the takes from both sets were blended
into one track per video, saved under the four names above. If Suno opens with violins regardless, make the temple opening as its own short track, then Extend it
with the Baroque prompt (see below). The Japanese sounds (temple bell, singing bowl, shakuhachi, koto) carry the
stillness, the Baroque or classical writing carries the motion, and one pentatonic melody (the yo scale, which sits on
major-key harmony) links them so the blend sounds intended. Taiko and chant are left out: drums fight the Baroque
pulse and chant breaks Instrumental. Fusion prompts vary more between takes than plain ones (some drift into new-age
ambient), so generate three or four. Chamber versions are for the bay crop, orchestral ones for the full view.

### `bay_solar_temple`: temple dawn into a violin concerto, chamber scale

Title: `First Light (temple, chamber)`

Prompt:
```
Instrumental. Dawn at a Japanese temple turning into an Italian Baroque violin concerto, chamber scale: shakuhachi, temple bell, singing bowl, solo violin, string quartet, harpsichord continuo, theorbo. Opens in near silence: one deep temple bell strike and its long decay, then a solo shakuhachi playing a slow, breathy pentatonic phrase over a soft sustained string drone. The violin takes up the shakuhachi's melody and the quartet harmonises it; harpsichord enters and the tempo slowly rises in a long crescendo. Just past halfway it breaks into a bright major-key Allegro: rapid violin runs, driving repeated-note bass, the pentatonic theme now the ritornello, joyful and exuberant, the shakuhachi joining at the climax. Ends on a warm cadence and a final soft bell. Acoustic, natural room. No drums, no synths, no vocals, no choir.
```

### `open_solar_temple`: temple dawn into a concerto grosso, full orchestra

Title: `First Light (temple)`

Prompt:
```
Instrumental. Dawn at a Japanese temple turning into a Baroque concerto grosso for full baroque orchestra: temple bell, shakuhachi, koto, strings, harpsichord, oboes, recorders, bassoon, natural trumpets and timpani. Opens in temple stillness: a deep bell strike and its long decay, a slow, breathy pentatonic shakuhachi phrase over hushed sustained strings, a few plucked koto notes. Then an Andante that warms and brightens as recorders and oboes take up the shakuhachi's melody over harpsichord continuo. About a third of the way in, trumpets and timpani enter for a jubilant major-key Allegro: rushing violin semiquavers, the pentatonic theme now the ritornello, call and response between soloists and full ensemble, the shakuhachi soaring at the climax. Closes with a broad, golden cadence and a final soft bell. Concert hall acoustic. No drum kit, no synths, no vocals, no choir.
```

### `bay_date_koto`: koto, cello and shakuhachi, chamber scale

Title: `Seasons at the Fence (koto, chamber)`

Prompt:
```
Instrumental flowing chamber piece for cello, koto and shakuhachi with a soft string quartet. Natural room. A gentle, singing pentatonic theme that keeps changing like the turning seasons. It opens with rippling koto arpeggios under a long, lyrical cello melody; then the shakuhachi answers the cello; a shadowed minor variation; a bright, lilting variation; a sparse, quiet variation with a distant temple bell and long silences; and finally the theme returns warm and full for a gentle close. Western harmony under Japanese colours, always moving, never abrupt. Intimate, reflective. Acoustic. No drums, no synths, no vocals.
```

### `open_date_koto`: theme and variations with koto and shakuhachi, orchestra

Title: `Seasons at the Fence (koto)`

Prompt:
```
Instrumental classical theme and variations for orchestra with koto and shakuhachi as soloists: strings, harp, flute, clarinet, oboe, horn. Concert hall acoustic. A simple, singing pentatonic theme in a lilting compound meter, legato, in constant flowing motion as the years pass. Koto and strings present the theme; the shakuhachi and flute pass it between them; then come a wistful minor variation, a playful pizzicato one with koto, a broad sunlit one for the full orchestra, and a sparse wintry one with a distant temple bell and long silences, before a finale where the theme returns warm and full-hearted. Western harmony under Japanese colours, seamless transitions, gradual key changes, no abrupt stops. No drums, no synths, no vocals, no choir.
```

### When Suno ignores the opening

Suno blends a prompt into one overall sound rather than following it in order, so "bell first, then violins" usually
comes out as violins from bar one. Two passes work better:

1. A short track with only the opening, as tags:
   `Japanese temple dawn, deep temple bell, solo shakuhachi, singing bowl, slow, sparse, meditative, pentatonic, long silences, instrumental`,
   with `violin, strings, orchestra, harpsichord, drums, vocals` in Exclude Styles if it's there.
2. Extend the best take from where the opening feels finished, with the Baroque style as a short tag list, then
   download the whole song.

## Putting the music on

```
zsh scripts/final_music.sh
```

writes `data/final/<name>_music.mp4` and a 1280 px `_music_small.mp4` for all four, via `scripts/add_music.py`. The
example tracks are all about 2:15 and full-voiced from the first bars, with no quiet-to-loud arrival to
anchor sunrise on, so the script spreads the photos evenly (`--even`) and lets the bay videos hold up to 1.4 s per
photo so the whole track plays. For a future track with a clear arrival, drop `--even`: add_music.py then lands the
first above-horizon photo on the first moment the smoothed loudness comes within 3 dB of its peak, or on
`--anchor-at <seconds>`.
