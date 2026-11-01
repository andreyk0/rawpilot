# Built-in profiles

Partial pp3s compiled into rawpilot (`src/option.rs`) and named in the config as
`partial = { builtin = "…" }`. Each holds only the keys it changes; rawpilot layers it over
the photographer's default profile.

- `film-iso-low.pp3`, `film-iso-medium.pp3`, `film-iso-high.pp3` — `film`. Unmodified
  copies of RawTherapee 5.13's bundled `Standard Film Curve - ISO Low/Medium/High.pp3`
  (© the RawTherapee authors, GPL-3.0-or-later, the license rawpilot is under too), kept
  here so a rawpilot build renders the same Film whatever RawTherapee version is installed.
  Several looks below start from Film's curve.
- `vivid.pp3` — `vivid`: a nature-postcard look, not a social-media one. Film's curve;
  Vibrance (pastels 30, saturated 10, skin protected); Local Contrast 0.25 at radius 80;
  Dehaze strength 8 with its saturation boost at 20.
- `warm.pp3` — `warm`: the scene as seen through bronze sunglasses. Film's curve a touch
  firmer; RGB curves lifting red (0.5 → 0.525) and lowering blue (0.5 → 0.465) at the
  midtones only, so the shift is relative to the shot's white balance and white stays white;
  Vibrance 20/10 with skin protected; Local Contrast 0.15.
- `warm-sun.pp3` — `warm_sun`: Warm and Sunny together: Warm's curve and colour shift with
  Sunny's Shadows & Highlights, Vibrance 20/8, Local Contrast 0.2.
- `portrait.pp3` — `portrait`: people. Film's curve with a softer top (0.66 → 0.86); a slight
  warm shift (red 0.51, blue 0.49 at the midtones); Shadows & Highlights 15/10; Vibrance
  10/0 with skin protected; Local Contrast off, so skin texture is not sharpened.
- `overcast.pp3` — `overcast`: flat grey light. A firmer S curve; a slight warm shift (red
  0.515, blue 0.48); Vibrance 25/10; Local Contrast 0.25; Dehaze 5. A first guess, tuned so
  far only on hazy sunny frames.
- `bright-sun.pp3` — `bright_sun`: hard midday light. Film's curve; Shadows & Highlights
  (shadows 30, highlights 20); Vibrance 20/5 and Local Contrast 0.2 so the opened shadows
  do not look faded.
- `dusk.pp3` — `dusk`: after sunset, a U-shaped histogram. A film-like curve that keeps the
  toe dark and rolls the top off (0.7 → 0.82, 0.9 → 0.95); highlight compression 40 and
  colour-propagation highlight recovery; Local Contrast 0.25 at radius 60; Vibrance 25/15;
  chroma noise reduction 15 with light luminance 10. Untried on night shots so far.
