# rawpilot usage

The commands in full. The README has the overview; [design.md](design.md) has the reasoning.

## process and cull

```
rawpilot cull <dir>               # rate, and write <dir>/.rawpilot/report.html
rawpilot process <dir>            # cull, plus the look and pre-rendered look options
rawpilot cull <dir> --dry-run     # decide and report, write no rating (or pp3, or render)
rawpilot cull <dir> --local-only  # no Gemini: sharpest frame of each burst, nothing uploaded
rawpilot scan <dir>               # list readable shots with capture time and AF box
```

`cull` writes ratings only. A pp3 that is missing is created from your default profile
without the exposure look, and the processing of a pp3 that exists is never changed.
`process` does what `cull` does, then also gives pp3s it creates or still owns the exposure
look, and renders every look option (`[[review.option]]` in the config: Camera, Film, Vivid,
Warm, Warm-sun, Sunny, Portrait, Overcast, Dusk; the built-in profiles are in `profiles/`) of
each shot rated at least `review.prerender_min_rating` stars into `<dir>/.rawpilot/looks/`: a
fit view and 1:1 crops of the AF box and any faces per render, for `rawpilot review` to show
without rendering anything. A render is made again only when something it was made from
changed (the shot, the option's profile, the exposure look, the render settings,
RawTherapee's default profile), so retuning a profile and reinstalling renders just the
affected ones; the run reports `preprocessing N/M` as it goes and, if any render failed, ends
in an error and retries only those next time. After a run with no failure, the renders of
options no longer in the config are removed from `<dir>/.rawpilot/looks/`. A look chosen or a
rating set in the review UI is yours: neither command overwrites it.

## review

```sh
rawpilot review <dir> [--port N] [--open]  # prints http://127.0.0.1:PORT/?t=TOKEN; open it in a browser
```

`--open` also hands that URL to `xdg-open`, so the default browser opens the page; if
`xdg-open` cannot be started, a warning is logged and the URL is still printed.

`review` serves a local page (127.0.0.1 only; the token in the printed URL signs the browser
in) for going through the shoot: a large fit view of the current shot in the middle of the
window (at most `review.centre_width_pct` of its width, narrower for a portrait), the AF and
face crops magnified (`review.region_zoom`) on both sides of it, most important nearest the
middle (a window too narrow for side columns puts them below), the file name, stars and
options at the top centre, and a filmstrip. The shots rated at least
`review.prerender_min_rating` are checked at launch and any missing render, or one made from
inputs other than today's (a retuned look profile), is made in the background, the shots
nearest the one on screen first; such a render is not shown or chosen until it is replaced.
The page opens on the first shot that is ready (a full progress page shows only while none
is) and the others join the filmstrip as they land, with `rendering N/M` in the bar; a filter
that admits unrendered shots does the same. RawTherapee's default profile is resolved in the
background at launch, and only a render, or the first rating of a shot that has no pp3, waits
for it. `review` remembers the last shot viewed in `<dir>/.rawpilot/review-state.json` and
opens there next time (on the next shot the filter shows when that one is hidden, on the
first when none follows, or when the file is missing or unreadable). The bar shows the scene
on screen (`scene 4/27`); scenes are the pipeline's (`group.scene_gap_s`).

Keys: `j`/`k` or the arrows step, `]` goes to the first shot of the next scene and `[` to the
start of this scene, then of the previous one, `J`/`K` to the next/previous shot with no look
chosen, `Home`/`End` to the first/last shot, `PageDown`/`PageUp` ten shots on/back (all of
these count the shots the filter shows), an option's letter (set per `[[review.option]]`; any
ASCII letter but `f j k`, in either case, which the page uses; the config is refused
otherwise, by `process` and `cull` too) only *views* it, `Tab` flips back to the previous
view, `Enter`/`Space` chooses the viewed option and steps on (only after an option key was
pressed on that shot), `0`-`5` rate, `f` steps the rating filter up a star (all, 2, 3, 4, 5;
it stops there), `F` down (it stops at all), `?` lists the keys. Replacing a pp3 edited in
RawTherapee asks first (`y`/`n`).

## finish

```sh
rawpilot finish <dir> --event 2026-11-01-autumn [--min-rating 3] [--media-root PATH] [--dry-run]
```

`finish` is for after `review`: it archives what you kept and files its JPEGs. It fits one
particular archive layout and an `import-media` script that is not part of this repository.
The keepers are the shots whose rating RawTherapee reads now (pp3 `Rank` or XMP, per its
`ThumbnailRankColorMode`) is at least `--min-rating` (default 3, one threshold). `--event` is
required: one folder name (no `/`, not `.` or `..`, no leading `-`, no control characters, at
most 100 characters). `--media-root` defaults to the `MEDIA_ROOT` environment variable and
must already hold a `photos-RAW` folder (an unmounted drive's empty mountpoint is refused;
`photos-RAW` is never created). In this order:

1. each keeper is exported to `<dir>/.rawpilot/export/<event>/<NEF name>.jpg` by
   `rawtherapee-cli` through that shot's own pp3, at JPEG quality 100 without chroma
   subsampling, carrying the camera's metadata unchanged;
2. only if every export worked, each keeper's NEF (the bytes, not a link), `.pp3`, `.rpl` and
   `.xmp` sidecar (when present) are copied to `<media-root>/photos-RAW/<event>/`, with
   `rawpilot/summary.json` (every shot of the folder: file, capture time, the rating rawpilot
   wrote, the current rating, the look chosen in `review`, keeper yes/no),
   `rawpilot/gemini-answers.jsonl` (the shoot's cached Gemini answers) and
   `rawpilot/report.html` beside them. Renders, thumbnails and measurements are not archived;
3. `import-media --media-root <media-root> <the JPEGs>` files the JPEGs under `photos/` by
   date, and its output is shown.

Nothing on the drive is ever replaced. A file already there with identical content counts as
done, so running again after an interruption completes the archive and a second run changes
nothing. A file there with different content (including a summary written for another
`--min-rating`, or a pp3 edited since) is refused and listed; `finish` carries on with what it
safely can and exits non-zero. The report ends with the photos and videos in the folder that
`finish` neither exports nor imports (`.MP4`, `.MOV`, camera `.JPG` ...), so videos are not
forgotten. The working folder's NEFs, pp3s and records are only read. `--dry-run` prints the
plan, what the drive holds for each file, and what would be refused, and writes nothing
anywhere (no lock either). Close RawTherapee first: it may write back the pp3s it holds.

## Everywhere

Only one rawpilot command works in a folder at a time (`process`, `cull`, `review` and
`finish` each hold `<dir>/.rawpilot/lock`); a second one refuses at once and names the first.

Run it with RawTherapee closed, or at least without the folder open in it: RawTherapee may
write back the pp3s it holds in memory.

Settings are in [config/default.toml](../config/default.toml); `--config FILE` replaces it
with a complete copy.

`tools/compare.py` compares a cull with a reference selection from the same shoot (and,
optionally, the frames that were edited) per frame, burst and take; see its header for usage.
