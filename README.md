# rawpilot

*Written entirely by AI (Claude), directed and reviewed by a human.*

My personal tool for one specific workflow: Nikon Z8 raw files, RawTherapee, Linux. It takes a
card dump of NEFs, pre-rates it so RawTherapee's file browser opens on a sorted shoot, and
pre-renders a few "looks" of each keeper to flip through before editing. It is not a general
photo tool, it assumes my camera, my software and my habits, and it is not supported.

## Why it is public

Some of the steps worked out here may be useful to someone building their own: reading the
camera's AF point straight from the NEF, judging frames on a neutral raw development rather
than the camera JPEG, grouping bursts and repeat attempts at a shot, reading how open eyes are
(through glasses where the raw allows), writing ratings RawTherapee reads without touching
the raw files, and a keyboard-driven review of pre-rendered looks. Take what helps.

## Ratings

Every frame gets a star rating, and the frames worth a look get a colour label:

| Rating | Label | Meaning |
| --- | --- | --- |
| ★★★★★ | Green | **Portfolio** — among the strongest photos of the whole shoot (ranked across the shoot, about the top 30% of keepers) |
| ★★★★ | Green | **Keeper** — a good photo worth editing |
| ★★★ | Yellow | **Ordinary** — the best of its shot, a fine record of the moment but unremarkable as a photograph |
| ★★ | Purple | **Best available** — the best of its shot although technically flawed (missed focus, blur, unrecoverable exposure); kept visible because nothing replaces it |
| ★★ | Blue | **Alternate** — another attempt at a shot whose best frame is elsewhere (its reason names that file) |
| ★ | — | Another usable frame of a burst whose best frame is rated above |
| (none) | — | Clearly worse than its burst's best (soft, blinked, misfired) |

So filtering RawTherapee to ≥3★ shows one frame per distinct shot, ≥4★ the keepers, and
5★ the highlights. Every shot keeps at least one frame at ≥2★: rawpilot demotes, it never
hides.

A rating rawpilot did not write is never overwritten, and a pp3 you have edited keeps its
processing. Each image's reasons are in its `.rpl` file and in the shoot's report.

## How it works

```
scan       the NEF headers are read in process: capture time, the camera's AF box and exposure
measure    the embedded full-size JPEG is decoded: sharpness at the AF point, clipping,
           a perceptual hash, a 1024 px thumbnail
develop    each raw is also given a neutral automatic development (libraw), so Gemini
           judges what an edit can make of a frame rather than the camera's JPEG look
group      near-identical frames seconds apart form bursts; bursts close in time, scenes
burst      Gemini picks each burst's best frame and judges every frame of it
faces      for people and wildlife: faces are located, cropped at full resolution, the eye
           strip of each is located and read for how open each eye is (through glasses where
           the raw allows) and how a person's frames order by it, plus expression and
           eyewear; the frame whose eyes are closest to each person's best, weighing those
           anyone would notice first (the main subject's first, none behind dark lenses),
           and the best faces wins
scene      Gemini compares the burst winners of nearby scenes side by side, merges repeat
           attempts at the same shot into one take, and tiers each take
shortlist  every keeper of the shoot is ranked together, so 5★ means the same everywhere
rate       the rule table turns all of the above into stars and colour labels
look       (process only) a pp3 rawpilot creates also opens at a corrected exposure: frames
           shot dark on purpose (negative exposure compensation) are brightened back
write      ratings go where RawTherapee reads them; each image gets a .rpl record
report     .rawpilot/report.html shows every frame as Gemini saw it, with its reasons
```

Ratings go wherever RawTherapee reads them, per its `ThumbnailRankColorMode`: `[General]
Rank` and `ColorLabel` in `DSC_xxxx.NEF.pp3` (a missing pp3 is created from your default
processing profile), or `xmp:Rating` and `xmp:Label` in the XMP sidecar. Each image also gets
`DSC_xxxx.NEF.rpl`: what rawpilot wrote, so it can tell its own ratings and pp3s from yours,
and what it measured and decided. Measurements, renders and Gemini answers are cached in
`<dir>/.rawpilot/`, so re-running costs nothing unless something changed. The NEFs are only
ever read, and nothing is deleted.

## Commands

```
rawpilot cull <dir>       # rate every frame and write the report
rawpilot process <dir>    # cull, plus an exposure fix and pre-rendered looks for review
rawpilot review <dir>     # a local, keyboard-driven web page to flip through looks and rate
rawpilot finish <dir>     # archive the keepers and export JPEGs (my archive layout only)
rawpilot scan <dir>       # list readable shots with capture time and AF box
```

The details of each are in [docs/usage.md](docs/usage.md); the design — invariants, how each
judgement is made and why, known limitations — is in [docs/design.md](docs/design.md).

## Building

```
cargo install --path .
```

It needs `dcraw_emu` (libraw) and `rawtherapee-cli` on `PATH`, both able to decode your
NEFs, and a Gemini API key in `GEMINI_API_KEY` unless run with `--local-only` (a cold run
costs about $1 per 1,000 frames).

## License

Copyright (C) 2026 Andrey Kartashov. [GPL-3.0-or-later](LICENSE). The Film profiles in `profiles/` are RawTherapee's own, under the
same license.
