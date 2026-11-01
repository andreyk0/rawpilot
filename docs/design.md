# rawpilot design

## Invariants

- NEFs are only ever read.
- Ratings go where RawTherapee reads them (its `ThumbnailRankColorMode`): the pp3's
  `[General] Rank/ColorLabel` in `procparams` mode, `xmp:Rating/Label` in `xmp` mode.
- A pp3 is only ever complete. RawTherapee fills a missing key with its built-in neutral
  value rather than the default profile, so a partial pp3 would silently reset the image's
  processing. A missing pp3 is created from the default profile as `rawtherapee-cli`
  resolves it; an existing one has at most its rating lines changed unless rawpilot owns it.
- A rating rawpilot did not write is never overwritten, and a pp3 the photographer has
  edited keeps its processing.
- A look chosen or a rating set in the review UI is the photographer's: it is recorded in
  the `.rpl`, and no batch command changes that pp3 or rating, even when the bytes are
  exactly what the batch would write.
- `cull` never changes the processing of an existing pp3; only `process` sets the look.
- Every take keeps its best frame at ≥2★. rawpilot demotes, it never hides.
- Re-running with unchanged inputs and config changes nothing and costs nothing.

## Signals

| Signal | Source | Notes |
| --- | --- | --- |
| Full-size preview | `JpgFromRaw` (8256×5504 on a Z8): SubIFD 0's `JPEGInterchangeFormat/Length` | Read straight from the NEF at that span. IFD0 and the later SubIFDs hold smaller previews. |
| Capture time | Exif `DateTimeOriginal` + `SubSecTimeOriginal` (1/100 s) + `OffsetTimeOriginal` | Bursts run at 7.5–20 fps. A frame without an offset is skipped. |
| AF box | MakerNote `AFInfo2` (version 0400): `AFAreaX/YPosition`, `AFAreaWidth/Height` in `AFImageWidth×Height` space | Only when `AFCoordinatesAvailable=1`; often absent. Another version gives no box. |
| Exposure | Exif `ExposureTime`, `FocalLength`, `ISO`, `ExposureBiasValue`; MakerNote `PictureControlData` (version 0310) `PictureControlName` | Negative compensation in sun is deliberate: the raw keeps the highlights. Another Picture Control version gives no name. |
| Raw data | libraw `dcraw_emu`, half size, linear | For the neutral development and raw clipping. |

`nef` reads these in process from the header region of each NEF (a TIFF file, little- or
big-endian) without touching the image data: ~0.03 ms per file against ~16 ms for exiftool.
Every offset in the file is bounds-checked, and a file whose Exif fields it cannot read is
skipped with a warning. The MakerNote only adds the AF box and Picture Control name: a
MakerNote that is absent, not Nikon's, truncated, or holds a field of an unknown version,
type or range leaves those unset (with a warning naming the file and the reason) rather than
dropping the shot, and layouts it does not know are never guessed.

## Judgement

**Sharpness** is Laplacian variance at full resolution: in a window around the AF box (at
least 384 px, or "featureless" when the window has no structure) and as the mean of the
sharpest tiles. Within a burst frames are compared on one figure: the AF window when every
member has one, otherwise tiles. It is only meaningful relative to the burst's sharpest.

**Grouping.** Consecutive frames within `burst_gap_s` and with similar perceptual hashes
form a burst; bursts within `scene_gap_s` form a scene.

**What Gemini sees.** A judge trained on finished photographs rates the rendering in front
of it, so it is shown a neutral automatic development of each raw rather than the camera
JPEG: exposure to a mid-grey key, a highlight shoulder, mild contrast and saturation, and
the share of the frame clipped in the raw. Flat renderings are marked down as readily as
dark ones, hence the contrast.

**Comparisons, not scores.** Asked for absolute scores, a language model compresses them;
side-by-side comparisons it makes consistently. Every pass is a comparison:

1. **Burst pass** — the frames of one burst (the sharpest 16 at most) with their measured
   sharpness: the best frame, and a verdict, aesthetic score, defects and reason per frame.
2. **Faces pass** — for people and wildlife bursts, five requests. Gemini first locates the
   faces (consistent person ids, the main subject marked). Each face is cut at full
   resolution from the same source the locate request showed — the raw when it was shown
   developed, else the camera preview, because Gemini's boxes follow that image's geometry
   and the preview's lens correction shifts it by tens of pixels — with one libraw decode
   per frame over the union of its faces, as ~768 px images: the whole face as it looks, and
   its eye band (30–85 % of the face box's height, a little wider than the box, and 1 % of
   the frame's size further out on every side to absorb the locate answer's error, which
   does not shrink with the face), rendered with a log curve that lifts the inside of
   tinted and dark lenses.
   Eyes behind sunglasses are legible in the raw where the JPEG's lens interior is blocks,
   but not in a band read at a glance, so the eyes are read from a *strip*. The *strip
   request* shows every eye band and asks for the box of the eye strip, like a carnival
   mask: from just above the eyebrows to the middle of the nose and across both eyes, the
   whole of any glasses included. The strip is cut from the same source, widened by 4 % of
   the band on every side for the box answer's error, exposed by its own median (so the
   inside of a lens gets the tonal range, which the face's exposure would give the skin),
   lifted, and enlarged to 1024 px wide. A frame whose strip files are all cached is not
   decoded again. Two requests then read the strips, each asking one thing and neither asked
   to rank or judge anything else. The *levels request* asks per eye (the one on the left
   and the one on the right of the picture) for a few words on what of it can be seen, then
   a level: 4 wide open, 3 open, 2 narrowed (the pupil still shows, as in a smile), 1 slit,
   0 shut, or null for an eye that cannot be seen; the note comes first in the schema on
   purpose, so that the model looks before it answers. The *compare request* asks, per
   person, to order their frames from the most to the least open eyes, a frame counting by
   its less open eye, frames that look alike sharing a group, and frames where the eyes
   cannot be seen listed apart. The *expression request* gets the whole face as it looks and
   asks each person's eyewear once (none, clear, tinted, dark, by how the eyes show through
   the lenses in the photograph as normally viewed), each expression, and the ranking of the
   frames by how good the faces look together, the main subject (whose crops are captioned
   as such) mattering most; eyes are expressly left out of it.
   The winner is chosen locally. For each person, a frame is measured against the best
   frame there is of them among the candidates. A face's level is that of its worse visible
   eye; a face whose eyes are both unseen, or a person with no face in the frame, counts as
   narrowed — below open and above slit and shut, because it is neither known to be a blink
   nor known to be open. The *level deficit* is the number of levels below the person's best
   frame; the *compare deficit* is the index of their group in the compare answer (0 = the
   most open), the group at index floor(groups / 2) for a frame where they are not visible or
   absent (the middle group of an odd number, the less open of two, group 0 for one). Each step
   decides only when all earlier ones tie:
   1. level deficits, tier by tier from the most to the least likely to be noticed in the
      photograph, each tier's deficits summed, smaller first: the main subject behind no,
      clear or tinted lenses; others bare; behind clear lenses; behind tinted lenses; and
      last anyone behind dark lenses, the main subject included, because nobody sees those;
   2. compare deficits, tier by tier the same way;
   3. fewest awkward expressions; 4. the expression request's ranking; 5. the burst pass's
      pick; 6. sharpness.

   Deficits are never summed across tiers: a deficit of one step on a bare face loses to any
   amount behind tinted or dark lenses, and a main subject behind dark lenses loses nothing
   to bare others. How noticeable a person's eyes are derives from their eyewear, one fact
   per person for the burst, never read per frame. Each face read is logged at debug level
   (`RUST_LOG=rawpilot::faces=debug`) with both eyes' levels and the compare group, to score
   a run against hand labels. If any faces request fails after its retry, the burst gets no
   faces result and keeps the burst pass's winner.
   Only frames at least `faces_sharpness_rel_min` as sharp as the burst's best are
   candidates.
3. **Scene pass** — burst winners of consecutive scenes side by side (up to 24): repeated
   attempts at the same shot are grouped into a take with one best photo, and each take is
   tiered portfolio / keeper / ordinary / reject relative to its neighbours. `reject` is
   reserved for technical failure no edit can save; composition is at worst `ordinary`, and
   the photograph is judged, not the rarity of the event.
4. **Shortlist** — every keeper or portfolio take of the shoot ranked once more, in
   interleaved chunks when there are more than `photos_max` (each chunk samples the whole
   shoot, chunks merged by relative place); the top `portfolio_share` become portfolio.

Model answers are validated against the request (indices in range, every item answered
once) and asked again once if invalid.

## Ratings

| Bucket | Rule | Rating | Label |
| --- | --- | --- | --- |
| Selected | best of its take, portfolio (5★) or keeper (4★) | 4–5★ | Green |
| Maybe | best of its take, ordinary | 3★ | Yellow |
| Coverage | best of its take, technically flawed | 2★ | Purple |
| Alternate | burst winner whose take's best is another burst's winner | 2★ | Blue |
| Review | any other frame | 1★ | — |
| Reject | judged a reject, or sharpness below `reject_sharpness_rel_max` of the burst's best | 0★ | — |

The 1★ bucket is called Review in the code (`rate::Bucket::Review`) and in the report; it is
not `rawpilot review`, the browser UI, which this document calls the review UI.

Without the scene pass (a failed request) a burst winner is rated from the burst pass's
verdict; with `--local-only` every burst winner is 3★.

## The look

RawTherapee's "Auto-Matched Curve" profile reproduces the camera JPEG, so a fresh image
opens as dark as a frame shot at negative compensation. A pp3 rawpilot creates or still owns
sets `[Exposure] Compensation` to the larger of undoing the shot's negative compensation and
`key_gap_fraction` of the gap, in stops of scene key, between the neutral development and
the camera JPEG — within `compensation_min/max_ev`, and left at the profile's value below
`compensation_threshold_ev` — plus `HighlightCompr` whenever exposure is raised. The full
gap over-brightens low-key scenes; half of it matches renders judged by eye.

## Storage and ownership

Each image's `DSC_xxxx.NEF.rpl` (JSON, next to the NEF) records what rawpilot last wrote —
rating, label, look, and the blake3 of the pp3 as rawpilot left it — committed and in
flight, plus the run's analysis of the image. From it:

- no pp3: create one from the default profile, with the rating and (`process`) the look;
- a pp3 byte-identical to what rawpilot wrote: nobody has edited it, so rating and look are
  kept up to date;
- any other pp3: the photographer's. Only the rating may change, and only while it is still
  rawpilot's; a rating-only write records no hash, so the file stays the photographer's.

Writes are ordered so an interruption anywhere is resumable by re-running: the `.rpl`
records the pending values, the file is written and read back, the `.rpl` commits. A
missing or unreadable `.rpl` claims nothing.

In `xmp` mode the sidecar is read and written in-process (`src/xmp.rs`), without a metadata
tool. A write changes only the bytes of `xmp:Rating` and `xmp:Label`: other properties,
other namespaces, the `<?xpacket?>` wrapper and its padding are kept, and each property
keeps the form it has (exiv2's attribute on `rdf:Description`, exiftool's element), found
by namespace URI rather than prefix. A missing property is added beside the other one, else
to the first `rdf:Description`; a missing sidecar is created as a minimal packet. A sidecar
that is not well-formed XMP is an error, never read as unrated.

The review UI adds three additive fields to the `.rpl` (older records without them read as
before): `faces` (where the faces pass located faces in this frame, in fractions of the
displayed image, kept across runs without the pass so renders need no Gemini), `chosen`
(the look option picked, `Pending` before its pp3 is written, with the digest of the pp3
that was there then (`prior`), and `Applied` after the read back, with the digest of the
pp3's processing, which a later rating does not change) and
`rated` (the rating was set in the UI). Rules:

- A pp3 with `chosen` is never owned by the batch, whatever its bytes; with `rated`, the
  batch leaves the rating alone, including an unrated zero.
- Choosing replaces a pp3's processing, keeping its `Rank` and `ColorLabel`. When the pp3 is
  hand-edited (it exists and is neither what the batch left nor what the UI wrote, including
  one RawTherapee wrote itself) the call refuses until it is repeated with the
  photographer's confirmation (`choice::OnHandEdited`; `Session::pp3_state` tells the UI
  which case it is). An interrupted choice (`Pending`, and the file is still the `prior`
  one) is finished by choosing again, without a question: the interrupted call already
  passed that gate for exactly those bytes. A file edited since is hand-edited again.
- Each UI write follows the batch's order: intent in the `.rpl`, file written and read
  back, completion recorded.
- A rating in `procparams` mode goes into the pp3 (created from the default profile, without
  the look, when missing), in `xmp` mode into the sidecar.

## Look options and renders

A look option is a name, a UI key and a partial pp3 (none, a file, one shipped in the binary,
or one of three files by ISO band), layered over the default profile; the exposure look is
applied last, to every option. The result is a `pp3::CompletePp3`, the only text rawpilot
renders from or writes for an option, so a partial profile never reaches disk. Every built-in
profile is a file in `profiles/` compiled into the binary, so a look does not change with the
installed RawTherapee. `Film` is RawTherapee 5.13's `Standard Film Curve - ISO
Low/Medium/High` (`profiles/film-iso-*.pp3`), which differ only in noise handling: Low (below
ISO 800) adds sharpening, Medium (800 to below 3200) enables a neutral denoise block, High
(3200 and up) adds median and luminance denoise and LMMSE demosaicing. All three turn on
automatic lens correction (distortion and vignetting). The others turn histogram matching
off so their own curve holds, and each answers a kind of scene; `profiles/README.md` has
their settings. `Vivid` aims at a nature postcard, not a social-media look. `Warm` is the
scene as seen through bronze sunglasses: the midtones shift warm relative to the shot's own
white balance, so white stays white. `Sunny` is for hard midday light: shadows opened and
highlights held back, with local contrast and vibrance so the result is not faded. `Warm-sun`
is the two together. `Portrait` keeps skin natural: a softer top, little vibrance and no
local contrast. `Overcast` firms up flat grey light; it is a first guess. `Dusk` is
for after sunset, when the histogram is U-shaped: deep blacks kept, the lights rolled off
rather than clipped, local contrast, vibrance for coloured lights, chroma noise reduction.
A softer, lower-contrast Film was tried and dropped: it read as a faded old print.

The exposure look an option is applied with is the one `process` measured for the shot,
recorded in the `.rpl` (`measured_look`) whether or not `process` could write it into the
pp3 (a hand-edited pp3 keeps its own processing). `process` and `review` both make a render
from the `.rpl` through `looks::Job::new`, so one (shot, option) has one key and one render
whichever of them makes it. `cull` leaves the recorded look as it is; a `process` that finds
no correction, or has the look turned off, clears it.

`process` renders each (shot, option) once at full size with `rawtherapee-cli` through
exactly that pp3, keeps a fit view and 1:1 crops (AF box padded to at least 512 px around its
centre; each located face padded by 25 % per side; one centre square when there is neither;
each capped at 1200 px), and deletes the render. Regions are decided in fractions of the
displayed image: the camera's AF box is turned from sensor orientation into it, and
`rawtherapee-cli` writes the image already oriented (a rotated frame comes out portrait).
Renders live under `<dir>/.rawpilot/looks/<NEF>/<option>/` with a `manifest.json` the UI
reads and the `profile.pp3` that choosing writes. The manifest holds two keys: one over
every input but RawTherapee's default profile (the shot, `looks::VERSION`, the digest of the
option's partial for the shot's ISO, the exposure look, the render parameters and the region
list) and the render's key, which adds the digest of the complete pp3. `looks::Fresh::check`
is the one place that compares them with today's inputs, and `process` redoes any render it
rejects. `review` shows and writes only a render that passed it: at launch the first key is
compared, because the default profile takes seconds to resolve, and when it resolves the full
keys are; a rejected render is queued like a missing one and cannot be chosen meanwhile.
`VERSION` covers code that changes a render's files without changing a key input (cutting,
resizing, JPEG encoding). A run of `process` with no failed render ends by removing the
directories of options no longer configured under `.rawpilot/looks/<NEF>/`; `review` removes
nothing. See the module docs of `looks` and `choice` for the API.

`<dir>/.rawpilot/` caches measurements and renderings (keyed by file identity and the
settings that affect them) and Gemini answers (keyed by a digest of the exact request:
prompt, captions, image bytes, model settings), plus `report.html`.

## Finish and the archive

`finish` (`src/finish/`) is the only command that writes outside the working folder, and
what it may write there is narrow: files under `<media-root>/photos-RAW/<event>/`, plus
whatever `import-media` files under `photos/`.

Types carry the checks. `EventName` is a single safe folder name, parsed at the command line.
`MediaRoot` exists only for a root whose `photos-RAW` is a directory, so an unmounted drive
cannot be written to and `photos-RAW` is never created; `EventDir` is the one folder inside it.
`FolderLock` (taken as `Run::Finish`) is required to run the stages, and `--dry-run` has none to
give: it only reads. A `Plan` is built from the ratings RawTherapee reads now, not the ones
rawpilot wrote, and fails before any work when no shot qualifies, a keeper has no pp3 to
export through, or two keepers would share an export.

What goes where:

| Where | What |
| --- | --- |
| `photos-RAW/<event>/` | per keeper: the NEF's bytes (links are followed), `<NEF>.pp3`, `<NEF>.rpl`, the XMP sidecar in RawTherapee's `XMPSidecarStyle` name; each only if it exists, and the pp3 is required |
| `photos-RAW/<event>/rawpilot/summary.json` | every shot of the folder: file, capture time, rating rawpilot wrote, current rating, look chosen in `review` (once applied), keeper |
| `photos-RAW/<event>/rawpilot/gemini-answers.jsonl` | one line per cached answer of `judge`, `scene`, `faces` and `shortlist`: pass, request digest, model version, usage, the reply verbatim |
| `photos-RAW/<event>/rawpilot/report.html` | a copy of `<dir>/.rawpilot/report.html` |
| `photos/YYYY/MM/DD/` | the exported JPEGs, filed by `import-media` |
| `<dir>/.rawpilot/export/<event>/` | the exported JPEGs, in the working folder |

Renders, thumbnails, developments and measurements are not archived. The summary and the
answers are generated in a fixed order from the same inputs, so an unchanged shoot produces the
same bytes.

Nothing on the drive is replaced (`archive`). Each file is written to a hidden temporary name
beside its destination, synced and read back against the blake3 of the source, then given its
name by a hard link, which fails when the name is taken (a rename would replace it); the
temporary name is then removed and the folder synced. A name that is taken is compared first:
identical content is done, anything else (different content, a folder, a link) is refused,
listed, and fails the run once everything safe has been done. The same decision runs when a
name is taken between the check and the link. The drive's file system must support hard
links; one that does not fails each file instead of falling back to a replacing rename.

The stages are ordered so that a failure leaves nothing half-done on the drive: exports first
(all of them are tried so every failure is listed), the archive only if they all worked, then
`import-media`, which skips existing targets itself. Interrupted at any point, running again
completes the work and does not repeat what is on the drive. Because an archived file is never
replaced, a file whose working copy changed after archiving (a pp3 edited, a summary written for
another threshold, new Gemini answers) is refused on a later run; the archive keeps what was
first written.

## Known limitations

- RawTherapee's first open of a folder of rawpilot-created pp3s is slow (about a second per
  image); the likely cause is the auto-matched curve, which the CLI-resolved profile leaves
  to be computed per image.
- A few High Efficiency NEFs (about one in 2,000) fail to decode in libraw and RawTherapee
  alike while their embedded JPEG reads fine; they are judged on the camera preview.
- Reflections can hide the eyes even from the raw: lenses that mirror the scenery show
  nothing behind them at any exposure, and the levels request answers null for them. Such
  a frame counts as narrowed: not penalised as a blink, but nor is it known to be good.
- The eye band is cut geometrically from Gemini's face box, which can sit a tenth of the
  face too high or low, and by up to 10 grid units whatever the face's size; the band's
  margins absorb that, and a box further off, or a head
  tilted past about 30°, can leave an eye outside it, and so outside the eye strip that is
  located within the band.
- Attempts at one shot that land in different scene-pass requests stay separate takes.
- The top tier has no notion of subject diversity: several takes of one landmark can all
  reach 5★.
