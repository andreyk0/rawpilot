//! Images of each located face, cut at full resolution, for the faces pass to read: the whole
//! face (`Crop::Face`, for expression), the band of it that holds the eyes (`Crop::EyeBand`,
//! where the model locates the eyes) and, once it has, the eye strip (`eye_strip_files`, for how
//! open the eyes are).
//!
//! A face is a few hundred pixels across in a 45 MP frame, and the eyes behind tinted or dark
//! lenses sit near the bottom of what a camera JPEG keeps: its tone curve and 8-bit blocks turn
//! the inside of a lens to mush. The raw still holds them. So when the request's images are
//! `Shown::Developed` the faces are cut from the raw (libraw's linear output) and when they
//! are `Shown::Camera` from the camera preview: Gemini's boxes are fractions of the image it
//! was shown, and the two differ in geometry by lens-distortion correction (up to ~36 px at
//! the edge of a frame), so the crop source is always the request's own.
//!
//! Each crop is rendered the way its purpose wants (`Crop::rendering`) and written to the
//! cache as a JPEG file; the request is built from those files' bytes, so a warm run sends
//! exactly what the cold run did and its answer is found in the cache. A frame whose files
//! are all present is not decoded again.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, Rgb32FImage, RgbImage, imageops};
use serde::Serialize;

use crate::cache::{Cache, write_atomic};
use crate::develop::{self, LinearU16};
use crate::judge::Shown;
use crate::preview;
use crate::shot::{Orientation, PixelRect, Shot};
use crate::tone;

/// A face box is padded by this fraction of its size on every side, for head and hair context
/// and for the subject moving a little between frames...
const PAD_FRAC: f64 = 0.25;
/// ...plus this many pixels, which covers the ~36 px by which the raw and camera geometries
/// differ at the edge of a frame.
const PAD_SLOP_PX: f64 = 48.0;
/// Long edge of a face image. Google documents only a token budget per image at each
/// `media_resolution` (560 for `MEDIA_RESOLUTION_MEDIUM`), not the pixel size it is read at;
/// 768 is the edge of the 768×768 tile Gemini's image tokenization is defined on, and a
/// 768 × ~576 face is well inside that budget. A crop already smaller is not enlarged.
const LONG_EDGE_PX: u32 = 768;
const JPEG_QUALITY: u8 = 90;
/// Bumped when the code that renders or cuts a face changes.
const VERSION: u32 = 4;

/// The eye strip is the located box widened by this fraction of the band's width and height on
/// every side, because the box answer is good to a few grid units, and kept inside the band.
const EYE_STRIP_MARGIN_FRAC: f64 = 0.04;
/// Width of an eye strip image: strips are enlarged to it when smaller, so that the pixels of a
/// lens interior are read at a size the model resolves.
const EYE_STRIP_WIDTH_PX: u32 = 1024;
/// Bumped when the code that renders or cuts an eye strip changes.
const EYE_STRIP_VERSION: u32 = 1;

/// How far a located box may sit from the face, as a fraction of the displayed image's width
/// and height. The locate answer is on a 0–1000 grid of a ~1024 px rendering and its error is
/// a few grid units whatever the face's size (the example shoot's boxes are good to 1–5); a
/// 10-unit bound leaves margin over that. Relative to a large face it is negligible, but a
/// face 2 % of the frame wide is off by half its own size, so the crop and the eye band are
/// widened by this on every side, on top of their relative geometry.
const LOCATE_ERROR_FRAC: f64 = 0.010;

/// The eye band, as fractions of the face box's height. Eyes sit at about 35–50 % of a box
/// that runs from hairline to chin; Gemini's boxes are looser than that (in the example
/// shoot lenses span 34–85 % of the box, the low end in a tilted head whose box sits high),
/// so the band takes 30–85 %, which holds both lenses of an upright or moderately tilted
/// head in all of them.
const BAND_TOP_FRAC: f64 = 0.30;
const BAND_BOTTOM_FRAC: f64 = 0.85;
/// The band extends past the face box's sides by this fraction of its width: a turned head's
/// far eye is near the box's edge.
const BAND_PAD_FRAC: f64 = 0.10;

/// Where a face's median luma lands, in linear light, in each rendering: sRGB ≈ 0.46 for
/// `Natural`; `EyesLifted` leaves the log curve room above the skin and puts it at 0.05.
const NATURAL_MEDIAN: f32 = 0.18;
const LIFTED_MEDIAN: f32 = 0.05;
/// Bounds of `Gain`: a crop whose median is below 1/320 of full scale (a black lens cap) is
/// not amplified past it.
const GAIN_RANGE: (f32, f32) = (1.0, 320.0);
const SHOULDER_START: f64 = 0.5;
/// Strength of the log curve `ln(1 + K·v) / ln(1 + K)`. At K = 800 a lens interior 20× darker
/// than the skin (v = 0.0025) reads 0.16 where the sRGB curve reads 0.03, and skin at the
/// median reads 0.55 with three times that still short of white.
const LIFT_K: f32 = 800.0;

/// Box in fractions (0–1) of the displayed image.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, serde::Deserialize)]
#[serde(try_from = "BoxFractions")]
pub struct Box2d {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// The stored shape of a `Box2d`, checked on the way in.
#[derive(serde::Deserialize)]
struct BoxFractions {
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl TryFrom<BoxFractions> for Box2d {
    type Error = String;

    fn try_from(b: BoxFractions) -> Result<Box2d, String> {
        let inside = |v: f64| (0.0..=1.0).contains(&v);
        if [b.x0, b.y0, b.x1, b.y1].into_iter().all(inside) && b.x0 < b.x1 && b.y0 < b.y1 {
            Ok(Box2d {
                x0: b.x0,
                y0: b.y0,
                x1: b.x1,
                y1: b.y1,
            })
        } else {
            Err(format!(
                "bad box {},{} to {},{}: not inside 0..1 with area",
                b.x0, b.y0, b.x1, b.y1
            ))
        }
    }
}

impl Box2d {
    /// The box of a model answer `[ymin, xmin, ymax, xmax]` on a 0–1000 grid; refused unless it
    /// lies inside the grid and has area.
    pub fn from_grid(grid: [i64; 4]) -> Result<Box2d> {
        let [ymin, xmin, ymax, xmax] = grid;
        ensure!(
            grid.iter().all(|v| (0..=1000).contains(v)) && ymin < ymax && xmin < xmax,
            "bad box {grid:?}"
        );
        let f = |v: i64| v as f64 / 1000.0;
        Ok(Box2d {
            x0: f(xmin),
            y0: f(ymin),
            x1: f(xmax),
            y1: f(ymax),
        })
    }
}

/// Where an eye strip lies in a face's eye band, in fractions (0–1) of the band.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BandBox(Box2d);

impl BandBox {
    /// The strip box of a model answer on the 0–1000 grid of the band image.
    pub fn from_grid(grid: [i64; 4]) -> Result<BandBox> {
        Box2d::from_grid(grid).map(BandBox)
    }
}

/// How a crop's pixels are turned into an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rendering {
    /// What the face looks like in a normal edit: exposure normalised to the face, a gentle
    /// shoulder, sRGB.
    Natural,
    /// A log curve that lifts the inside of tinted and dark lenses without blowing skin.
    EyesLifted,
}

/// Which part of a face is cut, for which purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crop {
    /// The face box with head and hair around it, as a viewer sees it: for expression.
    Face,
    /// The face box's eye region, cut tight: for locating the eye strip within it.
    EyeBand,
}

impl Crop {
    pub const ALL: [Crop; 2] = [Crop::Face, Crop::EyeBand];

    /// Seeing eyes needs the shadows lifted; judging expression needs the face as it looks.
    fn rendering(self) -> Rendering {
        match self {
            Crop::Face => Rendering::Natural,
            Crop::EyeBand => Rendering::EyesLifted,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Crop::Face => "face",
            Crop::EyeBand => "eye_band",
        }
    }
}

/// The cached files of one face, one per crop.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceFiles {
    face: PathBuf,
    eye_band: PathBuf,
}

impl FaceFiles {
    pub(crate) fn named(dir: &Path, stem: &str) -> FaceFiles {
        let path = |c: Crop| dir.join(format!("{stem}.{}.jpg", c.slug()));
        FaceFiles {
            face: path(Crop::Face),
            eye_band: path(Crop::EyeBand),
        }
    }

    pub fn get(&self, c: Crop) -> &Path {
        match c {
            Crop::Face => &self.face,
            Crop::EyeBand => &self.eye_band,
        }
    }

    fn all_present(&self) -> bool {
        Crop::ALL.iter().all(|&c| self.get(c).is_file())
    }
}

/// Everything that determines one face's files besides the shot's identity.
#[derive(Serialize)]
struct Params {
    shown: Shown,
    orientation: Orientation,
    /// The union of the frame's faces: what dcraw_emu was asked for decides the demosaic at
    /// its edges.
    union: PixelRect,
    face: FaceRect,
    pad_frac: f64,
    pad_slop_px: f64,
    locate_error_frac: f64,
    long_edge_px: u32,
    jpeg_quality: u8,
}

/// Each face of `layout`, cut from the source `shown` names into every `Crop`, written under
/// `dir` unless already there; in the boxes' order. One libraw decode serves the frame.
pub fn face_files(
    dir: &Path,
    shot: &Shot,
    shown: Shown,
    layout: &Layout,
) -> Result<Vec<FaceFiles>> {
    let files = layout
        .plan
        .faces
        .iter()
        .map(|face| {
            let params = Params {
                shown,
                orientation: layout.orientation,
                union: layout.plan.union,
                face: *face,
                pad_frac: PAD_FRAC,
                pad_slop_px: PAD_SLOP_PX,
                locate_error_frac: LOCATE_ERROR_FRAC,
                long_edge_px: LONG_EDGE_PX,
                jpeg_quality: JPEG_QUALITY,
            };
            let key = Cache::key_under("face-crop", VERSION, shot, &params)?;
            Ok(FaceFiles::named(
                dir,
                &format!("{}.{}", shot.name(), &key[..16]),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if files.iter().all(FaceFiles::all_present) {
        return Ok(files);
    }

    let union = layout.decode(shot, shown)?;
    for (face, files) in layout.plan.faces.iter().zip(&files) {
        for (c, image) in render_face(&union, face) {
            write_jpeg(files.get(c), &image)?;
        }
    }
    Ok(files)
}

/// Where a frame's faces are cut from: the plan over the displayed frame, and what it takes to
/// fetch its union from `shot`'s source. Built once per frame for both the face crops and the
/// eye strips, because it reads the source's size.
pub struct Layout {
    orientation: Orientation,
    /// The source image's size in stored (sensor) orientation.
    stored: (u32, u32),
    plan: Plan,
}

impl Layout {
    pub fn new(shot: &Shot, shown: Shown, boxes: &[Box2d]) -> Result<Layout> {
        let orientation = shot.orientation;
        let stored = match shown {
            Shown::Developed => develop::stored_size(&shot.file)?,
            Shown::Camera => preview::dimensions(shot)?,
        };
        let (w, h) = orientation.displayed_size(stored.0, stored.1);
        Ok(Layout {
            orientation,
            stored,
            plan: Plan::new(boxes, w, h),
        })
    }

    /// The union of the frame's faces from the source `shown` names: one libraw decode for
    /// the raw.
    fn decode(&self, shot: &Shot, shown: Shown) -> Result<Source> {
        let (o, union) = (self.orientation, self.plan.union);
        let source = match shown {
            Shown::Developed => Source::Raw16(develop::decode_crop(
                &shot.file,
                o.stored_rect(union, self.stored.0, self.stored.1),
            )?),
            Shown::Camera => Source::Camera(upright_cut(&preview::decode(shot)?, o, union)),
        };
        source
            .expect_size(union)
            .with_context(|| format!("cropping faces of {}", shot.name()))?;
        Ok(source)
    }

    /// Where the eye strip `strip` of `face` is cached under `dir`: named by everything that
    /// determines its bytes.
    fn eye_strip_path(
        &self,
        dir: &Path,
        shot: &Shot,
        shown: Shown,
        face: &FaceRect,
        strip: BandBox,
    ) -> Result<PathBuf> {
        let params = EyeStripParams {
            shown,
            orientation: self.orientation,
            union: self.plan.union,
            face: *face,
            strip,
            margin_frac: EYE_STRIP_MARGIN_FRAC,
            width_px: EYE_STRIP_WIDTH_PX,
            lifted_median: LIFTED_MEDIAN,
            lift_k: LIFT_K,
            gain_range: GAIN_RANGE,
            jpeg_quality: JPEG_QUALITY,
        };
        let key = Cache::key_under("eye-strip", EYE_STRIP_VERSION, shot, &params)?;
        Ok(dir.join(format!("{}.{}.eye_strip.jpg", shot.name(), &key[..16])))
    }
}

fn write_jpeg(path: &Path, image: &RgbImage) -> Result<()> {
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY).encode_image(image)?;
    write_atomic(path, &jpeg)
}

/// Everything that determines one eye strip's file besides the shot's identity.
#[derive(Serialize)]
struct EyeStripParams {
    shown: Shown,
    orientation: Orientation,
    union: PixelRect,
    face: FaceRect,
    strip: BandBox,
    margin_frac: f64,
    width_px: u32,
    lifted_median: f32,
    lift_k: f32,
    gain_range: (f32, f32),
    jpeg_quality: u8,
}

/// The eye strip of each face of `layout`, cut from the source `shown` names and written
/// under `dir` unless already there; in the faces' order. `strips` holds where in each eye
/// band the model located the strip. The union of the faces is decoded once, and not at all
/// when every strip is present.
pub fn eye_strip_files(
    dir: &Path,
    shot: &Shot,
    shown: Shown,
    layout: &Layout,
    strips: &[BandBox],
) -> Result<Vec<PathBuf>> {
    assert_eq!(strips.len(), layout.plan.faces.len(), "one strip per face");
    let paths = layout
        .plan
        .faces
        .iter()
        .zip(strips)
        .map(|(face, &strip)| layout.eye_strip_path(dir, shot, shown, face, strip))
        .collect::<Result<Vec<_>>>()?;
    if paths.iter().all(|p| p.is_file()) {
        return Ok(paths);
    }

    let union = layout.decode(shot, shown)?;
    for ((face, &strip), path) in layout.plan.faces.iter().zip(strips).zip(&paths) {
        if !path.is_file() {
            write_jpeg(path, &render_eye_strip(&union, face, strip))?;
        }
    }
    Ok(paths)
}

/// The eye strip of `face`: exposed by the strip's own pixels, so that the inside of a lens gets
/// the tonal range, lifted like an eye band, and enlarged or reduced to `EYE_STRIP_WIDTH_PX` wide.
fn render_eye_strip(union: &Source, face: &FaceRect, strip: BandBox) -> RgbImage {
    let pixels = union.cut(face.in_union(face.eye_strip_region(strip)));
    let lifted = pixels.render(Rendering::EyesLifted, pixels.gain());
    scale_to_width(&lifted, EYE_STRIP_WIDTH_PX)
}

/// `img` resampled to `width`, keeping its aspect: enlarged when narrower.
fn scale_to_width(img: &RgbImage, width: u32) -> RgbImage {
    let scale = f64::from(width) / f64::from(img.width());
    let height = ((f64::from(img.height()) * scale).round() as u32).max(1);
    imageops::resize(img, width, height, imageops::FilterType::Lanczos3)
}

/// Every `Crop` of `face`, rendered. Each part is cut from the union by itself, so that at
/// most one rectangle of one face is floating point at a time, and no copy of it is made to
/// cut from.
fn render_face(union: &Source, face: &FaceRect) -> [(Crop, RgbImage); Crop::ALL.len()] {
    let gain = union.cut(face.in_union(face.core)).gain();
    Crop::ALL.map(|c| {
        let image = union.cut(face.in_union(face.region(c))).fit(LONG_EDGE_PX);
        (c, image.render(c.rendering(), gain))
    })
}

/// One face in displayed pixels: the padded rectangle that is cut, and the face's own box and
/// eye band inside it. Both are relative to `crop`'s top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct FaceRect {
    crop: PixelRect,
    /// The face box, where the exposure is measured.
    core: PixelRect,
    band: PixelRect,
}

impl FaceRect {
    /// `r`, a rectangle relative to `crop`'s corner, relative to the union's corner. `crop`
    /// itself is already in the union's coordinates (`Plan::faces`).
    fn in_union(&self, r: PixelRect) -> PixelRect {
        PixelRect {
            x: self.crop.x + r.x,
            y: self.crop.y + r.y,
            ..r
        }
    }

    /// The part of `crop` that `c` shows.
    fn region(&self, c: Crop) -> PixelRect {
        match c {
            Crop::Face => PixelRect {
                x: 0,
                y: 0,
                ..self.crop
            },
            Crop::EyeBand => self.band,
        }
    }

    /// The part of `crop` that the eye strip `b` of the band covers, widened by
    /// `EYE_STRIP_MARGIN_FRAC` of the band on every side and kept inside it, at least a pixel.
    fn eye_strip_region(&self, b: BandBox) -> PixelRect {
        let band = self.band;
        assert!(band.w > 0 && band.h > 0, "an eye band has area");
        // The span of `lo..hi` (fractions of `len`) widened, as whole pixels inside `0..len`.
        let span = |len: u32, lo: f64, hi: f64| {
            let len_f = f64::from(len);
            let at = |frac: f64| (frac * len_f).round().clamp(0.0, len_f) as u32;
            let start = at(lo - EYE_STRIP_MARGIN_FRAC).min(len - 1);
            let end = at(hi + EYE_STRIP_MARGIN_FRAC).max(start + 1);
            (start, end)
        };
        let Box2d { x0, y0, x1, y1 } = b.0;
        let (sx, ex) = span(band.w, x0, x1);
        let (sy, ey) = span(band.h, y0, y1);
        PixelRect {
            x: band.x + sx,
            y: band.y + sy,
            w: ex - sx,
            h: ey - sy,
        }
    }
}

/// Height of a face in the report's contact strip, and of an eye band under it.
const CONTACT_FACE_HEIGHT_PX: u32 = 192;
const CONTACT_BAND_HEIGHT_PX: u32 = 96;
const CONTACT_GAP_PX: u32 = 6;

/// Makes the file at `path` the contact strip of `files`: the faces side by side with their eye
/// bands in a row under them, in one JPEG, for the report (it is not sent to the model). A
/// frame with no faces has no contact strip, and one left at `path` by an earlier run is
/// removed, so that the file exists exactly when the last call found faces.
pub fn sync_contact_strip(path: &Path, files: &[FaceFiles]) -> Result<()> {
    if files.is_empty() {
        return match std::fs::remove_file(path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                Err(e).with_context(|| format!("removing stale contact strip {}", path.display()))
            }
            Ok(()) | Err(_) => Ok(()),
        };
    }
    let faces = contact_row(files, Crop::Face, CONTACT_FACE_HEIGHT_PX)?;
    let bands = contact_row(files, Crop::EyeBand, CONTACT_BAND_HEIGHT_PX)?;
    let mut contact = RgbImage::new(
        faces.width().max(bands.width()),
        faces.height() + CONTACT_GAP_PX + bands.height(),
    );
    imageops::replace(&mut contact, &faces, 0, 0);
    imageops::replace(
        &mut contact,
        &bands,
        0,
        i64::from(faces.height() + CONTACT_GAP_PX),
    );
    write_jpeg(path, &contact)
}

/// The `crop` of every file, scaled to `height` and set side by side.
fn contact_row(files: &[FaceFiles], crop: Crop, height: u32) -> Result<RgbImage> {
    let images: Vec<RgbImage> = files
        .iter()
        .map(|f| {
            let image = image::open(f.get(crop))?.into_rgb8();
            let w = (image.width() * height).div_ceil(image.height());
            Ok(imageops::resize(
                &image,
                w.max(1),
                height,
                imageops::FilterType::Triangle,
            ))
        })
        .collect::<Result<_>>()?;
    let width = images
        .iter()
        .map(|f| f.width() + CONTACT_GAP_PX)
        .sum::<u32>()
        .saturating_sub(CONTACT_GAP_PX);
    let mut row = RgbImage::new(width, height);
    let mut x = 0;
    for f in &images {
        imageops::replace(&mut row, f, i64::from(x), 0);
        x += f.width() + CONTACT_GAP_PX;
    }
    Ok(row)
}

/// Where to cut, in displayed pixels: every face and the union of their padded rectangles,
/// which is the one region fetched from the source.
#[derive(Debug, Clone, PartialEq)]
struct Plan {
    union: PixelRect,
    /// Each face's `crop` relative to the union's top-left corner, in the boxes' order.
    faces: Vec<FaceRect>,
}

impl Plan {
    fn new(boxes: &[Box2d], w: u32, h: u32) -> Plan {
        assert!(!boxes.is_empty(), "a plan needs at least one face");
        let faces: Vec<FaceRect> = boxes.iter().map(|b| face_rect(*b, w, h)).collect();
        let x0 = faces.iter().map(|f| f.crop.x).min().expect("non-empty");
        let y0 = faces.iter().map(|f| f.crop.y).min().expect("non-empty");
        let x1 = faces
            .iter()
            .map(|f| f.crop.x + f.crop.w)
            .max()
            .expect("non-empty");
        let y1 = faces
            .iter()
            .map(|f| f.crop.y + f.crop.h)
            .max()
            .expect("non-empty");
        Plan {
            union: PixelRect {
                x: x0,
                y: y0,
                w: x1 - x0,
                h: y1 - y0,
            },
            faces: faces
                .iter()
                .map(|f| FaceRect {
                    crop: PixelRect {
                        x: f.crop.x - x0,
                        y: f.crop.y - y0,
                        ..f.crop
                    },
                    ..*f
                })
                .collect(),
        }
    }
}

/// `b` as whole pixels of a `w×h` displayed image, padded and kept inside it.
fn face_rect(b: Box2d, w: u32, h: u32) -> FaceRect {
    let (fw, fh) = (f64::from(w), f64::from(h));
    let (error_x, error_y) = (LOCATE_ERROR_FRAC * fw, LOCATE_ERROR_FRAC * fh);
    let pad_x = (b.x1 - b.x0) * fw * PAD_FRAC + PAD_SLOP_PX + error_x;
    let pad_y = (b.y1 - b.y0) * fh * PAD_FRAC + PAD_SLOP_PX + error_y;
    let (x0, y0) = ((b.x0 * fw).floor(), (b.y0 * fh).floor());
    let (x1, y1) = ((b.x1 * fw).ceil(), (b.y1 * fh).ceil());
    let (cx0, cy0) = ((x0 - pad_x).floor().max(0.0), (y0 - pad_y).floor().max(0.0));
    let (cx1, cy1) = ((x1 + pad_x).ceil().min(fw), (y1 + pad_y).ceil().min(fh));
    assert!(
        x0 < x1 && y0 < y1 && x1 <= fw && y1 <= fh,
        "box {b:?} leaves nothing of a {w}×{h} frame"
    );
    // The band lies inside the padded crop except at the frame's edge, where both are clipped.
    let band_pad = (x1 - x0) * BAND_PAD_FRAC + error_x;
    let (bx0, bx1) = (
        (x0 - band_pad).floor().max(cx0),
        (x1 + band_pad).ceil().min(cx1),
    );
    let (by0, by1) = (
        (y0 + (y1 - y0) * BAND_TOP_FRAC - error_y).floor().max(cy0),
        (y0 + (y1 - y0) * BAND_BOTTOM_FRAC + error_y)
            .ceil()
            .min(cy1),
    );
    FaceRect {
        crop: PixelRect {
            x: cx0 as u32,
            y: cy0 as u32,
            w: (cx1 - cx0) as u32,
            h: (cy1 - cy0) as u32,
        },
        core: PixelRect {
            x: (x0 - cx0) as u32,
            y: (y0 - cy0) as u32,
            w: (x1 - x0) as u32,
            h: (y1 - y0) as u32,
        },
        band: PixelRect {
            x: (bx0 - cx0) as u32,
            y: (by0 - cy0) as u32,
            w: (bx1 - bx0) as u32,
            h: (by1 - by0) as u32,
        },
    }
}

/// The factor that brings a face's median luma to 1. Both renderings are a curve of the pixel
/// times this, so it is made once, from the face and not the background around it, and
/// rendering asks for it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Gain(f32);

impl Gain {
    /// Of the median luma of `face`, within `GAIN_RANGE`.
    fn of_median_luma(face: &Rgb32FImage) -> Gain {
        let w = tone::LUMA.map(|w| w as f32);
        let mut luma: Vec<f32> = face
            .pixels()
            .map(|p| w[0] * p[0] + w[1] * p[1] + w[2] * p[2])
            .collect();
        let mid = luma.len() / 2;
        let (_, median, _) = luma.select_nth_unstable_by(mid, f32::total_cmp);
        Gain((1.0 / (*median).max(1e-6)).clamp(GAIN_RANGE.0, GAIN_RANGE.1))
    }
}

/// The displayed rectangle `r` of a stored-orientation image, upright.
fn upright_cut(stored: &RgbImage, o: Orientation, r: PixelRect) -> RgbImage {
    let s = o.stored_rect(r, stored.width(), stored.height());
    o.display(&imageops::crop_imm(stored, s.x, s.y, s.w, s.h).to_image())
}

/// The union of a frame's padded faces as the source gave it: 16-bit linear light from the
/// raw, or the 8-bit sRGB of the camera's JPEG. It is the one full-size image held while a
/// frame's faces are cut, so the raw stays 16-bit; only a face's own rectangle becomes
/// floating point (`Pixels`), and one at a time.
enum Source {
    Raw16(LinearU16),
    Camera(RgbImage),
}

impl Source {
    fn size(&self) -> (u32, u32) {
        match self {
            Source::Raw16(i) => i.dimensions(),
            Source::Camera(i) => i.dimensions(),
        }
    }

    /// dcraw_emu clips a box that runs past the image without saying so, and then the face
    /// rectangles no longer point at the faces: the size returned must be the size asked for.
    fn expect_size(&self, asked: PixelRect) -> Result<()> {
        let (w, h) = self.size();
        ensure!(
            (w, h) == (asked.w, asked.h),
            "asked for {}×{} pixels, got {w}×{h}",
            asked.w,
            asked.h
        );
        Ok(())
    }

    fn cut(&self, r: PixelRect) -> Pixels {
        match self {
            Source::Raw16(i) => Pixels::Linear(
                DynamicImage::ImageRgb16(imageops::crop_imm(i, r.x, r.y, r.w, r.h).to_image())
                    .into_rgb32f(),
            ),
            Source::Camera(i) => {
                Pixels::Srgb8(imageops::crop_imm(i, r.x, r.y, r.w, r.h).to_image())
            }
        }
    }
}

/// A face's pixels, ready to be rendered: linear light from the raw, or the 8-bit sRGB of the
/// camera's JPEG.
enum Pixels {
    Linear(Rgb32FImage),
    Srgb8(RgbImage),
}

impl Pixels {
    fn size(&self) -> (u32, u32) {
        match self {
            Pixels::Linear(i) => i.dimensions(),
            Pixels::Srgb8(i) => i.dimensions(),
        }
    }

    /// Scaled down, never up, so that the long edge is at most `long_edge`.
    fn fit(self, long_edge: u32) -> Pixels {
        let (w, h) = self.size();
        if w.max(h) <= long_edge {
            return self;
        }
        let scale = f64::from(long_edge) / f64::from(w.max(h));
        let (w, h) = (
            ((f64::from(w) * scale).round() as u32).max(1),
            ((f64::from(h) * scale).round() as u32).max(1),
        );
        let filter = imageops::FilterType::Lanczos3;
        match self {
            Pixels::Linear(i) => Pixels::Linear(imageops::resize(&i, w, h, filter)),
            Pixels::Srgb8(i) => Pixels::Srgb8(imageops::resize(&i, w, h, filter)),
        }
    }

    /// The exposure of these pixels: of a face box, or of an eye strip.
    fn gain(&self) -> Gain {
        match self {
            Pixels::Linear(face) => Gain::of_median_luma(face),
            Pixels::Srgb8(face) => Gain::of_median_luma(&to_linear(face)),
        }
    }

    fn render(&self, r: Rendering, gain: Gain) -> RgbImage {
        let Gain(g) = gain;
        match (self, r) {
            (Pixels::Linear(i), Rendering::Natural) => {
                let k = g * NATURAL_MEDIAN;
                encode(i, |v| natural_curve(v * k, k))
            }
            (Pixels::Linear(i), Rendering::EyesLifted) => {
                encode(i, |v| lifted_curve(v * g * LIFTED_MEDIAN))
            }
            (Pixels::Srgb8(i), Rendering::Natural) => stretch_contrast(i),
            // The JPEG's shadows are already blocky; lifting them is the best that is left.
            (Pixels::Srgb8(i), Rendering::EyesLifted) => {
                encode(&to_linear(i), |v| lifted_curve(v * g * LIFTED_MEDIAN))
            }
        }
    }
}

/// Display value (0–1) of a normalised linear value `v`, where `white` is what the sensor's
/// full scale became: a gentle shoulder lands it on white, then sRGB.
fn natural_curve(v: f32, white: f32) -> f32 {
    let toned = tone::shoulder(f64::from(v.max(0.0)), f64::from(white), SHOULDER_START);
    tone::linear_to_srgb(toned) as f32
}

/// Display value (0–1) of a normalised linear value `v`: a logarithm, so each stop of shadow
/// gets about the same share of the range, which is what opens the inside of a dark lens.
fn lifted_curve(v: f32) -> f32 {
    ((1.0 + LIFT_K * v.max(0.0)).ln() / (1.0 + LIFT_K).ln()).min(1.0)
}

fn encode(img: &Rgb32FImage, curve: impl Fn(f32) -> f32) -> RgbImage {
    let mut out = RgbImage::new(img.width(), img.height());
    for (o, p) in out.pixels_mut().zip(img.pixels()) {
        o.0 = p.0.map(|v| (curve(v) * 255.0).round() as u8);
    }
    out
}

fn to_linear(img: &RgbImage) -> Rgb32FImage {
    let lut = tone::srgb8_to_linear().map(|v| v as f32);
    let mut out = Rgb32FImage::new(img.width(), img.height());
    for (o, p) in out.pixels_mut().zip(img.pixels()) {
        o.0 = p.0.map(|v| lut[usize::from(v)]);
    }
    out
}

/// Stretches the crop's luma between its 1st and 99th percentile onto the full range, so a
/// face in shade or against the sun shows its eyes.
fn stretch_contrast(img: &RgbImage) -> RgbImage {
    let mut hist = [0u32; 256];
    for p in img.pixels() {
        let l =
            (0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])) as usize;
        hist[l.min(255)] += 1;
    }
    let total: u32 = hist.iter().sum();
    let pct = |q: f64| {
        let target = (f64::from(total) * q) as u32;
        let mut acc = 0;
        hist.iter()
            .position(|&c| {
                acc += c;
                acc > target
            })
            .unwrap_or(255) as f64
    };
    let (lo, hi) = (pct(0.01), pct(0.99));
    if hi - lo < 16.0 {
        return img.clone();
    }
    let mut out = img.clone();
    for p in out.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = ((f64::from(*c) - lo) / (hi - lo) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIENTATIONS: [Orientation; 4] = [
        Orientation::Normal,
        Orientation::Rotate90Cw,
        Orientation::Rotate180,
        Orientation::Rotate270Cw,
    ];

    /// A stored image whose every pixel encodes its own coordinates.
    fn coordinates(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([x as u8, y as u8, (x / 256 + 16 * (y / 256)) as u8])
        })
    }

    fn face(x0: f64, y0: f64, x1: f64, y1: f64) -> Box2d {
        Box2d { x0, y0, x1, y1 }
    }

    /// Cutting the union of two faces from the stored image, then each face from the union,
    /// gives the same pixels as cutting each padded face from the displayed image directly —
    /// in every orientation, so a rotation applied on the wrong side shows as a different image.
    #[test]
    fn union_then_face_cut_lands_on_the_displayed_faces() {
        // Large enough that no face's padding is clipped by the frame, so the union does not
        // start at the frame's corner.
        let stored = coordinates(600, 400);
        let boxes = [face(0.3, 0.3, 0.4, 0.45), face(0.55, 0.5, 0.7, 0.65)];
        for o in ORIENTATIONS {
            let shown = o.display(&stored);
            let plan = Plan::new(&boxes, shown.width(), shown.height());
            let union = Source::Camera(upright_cut(&stored, o, plan.union));
            assert_eq!(union.size(), (plan.union.w, plan.union.h), "{o:?}");
            for (b, rel) in boxes.iter().zip(&plan.faces) {
                let direct = face_rect(*b, shown.width(), shown.height()).crop;
                let want =
                    imageops::crop_imm(&shown, direct.x, direct.y, direct.w, direct.h).to_image();
                let Pixels::Srgb8(got) = union.cut(rel.crop) else {
                    unreachable!()
                };
                assert_eq!(got, want, "{o:?}");
            }
        }
    }

    #[test]
    fn padding_stays_inside_the_frame_and_the_core_is_the_box() {
        let f = face_rect(face(0.0, 0.5, 0.1, 0.6), 1000, 800);
        // The locate-error margin is 10 px across and 8 px down in a 1000×800 frame.
        assert_eq!((f.crop.x, f.crop.y), (0, 400 - 20 - 48 - 8));
        // 100 px wide box: 25 + 48 + 10 px each side, clipped at the left edge.
        assert_eq!(f.crop.w, 100 + 83);
        assert_eq!(f.crop.h, 80 + 2 * (20 + 48 + 8));
        assert_eq!(
            f.core,
            PixelRect {
                x: 0,
                y: 20 + 48 + 8,
                w: 100,
                h: 80
            }
        );
        let inner = face_rect(face(0.3, 0.5, 0.4, 0.6), 1000, 800);
        assert_eq!(
            (inner.crop.x, inner.crop.y),
            (300 - 25 - 48 - 10, 400 - 20 - 48 - 8)
        );
        assert_eq!((inner.core.x, inner.core.y), (25 + 48 + 10, 20 + 48 + 8));
        let edge = face_rect(face(0.9, 0.9, 1.0, 1.0), 1000, 800).crop;
        assert_eq!(edge.x + edge.w, 1000);
        assert_eq!(edge.y + edge.h, 800);
    }

    /// A synthetic upright face of 200×300 px with its eyes at 35–50 % of its height: the band
    /// is inside the face box vertically, holds both eyes' rows, reaches past the box's sides,
    /// and stays inside the crop and the frame, also for a face at the frame's edge.
    #[test]
    fn eye_band_holds_the_eyes_of_an_upright_face() {
        let (w, h) = (1000, 800);
        for b in [
            face(0.3, 0.2, 0.5, 0.575),
            face(0.0, 0.0, 0.2, 0.375),
            face(0.8, 0.625, 1.0, 1.0),
        ] {
            let f = face_rect(b, w, h);
            let (core, band, crop) = (f.core, f.band, f.crop);
            assert_eq!((core.w, core.h), (200, 300), "{b:?}");
            // Rows of the eyes' centres, relative to the crop.
            let eyes = (core.y + core.h * 35 / 100)..=(core.y + core.h * 50 / 100);
            assert!(
                band.y >= core.y && band.y + band.h <= core.y + core.h,
                "{b:?}: band rows inside the face box"
            );
            assert!(
                band.y <= *eyes.start() && *eyes.end() < band.y + band.h,
                "{b:?}: both eye rows inside the band"
            );
            assert!(
                band.x + band.w <= crop.w && band.y + band.h <= crop.h,
                "{b:?}: band inside the crop"
            );
            // Both eyes are at 20–80 % of the face's width, whatever the clipping at the frame.
            let eye_columns = [core.x + core.w * 20 / 100, core.x + core.w * 80 / 100];
            for x in eye_columns {
                assert!((band.x..band.x + band.w).contains(&x), "{b:?}: column {x}");
            }
            if b.x0 > 0.0 && b.x1 < 1.0 {
                assert!(
                    band.x < core.x && band.x + band.w > core.x + core.w,
                    "{b:?}"
                );
            }
        }
    }

    /// The eye band and face crop of a box that is `dx`, `dy` (fractions of the frame) off the
    /// true face, in frame pixels: whether they hold the true face's eyes (rows 35–50 % of its
    /// height, columns 20 % and 80 % of its width) and the whole true box, plus the band's size.
    fn locate_off_by(truth: Box2d, (dx, dy): (f64, f64), (w, h): (u32, u32)) -> (bool, (f64, f64)) {
        let (fw, fh) = (f64::from(w), f64::from(h));
        let located = face(truth.x0 + dx, truth.y0 + dy, truth.x1 + dx, truth.y1 + dy);
        let f = face_rect(located, w, h);
        let (cx, cy) = (f64::from(f.crop.x), f64::from(f.crop.y));
        let band = (
            (cx + f64::from(f.band.x))..(cx + f64::from(f.band.x + f.band.w)),
            (cy + f64::from(f.band.y))..(cy + f64::from(f.band.y + f.band.h)),
        );
        let (tw, th) = ((truth.x1 - truth.x0) * fw, (truth.y1 - truth.y0) * fh);
        let (tx, ty) = (truth.x0 * fw, truth.y0 * fh);
        let eye_rows = [ty + th * 0.35, ty + th * 0.50];
        let eye_columns = [tx + tw * 0.20, tx + tw * 0.80];
        let holds_eyes = eye_rows.iter().all(|y| band.1.contains(y))
            && eye_columns.iter().all(|x| band.0.contains(x));
        let crop_holds_face = cx <= tx
            && cy <= ty
            && tx + tw <= cx + f64::from(f.crop.w)
            && ty + th <= cy + f64::from(f.crop.h);
        (
            holds_eyes && crop_holds_face,
            (f64::from(f.band.w), f64::from(f.band.h)),
        )
    }

    /// The required tolerance, 10 grid units, written out and not read from the code's margin.
    const LOCATE_OFFSETS: [f64; 3] = [-0.010, 0.0, 0.010];

    /// A face 2 % of the frame wide, so that a locate error of 10 grid units is half the face:
    /// the band and the face crop hold the true face's eyes for every shift of the box by up to
    /// that error, vertically and horizontally.
    #[test]
    fn eye_band_holds_the_eyes_of_a_small_face_whose_box_is_off() {
        let truth = face(0.50, 0.40, 0.52, 0.43);
        for dx in LOCATE_OFFSETS {
            for dy in LOCATE_OFFSETS {
                let (ok, _) = locate_off_by(truth, (dx, dy), (6000, 4000));
                assert!(ok, "box off by ({dx}, {dy})");
            }
        }
    }

    /// A face 30 % of the frame wide: the same holds, and the margin adds no more than itself,
    /// on each side, to the band.
    #[test]
    fn the_locate_margin_adds_only_itself_to_a_large_face() {
        let (w, h) = (6000, 4000);
        let truth = face(0.30, 0.20, 0.60, 0.50);
        for dx in LOCATE_OFFSETS {
            for dy in LOCATE_OFFSETS {
                let (ok, (band_w, band_h)) = locate_off_by(truth, (dx, dy), (w, h));
                assert!(ok, "box off by ({dx}, {dy})");
                let (face_w, face_h) = (0.30 * f64::from(w), 0.30 * f64::from(h));
                let margin = (
                    LOCATE_ERROR_FRAC * f64::from(w),
                    LOCATE_ERROR_FRAC * f64::from(h),
                );
                // The box and the band each round their edges outwards to whole pixels.
                assert!(
                    band_w <= face_w * (1.0 + 2.0 * BAND_PAD_FRAC) + 2.0 * margin.0 + 4.0,
                    "{band_w}"
                );
                assert!(
                    band_h <= face_h * (BAND_BOTTOM_FRAC - BAND_TOP_FRAC) + 2.0 * margin.1 + 4.0,
                    "{band_h}"
                );
            }
        }
    }

    /// The box `-B 8000 5000 1000 1000` on an 8280×5520 raw comes back 280×520.
    #[test]
    fn a_clipped_decode_is_refused() {
        let asked = PixelRect {
            x: 8000,
            y: 5000,
            w: 1000,
            h: 1000,
        };
        let clipped = Source::Raw16(LinearU16::new(280, 520));
        assert!(clipped.expect_size(asked).is_err());
        let whole = Source::Raw16(LinearU16::new(1000, 1000));
        assert!(whole.expect_size(asked).is_ok());
    }

    #[test]
    fn curves_are_monotone_and_reach_white() {
        let steps = || (0..=2000).map(|i| f32::from(i as u16) * 0.001);
        for white in [1.0f32, 4.0, 20.0] {
            let ys: Vec<f32> = steps().map(|v| natural_curve(v, white)).collect();
            assert!(
                ys.windows(2).all(|w| w[0] <= w[1]),
                "natural, white {white}"
            );
            assert!((natural_curve(white, white) - 1.0).abs() < 1e-6);
        }
        let ys: Vec<f32> = steps().map(lifted_curve).collect();
        assert!(ys.windows(2).all(|w| w[0] <= w[1]), "lifted");
        assert_eq!(lifted_curve(0.0), 0.0);
        assert!((lifted_curve(1.0) - 1.0).abs() < 1e-6);
    }

    /// A scene point `s` times the face's median luma, through each rendering at a face gain
    /// of 4 (white = 4× the median's gain): deep shadows, such as the inside of a dark lens
    /// at a twentieth of the skin (s = 0.05), come out much brighter lifted than natural,
    /// while the skin itself does not clip.
    #[test]
    fn eyes_lifted_raises_deep_shadows_more_than_natural() {
        let white = 4.0 * NATURAL_MEDIAN;
        for s in [0.01f32, 0.02, 0.05, 0.1] {
            let n = natural_curve(s * NATURAL_MEDIAN, white);
            let l = lifted_curve(s * LIFTED_MEDIAN);
            assert!(l > 1.4 * n, "s = {s}: lifted {l}, natural {n}");
        }
        assert!(lifted_curve(LIFTED_MEDIAN) < 0.6);
        assert!(lifted_curve(3.0 * LIFTED_MEDIAN) < 0.8);
    }

    /// A face with median luma 0.02 is gained up 50-fold; a black crop is bounded by the gain
    /// range instead of amplified without limit.
    #[test]
    fn gain_normalises_the_median_and_is_bounded() {
        let grey = |v: f32| Rgb32FImage::from_pixel(4, 4, image::Rgb([v; 3]));
        assert!((Gain::of_median_luma(&grey(0.02)).0 - 50.0).abs() < 1e-3);
        assert_eq!(Gain::of_median_luma(&grey(0.0)).0, GAIN_RANGE.1);
        assert_eq!(Gain::of_median_luma(&grey(1.0)).0, GAIN_RANGE.0);
    }

    /// A dim face (0.05) in front of a bright background (0.8) is exposed for the face: the
    /// gain comes from the core, not from the crop's majority.
    #[test]
    fn gain_follows_the_face_box_not_the_background() {
        let core = PixelRect {
            x: 2,
            y: 2,
            w: 2,
            h: 2,
        };
        // 3277 of 65535 is 0.050004.
        let union = Source::Raw16(LinearU16::from_fn(6, 6, |x, y| {
            let inside = (2..4).contains(&x) && (2..4).contains(&y);
            image::Rgb([if inside { 3277 } else { 52428 }; 3])
        }));
        assert!((union.cut(core).gain().0 - 20.0).abs() < 1e-2);
    }

    /// The faces of a synthetic raw union, rendered from the 16-bit source, come out
    /// pixel-for-pixel as they did when the whole decode was converted to floating point
    /// first and each face cut from that (the reference below), in both crops and for faces
    /// of different gain, and the PPM parses to the same samples as the buffered decoder.
    #[test]
    fn rendering_from_the_16_bit_union_matches_the_floating_point_union() {
        let boxes = [face(0.3, 0.3, 0.4, 0.45), face(0.55, 0.5, 0.7, 0.65)];
        let plan = Plan::new(&boxes, 1000, 800);
        let (w, h) = (plan.union.w, plan.union.h);
        // A P6 file: a header and big-endian 16-bit samples that differ from pixel to pixel
        // and channel to channel, with every seventh column dim.
        let sample = |x: u32, y: u32, c: u32| -> u16 {
            ((x * 37 + y * 101 + c * 4001) % 65536) as u16
                / (1 + u16::from(x.is_multiple_of(7)) * 9)
        };
        let mut ppm = format!("P6\n{w} {h}\n65535\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    ppm.extend_from_slice(&sample(x, y, c).to_be_bytes());
                }
            }
        }
        let reference = image::load_from_memory_with_format(&ppm, image::ImageFormat::Pnm)
            .unwrap()
            .into_rgb32f();
        let streamed = develop::read_linear_u16(&ppm[..]).unwrap();
        assert_eq!(
            streamed.get_pixel(5, 3).0,
            [sample(5, 3, 0), sample(5, 3, 1), sample(5, 3, 2)]
        );
        let union = Source::Raw16(streamed);

        let cut = |r: PixelRect| {
            Pixels::Linear(imageops::crop_imm(&reference, r.x, r.y, r.w, r.h).to_image())
        };
        for f in &plan.faces {
            let cropped = cut(f.crop);
            let Pixels::Linear(cropped_f32) = &cropped else {
                unreachable!()
            };
            let rel = |r: PixelRect| {
                Pixels::Linear(imageops::crop_imm(cropped_f32, r.x, r.y, r.w, r.h).to_image())
            };
            let gain = rel(f.core).gain();
            let want: Vec<(Crop, RgbImage)> = Crop::ALL
                .iter()
                .map(|&c| {
                    let image = rel(f.region(c)).fit(LONG_EDGE_PX);
                    (c, image.render(c.rendering(), gain))
                })
                .collect();
            assert_eq!(render_face(&union, f).to_vec(), want);
        }
    }

    #[test]
    fn rendering_is_deterministic_and_never_enlarges() {
        let lin = Pixels::Linear(Rgb32FImage::from_fn(900, 700, |x, y| {
            image::Rgb([x as f32 / 900.0, y as f32 / 700.0, 0.01])
        }));
        let fitted = lin.fit(LONG_EDGE_PX);
        assert_eq!(fitted.size(), (768, 597));
        let gain = Gain(2.0);
        for c in Crop::ALL {
            let r = c.rendering();
            assert_eq!(fitted.render(r, gain), fitted.render(r, gain), "{r:?}");
        }
        let small = Pixels::Srgb8(RgbImage::new(100, 50)).fit(LONG_EDGE_PX);
        assert_eq!(small.size(), (100, 50));
    }

    /// Two faces of 100×200 and 300×200 pixels become 96 and 288 pixels wide at the contact
    /// strip's face height of 192, with the gap between; their bands (white faces, grey bands, the
    /// same shapes) are 48 and 144 wide at 96, in a row under them.
    #[test]
    fn contact_strip_sets_faces_side_by_side_with_their_bands_under() {
        let dir = tempfile::tempdir().unwrap();
        let files: Vec<FaceFiles> = [(100, 200), (300, 200)]
            .iter()
            .enumerate()
            .map(|(i, &(w, h))| {
                let f = FaceFiles::named(dir.path(), &format!("face{i}"));
                RgbImage::from_pixel(w, h, image::Rgb([255; 3]))
                    .save(f.get(Crop::Face))
                    .unwrap();
                RgbImage::from_pixel(w, h, image::Rgb([128; 3]))
                    .save(f.get(Crop::EyeBand))
                    .unwrap();
                f
            })
            .collect();
        let path = dir.path().join("contact.jpg");
        sync_contact_strip(&path, &files).unwrap();
        let contact = image::open(&path).unwrap().into_rgb8();
        assert_eq!(
            contact.dimensions(),
            (
                96 + CONTACT_GAP_PX + 288,
                CONTACT_FACE_HEIGHT_PX + CONTACT_GAP_PX + CONTACT_BAND_HEIGHT_PX
            )
        );
        assert!(contact.get_pixel(50, 96)[0] > 240, "first face");
        assert!(
            contact.get_pixel(96 + CONTACT_GAP_PX / 2, 96)[0] < 16,
            "gap"
        );
        assert!(
            contact.get_pixel(96 + CONTACT_GAP_PX + 50, 96)[0] > 240,
            "second face"
        );
        let band_y = CONTACT_FACE_HEIGHT_PX + CONTACT_GAP_PX + CONTACT_BAND_HEIGHT_PX / 2;
        assert!(
            contact.get_pixel(20, band_y)[0].abs_diff(128) < 8,
            "first band"
        );
        assert!(
            contact.get_pixel(48 + CONTACT_GAP_PX + 50, band_y)[0].abs_diff(128) < 8,
            "second band"
        );
        assert!(
            contact.get_pixel(20, CONTACT_FACE_HEIGHT_PX + CONTACT_GAP_PX / 2)[0] < 16,
            "gap between rows"
        );
    }

    /// The report shows a contact strip only where the file exists: a frame the last run found
    /// no faces in must not keep the one an earlier run made, and one never made is no error.
    #[test]
    fn a_frame_without_faces_has_no_contact_strip() {
        let dir = tempfile::tempdir().unwrap();
        let strip = dir.path().join("contact.jpg");
        let f = FaceFiles::named(dir.path(), "face");
        for c in Crop::ALL {
            RgbImage::new(8, 8).save(f.get(c)).unwrap();
        }
        sync_contact_strip(&strip, &[f]).unwrap();
        assert!(strip.is_file());
        sync_contact_strip(&strip, &[]).unwrap();
        assert!(!strip.exists());
        sync_contact_strip(&strip, &[]).unwrap();
    }

    /// A crop is rendered as its purpose wants, and the two purposes ask for different images.
    #[test]
    fn each_crop_has_its_own_rendering() {
        assert_eq!(Crop::Face.rendering(), Rendering::Natural);
        assert_eq!(Crop::EyeBand.rendering(), Rendering::EyesLifted);
    }

    #[test]
    fn camera_pixels_get_both_renderings() {
        let px = Pixels::Srgb8(RgbImage::from_fn(64, 64, |x, _| {
            image::Rgb([(x * 2) as u8; 3])
        }));
        let gain = px.gain();
        let (n, l) = (
            px.render(Rendering::Natural, gain),
            px.render(Rendering::EyesLifted, gain),
        );
        assert_eq!(n.dimensions(), l.dimensions());
        assert!(l.get_pixel(8, 0)[0] > n.get_pixel(8, 0)[0]);
    }

    fn rect(x: u32, y: u32, w: u32, h: u32) -> PixelRect {
        PixelRect { x, y, w, h }
    }

    /// A face whose crop is the whole 120×100 union, with a 100×80 face box and, inside it, a
    /// 100×20 eye band.
    fn banded_face() -> FaceRect {
        FaceRect {
            crop: rect(0, 0, 120, 100),
            core: rect(10, 10, 100, 80),
            band: rect(10, 40, 100, 20),
        }
    }

    /// The strip box is widened by 4 % of the band on every side and kept inside the band. The
    /// expected rectangles are worked out from a 200×100 band at (10, 20): a box at 25–75 %
    /// across and 30–60 % down becomes 21–79 % and 26–64 %, i.e. 42..158 and 26..64 px.
    #[test]
    fn eye_strip_region_is_widened_by_the_margin_and_kept_in_the_band() {
        let f = FaceRect {
            crop: rect(0, 0, 400, 300),
            core: rect(0, 0, 400, 300),
            band: rect(10, 20, 200, 100),
        };
        let strip = |grid| f.eye_strip_region(BandBox::from_grid(grid).unwrap());
        assert_eq!(strip([300, 250, 600, 750]), rect(10 + 42, 20 + 26, 116, 38));
        // Against the band's left edge and bottom, the margin does not leave it: 0..54 % across
        // and 46..100 % down.
        assert_eq!(strip([500, 0, 1000, 500]), rect(10, 20 + 46, 108, 54));
        assert_eq!(strip([0, 0, 1000, 1000]), f.band);
    }

    /// The strip is exposed by its own pixels and enlarged to 1024 px wide. A dark strip (0.01 of
    /// full scale) inside a bright face box (0.5) is gained to the lifted median, 0.05, and
    /// through the log curve `ln(1 + 800 v) / ln 801` that is 0.5554, i.e. 142 of 255. Exposed
    /// by the face box, whose median is the bright 0.5, it would come out near 22.
    #[test]
    fn strip_is_exposed_by_itself_and_enlarged_to_the_strip_width() {
        let f = banded_face();
        let union = Source::Raw16(LinearU16::from_fn(120, 100, |x, y| {
            let band = (10..110).contains(&x) && (40..60).contains(&y);
            // 655 of 65535 is 0.0099946, 32768 is 0.50000.
            image::Rgb([if band { 655 } else { 32768 }; 3])
        }));
        let whole_band = BandBox::from_grid([0, 0, 1000, 1000]).unwrap();
        let strip = render_eye_strip(&union, &f, whole_band);
        // A 100×20 band, 10.24 times wider.
        assert_eq!(strip.dimensions(), (EYE_STRIP_WIDTH_PX, 205));
        for (x, y) in [(5, 5), (512, 100), (1018, 200)] {
            let v = strip.get_pixel(x, y)[0];
            assert!(v.abs_diff(142) <= 1, "pixel ({x}, {y}) is {v}");
        }
    }

    #[test]
    fn scaling_to_the_strip_width_enlarges_and_reduces_keeping_the_aspect() {
        assert_eq!(
            scale_to_width(&RgbImage::new(100, 50), EYE_STRIP_WIDTH_PX).dimensions(),
            (1024, 512)
        );
        assert_eq!(
            scale_to_width(&RgbImage::new(2048, 500), EYE_STRIP_WIDTH_PX).dimensions(),
            (1024, 250)
        );
    }

    /// A strip file is named by everything that determines its bytes: the same inputs name the
    /// same file, and the strip box, the face and the source each change the name.
    #[test]
    fn eye_strip_files_are_named_by_what_determines_them() {
        let dir = tempfile::tempdir().unwrap();
        let nef = dir.path().join("DSC_0001.NEF");
        std::fs::write(&nef, [0u8; 8]).unwrap();
        let shot = Shot {
            file: nef,
            file_size: 8,
            taken_at: chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00+00:00").unwrap(),
            preview: crate::shot::PreviewSpan { offset: 0, len: 0 },
            af: None,
            orientation: Orientation::Normal,
            exposure: crate::shot::Exposure {
                shutter_s: 0.01,
                focal_mm: 50.0,
                iso: 100,
                picture_control: None,
                compensation_ev: None,
            },
        };
        let boxes = [face(0.3, 0.3, 0.4, 0.45), face(0.55, 0.5, 0.7, 0.65)];
        let layout = Layout {
            orientation: Orientation::Normal,
            stored: (1000, 800),
            plan: Plan::new(&boxes, 1000, 800),
        };
        let (a, b) = (&layout.plan.faces[0], &layout.plan.faces[1]);
        let strip = |g| BandBox::from_grid(g).unwrap();
        let path = |shown, face: &FaceRect, strip| {
            layout
                .eye_strip_path(dir.path(), &shot, shown, face, strip)
                .unwrap()
        };
        let base = path(Shown::Camera, a, strip([200, 100, 600, 900]));
        assert_eq!(base, path(Shown::Camera, a, strip([200, 100, 600, 900])));
        assert_ne!(base, path(Shown::Camera, a, strip([200, 100, 600, 901])));
        assert_ne!(base, path(Shown::Camera, b, strip([200, 100, 600, 900])));
        assert_ne!(base, path(Shown::Developed, a, strip([200, 100, 600, 900])));
        assert!(
            base.to_string_lossy().ends_with(".eye_strip.jpg"),
            "{base:?}"
        );
    }

    /// Checks against real raws and the dcraw_emu convention the code relies on. They need
    /// libraw's tools on PATH and read the example shoot in `RAWPILOT_NEF_DIR`, which is never
    /// written: `RAWPILOT_NEF_DIR=<dir> cargo test -- --ignored real_`.
    mod real {
        use super::*;
        use std::sync::OnceLock;

        fn shot(name: &str) -> Shot {
            static ALL: OnceLock<Vec<Shot>> = OnceLock::new();
            let all = ALL.get_or_init(|| {
                let dir = std::env::var_os("RAWPILOT_NEF_DIR").expect("RAWPILOT_NEF_DIR");
                crate::nef::scan(Path::new(&dir)).unwrap()
            });
            all.iter().find(|s| s.name() == name).unwrap().clone()
        }

        fn mean_abs_diff(a: &Rgb32FImage, b: &Rgb32FImage, border: u32) -> f32 {
            assert_eq!(a.dimensions(), b.dimensions());
            let (mut sum, mut n) = (0.0f64, 0u64);
            for y in border..a.height() - border {
                for x in border..a.width() - border {
                    for c in 0..3 {
                        sum += f64::from((a.get_pixel(x, y)[c] - b.get_pixel(x, y)[c]).abs());
                        n += 1;
                    }
                }
            }
            (sum / n as f64) as f32
        }

        /// The convention: `-B` is in stored coordinates and the output comes back upright, so
        /// a 90° orientation swaps the box's sides; a box past the edge is clipped.
        #[test]
        #[ignore]
        fn real_dcraw_crop_is_stored_box_upright_output() {
            for (name, o) in [
                ("DSC_3234.NEF", Orientation::Rotate270Cw),
                ("DSC_3310.NEF", Orientation::Normal),
            ] {
                let s = shot(name);
                assert_eq!(s.orientation, o);
                let (sw, sh) = develop::stored_size(&s.file).unwrap();
                assert_eq!((sw, sh), (8280, 5520));
                let r = PixelRect {
                    x: 1000,
                    y: 500,
                    w: 1200,
                    h: 400,
                };
                let img = develop::decode_crop(&s.file, r).unwrap();
                assert_eq!(img.dimensions(), o.displayed_size(r.w, r.h), "{name}");
                let again = develop::decode_crop(&s.file, r).unwrap();
                assert_eq!(img, again, "{name}: output is deterministic");
                let corner = PixelRect {
                    x: sw - 10,
                    y: sh - 20,
                    w: 100,
                    h: 100,
                };
                let clipped = develop::decode_crop(&s.file, corner).unwrap();
                assert_eq!(clipped.dimensions(), o.displayed_size(10, 20), "{name}");
            }
        }

        /// The union decode, cut by the plan, shows the same pixels as decoding each face's
        /// own rectangle (to within demosaic at the edges), including in a rotated file.
        #[test]
        #[ignore]
        fn real_union_cut_matches_decoding_each_face() {
            let s = shot("DSC_3234.NEF");
            let (sw, sh) = develop::stored_size(&s.file).unwrap();
            let (w, h) = s.orientation.displayed_size(sw, sh);
            let boxes = [face(0.40, 0.30, 0.46, 0.36), face(0.50, 0.38, 0.56, 0.44)];
            let plan = Plan::new(&boxes, w, h);
            let union =
                develop::decode_crop(&s.file, s.orientation.stored_rect(plan.union, sw, sh))
                    .unwrap();
            assert_eq!(union.dimensions(), (plan.union.w, plan.union.h));
            let union = Source::Raw16(union);
            for (b, rel) in boxes.iter().zip(&plan.faces) {
                let abs = face_rect(*b, w, h).crop;
                let direct = DynamicImage::ImageRgb16(
                    develop::decode_crop(&s.file, s.orientation.stored_rect(abs, sw, sh)).unwrap(),
                )
                .into_rgb32f();
                let Pixels::Linear(cut) = union.cut(rel.crop) else {
                    unreachable!()
                };
                assert!(mean_abs_diff(&cut, &direct, 16) < 0.01);
            }
        }

        /// Streaming dcraw_emu's output into the 16-bit image gives the samples of buffering
        /// the whole PPM and decoding it, the way the crop was once read.
        #[test]
        #[ignore]
        fn real_streamed_decode_equals_the_buffered_one() {
            let s = shot("DSC_3234.NEF");
            let r = PixelRect {
                x: 3000,
                y: 2000,
                w: 700,
                h: 500,
            };
            let out = std::process::Command::new("dcraw_emu")
                .args(["-4", "-W", "-w", "-o", "1", "-B"])
                .args([r.x, r.y, r.w, r.h].map(|v| v.to_string()))
                .args(["-Z", "-"])
                .arg(&s.file)
                .output()
                .unwrap();
            assert!(out.status.success());
            let buffered =
                image::load_from_memory_with_format(&out.stdout, image::ImageFormat::Pnm)
                    .unwrap()
                    .into_rgb32f();
            let streamed = develop::decode_crop(&s.file, r).unwrap();
            assert_eq!(DynamicImage::ImageRgb16(streamed).into_rgb32f(), buffered);
        }

        /// Renders real faces and eye bands, from the raw and from the camera JPEG, into the
        /// directory named by `RAWPILOT_CROP_OUT`, for looking at.
        #[test]
        #[ignore]
        fn real_render_faces_for_inspection() {
            let out = PathBuf::from(std::env::var("RAWPILOT_CROP_OUT").unwrap());
            // Single face; single face in an orientation-8 frame; two faces; two frames of a
            // burst in which the main subject wears sunglasses; a small face, ~3 % of the
            // frame wide, whose located box is off by a sizeable part of it.
            let cases: [(&str, Vec<Box2d>); 6] = [
                ("DSC_3310.NEF", vec![face(0.554, 0.547, 0.602, 0.601)]),
                ("DSC_3412.NEF", vec![face(0.421, 0.380, 0.457, 0.406)]),
                ("DSC_3443.NEF", vec![face(0.417, 0.630, 0.461, 0.668)]),
                (
                    "DSC_3197.NEF",
                    vec![
                        face(0.235, 0.265, 0.315, 0.415),
                        face(0.675, 0.405, 0.775, 0.555),
                    ],
                ),
                ("DSC_3470.NEF", vec![face(0.444, 0.403, 0.480, 0.459)]),
                ("DSC_3473.NEF", vec![face(0.444, 0.403, 0.479, 0.459)]),
            ];
            for (name, boxes) in cases {
                let s = shot(name);
                for shown in [Shown::Developed, Shown::Camera] {
                    let dir = out.join(format!("{shown:?}").to_lowercase());
                    std::fs::create_dir_all(&dir).unwrap();
                    let layout = Layout::new(&s, shown, &boxes).unwrap();
                    let files = face_files(&dir, &s, shown, &layout).unwrap();
                    assert_eq!(files, face_files(&dir, &s, shown, &layout).unwrap());
                    for f in &files {
                        for c in Crop::ALL {
                            assert!(f.get(c).is_file());
                        }
                    }
                    // The middle of each eye band as its strip, read back to be 1024 px wide.
                    let band = BandBox::from_grid([100, 0, 800, 1000]).unwrap();
                    let bands = vec![band; boxes.len()];
                    let strips = eye_strip_files(&dir, &s, shown, &layout, &bands).unwrap();
                    assert_eq!(
                        strips,
                        eye_strip_files(&dir, &s, shown, &layout, &bands).unwrap()
                    );
                    for p in &strips {
                        assert_eq!(image::image_dimensions(p).unwrap().0, EYE_STRIP_WIDTH_PX);
                    }
                }
            }
        }
    }
}
