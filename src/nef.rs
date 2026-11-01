//! In-process reader for the metadata rawpilot culls on, from the header region of a Nikon Z8
//! NEF (a TIFF file). Only the IFD tables and the handful of values wanted are read; the
//! 33 MB of image data is never touched. Every offset and count in the file is untrusted: a
//! region outside the file, or a value of the wrong type, is an `Err` for that file. The one
//! exception is the Nikon MakerNote, whose fields (AF box, Picture Control) only refine a
//! shot: a MakerNote or field that cannot be read is logged and left unknown.
//!
//! The IFDs are visited by a fixed descent (IFD0 → SubIFD 0, IFD0 → Exif IFD → MakerNote
//! IFD); next-IFD pointers are never followed, so a looping chain cannot occur.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use chrono::{DateTime, FixedOffset};
use tracing::warn;

use crate::shot::{AfBox, Exposure, Orientation, PixelRect, PreviewSpan, Shot};

const TAG_ORIENTATION: u16 = 0x0112;
const TAG_SUB_IFDS: u16 = 0x014a;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_PREVIEW_OFFSET: u16 = 0x0201;
const TAG_PREVIEW_LEN: u16 = 0x0202;
const TAG_EXPOSURE_TIME: u16 = 0x829a;
/// The Exif IFD's ISO, which is the value exiftool's plain `ISO` resolves to on a Z8 (the
/// MakerNote's own ISO is a float, 79.37 for the nominal 80).
const TAG_ISO: u16 = 0x8827;
const TAG_DATE_TIME_ORIGINAL: u16 = 0x9003;
const TAG_OFFSET_TIME_ORIGINAL: u16 = 0x9011;
const TAG_EXPOSURE_BIAS: u16 = 0x9204;
const TAG_FOCAL_LENGTH: u16 = 0x920a;
const TAG_MAKER_NOTE: u16 = 0x927c;
const TAG_SUB_SEC_TIME_ORIGINAL: u16 = 0x9291;
const TAG_PICTURE_CONTROL: u16 = 0x0023;
const TAG_AF_INFO2: u16 = 0x00b7;

const IFD_ENTRY_LEN: usize = 12;
const TIFF_HEADER_LEN: u64 = 8;
/// `Nikon\0`, a 2-byte version and 2 reserved bytes precede the MakerNote's own TIFF header.
const NIKON_MAKER_NOTE_PREFIX: &[u8; 6] = b"Nikon\0";
const NIKON_MAKER_NOTE_TIFF_AT: u64 = 10;
/// The longest ASCII value read: a date-time is 20 bytes, an offset time 7.
const ASCII_LEN_MAX: u64 = 64;

/// Byte order of one TIFF structure, fixed by its header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    fn u16(self, b: [u8; 2]) -> u16 {
        match self {
            ByteOrder::Little => u16::from_le_bytes(b),
            ByteOrder::Big => u16::from_be_bytes(b),
        }
    }

    fn u32(self, b: [u8; 4]) -> u32 {
        match self {
            ByteOrder::Little => u32::from_le_bytes(b),
            ByteOrder::Big => u32::from_be_bytes(b),
        }
    }

    /// The `u16` at `at` in `bytes`, if they are long enough.
    fn u16_at(self, bytes: &[u8], at: usize) -> Option<u16> {
        let b = bytes.get(at..at + 2)?;
        Some(self.u16([b[0], b[1]]))
    }
}

/// A TIFF header: the byte order and where the first IFD starts, relative to the header.
fn parse_tiff_header(b: &[u8]) -> Result<(ByteOrder, u32)> {
    let b: [u8; 8] = b.try_into().context("TIFF header is not 8 bytes")?;
    let order = match &b[..2] {
        b"II" => ByteOrder::Little,
        b"MM" => ByteOrder::Big,
        other => bail!("not a TIFF header: byte order mark {other:02x?}"),
    };
    ensure!(
        order.u16([b[2], b[3]]) == 42,
        "not a TIFF header: bad magic number"
    );
    Ok((order, order.u32([b[4], b[5], b[6], b[7]])))
}

/// The TIFF field types this reader understands; the rest only need their entry skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FieldType {
    Byte,
    Ascii,
    Short,
    Long,
    Rational,
    Undefined,
    SRational,
    /// A `Long` that is an IFD offset (SubIFD pointers).
    Ifd,
    Other(u16),
}

impl FieldType {
    fn from_code(code: u16) -> FieldType {
        match code {
            1 => FieldType::Byte,
            2 => FieldType::Ascii,
            3 => FieldType::Short,
            4 => FieldType::Long,
            5 => FieldType::Rational,
            7 => FieldType::Undefined,
            10 => FieldType::SRational,
            13 => FieldType::Ifd,
            other => FieldType::Other(other),
        }
    }

    fn unit_len(self) -> Result<u64> {
        match self {
            FieldType::Byte | FieldType::Ascii | FieldType::Undefined => Ok(1),
            FieldType::Short => Ok(2),
            FieldType::Long | FieldType::Ifd => Ok(4),
            FieldType::Rational | FieldType::SRational => Ok(8),
            FieldType::Other(code) => bail!("unsupported TIFF field type {code}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Entry {
    ty: FieldType,
    count: u32,
    /// The value itself when it fits in 4 bytes, otherwise its offset.
    field: [u8; 4],
}

/// A value of at most this many bytes lives in the entry's own `field`.
const INLINE_VALUE_LEN: usize = 4;

/// Where an entry's value is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    /// In the entry itself: its first `len` bytes.
    Inline { len: usize },
    /// `size` bytes at the absolute offset `at`.
    Stored { at: u64, size: u64 },
}

/// One decoded IFD table: its entries, and the base its value offsets are relative to.
#[derive(Debug)]
struct Ifd {
    base: u64,
    order: ByteOrder,
    entries: Vec<(u16, Entry)>,
}

impl Ifd {
    fn entry(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|(t, _)| *t == tag).map(|(_, e)| e)
    }
}

/// Random access to one file, bounds-checked against its length.
struct Reader<R> {
    src: R,
    len: u64,
}

impl<R: Read + Seek> Reader<R> {
    fn new(mut src: R) -> Result<Reader<R>> {
        let len = src.seek(SeekFrom::End(0)).context("finding file length")?;
        Ok(Reader { src, len })
    }

    fn check_range(&self, at: u64, n: u64) -> Result<()> {
        ensure!(
            at.checked_add(n).is_some_and(|end| end <= self.len),
            "{n} bytes at offset {at} run past the end of the {}-byte file",
            self.len
        );
        Ok(())
    }

    fn read_at(&mut self, at: u64, n: u64) -> Result<Vec<u8>> {
        self.check_range(at, n)?;
        self.src.seek(SeekFrom::Start(at))?;
        let mut buf = vec![0u8; usize::try_from(n)?];
        self.src.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// The IFD at `offset` from `base`.
    fn ifd(&mut self, base: u64, order: ByteOrder, offset: u32) -> Result<Ifd> {
        let at = base
            .checked_add(u64::from(offset))
            .context("IFD offset overflows")?;
        let count = self.read_at(at, 2)?;
        let count = u64::from(order.u16([count[0], count[1]]));
        let table = self.read_at(at + 2, count * IFD_ENTRY_LEN as u64)?;
        let (rows, _) = table.as_chunks::<IFD_ENTRY_LEN>();
        let entries = rows
            .iter()
            .map(|e| {
                let entry = Entry {
                    ty: FieldType::from_code(order.u16([e[2], e[3]])),
                    count: order.u32([e[4], e[5], e[6], e[7]]),
                    field: [e[8], e[9], e[10], e[11]],
                };
                (order.u16([e[0], e[1]]), entry)
            })
            .collect();
        Ok(Ifd {
            base,
            order,
            entries,
        })
    }

    /// The first `want` bytes (fewer if the value is shorter) of `tag`'s value, if present.
    fn value_prefix(&mut self, ifd: &Ifd, tag: u16, want: u64) -> Result<Option<(Entry, Vec<u8>)>> {
        let Some(&entry) = ifd.entry(tag) else {
            return Ok(None);
        };
        let location = self
            .locate(ifd, &entry)
            .with_context(|| format!("tag {tag:#06x}"))?;
        let bytes = match location {
            Location::Inline { len } => entry.field[..len].to_vec(),
            // The whole declared value must lie in the file, though only a prefix is read.
            Location::Stored { at, size } => self
                .check_range(at, size)
                .and_then(|()| self.read_at(at, size.min(want)))
                .with_context(|| format!("tag {tag:#06x}"))?,
        };
        Ok(Some((entry, bytes)))
    }

    /// Where `entry`'s value lives.
    fn locate(&self, ifd: &Ifd, entry: &Entry) -> Result<Location> {
        let size = u64::from(entry.count) * entry.ty.unit_len()?;
        if size <= INLINE_VALUE_LEN as u64 {
            return Ok(Location::Inline { len: size as usize });
        }
        let at = ifd
            .base
            .checked_add(u64::from(ifd.order.u32(entry.field)))
            .context("value offset overflows")?;
        Ok(Location::Stored { at, size })
    }

    /// The first element of an unsigned integer field.
    fn uint(&mut self, ifd: &Ifd, tag: u16) -> Result<Option<u32>> {
        let Some((entry, b)) = self.value_prefix(ifd, tag, 4)? else {
            return Ok(None);
        };
        let o = ifd.order;
        let v = match (entry.ty, b.as_slice()) {
            (FieldType::Short, [a, b, ..]) => u32::from(o.u16([*a, *b])),
            (FieldType::Long | FieldType::Ifd, [a, b, c, d, ..]) => o.u32([*a, *b, *c, *d]),
            (FieldType::Byte, [a, ..]) => u32::from(*a),
            _ => bail!("tag {tag:#06x}: not an unsigned integer ({entry:?})"),
        };
        Ok(Some(v))
    }

    /// A RATIONAL or SRATIONAL field's first element, as the quotient.
    fn ratio(&mut self, ifd: &Ifd, tag: u16) -> Result<Option<f64>> {
        let Some((entry, b)) = self.value_prefix(ifd, tag, 8)? else {
            return Ok(None);
        };
        let o = ifd.order;
        let [n0, n1, n2, n3, d0, d1, d2, d3] = b[..] else {
            bail!("tag {tag:#06x}: not a rational ({entry:?})");
        };
        let (num, den) = (o.u32([n0, n1, n2, n3]), o.u32([d0, d1, d2, d3]));
        let (num, den) = match entry.ty {
            FieldType::Rational => (f64::from(num), f64::from(den)),
            FieldType::SRational => (f64::from(num as i32), f64::from(den as i32)),
            _ => bail!("tag {tag:#06x}: not a rational ({entry:?})"),
        };
        ensure!(den != 0.0, "tag {tag:#06x}: rational with zero denominator");
        Ok(Some(num / den))
    }

    /// An ASCII field up to its first NUL.
    fn ascii(&mut self, ifd: &Ifd, tag: u16) -> Result<Option<String>> {
        let Some((entry, b)) = self.value_prefix(ifd, tag, ASCII_LEN_MAX)? else {
            return Ok(None);
        };
        ensure!(
            entry.ty == FieldType::Ascii,
            "tag {tag:#06x}: not ASCII ({entry:?})"
        );
        let text = b.split(|&c| c == 0).next().unwrap_or_default();
        let text = std::str::from_utf8(text).with_context(|| format!("tag {tag:#06x}"))?;
        Ok(Some(text.trim().to_owned()))
    }

    /// The first `want` bytes of an UNDEFINED field.
    fn blob(&mut self, ifd: &Ifd, tag: u16, want: u64) -> Result<Option<Vec<u8>>> {
        let Some((entry, b)) = self.value_prefix(ifd, tag, want)? else {
            return Ok(None);
        };
        ensure!(
            entry.ty == FieldType::Undefined,
            "tag {tag:#06x}: not UNDEFINED ({entry:?})"
        );
        Ok(Some(b))
    }

    fn header_ifd(&mut self) -> Result<Ifd> {
        let (order, first) = parse_tiff_header(&self.read_at(0, TIFF_HEADER_LEN)?)?;
        self.ifd(0, order, first)
    }
}

/// A tag the shot cannot be culled without.
fn required<T>(v: Option<T>, what: &str) -> Result<T> {
    v.with_context(|| format!("no {what}"))
}

/// Reads every `*.NEF` (any case) directly in `dir` (not recursive), in capture order; equal
/// timestamps (never within one body) fall back to file name. Files that cannot be described
/// well enough to cull are logged and left out.
pub fn scan(dir: &Path) -> Result<Vec<Shot>> {
    let paths = nef_paths(dir)?;
    let total = paths.len();
    let mut shots: Vec<Shot> = paths
        .into_iter()
        .filter_map(|path| {
            read_shot(&path)
                .inspect_err(|e| warn!(file = %path.display(), "skipping: {e:#}"))
                .ok()
        })
        .collect();
    if shots.len() < total {
        warn!("scan: {} of {total} files skipped", total - shots.len());
    }
    in_capture_order(&mut shots);
    Ok(shots)
}

fn in_capture_order(shots: &mut [Shot]) {
    shots.sort_by(|a, b| {
        a.taken_at
            .cmp(&b.taken_at)
            .then_with(|| a.file.cmp(&b.file))
    });
}

fn nef_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let is_nef = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("NEF"));
        // `is_file` follows symlinks: a folder of links to the originals is a supported layout.
        if is_nef && path.is_file() {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn read_shot(path: &Path) -> Result<Shot> {
    let file = File::open(path).context("opening")?;
    describe(&mut Reader::new(file)?, path)
}

/// The `Shot` for one NEF. `file` names it for the log only.
fn describe<R: Read + Seek>(r: &mut Reader<R>, file: &Path) -> Result<Shot> {
    let ifd0 = r.header_ifd().context("IFD0")?;
    let orientation = r.uint(&ifd0, TAG_ORIENTATION)?.unwrap_or(1);
    let preview = preview_span(r, &ifd0)?;
    let exif_at = required(r.uint(&ifd0, TAG_EXIF_IFD)?, "Exif IFD")?;
    let exif = r.ifd(0, ifd0.order, exif_at).context("Exif IFD")?;
    let maker_note = maker_note(r, &exif, file);

    let taken = required(r.ascii(&exif, TAG_DATE_TIME_ORIGINAL)?, "DateTimeOriginal")?;
    let subsec = r.ascii(&exif, TAG_SUB_SEC_TIME_ORIGINAL)?;
    let offset = required(
        r.ascii(&exif, TAG_OFFSET_TIME_ORIGINAL)?,
        "OffsetTimeOriginal",
    )?;
    Ok(Shot {
        file: file.to_owned(),
        file_size: r.len,
        taken_at: capture_time(&taken, subsec.as_deref(), &offset)?,
        preview,
        af: maker_note.af,
        orientation: Orientation::from_exif(u8::try_from(orientation)?)?,
        exposure: Exposure {
            shutter_s: required(r.ratio(&exif, TAG_EXPOSURE_TIME)?, "ExposureTime")?,
            focal_mm: required(r.ratio(&exif, TAG_FOCAL_LENGTH)?, "FocalLength")?,
            iso: required(r.uint(&exif, TAG_ISO)?, "ISO")?,
            picture_control: maker_note.picture_control,
            compensation_ev: r.ratio(&exif, TAG_EXPOSURE_BIAS)?,
        },
    })
}

/// The full-size JPEG: SubIFD 0's JPEGInterchangeFormat. (IFD0 holds a thumbnail and later
/// SubIFDs smaller previews.)
fn preview_span<R: Read + Seek>(r: &mut Reader<R>, ifd0: &Ifd) -> Result<PreviewSpan> {
    let sub = required(r.uint(ifd0, TAG_SUB_IFDS)?, "SubIFD 0")?;
    let sub = r.ifd(0, ifd0.order, sub).context("SubIFD 0")?;
    let span = PreviewSpan {
        offset: u64::from(required(
            r.uint(&sub, TAG_PREVIEW_OFFSET)?,
            "JpgFromRaw offset",
        )?),
        len: u64::from(required(
            r.uint(&sub, TAG_PREVIEW_LEN)?,
            "JpgFromRaw length",
        )?),
    };
    r.check_range(span.offset, span.len)
        .context("JpgFromRaw span")?;
    Ok(span)
}

/// `2026:10:18 14:49:32` with optional sub-second digits and a `+00:00` UTC offset, as
/// DateTimeOriginal, SubSecTimeOriginal and OffsetTimeOriginal carry them. A time with no
/// offset is refused rather than guessed.
fn capture_time(
    date_time: &str,
    subsec: Option<&str>,
    offset: &str,
) -> Result<DateTime<FixedOffset>> {
    let frac = subsec
        .filter(|s| !s.is_empty())
        .map_or(String::new(), |s| format!(".{s}"));
    let s = format!("{date_time}{frac}{offset}");
    DateTime::parse_from_str(&s, "%Y:%m:%d %H:%M:%S%.f%:z")
        .with_context(|| format!("unparseable capture time {s:?}"))
}

/// What the Nikon MakerNote adds. Everything in it is optional to the cull: a field is `None`
/// when the camera did not record it, its layout is not one this reader knows, or the
/// MakerNote or the field is damaged (each of the last two is logged with its reason).
#[derive(Debug, Default)]
struct MakerNote {
    af: Option<AfBox>,
    picture_control: Option<String>,
}

fn maker_note<R: Read + Seek>(r: &mut Reader<R>, exif: &Ifd, file: &Path) -> MakerNote {
    let ifd = match nikon_maker_note_ifd(r, exif) {
        Ok(ifd) => ifd,
        Err(e) => {
            warn!(file = %file.display(), "AF box and Picture Control unknown: {e:#}");
            return MakerNote::default();
        }
    };
    MakerNote {
        af: or_unknown(file, "AF box", af_box_of(r, &ifd)),
        picture_control: or_unknown(file, "Picture Control", picture_control_of(r, &ifd)),
    }
}

/// `what` of the shot, or `None` with a warning naming `file` and the reason when it cannot
/// be read.
fn or_unknown<T>(file: &Path, what: &str, field: Result<Option<T>>) -> Option<T> {
    field
        .inspect_err(|e| warn!(file = %file.display(), "{what} unknown: {e:#}"))
        .ok()
        .flatten()
}

/// The IFD inside the Nikon MakerNote; an error when there is no MakerNote or it is not a
/// readable Nikon one.
fn nikon_maker_note_ifd<R: Read + Seek>(r: &mut Reader<R>, exif: &Ifd) -> Result<Ifd> {
    let &entry = exif.entry(TAG_MAKER_NOTE).context("no MakerNote")?;
    let Location::Stored { at, size } = r.locate(exif, &entry).context("MakerNote")? else {
        bail!("MakerNote too short (fits in its entry)");
    };
    ensure!(
        entry.ty == FieldType::Undefined,
        "MakerNote is not UNDEFINED"
    );
    ensure!(
        size >= NIKON_MAKER_NOTE_TIFF_AT + TIFF_HEADER_LEN,
        "MakerNote too short ({size} bytes)"
    );
    let head = r.read_at(at, NIKON_MAKER_NOTE_TIFF_AT + TIFF_HEADER_LEN)?;
    ensure!(
        head.starts_with(NIKON_MAKER_NOTE_PREFIX),
        "MakerNote is not Nikon's"
    );
    let (order, first) = parse_tiff_header(&head[NIKON_MAKER_NOTE_TIFF_AT as usize..])
        .context("MakerNote TIFF header")?;
    r.ifd(at + NIKON_MAKER_NOTE_TIFF_AT, order, first)
        .context("MakerNote IFD")
}

fn af_box_of<R: Read + Seek>(r: &mut Reader<R>, maker_note: &Ifd) -> Result<Option<AfBox>> {
    r.blob(maker_note, TAG_AF_INFO2, AF_INFO2_V0400_LEN as u64)?
        .map_or(Ok(None), |b| af_box(&b, maker_note.order))
}

fn picture_control_of<R: Read + Seek>(
    r: &mut Reader<R>,
    maker_note: &Ifd,
) -> Result<Option<String>> {
    r.blob(
        maker_note,
        TAG_PICTURE_CONTROL,
        PICTURE_CONTROL_V0310_LEN as u64,
    )?
    .map(|b| picture_control_name(&b))
    .transpose()
}

/// AFInfo2 layouts this reader can decode. Any other version is refused, since the field
/// offsets differ between versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfInfo2Version {
    V0400,
}

impl AfInfo2Version {
    fn parse(ascii: &[u8]) -> Option<AfInfo2Version> {
        match ascii {
            b"0400" => Some(AfInfo2Version::V0400),
            _ => None,
        }
    }
}

/// Version 0400 (Z8, 80 bytes): the last field read, AFAreaHeight, ends at 0x4a.
const AF_INFO2_V0400_LEN: usize = 0x4a;

/// Whether the camera says the AF coordinates in AFInfo2 are valid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfCoordinates {
    Available,
    Unavailable,
}

impl AfCoordinates {
    fn from_byte(b: u8) -> AfCoordinates {
        if b == 1 {
            AfCoordinates::Available
        } else {
            AfCoordinates::Unavailable
        }
    }
}

/// The AF box only when the camera says its coordinates are valid. Nikon reports it in
/// `AFImageWidth×AFImageHeight` space, which on the Z8 equals the JpgFromRaw size; the
/// preview decoder rescales if a body ever reports a different space.
fn af_box(data: &[u8], order: ByteOrder) -> Result<Option<AfBox>> {
    let version = data
        .get(..4)
        .and_then(AfInfo2Version::parse)
        .ok_or_else(|| unsupported_version("AFInfo2", data))?;
    match version {
        AfInfo2Version::V0400 => {
            ensure!(
                data.len() >= AF_INFO2_V0400_LEN,
                "AFInfo2 0400 has {} bytes, needs {AF_INFO2_V0400_LEN}",
                data.len()
            );
            let u16_at = |at| u32::from(order.u16_at(data, at).expect("length checked above"));
            Ok(centred_box(
                AfCoordinates::from_byte(data[0x07]),
                (u16_at(0x3e), u16_at(0x40)),
                (u16_at(0x42), u16_at(0x44)),
                (u16_at(0x46), u16_at(0x48)),
            ))
        }
    }
}

/// The box of `size` centred on `centre`, in a `space`-sized frame; `None` unless the camera
/// reported coordinates and every dimension is nonzero.
fn centred_box(
    available: AfCoordinates,
    space: (u32, u32),
    centre: (u32, u32),
    size: (u32, u32),
) -> Option<AfBox> {
    let ((iw, ih), (cx, cy), (w, h)) = (space, centre, size);
    if available != AfCoordinates::Available || w == 0 || h == 0 || iw == 0 || ih == 0 {
        return None;
    }
    Some(AfBox {
        rect: PixelRect {
            x: cx.saturating_sub(w / 2),
            y: cy.saturating_sub(h / 2),
            w,
            h,
        },
        space_w: iw,
        space_h: ih,
    })
}

/// The error for a versioned MakerNote structure whose version this reader does not decode.
fn unsupported_version(what: &str, data: &[u8]) -> anyhow::Error {
    anyhow::anyhow!(
        "unsupported {what} version {:?}",
        String::from_utf8_lossy(data.get(..4).unwrap_or(data))
    )
}

/// PictureControlData layouts this reader can decode; any other version is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PictureControlVersion {
    V0310,
}

impl PictureControlVersion {
    fn parse(ascii: &[u8]) -> Option<PictureControlVersion> {
        match ascii {
            b"0310" => Some(PictureControlVersion::V0310),
            _ => None,
        }
    }
}

/// Version 0310 (Z8): the name is 20 bytes at 0x08, NUL padded; the whole name field ends at
/// 0x1c.
const PICTURE_CONTROL_V0310_LEN: usize = 0x1c;

fn picture_control_name(data: &[u8]) -> Result<String> {
    let version = data
        .get(..4)
        .and_then(PictureControlVersion::parse)
        .ok_or_else(|| unsupported_version("PictureControlData", data))?;
    match version {
        PictureControlVersion::V0310 => {
            let field = data
                .get(8..PICTURE_CONTROL_V0310_LEN)
                .with_context(|| format!("PictureControlData 0310 has {} bytes", data.len()))?;
            let name = field.split(|&c| c == 0).next().unwrap_or_default();
            Ok(std::str::from_utf8(name)
                .context("PictureControlName is not UTF-8")?
                .to_owned())
        }
    }
}

#[cfg(test)]
mod tests;
