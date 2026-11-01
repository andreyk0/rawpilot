//! Synthetic TIFF files built here from the spec (little- and big-endian, MakerNote in either
//! order) pin the reader's decisions and its refusals; the `#[ignore]`d differential test
//! compares it with exiftool on a real folder of NEFs.

use std::io::Cursor;

use serde::Deserialize;

use super::*;

/// A field value as it is written into a synthetic IFD.
enum Val {
    Short(u16),
    Long(u32),
    LongArray(Vec<u32>),
    Ascii(String),
    Rational(u32, u32),
    SRational(i32, i32),
    Undefined(Vec<u8>),
    /// An UNDEFINED value whose declared length (`count`) far exceeds the 5 bytes written.
    UndefinedOverlong {
        count: u32,
    },
}

fn u16_bytes(o: ByteOrder, v: u16) -> [u8; 2] {
    match o {
        ByteOrder::Little => v.to_le_bytes(),
        ByteOrder::Big => v.to_be_bytes(),
    }
}

fn u32_bytes(o: ByteOrder, v: u32) -> [u8; 4] {
    match o {
        ByteOrder::Little => v.to_le_bytes(),
        ByteOrder::Big => v.to_be_bytes(),
    }
}

impl Val {
    /// TIFF type code, element count and encoded bytes.
    fn encode(&self, o: ByteOrder) -> (u16, u32, Vec<u8>) {
        match self {
            Val::Short(v) => (3, 1, u16_bytes(o, *v).to_vec()),
            Val::Long(v) => (4, 1, u32_bytes(o, *v).to_vec()),
            Val::LongArray(vs) => (
                4,
                vs.len() as u32,
                vs.iter().flat_map(|v| u32_bytes(o, *v)).collect(),
            ),
            Val::Ascii(s) => {
                let mut b = s.as_bytes().to_vec();
                b.push(0);
                (2, b.len() as u32, b)
            }
            Val::Rational(n, d) => (5, 1, [u32_bytes(o, *n), u32_bytes(o, *d)].concat()),
            Val::SRational(n, d) => (
                10,
                1,
                [u32_bytes(o, *n as u32), u32_bytes(o, *d as u32)].concat(),
            ),
            Val::Undefined(b) => (7, b.len() as u32, b.clone()),
            Val::UndefinedOverlong { count } => (7, *count, vec![0; 5]),
        }
    }
}

/// A TIFF file under construction: an 8-byte header followed by IFDs, each followed by its
/// out-of-line values.
struct Builder {
    order: ByteOrder,
    buf: Vec<u8>,
}

impl Builder {
    fn new(order: ByteOrder) -> Builder {
        let mut buf = match order {
            ByteOrder::Little => b"II*\0".to_vec(),
            ByteOrder::Big => b"MM\0*".to_vec(),
        };
        buf.extend([0; 4]);
        Builder { order, buf }
    }

    fn next_offset(&self) -> u32 {
        self.buf.len() as u32
    }

    /// Appends an IFD (with no next IFD) and returns its offset.
    fn put_ifd(&mut self, entries: &[(u16, Val)]) -> u32 {
        let o = self.order;
        let start = self.buf.len();
        let table_len = 2 + entries.len() * 12 + 4;
        let mut table = u16_bytes(o, entries.len() as u16).to_vec();
        let mut data = Vec::new();
        for (tag, val) in entries {
            let (ty, count, mut bytes) = val.encode(o);
            table.extend(u16_bytes(o, *tag));
            table.extend(u16_bytes(o, ty));
            table.extend(u32_bytes(o, count));
            if bytes.len() <= 4 {
                bytes.resize(4, 0);
                table.extend(bytes);
            } else {
                table.extend(u32_bytes(o, (start + table_len + data.len()) as u32));
                data.extend(bytes);
                if data.len() % 2 == 1 {
                    data.push(0);
                }
            }
        }
        table.extend([0; 4]);
        self.buf.extend(table);
        self.buf.extend(data);
        start as u32
    }

    fn finish(mut self, first_ifd: u32) -> Vec<u8> {
        let b = u32_bytes(self.order, first_ifd);
        self.buf[4..8].copy_from_slice(&b);
        self.buf
    }
}

/// AFInfo2 as the Z8 writes it (80 bytes), holding a box centred on (2400, 1961), 154×156,
/// in an 8256×5504 frame.
fn af_info2(version: &[u8; 4], available: u8, o: ByteOrder) -> Vec<u8> {
    let mut b = vec![0u8; 80];
    b[..4].copy_from_slice(version);
    b[0x07] = available;
    for (at, v) in [
        (0x3e, 8256u16),
        (0x40, 5504),
        (0x42, 2400),
        (0x44, 1961),
        (0x46, 154),
        (0x48, 156),
    ] {
        b[at..at + 2].copy_from_slice(&u16_bytes(o, v));
    }
    b
}

/// PictureControlData as the Z8 writes it (108 bytes): version, a repeat of it, then the name
/// and the base name in 20-byte NUL-padded fields.
fn picture_control(version: &[u8; 4], name: &str) -> Vec<u8> {
    let mut b = vec![0u8; 108];
    b[..4].copy_from_slice(version);
    b[4..8].copy_from_slice(version);
    b[8..8 + name.len()].copy_from_slice(name.as_bytes());
    b[0x1c..0x1c + name.len()].copy_from_slice(name.as_bytes());
    b
}

// Tag numbers from the TIFF 6.0 and EXIF 2.3 tables and exiftool's dumps of real Z8 files,
// written out here independently of the reader's own constants (which they shadow).
const TAG_ORIENTATION: u16 = 0x0112;
const TAG_SUB_IFDS: u16 = 0x014a;
const TAG_PREVIEW_OFFSET: u16 = 0x0201;
const TAG_PREVIEW_LEN: u16 = 0x0202;
const TAG_EXPOSURE_TIME: u16 = 0x829a;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_ISO: u16 = 0x8827;
const TAG_DATE_TIME_ORIGINAL: u16 = 0x9003;
const TAG_OFFSET_TIME_ORIGINAL: u16 = 0x9011;
const TAG_EXPOSURE_BIAS: u16 = 0x9204;
const TAG_FOCAL_LENGTH: u16 = 0x920a;
const TAG_MAKER_NOTE: u16 = 0x927c;
const TAG_SUB_SEC_TIME_ORIGINAL: u16 = 0x9291;
const TAG_PICTURE_CONTROL: u16 = 0x0023;
const TAG_AF_INFO2: u16 = 0x00b7;

/// Placed inside the synthetic file's own header region: only its bounds are ever checked.
const PREVIEW_AT: u32 = 8;
const PREVIEW_LEN: u32 = 16;

/// What a synthetic NEF carries; `Opts::z8` is a well-formed Z8 file.
struct Opts {
    order: ByteOrder,
    mn_order: ByteOrder,
    date_time: &'static str,
    subsec: Option<&'static str>,
    offset_time: Option<&'static str>,
    af: Option<Val>,
    picture_control: Option<Val>,
    maker_note_magic: &'static [u8; 6],
    maker_note: bool,
    /// Replaces the whole MakerNote entry.
    maker_note_entry: Option<Val>,
    exposure_time: Val,
    exif_ifd_is_ifd0: bool,
}

impl Opts {
    fn z8(order: ByteOrder) -> Opts {
        Opts {
            order,
            mn_order: order,
            date_time: "2026:10:18 14:49:32",
            subsec: Some("49"),
            offset_time: Some("+00:00"),
            af: Some(Val::Undefined(af_info2(b"0400", 1, order))),
            picture_control: Some(Val::Undefined(picture_control(b"0310", "VIVID"))),
            maker_note_magic: b"Nikon\0",
            maker_note: true,
            maker_note_entry: None,
            exposure_time: Val::Rational(1, 250),
            exif_ifd_is_ifd0: false,
        }
    }

    fn with_mn_order(self, mn_order: ByteOrder) -> Opts {
        Opts {
            af: self
                .af
                .map(|_| Val::Undefined(af_info2(b"0400", 1, mn_order))),
            mn_order,
            ..self
        }
    }

    fn build(self) -> Vec<u8> {
        let mut mn = Builder::new(self.mn_order);
        let mut mn_entries = Vec::new();
        if let Some(af) = self.af {
            mn_entries.push((TAG_AF_INFO2, af));
        }
        if let Some(pc) = self.picture_control {
            mn_entries.push((TAG_PICTURE_CONTROL, pc));
        }
        let first = mn.put_ifd(&mn_entries);
        let mn_bytes = [
            &self.maker_note_magic[..],
            &[2, 0x11, 0, 0],
            &mn.finish(first),
        ]
        .concat();

        let mut b = Builder::new(self.order);
        // SubIFD 0 is the full-size JPEG; SubIFD 1 and IFD0 hold decoys.
        let sub0 = b.put_ifd(&[
            (TAG_PREVIEW_OFFSET, Val::Long(PREVIEW_AT)),
            (TAG_PREVIEW_LEN, Val::Long(PREVIEW_LEN)),
        ]);
        let sub1 = b.put_ifd(&[
            (TAG_PREVIEW_OFFSET, Val::Long(2)),
            (TAG_PREVIEW_LEN, Val::Long(3)),
        ]);
        let mut exif = vec![
            (TAG_EXPOSURE_TIME, self.exposure_time),
            (TAG_EXPOSURE_BIAS, Val::SRational(-4, 3)),
            (TAG_ISO, Val::Short(80)),
            (TAG_DATE_TIME_ORIGINAL, Val::Ascii(self.date_time.into())),
            (TAG_FOCAL_LENGTH, Val::Rational(350, 10)),
        ];
        if let Some(s) = self.subsec {
            exif.push((TAG_SUB_SEC_TIME_ORIGINAL, Val::Ascii(s.into())));
        }
        if let Some(o) = self.offset_time {
            exif.push((TAG_OFFSET_TIME_ORIGINAL, Val::Ascii(o.into())));
        }
        if self.maker_note {
            let entry = self.maker_note_entry.unwrap_or(Val::Undefined(mn_bytes));
            exif.push((TAG_MAKER_NOTE, entry));
        }
        let exif = b.put_ifd(&exif);
        let ifd0_at = b.next_offset();
        let ifd0 = b.put_ifd(&[
            (TAG_ORIENTATION, Val::Short(6)),
            (TAG_PREVIEW_OFFSET, Val::Long(4)),
            (TAG_PREVIEW_LEN, Val::Long(5)),
            (TAG_SUB_IFDS, Val::LongArray(vec![sub0, sub1])),
            (
                TAG_EXIF_IFD,
                Val::Long(if self.exif_ifd_is_ifd0 { ifd0_at } else { exif }),
            ),
        ]);
        b.finish(ifd0)
    }
}

fn describe_bytes(bytes: Vec<u8>) -> Result<Shot> {
    describe(&mut Reader::new(Cursor::new(bytes))?, Path::new("t.NEF"))
}

/// Written out from the values `Opts::z8` encodes.
fn expected_z8(file_size: u64) -> Shot {
    Shot {
        file: "t.NEF".into(),
        file_size,
        taken_at: DateTime::parse_from_rfc3339("2026-10-18T14:49:32.490+00:00").unwrap(),
        preview: PreviewSpan {
            offset: u64::from(PREVIEW_AT),
            len: u64::from(PREVIEW_LEN),
        },
        af: Some(AfBox {
            rect: PixelRect {
                x: 2323,
                y: 1883,
                w: 154,
                h: 156,
            },
            space_w: 8256,
            space_h: 5504,
        }),
        orientation: Orientation::Rotate90Cw,
        exposure: Exposure {
            shutter_s: 0.004,
            focal_mm: 35.0,
            iso: 80,
            picture_control: Some("VIVID".into()),
            compensation_ev: Some(-4.0 / 3.0),
        },
    }
}

fn shot_of(opts: Opts) -> Shot {
    describe_bytes(opts.build()).unwrap()
}

#[test]
fn reads_every_field_in_either_byte_order() {
    use ByteOrder::{Big, Little};
    for (outer, mn) in [(Little, Little), (Big, Big), (Little, Big), (Big, Little)] {
        let bytes = Opts::z8(outer).with_mn_order(mn).build();
        let len = bytes.len() as u64;
        assert_eq!(
            describe_bytes(bytes).unwrap(),
            expected_z8(len),
            "{outer:?}/{mn:?}"
        );
    }
}

#[test]
fn preview_is_subifd_zero_not_a_decoy() {
    let shot = shot_of(Opts::z8(ByteOrder::Little));
    assert_eq!(shot.preview, PreviewSpan { offset: 8, len: 16 });
}

#[test]
fn capture_time_without_subsec_is_whole_seconds() {
    let shot = shot_of(Opts {
        subsec: None,
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!(shot.taken_at.to_rfc3339(), "2026-10-18T14:49:32+00:00");
}

#[test]
fn rejects_time_without_offset() {
    let e = describe_bytes(
        Opts {
            offset_time: None,
            ..Opts::z8(ByteOrder::Little)
        }
        .build(),
    )
    .unwrap_err();
    assert!(format!("{e:#}").contains("OffsetTimeOriginal"), "{e:#}");
    assert!(capture_time("2026:10:18 14:49:32", Some("49"), "").is_err());
}

#[test]
fn parses_nikon_subsec_time() {
    let t = capture_time("2026:10:18 14:49:32", Some("49"), "+02:00").unwrap();
    assert_eq!(t.to_rfc3339(), "2026-10-18T14:49:32.490+02:00");
}

#[test]
fn unknown_af_version_gives_no_box_and_keeps_the_rest() {
    let shot = shot_of(Opts {
        af: Some(Val::Undefined(af_info2(b"0500", 1, ByteOrder::Little))),
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!(shot.af, None);
    assert_eq!(shot.exposure.picture_control.as_deref(), Some("VIVID"));
}

#[test]
fn truncated_af_info2_gives_no_box() {
    let shot = shot_of(Opts {
        af: Some(Val::Undefined(
            af_info2(b"0400", 1, ByteOrder::Little)[..0x40].to_vec(),
        )),
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!(shot.af, None);
}

#[test]
fn af_box_absent_when_camera_says_unavailable() {
    let shot = shot_of(Opts {
        af: Some(Val::Undefined(af_info2(b"0400", 0, ByteOrder::Little))),
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!(shot.af, None);
}

#[test]
fn af_box_absent_when_a_dimension_is_zero() {
    let mut af = af_info2(b"0400", 1, ByteOrder::Little);
    af[0x46..0x48].fill(0);
    let shot = shot_of(Opts {
        af: Some(Val::Undefined(af)),
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!(shot.af, None);
}

#[test]
fn unknown_picture_control_version_gives_no_name_and_keeps_the_rest() {
    let shot = shot_of(Opts {
        picture_control: Some(Val::Undefined(picture_control(b"0400", "VIVID"))),
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!(shot.exposure.picture_control, None);
    assert!(shot.af.is_some());
}

#[test]
fn absent_maker_note_tags_are_absent_not_errors() {
    let shot = shot_of(Opts {
        af: None,
        picture_control: None,
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!((shot.af, shot.exposure.picture_control), (None, None));
    let shot = shot_of(Opts {
        maker_note: false,
        ..Opts::z8(ByteOrder::Little)
    });
    assert_eq!((shot.af, shot.exposure.picture_control), (None, None));
}

/// The Exif fields are all that culling needs; a MakerNote or MakerNote field that cannot be
/// read leaves the shot in, without AF box and Picture Control.
#[test]
fn a_damaged_maker_note_keeps_the_shot_without_its_maker_note_fields() {
    let damaged = [
        (
            "not Nikon's",
            Opts {
                maker_note_magic: b"Canon\0",
                ..Opts::z8(ByteOrder::Little)
            },
            (false, false),
        ),
        (
            "runs past the end of the file",
            Opts {
                maker_note_entry: Some(Val::UndefinedOverlong { count: u32::MAX }),
                ..Opts::z8(ByteOrder::Little)
            },
            (false, false),
        ),
        (
            "AFInfo2 of the wrong type",
            Opts {
                af: Some(Val::Short(1)),
                ..Opts::z8(ByteOrder::Little)
            },
            (false, true),
        ),
        (
            "AFInfo2 out of range",
            Opts {
                af: Some(Val::UndefinedOverlong { count: u32::MAX }),
                ..Opts::z8(ByteOrder::Little)
            },
            (false, true),
        ),
        (
            "PictureControlData of the wrong type",
            Opts {
                picture_control: Some(Val::Short(1)),
                ..Opts::z8(ByteOrder::Little)
            },
            (true, false),
        ),
        (
            "PictureControlData out of range",
            Opts {
                picture_control: Some(Val::UndefinedOverlong { count: u32::MAX }),
                ..Opts::z8(ByteOrder::Little)
            },
            (true, false),
        ),
    ];
    for (what, opts, (has_af, has_picture_control)) in damaged {
        let shot = describe_bytes(opts.build()).unwrap_or_else(|e| panic!("{what}: {e:#}"));
        let want = expected_z8(shot.file_size);
        assert_eq!(shot.af.is_some(), has_af, "{what}: af");
        assert_eq!(
            shot.exposure.picture_control.is_some(),
            has_picture_control,
            "{what}: picture control"
        );
        assert_eq!(shot.af, has_af.then_some(want.af).flatten(), "{what}");
        assert_eq!(shot.taken_at, want.taken_at, "{what}");
        assert_eq!(shot.preview, want.preview, "{what}");
        assert_eq!(shot.exposure.iso, want.exposure.iso, "{what}");
    }
}

#[test]
fn rejects_zero_denominator() {
    let r = describe_bytes(
        Opts {
            exposure_time: Val::Rational(1, 0),
            ..Opts::z8(ByteOrder::Little)
        }
        .build(),
    );
    assert!(r.is_err());
}

#[test]
fn exif_ifd_pointing_back_at_ifd0_terminates_with_an_error() {
    let e = describe_bytes(
        Opts {
            exif_ifd_is_ifd0: true,
            ..Opts::z8(ByteOrder::Little)
        }
        .build(),
    )
    .unwrap_err();
    assert!(format!("{e:#}").contains("DateTimeOriginal"), "{e:#}");
}

/// Offset of the first IFD entry's value field after the header, in a little-endian file
/// whose first IFD is `Opts::build`'s SubIFD 0 (entry 0 is the preview offset).
const SUB0_PREVIEW_OFFSET_FIELD: usize = 8 + 2 + 8;

#[test]
fn rejects_preview_past_end_of_file() {
    let mut bytes = Opts::z8(ByteOrder::Little).build();
    let past = (bytes.len() as u32 - PREVIEW_LEN + 1).to_le_bytes();
    bytes[SUB0_PREVIEW_OFFSET_FIELD..SUB0_PREVIEW_OFFSET_FIELD + 4].copy_from_slice(&past);
    let e = describe_bytes(bytes).unwrap_err();
    assert!(format!("{e:#}").contains("past the end"), "{e:#}");
}

#[test]
fn rejects_preview_span_that_overflows() {
    let mut bytes = Opts::z8(ByteOrder::Little).build();
    bytes[SUB0_PREVIEW_OFFSET_FIELD..SUB0_PREVIEW_OFFSET_FIELD + 4]
        .copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(describe_bytes(bytes).is_err());
}

#[test]
fn rejects_overlong_value_counts_and_offsets() {
    let good = Opts::z8(ByteOrder::Little).build();
    let ifd0 = Reader::new(Cursor::new(good.clone()))
        .unwrap()
        .header_ifd()
        .unwrap();
    let first_ifd = u32::from_le_bytes(good[4..8].try_into().unwrap()) as usize;
    let sub_ifds_entry = first_ifd
        + 2
        + 12 * ifd0
            .entries
            .iter()
            .position(|(t, _)| *t == TAG_SUB_IFDS)
            .unwrap();

    let mut bad_offset = good.clone();
    bad_offset[sub_ifds_entry + 8..sub_ifds_entry + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    let e = describe_bytes(bad_offset).unwrap_err();
    assert!(format!("{e:#}").contains("past the end"), "{e:#}");

    // An element count whose byte length (count × 4) exceeds the file.
    let mut bad_count = good;
    bad_count[sub_ifds_entry + 4..sub_ifds_entry + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    let e = describe_bytes(bad_count).unwrap_err();
    assert!(format!("{e:#}").contains("past the end"), "{e:#}");
}

/// Whatever the damage, the reader answers `Ok` or `Err`: no panic, no huge allocation, no
/// hang. Every truncation, and every byte overwritten with 0x00, 0xFF and 0x80.
#[test]
fn damaged_files_never_panic() {
    for order in [ByteOrder::Little, ByteOrder::Big] {
        let good = Opts::z8(order).build();
        for n in 0..good.len() {
            let _ = describe_bytes(good[..n].to_vec());
        }
        for at in 0..good.len() {
            for v in [0x00, 0xff, 0x80] {
                let mut bad = good.clone();
                bad[at] = v;
                let _ = describe_bytes(bad);
            }
        }
    }
}

#[test]
fn rejects_non_tiff_input() {
    assert!(describe_bytes(b"\xff\xd8\xff\xe0not a tiff at all".to_vec()).is_err());
    assert!(describe_bytes(Vec::new()).is_err());
}

#[test]
fn shots_are_ordered_by_capture_time_then_file_name() {
    let at = |name: &str, t: &str| Shot {
        file: name.into(),
        taken_at: DateTime::parse_from_rfc3339(t).unwrap(),
        ..expected_z8(0)
    };
    let mut shots = vec![
        at("d.NEF", "2026-10-18T14:49:33+00:00"),
        at("c.NEF", "2026-10-18T14:49:32.500+00:00"),
        at("b.NEF", "2026-10-18T14:49:32.500+00:00"),
        at("a.NEF", "2026-10-18T14:49:34+00:00"),
    ];
    in_capture_order(&mut shots);
    let names: Vec<String> = shots.iter().map(Shot::name).collect();
    assert_eq!(names, ["b.NEF", "c.NEF", "d.NEF", "a.NEF"]);
}

#[test]
fn scan_takes_nef_files_in_any_case_in_capture_order_and_skips_broken_ones() {
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, bytes: Vec<u8>| std::fs::write(dir.path().join(name), bytes).unwrap();
    // File-name order is the reverse of capture order.
    write(
        "a.nef",
        Opts {
            date_time: "2026:10:18 15:00:00",
            ..Opts::z8(ByteOrder::Little)
        }
        .build(),
    );
    write("b.NEF", Opts::z8(ByteOrder::Big).build());
    write("broken.NEF", b"II*\0garbage".to_vec());
    write("notes.txt", Opts::z8(ByteOrder::Little).build());
    std::fs::create_dir(dir.path().join("sub.NEF")).unwrap();
    write("sub.NEF/inner.NEF", Opts::z8(ByteOrder::Little).build());

    let names: Vec<String> = scan(dir.path()).unwrap().iter().map(Shot::name).collect();
    assert_eq!(names, ["b.NEF", "a.nef"]);
}

/// The reader against exiftool, on a real folder: `RAWPILOT_NEF_DIR=<dir> cargo test --
/// --ignored differential --nocapture`. Everything must be equal except floats, which
/// exiftool prints to 10 significant digits.
#[test]
#[ignore = "needs exiftool and RAWPILOT_NEF_DIR"]
fn differential_against_exiftool() {
    let dir = PathBuf::from(std::env::var_os("RAWPILOT_NEF_DIR").expect("RAWPILOT_NEF_DIR"));
    let t = std::time::Instant::now();
    let ours = scan(&dir).unwrap();
    let ours_s = t.elapsed();
    let t = std::time::Instant::now();
    let theirs = exiftool_shots(&dir);
    let theirs_s = t.elapsed();

    let mut mismatches = Vec::new();
    let mut agree = 0;
    for (file, expected) in &theirs {
        match (ours.iter().find(|s| &s.file == file), expected) {
            (Some(got), Ok(want)) if shots_agree(got, want) => agree += 1,
            (None, Err(_)) => agree += 1,
            (got, want) => mismatches.push(format!("{}: {got:?} vs {want:?}", file.display())),
        }
    }
    let theirs_ok = theirs.iter().filter(|(_, r)| r.is_ok()).count();
    if ours.len() != theirs_ok {
        mismatches.push(format!("{} shots vs {theirs_ok} from exiftool", ours.len()));
    }
    println!(
        "files {}  ours {}  agree {agree}  mismatches {}  scan {ours_s:?}  exiftool {theirs_s:?}",
        theirs.len(),
        ours.len(),
        mismatches.len()
    );
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
}

fn shots_agree(a: &Shot, b: &Shot) -> bool {
    let (ea, eb) = (&a.exposure, &b.exposure);
    a.file == b.file
        && a.file_size == b.file_size
        && a.taken_at == b.taken_at
        && a.preview == b.preview
        && a.af == b.af
        && a.orientation == b.orientation
        && close(ea.shutter_s, eb.shutter_s)
        && close(ea.focal_mm, eb.focal_mm)
        && ea.iso == eb.iso
        && ea.picture_control == eb.picture_control
        && match (ea.compensation_ev, eb.compensation_ev) {
            (Some(x), Some(y)) => close(x, y),
            (None, None) => true,
            _ => false,
        }
}

/// `exiftool -j -n` for every NEF in `dir`: the oracle. A file exiftool cannot fully describe
/// is an `Err`, as it is skipped by the reader.
fn exiftool_shots(dir: &Path) -> Vec<(PathBuf, Result<Shot>)> {
    let out = std::process::Command::new("exiftool")
        .args(["-j", "-n", "-q", "-q", "-ext", "NEF"])
        .args([
            "-FileSize",
            "-SubSecDateTimeOriginal",
            "-JpgFromRawStart",
            "-JpgFromRawLength",
            "-AFCoordinatesAvailable",
            "-AFImageWidth",
            "-AFImageHeight",
            "-AFAreaXPosition",
            "-AFAreaYPosition",
            "-AFAreaWidth",
            "-AFAreaHeight",
            "-ExposureTime",
            "-FocalLength",
            "-ISO",
            "-PictureControlName",
            "-ExposureCompensation",
            "-Orientation",
        ])
        .arg(dir)
        .output()
        .unwrap();
    let records: Vec<ExiftoolRecord> = serde_json::from_slice(&out.stdout).unwrap();
    records
        .into_iter()
        .map(|r| (r.source_file.clone(), r.into_shot()))
        .collect()
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct ExiftoolRecord {
    source_file: PathBuf,
    file_size: Option<u64>,
    sub_sec_date_time_original: Option<String>,
    jpg_from_raw_start: Option<u64>,
    jpg_from_raw_length: Option<u64>,
    #[serde(rename = "AFCoordinatesAvailable")]
    af_available: Option<u8>,
    #[serde(rename = "AFImageWidth")]
    af_image_width: Option<u32>,
    #[serde(rename = "AFImageHeight")]
    af_image_height: Option<u32>,
    #[serde(rename = "AFAreaXPosition")]
    af_x: Option<u32>,
    #[serde(rename = "AFAreaYPosition")]
    af_y: Option<u32>,
    #[serde(rename = "AFAreaWidth")]
    af_w: Option<u32>,
    #[serde(rename = "AFAreaHeight")]
    af_h: Option<u32>,
    exposure_time: Option<f64>,
    focal_length: Option<f64>,
    #[serde(rename = "ISO")]
    iso: Option<u32>,
    picture_control_name: Option<String>,
    exposure_compensation: Option<f64>,
    orientation: Option<u8>,
}

impl ExiftoolRecord {
    fn into_shot(self) -> Result<Shot> {
        let af = match (
            self.af_available,
            self.af_image_width,
            self.af_image_height,
            self.af_x,
            self.af_y,
            self.af_w,
            self.af_h,
        ) {
            (Some(1), Some(iw), Some(ih), Some(x), Some(y), Some(w), Some(h))
                if w != 0 && h != 0 && iw != 0 && ih != 0 =>
            {
                Some(AfBox {
                    rect: PixelRect {
                        x: x.saturating_sub(w / 2),
                        y: y.saturating_sub(h / 2),
                        w,
                        h,
                    },
                    space_w: iw,
                    space_h: ih,
                })
            }
            _ => None,
        };
        let taken = self.sub_sec_date_time_original.context("no time")?;
        Ok(Shot {
            file: self.source_file,
            file_size: self.file_size.context("no size")?,
            taken_at: DateTime::parse_from_str(&taken, "%Y:%m:%d %H:%M:%S%.f%:z")?,
            preview: PreviewSpan {
                offset: self.jpg_from_raw_start.context("no preview")?,
                len: self.jpg_from_raw_length.context("no preview")?,
            },
            af,
            orientation: Orientation::from_exif(self.orientation.unwrap_or(1))?,
            exposure: Exposure {
                shutter_s: self.exposure_time.context("no shutter")?,
                focal_mm: self.focal_length.context("no focal")?,
                iso: self.iso.context("no iso")?,
                picture_control: self.picture_control_name,
                compensation_ev: self.exposure_compensation,
            },
        })
    }
}
