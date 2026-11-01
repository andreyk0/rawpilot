//! A neutral automatic development of the raw file, for judging.
//!
//! The camera's JPEG shows one rendering of the frame: its exposure, its Picture Control tone
//! curve, its clipping. A judge trained on finished photographs rates that rendering, so it
//! prefers a bright frame whose JPEG looks good over a darker one that would edit better. This
//! renders every frame the way a basic edit would — exposure normalised to the scene, raw
//! highlights compressed rather than clipped, mild contrast and saturation — so frames are
//! compared on what they can become. It also measures what no edit can fix: the share of the
//! frame clipped in the raw itself.
//!
//! Decoding is libraw's `dcraw_emu` at half size (2×2 binning, no demosaic), ~1.1 s per 45 MP
//! frame.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};

use anyhow::{Context, Result, anyhow, ensure};
use image::codecs::pnm::PnmDecoder;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

use crate::config::DevelopConfig;
use crate::shot::PixelRect;
use crate::tone;

/// Mid grey in linear light: the scene's log-average luminance is pushed here.
const MID_GREY: f32 = 0.18;
/// Linear level above which values roll off towards white instead of clipping. Lower than
/// `exposure`'s because a raw push is larger and has more highlight range to fit.
const SHOULDER_START: f64 = 0.5;
/// A raw channel at or above this (of full scale, after `-H 0` clipping) is clipped.
const RAW_CLIPPED_MIN: f32 = 0.995;
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DevelopInfo {
    /// Exposure change applied to reach a mid-grey key, in EV.
    pub push_ev: f32,
    /// Fraction of the frame (0–1) clipped in the raw: highlight detail no edit can recover.
    pub raw_clipped: f32,
}

/// 16-bit linear-light RGB, full scale 65535.
pub type LinearU16 = ImageBuffer<Rgb<u16>, Vec<u16>>;

/// Linear-light RGB in [0, 1], row-major.
pub struct LinearF32 {
    pub width: u32,
    pub height: u32,
    pub px: Vec<[f32; 3]>,
}

/// Decodes `nef` at half size and box-averages it by the largest whole factor that keeps the
/// long edge at least `long_edge` pixels (1035 px for a Z8 at 1024), so the judge gets no
/// fewer pixels than the camera thumbnail gives it. dcraw_emu applies the EXIF orientation
/// itself.
pub fn decode(nef: &Path, long_edge: u32) -> Result<(LinearF32, f32)> {
    let out = Command::new("dcraw_emu")
        // -h half size, -4 linear 16-bit, -W no auto-brighten, -w camera white balance,
        // -o 1 sRGB primaries, -H 0 clip highlights, -Z - to stdout.
        .args(["-h", "-4", "-W", "-w", "-o", "1", "-H", "0", "-Z", "-"])
        .arg(nef)
        .output()
        .context("running dcraw_emu (is libraw installed?)")?;
    ensure!(
        out.status.success(),
        "dcraw_emu failed on {} ({}): {}",
        nef.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let img = image::load_from_memory_with_format(&out.stdout, ImageFormat::Pnm)
        .with_context(|| format!("parsing dcraw_emu output for {}", nef.display()))?
        .into_rgb16();
    let (w, h) = img.dimensions();
    let factor = (w.max(h) / long_edge).max(1);
    let (ow, oh) = (w / factor, h / factor);
    ensure!(
        ow > 0 && oh > 0,
        "{}: decoded image too small",
        nef.display()
    );
    let mut px = vec![[0f32; 3]; (ow * oh) as usize];
    let mut clipped = 0u64;
    let norm = 1.0 / (f32::from(u16::MAX) * (factor * factor) as f32);
    for y in 0..oh * factor {
        for x in 0..ow * factor {
            let p = img.get_pixel(x, y).0;
            if p.iter()
                .any(|&c| f32::from(c) / f32::from(u16::MAX) >= RAW_CLIPPED_MIN)
            {
                clipped += 1;
            }
            let o = &mut px[((y / factor) * ow + x / factor) as usize];
            for c in 0..3 {
                o[c] += f32::from(p[c]) * norm;
            }
        }
    }
    let counted = u64::from(ow * factor) * u64::from(oh * factor);
    Ok((
        LinearF32 {
            width: ow,
            height: oh,
            px,
        },
        clipped as f32 / counted as f32,
    ))
}

/// Size of the full raw output in the stored (sensor) orientation, from libraw's
/// `raw-identify -s`, which prints `make model width height` and reads no pixels.
pub fn stored_size(nef: &Path) -> Result<(u32, u32)> {
    let out = Command::new("raw-identify")
        .arg("-s")
        .arg(nef)
        .output()
        .context("running raw-identify (is libraw installed?)")?;
    ensure!(
        out.status.success(),
        "raw-identify failed on {} ({}): {}",
        nef.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    parse_identify_size(&text)
        .with_context(|| format!("raw-identify printed {text:?} for {}", nef.display()))
}

/// The size on a `make model width height` line; the model may contain spaces, so the two
/// numbers are taken from the end.
fn parse_identify_size(text: &str) -> Result<(u32, u32)> {
    let mut words = text.split_whitespace().rev();
    let h = words.next().context("no height")?.parse()?;
    let w = words.next().context("no width")?.parse()?;
    ensure!(w > 0 && h > 0, "empty size {w}×{h}");
    Ok((w, h))
}

/// The `stored` rectangle (stored-orientation pixel coordinates of the full raw output) at full
/// resolution, as 16-bit linear-light RGB (full scale 65535) with camera white balance and
/// sRGB primaries. dcraw_emu applies the EXIF orientation to its output, so the result is
/// upright: a 90° orientation returns `stored.h × stored.w` pixels. libraw unpacks the whole
/// raw whatever the box, so one call should cover everything wanted from a frame. A box
/// running past the image is clipped silently; the caller checks the returned size.
///
/// The output is parsed as it streams out of dcraw_emu straight into the 16-bit image, so a
/// large box costs 6 bytes a pixel once, not the PPM bytes and a converted copy besides.
pub fn decode_crop(nef: &Path, stored: PixelRect) -> Result<LinearU16> {
    let mut child = Command::new("dcraw_emu")
        // -4 linear 16-bit, -W no auto-brighten, -w camera white balance, -o 1 sRGB
        // primaries, -B crop box, -Z - to stdout.
        .args(["-4", "-W", "-w", "-o", "1", "-B"])
        .args([stored.x, stored.y, stored.w, stored.h].map(|v| v.to_string()))
        .args(["-Z", "-"])
        .arg(nef)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("running dcraw_emu (is libraw installed?)")?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let mut stderr = child.stderr.take().expect("stderr is piped");
    // Drained on its own thread so that a chatty child cannot block on a full pipe while its
    // output is being read.
    let messages = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    // The reader is dropped when parsing ends, so a child still writing after a bad header
    // gets a broken pipe instead of waiting.
    let parsed = read_linear_u16(BufReader::new(stdout));
    let status = child.wait().context("waiting for dcraw_emu")?;
    let messages = messages.join().expect("stderr reader does not panic");
    crop_outcome(parsed, status, &messages, nef, stored)
}

/// What a `dcraw_emu` run came to, from its parsed output and its exit status. A failed run
/// is reported by its status and messages; when its output did not parse either, that is
/// said too, because dropping the reader makes a child still writing die on a broken pipe,
/// and the status then hides the parse error that caused it.
fn crop_outcome(
    parsed: Result<LinearU16>,
    status: ExitStatus,
    messages: &str,
    nef: &Path,
    stored: PixelRect,
) -> Result<LinearU16> {
    // Exit status 2 is both "cannot unpack" and "bad crop box".
    if status.success() {
        return parsed.with_context(|| format!("parsing dcraw_emu output for {}", nef.display()));
    }
    let failed = anyhow!(
        "dcraw_emu failed on {} for box {stored:?} ({status}): {messages}",
        nef.display(),
    );
    Err(match parsed {
        Ok(_) => failed,
        Err(parse) => failed.context(format!("its output did not parse either: {parse:#}")),
    })
}

/// A PNM stream as 16-bit RGB, without an intermediate copy of its bytes.
pub fn read_linear_u16(reader: impl BufRead) -> Result<LinearU16> {
    let decoder = PnmDecoder::new(reader)?;
    Ok(DynamicImage::from_decoder(decoder)?.into_rgb16())
}

/// The rendering: exposure to a mid-grey key, contrast as a power around mid grey, a
/// highlight shoulder that lands the brightest raw value on white, then saturation.
pub fn render(lin: &LinearF32, raw_clipped: f32, cfg: &DevelopConfig) -> (RgbImage, DevelopInfo) {
    let key = log_average_luminance(lin);
    let push_ev = (MID_GREY / key)
        .log2()
        .clamp(cfg.push_ev_min, cfg.push_ev_max);
    let gain = push_ev.exp2();
    let contrast = |v: f32| MID_GREY * (v.max(0.0) / MID_GREY).powf(cfg.contrast);
    let white = contrast(gain);
    let mut out = RgbImage::new(lin.width, lin.height);
    for (o, p) in out.pixels_mut().zip(&lin.px) {
        let toned = p.map(|c| {
            tone::shoulder(
                f64::from(contrast(c * gain)),
                f64::from(white),
                SHOULDER_START,
            ) as f32
        });
        let y = luma(toned);
        o.0 = toned.map(|c| {
            let saturated = (y + cfg.saturation * (c - y)).clamp(0.0, 1.0);
            (tone::linear_to_srgb(f64::from(saturated)) * 255.0).round() as u8
        });
    }
    (
        out,
        DevelopInfo {
            push_ev,
            raw_clipped,
        },
    )
}

fn luma(p: [f32; 3]) -> f32 {
    p.iter().zip(LUMA).map(|(c, w)| c * w).sum()
}

/// Geometric mean of luminance, the usual "key" of a scene.
fn log_average_luminance(lin: &LinearF32) -> f32 {
    let sum: f64 = lin
        .px
        .iter()
        .map(|&p| f64::from(luma(p).max(1e-6)).ln())
        .sum();
    (sum / lin.px.len() as f64).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DevelopConfig {
        DevelopConfig {
            push_ev_min: -1.0,
            push_ev_max: 3.0,
            contrast: 1.2,
            saturation: 1.25,
        }
    }

    fn flat(v: f32) -> LinearF32 {
        LinearF32 {
            width: 2,
            height: 2,
            px: vec![[v; 3]; 4],
        }
    }

    /// Grey at 0.045 linear needs log2(0.18 / 0.045) = +2 EV to reach mid grey; contrast and
    /// saturation leave mid grey alone, and sRGB(0.18) = 0.4614 → 118.
    #[test]
    fn grey_scene_is_pushed_to_mid_grey() {
        let (img, info) = render(&flat(0.045), 0.0, &cfg());
        assert!((info.push_ev - 2.0).abs() < 1e-4, "{}", info.push_ev);
        assert_eq!(img.get_pixel(0, 0).0, [118, 118, 118]);
    }

    /// A near-black frame (lens cap) would need +7 EV; the push stops at the configured
    /// maximum rather than amplifying noise into a grey frame.
    #[test]
    fn identify_size_comes_from_the_end_of_the_line() {
        assert_eq!(
            parse_identify_size("/x/DSC_1.NEF\tNikon\tZ 8\t8280\t5520\n").unwrap(),
            (8280, 5520)
        );
        assert!(parse_identify_size("").is_err());
        assert!(parse_identify_size("Nikon 0 5520").is_err());
    }

    fn exited(code: i32) -> ExitStatus {
        use std::os::unix::process::ExitStatusExt;
        ExitStatus::from_raw(code << 8)
    }

    fn crop_error(parsed: Result<LinearU16>, status: ExitStatus) -> String {
        let stored = PixelRect {
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        };
        let e = crop_outcome(parsed, status, "stderr text", Path::new("a.NEF"), stored);
        format!("{:#}", e.unwrap_err())
    }

    /// `crop_outcome` keeps both causes: a failed run whose output did not parse names the
    /// exit status and the parse error, and each alone names only itself.
    #[test]
    fn a_failed_run_with_unparsable_output_reports_both() {
        let bad = || read_linear_u16(&b"not a pnm"[..]);
        let both = crop_error(bad(), exited(2));
        assert!(both.contains("exit status: 2"), "{both}");
        assert!(both.contains("stderr text"), "{both}");
        assert!(both.contains("did not parse"), "{both}");

        let status_only = crop_error(Ok(LinearU16::new(1, 1)), exited(2));
        assert!(status_only.contains("exit status: 2"), "{status_only}");
        assert!(!status_only.contains("did not parse"), "{status_only}");

        let parse_only = crop_error(bad(), exited(0));
        assert!(
            parse_only.contains("parsing dcraw_emu output"),
            "{parse_only}"
        );
        assert!(!parse_only.contains("exit status"), "{parse_only}");
    }

    #[test]
    fn push_is_bounded() {
        let (_, info) = render(&flat(0.0014), 0.0, &cfg());
        assert_eq!(info.push_ev, 3.0);
    }

    /// The full-scale raw value, pushed by any amount, still lands on display white. Key
    /// here = 0.02^(3/4) = 0.0532, so the push is log2(0.18 / 0.0532) = +1.76 EV.
    #[test]
    fn raw_white_renders_white() {
        let mut lin = flat(0.02);
        lin.px[0] = [1.0; 3];
        let (img, info) = render(&lin, 0.0, &cfg());
        assert!((info.push_ev - 1.76).abs() < 0.01, "{}", info.push_ev);
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255]);
    }
}
