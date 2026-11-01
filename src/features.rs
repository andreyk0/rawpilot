//! Per-frame measurements taken from the full-resolution preview. All of them are only
//! meaningful relative to other frames of the same burst: absolute Laplacian variance depends
//! on ISO, subject and Picture Control sharpening.

use image::imageops::{self, FilterType};
use image::{GrayImage, RgbImage};
use serde::{Deserialize, Serialize};

use crate::config::FeaturesConfig;
use crate::shot::{Orientation, PixelRect, Shot};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AfSharpness {
    /// The camera recorded no AF coordinates for this frame.
    NoBox,
    /// The AF window has too little structure for its sharpness to mean anything.
    Featureless {
        texture: f64,
    },
    Measured {
        sharpness: f64,
        texture: f64,
    },
}

impl AfSharpness {
    pub fn measured(&self) -> Option<f64> {
        match *self {
            AfSharpness::Measured { sharpness, .. } => Some(sharpness),
            AfSharpness::NoBox | AfSharpness::Featureless { .. } => None,
        }
    }
}

/// Fractions of the frame (0–1) at the ends of the luma range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Clipping {
    pub highlights: f64,
    pub shadows: f64,
}

/// 64-bit difference hash of the displayed frame: near-identical frames differ in few bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DHash(pub u64);

impl DHash {
    pub fn distance(self, other: DHash) -> u32 {
        (self.0 ^ other.0).count_ones()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Features {
    pub af: AfSharpness,
    /// Mean Laplacian variance of the sharpest tiles.
    pub tiles_sharpness: f64,
    pub clipping: Clipping,
    pub dhash: DHash,
}

/// Luma at or beyond these values counts as clipped. JPEG ringing keeps true clipping a few
/// codes short of 0/255.
const HIGHLIGHT_MIN: u8 = 253;
const SHADOW_MAX: u8 = 2;
/// Scale of the copy texture is measured on: coarse enough that defocus barely changes it.
const TEXTURE_SCALE: u32 = 8;

pub fn measure(shot: &Shot, rgb: &RgbImage, cfg: &FeaturesConfig) -> (Features, RgbImage) {
    let luma = imageops::grayscale(rgb);
    let thumb = thumbnail(rgb, shot.orientation, cfg.thumb_long_edge_px);
    let features = Features {
        af: af_sharpness(shot, &luma, cfg),
        tiles_sharpness: tiles_sharpness(&luma, cfg.tiles_x, cfg.tiles_y, cfg.top_tiles),
        clipping: clipping(&luma),
        dhash: dhash(&imageops::grayscale(&thumb)),
    };
    (features, thumb)
}

fn af_sharpness(shot: &Shot, luma: &GrayImage, cfg: &FeaturesConfig) -> AfSharpness {
    let Some(af) = shot.af else {
        return AfSharpness::NoBox;
    };
    let window = expand_to_min(
        af.in_frame(luma.width(), luma.height()),
        cfg.af_window_min_px,
        luma,
    );
    let texture = texture(luma, window);
    if texture < cfg.af_texture_min {
        return AfSharpness::Featureless { texture };
    }
    AfSharpness::Measured {
        sharpness: laplacian_variance(luma, window),
        texture,
    }
}

/// `r` grown symmetrically to at least `min` on each side, shifted to stay inside the frame.
fn expand_to_min(r: PixelRect, min: u32, frame: &GrayImage) -> PixelRect {
    let grow = |pos: u32, len: u32, bound: u32| {
        let len2 = len.max(min).min(bound);
        let centre = pos + len / 2;
        let start = centre.saturating_sub(len2 / 2).min(bound - len2);
        (start, len2)
    };
    let (x, w) = grow(r.x, r.w, frame.width());
    let (y, h) = grow(r.y, r.h, frame.height());
    PixelRect { x, y, w, h }
}

/// Standard deviation of luma over `r`, measured on a `TEXTURE_SCALE`-times box-downscaled
/// copy so that it reflects structure rather than fine focus.
fn texture(luma: &GrayImage, r: PixelRect) -> f64 {
    let s = TEXTURE_SCALE;
    let (cols, rows) = (r.w / s, r.h / s);
    if cols == 0 || rows == 0 {
        return 0.0;
    }
    let mut cells = Vec::with_capacity((cols * rows) as usize);
    for cy in 0..rows {
        for cx in 0..cols {
            let mut sum = 0u32;
            for y in 0..s {
                for x in 0..s {
                    sum += u32::from(luma.get_pixel(r.x + cx * s + x, r.y + cy * s + y)[0]);
                }
            }
            cells.push(f64::from(sum) / f64::from(s * s));
        }
    }
    let n = cells.len() as f64;
    let mean = cells.iter().sum::<f64>() / n;
    (cells.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / n).sqrt()
}

/// Running sums of the 4-neighbour Laplacian over a set of pixels.
#[derive(Default, Clone, Copy)]
struct LaplacianAcc {
    n: u64,
    sum: i64,
    sum_sq: u64,
}

impl LaplacianAcc {
    fn add(&mut self, l: i32) {
        self.n += 1;
        self.sum += i64::from(l);
        self.sum_sq += u64::from(l.unsigned_abs()).pow(2);
    }

    fn variance(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let n = self.n as f64;
        let mean = self.sum as f64 / n;
        self.sum_sq as f64 / n - mean * mean
    }
}

fn laplacian_at(luma: &GrayImage, x: u32, y: u32) -> i32 {
    let p = |x: u32, y: u32| i32::from(luma.get_pixel(x, y)[0]);
    4 * p(x, y) - p(x - 1, y) - p(x + 1, y) - p(x, y - 1) - p(x, y + 1)
}

/// Laplacian variance over `r`, skipping the frame's outermost pixels where the kernel would
/// reach outside the image.
fn laplacian_variance(luma: &GrayImage, r: PixelRect) -> f64 {
    let (w, h) = luma.dimensions();
    let mut acc = LaplacianAcc::default();
    for y in r.y.max(1)..(r.y + r.h).min(h - 1) {
        for x in r.x.max(1)..(r.x + r.w).min(w - 1) {
            acc.add(laplacian_at(luma, x, y));
        }
    }
    acc.variance()
}

/// One pass over the frame accumulating a Laplacian variance per tile; returns the mean of
/// the `top` sharpest. Taking the best tiles rather than the whole frame keeps a sharp subject
/// against a blurred background from being averaged away.
fn tiles_sharpness(luma: &GrayImage, tiles_x: u32, tiles_y: u32, top: u32) -> f64 {
    let (w, h) = luma.dimensions();
    let mut accs = vec![LaplacianAcc::default(); (tiles_x * tiles_y) as usize];
    for y in 1..h - 1 {
        let ty = (y * tiles_y / h) as usize;
        for x in 1..w - 1 {
            let tx = (x * tiles_x / w) as usize;
            accs[ty * tiles_x as usize + tx].add(laplacian_at(luma, x, y));
        }
    }
    let mut vars: Vec<f64> = accs.iter().map(LaplacianAcc::variance).collect();
    vars.sort_by(|a, b| b.total_cmp(a));
    let top = &vars[..top as usize];
    top.iter().sum::<f64>() / top.len() as f64
}

fn clipping(luma: &GrayImage) -> Clipping {
    let mut hist = [0u64; 256];
    for p in luma.pixels() {
        hist[p[0] as usize] += 1;
    }
    let n = (luma.width() as f64) * (luma.height() as f64);
    let count = |range: std::ops::RangeInclusive<u8>| -> f64 {
        range.map(|v| hist[v as usize]).sum::<u64>() as f64 / n
    };
    Clipping {
        highlights: count(HIGHLIGHT_MIN..=255),
        shadows: count(0..=SHADOW_MAX),
    }
}

/// Classic dHash: 9×8 grey thumbnail, one bit per horizontal neighbour pair.
fn dhash(gray: &GrayImage) -> DHash {
    let small = imageops::resize(gray, 9, 8, FilterType::Triangle);
    let mut bits = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let brighter = small.get_pixel(x, y)[0] < small.get_pixel(x + 1, y)[0];
            bits = (bits << 1) | u64::from(brighter);
        }
    }
    DHash(bits)
}

/// Display-oriented thumbnail with the long edge at `long_edge` pixels.
fn thumbnail(rgb: &RgbImage, orientation: Orientation, long_edge: u32) -> RgbImage {
    let (w, h) = rgb.dimensions();
    let scale = f64::from(long_edge) / f64::from(w.max(h));
    let (tw, th) = (
        (f64::from(w) * scale).round() as u32,
        (f64::from(h) * scale).round() as u32,
    );
    orientation.display(&imageops::thumbnail(rgb, tw, th))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: u32, h: u32, cell: u32) -> GrayImage {
        GrayImage::from_fn(w, h, |x, y| {
            image::Luma([if (x / cell + y / cell).is_multiple_of(2) {
                0
            } else {
                200
            }])
        })
    }

    #[test]
    fn flat_image_has_zero_laplacian_variance() {
        let flat = GrayImage::from_pixel(64, 64, image::Luma([128]));
        let r = PixelRect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        };
        assert_eq!(laplacian_variance(&flat, r), 0.0);
    }

    #[test]
    fn blur_lowers_laplacian_variance() {
        let sharp = checkerboard(128, 128, 4);
        let blurred = imageops::blur(&sharp, 2.0);
        let r = PixelRect {
            x: 0,
            y: 0,
            w: 128,
            h: 128,
        };
        assert!(laplacian_variance(&sharp, r) > 4.0 * laplacian_variance(&blurred, r));
    }

    #[test]
    fn texture_survives_blur_that_kills_sharpness() {
        let sharp = checkerboard(256, 256, 32);
        let blurred = imageops::blur(&sharp, 2.0);
        let r = PixelRect {
            x: 0,
            y: 0,
            w: 256,
            h: 256,
        };
        let (ts, tb) = (texture(&sharp, r), texture(&blurred, r));
        assert!(tb > 0.8 * ts, "texture {tb} vs {ts}");
    }

    /// One sharp tile among flat ones: the best-tile mean must find it, a whole-frame
    /// variance would dilute it 48-fold.
    #[test]
    fn tiles_sharpness_finds_the_one_sharp_tile() {
        let mut img = GrayImage::from_pixel(80, 60, image::Luma([100]));
        let board = checkerboard(10, 10, 1);
        imageops::replace(&mut img, &board, 30, 20);
        let one_tile = laplacian_variance(
            &img,
            PixelRect {
                x: 30,
                y: 20,
                w: 10,
                h: 10,
            },
        );
        assert!((tiles_sharpness(&img, 8, 6, 1) - one_tile).abs() < 1e-9);
    }

    #[test]
    fn af_window_is_grown_and_kept_inside_frame() {
        let frame = GrayImage::new(1000, 800);
        let r = expand_to_min(
            PixelRect {
                x: 950,
                y: 10,
                w: 40,
                h: 40,
            },
            300,
            &frame,
        );
        assert_eq!(
            r,
            PixelRect {
                x: 700,
                y: 0,
                w: 300,
                h: 300
            }
        );
    }

    #[test]
    fn clipping_counts_extremes() {
        let img = GrayImage::from_fn(10, 10, |x, _| {
            image::Luma([[0u8, 2, 3, 252, 253, 255, 128, 128, 128, 128][x as usize]])
        });
        let c = clipping(&img);
        assert_eq!(c.shadows, 0.2);
        assert_eq!(c.highlights, 0.2);
    }

    #[test]
    fn dhash_distance_small_for_similar_large_for_different() {
        let a = GrayImage::from_fn(90, 80, |x, y| image::Luma([(x * 2 + y) as u8]));
        let mut b = a.clone();
        b.put_pixel(0, 0, image::Luma([255]));
        let c = imageops::flip_horizontal(&a);
        assert!(dhash(&a).distance(dhash(&b)) <= 2);
        assert!(dhash(&a).distance(dhash(&c)) >= 20);
    }
}
