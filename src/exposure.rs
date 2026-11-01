//! Undoing deliberate under-exposure for viewing. In bright sun the photographer often dials
//! in negative exposure compensation to keep highlights, and brightens in editing. The
//! camera's JPEG then looks dark, and a judge looking at it would mark down frames that are
//! fine once edited.

use image::RgbImage;

use crate::tone::{linear_to_srgb, shoulder, srgb_to_linear};

/// Linear-light level above which brightened values are compressed instead of clipped, so the
/// highlights the raw still holds stay distinguishable.
const SHOULDER_START: f64 = 0.8;

/// `rgb` brightened by `-compensation_ev` stops, as editing would; unchanged unless the
/// compensation is negative.
pub fn undo_negative_compensation(rgb: &RgbImage, compensation_ev: Option<f64>) -> RgbImage {
    let Some(ev) = compensation_ev.filter(|&ev| ev < 0.0) else {
        return rgb.clone();
    };
    let gain = (-ev).exp2();
    let lut: Vec<u8> = (0..=255u8).map(|v| brighten(v, gain)).collect();
    let mut out = rgb.clone();
    for p in out.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = lut[*c as usize];
        }
    }
    out
}

/// The JPEG's white becomes `gain` in linear light; the shoulder lands it back on white.
fn brighten(v: u8, gain: f64) -> u8 {
    assert!(gain >= 1.0, "brighten only ever brightens");
    let linear = srgb_to_linear(f64::from(v) / 255.0) * gain;
    (linear_to_srgb(shoulder(linear, gain, SHOULDER_START)) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 128/255 = 0.50196 is 0.21586 linear; one stop doubles it to 0.43173, which is sRGB
    /// 0.68845 = 175.55 → 176.
    #[test]
    fn one_stop_doubles_linear_light() {
        let img = RgbImage::from_pixel(1, 1, image::Rgb([128, 128, 128]));
        let out = undo_negative_compensation(&img, Some(-1.0));
        assert_eq!(out.get_pixel(0, 0).0, [176, 176, 176]);
    }

    #[test]
    fn zero_or_positive_compensation_is_left_alone() {
        let img = RgbImage::from_pixel(1, 1, image::Rgb([128, 64, 32]));
        assert_eq!(undo_negative_compensation(&img, Some(0.33)), img);
        assert_eq!(undo_negative_compensation(&img, None), img);
    }

    /// Bright tones are compressed rather than clipped. By hand for one stop: 230 is 0.791
    /// linear, ×2 = 1.582, t = 0.652, s = 6, f = 0.918, out = 0.9836 linear = sRGB 253.
    #[test]
    fn highlights_roll_off_instead_of_clipping() {
        assert_eq!(brighten(230, 2.0), 253);
        assert_eq!(brighten(255, 2.0), 255);
        let outs: Vec<u8> = (0..=255u8).map(|v| brighten(v, 2.5)).collect();
        assert!(outs.windows(2).all(|w| w[0] <= w[1]), "{outs:?}");
    }
}
