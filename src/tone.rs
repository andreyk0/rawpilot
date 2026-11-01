//! Tone primitives shared by every rendering rawpilot makes for judging.

/// Identity up to `start`; above it `[start, white]` maps onto `[start, 1]` by
/// `f(t) = s·t / (1 + (s−1)·t)` with `s = (white − start) / (1 − start)`, which keeps the
/// slope continuous with the identity part and lands `white` exactly on 1. Highlights are
/// compressed rather than clipped.
pub fn shoulder(x: f64, white: f64, start: f64) -> f64 {
    assert!(
        (0.0..1.0).contains(&start),
        "shoulder start {start} outside [0, 1)"
    );
    if x <= start || white <= 1.0 {
        return x.min(1.0);
    }
    let room = 1.0 - start;
    let t = ((x - start) / (white - start)).min(1.0);
    let s = (white - start) / room;
    start + room * (s * t / (1.0 + (s - 1.0) * t))
}

/// Rec. 709 luma weights of linear red, green and blue (also sRGB's): the luminance of a
/// linear-light colour is the sum of its channels times these.
pub const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Linear light of each 8-bit sRGB code value, indexed by the code.
pub fn srgb8_to_linear() -> [f64; 256] {
    std::array::from_fn(|v| srgb_to_linear(v as f64 / 255.0))
}

/// Geometric mean of linear-light luminance of an sRGB image: the scene's "key".
pub fn log_average_srgb(img: &image::RgbImage) -> f64 {
    let lut = srgb8_to_linear();
    let n = f64::from(img.width()) * f64::from(img.height());
    let sum: f64 = img
        .pixels()
        .map(|p| {
            let y = LUMA[0] * lut[p[0] as usize]
                + LUMA[1] * lut[p[1] as usize]
                + LUMA[2] * lut[p[2] as usize];
            y.max(1e-4).ln()
        })
        .sum();
    (sum / n).exp()
}

pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_lands_on_one_and_below_start_is_identity() {
        assert!((shoulder(3.0, 3.0, 0.5) - 1.0).abs() < 1e-12);
        assert_eq!(shoulder(0.3, 3.0, 0.5), 0.3);
        assert!(shoulder(2.0, 3.0, 0.5) < 0.999);
    }

    /// With no push (white at or below 1) nothing needs compressing: identity, clipped at 1.
    #[test]
    fn no_push_is_identity() {
        assert_eq!(shoulder(0.7, 0.8, 0.5), 0.7);
        assert_eq!(shoulder(1.2, 1.0, 0.5), 1.0);
    }

    #[test]
    fn monotonic_across_the_shoulder() {
        let ys: Vec<f64> = (0..=100)
            .map(|i| shoulder(f64::from(i) * 0.03, 3.0, 0.5))
            .collect();
        assert!(ys.windows(2).all(|w| w[0] <= w[1]));
    }

    /// A uniform image's key is its own luminance: sRGB 118 is linear 0.1812.
    #[test]
    fn key_of_uniform_grey() {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([118, 118, 118]));
        assert!((log_average_srgb(&img) - 0.1812).abs() < 1e-3);
    }

    /// The table has one entry per 8-bit code, from linear 0 to linear 1, and 118 lands on
    /// 0.1812 (as in the key test above); Rec. 709 weights sum to 1 so grey keeps its level.
    #[test]
    fn code_table_and_luma_weights() {
        let lut = srgb8_to_linear();
        assert_eq!((lut[0], lut[255]), (0.0, 1.0));
        assert!((lut[118] - 0.1812).abs() < 1e-3);
        assert!(lut.windows(2).all(|w| w[0] < w[1]));
        assert!((LUMA.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// Published sRGB reference points: linear 0.18 ↔ 0.4614, and the two ends.
    #[test]
    fn srgb_round_trip_and_reference_points() {
        assert!((linear_to_srgb(0.18) - 0.4614).abs() < 1e-4);
        assert!((srgb_to_linear(0.4614) - 0.18).abs() < 1e-4);
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-12);
    }
}
