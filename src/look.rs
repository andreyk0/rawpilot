//! The exposure RawTherapee opens a fresh image with.
//!
//! RawTherapee's "Auto-Matched Curve" default profile fits a tone curve that reproduces the
//! camera's embedded JPEG, so a fresh image opens looking like that JPEG — dark when shot
//! with negative exposure compensation to protect highlights in sun. The fix is measured
//! against that JPEG: how much brighter the neutral development (`develop`) is, in stops of
//! scene key. Judged on rawtherapee-cli renders, the full difference overshoots (+2.5 EV on
//! a −1.33 EV forest frame that looks right at +1.33 EV: a mid-grey key over-brightens
//! naturally low-key scenes), so only part of it is applied, and never
//! less than undoing the photographer's own negative compensation. Highlight compression
//! keeps a brightened sky from blowing out.

use anyhow::{Context, Result};
use image::RgbImage;

use crate::analyze::Analyzed;
use crate::cache::{self, Cache};
use crate::config::LookConfig;
use crate::pp3::Look;
use crate::tone;

/// The look for `a`, or `None` when it has no development to measure against or the
/// correction is too small to be worth overriding RawTherapee's default.
pub fn for_shot(cache: &Cache, a: &Analyzed, cfg: &LookConfig) -> Result<Option<Look>> {
    if a.developed.is_none() {
        return Ok(None);
    }
    let open = |kind| -> Result<RgbImage> {
        let p = cache.image_path(kind, &a.shot);
        Ok(image::open(&p)
            .with_context(|| format!("reading {}", p.display()))?
            .into_rgb8())
    };
    let camera_key = tone::log_average_srgb(&open(cache::FEATURES)?);
    let developed_key = tone::log_average_srgb(&open(cache::DEVELOPED)?);
    Ok(compensation(
        (developed_key / camera_key).log2(),
        a.shot.exposure.compensation_ev,
        cfg,
    )
    .map(|ev| Look {
        compensation_ev: ev,
        highlight_compr: if ev > 0.0 { cfg.highlight_compr } else { 0 },
    }))
}

/// `key_gap_ev`: how many stops brighter the development's key is than the camera JPEG's.
fn compensation(key_gap_ev: f64, shot_ev: Option<f64>, cfg: &LookConfig) -> Option<f64> {
    let from_gap = cfg.key_gap_fraction * key_gap_ev;
    // Undoing deliberate negative compensation is a floor; without it the gap alone decides,
    // in either direction.
    let ev = match shot_ev.filter(|&ev| ev < 0.0) {
        Some(shot) => from_gap.max(-shot),
        None => from_gap,
    }
    .clamp(cfg.compensation_min_ev, cfg.compensation_max_ev);
    (ev.abs() >= cfg.compensation_threshold_ev).then_some(ev)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LookConfig {
        LookConfig {
            enabled: true,
            key_gap_fraction: 0.5,
            compensation_min_ev: -1.0,
            compensation_max_ev: 2.0,
            compensation_threshold_ev: 0.25,
            highlight_compr: 100,
        }
    }

    /// Measured frames (key gap, shot compensation) and the outcome rawtherapee-cli renders
    /// favoured.
    #[test]
    fn measured_frames() {
        // Forest at −1.33 EV, key gap +2.52: undoing the shot's −1.33 wins.
        assert_eq!(compensation(2.52, Some(-1.333), &cfg()), Some(1.333));
        // Couple in shade at 0 EV, key gap +1.86: half the gap.
        assert_eq!(compensation(1.86, Some(0.0), &cfg()), Some(0.93));
        // Sunny portrait, key gap −0.07: leave RawTherapee's default alone.
        assert_eq!(compensation(-0.07, Some(0.0), &cfg()), None);
    }

    #[test]
    fn bounded_both_ways() {
        assert_eq!(compensation(9.0, None, &cfg()), Some(2.0));
        assert_eq!(compensation(-9.0, None, &cfg()), Some(-1.0));
    }
}
