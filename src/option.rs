//! Look options: a name, a key letter for the review UI, and the processing the option
//! layers over the photographer's default profile.
//!
//! Configured under `[[review.option]]`. An option's partial is read and parsed once, when the
//! options are loaded, so a missing or malformed profile fails before any rendering starts.

use std::fs;

use anyhow::{Context, Result};

use crate::config::{Builtin, IsoBands, OptionConfig, OptionName, PartialSource};
use crate::pp3::Partial;

const FILM_ISO_LOW: &str = include_str!("../profiles/film-iso-low.pp3");
const FILM_ISO_MEDIUM: &str = include_str!("../profiles/film-iso-medium.pp3");
const FILM_ISO_HIGH: &str = include_str!("../profiles/film-iso-high.pp3");
const VIVID: &str = include_str!("../profiles/vivid.pp3");
const WARM: &str = include_str!("../profiles/warm.pp3");
const WARM_SUN: &str = include_str!("../profiles/warm-sun.pp3");
const BRIGHT_SUN: &str = include_str!("../profiles/bright-sun.pp3");
const PORTRAIT: &str = include_str!("../profiles/portrait.pp3");
const OVERCAST: &str = include_str!("../profiles/overcast.pp3");
const DUSK: &str = include_str!("../profiles/dusk.pp3");

/// The built-in Film's bands. The three profiles differ only in noise handling: Low adds
/// post-demosaic sharpening, Medium enables the (neutral) denoise block, High adds median +
/// luminance denoise and LMMSE demosaicing. Low below ISO 800, where a Z8 frame is clean
/// enough to sharpen; Medium from 800, where noise starts to show at 100 %; High from 3200,
/// where luminance noise shows at screen size and the stronger denoise earns its softness.
const FILM_THRESHOLDS: IsoThresholds = IsoThresholds {
    medium_from: 800,
    high_from: 3200,
};

#[derive(Debug, Clone)]
pub struct LookOption {
    pub name: OptionName,
    /// The letter the review UI binds to the option.
    pub key: char,
    layer: Layer,
}

#[derive(Debug, Clone)]
enum Layer {
    Fixed(Partial),
    ByIso {
        low: Partial,
        medium: Partial,
        high: Partial,
        thresholds: IsoThresholds,
    },
}

/// Where the ISO bands start: Medium at `medium_from`, High at `high_from`, Low below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IsoThresholds {
    medium_from: u32,
    high_from: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IsoBand {
    Low,
    Medium,
    High,
}

impl IsoThresholds {
    fn band(self, iso: u32) -> IsoBand {
        if iso >= self.high_from {
            IsoBand::High
        } else if iso >= self.medium_from {
            IsoBand::Medium
        } else {
            IsoBand::Low
        }
    }
}

impl LookOption {
    pub fn load_all(configured: &[OptionConfig]) -> Result<Vec<LookOption>> {
        configured
            .iter()
            .map(|c| {
                Ok(LookOption {
                    name: c.name.clone(),
                    key: c.key,
                    layer: Layer::load(&c.partial)
                        .with_context(|| format!("look option {}", c.name))?,
                })
            })
            .collect()
    }

    /// The processing this option layers over the default profile for a shot taken at `iso`.
    pub fn partial_for(&self, iso: u32) -> &Partial {
        match &self.layer {
            Layer::Fixed(p) => p,
            Layer::ByIso {
                low,
                medium,
                high,
                thresholds,
            } => match thresholds.band(iso) {
                IsoBand::Low => low,
                IsoBand::Medium => medium,
                IsoBand::High => high,
            },
        }
    }
}

impl Layer {
    fn load(source: &PartialSource) -> Result<Layer> {
        Ok(match source {
            PartialSource::None => Layer::Fixed(Partial::none()),
            PartialSource::File(path) => Layer::Fixed(read_partial(path)?),
            PartialSource::Builtin(Builtin::Film) => Layer::ByIso {
                low: built_in(FILM_ISO_LOW, "film-iso-low")?,
                medium: built_in(FILM_ISO_MEDIUM, "film-iso-medium")?,
                high: built_in(FILM_ISO_HIGH, "film-iso-high")?,
                thresholds: FILM_THRESHOLDS,
            },
            PartialSource::Builtin(Builtin::Vivid) => Layer::Fixed(built_in(VIVID, "vivid")?),
            PartialSource::Builtin(Builtin::Warm) => Layer::Fixed(built_in(WARM, "warm")?),
            PartialSource::Builtin(Builtin::WarmSun) => {
                Layer::Fixed(built_in(WARM_SUN, "warm-sun")?)
            }
            PartialSource::Builtin(Builtin::BrightSun) => {
                Layer::Fixed(built_in(BRIGHT_SUN, "bright-sun")?)
            }
            PartialSource::Builtin(Builtin::Portrait) => {
                Layer::Fixed(built_in(PORTRAIT, "portrait")?)
            }
            PartialSource::Builtin(Builtin::Overcast) => {
                Layer::Fixed(built_in(OVERCAST, "overcast")?)
            }
            PartialSource::Builtin(Builtin::Dusk) => Layer::Fixed(built_in(DUSK, "dusk")?),
            PartialSource::IsoBands(IsoBands {
                low,
                medium,
                high,
                medium_from_iso,
                high_from_iso,
            }) => Layer::ByIso {
                low: read_partial(low)?,
                medium: read_partial(medium)?,
                high: read_partial(high)?,
                thresholds: IsoThresholds {
                    medium_from: *medium_from_iso,
                    high_from: *high_from_iso,
                },
            },
        })
    }
}

fn built_in(text: &str, name: &str) -> Result<Partial> {
    Partial::parse(text).with_context(|| format!("the built-in profile profiles/{name}.pp3"))
}

fn read_partial(path: &std::path::Path) -> Result<Partial> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Partial::parse(&text).with_context(|| format!("in {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Goal: the ISO bands pick the right profile at and around each threshold. Method:
    /// three one-key profiles that name their band, selected at the boundaries.
    #[test]
    fn iso_selects_its_band_at_the_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        for band in ["Low", "Medium", "High"] {
            fs::write(
                dir.path().join(format!("{band}.pp3")),
                format!("[Test]\nBand={band}\n"),
            )
            .unwrap();
        }
        let path = |b: &str| dir.path().join(format!("{b}.pp3"));
        let option = LookOption::load_all(&[OptionConfig {
            name: "Film".to_owned().try_into().unwrap(),
            key: 'f',
            partial: PartialSource::IsoBands(IsoBands {
                low: path("Low"),
                medium: path("Medium"),
                high: path("High"),
                medium_from_iso: 800,
                high_from_iso: 3200,
            }),
        }])
        .unwrap()
        .remove(0);
        let band_at = |iso| option.partial_for(iso).get("Test", "Band").unwrap();
        assert_eq!(band_at(64), "Low");
        assert_eq!(band_at(799), "Low");
        assert_eq!(band_at(800), "Medium");
        assert_eq!(band_at(3199), "Medium");
        assert_eq!(band_at(3200), "High");
        assert_eq!(band_at(25600), "High");
    }

    #[test]
    fn a_missing_profile_fails_the_load() {
        let missing = OptionConfig {
            name: "Film".to_owned().try_into().unwrap(),
            key: 'f',
            partial: PartialSource::File("/nonexistent/x.pp3".into()),
        };
        assert!(LookOption::load_all(&[missing]).is_err());
    }

    /// Each built-in look is its own file, and every one but Film turns histogram matching
    /// off, or the matched curve would replace its own. The file names are written here
    /// rather than shared with the loader, so a look wired to another's file fails.
    #[test]
    fn each_built_in_look_is_its_own_file() {
        let read = |name: &str| {
            let path = format!("{}/profiles/{name}.pp3", env!("CARGO_MANIFEST_DIR"));
            Partial::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
        };
        for (builtin, file) in [
            (Builtin::Vivid, "vivid"),
            (Builtin::Warm, "warm"),
            (Builtin::WarmSun, "warm-sun"),
            (Builtin::BrightSun, "bright-sun"),
            (Builtin::Portrait, "portrait"),
            (Builtin::Overcast, "overcast"),
            (Builtin::Dusk, "dusk"),
        ] {
            let p = built_in_option(builtin).partial_for(100).clone();
            assert_eq!(p, read(file), "{builtin:?}");
            assert_eq!(
                p.get("Exposure", "HistogramMatching"),
                Some("false"),
                "{builtin:?}"
            );
        }
    }

    /// Warm shifts the midtones toward red and away from blue, relative to the shot's white
    /// balance; its ends stay put, so white stays white and black black.
    #[test]
    fn warm_moves_the_midtones_only() {
        let warm = built_in_option(Builtin::Warm);
        let p = warm.partial_for(100);
        assert_eq!(p.get("RGB Curves", "rCurve"), Some("1;0;0;0.5;0.525;1;1;"));
        assert_eq!(p.get("RGB Curves", "bCurve"), Some("1;0;0;0.5;0.465;1;1;"));
    }

    fn built_in_option(builtin: Builtin) -> LookOption {
        LookOption::load_all(&[OptionConfig {
            name: "X".to_owned().try_into().unwrap(),
            key: 'x',
            partial: PartialSource::Builtin(builtin),
        }])
        .unwrap()
        .remove(0)
    }

    /// The built-in Film is the shipped copy of each ISO band: only Low sharpens after
    /// demosaicing, only High demosaics with LMMSE, and all three share the film curve.
    #[test]
    fn built_in_film_picks_its_band_by_iso() {
        let film = built_in_option(Builtin::Film);
        let sharpens = |iso| {
            film.partial_for(iso)
                .get("PostDemosaicSharpening", "Enabled")
        };
        let demosaic = |iso| film.partial_for(iso).get("RAW Bayer", "Method");
        assert_eq!(sharpens(799), Some("true"));
        assert_eq!(sharpens(800), None);
        assert_eq!(demosaic(3199), None);
        assert_eq!(demosaic(3200), Some("lmmse"));
        for iso in [100, 800, 3200] {
            assert_eq!(
                film.partial_for(iso).get("Exposure", "CurveMode"),
                Some("FilmLike")
            );
        }
    }

    /// Vivid's vibrance protects skin, and the matched curve is off so the film curve holds.
    #[test]
    fn built_in_vivid_protects_skin() {
        let vivid = built_in_option(Builtin::Vivid);
        let p = vivid.partial_for(100);
        assert_eq!(p.get("Exposure", "HistogramMatching"), Some("false"));
        assert_eq!(p.get("Vibrance", "Enabled"), Some("true"));
        assert_eq!(p.get("Vibrance", "ProtectSkins"), Some("true"));
        assert_eq!(p.get("Local Contrast", "Enabled"), Some("true"));
    }

    #[test]
    fn option_names_are_safe_directory_names() {
        for bad in ["", "a/b", "..", "-x", "a b", &"x".repeat(33)] {
            assert!(OptionName::try_from(bad.to_owned()).is_err(), "{bad:?}");
        }
        for good in ["Camera", "film_2", "A-b"] {
            assert!(OptionName::try_from(good.to_owned()).is_ok(), "{good:?}");
        }
    }
}
