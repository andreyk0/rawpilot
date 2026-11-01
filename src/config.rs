//! One config source: the embedded `config/default.toml`, or a complete replacement given on
//! the command line. There are no code defaults for any key.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

const DEFAULT_TOML: &str = include_str!("../config/default.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub features: FeaturesConfig,
    pub group: GroupConfig,
    pub rate: RateConfig,
    pub judge: JudgeConfig,
    pub develop: DevelopConfig,
    pub look: LookConfig,
    pub shortlist: ShortlistConfig,
    pub review: ReviewConfig,
}

/// A look option's name. It names the option's directory in the cache and is what the `.rpl`
/// records a choice by, so it is ASCII letters, digits, `-` and `_`, starting with a letter or
/// digit. Built only by parsing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct OptionName(String);

impl OptionName {
    const LEN_MAX: usize = 32;

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for OptionName {
    type Error = String;

    fn try_from(s: String) -> Result<OptionName, String> {
        let first_ok = s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric());
        let rest_ok = s
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if first_ok && rest_ok && s.len() <= OptionName::LEN_MAX {
            Ok(OptionName(s))
        } else {
            Err(format!(
                "option name {s:?} must be 1-{} ASCII letters, digits, '-' or '_', starting with a letter or digit",
                OptionName::LEN_MAX
            ))
        }
    }
}

impl From<OptionName> for String {
    fn from(n: OptionName) -> String {
        n.0
    }
}

impl std::fmt::Display for OptionName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The share of the window's width, in percent, the review's centre column (the fit view)
/// may take. Built only by parsing, so a value held is in range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct CentreWidthPct(u8);

impl CentreWidthPct {
    const RANGE: std::ops::RangeInclusive<u8> = 30..=85;

    pub fn get(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for CentreWidthPct {
    type Error = String;

    fn try_from(pct: u8) -> Result<CentreWidthPct, String> {
        if CentreWidthPct::RANGE.contains(&pct) {
            Ok(CentreWidthPct(pct))
        } else {
            Err(format!(
                "review.centre_width_pct must be in {}..={}, not {pct}",
                CentreWidthPct::RANGE.start(),
                CentreWidthPct::RANGE.end()
            ))
        }
    }
}

impl From<CentreWidthPct> for u8 {
    fn from(pct: CentreWidthPct) -> u8 {
        pct.0
    }
}

/// Processing a look option layers over the photographer's default profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartialSource {
    /// Nothing: the default profile as it is.
    None,
    /// A pp3 holding some keys, at this path.
    File(PathBuf),
    /// A pp3 shipped inside rawpilot.
    Builtin(Builtin),
    /// One of three pp3s chosen by the shot's ISO.
    IsoBands(IsoBands),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Builtin {
    /// RawTherapee's Standard Film Curve, by ISO band (`profiles/film-iso-*.pp3`).
    Film,
    /// `profiles/vivid.pp3`.
    Vivid,
    /// `profiles/warm.pp3`.
    Warm,
    /// `profiles/warm-sun.pp3`.
    WarmSun,
    /// `profiles/bright-sun.pp3`.
    BrightSun,
    /// `profiles/portrait.pp3`.
    Portrait,
    /// `profiles/overcast.pp3`.
    Overcast,
    /// `profiles/dusk.pp3`.
    Dusk,
}

/// Low below `medium_from_iso`, Medium from there to below `high_from_iso`, High from there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsoBands {
    pub low: PathBuf,
    pub medium: PathBuf,
    pub high: PathBuf,
    pub medium_from_iso: u32,
    pub high_from_iso: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptionConfig {
    pub name: OptionName,
    /// The letter the review UI binds to the option.
    pub key: char,
    pub partial: PartialSource,
}

/// Pre-rendered look options for `rawpilot review`; see `looks`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewConfig {
    /// Long edge of the fit view.
    pub fit_long_edge_px: u32,
    pub fit_jpeg_quality: u8,
    /// A 1:1 region larger than this on either side is scaled down to fit.
    pub region_max_px: u32,
    pub region_jpeg_quality: u8,
    /// The AF region is at least this many pixels on each side, around the AF box's centre.
    pub af_region_min_px: u32,
    /// Each face region extends the face box by this fraction of its size on every side.
    pub face_pad_fraction: f64,
    /// The region of a shot with no AF box and no faces: a square at the centre.
    pub centre_region_px: u32,
    /// Quality of the full-size render the views are cut from; it is deleted afterwards.
    pub render_jpeg_quality: u8,
    /// `rawtherapee-cli` processes running at once; each is multithreaded itself.
    pub render_jobs: usize,
    /// `process` renders every option for the shots rated at least this many stars.
    pub prerender_min_rating: u8,
    /// `review` shows a region crop at this multiple of its pixel size (CSS pixels).
    pub region_zoom: f64,
    /// Share of the window's width the centre column, the fit view, takes at most; the
    /// region crops sit on both sides of it.
    pub centre_width_pct: CentreWidthPct,
    /// Height of the filmstrip's thumbnails.
    pub thumb_height_px: u32,
    /// The filmstrip shows this many shots on each side of the current one.
    pub filmstrip_reach: usize,
    #[serde(rename = "option")]
    pub options: Vec<OptionConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortlistConfig {
    pub portfolio_share: f64,
    pub photos_max: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookConfig {
    pub enabled: bool,
    pub key_gap_fraction: f64,
    pub compensation_min_ev: f64,
    pub compensation_max_ev: f64,
    pub compensation_threshold_ev: f64,
    pub highlight_compr: u32,
}

/// What Gemini is shown for each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preview {
    /// The camera's embedded JPEG, brightened back by any negative exposure compensation.
    Camera,
    /// A neutral automatic development of the raw (`develop`).
    Developed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    pub preview: Preview,
    pub model: String,
    pub media_resolution: String,
    pub thinking_level: String,
    pub frames_per_request_max: usize,
    pub scene_photos_per_request_max: usize,
    pub faces_sharpness_rel_min: f64,
    pub request_timeout_s: u64,
    pub attempts_max: u32,
    pub concurrency: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaturesConfig {
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub top_tiles: u32,
    pub af_window_min_px: u32,
    pub af_texture_min: f64,
    pub thumb_long_edge_px: u32,
    pub thumb_jpeg_quality: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupConfig {
    pub burst_gap_s: f64,
    pub burst_dhash_max: u32,
    pub scene_gap_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevelopConfig {
    pub push_ev_min: f32,
    pub push_ev_max: f32,
    pub contrast: f32,
    pub saturation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateConfig {
    pub reject_sharpness_rel_max: f64,
    pub excellent_aesthetic_min: u8,
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Config> {
        let text = match path {
            Some(p) => std::fs::read_to_string(p)
                .with_context(|| format!("reading config {}", p.display()))?,
            None => DEFAULT_TOML.to_owned(),
        };
        let cfg: Config = toml::from_str(&text).context("parsing config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        let f = &self.features;
        ensure!(
            f.tiles_x >= 1 && f.tiles_y >= 1,
            "features.tiles_* must be ≥ 1"
        );
        ensure!(
            (1..=f.tiles_x * f.tiles_y).contains(&f.top_tiles),
            "features.top_tiles must be in 1..=tiles_x*tiles_y"
        );
        ensure!(
            f.thumb_long_edge_px >= 64,
            "features.thumb_long_edge_px must be ≥ 64"
        );
        ensure!(
            (1..=100).contains(&f.thumb_jpeg_quality),
            "features.thumb_jpeg_quality must be in 1..=100"
        );
        let g = &self.group;
        ensure!(g.burst_gap_s > 0.0, "group.burst_gap_s must be > 0");
        ensure!(
            g.burst_dhash_max <= 64,
            "group.burst_dhash_max must be ≤ 64"
        );
        ensure!(
            g.scene_gap_s >= g.burst_gap_s,
            "group.scene_gap_s must be ≥ burst_gap_s"
        );
        let r = &self.rate;
        ensure!(
            (0.0..1.0).contains(&r.reject_sharpness_rel_max),
            "rate.reject_sharpness_rel_max must be in [0, 1)"
        );
        ensure!(
            (1..=5).contains(&r.excellent_aesthetic_min),
            "rate.excellent_aesthetic_min must be in 1..=5"
        );
        let s = &self.shortlist;
        ensure!(
            s.portfolio_share > 0.0 && s.portfolio_share <= 1.0,
            "shortlist.portfolio_share must be in (0, 1]"
        );
        ensure!(
            (2..=64).contains(&s.photos_max),
            "shortlist.photos_max must be in 2..=64"
        );
        let l = &self.look;
        ensure!(
            (0.0..=1.0).contains(&l.key_gap_fraction),
            "look.key_gap_fraction must be in [0, 1]"
        );
        ensure!(
            l.compensation_min_ev <= 0.0 && l.compensation_max_ev >= 0.0,
            "look.compensation_min_ev must be ≤ 0 and compensation_max_ev ≥ 0"
        );
        ensure!(
            l.compensation_threshold_ev >= 0.0,
            "look.compensation_threshold_ev must be ≥ 0"
        );
        ensure!(
            l.highlight_compr <= 500,
            "look.highlight_compr must be in 0..=500 (RawTherapee's range)"
        );
        let d = &self.develop;
        ensure!(
            d.push_ev_min <= 0.0 && d.push_ev_max >= 0.0,
            "develop.push_ev_min must be ≤ 0 and push_ev_max ≥ 0"
        );
        ensure!(
            (0.5..=2.0).contains(&d.contrast),
            "develop.contrast must be in 0.5..=2"
        );
        ensure!(
            (0.0..=2.0).contains(&d.saturation),
            "develop.saturation must be in 0..=2"
        );
        let j = &self.judge;
        ensure!(
            (1..=32).contains(&j.frames_per_request_max),
            "judge.frames_per_request_max must be in 1..=32"
        );
        ensure!(
            (0.0..=1.0).contains(&j.faces_sharpness_rel_min),
            "judge.faces_sharpness_rel_min must be in [0, 1]"
        );
        ensure!(
            (2..=48).contains(&j.scene_photos_per_request_max),
            "judge.scene_photos_per_request_max must be in 2..=48"
        );
        ensure!(
            j.request_timeout_s >= 1,
            "judge.request_timeout_s must be ≥ 1"
        );
        ensure!(
            (1..=5).contains(&j.attempts_max),
            "judge.attempts_max must be in 1..=5"
        );
        ensure!(
            (1..=16).contains(&j.concurrency),
            "judge.concurrency must be in 1..=16"
        );
        self.review.validate()
    }
}

impl ReviewConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            (256..=8192).contains(&self.fit_long_edge_px),
            "review.fit_long_edge_px must be in 256..=8192"
        );
        for (q, name) in [
            (self.fit_jpeg_quality, "fit_jpeg_quality"),
            (self.region_jpeg_quality, "region_jpeg_quality"),
            (self.render_jpeg_quality, "render_jpeg_quality"),
        ] {
            ensure!((1..=100).contains(&q), "review.{name} must be in 1..=100");
        }
        ensure!(
            (64..=4096).contains(&self.region_max_px),
            "review.region_max_px must be in 64..=4096"
        );
        ensure!(
            (1..=self.region_max_px).contains(&self.af_region_min_px),
            "review.af_region_min_px must be in 1..=region_max_px"
        );
        ensure!(
            (1..=self.region_max_px).contains(&self.centre_region_px),
            "review.centre_region_px must be in 1..=region_max_px"
        );
        ensure!(
            (0.0..=2.0).contains(&self.face_pad_fraction),
            "review.face_pad_fraction must be in [0, 2]"
        );
        ensure!(
            (1..=16).contains(&self.render_jobs),
            "review.render_jobs must be in 1..=16"
        );
        ensure!(
            self.prerender_min_rating <= 5,
            "review.prerender_min_rating must be in 0..=5"
        );
        ensure!(
            (0.25..=4.0).contains(&self.region_zoom),
            "review.region_zoom must be in 0.25..=4"
        );
        ensure!(
            (48..=240).contains(&self.thumb_height_px),
            "review.thumb_height_px must be in 48..=240"
        );
        ensure!(
            (1..=50).contains(&self.filmstrip_reach),
            "review.filmstrip_reach must be in 1..=50"
        );
        ensure!(
            !self.options.is_empty(),
            "review.option needs at least one option"
        );
        let reserved = crate::review::reserved_option_letters();
        for (i, o) in self.options.iter().enumerate() {
            ensure!(
                o.key.is_ascii_alphabetic(),
                "review.option {}: key {:?} must be an ASCII letter",
                o.name,
                o.key
            );
            ensure!(
                !reserved.contains(&o.key.to_ascii_lowercase()),
                "review.option {}: key {:?} is one the review page uses itself ({}); case does \
                 not matter",
                o.name,
                o.key,
                reserved
                    .iter()
                    .map(char::to_string)
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            for p in &self.options[..i] {
                ensure!(
                    !p.name.as_str().eq_ignore_ascii_case(o.name.as_str()),
                    "review.option names {} and {} are the same",
                    p.name,
                    o.name
                );
                ensure!(
                    !p.key.eq_ignore_ascii_case(&o.key),
                    "review.option {} and {} share the key {:?}",
                    p.name,
                    o.name,
                    o.key
                );
            }
            if let PartialSource::IsoBands(b) = &o.partial {
                ensure!(
                    b.medium_from_iso < b.high_from_iso,
                    "review.option {}: medium_from_iso must be below high_from_iso",
                    o.name
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_default_is_valid() {
        let cfg = Config::load(None).unwrap();
        let names: Vec<&str> = cfg.review.options.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "Camera", "Film", "Vivid", "Warm", "Warm-sun", "Sunny", "Portrait", "Overcast",
                "Dusk"
            ]
        );
    }

    fn with_review(edit: impl FnOnce(&mut Config)) -> Result<Config> {
        let mut cfg = Config::load(None).unwrap();
        edit(&mut cfg);
        cfg.validate().map(|()| cfg)
    }

    #[test]
    fn options_must_differ_in_name_and_key() {
        assert!(with_review(|c| c.review.options[1].key = 'c').is_err());
        assert!(with_review(|c| c.review.options[1].key = 'C').is_err());
        assert!(
            with_review(|c| c.review.options[1].name = c.review.options[0].name.clone()).is_err()
        );
        assert!(with_review(|c| c.review.options.clear()).is_err());
        assert!(with_review(|c| c.review.options[1].key = '1').is_err());
    }

    /// Goal: a key the review page binds itself is refused where the configuration is loaded,
    /// so `process` and `cull` fail on it too instead of leaving it for `review` to find.
    /// Mutation: drop the reserved-letter check from `ReviewConfig::validate`: the loop goes
    /// red. `y` and `n` answer a question, a table of their own, and stay available.
    #[test]
    fn an_option_cannot_take_a_key_the_review_page_uses() {
        for key in ['j', 'k', 'f', 'J', 'K', 'F'] {
            let refused = with_review(|c| c.review.options[1].key = key).unwrap_err();
            assert!(format!("{refused}").contains("f j k"), "{key}: {refused}");
        }
        for key in ['y', 'n', 'v', 'a', 'Z'] {
            assert!(
                with_review(|c| c.review.options[1].key = key).is_ok(),
                "{key}"
            );
        }
    }

    /// Goal: `[review] centre_width_pct` is the key, defaults to 60, is range-checked where
    /// the file is parsed, and a config still saying `fit_width_pct` is refused.
    /// Method: the embedded default with the line edited, parsed as `Config::load` does.
    fn parse_with_centre_line(line: &str) -> Result<Config, toml::de::Error> {
        let edited = DEFAULT_TOML.replace("centre_width_pct = 60", line);
        assert_ne!(
            edited, DEFAULT_TOML,
            "the default no longer has the line the test edits"
        );
        toml::from_str(&edited)
    }

    /// Mutation: add `#[serde(alias = "fit_width_pct")]` to the field: the last assert goes
    /// red; a different default than 60 in `default.toml` fails the first.
    #[test]
    fn the_centre_column_share_is_centre_width_pct_defaulting_to_60() {
        assert_eq!(
            Config::load(None).unwrap().review.centre_width_pct.get(),
            60
        );
        assert_eq!(
            parse_with_centre_line("centre_width_pct = 45")
                .unwrap()
                .review
                .centre_width_pct
                .get(),
            45
        );
        assert!(parse_with_centre_line("fit_width_pct = 60").is_err());
    }

    /// Mutation: widen `CentreWidthPct::RANGE` to `0..=100`: the 29, 86 and 0 rows go red.
    #[test]
    fn the_centre_column_share_is_range_checked_at_parse() {
        for pct in [0, 29, 86, 100, 255] {
            assert!(
                parse_with_centre_line(&format!("centre_width_pct = {pct}")).is_err(),
                "{pct}"
            );
        }
        for pct in [30, 85] {
            assert!(
                parse_with_centre_line(&format!("centre_width_pct = {pct}")).is_ok(),
                "{pct}"
            );
        }
    }

    #[test]
    fn iso_bands_must_ascend() {
        assert!(
            with_review(|c| {
                c.review.options[1].partial = PartialSource::IsoBands(IsoBands {
                    low: "low.pp3".into(),
                    medium: "medium.pp3".into(),
                    high: "high.pp3".into(),
                    medium_from_iso: 800,
                    high_from_iso: 800,
                });
            })
            .is_err()
        );
    }
}
