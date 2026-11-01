//! RawTherapee `.pp3` processing profiles as a rating store. In `procparams` mode RawTherapee
//! takes a thumbnail's rank and colour from `[General] Rank` and `ColorLabel` here.
//!
//! A pp3 must be complete: RawTherapee fills a missing key with its built-in neutral value,
//! not with the user's default profile, so a pp3 holding only a rank would silently reset the
//! image's processing. New pp3s therefore start from the `DefaultProfile` that
//! `rtcli::default_profile_pp3` resolves, never from scratch; one with a look option's
//! processing is a `CompletePp3`. An existing pp3 has its rank and colour lines edited, and
//! its exposure keys where rawpilot owns it.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};

use crate::rate::Label;
use crate::sidecar::{FoundLabel, FoundMark, Mark};

pub fn path_for(nef: &Path) -> PathBuf {
    let mut p = nef.as_os_str().to_owned();
    p.push(".pp3");
    PathBuf::from(p)
}

/// RawTherapee's colour label numbering.
fn label_index(l: Label) -> u8 {
    match l {
        Label::Red => 1,
        Label::Yellow => 2,
        Label::Green => 3,
        Label::Blue => 4,
        Label::Purple => 5,
    }
}

fn label_from_index(n: i64) -> Option<FoundLabel> {
    let known = match n {
        0 => return None,
        1 => Label::Red,
        2 => Label::Yellow,
        3 => Label::Green,
        4 => Label::Blue,
        5 => Label::Purple,
        other => return Some(FoundLabel::Other(other.to_string())),
    };
    Some(FoundLabel::Known(known))
}

/// Line range `[header, end)` of the section `[name]`.
fn section(lines: &[&str], name: &str) -> Option<(usize, usize)> {
    let header = format!("[{name}]");
    let start = lines.iter().position(|l| l.trim() == header)?;
    let end = lines[start + 1..]
        .iter()
        .position(|l| l.trim_start().starts_with('['))
        .map_or(lines.len(), |i| start + 1 + i);
    Some((start, end))
}

fn key_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (k, v) = line.split_once('=')?;
    (k.trim() == key).then(|| v.trim())
}

pub fn read_mark(text: &str) -> FoundMark {
    let lines: Vec<&str> = text.lines().collect();
    let Some((start, end)) = section(&lines, "General") else {
        return FoundMark::default();
    };
    let value = |key: &str| {
        lines[start + 1..end]
            .iter()
            .find_map(|l| key_value(l, key))
            .and_then(|v| v.parse::<i64>().ok())
    };
    FoundMark {
        rating: value("Rank"),
        label: value("ColorLabel").and_then(label_from_index),
    }
}

/// `text` with `[General] Rank` and `ColorLabel` set to `m`, every other line unchanged.
pub fn with_mark(text: &str, m: Mark) -> Result<String> {
    with_keys(
        text,
        "General",
        &[
            ("Rank", m.rating.get().to_string()),
            ("ColorLabel", m.label.map_or(0, label_index).to_string()),
        ],
    )
}

/// The exposure RawTherapee should open a fresh image with (`look::for_shot`).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Look {
    pub compensation_ev: f64,
    /// RawTherapee's highlight compression, 0–500.
    pub highlight_compr: u32,
}

/// `text` with `[Exposure] Compensation` and `HighlightCompr` set to `look`, every other line
/// unchanged.
pub fn with_look(text: &str, look: Look) -> Result<String> {
    with_keys(
        text,
        "Exposure",
        &[
            ("Compensation", format!("{:.2}", look.compensation_ev)),
            ("HighlightCompr", look.highlight_compr.to_string()),
        ],
    )
}

/// blake3 of a pp3's exact bytes: how rawpilot recognises a file it wrote or rendered from.
pub fn digest(text: &str) -> String {
    blake3::hash(text.as_bytes()).to_hex().to_string()
}

/// blake3 of a pp3 without its `[General]` `Rank` and `ColorLabel` lines: the digest of its
/// processing, unchanged by rating the image.
pub fn processing_digest(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let general = section(&lines, "General");
    let kept: Vec<&str> = lines
        .iter()
        .enumerate()
        .filter(|&(i, l)| {
            let is_mark = ["Rank", "ColorLabel"]
                .iter()
                .any(|k| key_value(l, k).is_some());
            !(is_mark && general.is_some_and(|(start, end)| start < i && i < end))
        })
        .map(|(_, l)| *l)
        .collect();
    digest(&kept.join("\n"))
}

/// The photographer's default processing profile as `rawtherapee-cli` resolves it: complete,
/// image-independent. Built only by `DefaultProfile::parse`, which `rtcli` calls on what
/// RawTherapee wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultProfile(String);

impl DefaultProfile {
    pub fn parse(text: String) -> Result<DefaultProfile> {
        ensure!(
            text.lines().any(|l| l.trim() == "[General]"),
            "the default profile has no [General] section"
        );
        Ok(DefaultProfile(text))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Some keys of a pp3, to be layered over a complete profile. It is never written to disk as
/// it is: RawTherapee fills a missing key with its neutral value, so a partial pp3 would
/// reset whatever it does not mention.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Partial(Vec<PartialSection>);

#[derive(Debug, Clone, PartialEq, Eq)]
struct PartialSection {
    name: String,
    keys: Vec<(String, String)>,
}

impl Partial {
    /// The layer that changes nothing.
    pub fn none() -> Partial {
        Partial::default()
    }

    /// blake3 of the sections and keys in their order, so that any change to what the layer
    /// sets changes it.
    pub fn digest(&self) -> String {
        let mut text = String::new();
        for section in &self.0 {
            text.push_str(&format!("[{}]\n", section.name));
            for (key, value) in &section.keys {
                text.push_str(&format!("{key}={value}\n"));
            }
        }
        digest(&text)
    }

    /// The `[section]` / `key=value` lines of a pp3; blank lines are skipped, anything else is
    /// refused.
    pub fn parse(text: &str) -> Result<Partial> {
        let mut sections: Vec<PartialSection> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end();
            if line.trim().is_empty() {
                continue;
            }
            if let Some(name) = line
                .trim()
                .strip_prefix('[')
                .and_then(|l| l.strip_suffix(']'))
            {
                ensure!(
                    sections.iter().all(|s| s.name != name),
                    "line {}: section [{name}] appears twice",
                    n + 1
                );
                sections.push(PartialSection {
                    name: name.to_owned(),
                    keys: Vec::new(),
                });
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                bail!(
                    "line {}: not a section header or key=value: {line:?}",
                    n + 1
                );
            };
            let Some(section) = sections.last_mut() else {
                bail!("line {}: key before any section: {line:?}", n + 1);
            };
            section
                .keys
                .push((key.trim().to_owned(), value.trim().to_owned()));
        }
        Ok(Partial(sections))
    }
}

#[cfg(test)]
impl Partial {
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|s| s.name == section)?
            .keys
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// A pp3 holding every key of the photographer's default profile, with an option's partial
/// layered over it and the shot's exposure look applied over that. This is the only pp3 text
/// rawpilot renders from or writes for a look option, and `compose` is the only way to make
/// one besides re-reading a stored copy whose digest vouches for it (`from_stored`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletePp3 {
    text: String,
    digest: String,
}

impl CompletePp3 {
    /// `default` with `partial` layered over it, then `look` (when the shot has one) applied.
    pub fn compose(
        default: &DefaultProfile,
        partial: &Partial,
        look: Option<Look>,
    ) -> Result<CompletePp3> {
        let layered = with_partial(default.as_str(), partial)?;
        let text = match look {
            Some(look) => with_look(&layered, look)?,
            None => layered,
        };
        Ok(CompletePp3::of(text))
    }

    /// A stored copy of a composed pp3, accepted only if it is the text `digest` names.
    pub fn from_stored(text: String, digest: &str) -> Result<CompletePp3> {
        let found = self::digest(&text);
        ensure!(
            found == digest,
            "stored pp3 digest {found} differs from the recorded {digest}"
        );
        Ok(CompletePp3 {
            text,
            digest: found,
        })
    }

    fn of(text: String) -> CompletePp3 {
        let digest = self::digest(&text);
        CompletePp3 { text, digest }
    }

    /// The same processing with `[General] Rank` and `ColorLabel` set to `m`.
    pub fn with_mark(&self, m: Mark) -> Result<CompletePp3> {
        Ok(CompletePp3::of(with_mark(&self.text, m)?))
    }

    /// The same processing carrying the `Rank` and `ColorLabel` lines `existing` has, verbatim
    /// (a value another tool wrote is kept); a key `existing` lacks is left as it is here.
    pub fn carrying_mark_of(&self, existing: &str) -> Result<CompletePp3> {
        let lines: Vec<&str> = existing.lines().collect();
        let carried: Vec<(&str, String)> = section(&lines, "General")
            .map(|(start, end)| {
                ["Rank", "ColorLabel"]
                    .into_iter()
                    .filter_map(|key| {
                        lines[start + 1..end]
                            .iter()
                            .find_map(|l| key_value(l, key))
                            .map(|v| (key, v.to_owned()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(CompletePp3::of(with_keys(&self.text, "General", &carried)?))
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
}

/// `text` with every key of `partial` set: in its section where that exists, the section
/// appended otherwise.
fn with_partial(text: &str, partial: &Partial) -> Result<String> {
    let mut out = text.to_owned();
    for s in &partial.0 {
        let lines: Vec<&str> = out.lines().collect();
        if section(&lines, &s.name).is_some() {
            let keys: Vec<(&str, String)> = s
                .keys
                .iter()
                .map(|(k, v)| (k.as_str(), v.clone()))
                .collect();
            out = with_keys(&out, &s.name, &keys)?;
        } else {
            if !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&format!("\n[{}]\n", s.name));
            for (k, v) in &s.keys {
                out.push_str(&format!("{k}={v}\n"));
            }
        }
    }
    Ok(out)
}

/// `text` with each `key=value` set in `[section]`: replacing the key's line where present,
/// inserting it after the header otherwise. Same-named keys in other sections are untouched.
fn with_keys(text: &str, section_name: &str, values: &[(&str, String)]) -> Result<String> {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let borrowed: Vec<&str> = lines.iter().map(String::as_str).collect();
    let Some((start, mut end)) = section(&borrowed, section_name) else {
        bail!("pp3 has no [{section_name}] section");
    };
    let mut insert_at = start + 1;
    for (key, value) in values {
        let line = format!("{key}={value}");
        match (start + 1..end).find(|&i| key_value(&lines[i], key).is_some()) {
            Some(i) => lines[i] = line,
            None => {
                lines.insert(insert_at, line);
                insert_at += 1;
                end += 1;
            }
        }
    }
    let mut out = lines.join("\n");
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate::Stars;

    /// The head of a pp3 RawTherapee 5.13 writes when an unrated image is opened: no Rank line.
    const OPENED: &str = "[Version]\nAppVersion=5.13\nVersion=353\n\n[General]\nColorLabel=0\nInTrash=false\n\n[Exposure]\nAuto=false\n";

    fn mark(r: u8, l: Option<Label>) -> Mark {
        Mark {
            rating: Stars::new(r),
            label: l,
        }
    }

    #[test]
    fn opened_but_unrated_pp3_has_no_mark() {
        assert!(read_mark(OPENED).is_empty());
    }

    /// RawTherapee's rank 0 is "unrated", not a decision to protect.
    #[test]
    fn explicit_rank_zero_is_unrated() {
        assert!(read_mark("[General]\nRank=0\nColorLabel=0\n").is_empty());
    }

    #[test]
    fn sets_existing_and_inserts_missing_keys_in_general_only() {
        let out = with_mark(OPENED, mark(4, Some(Label::Green))).unwrap();
        assert_eq!(
            out,
            "[Version]\nAppVersion=5.13\nVersion=353\n\n[General]\nRank=4\nColorLabel=3\nInTrash=false\n\n[Exposure]\nAuto=false\n"
        );
        assert_eq!(
            read_mark(&out),
            FoundMark::from(mark(4, Some(Label::Green)))
        );
    }

    #[test]
    fn rewriting_replaces_rather_than_duplicates() {
        let once = with_mark(OPENED, mark(4, Some(Label::Green))).unwrap();
        let twice = with_mark(&once, mark(1, None)).unwrap();
        assert_eq!(twice.matches("Rank=").count(), 1);
        assert_eq!(read_mark(&twice), FoundMark::from(mark(1, None)));
    }

    /// Keys of the same name in other sections (e.g. a curve's own `Rank`) are not touched.
    #[test]
    fn same_key_outside_general_is_ignored() {
        let text = "[General]\nColorLabel=2\n\n[Other]\nRank=9\n";
        assert_eq!(read_mark(text).rating, None);
        let out = with_mark(text, mark(3, None)).unwrap();
        assert!(out.ends_with("[Other]\nRank=9\n"), "{out}");
    }

    #[test]
    fn foreign_colour_index_is_kept_verbatim() {
        let m = read_mark("[General]\nRank=2\nColorLabel=7\n");
        assert_eq!(m.label, Some(FoundLabel::Other("7".into())));
    }

    #[test]
    fn missing_general_section_is_an_error() {
        assert!(with_mark("[Exposure]\nAuto=false\n", mark(1, None)).is_err());
    }

    /// The second key is the section's last line, and the first key's insertion shifts it:
    /// it must still be found and replaced, not inserted again.
    #[test]
    fn insertion_does_not_hide_a_later_key_at_the_section_end() {
        let out = with_mark(
            "[General]\nInTrash=false\nColorLabel=1\n[Next]\n",
            mark(2, None),
        )
        .unwrap();
        assert_eq!(
            out,
            "[General]\nRank=2\nInTrash=false\nColorLabel=0\n[Next]\n"
        );
    }

    /// Exposure keys go into [Exposure], replacing RawTherapee's own values there.
    #[test]
    fn look_sets_exposure_keys() {
        let text = "[General]\nRank=1\n\n[Exposure]\nAuto=false\nCompensation=0\nHighlightCompr=0\n\n[Other]\nCompensation=7\n";
        let out = with_look(
            text,
            Look {
                compensation_ev: 1.333,
                highlight_compr: 100,
            },
        )
        .unwrap();
        assert_eq!(
            out,
            "[General]\nRank=1\n\n[Exposure]\nAuto=false\nCompensation=1.33\nHighlightCompr=100\n\n[Other]\nCompensation=7\n"
        );
    }

    /// Goal: a pp3 composed for a look option never loses processing the default profile
    /// holds. Method: compose over a default with several sections, then compare the
    /// composed text key by key (through `Partial::parse`, an independent reading of pp3
    /// text) against the default, the partial and the look.
    const DEFAULT: &str = "[Version]\nAppVersion=5.13\n\n[General]\nRank=1\nColorLabel=0\n\n[Exposure]\nAuto=false\nCompensation=0\nHighlightCompr=0\nHistogramMatching=true\nCurveMode=Standard\nCurve=0;\n\n[Directional Pyramid Denoising]\nEnabled=false\nChroma=15\n\n[RAW Bayer]\nMethod=amaze\n";

    const FILM: &str = "[Exposure]\nHistogramMatching=false\nCurveMode=FilmLike\nCurve=1;0;0;1;1;\n\n[Directional Pyramid Denoising]\nEnabled=true\n\n[PostDemosaicSharpening]\nEnabled=true\n";

    fn parsed(text: &str) -> Vec<(String, String, String)> {
        Partial::parse(text)
            .unwrap()
            .0
            .into_iter()
            .flat_map(|s| s.keys.into_iter().map(move |(k, v)| (s.name.clone(), k, v)))
            .collect()
    }

    fn look() -> Look {
        Look {
            compensation_ev: 1.5,
            highlight_compr: 100,
        }
    }

    #[test]
    fn composed_pp3_is_complete_and_carries_partial_and_look() {
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let partial = Partial::parse(FILM).unwrap();
        let composed = CompletePp3::compose(&default, &partial, Some(look())).unwrap();
        let out = parsed(composed.as_str());
        let value = |section: &str, key: &str| {
            out.iter()
                .find(|(s, k, _)| s == section && k == key)
                .map(|(_, _, v)| v.as_str())
        };
        // Every key of the default is still there ...
        for (s, k, _) in parsed(DEFAULT) {
            assert!(value(&s, &k).is_some(), "[{s}] {k} lost");
        }
        // ... with the partial's values where it has them, new sections appended ...
        assert_eq!(value("Exposure", "HistogramMatching"), Some("false"));
        assert_eq!(value("Exposure", "CurveMode"), Some("FilmLike"));
        assert_eq!(value("Exposure", "Curve"), Some("1;0;0;1;1;"));
        assert_eq!(
            value("Directional Pyramid Denoising", "Enabled"),
            Some("true")
        );
        assert_eq!(value("PostDemosaicSharpening", "Enabled"), Some("true"));
        // ... the look over everything ...
        assert_eq!(value("Exposure", "Compensation"), Some("1.50"));
        assert_eq!(value("Exposure", "HighlightCompr"), Some("100"));
        // ... and the default's other values untouched.
        assert_eq!(value("Directional Pyramid Denoising", "Chroma"), Some("15"));
        assert_eq!(value("RAW Bayer", "Method"), Some("amaze"));
        assert_eq!(value("General", "Rank"), Some("1"));
        assert_eq!(composed.digest(), digest(composed.as_str()));
    }

    /// The look is applied after the partial, so a partial that sets exposure keys cannot
    /// undo it.
    #[test]
    fn look_wins_over_the_partial() {
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let partial = Partial::parse("[Exposure]\nCompensation=-3\n").unwrap();
        let composed = CompletePp3::compose(&default, &partial, Some(look())).unwrap();
        assert!(composed.as_str().contains("Compensation=1.50\n"));
        assert!(!composed.as_str().contains("Compensation=-3"));
    }

    #[test]
    fn no_partial_and_no_look_is_the_default() {
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let composed = CompletePp3::compose(&default, &Partial::none(), None).unwrap();
        assert_eq!(composed.as_str(), DEFAULT);
    }

    #[test]
    fn partial_parse_refuses_what_is_not_a_pp3() {
        assert!(Partial::parse("Rank=1\n").is_err(), "key before a section");
        assert!(Partial::parse("[A]\nnot a key\n").is_err());
        assert!(Partial::parse("[A]\nx=1\n[A]\ny=2\n").is_err(), "twice");
        assert!(Partial::parse("").unwrap() == Partial::none());
    }

    #[test]
    fn carrying_a_mark_keeps_foreign_values_and_the_rest_of_the_processing() {
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let composed =
            CompletePp3::compose(&default, &Partial::parse(FILM).unwrap(), None).unwrap();
        let existing = "[General]\nRank=4\nColorLabel=7\n\n[Exposure]\nCompensation=9\n";
        let carried = composed.carrying_mark_of(existing).unwrap();
        assert!(carried.as_str().contains("Rank=4\nColorLabel=7\n"));
        assert!(carried.as_str().contains("CurveMode=FilmLike"));
        assert!(!carried.as_str().contains("Compensation=9"));
        // An existing pp3 without a rank leaves the composed one's as it is.
        let bare = composed
            .carrying_mark_of("[General]\nColorLabel=2\n")
            .unwrap();
        assert!(bare.as_str().contains("Rank=1\nColorLabel=2\n"));
    }

    #[test]
    fn a_stored_copy_is_accepted_only_if_its_digest_matches() {
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let c = CompletePp3::compose(&default, &Partial::none(), None).unwrap();
        let text = c.as_str().to_owned();
        assert_eq!(
            CompletePp3::from_stored(text.clone(), c.digest()).unwrap(),
            c
        );
        assert!(CompletePp3::from_stored(format!("{text}x"), c.digest()).is_err());
    }
}
