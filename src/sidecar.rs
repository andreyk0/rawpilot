//! Where ratings are stored, and whether rawpilot may touch a stored rating.
//!
//! RawTherapee reads ranks from one of two places, chosen by its `ThumbnailRankColorMode`;
//! rawpilot writes to the same one. It may overwrite a rating and label only when they are
//! absent or equal to values rawpilot itself wrote, as recorded in the image's `.rpl`
//! (`rpl::Written`), and never one set in the review UI (`rpl::Rpl::rated`).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::rate::{Label, Stars};
use crate::rpl::Rpl;

/// RawTherapee's `[Metadata] XMPSidecarStyle`: `std` = `DSC_1.xmp`, `ext` = `DSC_1.NEF.xmp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarStyle {
    Std,
    Ext,
}

/// Where RawTherapee takes a thumbnail's rank and colour from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankSource {
    Xmp,
    ProcParams,
}

pub struct RtOptions {
    pub sidecar_style: SidecarStyle,
    pub rank_source: RankSource,
    /// `CustomProfileBuilderPath` is set: an external program picks each image's profile.
    pub custom_profile_builder: bool,
    /// The directory holding `options`, where `dynamicprofile.cfg` also lives.
    pub config_dir: PathBuf,
}

/// Where rawpilot puts ratings: wherever RawTherapee will read them from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Store {
    Xmp(SidecarStyle),
    Pp3,
}

impl RtOptions {
    pub fn store(&self) -> Store {
        match self.rank_source {
            RankSource::Xmp => Store::Xmp(self.sidecar_style),
            RankSource::ProcParams => Store::Pp3,
        }
    }
}

pub fn default_rt_options_path() -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .context("neither XDG_CONFIG_HOME nor HOME is set")?;
    Ok(base.join("RawTherapee").join("options"))
}

pub fn read_rt_options(path: &Path) -> Result<RtOptions> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading RawTherapee options {}", path.display()))?;
    let config_dir = path
        .parent()
        .context("RawTherapee options path has no directory")?;
    parse_rt_options(&text, config_dir).with_context(|| format!("in {}", path.display()))
}

fn ini_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim() == key)
        .map(|(_, v)| v.trim())
}

fn parse_rt_options(text: &str, config_dir: &Path) -> Result<RtOptions> {
    let sidecar_style = match ini_value(text, "XMPSidecarStyle") {
        Some("std") => SidecarStyle::Std,
        Some("ext") => SidecarStyle::Ext,
        other => bail!("unrecognised XMPSidecarStyle {other:?}"),
    };
    let rank_source = match ini_value(text, "ThumbnailRankColorMode") {
        Some("xmp") => RankSource::Xmp,
        Some("procparams") => RankSource::ProcParams,
        other => bail!("unrecognised ThumbnailRankColorMode {other:?}"),
    };
    Ok(RtOptions {
        sidecar_style,
        rank_source,
        custom_profile_builder: ini_value(text, "CustomProfileBuilderPath")
            .is_some_and(|v| !v.is_empty()),
        config_dir: config_dir.to_owned(),
    })
}

pub fn sidecar_path(nef: &Path, style: SidecarStyle) -> PathBuf {
    match style {
        SidecarStyle::Std => nef.with_extension("xmp"),
        SidecarStyle::Ext => {
            let mut p = nef.as_os_str().to_owned();
            p.push(".xmp");
            PathBuf::from(p)
        }
    }
}

/// A rating and label as rawpilot writes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mark {
    pub rating: Stars,
    pub label: Option<Label>,
}

/// A stored colour label: one of RawTherapee's five, or whatever another tool wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoundLabel {
    Known(Label),
    Other(String),
}

/// What a store holds now. Other tools write values rawpilot never would (`Rating=-1`, label
/// names other than RawTherapee's), so those are kept verbatim.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoundMark {
    pub rating: Option<i64>,
    pub label: Option<FoundLabel>,
}

impl FoundMark {
    /// No decision recorded: rating 0 is how both stores say "unrated".
    pub fn is_empty(&self) -> bool {
        self.rating.is_none_or(|r| r == 0) && self.label.is_none()
    }

    fn equals(&self, m: Mark) -> bool {
        *self == FoundMark::from(m)
    }
}

/// How a store reads back after rawpilot writes `m`.
impl From<Mark> for FoundMark {
    fn from(m: Mark) -> FoundMark {
        FoundMark {
            rating: Some(i64::from(m.rating.get())),
            label: m.label.map(FoundLabel::Known),
        }
    }
}

/// The `xmp:Label` value RawTherapee writes and reads for `l`.
pub fn xmp_label(l: Label) -> &'static str {
    match l {
        Label::Red => "Red",
        Label::Yellow => "Yellow",
        Label::Green => "Green",
        Label::Blue => "Blue",
        Label::Purple => "Purple",
    }
}

pub fn parse_xmp_label(s: &str) -> FoundLabel {
    [
        Label::Red,
        Label::Yellow,
        Label::Green,
        Label::Blue,
        Label::Purple,
    ]
    .into_iter()
    .find(|&l| xmp_label(l) == s)
    .map_or_else(|| FoundLabel::Other(s.to_owned()), FoundLabel::Known)
}

/// The mark in `nef`'s pp3, or empty when it has none.
pub fn read_pp3_mark(nef: &Path) -> Result<FoundMark> {
    Ok(read_pp3(nef)?
        .map(|t| crate::pp3::read_mark(&t))
        .unwrap_or_default())
}

pub fn read_pp3(nef: &Path) -> Result<Option<String>> {
    let path = crate::pp3::path_for(nef);
    match fs::read_to_string(&path) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    Write,
    /// Already holds exactly the target values.
    Unchanged,
    /// Holds a rating or label rawpilot did not write.
    KeepHuman(FoundMark),
}

/// `current` is what the store holds. `elsewhere` is a mark in the other store that still
/// counts as a human decision while `current` is empty: a rank in a pp3 when writing XMP is
/// how a rating made before switching RawTherapee to `xmp` mode looks. A rating set in the
/// review UI (`record.rated`) is the photographer's whatever it holds, even none.
pub fn plan(target: Mark, current: &FoundMark, elsewhere: &FoundMark, record: &Rpl) -> Plan {
    if current.equals(target) {
        return Plan::Unchanged;
    }
    if record.rated.is_some() {
        return Plan::KeepHuman(current.clone());
    }
    if current.is_empty() {
        return if elsewhere.is_empty() {
            Plan::Write
        } else {
            Plan::KeepHuman(elsewhere.clone())
        };
    }
    if record.written.owns_mark(current) {
        Plan::Write
    } else {
        Plan::KeepHuman(current.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark(r: u8, l: Option<Label>) -> Mark {
        Mark {
            rating: Stars::new(r),
            label: l,
        }
    }

    fn found(r: Option<i64>, l: Option<&str>) -> FoundMark {
        FoundMark {
            rating: r,
            label: l.map(parse_xmp_label),
        }
    }

    #[test]
    fn sidecar_names_follow_rt_style() {
        let nef = Path::new("/p/DSC_3194.NEF");
        assert_eq!(
            sidecar_path(nef, SidecarStyle::Std),
            Path::new("/p/DSC_3194.xmp")
        );
        assert_eq!(
            sidecar_path(nef, SidecarStyle::Ext),
            Path::new("/p/DSC_3194.NEF.xmp")
        );
    }

    /// Lines as they appear in RawTherapee 5.13's options file.
    #[test]
    fn parses_rt_options() {
        let o = parse_rt_options(
            "[File Browser]\nThumbnailRankColorMode=procparams\n[Metadata]\nXMPSidecarStyle=std\n\
             [Profiles]\nCustomProfileBuilderPath=\n",
            Path::new("/rt"),
        )
        .unwrap();
        assert_eq!(o.sidecar_style, SidecarStyle::Std);
        assert_eq!(o.rank_source, RankSource::ProcParams);
        assert!(!o.custom_profile_builder);
        assert_eq!(o.store(), Store::Pp3);
    }

    #[test]
    fn empty_sidecar_is_written() {
        let p = plan(
            mark(3, Some(Label::Yellow)),
            &found(None, None),
            &found(None, None),
            &Rpl::default(),
        );
        assert_eq!(p, Plan::Write);
    }

    #[test]
    fn foreign_rating_is_kept() {
        let p = plan(
            mark(3, None),
            &found(Some(5), None),
            &found(None, None),
            &Rpl::default(),
        );
        assert_eq!(p, Plan::KeepHuman(found(Some(5), None)));
    }

    #[test]
    fn pp3_rank_protects_a_file_without_xmp_mark() {
        let p = plan(
            mark(3, None),
            &found(None, None),
            &found(Some(4), None),
            &Rpl::default(),
        );
        assert_eq!(p, Plan::KeepHuman(found(Some(4), None)));
    }

    #[test]
    fn own_committed_or_pending_value_is_overwritten() {
        let snap = |m| crate::rpl::Snapshot {
            mark: m,
            pp3_blake3: None,
            look: None,
        };
        let mut l = Rpl::default();
        l.written.begin(snap(mark(1, None)));
        l.written.commit();
        l.written.begin(snap(mark(3, Some(Label::Yellow))));
        let target = mark(4, Some(Label::Green));
        for xmp in [found(Some(1), None), found(Some(3), Some("Yellow"))] {
            assert_eq!(plan(target, &xmp, &found(None, None), &l), Plan::Write);
        }
        // A human change away from both is kept.
        let human = found(Some(3), None);
        assert_eq!(
            plan(target, &human, &found(None, None), &l),
            Plan::KeepHuman(human.clone())
        );
    }

    #[test]
    fn same_value_is_unchanged() {
        let p = plan(
            mark(2, Some(Label::Purple)),
            &found(Some(2), Some("Purple")),
            &found(None, None),
            &Rpl::default(),
        );
        assert_eq!(p, Plan::Unchanged);
    }

    #[test]
    fn pp3_general_section_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let nef = dir.path().join("DSC_1.NEF");
        fs::write(
            dir.path().join("DSC_1.NEF.pp3"),
            "[Version]\nAppVersion=5.13\n\n[General]\nRank=3\nColorLabel=0\nInTrash=false\n\n[Exposure]\nAuto=false\n",
        )
        .unwrap();
        assert_eq!(read_pp3_mark(&nef).unwrap(), found(Some(3), None));
        assert_eq!(
            read_pp3_mark(&dir.path().join("none.NEF")).unwrap(),
            FoundMark::default()
        );
    }
}
