//! `DSC_xxxx.NEF.rpl`: rawpilot's record for one image, kept next to it so it travels with
//! the folder.
//!
//! Two parts. `written` is what rawpilot last put into RawTherapee's files for this image —
//! the rating and label, the look, and a hash of the pp3 as rawpilot left it — which is how
//! rawpilot tells its own values from the photographer's on the next run. `analysis` is what
//! the run measured and decided, kept for review and for comparison with the
//! photographer's final choices (`tools/compare.py`).
//!
//! The record states facts about files; it is derived, so a missing or unreadable `.rpl`
//! only means rawpilot cannot claim anything as its own, and treats every existing rating
//! and pp3 as the photographer's.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::cache::write_atomic;
use crate::config::OptionName;
use crate::facecrop::Box2d;
use crate::pp3::{self, Look};
use crate::sidecar::{FoundMark, Mark};

const FORMAT: u32 = 1;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Rpl {
    pub format: u32,
    #[serde(default)]
    pub written: Written,
    /// Free-form so the analysis can grow without a format change; see `pipeline::analyses`.
    #[serde(default)]
    pub analysis: Option<serde_json::Value>,
    /// Where the faces pass found faces in this frame, for the review renders to crop around
    /// without calling Gemini again. `None` when the pass has not read this frame; `Some`
    /// of an empty list when it read it and found no face. Kept across runs that do not run
    /// the pass (`--local-only`).
    #[serde(default)]
    pub faces: Option<Vec<FaceBox>>,
    /// The exposure look `process` last measured for this frame, which every render of it is
    /// made with (`looks::Job::new`), whether or not `process` could write it into the pp3.
    /// `None` when `process` found no correction worth making, has the look turned off, or
    /// has not run; `cull` leaves it as it is. Records from before the field was kept have
    /// none.
    #[serde(default)]
    pub measured_look: Option<Look>,
    /// The look option chosen in the review UI. While present, the pp3 is the
    /// photographer's, whatever its bytes.
    #[serde(default)]
    pub chosen: Option<Chosen>,
    /// The rating was set in the review UI. While present, it is the photographer's,
    /// whatever it holds.
    #[serde(default)]
    pub rated: Option<RatedInUi>,
}

/// One located face: who, and where in the displayed image.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FaceBox {
    /// Gemini's person id, consistent within a burst.
    pub person: usize,
    pub at: Box2d,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub option: OptionName,
    pub at: DateTime<Utc>,
    /// `pp3::processing_digest` of the pp3 the choice writes: rating the image later does not
    /// change it.
    pub processing_blake3: String,
}

/// A choice and how far its write got. `Pending` is recorded before the pp3 is touched, so
/// an interruption anywhere leaves the pp3 protected; `Applied` once the file was read back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Chosen {
    Pending(PendingChoice),
    Applied(Choice),
}

impl Chosen {
    pub fn choice(&self) -> &Choice {
        match self {
            Chosen::Pending(p) => &p.choice,
            Chosen::Applied(c) => c,
        }
    }
}

/// A choice being written, and what the pp3 was when it began.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingChoice {
    #[serde(flatten)]
    pub choice: Choice,
    #[serde(default)]
    pub prior: PriorPp3,
}

/// What stood at the pp3's path when a choice began writing. An interruption leaves the
/// file as it was or as the choice makes it; a file that is neither was edited since.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorPp3 {
    /// A record from before this was kept: no file is taken to be the one that was there.
    #[default]
    Unrecorded,
    NoFile,
    /// `pp3::processing_digest` of the file: a rating changed since (by a batch run or in
    /// the UI) leaves it the file that was there.
    File {
        processing_blake3: String,
    },
}

impl PriorPp3 {
    /// `existing` is the file's content, `None` when there is no file.
    pub fn of(existing: Option<&str>) -> PriorPp3 {
        match existing {
            Some(text) => PriorPp3::File {
                processing_blake3: pp3::processing_digest(text),
            },
            None => PriorPp3::NoFile,
        }
    }

    fn is_file_with(&self, text: &str) -> bool {
        match self {
            PriorPp3::File { processing_blake3 } => {
                *processing_blake3 == pp3::processing_digest(text)
            }
            PriorPp3::Unrecorded | PriorPp3::NoFile => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatedInUi {
    pub at: DateTime<Utc>,
}

pub fn now() -> DateTime<Utc> {
    DateTime::from(SystemTime::now())
}

/// Who made the processing in an image's pp3, as far as rawpilot can tell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pp3State {
    /// There is no pp3.
    Absent,
    /// Byte-identical to what a batch run wrote; nobody has changed it.
    Generated,
    /// Byte-identical to what the review UI wrote for this option.
    Chosen { option: OptionName },
    /// The review UI began writing this option and was interrupted; the file is still what
    /// it was when the choice began. Choosing again is safe: the interrupted choice already
    /// passed the hand-edit check for exactly these bytes.
    Interrupted { option: OptionName },
    /// A pp3 rawpilot cannot vouch for: edited since rawpilot or the UI wrote it, or written
    /// by RawTherapee or another tool. Replacing its processing needs the photographer's
    /// confirmation.
    HandEdited,
}

/// What rawpilot last wrote, and what it was about to write when a run was interrupted:
/// either one is recognised as rawpilot's own.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Written {
    pub committed: Option<Snapshot>,
    pub pending: Option<Snapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub mark: Mark,
    /// blake3 of the pp3 exactly as rawpilot wrote it; `None` when rawpilot wrote no pp3.
    pub pp3_blake3: Option<String>,
    pub look: Option<Look>,
}

pub fn path_for(nef: &Path) -> PathBuf {
    let mut p = nef.as_os_str().to_owned();
    p.push(".rpl");
    PathBuf::from(p)
}

impl Rpl {
    /// The record next to `nef`; empty when there is none or it cannot be read.
    pub fn load(nef: &Path) -> Rpl {
        let path = path_for(nef);
        match fs::read_to_string(&path) {
            Ok(text) => match serde_json::from_str::<Rpl>(&text) {
                Ok(r) => r,
                Err(e) => {
                    warn!(file = %path.display(), "unreadable, claiming nothing as rawpilot's: {e}");
                    Rpl::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Rpl::default(),
            Err(e) => {
                warn!(file = %path.display(), "unreadable, claiming nothing as rawpilot's: {e}");
                Rpl::default()
            }
        }
    }

    /// Whether a batch run may rewrite the processing of the pp3 holding `pp3_text`: it is
    /// byte-for-byte what rawpilot's batch left, and no choice made in the review UI claims
    /// it (the UI's pp3 can equal the batch's, so the bytes alone cannot tell).
    pub fn owns_pp3(&self, pp3_text: &str) -> bool {
        self.chosen.is_none() && self.written.owns_pp3(pp3_text)
    }

    /// `pp3_text` is the file's content, `None` when there is no file.
    pub fn pp3_state(&self, pp3_text: Option<&str>) -> Pp3State {
        let Some(text) = pp3_text else {
            return Pp3State::Absent;
        };
        match &self.chosen {
            Some(chosen) if pp3::processing_digest(text) == chosen.choice().processing_blake3 => {
                Pp3State::Chosen {
                    option: chosen.choice().option.clone(),
                }
            }
            Some(Chosen::Pending(p)) if p.prior.is_file_with(text) => Pp3State::Interrupted {
                option: p.choice.option.clone(),
            },
            Some(Chosen::Pending(_) | Chosen::Applied(_)) => Pp3State::HandEdited,
            None if self.written.owns_pp3(text) => Pp3State::Generated,
            None => Pp3State::HandEdited,
        }
    }

    pub fn save(&mut self, nef: &Path) -> Result<()> {
        self.format = FORMAT;
        let path = path_for(nef);
        write_atomic(&path, serde_json::to_string_pretty(self)?.as_bytes())
            .with_context(|| format!("writing {}", path.display()))
    }
}

impl Written {
    fn snapshots(&self) -> impl Iterator<Item = &Snapshot> {
        self.committed.iter().chain(self.pending.iter())
    }

    /// The stored rating and label are ones rawpilot wrote.
    pub fn owns_mark(&self, found: &FoundMark) -> bool {
        self.snapshots().any(|s| *found == FoundMark::from(s.mark))
    }

    /// The pp3 is byte-for-byte as rawpilot left it: nobody has edited it since.
    pub fn owns_pp3(&self, pp3_text: &str) -> bool {
        let d = pp3::digest(pp3_text);
        self.snapshots()
            .any(|s| s.pp3_blake3.as_deref() == Some(d.as_str()))
    }

    /// The look last written, if rawpilot wrote one.
    pub fn look(&self) -> Option<Look> {
        self.committed.as_ref().and_then(|s| s.look)
    }

    pub fn begin(&mut self, s: Snapshot) {
        self.pending = Some(s);
    }

    pub fn commit(&mut self) {
        let pending = self.pending.take().expect("commit follows begin");
        self.committed = Some(pending);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate::{Label, Stars};

    fn snap(r: u8, pp3: Option<&str>) -> Snapshot {
        Snapshot {
            mark: Mark {
                rating: Stars::new(r),
                label: Some(Label::Green),
            },
            pp3_blake3: pp3.map(pp3::digest),
            look: None,
        }
    }

    #[test]
    fn committed_and_pending_are_both_owned() {
        let mut w = Written::default();
        w.begin(snap(4, Some("a")));
        w.commit();
        w.begin(snap(5, Some("b")));
        assert!(w.owns_pp3("a"));
        assert!(w.owns_pp3("b"));
        assert!(!w.owns_pp3("a edited"));
        assert!(w.owns_mark(&FoundMark::from(snap(4, None).mark)));
        assert!(w.owns_mark(&FoundMark::from(snap(5, None).mark)));
        assert!(!w.owns_mark(&FoundMark::from(snap(3, None).mark)));
    }

    #[test]
    fn empty_record_owns_nothing() {
        let w = Written::default();
        assert!(!w.owns_pp3(""));
        assert!(!w.owns_mark(&FoundMark::default()));
    }

    /// Records written before `measured_look` was kept still load, without a look.
    #[test]
    fn a_record_without_a_measured_look_loads() {
        let dir = tempfile::tempdir().unwrap();
        let nef = dir.path().join("DSC_1.NEF");
        fs::write(
            path_for(&nef),
            r#"{"format":1,"written":{"committed":null,"pending":null},"analysis":null,"faces":[],"chosen":null,"rated":null}"#,
        )
        .unwrap();
        let r = Rpl::load(&nef);
        assert_eq!(r.measured_look, None);
        assert_eq!(r.faces, Some(vec![]), "the record was read, not defaulted");
    }

    /// The record lives beside the NEF, so moving the folder moves it too.
    #[test]
    fn round_trips_beside_the_nef_and_survives_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let nef = dir.path().join("DSC_1.NEF");
        let mut r = Rpl::default();
        r.written.begin(snap(4, Some("x")));
        r.written.commit();
        r.save(&nef).unwrap();
        assert!(dir.path().join("DSC_1.NEF.rpl").is_file());
        assert!(Rpl::load(&nef).written.owns_pp3("x"));
        fs::write(path_for(&nef), "not json").unwrap();
        assert!(!Rpl::load(&nef).written.owns_pp3("x"));
    }
}
