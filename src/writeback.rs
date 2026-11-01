//! Puts decided ratings (and, for a pp3 rawpilot owns, the look) where RawTherapee reads
//! them, and keeps each image's `.rpl` record.
//!
//! What rawpilot may change in a pp3 depends on who last touched it:
//! - no pp3: create one from the default profile, with the look and the rating;
//! - a pp3 byte-identical to what rawpilot's batch last wrote (`rpl::Rpl::owns_pp3`; one the
//!   review UI wrote is never owned, whatever its bytes): nobody has
//!   edited it, so rating and look are both brought up to date;
//! - any other pp3: the photographer's; only the rating may change, and only under the
//!   human-rating guard in `sidecar::plan`.
//!
//! Order of effects, so an interruption anywhere is resumable by re-running: the `.rpl`
//! records the pending values (and the pending pp3's hash) first, then the file is written
//! and read back, then the `.rpl` commits. A file holding either the pending or the
//! committed values is recognised as rawpilot's own on the next run.

use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use tracing::info;

use crate::cache::write_atomic;
use crate::lock::FolderLock;
use crate::pp3::{self, CompletePp3, Look, Partial};
use crate::rpl::{self, FaceBox, Rpl, Snapshot};
use crate::rtcli::DefaultProfiles;
use crate::shot::Shot;
use crate::sidecar::{self, FoundMark, Mark, Plan, RtOptions, SidecarStyle, Store};
use crate::xmp;

#[derive(Debug, Default)]
pub struct Outcome {
    pub written: usize,
    /// Of `written`, pp3s that did not exist and were created from the default profile.
    pub created: usize,
    pub unchanged: usize,
    pub kept_human: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Write,
    DryRun,
}

/// What a run knows about a shot's exposure look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Measure {
    /// The run does not measure looks (`cull`): the `.rpl`'s look and every pp3's processing
    /// stay as they are.
    Skipped,
    /// The look the run measured, `None` when there is no correction to make or the
    /// configuration turns the look off.
    Measured(Option<Look>),
}

impl Measure {
    /// The look to put into a pp3.
    fn for_pp3(self) -> Option<Look> {
        match self {
            Measure::Skipped | Measure::Measured(None) => None,
            Measure::Measured(Some(look)) => Some(look),
        }
    }
}

/// What rawpilot wants for one shot.
#[derive(Debug, Clone)]
pub struct Target<'a> {
    pub shot: &'a Shot,
    pub mark: Mark,
    /// Recorded in the `.rpl` for the renders, and written into a pp3 rawpilot creates or
    /// still owns.
    pub look: Measure,
    /// Stored in the `.rpl` as the run's record of this image.
    pub analysis: serde_json::Value,
    /// The faces the faces pass located in this frame, when it read it; `None` keeps what the
    /// `.rpl` already holds.
    pub faces: Option<Vec<FaceBox>>,
}

/// The pp3 store's decision for one image.
enum Pp3Action {
    Create,
    /// Replace the pp3 rawpilot owns with this text (rating and look up to date).
    Update(String),
    /// Set the rating in the photographer's pp3, leaving its processing alone.
    Rate(String),
    Leave(Plan),
}

pub fn apply(
    targets: &[Target],
    rt: &RtOptions,
    defaults: &DefaultProfiles,
    mode: Mode,
    _folder: &FolderLock,
) -> Result<Outcome> {
    let store = rt.store();
    let mut records: Vec<Rpl> = targets.iter().map(|t| Rpl::load(&t.shot.file)).collect();
    for (r, t) in records.iter_mut().zip(targets) {
        r.analysis = Some(t.analysis.clone());
        if let Some(faces) = &t.faces {
            r.faces = Some(faces.clone());
        }
        if let Measure::Measured(look) = t.look {
            r.measured_look = look;
        }
    }
    let mut outcome = Outcome::default();
    match store {
        Store::Xmp(style) => apply_xmp(targets, &mut records, style, mode, &mut outcome)?,
        Store::Pp3 => apply_pp3(targets, &mut records, defaults, mode, &mut outcome)?,
    }
    if let Mode::Write = mode {
        for (r, t) in records.iter_mut().zip(targets) {
            r.save(&t.shot.file)?;
        }
    }
    Ok(outcome)
}

fn count_plan(outcome: &mut Outcome, t: &Target, p: &Plan) {
    match p {
        Plan::Write => {}
        Plan::Unchanged => outcome.unchanged += 1,
        Plan::KeepHuman(found) => {
            outcome.kept_human += 1;
            info!(file = %t.shot.name(), ?found, "keeping rating rawpilot did not write");
        }
    }
}

fn apply_xmp(
    targets: &[Target],
    records: &mut [Rpl],
    style: SidecarStyle,
    mode: Mode,
    outcome: &mut Outcome,
) -> Result<()> {
    let paths: Vec<PathBuf> = targets
        .iter()
        .map(|t| sidecar::sidecar_path(&t.shot.file, style))
        .collect();
    let found = xmp::read_marks(&paths)?;
    let mut writes: Vec<usize> = Vec::new();
    for (i, t) in targets.iter().enumerate() {
        let now = found[i].clone().unwrap_or_default();
        let pp3 = sidecar::read_pp3_mark(&t.shot.file)?;
        let p = sidecar::plan(t.mark, &now, &pp3, &records[i]);
        count_plan(outcome, t, &p);
        if p == Plan::Write {
            writes.push(i);
        }
    }
    outcome.written = writes.len();
    if let Mode::DryRun = mode {
        return Ok(());
    }
    for &i in &writes {
        records[i].written.begin(Snapshot {
            mark: targets[i].mark,
            pp3_blake3: None,
            look: None,
        });
        records[i].save(&targets[i].shot.file)?;
    }
    let batch: Vec<(PathBuf, Mark)> = writes
        .iter()
        .map(|&i| (paths[i].clone(), targets[i].mark))
        .collect();
    xmp::write_marks(&batch)?;
    let back = xmp::read_marks(&paths)?;
    for &i in &writes {
        verify(
            &paths[i],
            back[i].clone().unwrap_or_default(),
            targets[i].mark,
        )?;
        records[i].written.commit();
    }
    Ok(())
}

fn apply_pp3(
    targets: &[Target],
    records: &mut [Rpl],
    defaults: &DefaultProfiles,
    mode: Mode,
    outcome: &mut Outcome,
) -> Result<()> {
    for (t, r) in targets.iter().zip(records.iter_mut()) {
        let action = decide_pp3(t, r)?;
        if let Pp3Action::Leave(p) = &action {
            count_plan(outcome, t, p);
            continue;
        }
        if let Pp3Action::Create = action {
            outcome.created += 1;
        }
        outcome.written += 1;
        if let Mode::DryRun = mode {
            info!(file = %t.shot.name(), mark = ?t.mark, look = ?t.look.for_pp3(), "would write");
            continue;
        }
        // The look recorded is the one in the file: ours where rawpilot owns the processing,
        // whatever was recorded before where it does not.
        let text = match &action {
            Pp3Action::Create => {
                CompletePp3::compose(defaults.get()?.as_ref(), &Partial::none(), t.look.for_pp3())?
                    .with_mark(t.mark)?
                    .as_str()
                    .to_owned()
            }
            Pp3Action::Update(text) | Pp3Action::Rate(text) => text.clone(),
            Pp3Action::Leave(_) => unreachable!("handled above"),
        };
        r.written
            .begin(snapshot_after(&action, &text, t, &r.written));
        r.save(&t.shot.file)?;
        let path = pp3::path_for(&t.shot.file);
        write_atomic(&path, text.as_bytes())?;
        verify(&path, sidecar::read_pp3_mark(&t.shot.file)?, t.mark)?;
        r.written.commit();
    }
    Ok(())
}

/// The record of a pp3 write. Only a pp3 whose processing is rawpilot's (created or owned)
/// is recorded by hash: after a rating-only write into the photographer's pp3, the file
/// still belongs to the photographer, and recording its new hash would hand its processing
/// to rawpilot on the next run.
fn snapshot_after(action: &Pp3Action, text: &str, t: &Target, before: &rpl::Written) -> Snapshot {
    let (pp3_blake3, look) = match action {
        Pp3Action::Create => (Some(pp3::digest(text)), t.look.for_pp3()),
        Pp3Action::Update(_) => (Some(pp3::digest(text)), t.look.for_pp3().or(before.look())),
        Pp3Action::Rate(_) => (None, None),
        Pp3Action::Leave(_) => unreachable!("nothing is written"),
    };
    Snapshot {
        mark: t.mark,
        pp3_blake3,
        look,
    }
}

fn decide_pp3(t: &Target, r: &Rpl) -> Result<Pp3Action> {
    let Some(text) = sidecar::read_pp3(&t.shot.file)? else {
        return Ok(Pp3Action::Create);
    };
    decide_existing_pp3(&text, t.mark, t.look.for_pp3(), r)
}

/// For an existing pp3 `text`. A look of `None` leaves an owned pp3's exposure as it is:
/// the default profile's values are not known here to restore.
fn decide_existing_pp3(text: &str, mark: Mark, look: Option<Look>, r: &Rpl) -> Result<Pp3Action> {
    if r.owns_pp3(text) {
        let looked = match look {
            Some(look) => pp3::with_look(text, look)?,
            None => text.to_owned(),
        };
        // A rating set in the review UI stays as it is in the file.
        let desired = match r.rated {
            Some(_) => looked,
            None => pp3::with_mark(&looked, mark)?,
        };
        return Ok(if desired == text {
            Pp3Action::Leave(Plan::Unchanged)
        } else {
            Pp3Action::Update(desired)
        });
    }
    // In procparams mode RawTherapee ignores XMP, so nothing elsewhere protects a file.
    let current = pp3::read_mark(text);
    Ok(
        match sidecar::plan(mark, &current, &FoundMark::default(), r) {
            Plan::Write => Pp3Action::Rate(pp3::with_mark(text, mark)?),
            p @ (Plan::Unchanged | Plan::KeepHuman(_)) => Pp3Action::Leave(p),
        },
    )
}

/// The rating each of `nefs` holds now in the store RawTherapee reads, 0 for none.
pub fn current_ratings(nefs: &[&Path], rt: &RtOptions) -> Result<Vec<i64>> {
    Ok(current_marks(nefs, rt)?
        .iter()
        .map(|m| m.rating.unwrap_or(0))
        .collect())
}

/// The rating and label each of `nefs` holds now in the store RawTherapee reads.
pub fn current_marks(nefs: &[&Path], rt: &RtOptions) -> Result<Vec<FoundMark>> {
    Ok(match rt.store() {
        Store::Pp3 => nefs
            .iter()
            .map(|nef| sidecar::read_pp3_mark(nef))
            .collect::<Result<_>>()?,
        Store::Xmp(style) => {
            let paths: Vec<PathBuf> = nefs
                .iter()
                .map(|nef| sidecar::sidecar_path(nef, style))
                .collect();
            xmp::read_marks(&paths)?
                .into_iter()
                .map(Option::unwrap_or_default)
                .collect()
        }
    })
}

/// Postcondition: the file reads back exactly as written.
pub(crate) fn verify(path: &Path, found: FoundMark, m: Mark) -> Result<()> {
    let expected = FoundMark::from(m);
    ensure!(
        found == expected,
        "{}: wrote {expected:?}, read back {found:?}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate::{Label, Stars};

    const PP3: &str =
        "[General]\nRank=1\nColorLabel=3\n\n[Exposure]\nCompensation=0\nHighlightCompr=0\n";

    fn mark(r: u8) -> Mark {
        Mark {
            rating: Stars::new(r),
            label: Some(Label::Green),
        }
    }

    fn look(ev: f64) -> Look {
        Look {
            compensation_ev: ev,
            highlight_compr: 100,
        }
    }

    /// A record claiming `text` as written by rawpilot with `m`.
    fn owning(text: &str, m: Mark) -> Rpl {
        let mut r = Rpl::default();
        r.written.begin(Snapshot {
            mark: m,
            pp3_blake3: Some(pp3::digest(text)),
            look: None,
        });
        r.written.commit();
        r
    }

    #[test]
    fn owned_pp3_gets_rating_and_look() {
        let r = owning(PP3, mark(1));
        let Pp3Action::Update(out) =
            decide_existing_pp3(PP3, mark(4), Some(look(1.0)), &r).unwrap()
        else {
            panic!("expected an update");
        };
        assert!(out.contains("Rank=4"));
        assert!(out.contains("Compensation=1.00"));
    }

    #[test]
    fn owned_pp3_already_current_is_left() {
        let current = pp3::with_mark(&pp3::with_look(PP3, look(1.0)).unwrap(), mark(4)).unwrap();
        let r = owning(&current, mark(4));
        assert!(matches!(
            decide_existing_pp3(&current, mark(4), Some(look(1.0)), &r).unwrap(),
            Pp3Action::Leave(Plan::Unchanged)
        ));
    }

    /// The photographer edited the processing (the hash no longer matches) but not the
    /// rating rawpilot wrote: the rating may change, the processing may not.
    #[test]
    fn edited_pp3_gets_rating_only() {
        let r = owning(PP3, mark(1));
        let edited = PP3.replace("HighlightCompr=0", "HighlightCompr=40");
        let Pp3Action::Rate(out) =
            decide_existing_pp3(&edited, mark(4), Some(look(1.0)), &r).unwrap()
        else {
            panic!("expected a rating-only write");
        };
        assert!(out.contains("Rank=4"));
        assert!(out.contains("Compensation=0\n"));
        assert!(out.contains("HighlightCompr=40"));
    }

    /// Re-rated by the photographer: nothing changes.
    #[test]
    fn human_rating_is_kept() {
        let r = owning(PP3, mark(1));
        let rerated = PP3.replace("Rank=1", "Rank=5");
        assert!(matches!(
            decide_existing_pp3(&rerated, mark(4), Some(look(1.0)), &r).unwrap(),
            Pp3Action::Leave(Plan::KeepHuman(_))
        ));
    }

    /// Two runs over the photographer's pp3: the first sets only the rating, and the second
    /// must still treat the processing as theirs — no look, no further write.
    #[test]
    fn rating_only_write_does_not_take_over_the_pp3() {
        let theirs = PP3
            .replace("Rank=1", "Rank=0")
            .replace("ColorLabel=3", "ColorLabel=0");
        let shot = crate::shot::Shot {
            file: "x.NEF".into(),
            file_size: 0,
            taken_at: chrono::DateTime::parse_from_rfc3339("2026-10-18T14:00:00+00:00").unwrap(),
            preview: crate::shot::PreviewSpan { offset: 0, len: 0 },
            af: None,
            orientation: crate::shot::Orientation::Normal,
            exposure: crate::shot::Exposure {
                shutter_s: 0.01,
                focal_mm: 50.0,
                iso: 100,
                picture_control: None,
                compensation_ev: None,
            },
        };
        let t = Target {
            shot: &shot,
            mark: mark(4),
            look: Measure::Measured(Some(look(1.0))),
            analysis: serde_json::Value::Null,
            faces: None,
        };
        let mut r = Rpl::default();
        let first = decide_existing_pp3(&theirs, t.mark, t.look.for_pp3(), &r).unwrap();
        let Pp3Action::Rate(text) = &first else {
            panic!("first run should set the rating only");
        };
        r.written
            .begin(snapshot_after(&first, text, &t, &r.written));
        r.written.commit();
        let second = decide_existing_pp3(text, t.mark, t.look.for_pp3(), &r).unwrap();
        assert!(
            matches!(second, Pp3Action::Leave(Plan::Unchanged)),
            "second run must leave the photographer's pp3 alone"
        );
    }

    /// With no `.rpl` (or an unreadable one), an existing pp3 is the photographer's.
    #[test]
    fn without_a_record_processing_is_theirs() {
        let Pp3Action::Rate(out) = decide_existing_pp3(
            &PP3.replace("Rank=1", "Rank=0")
                .replace("ColorLabel=3", "ColorLabel=0"),
            mark(4),
            Some(look(1.0)),
            &Rpl::default(),
        )
        .unwrap() else {
            panic!("expected a rating-only write");
        };
        assert!(out.contains("Compensation=0\n"));
    }

    fn chosen(text: &str) -> rpl::Chosen {
        rpl::Chosen::Applied(rpl::Choice {
            option: "Camera".to_owned().try_into().unwrap(),
            at: rpl::now(),
            processing_blake3: pp3::processing_digest(text),
        })
    }

    /// Goal: a pp3 the review UI wrote is the photographer's even when its bytes are exactly
    /// what the batch wrote (picking Camera). Method: a record that owns `PP3` by digest,
    /// with and without a UI choice, run through the batch's decision.
    #[test]
    fn a_ui_choice_is_not_owned_even_when_the_bytes_equal_the_batchs() {
        let mut r = owning(PP3, mark(1));
        assert!(matches!(
            decide_existing_pp3(PP3, mark(4), Some(look(1.0)), &r).unwrap(),
            Pp3Action::Update(_)
        ));
        r.chosen = Some(chosen(PP3));
        let Pp3Action::Rate(out) = decide_existing_pp3(PP3, mark(4), Some(look(1.0)), &r).unwrap()
        else {
            panic!("expected a rating-only write");
        };
        assert!(out.contains("Rank=4"));
        assert!(out.contains("Compensation=0\n"), "processing untouched");
        assert!(!out.contains("HighlightCompr=100"));
    }

    /// A rating set in the UI is kept by the batch, wherever the batch finds it: in a pp3
    /// the batch owns (which still gets its look), in one it does not, and as an unrated
    /// zero.
    #[test]
    fn a_ui_rating_survives_the_batch() {
        let mut r = owning(PP3, mark(1));
        r.rated = Some(rpl::RatedInUi { at: rpl::now() });
        let Pp3Action::Update(out) =
            decide_existing_pp3(PP3, mark(4), Some(look(1.0)), &r).unwrap()
        else {
            panic!("an owned pp3 still takes the look");
        };
        assert!(out.contains("Rank=1"), "rating kept: {out}");
        assert!(out.contains("Compensation=1.00"));

        let edited = PP3.replace("HighlightCompr=0", "HighlightCompr=40");
        assert!(matches!(
            decide_existing_pp3(&edited, mark(4), Some(look(1.0)), &r).unwrap(),
            Pp3Action::Leave(Plan::KeepHuman(_))
        ));

        let unrated = PP3
            .replace("Rank=1", "Rank=0")
            .replace("ColorLabel=3", "ColorLabel=0");
        assert!(matches!(
            decide_existing_pp3(&unrated, mark(4), Some(look(1.0)), &r).unwrap(),
            Pp3Action::Leave(Plan::KeepHuman(_))
        ));
    }

    /// `cull` passes no look: an owned pp3's processing is left exactly as it is.
    #[test]
    fn a_batch_without_a_look_leaves_an_owned_pp3s_processing() {
        let r = owning(PP3, mark(1));
        let Pp3Action::Update(out) = decide_existing_pp3(PP3, mark(4), None, &r).unwrap() else {
            panic!("the rating still updates");
        };
        assert_eq!(out, PP3.replace("Rank=1", "Rank=4"));
    }

    fn shot_at(file: PathBuf) -> crate::shot::Shot {
        crate::shot::Shot {
            file,
            file_size: 1,
            taken_at: chrono::DateTime::parse_from_rfc3339("2026-10-18T14:00:00+00:00").unwrap(),
            preview: crate::shot::PreviewSpan { offset: 0, len: 0 },
            af: None,
            orientation: crate::shot::Orientation::Normal,
            exposure: crate::shot::Exposure {
                shutter_s: 0.01,
                focal_mm: 50.0,
                iso: 100,
                picture_control: None,
                compensation_ev: None,
            },
        }
    }

    /// Goal: the look the renders are made with is the look `process` measured, recorded
    /// whether or not it could write the look into the pp3, and `cull` leaves it be.
    /// Method: the real `apply` over a folder holding one NEF stand-in whose pp3 is the
    /// photographer's (so the look cannot be written), then a `cull` and a `process` that
    /// finds no correction. Mutation: record the look only when the pp3 is written: the
    /// first assertion goes red; let `Skipped` clear it: the second does.
    #[test]
    fn the_measured_look_is_recorded_whether_or_not_the_pp3_takes_it() {
        let dir = tempfile::tempdir().unwrap();
        let nef = dir.path().join("DSC_1.NEF");
        std::fs::write(&nef, b"x").unwrap();
        let theirs = PP3
            .replace("Rank=1", "Rank=0")
            .replace("ColorLabel=3", "ColorLabel=0");
        std::fs::write(pp3::path_for(&nef), &theirs).unwrap();
        let rt = RtOptions {
            sidecar_style: SidecarStyle::Ext,
            rank_source: sidecar::RankSource::ProcParams,
            custom_profile_builder: false,
            config_dir: dir.path().to_owned(),
        };
        let defaults = DefaultProfiles::resolving_with(|| unreachable!("a pp3 exists"));
        let folder = FolderLock::acquire(dir.path(), crate::lock::Run::Process).unwrap();
        let shot = shot_at(nef.clone());
        let run = |look: Measure| {
            let target = Target {
                shot: &shot,
                mark: mark(4),
                look,
                analysis: serde_json::Value::Null,
                faces: None,
            };
            apply(&[target], &rt, &defaults, Mode::Write, &folder).unwrap();
            Rpl::load(&nef).measured_look
        };

        assert_eq!(run(Measure::Measured(Some(look(1.0)))), Some(look(1.0)));
        assert_eq!(
            std::fs::read_to_string(pp3::path_for(&nef)).unwrap(),
            PP3.replace("Rank=1", "Rank=4"),
            "only the rating is written; the photographer's processing is untouched"
        );
        assert_eq!(run(Measure::Skipped), Some(look(1.0)), "cull keeps it");
        assert_eq!(run(Measure::Measured(None)), None, "process found none");
    }
}
