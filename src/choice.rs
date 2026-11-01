//! What the review UI does to a shot: pick one of its pre-rendered look options, and set its
//! rating. Both are the photographer's explicit decisions, so both are recorded in the
//! image's `.rpl` (`rpl::Chosen`, `rpl::RatedInUi`) and no batch command (`process`, `cull`)
//! changes the pp3 or the rating afterwards, even when the bytes equal what the batch would
//! write: picking `Camera` writes rawpilot's own default pp3.
//!
//! # API
//!
//! - `Session::pp3_state(nef)`: who made the processing now in the pp3
//!   (`rpl::Pp3State`). `HandEdited` means the photographer's own RawTherapee edits may be
//!   lost by `choose`, which the UI asks about first.
//! - `Session::choose(nef, render, OnHandEdited)`: writes the pp3 that was rendered
//!   (`looks::Fresh`, a render compared with today's inputs; its `pp3` is complete) and
//!   records the choice. Over a hand-edited
//!   pp3 it does nothing and answers `Choose::NeedsConfirmation` unless called with
//!   `OnHandEdited::Replace`. The pp3's `Rank` and `ColorLabel` lines are kept.
//! - `Session::rate(nef, stars)`: writes the rating where RawTherapee reads it (the pp3's
//!   `[General]` or the XMP sidecar, per `ThumbnailRankColorMode`) and records it. The colour
//!   label is the one the file holds at that moment. A pp3 that does not exist is created
//!   from the default profile, without the exposure look.
//!
//! # Interruption
//!
//! Each call records its intent in the `.rpl` first, writes the file, reads it back, and
//! records completion. A process that dies before completion leaves `Chosen::Pending`, which
//! holds the digest of the pp3 that was there when the choice began; the pp3 stays protected
//! from the batch. While the file still has that digest `Pp3State::Interrupted` reports it
//! and choosing again finishes it without asking; a file edited since is `HandEdited`, asked
//! about like any other. A choice that returns an error puts the record back as it was. A
//! rating that stops early is recorded as rated, and the batch keeps whatever the store then
//! holds. Running the call again completes either.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use tracing::warn;

use crate::cache::write_atomic;
use crate::config::OptionName;
use crate::lock::FolderLock;
use crate::looks::{Fresh, Store as LooksStore};
use crate::pp3::{self, CompletePp3, Partial};
use crate::rate::{Label, Stars};
use crate::rpl::{
    self, Choice, Chosen, PendingChoice, Pp3State, PriorPp3, RatedInUi, Rpl, Snapshot,
};
use crate::rtcli::DefaultProfiles;
use crate::sidecar::{self, FoundLabel, Mark, RtOptions, Store};
use crate::{writeback, xmp};

/// What `choose` does with a pp3 that may hold the photographer's RawTherapee edits
/// (`Pp3State::HandEdited`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnHandEdited {
    /// Leave it and say so: the UI asks the photographer first.
    Refuse,
    /// The photographer confirmed: replace the processing.
    Replace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choose {
    Applied,
    /// The pp3 is hand-edited and the call was `OnHandEdited::Refuse`; nothing was written.
    NeedsConfirmation,
}

pub struct Session {
    looks: LooksStore,
    rt: RtOptions,
    defaults: DefaultProfiles,
}

impl Session {
    /// A session in the folder `_folder` holds; the caller keeps the lock for the session's
    /// life.
    pub fn new(
        looks: LooksStore,
        rt: RtOptions,
        defaults: DefaultProfiles,
        _folder: &FolderLock,
    ) -> Session {
        Session {
            looks,
            rt,
            defaults,
        }
    }

    pub fn looks(&self) -> &LooksStore {
        &self.looks
    }

    /// The shoot's default profile, shared with whoever renders.
    pub fn defaults(&self) -> &DefaultProfiles {
        &self.defaults
    }

    /// Whether rating a shot that has no pp3 creates one, from the default profile.
    pub fn rating_creates_pp3(&self) -> bool {
        match self.rt.store() {
            Store::Pp3 => true,
            Store::Xmp(_) => false,
        }
    }

    pub fn pp3_state(&self, nef: &Path) -> Result<Pp3State> {
        let existing = sidecar::read_pp3(nef)?;
        Ok(Rpl::load(nef).pp3_state(existing.as_deref()))
    }

    /// Writes `render`, a render of `nef` that was compared with today's inputs.
    pub fn choose(&self, nef: &Path, render: &Fresh, on: OnHandEdited) -> Result<Choose> {
        let option = &render.manifest.option;
        let rendered_pp3 = render.pp3()?;
        let existing = sidecar::read_pp3(nef)?;
        let mut record = Rpl::load(nef);
        if record.pp3_state(existing.as_deref()) == Pp3State::HandEdited
            && on == OnHandEdited::Refuse
        {
            return Ok(Choose::NeedsConfirmation);
        }
        let to_write = match &existing {
            Some(text) => rendered_pp3.carrying_mark_of(text)?,
            None => rendered_pp3,
        };
        let choice = Choice {
            option: option.clone(),
            at: rpl::now(),
            processing_blake3: pp3::processing_digest(to_write.as_str()),
        };
        let previous = record.chosen.replace(Chosen::Pending(PendingChoice {
            choice: choice.clone(),
            prior: PriorPp3::of(existing.as_deref()),
        }));
        record.save(nef)?;

        if let Err(e) = write_and_verify(nef, option, &to_write) {
            // Nothing is Pending any more: whatever the file holds, the record says what it
            // said before this call.
            record.chosen = previous;
            if let Err(restore) = record.save(nef) {
                warn!("{}: cannot restore the record: {restore:#}", nef.display());
            }
            return Err(e);
        }

        record.chosen = Some(Chosen::Applied(choice));
        record.save(nef)?;
        Ok(Choose::Applied)
    }

    /// The colour label the store holds now, read at write time: a label set in RawTherapee
    /// since the session began is kept. A label another tool wrote that is not one of
    /// RawTherapee's five is not carried.
    fn label_now(&self, nef: &Path) -> Result<Option<Label>> {
        let found = writeback::current_marks(&[nef], &self.rt)?;
        Ok(match found.into_iter().next().and_then(|m| m.label) {
            Some(FoundLabel::Known(label)) => Some(label),
            Some(FoundLabel::Other(_)) | None => None,
        })
    }

    /// Sets the rating to `stars`, keeping the label the store holds.
    pub fn rate(&self, nef: &Path, stars: Stars) -> Result<()> {
        let mark = Mark {
            rating: stars,
            label: self.label_now(nef)?,
        };
        let mut record = Rpl::load(nef);
        match self.rt.store() {
            Store::Pp3 => self.rate_in_pp3(nef, &mut record, mark),
            Store::Xmp(style) => {
                let path = sidecar::sidecar_path(nef, style);
                record.written.begin(Snapshot {
                    mark,
                    pp3_blake3: None,
                    look: None,
                });
                record.rated = Some(RatedInUi { at: rpl::now() });
                record.save(nef)?;
                xmp::write_marks(&[(path.clone(), mark)])?;
                let back = xmp::read_marks(std::slice::from_ref(&path))?;
                writeback::verify(
                    &path,
                    back.into_iter().next().flatten().unwrap_or_default(),
                    mark,
                )?;
                record.written.commit();
                record.save(nef)
            }
        }
    }

    fn rate_in_pp3(&self, nef: &Path, record: &mut Rpl, mark: Mark) -> Result<()> {
        let existing = sidecar::read_pp3(nef)?;
        let state = record.pp3_state(existing.as_deref());
        let text = match &existing {
            Some(text) => pp3::with_mark(text, mark)?,
            None => CompletePp3::compose(self.defaults.get()?.as_ref(), &Partial::none(), None)?
                .with_mark(mark)?
                .as_str()
                .to_owned(),
        };
        // As in a batch rating: the file's hash is recorded only where its processing is
        // rawpilot's batch's, so a rating does not hand a photographer's pp3 to the batch.
        let batch_owned = matches!(state, Pp3State::Absent | Pp3State::Generated);
        record.written.begin(Snapshot {
            mark,
            pp3_blake3: batch_owned.then(|| pp3::digest(&text)),
            look: if batch_owned {
                record.written.look()
            } else {
                None
            },
        });
        record.rated = Some(RatedInUi { at: rpl::now() });
        record.save(nef)?;

        let path = pp3::path_for(nef);
        write_atomic(&path, text.as_bytes())
            .with_context(|| format!("writing {}", path.display()))?;
        writeback::verify(&path, sidecar::read_pp3_mark(nef)?, mark)?;
        record.written.commit();
        record.save(nef)
    }
}

/// Writes `to_write` as the shot's pp3 and reads it back.
fn write_and_verify(nef: &Path, option: &OptionName, to_write: &CompletePp3) -> Result<()> {
    let path = pp3::path_for(nef);
    write_atomic(&path, to_write.as_str().as_bytes())?;
    let back = sidecar::read_pp3(nef)?;
    ensure!(
        back.as_deref() == Some(to_write.as_str()),
        "{}: wrote the pp3 of {option} but read back something else",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Goal: what the UI writes survives the batch even when its bytes are the batch's own,
    //! a hand-edited pp3 is never replaced unasked, and an interrupted choice is recognised
    //! and finished. Method: a folder with a NEF stand-in, a hand-made render in the looks
    //! store, and the real batch decisions (`sidecar::plan`, `Rpl::owns_pp3`) consulted
    //! after each UI call. No rawtherapee-cli is involved.

    use std::fs;

    use super::*;
    use crate::config::{Config, OptionConfig, PartialSource};
    use crate::looks::{Against, Inputs, Manifest, RenderParams, Standing, View};
    use crate::option::LookOption;
    use crate::rate::{Label, Stars};
    use crate::shot::{Exposure, Orientation, PreviewSpan, Shot};
    use crate::sidecar::{FoundMark, Plan, RankSource, SidecarStyle};

    /// What the render of "Camera" holds; its Rank/ColorLabel are the default's.
    const RENDERED: &str = "[General]\nRank=0\nColorLabel=0\n\n[Exposure]\nCompensation=1.00\n";

    struct Fixture {
        dir: tempfile::TempDir,
        nef: std::path::PathBuf,
        session: Session,
        /// The render of "Camera", compared with the inputs it was made from.
        fresh: Fresh,
    }

    fn mark(r: u8) -> Mark {
        Mark {
            rating: Stars::new(r),
            label: Some(Label::Green),
        }
    }

    fn fixture(rank_source: RankSource) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let nef = dir.path().join("DSC_1.NEF");
        fs::write(&nef, b"x").unwrap();
        let cache = dir.path().join(".rawpilot");
        let option: OptionName = "Camera".to_owned().try_into().unwrap();
        let render = cache.join("looks").join("DSC_1.NEF").join("Camera");
        fs::create_dir_all(&render).unwrap();
        fs::write(render.join("fit.jpg"), b"").unwrap();
        fs::write(render.join("profile.pp3"), RENDERED).unwrap();
        let shot = Shot {
            file: nef.clone(),
            file_size: 1,
            taken_at: chrono::DateTime::parse_from_rfc3339("2026-01-01T10:00:00+00:00").unwrap(),
            preview: PreviewSpan { offset: 0, len: 0 },
            af: None,
            orientation: Orientation::Normal,
            exposure: Exposure {
                shutter_s: 0.01,
                focal_mm: 50.0,
                iso: 100,
                picture_control: None,
                compensation_ev: None,
            },
        };
        let look_option = LookOption::load_all(&[OptionConfig {
            name: option.clone(),
            key: 'c',
            partial: PartialSource::None,
        }])
        .unwrap()
        .remove(0);
        let params = RenderParams::of(&Config::load(None).unwrap().review);
        let inputs = Inputs::new(&shot, &look_option, &Rpl::default(), &params).unwrap();
        let manifest = Manifest {
            key: "k".into(),
            inputs_key: Some(inputs.key().to_owned()),
            option,
            pp3_blake3: pp3::digest(RENDERED),
            full_w: 10,
            full_h: 10,
            fit: View {
                file: "fit.jpg".into(),
                w: 10,
                h: 10,
            },
            regions: vec![],
        };
        fs::write(
            render.join("manifest.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        let rt = RtOptions {
            sidecar_style: SidecarStyle::Ext,
            rank_source,
            custom_profile_builder: false,
            config_dir: dir.path().to_owned(),
        };
        let defaults = DefaultProfiles::new(&rt, nef.clone(), cache.join("rt-template"));
        let folder = FolderLock::acquire(dir.path(), crate::lock::Run::Review).unwrap();
        let session = Session::new(LooksStore::new(&cache), rt, defaults, &folder);
        let fresh = match session.looks().standing(Against::Inputs(&inputs)).unwrap() {
            Standing::Fresh(fresh) => fresh,
            other => panic!("the hand-made render is not fresh: {other:?}"),
        };
        Fixture {
            dir,
            nef,
            session,
            fresh,
        }
    }

    fn camera() -> OptionName {
        "Camera".to_owned().try_into().unwrap()
    }

    fn read(nef: &Path) -> Option<String> {
        sidecar::read_pp3(nef).unwrap()
    }

    #[test]
    fn choosing_writes_the_rendered_pp3_and_records_the_choice() {
        let f = fixture(RankSource::ProcParams);
        let out = f
            .session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();
        assert_eq!(out, Choose::Applied);
        assert_eq!(read(&f.nef).as_deref(), Some(RENDERED));
        assert_eq!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Chosen { option: camera() }
        );
        assert!(matches!(Rpl::load(&f.nef).chosen, Some(Chosen::Applied(_))));
    }

    /// The batch wrote exactly the processing that Camera renders (that is what Camera is).
    /// After the UI picks Camera, the batch must treat the identical bytes as the
    /// photographer's: no processing change, whatever its look says, and a rating-only edit.
    #[test]
    fn a_chosen_pp3_equal_to_the_batchs_is_still_not_the_batchs() {
        let f = fixture(RankSource::ProcParams);
        let batch = RENDERED.replace("Rank=0", "Rank=3");
        fs::write(pp3::path_for(&f.nef), &batch).unwrap();
        let mut record = Rpl::default();
        record.written.begin(Snapshot {
            mark: mark(3),
            pp3_blake3: Some(pp3::digest(&batch)),
            look: None,
        });
        record.written.commit();
        record.save(&f.nef).unwrap();
        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::Generated);
        assert!(Rpl::load(&f.nef).owns_pp3(&batch));

        f.session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();

        assert_eq!(
            read(&f.nef).as_deref(),
            Some(batch.as_str()),
            "bytes unchanged"
        );
        let after = Rpl::load(&f.nef);
        assert!(!after.owns_pp3(&batch), "no longer the batch's to rewrite");
        assert_eq!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Chosen { option: camera() }
        );
    }

    #[test]
    fn a_hand_edited_pp3_is_replaced_only_when_confirmed() {
        let f = fixture(RankSource::ProcParams);
        let theirs = "[General]\nRank=4\nColorLabel=5\n\n[Exposure]\nCompensation=-0.5\n";
        fs::write(pp3::path_for(&f.nef), theirs).unwrap();
        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::HandEdited);

        let out = f
            .session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();
        assert_eq!(out, Choose::NeedsConfirmation);
        assert_eq!(read(&f.nef).as_deref(), Some(theirs));
        assert!(
            !crate::rpl::path_for(&f.nef).exists(),
            "nothing recorded either"
        );

        let out = f
            .session
            .choose(&f.nef, &f.fresh, OnHandEdited::Replace)
            .unwrap();
        assert_eq!(out, Choose::Applied);
        // Their processing is replaced; their rating and label are kept.
        assert_eq!(
            read(&f.nef).as_deref(),
            Some("[General]\nRank=4\nColorLabel=5\n\n[Exposure]\nCompensation=1.00\n")
        );
    }

    /// An interruption between recording the choice and reading the pp3 back leaves a
    /// pending choice; choosing again finishes it without a confirmation, and meanwhile the
    /// batch keeps its hands off.
    #[test]
    fn an_interrupted_choice_is_reported_protected_and_finished_by_choosing_again() {
        let f = fixture(RankSource::ProcParams);
        let theirs = "[General]\nRank=4\n\n[Exposure]\nCompensation=-0.5\n";
        fs::write(pp3::path_for(&f.nef), theirs).unwrap();
        pending_over(&f.nef, Some(theirs));
        assert_eq!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Interrupted { option: camera() }
        );
        assert!(!Rpl::load(&f.nef).owns_pp3(theirs));

        let out = f
            .session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();
        assert_eq!(out, Choose::Applied);
        assert_eq!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Chosen { option: camera() }
        );
    }

    /// Records a choice of Camera that began over `before` (the file's content, `None` for
    /// no file) and stopped.
    fn pending_over(nef: &Path, before: Option<&str>) {
        Rpl {
            chosen: Some(Chosen::Pending(PendingChoice {
                choice: Choice {
                    option: camera(),
                    at: rpl::now(),
                    processing_blake3: pp3::processing_digest(RENDERED),
                },
                prior: PriorPp3::of(before),
            })),
            ..Rpl::default()
        }
        .save(nef)
        .unwrap();
    }

    /// Goal: an interrupted choice skips the question only while the file is what it was
    /// when the choice began. Method: a pending choice over the photographer's pp3, which
    /// the photographer then edits in RawTherapee. Mutation: let `pp3_state` answer
    /// `Interrupted` for any pending choice whose file is not the rendered one: the edited
    /// row writes without asking.
    #[test]
    fn a_pp3_edited_after_an_interrupted_choice_is_asked_about_again() {
        let f = fixture(RankSource::ProcParams);
        let theirs = "[General]\nRank=4\n\n[Exposure]\nCompensation=-0.5\n";
        fs::write(pp3::path_for(&f.nef), theirs).unwrap();
        pending_over(&f.nef, Some(theirs));

        let edited = theirs.replace("-0.5", "-0.7");
        fs::write(pp3::path_for(&f.nef), &edited).unwrap();

        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::HandEdited);
        let out = f
            .session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();
        assert_eq!(out, Choose::NeedsConfirmation);
        assert_eq!(read(&f.nef).as_deref(), Some(edited.as_str()));
    }

    /// A rating changed since the interrupted choice (by a batch run or in the UI) is not an
    /// edit of the processing: the choice still finishes without asking.
    #[test]
    fn a_rating_changed_after_an_interrupted_choice_is_still_interrupted() {
        let f = fixture(RankSource::ProcParams);
        let theirs = "[General]\nRank=4\n\n[Exposure]\nCompensation=-0.5\n";
        fs::write(pp3::path_for(&f.nef), theirs).unwrap();
        pending_over(&f.nef, Some(theirs));
        fs::write(pp3::path_for(&f.nef), theirs.replace("Rank=4", "Rank=2")).unwrap();
        assert!(matches!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Interrupted { .. }
        ));
    }

    /// A pp3 that appeared after a choice began over no file is somebody's too.
    #[test]
    fn a_pp3_created_after_an_interrupted_choice_is_hand_edited() {
        let f = fixture(RankSource::ProcParams);
        pending_over(&f.nef, None);
        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::Absent);
        fs::write(pp3::path_for(&f.nef), "[General]\nRank=2\n").unwrap();
        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::HandEdited);
    }

    /// A record written before the prior file was kept: the file is asked about unless it
    /// is already what the choice writes.
    #[test]
    fn a_pending_record_without_a_prior_file_loads_and_asks() {
        let f = fixture(RankSource::ProcParams);
        let theirs = "[General]\nRank=4\n\n[Exposure]\nCompensation=-0.5\n";
        fs::write(pp3::path_for(&f.nef), theirs).unwrap();
        let legacy = serde_json::json!({
            "format": 1,
            "chosen": { "Pending": {
                "option": "Camera",
                "at": "2026-10-18T14:00:00Z",
                "processing_blake3": pp3::processing_digest(RENDERED),
            } },
        });
        fs::write(rpl::path_for(&f.nef), legacy.to_string()).unwrap();

        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::HandEdited);
        fs::write(pp3::path_for(&f.nef), RENDERED).unwrap();
        assert_eq!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Chosen { option: camera() }
        );
    }

    /// Goal: a choice that fails leaves the record as it was, whatever the failed call
    /// recorded first. Method: the pp3's temporary file name is taken by a directory, so the
    /// write fails after `Pending` is saved. Mutation: drop the restore in `choose`: the
    /// record stays `Pending` and the second assertion goes red.
    #[test]
    fn a_failed_choice_puts_the_record_back() {
        let f = fixture(RankSource::ProcParams);
        let theirs = "[General]\nRank=4\n\n[Exposure]\nCompensation=-0.5\n";
        fs::write(pp3::path_for(&f.nef), theirs).unwrap();
        let mut tmp = pp3::path_for(&f.nef).into_os_string();
        tmp.push(".tmp");
        fs::create_dir(&tmp).unwrap();

        let before = Rpl::load(&f.nef).chosen;
        assert!(
            f.session
                .choose(&f.nef, &f.fresh, OnHandEdited::Replace)
                .is_err()
        );
        assert_eq!(Rpl::load(&f.nef).chosen, before);
        assert_eq!(read(&f.nef).as_deref(), Some(theirs));
        assert_eq!(f.session.pp3_state(&f.nef).unwrap(), Pp3State::HandEdited);
    }

    /// A rating made in the UI is in the pp3 and the batch keeps it; it does not turn a
    /// chosen pp3 into a hand-edited one.
    #[test]
    fn a_ui_rating_is_written_recorded_and_kept_from_the_batch() {
        let f = fixture(RankSource::ProcParams);
        f.session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();
        f.session.rate(&f.nef, Stars::new(5)).unwrap();

        assert_eq!(
            sidecar::read_pp3_mark(&f.nef).unwrap(),
            FoundMark::from(Mark {
                rating: Stars::new(5),
                label: None
            }),
            "the render holds no label, so none is invented"
        );
        assert_eq!(
            f.session.pp3_state(&f.nef).unwrap(),
            Pp3State::Chosen { option: camera() },
            "rating does not change the processing"
        );
        let record = Rpl::load(&f.nef);
        assert!(record.rated.is_some());
        // The batch would rate it 2; the current 5 is the photographer's.
        let current = sidecar::read_pp3_mark(&f.nef).unwrap();
        assert!(matches!(
            sidecar::plan(mark(2), &current, &FoundMark::default(), &record),
            Plan::KeepHuman(_)
        ));
    }

    /// Goal: a rating keeps the colour label the file holds when it is written, not one read
    /// when the session began. Method: the label is set in the store after the session
    /// exists (as RawTherapee would), then the UI rates. Mutation: build the `Mark` with
    /// `label: None` in `Session::rate`: both rows go red.
    #[test]
    fn a_ui_rating_keeps_the_label_the_file_holds_now() {
        let f = fixture(RankSource::ProcParams);
        f.session
            .choose(&f.nef, &f.fresh, OnHandEdited::Refuse)
            .unwrap();
        let labelled = pp3::with_mark(&read(&f.nef).unwrap(), mark(0)).unwrap();
        fs::write(pp3::path_for(&f.nef), labelled).unwrap();

        f.session.rate(&f.nef, Stars::new(5)).unwrap();

        assert_eq!(
            sidecar::read_pp3_mark(&f.nef).unwrap(),
            FoundMark::from(mark(5))
        );

        let f = fixture(RankSource::Xmp);
        let path = sidecar::sidecar_path(&f.nef, SidecarStyle::Ext);
        crate::xmp::write_marks(&[(path.clone(), mark(1))]).unwrap();
        f.session.rate(&f.nef, Stars::new(5)).unwrap();
        assert_eq!(
            crate::xmp::read_marks(&[path]).unwrap(),
            [Some(FoundMark::from(mark(5)))]
        );
    }

    #[test]
    fn a_ui_rating_in_xmp_mode_goes_to_the_sidecar() {
        let f = fixture(RankSource::Xmp);
        f.session.rate(&f.nef, Stars::new(4)).unwrap();
        let sidecar = sidecar::sidecar_path(&f.nef, SidecarStyle::Ext);
        let found = crate::xmp::read_marks(&[sidecar]).unwrap();
        assert_eq!(
            found,
            [Some(FoundMark::from(Mark {
                rating: Stars::new(4),
                label: None
            }))]
        );
        assert!(Rpl::load(&f.nef).rated.is_some());
        assert!(
            read(&f.nef).is_none(),
            "no pp3 is created for a rating in xmp mode"
        );
        let _ = &f.dir;
    }
}
