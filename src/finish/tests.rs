//! Goal: `finish` exports every keeper first, archives only after all exports worked, never
//! replaces or loses a file on the drive, treats a re-run as a no-op, reports and fails on
//! what it cannot do safely, and under `--dry-run` writes nothing. Method: a temp working
//! folder (NEFs as symlinks to a "card" folder, as in real use), a temp media root, and a
//! fake `Tools` that records its calls in order and writes stand-in JPEGs; shots are built
//! directly (the NEF header reader is not under test here). Expected archive contents are
//! written out, not derived from the code.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use chrono::DateTime;

use super::*;
use crate::pp3;
use crate::rate::{Label, Stars};
use crate::rpl::{Rpl, Snapshot};
use crate::shot::{Exposure, Orientation, PreviewSpan};
use crate::sidecar::{Mark, RankSource, SidecarStyle};

fn shot_at(file: PathBuf, minute: u32) -> Shot {
    Shot {
        file,
        file_size: 1,
        taken_at: DateTime::parse_from_rfc3339(&format!("2026-11-01T10:{minute:02}:00+00:00"))
            .unwrap(),
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
    }
}

fn rt(rank_source: RankSource) -> RtOptions {
    RtOptions {
        sidecar_style: SidecarStyle::Std,
        rank_source,
        custom_profile_builder: false,
        config_dir: PathBuf::new(),
    }
}

fn pp3_text(name: &str, rank: i64) -> String {
    format!("[General]\nRank={rank}\nColorLabel=0\n\n[Exposure]\nCompensation=0\n# {name}\n")
}

fn xmp_text(rating: i64) -> String {
    format!(
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF \
         xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description \
         xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\" xmp:Rating=\"{rating}\"/></rdf:RDF>\
         </x:xmpmeta>"
    )
}

/// A working folder of symlinks to a card, and a media root.
struct Fx {
    card: tempfile::TempDir,
    work: tempfile::TempDir,
    media: tempfile::TempDir,
    shots: RefCell<Vec<Shot>>,
}

impl Fx {
    fn new() -> Fx {
        let fx = Fx {
            card: tempfile::tempdir().unwrap(),
            work: tempfile::tempdir().unwrap(),
            media: tempfile::tempdir().unwrap(),
            shots: RefCell::new(Vec::new()),
        };
        fs::create_dir(fx.media.path().join("photos-RAW")).unwrap();
        fx
    }

    /// A shot whose pp3 holds `rank` (none when `None`), reached through a link.
    fn add(&self, name: &str, rank: Option<i64>) -> PathBuf {
        let real = self.card.path().join(name);
        fs::write(&real, format!("NEF bytes of {name}")).unwrap();
        let nef = self.work.path().join(name);
        symlink(&real, &nef).unwrap();
        if let Some(rank) = rank {
            fs::write(pp3::path_for(&nef), pp3_text(name, rank)).unwrap();
        }
        let minute = u32::try_from(self.shots.borrow().len()).unwrap();
        self.shots.borrow_mut().push(shot_at(nef.clone(), minute));
        nef
    }

    fn record(&self, nef: &Path, rawpilot_rating: u8) {
        let mut rpl = Rpl::default();
        rpl.written.committed = Some(Snapshot {
            mark: Mark {
                rating: Stars::new(rawpilot_rating),
                label: Some(Label::Green),
            },
            pp3_blake3: None,
            look: None,
        });
        rpl.save(nef).unwrap();
    }

    fn args(&self, min: u8, mode: Mode) -> Args {
        Args {
            dir: self.work.path().to_owned(),
            event: EventName::parse("2026-11-01-test").unwrap(),
            min_rating: Stars::new(min),
            media: MediaRoot::open(self.media.path()).unwrap(),
            mode,
        }
    }

    fn event_dir(&self) -> PathBuf {
        self.media.path().join("photos-RAW/2026-11-01-test")
    }

    fn run(&self, args: &Args, rt: &RtOptions, tools: &FakeTools) -> Result<()> {
        let shots = self.shots.borrow().clone();
        run_with(args, rt, tools, move |_| Ok(shots.clone()))
    }
}

/// Every file under `root` with its content (links as their target), by relative path.
fn snapshot(root: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let rel = path.strip_prefix(root).unwrap().display().to_string();
            if path.is_dir() {
                out.insert(format!("{rel}/"), String::new());
                walk(root, &path, out);
            } else {
                let text = String::from_utf8_lossy(&fs::read(&path).unwrap()).into_owned();
                out.insert(rel, text);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

#[derive(Default)]
struct FakeTools {
    calls: RefCell<Vec<String>>,
    /// The `(pp3 content, nef)` of each export.
    exported: RefCell<Vec<(String, PathBuf)>>,
    /// Whether the drive's event folder existed at each export.
    drive_touched_at_export: RefCell<Vec<bool>>,
    imported: RefCell<Vec<Vec<PathBuf>>>,
    event_dir: PathBuf,
    fail_export_of: Option<&'static str>,
    fail_import: bool,
}

impl Tools for FakeTools {
    fn export_jpeg(&self, pp3: &Path, nef: &Path, out: &Path) -> Result<()> {
        let name = nef.file_name().unwrap().to_string_lossy().into_owned();
        self.calls.borrow_mut().push(format!("export {name}"));
        self.drive_touched_at_export
            .borrow_mut()
            .push(self.event_dir.exists());
        assert!(!out.exists(), "an old export would pass for this one");
        if self.fail_export_of == Some(name.as_str()) {
            bail!("RawTherapee did not like {name}");
        }
        let pp3_content = fs::read_to_string(pp3).unwrap();
        fs::write(out, format!("JPEG of {name}")).unwrap();
        self.exported
            .borrow_mut()
            .push((pp3_content, nef.to_owned()));
        Ok(())
    }

    fn import(&self, media: &MediaRoot, jpegs: &[PathBuf]) -> Result<()> {
        self.calls.borrow_mut().push("import".into());
        assert_eq!(media.path(), self.media_path());
        self.imported.borrow_mut().push(jpegs.to_vec());
        if self.fail_import {
            bail!("import-media failed (exit status: 1)");
        }
        Ok(())
    }
}

impl FakeTools {
    fn for_fixture(fx: &Fx) -> FakeTools {
        FakeTools {
            event_dir: fx.event_dir(),
            ..FakeTools::default()
        }
    }

    fn media_path(&self) -> &Path {
        self.event_dir.parent().unwrap().parent().unwrap()
    }

    fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }
}

/// Four shots rated 5, 3, 2 and 4 in their pp3s, records for two of them, one XMP sidecar,
/// the cache files a Gemini run leaves, and other media in the folder.
fn shoot() -> (Fx, FakeTools) {
    let fx = Fx::new();
    let a = fx.add("DSC_0001.NEF", Some(5));
    fx.add("DSC_0002.NEF", Some(3));
    let c = fx.add("DSC_0003.NEF", Some(2));
    fx.add("DSC_0004.NEF", Some(4));
    fx.record(&a, 4);
    fx.record(&c, 2);
    fs::write(a.with_extension("xmp"), xmp_text(5)).unwrap();
    let cache = fx.work.path().join(".rawpilot");
    fs::create_dir_all(cache.join("judge")).unwrap();
    fs::write(
        cache.join("judge/abc.json"),
        r#"{"model_version":"m-1","usage":{"prompt_tokens":7,"output_tokens":3},"text":"{\"winner\":1}"}"#,
    )
    .unwrap();
    fs::write(cache.join("report.html"), "<html>report</html>").unwrap();
    for other in ["clip.MP4", "IMG_9.jpg", "notes.txt"] {
        fs::write(fx.work.path().join(other), "x").unwrap();
    }
    let tools = FakeTools::for_fixture(&fx);
    (fx, tools)
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// Mutation: archive another set of files (drop `.rpl` or `.xmp`, archive a non-keeper, keep
/// the link instead of the bytes), name them differently, or leave the working folder
/// changed: red.
#[test]
fn a_run_archives_each_keepers_files_and_the_shoot_description() {
    let (fx, tools) = shoot();
    let before = snapshot(fx.work.path());
    fx.run(
        &fx.args(3, Mode::Write),
        &rt(RankSource::ProcParams),
        &tools,
    )
    .unwrap();

    let event = fx.event_dir();
    assert_eq!(
        names(&event),
        [
            "DSC_0001.NEF",
            "DSC_0001.NEF.pp3",
            "DSC_0001.NEF.rpl",
            "DSC_0001.xmp",
            "DSC_0002.NEF",
            "DSC_0002.NEF.pp3",
            "DSC_0004.NEF",
            "DSC_0004.NEF.pp3",
            "rawpilot",
        ]
    );
    assert_eq!(
        fs::read(event.join("DSC_0001.NEF")).unwrap(),
        b"NEF bytes of DSC_0001.NEF"
    );
    assert!(
        fs::symlink_metadata(event.join("DSC_0001.NEF"))
            .unwrap()
            .is_file()
    );
    assert_eq!(
        fs::read_to_string(event.join("DSC_0002.NEF.pp3")).unwrap(),
        pp3_text("DSC_0002.NEF", 3)
    );
    assert_eq!(
        fs::read_to_string(event.join("DSC_0001.xmp")).unwrap(),
        xmp_text(5)
    );
    assert_eq!(
        names(&event.join("rawpilot")),
        ["gemini-answers.jsonl", "report.html", "summary.json"]
    );
    assert_eq!(
        fs::read_to_string(event.join("rawpilot/report.html")).unwrap(),
        "<html>report</html>"
    );
    assert_eq!(
        fs::read_to_string(event.join("rawpilot/gemini-answers.jsonl")).unwrap(),
        concat!(
            r#"{"pass":"judge","digest":"abc","model_version":"m-1","#,
            r#""usage":{"prompt_tokens":7,"output_tokens":3},"text":"{\"winner\":1}"}"#,
            "\n"
        )
    );
    let summary: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(event.join("rawpilot/summary.json")).unwrap())
            .unwrap();
    let want = serde_json::json!({
        "event": "2026-11-01-test",
        "min_rating": 3,
        "shots": [
            {"file": "DSC_0001.NEF", "taken_at": "2026-11-01T10:00:00Z", "rawpilot_rating": 4,
             "current_rating": 5, "look": null, "keeper": true},
            {"file": "DSC_0002.NEF", "taken_at": "2026-11-01T10:01:00Z", "rawpilot_rating": null,
             "current_rating": 3, "look": null, "keeper": true},
            {"file": "DSC_0003.NEF", "taken_at": "2026-11-01T10:02:00Z", "rawpilot_rating": 2,
             "current_rating": 2, "look": null, "keeper": false},
            {"file": "DSC_0004.NEF", "taken_at": "2026-11-01T10:03:00Z", "rawpilot_rating": null,
             "current_rating": 4, "look": null, "keeper": true},
        ]
    });
    assert_eq!(summary, want);

    // Only the lock and the exports are new in the working folder.
    let after = snapshot(fx.work.path());
    let changed: Vec<&String> = after
        .iter()
        .filter(|(k, v)| before.get(*k) != Some(*v))
        .map(|(k, _)| k)
        .collect();
    assert!(
        changed.iter().all(|k| k.starts_with(".rawpilot/")),
        "{changed:?}"
    );
    assert!(
        before.iter().all(|(k, v)| after.get(k) == Some(v)),
        "something was lost or changed"
    );
}

/// Mutation: choose by the rating rawpilot recorded instead of the current one: the demoted
/// shot (recorded 5, now 2) is exported and the promoted one (recorded 1, now 5) is not: red.
#[test]
fn keepers_are_chosen_by_the_current_rating_not_the_one_rawpilot_wrote() {
    let fx = Fx::new();
    let demoted = fx.add("DSC_0001.NEF", Some(2));
    let promoted = fx.add("DSC_0002.NEF", Some(5));
    fx.record(&demoted, 5);
    fx.record(&promoted, 1);
    let tools = FakeTools::for_fixture(&fx);
    fx.run(
        &fx.args(3, Mode::Write),
        &rt(RankSource::ProcParams),
        &tools,
    )
    .unwrap();
    assert_eq!(tools.calls(), ["export DSC_0002.NEF", "import"]);
}

/// In xmp mode the rating is the sidecar's; the pp3's rank is not read. Mutation: read the
/// pp3 whatever the mode: both rows flip and this goes red.
#[test]
fn the_rating_is_read_where_raw_therapee_reads_it() {
    let fx = Fx::new();
    let a = fx.add("DSC_0001.NEF", Some(5));
    let b = fx.add("DSC_0002.NEF", Some(1));
    fs::write(a.with_extension("xmp"), xmp_text(1)).unwrap();
    fs::write(b.with_extension("xmp"), xmp_text(4)).unwrap();
    let tools = FakeTools::for_fixture(&fx);
    fx.run(&fx.args(3, Mode::Write), &rt(RankSource::Xmp), &tools)
        .unwrap();
    assert_eq!(tools.calls(), ["export DSC_0002.NEF", "import"]);
}

/// Mutation: archive before exporting (the drive folder exists at the first export), or
/// export with a pp3 other than the shot's own: red.
#[test]
fn each_keeper_is_exported_through_its_own_pp3_before_the_drive_is_touched() {
    let (fx, tools) = shoot();
    fx.run(
        &fx.args(3, Mode::Write),
        &rt(RankSource::ProcParams),
        &tools,
    )
    .unwrap();
    assert_eq!(
        tools.calls(),
        [
            "export DSC_0001.NEF",
            "export DSC_0002.NEF",
            "export DSC_0004.NEF",
            "import"
        ]
    );
    assert_eq!(
        *tools.drive_touched_at_export.borrow(),
        [false, false, false]
    );
    let exported = tools.exported.borrow();
    let want: Vec<String> = [
        ("DSC_0001.NEF", 5),
        ("DSC_0002.NEF", 3),
        ("DSC_0004.NEF", 4),
    ]
    .into_iter()
    .map(|(n, r)| pp3_text(n, r))
    .collect();
    let got: Vec<String> = exported.iter().map(|(p, _)| p.clone()).collect();
    assert_eq!(got, want);
    let export_dir = fx.work.path().join(".rawpilot/export/2026-11-01-test");
    assert_eq!(
        *tools.imported.borrow(),
        [vec![
            export_dir.join("DSC_0001.jpg"),
            export_dir.join("DSC_0002.jpg"),
            export_dir.join("DSC_0004.jpg"),
        ]]
    );
}

/// Mutation: archive or import after a failed export, or stop at the first failure (the
/// other export is then not attempted): red.
#[test]
fn a_failed_export_stops_before_the_drive_and_still_tries_the_other_shots() {
    let (fx, mut tools) = shoot();
    tools.fail_export_of = Some("DSC_0002.NEF");
    let err = fx
        .run(
            &fx.args(3, Mode::Write),
            &rt(RankSource::ProcParams),
            &tools,
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("1 exports failed"), "{err:#}");
    assert_eq!(
        tools.calls(),
        [
            "export DSC_0001.NEF",
            "export DSC_0002.NEF",
            "export DSC_0004.NEF"
        ]
    );
    assert!(!fx.event_dir().exists());
    assert_eq!(
        names(&fx.media.path().join("photos-RAW")),
        Vec::<String>::new()
    );
}

/// Mutation: make an existing identical file an error, or archive again in a way that
/// changes it: the second run goes red.
#[test]
fn running_again_changes_nothing_and_succeeds() {
    let (fx, tools) = shoot();
    let args = fx.args(3, Mode::Write);
    fx.run(&args, &rt(RankSource::ProcParams), &tools).unwrap();
    let first = snapshot(fx.media.path());
    fx.run(&args, &rt(RankSource::ProcParams), &tools).unwrap();
    assert_eq!(snapshot(fx.media.path()), first);
}

/// A run interrupted after some files: the archive holds part of the set, and running again
/// completes it. Mutation: refuse an identical file instead of counting it done: red.
#[test]
fn a_run_after_an_interruption_completes_the_archive() {
    let (fx, tools) = shoot();
    let args = fx.args(3, Mode::Write);
    fx.run(&args, &rt(RankSource::ProcParams), &tools).unwrap();
    let complete = snapshot(fx.media.path());
    fs::remove_file(fx.event_dir().join("DSC_0004.NEF.pp3")).unwrap();
    fs::remove_file(fx.event_dir().join("rawpilot/summary.json")).unwrap();
    fx.run(&args, &rt(RankSource::ProcParams), &tools).unwrap();
    assert_eq!(snapshot(fx.media.path()), complete);
}

/// Mutation: overwrite the archived file (or skip the check and report success): the file
/// changes or the run succeeds, and this goes red. The other files are still archived and
/// the JPEGs imported, as far as is safe.
#[test]
fn a_file_with_other_content_on_the_drive_is_refused_and_the_run_fails() {
    let (fx, tools) = shoot();
    let args = fx.args(3, Mode::Write);
    fx.run(&args, &rt(RankSource::ProcParams), &tools).unwrap();
    let theirs = fx.event_dir().join("DSC_0002.NEF");
    fs::write(&theirs, b"edited on the drive").unwrap();
    fs::remove_file(fx.event_dir().join("DSC_0004.NEF")).unwrap();

    let err = fx
        .run(&args, &rt(RankSource::ProcParams), &tools)
        .unwrap_err();
    assert!(format!("{err:#}").contains("1 files refused"), "{err:#}");
    assert_eq!(fs::read(&theirs).unwrap(), b"edited on the drive");
    assert_eq!(
        fs::read(fx.event_dir().join("DSC_0004.NEF")).unwrap(),
        b"NEF bytes of DSC_0004.NEF"
    );
    assert_eq!(tools.calls().last().unwrap(), "import");
}

/// Mutation: report success when the import fails: red. The archive is still complete.
#[test]
fn a_failed_import_fails_the_run_after_the_archive_is_made() {
    let (fx, mut tools) = shoot();
    tools.fail_import = true;
    let err = fx
        .run(
            &fx.args(3, Mode::Write),
            &rt(RankSource::ProcParams),
            &tools,
        )
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("import-media failed"),
        "{err:#}"
    );
    assert!(fx.event_dir().join("DSC_0001.NEF").is_file());
}

/// Mutation: list NEFs or notes, drop an extension (or its upper-case form), or leave the
/// list unsorted: red.
#[test]
fn photos_and_videos_not_exported_are_listed_by_name() {
    let (fx, _) = shoot();
    let shots = fx.shots.borrow();
    let plan = Plan::build(&Inputs {
        dir: fx.work.path(),
        event: &EventName::parse("e").unwrap(),
        min_rating: Stars::new(3),
        rt: &rt(RankSource::ProcParams),
        shots: &shots,
        answers: &[],
    })
    .unwrap();
    let listed: Vec<String> = plan
        .other_media
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(listed, ["IMG_9.jpg", "clip.MP4"]);
}

/// Mutation: report a look that is only `Pending`, or none when `Applied`: red.
#[test]
fn the_chosen_look_is_the_one_the_review_applied() {
    let (fx, _) = shoot();
    let choice = |option: &str| crate::rpl::Choice {
        option: option.to_owned().try_into().unwrap(),
        at: crate::rpl::now(),
        processing_blake3: String::new(),
    };
    let look_of = |chosen: Option<crate::rpl::Chosen>| {
        let record = Rpl {
            chosen,
            ..Rpl::default()
        };
        let shot = fx.shots.borrow()[1].clone();
        let mark = crate::sidecar::FoundMark::default();
        shoot::Frame::new(&shot, &record, &mark, Stars::new(3))
            .look
            .map(|l| l.as_str().to_owned())
    };
    assert_eq!(
        look_of(Some(crate::rpl::Chosen::Applied(choice("Warm")))),
        Some("Warm".to_owned())
    );
    let pending = crate::rpl::Chosen::Pending(crate::rpl::PendingChoice {
        choice: choice("Warm"),
        prior: crate::rpl::PriorPp3::NoFile,
    });
    assert_eq!(look_of(Some(pending)), None);
    assert_eq!(look_of(None), None);
}

/// Mutation: write anything (take the lock, make a folder, copy, export, import) in dry-run
/// mode: the snapshots differ or a tool is called, red. The refusal is predicted.
#[test]
fn a_dry_run_writes_nothing_anywhere_and_predicts_a_refusal() {
    let (fx, tools) = shoot();
    let work_before = snapshot(fx.work.path());
    let media_before = snapshot(fx.media.path());
    fx.run(
        &fx.args(3, Mode::DryRun),
        &rt(RankSource::ProcParams),
        &tools,
    )
    .unwrap();
    assert_eq!(snapshot(fx.work.path()), work_before);
    assert_eq!(snapshot(fx.media.path()), media_before);
    assert_eq!(tools.calls(), Vec::<String>::new());

    fx.run(
        &fx.args(3, Mode::Write),
        &rt(RankSource::ProcParams),
        &tools,
    )
    .unwrap();
    fs::write(fx.event_dir().join("DSC_0002.NEF"), b"edited").unwrap();
    let work_before = snapshot(fx.work.path());
    let media_before = snapshot(fx.media.path());
    let err = fx
        .run(
            &fx.args(3, Mode::DryRun),
            &rt(RankSource::ProcParams),
            &tools,
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("would be refused"), "{err:#}");
    assert_eq!(snapshot(fx.work.path()), work_before);
    assert_eq!(snapshot(fx.media.path()), media_before);
}

/// Mutation: skip the lock: the run goes ahead and this goes red.
#[test]
fn finish_waits_for_the_folder_lock() {
    let (fx, tools) = shoot();
    let _review = FolderLock::acquire(fx.work.path(), Run::Review).unwrap();
    let err = fx
        .run(
            &fx.args(3, Mode::Write),
            &rt(RankSource::ProcParams),
            &tools,
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("`rawpilot review`"), "{err:#}");
    assert_eq!(tools.calls(), Vec::<String>::new());
    assert!(!fx.event_dir().exists());
}

/// Mutation: let a keeper without a pp3 through (it is exported with no profile), or let an
/// empty selection succeed: red. Nothing is exported either way.
#[test]
fn a_run_that_cannot_export_every_keeper_or_has_none_does_nothing() {
    let fx = Fx::new();
    fx.add("DSC_0001.NEF", Some(5));
    let no_pp3 = fx.add("DSC_0002.NEF", None);
    fs::write(no_pp3.with_extension("xmp"), xmp_text(5)).unwrap();
    let tools = FakeTools::for_fixture(&fx);
    // In xmp mode the second shot is a keeper by its sidecar but has no pp3.
    let err = fx
        .run(&fx.args(3, Mode::Write), &rt(RankSource::Xmp), &tools)
        .unwrap_err();
    assert!(
        format!("{err:#}").contains("no pp3 to export with (DSC_0002.NEF)"),
        "{err:#}"
    );
    assert_eq!(tools.calls(), Vec::<String>::new());

    let none = Fx::new();
    none.add("DSC_0001.NEF", Some(1));
    let tools = FakeTools::for_fixture(&none);
    let err = none
        .run(
            &none.args(3, Mode::Write),
            &rt(RankSource::ProcParams),
            &tools,
        )
        .unwrap_err();
    assert!(format!("{err:#}").contains("rated at least 3"), "{err:#}");
    assert_eq!(tools.calls(), Vec::<String>::new());
}
