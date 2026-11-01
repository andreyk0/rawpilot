//! What `finish` will do for a folder: which shots are keepers, which files of theirs go to
//! the drive and under which names, which JPEG each is exported to, and what else in the
//! folder is left for the photographer.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use super::archive::Content;
use super::event::EventName;
use super::media_root::EventDir;
use super::shoot::{self, Frame, Keeping};
use crate::gemini::StoredAnswer;
use crate::rate::Stars;
use crate::rpl::{self, Rpl};
use crate::shot::Shot;
use crate::sidecar::{self, RtOptions};
use crate::{pp3, writeback};

/// The summary and the Gemini answers are archived under these names in `rawpilot/`.
const SUMMARY_NAME: &str = "summary.json";
const ANSWERS_NAME: &str = "gemini-answers.jsonl";
const REPORT_NAME: &str = "report.html";

/// Files in the folder that finish neither exports nor archives, listed so a video is not
/// forgotten. Lower case.
const OTHER_MEDIA_EXTENSIONS: [&str; 11] = [
    "jpg", "jpeg", "heic", "heif", "mp4", "mov", "m4v", "mts", "m2ts", "avi", "mkv",
];

/// A keeper and the files of it that go to the drive.
#[derive(Debug, Clone)]
pub struct Keeper {
    pub nef: PathBuf,
    /// Required: the export develops the shot with exactly this profile.
    pub pp3: PathBuf,
    /// rawpilot's record, if the shot has one.
    pub rpl: Option<PathBuf>,
    /// RawTherapee's XMP sidecar, if the shot has one.
    pub xmp: Option<PathBuf>,
    /// Where the JPEG is exported before `import-media` files it.
    pub export: PathBuf,
}

impl Keeper {
    pub fn name(&self) -> String {
        file_name(&self.nef)
    }

    /// The files archived for this shot, NEF first.
    fn sources(&self) -> impl Iterator<Item = &Path> {
        [
            Some(&self.nef),
            Some(&self.pp3),
            self.rpl.as_ref(),
            self.xmp.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(PathBuf::as_path)
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// One thing to put on the drive.
#[derive(Debug)]
pub struct Item<'a> {
    pub dest: PathBuf,
    pub content: Content<'a>,
}

/// What describes the whole shoot, as archived under `rawpilot/`.
#[derive(Debug)]
pub struct ShootFiles {
    summary: String,
    answers: String,
    /// `<dir>/.rawpilot/report.html`, when `cull` or `process` wrote one.
    report: Option<PathBuf>,
}

#[derive(Debug)]
pub struct Plan {
    pub frames: Vec<Frame>,
    pub keepers: Vec<Keeper>,
    shoot: ShootFiles,
    pub other_media: Vec<PathBuf>,
}

/// Everything the plan is made from.
pub struct Inputs<'a> {
    pub dir: &'a Path,
    pub event: &'a EventName,
    pub min_rating: Stars,
    pub rt: &'a RtOptions,
    pub shots: &'a [Shot],
    pub answers: &'a [StoredAnswer],
}

impl Plan {
    /// Reads the ratings where RawTherapee reads them, the shots' records and the folder's
    /// files. Fails, before anything is done, when no shot is a keeper, when a keeper has no
    /// pp3 to export with, or when two keepers would export to one JPEG.
    pub fn build(inputs: &Inputs) -> Result<Plan> {
        let files: Vec<&Path> = inputs.shots.iter().map(|s| s.file.as_path()).collect();
        let marks = writeback::current_marks(&files, inputs.rt)?;
        let frames: Vec<Frame> = inputs
            .shots
            .iter()
            .zip(&marks)
            .map(|(shot, mark)| Frame::new(shot, &Rpl::load(&shot.file), mark, inputs.min_rating))
            .collect();

        let export_dir = inputs
            .dir
            .join(".rawpilot")
            .join("export")
            .join(inputs.event.as_str());
        let mut keepers = Vec::new();
        for (shot, frame) in inputs.shots.iter().zip(&frames) {
            if frame.keeping == Keeping::Keeper {
                keepers.push(keeper_of(shot, inputs.rt, &export_dir)?);
            }
        }
        if keepers.is_empty() {
            bail!(
                "no shot in {} is rated at least {} stars (where RawTherapee reads ratings)",
                inputs.dir.display(),
                inputs.min_rating.get()
            );
        }
        let missing: Vec<String> = keepers
            .iter()
            .filter(|k| !k.pp3.is_file())
            .map(Keeper::name)
            .collect();
        if !missing.is_empty() {
            bail!(
                "{} keepers have no pp3 to export with ({}); open them in RawTherapee or run \
                 `rawpilot cull` first",
                missing.len(),
                missing.join(", ")
            );
        }
        let mut exports = HashSet::new();
        for k in &keepers {
            if !exports.insert(k.export.clone()) {
                bail!("two keepers would export to {}", k.export.display());
            }
        }

        let cache_root = inputs.dir.join(".rawpilot");
        let report = cache_root.join(REPORT_NAME);
        Ok(Plan {
            shoot: ShootFiles {
                summary: shoot::summary_json(inputs.event, inputs.min_rating, &frames),
                answers: answers_jsonl(inputs.answers),
                report: report.is_file().then_some(report),
            },
            frames,
            keepers,
            other_media: other_media(inputs.dir)?,
        })
    }

    /// Every file that goes to the drive under `event`, keepers' first.
    pub fn archive_items(&self, event: &EventDir) -> Vec<Item<'_>> {
        let keepers = self.keepers.iter().flat_map(|k| {
            k.sources().map(|src| Item {
                dest: event.file(&file_name(src)),
                content: Content::File(src),
            })
        });
        let shoot = [
            Some(Item {
                dest: event.shoot_file(SUMMARY_NAME),
                content: Content::Bytes(self.shoot.summary.as_bytes()),
            }),
            Some(Item {
                dest: event.shoot_file(ANSWERS_NAME),
                content: Content::Bytes(self.shoot.answers.as_bytes()),
            }),
            self.shoot.report.as_deref().map(|report| Item {
                dest: event.shoot_file(REPORT_NAME),
                content: Content::File(report),
            }),
        ];
        keepers.chain(shoot.into_iter().flatten()).collect()
    }
}

fn keeper_of(shot: &Shot, rt: &RtOptions, export_dir: &Path) -> Result<Keeper> {
    let nef = shot.file.clone();
    let stem = nef
        .file_stem()
        .with_context(|| format!("{} has no file name", nef.display()))?;
    let present = |p: PathBuf| p.is_file().then_some(p);
    Ok(Keeper {
        pp3: pp3::path_for(&nef),
        rpl: present(rpl::path_for(&nef)),
        xmp: present(sidecar::sidecar_path(&nef, rt.sidecar_style)),
        export: export_dir.join(format!("{}.jpg", stem.to_string_lossy())),
        nef,
    })
}

/// One JSON object per line: the pass, the digest, and the model's reply verbatim.
fn answers_jsonl(answers: &[StoredAnswer]) -> String {
    answers
        .iter()
        .map(|a| {
            let mut line = serde_json::to_string(a).expect("an answer is plain data");
            line.push('\n');
            line
        })
        .collect()
}

/// The photos and videos directly in `dir` that are not NEFs, by name.
fn other_media(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let is_media = path.is_file()
            && path.extension().is_some_and(|e| {
                OTHER_MEDIA_EXTENSIONS
                    .iter()
                    .any(|m| e.eq_ignore_ascii_case(m))
            });
        if is_media {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}
