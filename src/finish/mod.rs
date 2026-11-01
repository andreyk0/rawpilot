//! `rawpilot finish DIR --event NAME`: after `review`, archive what was kept and file its JPEGs
//! in the media collection.
//!
//! The keepers are the shots whose rating RawTherapee reads now is at least `--min-rating`.
//! In this order:
//!
//! 1. each keeper's JPEG is exported into `<DIR>/.rawpilot/export/<event>/` through that
//!    shot's own pp3, at the top quality;
//! 2. only if every export worked, each keeper's NEF, pp3, `.rpl` and XMP sidecar go to
//!    `<media-root>/photos-RAW/<event>/`, with the shoot's summary, its cached Gemini answers
//!    and its report in `rawpilot/` beside them (`archive` never replaces a file there);
//! 3. `import-media` files the JPEGs under `<media-root>/photos/` by date.
//!
//! The working folder's NEFs, pp3s and records are only read. `--dry-run` prints the plan and
//! writes nothing, to the folder or the drive.
//!
//! | Module | |
//! | --- | --- |
//! | `event` | the event name, safe as one folder name |
//! | `media_root` | the drive, and the one folder on it `finish` writes to |
//! | `archive` | copying onto the drive without replacing anything |
//! | `shoot` | each shot's facts and the shoot summary |
//! | `plan` | keepers, files, destinations |
//! | `tools` | `rawtherapee-cli` and `import-media` behind a trait |
//! | `outcome` | what a run did, and its verdict |

mod archive;
mod event;
mod media_root;
mod outcome;
mod plan;
mod shoot;
mod tools;

#[cfg(test)]
mod tests;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use tracing::{info, warn};

use self::archive::Standing;
pub use self::event::EventName;
use self::media_root::EventDir;
pub use self::media_root::MediaRoot;
use self::outcome::{Archiving, Exports, Importing, Report};
use self::plan::{Inputs, Plan};
pub use self::tools::{Installed, Tools};
use crate::lock::{FolderLock, Run};
use crate::rate::Stars;
use crate::shot::Shot;
use crate::sidecar::RtOptions;
use crate::writeback::Mode;
use crate::{gemini, nef};

pub struct Args {
    pub dir: PathBuf,
    pub event: EventName,
    pub min_rating: Stars,
    pub media: MediaRoot,
    pub mode: Mode,
}

pub fn run(args: &Args, rt: &RtOptions, tools: &dyn Tools) -> Result<()> {
    run_with(args, rt, tools, nef::scan)
}

/// `run`, with the reading of the folder's shots as a parameter.
fn run_with(
    args: &Args,
    rt: &RtOptions,
    tools: &dyn Tools,
    scan: impl Fn(&Path) -> Result<Vec<Shot>>,
) -> Result<()> {
    match args.mode {
        // Reads only: no lock is taken, no folder is made.
        Mode::DryRun => preview(&args.dir, &prepare(args, rt, &scan)?, args),
        Mode::Write => {
            let folder = FolderLock::acquire(&args.dir, Run::Finish)?;
            warn!("RawTherapee should be closed: it may write back the pp3s it holds");
            let plan = prepare(args, rt, &scan)?;
            let report = execute(&plan, args, tools, &folder);
            report.print();
            list_other_media(&plan);
            report.verdict()
        }
    }
}

fn prepare(
    args: &Args,
    rt: &RtOptions,
    scan: &impl Fn(&Path) -> Result<Vec<Shot>>,
) -> Result<Plan> {
    let shots = scan(&args.dir)?;
    ensure!(
        !shots.is_empty(),
        "no readable NEFs in {}",
        args.dir.display()
    );
    let answers = gemini::stored_answers(&args.dir.join(".rawpilot"))?;
    Plan::build(&Inputs {
        dir: &args.dir,
        event: &args.event,
        min_rating: args.min_rating,
        rt,
        shots: &shots,
        answers: &answers,
    })
}

/// The `FolderLock` is the witness that nothing else of rawpilot's works in the folder.
fn execute(plan: &Plan, args: &Args, tools: &dyn Tools, _folder: &FolderLock) -> Report {
    let mut report = Report {
        shots: plan.frames.len(),
        keepers: plan.keepers.len(),
        exports: export_all(plan, tools),
        archiving: Archiving::NotAttempted,
        importing: Importing::NotAttempted,
    };
    let Exports::Done(jpegs) = &report.exports else {
        return report;
    };
    report.archiving = archive_all(plan, &args.media.event_dir(&args.event));
    report.importing = match tools.import(&args.media, jpegs) {
        Ok(()) => Importing::Done,
        Err(e) => Importing::Failed(format!("{e:#}")),
    };
    report
}

/// Exports every keeper, carrying on past a failure so that all of them are listed.
fn export_all(plan: &Plan, tools: &dyn Tools) -> Exports {
    let mut jpegs = Vec::new();
    let mut failures = Vec::new();
    for (i, keeper) in plan.keepers.iter().enumerate() {
        info!(
            "exporting {}/{}: {}",
            i + 1,
            plan.keepers.len(),
            keeper.name()
        );
        match export_one(keeper, tools) {
            Ok(()) => jpegs.push(keeper.export.clone()),
            Err(e) => failures.push((keeper.name(), format!("{e:#}"))),
        }
    }
    if failures.is_empty() {
        Exports::Done(jpegs)
    } else {
        Exports::Failed(failures)
    }
}

fn export_one(keeper: &plan::Keeper, tools: &dyn Tools) -> Result<()> {
    let dir = keeper.export.parent().context("an export has a folder")?;
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    // An export left by an earlier run must not pass for this one's.
    match fs::remove_file(&keeper.export) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("removing {}", keeper.export.display())),
    }
    tools.export_jpeg(&keeper.pp3, &keeper.nef, &keeper.export)
}

fn archive_all(plan: &Plan, event: &EventDir) -> Archiving {
    if let Err(e) = event.create() {
        return Archiving::Failed(format!("creating the event folder: {e}"));
    }
    let items = plan.archive_items(event);
    let total = items.len();
    Archiving::Done(
        items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                info!("archiving {}/{total}: {}", i + 1, item.dest.display());
                let done = archive::archive(item.content, &item.dest);
                (item.dest, done)
            })
            .collect(),
    )
}

/// The plan, and what the drive holds for each file now; nothing is written. An error when a
/// file would be refused or the drive cannot be read, so a script sees what a run would do.
fn preview(dir: &Path, plan: &Plan, args: &Args) -> Result<()> {
    println!(
        "dry run: {} keepers of {} shots in {} (nothing is written)",
        plan.keepers.len(),
        plan.frames.len(),
        dir.display()
    );
    for keeper in &plan.keepers {
        println!("  export  {} -> {}", keeper.name(), keeper.export.display());
    }
    let mut unsafe_files = 0;
    for item in plan.archive_items(&args.media.event_dir(&args.event)) {
        let verdict = match archive::standing(item.content, &item.dest) {
            Ok(Standing::Absent) => "would copy".to_owned(),
            Ok(Standing::Identical) => "already there".to_owned(),
            Ok(Standing::Differs) => {
                unsafe_files += 1;
                format!(
                    "WOULD REFUSE ({})",
                    archive::Refusal::DifferentContent.describe()
                )
            }
            Ok(Standing::NotAFile) => {
                unsafe_files += 1;
                format!("WOULD REFUSE ({})", archive::Refusal::NotAFile.describe())
            }
            Err(e) => {
                unsafe_files += 1;
                format!("CANNOT TELL ({e:#})")
            }
        };
        println!("  archive {verdict}: {}", item.dest.display());
    }
    println!(
        "  import-media --media-root {} {} JPEGs",
        args.media.path().display(),
        plan.keepers.len()
    );
    list_other_media(plan);
    if unsafe_files > 0 {
        bail!("{unsafe_files} files would be refused or could not be checked");
    }
    Ok(())
}

fn list_other_media(plan: &Plan) {
    if plan.other_media.is_empty() {
        return;
    }
    println!(
        "{} other photos and videos in the folder, neither exported nor imported by finish:",
        plan.other_media.len()
    );
    for path in &plan.other_media {
        println!("  {}", path.display());
    }
}
