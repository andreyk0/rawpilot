//! What a `finish` run did, stage by stage, and what it tells the photographer.

use std::path::PathBuf;

use anyhow::{Result, bail};

use super::archive::Archived;

/// The JPEG exports. A failure on any shot stops the run before the drive is touched.
#[derive(Debug)]
pub enum Exports {
    Done(Vec<PathBuf>),
    /// The shots that could not be exported, with why.
    Failed(Vec<(String, String)>),
}

#[derive(Debug)]
pub enum Archiving {
    /// The exports failed, so the drive was not touched.
    NotAttempted,
    /// The event folder could not be made.
    Failed(String),
    /// What became of each file, by destination.
    Done(Vec<(PathBuf, Archived)>),
}

#[derive(Debug)]
pub enum Importing {
    /// The exports failed, so nothing was filed.
    NotAttempted,
    Done,
    Failed(String),
}

#[derive(Debug)]
pub struct Report {
    pub shots: usize,
    pub keepers: usize,
    pub exports: Exports,
    pub archiving: Archiving,
    pub importing: Importing,
}

/// How many files ended each way.
#[derive(Debug, Default, PartialEq, Eq)]
struct Tally {
    copied: usize,
    already_there: usize,
    refused: usize,
    failed: usize,
}

impl Report {
    fn tally(&self) -> Tally {
        let mut t = Tally::default();
        if let Archiving::Done(files) = &self.archiving {
            for (_, a) in files {
                match a {
                    Archived::Copied => t.copied += 1,
                    Archived::AlreadyThere => t.already_there += 1,
                    Archived::Refused(_) => t.refused += 1,
                    Archived::Failed(_) => t.failed += 1,
                }
            }
        }
        t
    }

    /// Prints the totals and every file or shot that went wrong.
    pub fn print(&self) {
        println!("finish: {} keepers of {} shots", self.keepers, self.shots);
        match &self.exports {
            Exports::Done(jpegs) => println!("exported {} JPEGs", jpegs.len()),
            Exports::Failed(failures) => {
                println!(
                    "{} exports FAILED; nothing was archived or imported:",
                    failures.len()
                );
                for (shot, why) in failures {
                    println!("  {shot}: {why}");
                }
            }
        }
        match &self.archiving {
            Archiving::NotAttempted => {}
            Archiving::Failed(why) => println!("archive FAILED: {why}"),
            Archiving::Done(files) => {
                let t = self.tally();
                println!(
                    "archive: {} files, {} copied, {} already there, {} refused, {} failed",
                    files.len(),
                    t.copied,
                    t.already_there,
                    t.refused,
                    t.failed
                );
                for (dest, a) in files {
                    match a {
                        Archived::Refused(why) => {
                            println!("  REFUSED {}: {}", dest.display(), why.describe());
                        }
                        Archived::Failed(why) => println!("  FAILED {}: {why}", dest.display()),
                        Archived::Copied | Archived::AlreadyThere => {}
                    }
                }
            }
        }
        match &self.importing {
            Importing::NotAttempted => {}
            Importing::Done => println!("import-media: done"),
            Importing::Failed(why) => println!("import-media FAILED: {why}"),
        }
    }

    /// Ok only when every stage ran and succeeded; otherwise an error naming each stage
    /// that did not, after the stages that could still run have run.
    pub fn verdict(&self) -> Result<()> {
        let mut problems = Vec::new();
        if let Exports::Failed(f) = &self.exports {
            problems.push(format!("{} exports failed", f.len()));
        }
        match &self.archiving {
            Archiving::Failed(why) => problems.push(format!("could not archive: {why}")),
            Archiving::NotAttempted | Archiving::Done(_) => {}
        }
        let t = self.tally();
        if t.refused + t.failed > 0 {
            problems.push(format!(
                "{} files refused and {} failed on the drive",
                t.refused, t.failed
            ));
        }
        if let Importing::Failed(why) = &self.importing {
            problems.push(format!("import-media failed: {why}"));
        }
        if problems.is_empty() {
            Ok(())
        } else {
            bail!("{}", problems.join("; "))
        }
    }
}
