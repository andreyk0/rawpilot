//! One rawpilot per folder.
//!
//! `process`, `cull`, `review` and `finish` each hold an exclusive lock on `<dir>/.rawpilot/lock` for
//! their whole run. Two of them in one folder would race on the render scratch directory, on
//! the `.rpl` records (a batch saving its in-memory copy over a choice made in the UI) and on
//! the render directories themselves.
//!
//! The lock is the operating system's `flock` on the file, not the file's existence: a file
//! left by a crashed run holds no lock and blocks nobody. The holder writes its pid and
//! command into the file so a refused second run can say who is in the way; the text is
//! advisory and only read after the lock was refused.
//!
//! [`FolderLock`] is a witness: the code that writes into a folder asks for a `&FolderLock`,
//! and the only way to have one is to have taken the lock.

use std::fmt;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Which command is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Run {
    Process,
    Cull,
    Review,
    Finish,
}

impl Run {
    const ALL: [Run; 4] = [Run::Process, Run::Cull, Run::Review, Run::Finish];

    fn name(self) -> &'static str {
        match self {
            Run::Process => "process",
            Run::Cull => "cull",
            Run::Review => "review",
            Run::Finish => "finish",
        }
    }
}

impl fmt::Display for Run {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What the lock file says about whoever wrote it last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Holder {
    Known {
        pid: u32,
        run: Run,
    },
    /// The file is empty or unreadable: the holder is between taking the lock and writing it.
    Unknown,
}

impl Holder {
    fn parse(text: &str) -> Holder {
        let mut words = text.split_whitespace();
        let (Some(pid), Some(run), None) = (words.next(), words.next(), words.next()) else {
            return Holder::Unknown;
        };
        match (pid.parse(), Run::ALL.into_iter().find(|r| r.name() == run)) {
            (Ok(pid), Some(run)) => Holder::Known { pid, run },
            _ => Holder::Unknown,
        }
    }
}

impl fmt::Display for Holder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Holder::Known { pid, run } => write!(f, "`rawpilot {run}` (pid {pid})"),
            Holder::Unknown => f.write_str("another rawpilot"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "{holder} is already running in {}; {} before running `rawpilot {wanted}` here",
    dir.display(),
    way_out(*holder)
)]
struct Busy {
    holder: Holder,
    dir: PathBuf,
    wanted: Run,
}

/// What the photographer can do about the run in the way.
fn way_out(holder: Holder) -> &'static str {
    match holder {
        // `review` serves until it is stopped.
        Holder::Known {
            run: Run::Review, ..
        } => "stop it (Ctrl-C in its terminal)",
        // Once its ratings are written, stopping `process` loses no finished render:
        // `review` makes the missing ones, nearest the shot on screen first.
        Holder::Known {
            run: Run::Process, ..
        } => "wait for it to finish, or stop it (Ctrl-C): it loses no finished render",
        Holder::Known {
            run: Run::Cull | Run::Finish,
            ..
        }
        | Holder::Unknown => "wait for it to finish, or stop it",
    }
}

/// The exclusive lock on a folder, held until dropped.
#[derive(Debug)]
pub struct FolderLock {
    dir: PathBuf,
    // Closing the file releases the lock.
    _file: File,
}

impl FolderLock {
    /// Takes the lock on `dir` for `run`, or refuses at once naming the command that holds
    /// it. Creates `<dir>/.rawpilot/`.
    pub fn acquire(dir: &Path, run: Run) -> Result<FolderLock> {
        let root = dir.join(".rawpilot");
        fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        let path = root.join("lock");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                let holder =
                    fs::read_to_string(&path).map_or(Holder::Unknown, |text| Holder::parse(&text));
                return Err(Busy {
                    holder,
                    dir: dir.to_owned(),
                    wanted: run,
                }
                .into());
            }
            Err(TryLockError::Error(e)) => {
                return Err(e).with_context(|| format!("locking {}", path.display()));
            }
        }
        file.set_len(0)
            .and_then(|()| writeln!(file, "{} {run}", std::process::id()))
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(FolderLock {
            dir: dir.to_owned(),
            _file: file,
        })
    }

    /// The folder this lock holds.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    //! Goal: a second rawpilot in a folder is refused while the first runs, told who holds
    //! the folder, and admitted once the first is gone, even when the first left its file
    //! behind. Method: real locks on a temp folder, in one process (a lock belongs to the
    //! open file, so a second open conflicts with the first just as another process would).

    use super::*;

    /// Mutation: make `acquire` skip `try_lock`: the refusal rows go red.
    #[test]
    fn a_second_run_is_refused_naming_the_first_until_it_ends() {
        let dir = tempfile::tempdir().unwrap();
        let first = FolderLock::acquire(dir.path(), Run::Review).unwrap();
        for wanted in [Run::Process, Run::Cull, Run::Review, Run::Finish] {
            let refused = FolderLock::acquire(dir.path(), wanted).unwrap_err();
            let message = format!("{refused:#}");
            assert!(
                message.contains(&format!("`rawpilot review` (pid {})", std::process::id())),
                "{message}"
            );
            assert!(
                message.contains(&format!("`rawpilot {wanted}` here")),
                "{message}"
            );
            assert!(
                message.contains("stop it (Ctrl-C in its terminal)"),
                "{message}"
            );
        }
        drop(first);
        let process = FolderLock::acquire(dir.path(), Run::Process).unwrap();
        let message = format!(
            "{:#}",
            FolderLock::acquire(dir.path(), Run::Review).unwrap_err()
        );
        assert!(message.contains("loses no finished render"), "{message}");
        drop(process);
    }

    /// Mutation: refuse when the file exists instead of when it is locked: red.
    #[test]
    fn a_file_left_by_a_crashed_run_blocks_nobody() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join(".rawpilot")).unwrap();
        fs::write(dir.path().join(".rawpilot/lock"), "99999 process\n").unwrap();
        FolderLock::acquire(dir.path(), Run::Cull).unwrap();
        let text = fs::read_to_string(dir.path().join(".rawpilot/lock")).unwrap();
        assert_eq!(text, format!("{} cull\n", std::process::id()));
    }

    #[test]
    fn the_holder_text_is_read_strictly() {
        assert_eq!(
            Holder::parse("42 process\n"),
            Holder::Known {
                pid: 42,
                run: Run::Process
            }
        );
        for bad in ["", "42", "42 sing", "x process", "42 process extra"] {
            assert_eq!(Holder::parse(bad), Holder::Unknown, "{bad:?}");
        }
    }
}
