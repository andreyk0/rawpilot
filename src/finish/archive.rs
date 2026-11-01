//! Putting content onto the media drive without ever replacing anything there.
//!
//! The drive is the permanent copy, so a name that is taken stays as it is. The content is
//! written to a temporary name beside its destination, synced, and read back against the
//! blake3 of the source; only then is it given its real name, by a hard link, which fails
//! when the name exists (a rename would silently replace it). The temporary name is then
//! removed. An interrupted copy leaves at worst a temporary file, which the next run
//! overwrites, and never a half-written real name, so running again is the recovery.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use tracing::warn;

/// What is to be put at a destination: a file (followed through links, as the working
/// folder's NEFs are often links), or bytes made in memory.
#[derive(Debug, Clone, Copy)]
pub enum Content<'a> {
    File(&'a Path),
    Bytes(&'a [u8]),
}

impl Content<'_> {
    fn len(&self) -> Result<u64> {
        match self {
            Content::File(p) => Ok(fs::metadata(p)
                .with_context(|| format!("looking at {}", p.display()))?
                .len()),
            Content::Bytes(b) => Ok(b.len() as u64),
        }
    }

    fn hash(&self) -> Result<blake3::Hash> {
        match self {
            Content::File(p) => hash_of(p),
            Content::Bytes(b) => Ok(blake3::hash(b)),
        }
    }

    /// Writes the content to `tmp` (replacing a temporary file an interrupted run left),
    /// syncs it, and returns the blake3 of the bytes it read from the source.
    fn write_to(&self, tmp: &Path) -> io::Result<blake3::Hash> {
        let mut to = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(tmp)?;
        let mut hasher = blake3::Hasher::new();
        match self {
            Content::File(src) => {
                let mut from = File::open(src)?;
                let mut buf = vec![0u8; 1 << 20];
                loop {
                    let n = from.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buf[..n]);
                    to.write_all(&buf[..n])?;
                }
            }
            Content::Bytes(b) => {
                hasher.update(b);
                to.write_all(b)?;
            }
        }
        to.sync_all()?;
        Ok(hasher.finalize())
    }
}

/// What stands at a destination, compared with the content about to be put there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    Absent,
    /// A regular file with the content.
    Identical,
    /// A regular file with other content.
    Differs,
    /// Something that is not a regular file: a folder, or a link (whatever it points to).
    NotAFile,
}

/// How content got on the drive, or why it did not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Archived {
    Copied,
    /// The destination already held this content (a re-run after an interruption).
    AlreadyThere,
    Refused(Refusal),
    /// Neither copied nor refused: something went wrong reading, writing or verifying.
    Failed(String),
}

/// Why a destination was left alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    DifferentContent,
    NotAFile,
}

impl Refusal {
    pub fn describe(self) -> &'static str {
        match self {
            Refusal::DifferentContent => "a file with different content is already there",
            Refusal::NotAFile => "something that is not a regular file is already there",
        }
    }
}

/// What stands at `dest`, compared with `content`. Reads both only when their sizes agree.
pub fn standing(content: Content, dest: &Path) -> Result<Standing> {
    let meta = match fs::symlink_metadata(dest) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Standing::Absent),
        Err(e) => return Err(e).with_context(|| format!("looking at {}", dest.display())),
    };
    if !meta.is_file() {
        return Ok(Standing::NotAFile);
    }
    if meta.len() != content.len()? || content.hash()? != hash_of(dest)? {
        return Ok(Standing::Differs);
    }
    Ok(Standing::Identical)
}

fn hash_of(path: &Path) -> Result<blake3::Hash> {
    let mut hasher = blake3::Hasher::new();
    hasher
        .update_reader(File::open(path).with_context(|| format!("opening {}", path.display()))?)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(hasher.finalize())
}

/// Puts `content` at `dest` unless something is there. `dest`'s folder exists.
pub fn archive(content: Content, dest: &Path) -> Archived {
    match archive_with(content, dest, |c: &Content, tmp: &Path| c.write_to(tmp)) {
        Ok(a) => a,
        Err(e) => Archived::Failed(format!("{e:#}")),
    }
}

/// `archive`, with the write to the temporary file as a parameter: it writes the content to
/// its second argument and returns the blake3 of what it read from the source.
fn archive_with(
    content: Content,
    dest: &Path,
    write: impl Fn(&Content, &Path) -> io::Result<blake3::Hash>,
) -> Result<Archived> {
    if let Some(done) = settled(standing(content, dest)?) {
        return Ok(done);
    }
    let tmp = temp_name(dest)?;
    let linked = copy_and_link(content, dest, &tmp, write);
    // The real name, if it was made, is a second link to the same bytes.
    match fs::remove_file(&tmp) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => warn!(file = %tmp.display(), "could not remove the temporary copy: {e}"),
    }
    linked
}

/// The outcome when something stands at the destination; `None` when nothing does.
fn settled(standing: Standing) -> Option<Archived> {
    match standing {
        Standing::Absent => None,
        Standing::Identical => Some(Archived::AlreadyThere),
        Standing::Differs => Some(Archived::Refused(Refusal::DifferentContent)),
        Standing::NotAFile => Some(Archived::Refused(Refusal::NotAFile)),
    }
}

fn copy_and_link(
    content: Content,
    dest: &Path,
    tmp: &Path,
    write: impl Fn(&Content, &Path) -> io::Result<blake3::Hash>,
) -> Result<Archived> {
    let wanted = write(&content, tmp).with_context(|| format!("copying to {}", tmp.display()))?;
    let written = hash_of(tmp)?;
    ensure!(
        written == wanted,
        "{} does not read back as the source of {}: not archiving it",
        tmp.display(),
        dest.display()
    );
    match fs::hard_link(tmp, dest) {
        Ok(()) => {
            sync_dir_of(dest)?;
            Ok(Archived::Copied)
        }
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            // Taken since `standing` looked: whatever is there stays.
            match settled(standing(content, dest)?) {
                Some(done) => Ok(done),
                None => bail!("{} appeared and went again", dest.display()),
            }
        }
        Err(e) => Err(e).with_context(|| {
            format!(
                "giving {} its name (the drive must support hard links)",
                dest.display()
            )
        }),
    }
}

fn temp_name(dest: &Path) -> Result<PathBuf> {
    let name = dest
        .file_name()
        .with_context(|| format!("{} has no file name", dest.display()))?;
    let mut tmp = std::ffi::OsString::from(".");
    tmp.push(name);
    tmp.push(".rawpilot-tmp");
    Ok(dest.with_file_name(tmp))
}

fn sync_dir_of(path: &Path) -> Result<()> {
    let dir = path.parent().context("a destination has a folder")?;
    File::open(dir)
        .and_then(|d| d.sync_all())
        .with_context(|| format!("syncing {}", dir.display()))
}

#[cfg(test)]
mod tests {
    //! Goal: content reaches the drive only under a name nothing holds, byte for byte, and
    //! anything already there is left exactly as it is. Method: real files in a temp folder
    //! (the source reached through a symlink, as the working folder's NEFs are), with the
    //! destination in each state; the verify step is driven through `archive_with`'s write
    //! parameter with a writer that corrupts the copy.

    use std::os::unix::fs::symlink;

    use super::*;

    struct Scene {
        _dir: tempfile::TempDir,
        src: PathBuf,
        dest: PathBuf,
    }

    fn scene() -> Scene {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real.NEF");
        fs::write(&real, b"raw bytes").unwrap();
        let src = dir.path().join("link.NEF");
        symlink(&real, &src).unwrap();
        let dest_dir = dir.path().join("drive");
        fs::create_dir(&dest_dir).unwrap();
        Scene {
            dest: dest_dir.join("DSC_1.NEF"),
            src,
            _dir: dir,
        }
    }

    fn names_in(path: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// Mutation: copy the link instead of the bytes it points to, or leave the temporary
    /// name behind: red.
    #[test]
    fn an_absent_destination_gets_the_bytes_and_no_temporary_file() {
        let s = scene();
        assert_eq!(archive(Content::File(&s.src), &s.dest), Archived::Copied);
        assert_eq!(fs::read(&s.dest).unwrap(), b"raw bytes");
        assert!(fs::symlink_metadata(&s.dest).unwrap().is_file());
        assert_eq!(names_in(&s.dest), ["DSC_1.NEF"]);
    }

    /// Mutation: make `Content::Bytes` write nothing, or hash other bytes than it writes: red.
    #[test]
    fn bytes_made_in_memory_are_archived_the_same_way() {
        let s = scene();
        let bytes = Content::Bytes(b"{\"shots\":[]}\n");
        assert_eq!(standing(bytes, &s.dest).unwrap(), Standing::Absent);
        assert_eq!(archive(bytes, &s.dest), Archived::Copied);
        assert_eq!(fs::read(&s.dest).unwrap(), b"{\"shots\":[]}\n");
        assert_eq!(standing(bytes, &s.dest).unwrap(), Standing::Identical);
        assert_eq!(archive(bytes, &s.dest), Archived::AlreadyThere);
        assert_eq!(
            archive(Content::Bytes(b"{\"shots\":[1]}\n"), &s.dest),
            Archived::Refused(Refusal::DifferentContent)
        );
    }

    /// Mutation: compare only sizes (the first row has the same size), or treat an existing
    /// file as done without comparing: the different-content rows go red.
    #[test]
    fn a_taken_name_is_done_when_identical_and_refused_otherwise_and_never_touched() {
        let s = scene();
        fs::write(&s.dest, b"raw bytes").unwrap();
        assert_eq!(
            archive(Content::File(&s.src), &s.dest),
            Archived::AlreadyThere
        );

        for other in [&b"raw BYTES"[..], b"shorter", b""] {
            fs::write(&s.dest, other).unwrap();
            assert_eq!(
                archive(Content::File(&s.src), &s.dest),
                Archived::Refused(Refusal::DifferentContent),
                "{other:?}"
            );
            assert_eq!(fs::read(&s.dest).unwrap(), other);
            assert_eq!(names_in(&s.dest), ["DSC_1.NEF"]);
        }
    }

    /// Mutation: follow links at the destination (`metadata` for `symlink_metadata`), or
    /// treat a folder as absent: red.
    #[test]
    fn a_link_or_folder_at_the_destination_is_refused() {
        let s = scene();
        symlink(&s.src, &s.dest).unwrap();
        assert_eq!(
            archive(Content::File(&s.src), &s.dest),
            Archived::Refused(Refusal::NotAFile)
        );
        fs::remove_file(&s.dest).unwrap();
        fs::create_dir(&s.dest).unwrap();
        assert_eq!(
            archive(Content::File(&s.src), &s.dest),
            Archived::Refused(Refusal::NotAFile)
        );
    }

    /// The name is taken after `standing` looked (the window a rename would lose a file in):
    /// the writer below creates it in that window. Mutation: give the name with `fs::rename`
    /// instead of `fs::hard_link`: the other file is replaced and this goes red.
    #[test]
    fn a_name_taken_during_the_copy_is_not_replaced() {
        let s = scene();
        let dest = s.dest.clone();
        let taken_meanwhile = move |content: &Content, tmp: &Path| {
            fs::write(&dest, b"someone else's file").unwrap();
            content.write_to(tmp)
        };
        assert_eq!(
            archive_with(Content::File(&s.src), &s.dest, taken_meanwhile).unwrap(),
            Archived::Refused(Refusal::DifferentContent)
        );
        assert_eq!(fs::read(&s.dest).unwrap(), b"someone else's file");
        assert_eq!(names_in(&s.dest), ["DSC_1.NEF"]);
    }

    /// Mutation: skip the read-back comparison: the corrupted copy is archived and this
    /// goes red.
    #[test]
    fn a_copy_that_does_not_read_back_is_not_given_its_name() {
        let s = scene();
        let corrupting = |content: &Content, tmp: &Path| {
            let hash = content.write_to(tmp)?;
            fs::write(tmp, b"raw bytez")?;
            Ok(hash)
        };
        let err = archive_with(Content::File(&s.src), &s.dest, corrupting).unwrap_err();
        assert!(format!("{err:#}").contains("does not read back"), "{err:#}");
        assert_eq!(names_in(&s.dest), Vec::<String>::new());
    }

    /// An interruption left a temporary file (and nothing under the real name): the run
    /// that follows completes. Mutation: open the temporary file with `create_new`: red.
    #[test]
    fn a_temporary_file_left_by_an_interrupted_run_is_overwritten() {
        let s = scene();
        fs::write(temp_name(&s.dest).unwrap(), b"half of a co").unwrap();
        assert_eq!(archive(Content::File(&s.src), &s.dest), Archived::Copied);
        assert_eq!(fs::read(&s.dest).unwrap(), b"raw bytes");
        assert_eq!(names_in(&s.dest), ["DSC_1.NEF"]);
    }

    /// Mutation: report a missing source as copied or refused: red.
    #[test]
    fn a_missing_source_fails_and_leaves_nothing() {
        let s = scene();
        fs::remove_file(fs::canonicalize(&s.src).unwrap()).unwrap();
        assert!(matches!(
            archive(Content::File(&s.src), &s.dest),
            Archived::Failed(_)
        ));
        assert_eq!(names_in(&s.dest), Vec::<String>::new());
    }
}
