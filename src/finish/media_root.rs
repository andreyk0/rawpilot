//! The photographer's media drive, as far as `finish` may touch it: one event folder under
//! `photos-RAW/`.

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::event::EventName;

/// Where the archived NEFs go under the media root.
const ARCHIVE_DIR: &str = "photos-RAW";
/// Under the event folder, the files that describe the whole shoot.
const SHOOT_DIR: &str = "rawpilot";

/// A media root whose `photos-RAW` exists as a directory. An unmounted drive leaves an empty
/// mountpoint, which has no `photos-RAW`: holding a `MediaRoot` proves the drive is there, and
/// nothing creates `photos-RAW`.
#[derive(Debug, Clone)]
pub struct MediaRoot {
    root: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum MediaRootError {
    #[error(
        "no media root: pass --media-root PATH or set MEDIA_ROOT (the folder that holds \
         photos-RAW and photos)"
    )]
    Unset,
    #[error(
        "{} is not a directory; is the media drive mounted?",
        .0.join(ARCHIVE_DIR).display()
    )]
    NoArchive(PathBuf),
}

impl MediaRoot {
    pub fn open(root: &Path) -> Result<MediaRoot, MediaRootError> {
        if root.join(ARCHIVE_DIR).is_dir() {
            Ok(MediaRoot {
                root: root.to_owned(),
            })
        } else {
            Err(MediaRootError::NoArchive(root.to_owned()))
        }
    }

    /// The root `--media-root` names, else the `MEDIA_ROOT` environment variable's value
    /// (`env`, passed in so that only `main` reads the environment); empty counts as unset.
    pub fn choose(flag: Option<PathBuf>, env: Option<OsString>) -> Result<PathBuf, MediaRootError> {
        flag.or_else(|| env.filter(|v| !v.is_empty()).map(PathBuf::from))
            .ok_or(MediaRootError::Unset)
    }

    /// What `import-media` is told as its media root.
    pub fn path(&self) -> &Path {
        &self.root
    }

    pub fn event_dir(&self, event: &EventName) -> EventDir {
        EventDir {
            dir: self.root.join(ARCHIVE_DIR).join(event.as_str()),
        }
    }
}

/// `<media-root>/photos-RAW/<event>/`, which may not exist yet.
#[derive(Debug, Clone)]
pub struct EventDir {
    dir: PathBuf,
}

impl EventDir {
    /// Where an archived file of that name goes.
    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// Where a file describing the whole shoot goes.
    pub fn shoot_file(&self, name: &str) -> PathBuf {
        self.dir.join(SHOOT_DIR).join(name)
    }

    /// Makes the event folder and its `rawpilot/` folder if absent. `photos-RAW` exists
    /// (`MediaRoot` proves it), so this adds nothing above the event.
    pub fn create(&self) -> io::Result<()> {
        fs::create_dir_all(self.dir.join(SHOOT_DIR))
    }
}

#[cfg(test)]
mod tests {
    //! Goal: the media root is only ever one whose `photos-RAW` exists, and the places
    //! `finish` writes are inside `photos-RAW/<event>/`. Method: temp folders.

    use super::*;

    /// Mutation: make `open` accept a root without `photos-RAW`, or create it: red.
    #[test]
    fn a_root_without_photos_raw_is_refused_and_left_alone() {
        let root = tempfile::tempdir().unwrap();
        assert!(matches!(
            MediaRoot::open(root.path()),
            Err(MediaRootError::NoArchive(_))
        ));
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);

        fs::write(root.path().join(ARCHIVE_DIR), "a file, not a folder").unwrap();
        assert!(MediaRoot::open(root.path()).is_err());
        fs::remove_file(root.path().join(ARCHIVE_DIR)).unwrap();

        fs::create_dir(root.path().join(ARCHIVE_DIR)).unwrap();
        MediaRoot::open(root.path()).unwrap();
    }

    /// Mutation: let `create` build `photos-RAW` too (`create_dir_all` from the root), or
    /// join the event onto the root instead of `photos-RAW`: red.
    #[test]
    fn the_event_folder_is_made_inside_photos_raw() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("photos-RAW")).unwrap();
        let media = MediaRoot::open(root.path()).unwrap();
        let event = EventName::parse("2026-11-01-autumn").unwrap();
        let dir = media.event_dir(&event);
        assert!(!root.path().join("photos-RAW/2026-11-01-autumn").exists());
        dir.create().unwrap();
        assert!(
            root.path()
                .join("photos-RAW/2026-11-01-autumn/rawpilot")
                .is_dir()
        );
        assert_eq!(
            dir.file("DSC_1.NEF"),
            root.path().join("photos-RAW/2026-11-01-autumn/DSC_1.NEF")
        );
        assert_eq!(
            dir.shoot_file("summary.json"),
            root.path()
                .join("photos-RAW/2026-11-01-autumn/rawpilot/summary.json")
        );
    }

    /// Mutation: let the environment win over the flag, or take an empty value: red.
    #[test]
    fn the_flag_wins_over_the_environment_and_empty_is_unset() {
        let flag = || Some(PathBuf::from("/flag"));
        let env = |v: &str| Some(OsString::from(v));
        assert_eq!(
            MediaRoot::choose(flag(), env("/env")).unwrap(),
            PathBuf::from("/flag")
        );
        assert_eq!(
            MediaRoot::choose(None, env("/env")).unwrap(),
            PathBuf::from("/env")
        );
        assert!(matches!(
            MediaRoot::choose(None, env("")),
            Err(MediaRootError::Unset)
        ));
        assert!(matches!(
            MediaRoot::choose(None, None),
            Err(MediaRootError::Unset)
        ));
    }
}
