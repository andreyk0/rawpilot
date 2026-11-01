//! The image routes: `/img/<shot>/<option>/<fit|rN>`.
//!
//! The request names a shot and an option by index and the picture by [`ImgRef`]; the file is
//! the one the render's manifest names for it. No part of the path comes from the client, and
//! [`LooksRoot`] refuses a manifest that names a file outside `<dir>/.rawpilot/looks/`.

use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::command::{OptionIdx, ShotIdx};
use crate::looks::{Rendered, View};

/// Which picture of a render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImgRef {
    /// The whole frame.
    Fit,
    /// The `n`th region crop.
    Region(usize),
}

impl fmt::Display for ImgRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImgRef::Fit => f.write_str("fit"),
            ImgRef::Region(n) => write!(f, "r{n}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?} is not a picture: expected fit or r<number>")]
pub struct ImgRefError(String);

impl ImgRef {
    pub fn parse(text: &str) -> Result<ImgRef, ImgRefError> {
        let region = text
            .strip_prefix('r')
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|digits| digits.parse().ok());
        match (text, region) {
            ("fit", _) => Ok(ImgRef::Fit),
            (_, Some(n)) => Ok(ImgRef::Region(n)),
            _ => Err(ImgRefError(text.to_owned())),
        }
    }

    /// The picture's file within `rendered`, if it has one.
    pub fn view(self, rendered: &Rendered) -> Option<&View> {
        match self {
            ImgRef::Fit => Some(&rendered.manifest.fit),
            ImgRef::Region(n) => rendered.manifest.regions.get(n).map(|r| &r.view),
        }
    }
}

/// The URL of a picture. `version` (the render's key) only keeps a browser from showing a
/// picture of an earlier render under the same index.
pub fn url(shot: ShotIdx, option: OptionIdx, img: ImgRef, version: &str) -> String {
    let version: String = version
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect();
    format!("/img/{shot}/{option}/{img}?k={version}")
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("{} is outside the looks directory", .0.display())]
    Outside(PathBuf),
    #[error("{}: {source}", path.display())]
    Unreadable { path: PathBuf, source: io::Error },
}

/// `<dir>/.rawpilot/looks/`, resolved: the only place pictures are read from.
#[derive(Debug, Clone)]
pub struct LooksRoot(PathBuf);

impl LooksRoot {
    pub fn open(looks_dir: &Path) -> io::Result<LooksRoot> {
        fs::canonicalize(looks_dir).map(LooksRoot)
    }

    /// The bytes of `path`, once every symlink and `..` in it is resolved and it is found
    /// below the root.
    pub fn read(&self, path: &Path) -> Result<Vec<u8>, ImageError> {
        let unreadable = |source| ImageError::Unreadable {
            path: path.to_owned(),
            source,
        };
        let resolved = fs::canonicalize(path).map_err(unreadable)?;
        if !resolved.starts_with(&self.0) {
            return Err(ImageError::Outside(path.to_owned()));
        }
        fs::read(&resolved).map_err(unreadable)
    }
}

#[cfg(test)]
mod tests {
    //! Goal: pictures come only from below the looks directory. Method: a temp directory
    //! with a looks tree and a secret beside it; paths that name the secret by `..` and by a
    //! symlink are read through `LooksRoot`.

    use super::*;

    #[test]
    fn picture_names_parse_strictly() {
        assert_eq!(ImgRef::parse("fit"), Ok(ImgRef::Fit));
        assert_eq!(ImgRef::parse("r0"), Ok(ImgRef::Region(0)));
        assert_eq!(ImgRef::parse("r12"), Ok(ImgRef::Region(12)));
        for bad in [
            "", "r", "R1", "r-1", "r+1", "r1x", "fit.jpg", "../fit", "r1/../x", "FIT",
        ] {
            assert!(ImgRef::parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(ImgRef::Region(3).to_string(), "r3");
    }

    /// Mutation: drop the `starts_with` check: both escape rows read the secret.
    #[test]
    fn a_path_outside_the_looks_directory_is_not_read() {
        let dir = tempfile::tempdir().unwrap();
        let looks = dir.path().join("looks");
        fs::create_dir_all(looks.join("DSC_1.NEF/Camera")).unwrap();
        fs::write(looks.join("DSC_1.NEF/Camera/fit.jpg"), b"picture").unwrap();
        fs::write(dir.path().join("secret.txt"), b"secret").unwrap();
        std::os::unix::fs::symlink(dir.path().join("secret.txt"), looks.join("link.jpg")).unwrap();
        let root = LooksRoot::open(&looks).unwrap();

        assert_eq!(
            root.read(&looks.join("DSC_1.NEF/Camera/fit.jpg")).unwrap(),
            b"picture"
        );
        for escape in [
            looks.join("../secret.txt"),
            looks.join("DSC_1.NEF/Camera/../../../secret.txt"),
            looks.join("link.jpg"),
            dir.path().join("secret.txt"),
        ] {
            assert!(
                matches!(root.read(&escape), Err(ImageError::Outside(_))),
                "{}",
                escape.display()
            );
        }
    }
}
