//! The two programs `finish` runs: RawTherapee's command line, to develop each keeper, and
//! the photographer's `import-media`, to file the JPEGs by date. Behind a trait so that the
//! rest runs, and is tested, without either.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};

use super::media_root::MediaRoot;
use crate::rtcli;

/// JPEG quality of the exports: the top of the scale. Chroma is not subsampled (`rtcli`
/// passes `-js3`), and the camera's metadata is carried unchanged (`Metadata::Camera`), so
/// `import-media` names the file after the same camera as the photographer's earlier imports.
const EXPORT_JPEG_QUALITY: u8 = 100;

pub trait Tools {
    /// Develops `nef` through exactly the profile `pp3` into the JPEG `out`.
    fn export_jpeg(&self, pp3: &Path, nef: &Path, out: &Path) -> Result<()>;

    /// Files `jpegs` under the media root by date (`import-media`), showing its output. An
    /// error when it exits with a failure.
    fn import(&self, media: &MediaRoot, jpegs: &[PathBuf]) -> Result<()>;
}

/// The programs installed on this machine.
pub struct Installed;

impl Tools for Installed {
    fn export_jpeg(&self, pp3: &Path, nef: &Path, out: &Path) -> Result<()> {
        rtcli::render_jpeg(pp3, nef, out, EXPORT_JPEG_QUALITY, rtcli::Metadata::Camera)
    }

    fn import(&self, media: &MediaRoot, jpegs: &[PathBuf]) -> Result<()> {
        let status = Command::new("import-media")
            .arg("--media-root")
            .arg(media.path())
            .args(jpegs)
            .status()
            .context("running import-media")?;
        ensure!(status.success(), "import-media failed ({status})");
        Ok(())
    }
}
