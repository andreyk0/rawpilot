//! A complete pp3 for a fresh NEF, as RawTherapee itself would build it.
//!
//! `rawtherapee-cli -d` resolves the default raw profile (neutral values overlaid with
//! Preferences > Default Processing Profile) and writes the result next to its output. That
//! resolved profile is the same for every image unless a per-image rule chooses the
//! profile, so one conversion per run yields the pp3 for all missing files — measured
//! byte-identical across NEFs at different ISOs, where converting each file would
//! cost ~7 s apiece. The image-dependent values are written as placeholders
//! (`CurveFromHistogramMatching=false` with a linear curve, `Temperature=6504` under
//! `Setting=Camera`) where a pp3 the GUI saves holds the computed curve and camera white
//! balance, so RawTherapee is expected to compute them when the image is opened.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use tracing::info;

use crate::pp3::DefaultProfile;
use crate::sidecar::RtOptions;

/// Refuses when the profile could differ per image: a dynamic profile rule or a custom
/// profile builder would make one shared pp3 wrong for some files.
pub fn check_profile_is_shared(rt_config_dir: &Path, custom_profile_builder: bool) -> Result<()> {
    ensure!(
        !custom_profile_builder,
        "RawTherapee has a CustomProfileBuilderPath set, so each image may get its own \
         profile; rawpilot cannot create pp3s for it. Open the folder in RawTherapee first, or \
         switch ThumbnailRankColorMode to xmp."
    );
    let rules = rt_config_dir.join("dynamicprofile.cfg");
    match fs::read_to_string(&rules) {
        Ok(text) if text.lines().any(|l| l.trim_start().starts_with('[')) => bail!(
            "{} has dynamic profile rules, so each image may get its own profile; rawpilot \
             cannot create pp3s for it. Open the folder in RawTherapee first, or switch \
             ThumbnailRankColorMode to xmp.",
            rules.display()
        ),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("reading {}", rules.display())),
    }
}

/// The default profile of a shoot, resolved by the first caller that needs it (one
/// `rawtherapee-cli` conversion, ~9 s) and shared by the rest, who wait for it. It is refused
/// up front when the profile could differ per image. A failure is kept too: a missing
/// RawTherapee or a profile that differs per image does not mend itself within a run, and
/// every later caller gets the same error at once instead of waiting for another attempt.
pub struct DefaultProfiles {
    resolve: Box<dyn Fn() -> Result<DefaultProfile> + Send + Sync>,
    resolution: Mutex<Resolution>,
}

enum Resolution {
    Unresolved,
    Resolved(Arc<DefaultProfile>),
    Failed(String),
}

impl DefaultProfiles {
    /// `sample` is any NEF of the shoot; `work_dir` holds the throwaway conversion.
    pub fn new(rt: &RtOptions, sample: PathBuf, work_dir: PathBuf) -> DefaultProfiles {
        let rt_config_dir = rt.config_dir.clone();
        let custom_profile_builder = rt.custom_profile_builder;
        DefaultProfiles::resolving_with(move || {
            check_profile_is_shared(&rt_config_dir, custom_profile_builder)?;
            default_profile_pp3(&sample, &work_dir)
        })
    }

    /// Profiles made by `resolve`, which runs at most once.
    pub fn resolving_with(
        resolve: impl Fn() -> Result<DefaultProfile> + Send + Sync + 'static,
    ) -> DefaultProfiles {
        DefaultProfiles {
            resolve: Box::new(resolve),
            resolution: Mutex::new(Resolution::Unresolved),
        }
    }

    pub fn get(&self) -> Result<Arc<DefaultProfile>> {
        let mut resolution = self.resolution.lock().expect("no panic while resolving");
        match &*resolution {
            Resolution::Resolved(p) => return Ok(Arc::clone(p)),
            Resolution::Failed(why) => bail!("{why}"),
            Resolution::Unresolved => {}
        }
        match (self.resolve)() {
            Ok(p) => {
                let p = Arc::new(p);
                *resolution = Resolution::Resolved(Arc::clone(&p));
                Ok(p)
            }
            Err(e) => {
                *resolution = Resolution::Failed(format!("{e:#}"));
                Err(e)
            }
        }
    }
}

/// Which metadata a rendered JPEG carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metadata {
    /// What the profile's `[MetaData]` says; RawTherapee rewrites some tags (`Model=Z 8` for
    /// the camera's `NIKON Z 8`). Enough for the review UI's renders.
    AsProfile,
    /// The camera's tags unchanged (RawTherapee's tunnel mode), as NX Studio exports carry
    /// them, so a JPEG is filed under the same camera name as the photographer's earlier
    /// ones. The pixels are already turned upright and no orientation tag is carried.
    Camera,
}

const CAMERA_METADATA: &str = "[MetaData]\nMode=0\n";

/// Renders `nef` through exactly the pp3 at `pp3` (no default profile, no sidecar) into the
/// JPEG `out`, at the image's displayed orientation and full size.
pub fn render_jpeg(
    pp3: &Path,
    nef: &Path,
    out: &Path,
    jpeg_quality: u8,
    metadata: Metadata,
) -> Result<()> {
    let mut command = Command::new("rawtherapee-cli");
    // -q: no cache; -Y: overwrite; -js3: no chroma subsampling.
    command
        .args(["-q", "-Y", "-js3"])
        .arg(format!("-j{jpeg_quality}"))
        .arg("-p")
        .arg(pp3);
    let layer = match metadata {
        Metadata::AsProfile => None,
        Metadata::Camera => {
            let layer = out.with_extension("metadata.pp3");
            fs::write(&layer, CAMERA_METADATA)
                .with_context(|| format!("writing {}", layer.display()))?;
            command.arg("-p").arg(&layer);
            Some(layer)
        }
    };
    let result = command.arg("-o").arg(out).arg("-c").arg(nef).output();
    if let Some(layer) = layer {
        let _ = fs::remove_file(layer);
    }
    let result = result.context("running rawtherapee-cli")?;
    ensure!(
        result.status.success(),
        "rawtherapee-cli failed on {} ({}): {}",
        nef.display(),
        result.status,
        String::from_utf8_lossy(&result.stderr)
    );
    ensure!(out.is_file(), "rawtherapee-cli wrote no {}", out.display());
    Ok(())
}

/// Runs one conversion of `sample` into `work_dir` and returns the pp3 RawTherapee wrote.
pub fn default_profile_pp3(sample: &Path, work_dir: &Path) -> Result<DefaultProfile> {
    fs::create_dir_all(work_dir)?;
    let stem = sample
        .file_stem()
        .context("sample NEF has no file name")?
        .to_string_lossy()
        .into_owned();
    let jpg = work_dir.join(format!("{stem}.jpg"));
    let pp3 = work_dir.join(format!("{stem}.jpg.pp3"));
    let _ = fs::remove_file(&pp3);
    info!(sample = %sample.display(), "resolving the default profile with rawtherapee-cli");
    let out = Command::new("rawtherapee-cli")
        // -O writes the pp3 beside the output; -f and -j1 make the throwaway JPEG cheap.
        .args(["-q", "-d", "-f", "-Y", "-j1"])
        .arg("-O")
        .arg(work_dir.join(""))
        .arg("-c")
        .arg(sample)
        .output()
        .context("running rawtherapee-cli")?;
    ensure!(
        out.status.success(),
        "rawtherapee-cli failed ({}): {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = fs::read_to_string(&pp3)
        .with_context(|| format!("rawtherapee-cli wrote no {}", pp3.display()))?;
    let _ = fs::remove_file(&jpg);
    DefaultProfile::parse(text).with_context(|| format!("in {}", pp3.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_profile_without_rules_or_builder() {
        let dir = tempfile::tempdir().unwrap();
        check_profile_is_shared(dir.path(), false).unwrap();
        fs::write(dir.path().join("dynamicprofile.cfg"), "\n").unwrap();
        check_profile_is_shared(dir.path(), false).unwrap();
    }

    #[test]
    fn refuses_dynamic_profile_rules() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("dynamicprofile.cfg"),
            "[1000_0]\nprofilepath=${G}/Pop 1\n",
        )
        .unwrap();
        assert!(check_profile_is_shared(dir.path(), false).is_err());
    }

    #[test]
    fn refuses_custom_profile_builder() {
        let dir = tempfile::tempdir().unwrap();
        assert!(check_profile_is_shared(dir.path(), true).is_err());
    }
}
