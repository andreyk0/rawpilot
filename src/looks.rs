//! Pre-rendered look options: what `process` precomputes so that nothing renders while the
//! photographer goes through a shoot in `rawpilot review`.
//!
//! One *render* per (shot, look option). Its pp3 is the complete profile the option would
//! write (`pp3::CompletePp3`: the default profile, the option's partial, the exposure look),
//! so a render shows exactly what choosing the option will produce. `rawtherapee-cli`
//! renders it once at full size; from that the module keeps a *fit view* (the whole frame at
//! `review.fit_long_edge_px`) and a 1:1 crop per region (`regions`: the AF box, each located
//! face, or the centre), and deletes the full render.
//!
//! # Layout
//!
//! ```text
//! <dir>/.rawpilot/looks/<NEF name>/<option>/
//!     manifest.json   what the UI reads: `Manifest`
//!     profile.pp3     the complete pp3 that was rendered, which `choose` writes
//!     fit.jpg
//!     region-<n>.jpg  one per `Manifest::regions` entry
//! ```
//!
//! A render is built in `<option>.partial/` and renamed into place, so `<option>/` is either
//! absent or whole, and an interrupted run is redone by the next one.
//!
//! # Keys and freshness
//!
//! A render is a function of its inputs, and two keys record which. `Manifest::inputs_key` is
//! the `Cache::key_under` of the shot's identity, `VERSION`, the digest of the option's
//! partial for the shot's ISO, the exposure look, the render parameters and the region list:
//! every input but RawTherapee's default profile, which takes seconds to resolve
//! (`Inputs`). `Manifest::key` adds the digest of the complete pp3 (`Job`). A render counts
//! only when [`Fresh::check`] has compared its manifest with today's inputs; `process` redoes
//! every render that fails the check, and `review` neither shows nor writes one. Before the
//! default profile is known only the inputs key can be compared (`Against::Inputs`).
//!
//! # Reading
//!
//! `Store::standing` returns the render of an option for a shot as [`Standing`]: missing (not
//! rendered, a failed render, a render in progress), stale, or fresh. `profile.pp3` of a
//! fresh render is what choosing it writes.
//!
//! # Rendering
//!
//! `render_all` runs at most `review.render_jobs` `rawtherapee-cli` processes at once and
//! reports each finished job through a callback (`Progress`), from the worker thread. A job
//! that fails is recorded in the `Report` and logged, never fatal to the batch, and has left
//! no manifest, so the next run tries it again. Full renders and their pp3s are written under
//! `<dir>/.rawpilot/tmp/` and removed on success and failure alike; the directory is emptied
//! when a batch starts, which is safe because the run holds the folder's `lock::FolderLock`
//! and no other rawpilot is working in it.

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use image::RgbImage;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::cache::{Cache, write_atomic};
use crate::config::{OptionName, ReviewConfig};
use crate::facecrop::Box2d;
use crate::option::LookOption;
use crate::pp3::{CompletePp3, DefaultProfile, Look, Partial};
use crate::regions::{self, Cut, Region, Sizing};
use crate::rpl::Rpl;
use crate::rtcli;
use crate::shot::{PixelRect, Shot};

/// Bumped when code changes the files a given key produces: cutting the fit view and the
/// crops, resizing, JPEG encoding.
const VERSION: u32 = 1;
const MANIFEST_FILE: &str = "manifest.json";
const PROFILE_FILE: &str = "profile.pp3";
const FIT_FILE: &str = "fit.jpg";

/// Everything besides the shot, the pp3 and the regions that determines a render's files.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RenderParams {
    pub fit_long_edge_px: u32,
    pub fit_jpeg_quality: u8,
    pub region_jpeg_quality: u8,
    pub render_jpeg_quality: u8,
    pub sizing: Sizing,
}

impl RenderParams {
    pub fn of(cfg: &ReviewConfig) -> RenderParams {
        RenderParams {
            fit_long_edge_px: cfg.fit_long_edge_px,
            fit_jpeg_quality: cfg.fit_jpeg_quality,
            region_jpeg_quality: cfg.region_jpeg_quality,
            render_jpeg_quality: cfg.render_jpeg_quality,
            sizing: Sizing::of(cfg),
        }
    }
}

/// An image file of a render and its size in pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct View {
    /// Relative to the render's directory.
    pub file: String,
    pub w: u32,
    pub h: u32,
}

/// A 1:1 crop of the full render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionView {
    /// What the crop is of: the AF box, a person's face, or the centre.
    pub region: Region,
    /// The area the crop covers, in fractions of the displayed image.
    pub crop: Box2d,
    /// The same area in pixels of the full render.
    pub rect: PixelRect,
    /// The file is `rect` at 1:1, or scaled down to `review.region_max_px`.
    pub view: View,
}

/// What the UI reads for one (shot, option).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub key: String,
    /// `Inputs::key`. `None` in a manifest written before it was kept, which never counts
    /// as fresh: such a render cannot be compared with today's inputs without the default
    /// profile, so it is made again.
    #[serde(default)]
    pub inputs_key: Option<String>,
    pub option: OptionName,
    /// blake3 of the complete pp3 that was rendered; `profile.pp3` is that text.
    pub pp3_blake3: String,
    /// The size of the full render, in displayed orientation.
    pub full_w: u32,
    pub full_h: u32,
    pub fit: View,
    pub regions: Vec<RegionView>,
}

/// A render on disk.
#[derive(Debug, Clone)]
pub struct Rendered {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Rendered {
    pub fn path(&self, view: &View) -> PathBuf {
        self.dir.join(&view.file)
    }

    /// The complete pp3 that was rendered, verified against the manifest's digest.
    pub fn pp3(&self) -> Result<CompletePp3> {
        let path = self.dir.join(PROFILE_FILE);
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        CompletePp3::from_stored(text, &self.manifest.pp3_blake3)
            .with_context(|| format!("in {}", path.display()))
    }
}

/// What a render is compared with: today's inputs of its (shot, option), with or without the
/// default profile.
#[derive(Debug, Clone, Copy)]
pub enum Against<'a> {
    /// Everything but the default profile, which is not resolved yet.
    Inputs(&'a Inputs),
    /// Every input.
    Job(&'a Job),
}

impl Against<'_> {
    fn shot(&self) -> &Shot {
        match self {
            Against::Inputs(i) => &i.shot,
            Against::Job(j) => &j.shot,
        }
    }

    fn option(&self) -> &OptionName {
        match self {
            Against::Inputs(i) => &i.option,
            Against::Job(j) => &j.option,
        }
    }
}

/// Why a render does not match today's inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Stale {
    #[error("its manifest does not record its inputs")]
    InputsNotRecorded,
    #[error(
        "the shot, the option's profile, the exposure look, the regions or the render parameters changed"
    )]
    InputsChanged,
    #[error("RawTherapee's default profile changed")]
    DefaultProfileChanged,
}

/// A render that [`Fresh::check`] found to match the inputs it was compared with. Every
/// consumer that shows a render or writes its pp3 asks for this type, so none can use a
/// render that was not compared.
///
/// Compared `Against::Inputs` it is known to match everything but the default profile; the
/// holder compares again `Against::Job` once the profile is known.
#[derive(Debug, Clone)]
pub struct Fresh(Rendered);

impl std::ops::Deref for Fresh {
    type Target = Rendered;

    fn deref(&self) -> &Rendered {
        &self.0
    }
}

impl Fresh {
    /// The one decision of whether a render is today's: `process` and `review` both use it.
    pub fn check(rendered: Rendered, against: Against<'_>) -> Result<Fresh, Stale> {
        let (inputs_key, job_key) = match against {
            Against::Inputs(i) => (&i.key, None),
            Against::Job(j) => (&j.inputs_key, Some(&j.key)),
        };
        match rendered.manifest.inputs_key.as_ref() {
            None => Err(Stale::InputsNotRecorded),
            Some(recorded) if recorded != inputs_key => Err(Stale::InputsChanged),
            Some(_) => match job_key {
                Some(key) if rendered.manifest.key != *key => Err(Stale::DefaultProfileChanged),
                Some(_) | None => Ok(Fresh(rendered)),
            },
        }
    }
}

/// A render of an option for a shot, as found on disk and compared.
#[derive(Debug)]
pub enum Standing {
    /// None yet: not rendered, a failed render, a render in progress.
    Missing,
    Stale(Stale),
    Fresh(Fresh),
}

/// `<dir>/.rawpilot/looks/`.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn new(cache_root: &Path) -> Store {
        Store {
            root: cache_root.join("looks"),
        }
    }

    /// `<dir>/.rawpilot/looks/`: every file of every render is below it.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn dir(&self, nef: &Path, option: &OptionName) -> Result<PathBuf> {
        let name = nef.file_name().context("NEF path has no file name")?;
        Ok(self.root.join(name).join(option.as_str()))
    }

    /// The render of the (shot, option) `against` names, compared with `against`. An error
    /// when a render is there and cannot be read.
    pub fn standing(&self, against: Against<'_>) -> Result<Standing> {
        Ok(
            match self.rendered(&against.shot().file, against.option())? {
                None => Standing::Missing,
                Some(rendered) => match Fresh::check(rendered, against) {
                    Ok(fresh) => Standing::Fresh(fresh),
                    Err(why) => Standing::Stale(why),
                },
            },
        )
    }

    /// Removes the renders of options that are not in `configured`: every directory directly
    /// under a NEF's directory in the store whose name is not a configured option, which
    /// includes the leftovers of a render that was interrupted. Nothing outside the store is
    /// touched, and a symlink is removed as a link, never followed. Returns how many
    /// directories went.
    pub fn prune_unconfigured(&self, configured: &[LookOption]) -> Result<usize> {
        let listing = |dir: &Path| -> Result<Vec<fs::DirEntry>> {
            match fs::read_dir(dir) {
                Ok(entries) => entries
                    .collect::<std::io::Result<_>>()
                    .with_context(|| format!("listing {}", dir.display())),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
                Err(e) => Err(e).with_context(|| format!("listing {}", dir.display())),
            }
        };
        let mut removed = 0;
        for shot_dir in listing(&self.root)? {
            if !shot_dir.file_type()?.is_dir() {
                continue;
            }
            for entry in listing(&shot_dir.path())? {
                let name = entry.file_name();
                let keep = name
                    .to_str()
                    .is_some_and(|n| configured.iter().any(|o| o.name.as_str() == n));
                if keep || !entry.file_type()?.is_dir() {
                    continue;
                }
                fs::remove_dir_all(entry.path())
                    .with_context(|| format!("removing {}", entry.path().display()))?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// The render of `option` for `nef`: `None` when there is none, an error when there is
    /// one that cannot be read (a manifest that does not parse, a file it names missing).
    pub fn rendered(&self, nef: &Path, option: &OptionName) -> Result<Option<Rendered>> {
        let dir = self.dir(nef, option)?;
        let text = match fs::read_to_string(dir.join(MANIFEST_FILE)) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        };
        let manifest: Manifest = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a manifest", dir.join(MANIFEST_FILE).display()))?;
        ensure!(
            manifest.option == *option,
            "{} is the manifest of option {}",
            dir.display(),
            manifest.option
        );
        let rendered = Rendered { dir, manifest };
        let files = std::iter::once(PROFILE_FILE.to_owned())
            .chain(std::iter::once(rendered.manifest.fit.file.clone()))
            .chain(
                rendered
                    .manifest
                    .regions
                    .iter()
                    .map(|r| r.view.file.clone()),
            );
        for file in files {
            ensure!(
                rendered.dir.join(&file).is_file(),
                "{} names {file}, which is missing",
                rendered.dir.display()
            );
        }
        Ok(Some(rendered))
    }
}

/// Every input of a render's key except RawTherapee's default profile: what can be compared
/// with a render on disk before the profile is resolved.
#[derive(Debug, Clone)]
pub struct Inputs {
    shot: Shot,
    option: OptionName,
    partial: Partial,
    look: Option<Look>,
    regions: Vec<Region>,
    key: String,
}

#[derive(Serialize)]
struct InputsKeyParts<'a> {
    partial_blake3: &'a str,
    look: Option<Look>,
    params: &'a RenderParams,
    regions: &'a [Region],
}

#[derive(Serialize)]
struct JobKeyParts<'a> {
    inputs_key: &'a str,
    pp3_blake3: &'a str,
}

#[cfg(test)]
impl Inputs {
    /// The key a manifest must carry as `inputs_key` to match these inputs.
    pub fn key(&self) -> &str {
        &self.key
    }
}

impl Inputs {
    /// The inputs for `shot` through `option`. The exposure look and the faces come from the
    /// shot's `.rpl`, the one place `process` leaves what it measured and found, so `process`
    /// and `review` make the same render, with the same key, for the same inputs.
    pub fn new(
        shot: &Shot,
        option: &LookOption,
        record: &Rpl,
        params: &RenderParams,
    ) -> Result<Inputs> {
        let partial = option.partial_for(shot.exposure.iso).clone();
        let regions = regions::plan(shot, record.faces.as_deref());
        let key = Cache::key_under(
            "look-render-inputs",
            VERSION,
            shot,
            &InputsKeyParts {
                partial_blake3: &partial.digest(),
                look: record.measured_look,
                params,
                regions: &regions,
            },
        )?;
        Ok(Inputs {
            shot: shot.clone(),
            option: option.name.clone(),
            partial,
            look: record.measured_look,
            regions,
            key,
        })
    }
}

/// One render to make: a shot through one option's complete pp3.
#[derive(Debug, Clone)]
pub struct Job {
    shot: Shot,
    option: OptionName,
    pp3: CompletePp3,
    regions: Vec<Region>,
    inputs_key: String,
    key: String,
}

#[cfg(test)]
impl Job {
    /// The key a manifest must carry as `key` for the render to match this job.
    pub fn key(&self) -> &str {
        &self.key
    }
}

impl Job {
    /// The job for `inputs` over RawTherapee's `default` profile.
    pub fn of(inputs: Inputs, default: &DefaultProfile) -> Result<Job> {
        let pp3 = CompletePp3::compose(default, &inputs.partial, inputs.look)?;
        let key = Cache::key_under(
            "look-render",
            VERSION,
            &inputs.shot,
            &JobKeyParts {
                inputs_key: &inputs.key,
                pp3_blake3: pp3.digest(),
            },
        )?;
        Ok(Job {
            shot: inputs.shot,
            option: inputs.option,
            pp3,
            regions: inputs.regions,
            inputs_key: inputs.key,
            key,
        })
    }

    /// The job for `shot` through `option`: `Inputs::new` over `default`.
    pub fn new(
        shot: &Shot,
        option: &LookOption,
        default: &DefaultProfile,
        record: &Rpl,
        params: &RenderParams,
    ) -> Result<Job> {
        Job::of(Inputs::new(shot, option, record, params)?, default)
    }
}

/// How one job ended.
#[derive(Debug)]
pub enum Status {
    /// A render under the job's key was already there.
    Cached,
    Rendered {
        took: Duration,
    },
    Failed {
        error: String,
    },
}

/// One job finished; `done` of `total` jobs have.
#[derive(Debug)]
pub struct Progress<'a> {
    /// The job's position in the jobs given to `render_all`.
    pub job: usize,
    pub done: usize,
    pub total: usize,
    pub shot: &'a str,
    pub option: &'a OptionName,
    pub status: &'a Status,
}

/// How every job of a `render_all` call ended.
#[derive(Debug, Default)]
pub struct Report {
    /// One per job, in the order the jobs were given.
    pub statuses: Vec<Status>,
}

impl Report {
    pub fn cached(&self) -> usize {
        self.statuses
            .iter()
            .filter(|s| matches!(s, Status::Cached))
            .count()
    }

    pub fn rendered(&self) -> usize {
        self.statuses
            .iter()
            .filter(|s| matches!(s, Status::Rendered { .. }))
            .count()
    }

    pub fn failed(&self) -> usize {
        self.statuses
            .iter()
            .filter(|s| matches!(s, Status::Failed { .. }))
            .count()
    }
}

/// Runs `jobs`, at most `jobs_max` at a time, calling `progress` as each finishes. Fails only
/// when the cache directories cannot be prepared; a job's failure is in the report, whose
/// statuses line up with `jobs`.
pub fn render_all(
    jobs: &[Job],
    params: &RenderParams,
    jobs_max: usize,
    cache_root: &Path,
    progress: &(dyn Fn(&Progress) + Sync),
) -> Result<Report> {
    assert!(jobs_max >= 1, "at least one render at a time");
    let store = Store::new(cache_root);
    let tmp = cache_root.join("tmp");
    // A batch interrupted before it removed its temporary files leaves them here.
    if tmp.exists() {
        fs::remove_dir_all(&tmp).with_context(|| format!("clearing {}", tmp.display()))?;
    }
    fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    fs::create_dir_all(&store.root)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs_max)
        .build()?;
    let done = AtomicUsize::new(0);
    let total = jobs.len();
    let statuses: Vec<Status> = pool.install(|| {
        jobs.par_iter()
            .enumerate()
            .map(|(n, job)| {
                let status = match run(job, params, &store, &tmp, n) {
                    Ok(status) => status,
                    Err(e) => Status::Failed {
                        error: format!("{e:#}"),
                    },
                };
                progress(&Progress {
                    job: n,
                    done: done.fetch_add(1, Ordering::SeqCst) + 1,
                    total,
                    shot: &job.shot.name(),
                    option: &job.option,
                    status: &status,
                });
                status
            })
            .collect()
    });
    for (job, status) in jobs.iter().zip(&statuses) {
        if let Status::Failed { error } = status {
            warn!(shot = %job.shot.name(), option = %job.option, "render failed: {error}");
        }
    }
    let report = Report { statuses };
    info!(
        "renders: {} made, {} already cached, {} failed",
        report.rendered(),
        report.cached(),
        report.failed()
    );
    Ok(report)
}

/// Removes its files when dropped: after a render that succeeded and after one that failed.
struct Temp(Vec<PathBuf>);

impl Drop for Temp {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = fs::remove_file(p);
        }
    }
}

fn run(job: &Job, params: &RenderParams, store: &Store, tmp: &Path, n: usize) -> Result<Status> {
    let dir = store.dir(&job.shot.file, &job.option)?;
    if let Ok(Standing::Fresh(_)) = store.standing(Against::Job(job)) {
        return Ok(Status::Cached);
    }
    let started = Instant::now();
    let stem = format!("{n}-{}-{}", job.shot.name(), job.option);
    let pp3_path = tmp.join(format!("{stem}.pp3"));
    let jpg_path = tmp.join(format!("{stem}.jpg"));
    let _cleanup = Temp(vec![pp3_path.clone(), jpg_path.clone()]);
    fs::write(&pp3_path, job.pp3.as_str())
        .with_context(|| format!("writing {}", pp3_path.display()))?;
    rtcli::render_jpeg(
        &pp3_path,
        &job.shot.file,
        &jpg_path,
        params.render_jpeg_quality,
        rtcli::Metadata::AsProfile,
    )?;
    let full = image::open(&jpg_path)
        .with_context(|| format!("decoding {}", jpg_path.display()))?
        .into_rgb8();
    install(job, params, &full, &dir)?;
    Ok(Status::Rendered {
        took: started.elapsed(),
    })
}

/// Cuts the views from `full`, writes them with the manifest into a fresh directory next to
/// `dir`, and swaps it in.
fn install(job: &Job, params: &RenderParams, full: &RgbImage, dir: &Path) -> Result<()> {
    let partial = dir.with_extension("partial");
    if partial.exists() {
        fs::remove_dir_all(&partial)?;
    }
    fs::create_dir_all(&partial).with_context(|| format!("creating {}", partial.display()))?;
    let size = full.dimensions();

    let (fit_w, fit_h) = regions::scaled_down(size.0, size.1, params.fit_long_edge_px);
    write_view(
        &partial.join(FIT_FILE),
        &resized(full, fit_w, fit_h),
        params.fit_jpeg_quality,
    )?;
    let fit = View {
        file: FIT_FILE.to_owned(),
        w: fit_w,
        h: fit_h,
    };

    let mut regions = Vec::new();
    for (i, region) in job.regions.iter().enumerate() {
        let cut = Cut::of(region, size, &params.sizing);
        let r = cut.rect;
        let crop = imageops::crop_imm(full, r.x, r.y, r.w, r.h).to_image();
        let file = format!("region-{i}.jpg");
        write_view(
            &partial.join(&file),
            &resized(&crop, cut.out_w, cut.out_h),
            params.region_jpeg_quality,
        )?;
        regions.push(RegionView {
            region: *region,
            crop: cut.as_box(size),
            rect: r,
            view: View {
                file,
                w: cut.out_w,
                h: cut.out_h,
            },
        });
    }

    write_atomic(&partial.join(PROFILE_FILE), job.pp3.as_str().as_bytes())?;
    let manifest = Manifest {
        key: job.key.clone(),
        inputs_key: Some(job.inputs_key.clone()),
        option: job.option.clone(),
        pp3_blake3: job.pp3.digest().to_owned(),
        full_w: size.0,
        full_h: size.1,
        fit,
        regions,
    };
    write_atomic(
        &partial.join(MANIFEST_FILE),
        serde_json::to_string_pretty(&manifest)?.as_bytes(),
    )?;
    if dir.exists() {
        fs::remove_dir_all(dir).with_context(|| format!("replacing {}", dir.display()))?;
    }
    fs::rename(&partial, dir).with_context(|| format!("installing {}", dir.display()))
}

fn resized(img: &RgbImage, w: u32, h: u32) -> Cow<'_, RgbImage> {
    if img.dimensions() == (w, h) {
        Cow::Borrowed(img)
    } else {
        Cow::Owned(imageops::resize(img, w, h, FilterType::Lanczos3))
    }
}

fn write_view(path: &Path, img: &RgbImage, quality: u8) -> Result<()> {
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, quality).encode_image(img)?;
    write_atomic(path, &jpeg)
}

#[cfg(test)]
mod tests {
    //! Goal: a render is reused exactly when everything that determines it is unchanged, its
    //! views and manifest are cut from the full render as configured, and a failing render
    //! is reported without leaving anything behind. Method: keys are compared across
    //! single-input changes; `install` is run on a synthetic full render; failure is
    //! provoked with a NEF that rawtherapee-cli cannot read (or its absence).

    use std::sync::Mutex;

    use super::*;
    use crate::config::{OptionConfig, PartialSource};
    use crate::pp3::Look;
    use crate::rpl::FaceBox;
    use crate::shot::{AfBox, Exposure, Orientation, PreviewSpan};

    const DEFAULT: &str =
        "[General]\nRank=1\n\n[Exposure]\nCompensation=0\nHistogramMatching=true\n";

    fn params() -> RenderParams {
        RenderParams {
            fit_long_edge_px: 2560,
            fit_jpeg_quality: 88,
            region_jpeg_quality: 92,
            render_jpeg_quality: 95,
            sizing: Sizing {
                af_min_px: 512,
                face_pad_fraction: 0.25,
                centre_px: 1024,
                max_px: 1200,
            },
        }
    }

    fn shot_in(dir: &Path, af: Option<AfBox>) -> Shot {
        let file = dir.join("DSC_1.NEF");
        if !file.exists() {
            fs::write(&file, b"not a NEF").unwrap();
        }
        Shot {
            file,
            file_size: 9,
            taken_at: chrono::DateTime::parse_from_rfc3339("2026-10-18T14:00:00+00:00").unwrap(),
            preview: PreviewSpan { offset: 0, len: 0 },
            af,
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

    fn option(name: &str, partial: &str) -> LookOption {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("p.pp3");
        fs::write(&file, partial).unwrap();
        LookOption::load_all(&[OptionConfig {
            name: name.to_owned().try_into().unwrap(),
            key: 'x',
            partial: PartialSource::File(file),
        }])
        .unwrap()
        .remove(0)
    }

    fn key(
        shot: &Shot,
        default: &str,
        partial: &str,
        look: Option<Look>,
        faces: Option<&[FaceBox]>,
        p: &RenderParams,
    ) -> String {
        let record = Rpl {
            measured_look: look,
            faces: faces.map(<[FaceBox]>::to_vec),
            ..Rpl::default()
        };
        Job::new(
            shot,
            &option("Film", partial),
            &DefaultProfile::parse(default.to_owned()).unwrap(),
            &record,
            p,
        )
        .unwrap()
        .key
    }

    fn af() -> AfBox {
        AfBox {
            rect: PixelRect {
                x: 100,
                y: 100,
                w: 50,
                h: 50,
            },
            space_w: 1000,
            space_h: 700,
        }
    }

    fn face(x0: f64) -> FaceBox {
        FaceBox {
            person: 1,
            at: Box2d {
                x0,
                y0: 0.2,
                x1: x0 + 0.1,
                y1: 0.3,
            },
        }
    }

    #[test]
    fn the_key_covers_the_pp3_the_parameters_and_the_regions() {
        let dir = tempfile::tempdir().unwrap();
        let shot = shot_in(dir.path(), Some(af()));
        let partial = "[Exposure]\nCurveMode=FilmLike\n";
        let look = Some(Look {
            compensation_ev: 1.0,
            highlight_compr: 100,
        });
        let faces = [face(0.1)];
        let base = key(&shot, DEFAULT, partial, look, Some(&faces), &params());
        assert_eq!(
            base,
            key(&shot, DEFAULT, partial, look, Some(&faces), &params()),
            "the key is deterministic"
        );

        let differs = |k: String, what: &str| assert_ne!(base, k, "{what} must change the key");
        differs(
            key(
                &shot,
                &DEFAULT.replace("Rank=1", "Rank=1\nColorLabel=0"),
                partial,
                look,
                Some(&faces),
                &params(),
            ),
            "the default profile",
        );
        differs(
            key(
                &shot,
                DEFAULT,
                "[Exposure]\nCurveMode=Standard\n",
                look,
                Some(&faces),
                &params(),
            ),
            "the option's partial",
        );
        differs(
            key(
                &shot,
                DEFAULT,
                partial,
                Some(Look {
                    compensation_ev: 1.5,
                    highlight_compr: 100,
                }),
                Some(&faces),
                &params(),
            ),
            "the exposure look",
        );
        differs(
            key(&shot, DEFAULT, partial, None, Some(&faces), &params()),
            "no exposure look",
        );
        differs(
            key(&shot, DEFAULT, partial, look, Some(&[face(0.5)]), &params()),
            "a moved face",
        );
        differs(
            key(&shot, DEFAULT, partial, look, None, &params()),
            "no faces",
        );
        differs(
            key(
                &shot_in(dir.path(), None),
                DEFAULT,
                partial,
                look,
                Some(&faces),
                &params(),
            ),
            "no AF box",
        );
        for p in [
            RenderParams {
                fit_long_edge_px: 1920,
                ..params()
            },
            RenderParams {
                fit_jpeg_quality: 80,
                ..params()
            },
            RenderParams {
                region_jpeg_quality: 80,
                ..params()
            },
            RenderParams {
                render_jpeg_quality: 90,
                ..params()
            },
            RenderParams {
                sizing: Sizing {
                    max_px: 800,
                    ..params().sizing
                },
                ..params()
            },
            RenderParams {
                sizing: Sizing {
                    face_pad_fraction: 0.5,
                    ..params().sizing
                },
                ..params()
            },
        ] {
            differs(
                key(&shot, DEFAULT, partial, look, Some(&faces), &p),
                "a render parameter",
            );
        }
    }

    fn gradient(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        })
    }

    #[test]
    fn install_cuts_the_views_and_replaces_an_older_render() {
        let dir = tempfile::tempdir().unwrap();
        let shot = shot_in(dir.path(), Some(af()));
        let job = Job::new(
            &shot,
            &option("Film", "[Exposure]\nCurveMode=FilmLike\n"),
            &DefaultProfile::parse(DEFAULT.to_owned()).unwrap(),
            &Rpl {
                faces: Some(vec![face(0.5)]),
                ..Rpl::default()
            },
            &params(),
        )
        .unwrap();
        let store = Store::new(dir.path());
        let target = store.dir(&shot.file, &job.option).unwrap();
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("stale.txt"), "x").unwrap();

        // 4000×2000: the fit view scales to 2560×1280; the AF box (50×50 of 1000×700) is
        // padded to 512 px; the face (0.1 of 4000 = 400 px wide, 0.1 of 2000 = 200 high) to
        // 600×300.
        install(&job, &params(), &gradient(4000, 2000), &target).unwrap();

        assert!(!target.join("stale.txt").exists());
        assert!(!target.with_extension("partial").exists());
        let r = store.rendered(&shot.file, &job.option).unwrap().unwrap();
        assert_eq!(r.manifest.key, job.key);
        assert_eq!((r.manifest.full_w, r.manifest.full_h), (4000, 2000));
        assert_eq!((r.manifest.fit.w, r.manifest.fit.h), (2560, 1280));
        let on_disk = |v: &View| image::image_dimensions(r.path(v)).unwrap();
        assert_eq!(on_disk(&r.manifest.fit), (2560, 1280));
        let sizes: Vec<(u32, u32)> = r
            .manifest
            .regions
            .iter()
            .map(|x| (x.view.w, x.view.h))
            .collect();
        assert_eq!(sizes, [(512, 512), (600, 300)]);
        for region in &r.manifest.regions {
            assert_eq!(on_disk(&region.view), (region.view.w, region.view.h));
        }
        assert_eq!(r.pp3().unwrap(), job.pp3);
    }

    #[test]
    fn a_manifest_naming_a_missing_file_is_an_error_and_none_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let shot = shot_in(dir.path(), None);
        let name: OptionName = "Film".to_owned().try_into().unwrap();
        let store = Store::new(dir.path());
        assert!(store.rendered(&shot.file, &name).unwrap().is_none());
        let job = Job::new(
            &shot,
            &option("Film", ""),
            &DefaultProfile::parse(DEFAULT.to_owned()).unwrap(),
            &Rpl::default(),
            &params(),
        )
        .unwrap();
        let target = store.dir(&shot.file, &name).unwrap();
        install(&job, &params(), &gradient(600, 400), &target).unwrap();
        fs::remove_file(target.join("region-0.jpg")).unwrap();
        assert!(store.rendered(&shot.file, &name).is_err());
    }

    #[test]
    fn a_failing_render_is_reported_leaves_nothing_and_does_not_stop_the_batch() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join(".rawpilot");
        let shot = shot_in(dir.path(), None);
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let jobs: Vec<Job> = ["Film", "Gentle"]
            .into_iter()
            .map(|n| Job::new(&shot, &option(n, ""), &default, &Rpl::default(), &params()).unwrap())
            .collect();
        let seen = Mutex::new(Vec::new());
        let report = render_all(&jobs, &params(), 2, &cache, &|p| {
            assert_eq!(p.total, 2);
            assert_eq!(jobs[p.job].option, *p.option, "the job index names the job");
            assert!(matches!(p.status, Status::Failed { .. }));
            seen.lock().unwrap().push(p.done);
        })
        .unwrap();
        assert_eq!(report.failed(), 2);
        assert_eq!((report.rendered(), report.cached()), (0, 0));
        assert_eq!(
            report.statuses.len(),
            jobs.len(),
            "one status per job, in the jobs' order"
        );
        let mut done = seen.into_inner().unwrap();
        done.sort_unstable();
        assert_eq!(done, [1, 2]);
        assert_eq!(fs::read_dir(cache.join("tmp")).unwrap().count(), 0);
        let store = Store::new(&cache);
        for j in &jobs {
            assert!(store.rendered(&shot.file, &j.option).unwrap().is_none());
        }
    }

    /// A render under the job's key is not redone; under another key it is.
    #[test]
    fn a_cached_render_is_not_redone() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join(".rawpilot");
        let shot = shot_in(dir.path(), None);
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let job = Job::new(
            &shot,
            &option("Film", ""),
            &default,
            &Rpl::default(),
            &params(),
        )
        .unwrap();
        let store = Store::new(&cache);
        install(
            &job,
            &params(),
            &gradient(600, 400),
            &store.dir(&shot.file, &job.option).unwrap(),
        )
        .unwrap();
        let report = render_all(std::slice::from_ref(&job), &params(), 1, &cache, &|p| {
            assert!(matches!(p.status, Status::Cached));
        })
        .unwrap();
        assert_eq!(
            (report.cached(), report.rendered(), report.failed()),
            (1, 0, 0)
        );
        // The same inputs but another partial: another key, so it is rendered (and here fails).
        let changed = Job::new(
            &shot,
            &option("Film", "[Exposure]\nCurveMode=FilmLike\n"),
            &default,
            &Rpl::default(),
            &params(),
        )
        .unwrap();
        let report = render_all(&[changed], &params(), 1, &cache, &|_| {}).unwrap();
        assert_eq!((report.cached(), report.failed()), (0, 1));
    }

    fn job_over(shot: &Shot, default: &str, partial: &str) -> Job {
        Job::new(
            shot,
            &option("Film", partial),
            &DefaultProfile::parse(default.to_owned()).unwrap(),
            &Rpl::default(),
            &params(),
        )
        .unwrap()
    }

    /// Goal: the inputs key is every input of the full key but the default profile, so a
    /// render can be compared before the profile is known. Method: keys compared across
    /// single-input changes. Mutation: leave the partial's digest out of `InputsKeyParts`:
    /// the partial row goes red; hash the default profile into it: the first assertion does.
    #[test]
    fn the_inputs_key_is_the_key_without_the_default_profile() {
        let dir = tempfile::tempdir().unwrap();
        let shot = shot_in(dir.path(), Some(af()));
        let partial = "[Exposure]\nCurveMode=FilmLike\n";
        let base = job_over(&shot, DEFAULT, partial);

        let other_default = job_over(&shot, &DEFAULT.replace("Rank=1", "Rank=2"), partial);
        assert_eq!(base.inputs_key, other_default.inputs_key);
        assert_ne!(base.key, other_default.key);

        let inputs = |shot: &Shot, partial: &str, p: &RenderParams| {
            Inputs::new(shot, &option("Film", partial), &Rpl::default(), p)
                .unwrap()
                .key
        };
        assert_eq!(base.inputs_key, inputs(&shot, partial, &params()));
        assert_ne!(
            base.inputs_key,
            inputs(&shot, "[Exposure]\nCurveMode=Standard\n", &params()),
            "the option's partial"
        );
        assert_ne!(
            base.inputs_key,
            inputs(&shot_in(dir.path(), None), partial, &params()),
            "the regions"
        );
        assert_ne!(
            base.inputs_key,
            inputs(
                &shot,
                partial,
                &RenderParams {
                    fit_jpeg_quality: 80,
                    ..params()
                }
            ),
            "a render parameter"
        );
        let looked = Inputs::new(
            &shot,
            &option("Film", partial),
            &Rpl {
                measured_look: Some(Look {
                    compensation_ev: 1.5,
                    highlight_compr: 100,
                }),
                ..Rpl::default()
            },
            &params(),
        )
        .unwrap();
        assert_ne!(base.inputs_key, looked.key, "the exposure look");
    }

    /// Goal: a render is fresh exactly when its manifest matches today's inputs, with the
    /// strength of the check chosen by whether the default profile is known. Method: one
    /// installed render compared against changed inputs, a changed default profile, and a
    /// manifest from before `inputs_key` existed. Mutation: drop the `inputs_key` comparison
    /// in `Fresh::check`: the changed-partial row goes red; drop the job-key comparison: the
    /// default-profile row does; treat `None` as fresh: the legacy row does.
    #[test]
    fn a_render_is_fresh_only_against_todays_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join(".rawpilot");
        let store = Store::new(&cache);
        let shot = shot_in(dir.path(), None);
        let partial = "[Exposure]\nCurveMode=FilmLike\n";
        let job = job_over(&shot, DEFAULT, partial);
        let inputs = |partial: &str| {
            Inputs::new(&shot, &option("Film", partial), &Rpl::default(), &params()).unwrap()
        };
        assert!(matches!(
            store.standing(Against::Inputs(&inputs(partial))).unwrap(),
            Standing::Missing
        ));
        let target = store.dir(&shot.file, &job.option).unwrap();
        install(&job, &params(), &gradient(600, 400), &target).unwrap();

        let stale_because = |against: Against| match store.standing(against).unwrap() {
            Standing::Stale(why) => Some(why),
            Standing::Fresh(_) | Standing::Missing => None,
        };
        assert!(matches!(
            store.standing(Against::Inputs(&inputs(partial))).unwrap(),
            Standing::Fresh(_)
        ));
        assert!(matches!(
            store.standing(Against::Job(&job)).unwrap(),
            Standing::Fresh(_)
        ));
        assert_eq!(
            stale_because(Against::Inputs(&inputs("[Exposure]\nCurveMode=Standard\n"))),
            Some(Stale::InputsChanged)
        );
        let retuned = job_over(&shot, &DEFAULT.replace("Rank=1", "Rank=2"), partial);
        assert!(
            matches!(
                store.standing(Against::Inputs(&inputs(partial))).unwrap(),
                Standing::Fresh(_)
            ),
            "the quick check cannot see the default profile"
        );
        assert_eq!(
            stale_because(Against::Job(&retuned)),
            Some(Stale::DefaultProfileChanged)
        );

        let manifest = target.join(MANIFEST_FILE);
        let mut json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("inputs_key");
        fs::write(&manifest, json.to_string()).unwrap();
        assert_eq!(
            stale_because(Against::Inputs(&inputs(partial))),
            Some(Stale::InputsNotRecorded)
        );
        assert_eq!(
            stale_because(Against::Job(&job)),
            Some(Stale::InputsNotRecorded)
        );
    }

    /// `process` redoes a render made under another default profile. Mutation: have `run`
    /// compare nothing but the file's presence: the render is reported cached.
    #[test]
    fn a_render_under_another_default_profile_is_redone() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join(".rawpilot");
        let shot = shot_in(dir.path(), None);
        let job = job_over(&shot, DEFAULT, "");
        install(
            &job,
            &params(),
            &gradient(600, 400),
            &Store::new(&cache).dir(&shot.file, &job.option).unwrap(),
        )
        .unwrap();
        let retuned = job_over(&shot, &DEFAULT.replace("Rank=1", "Rank=2"), "");
        let report = render_all(&[retuned], &params(), 1, &cache, &|_| {}).unwrap();
        // The NEF stand-in cannot be rendered, so the redo fails: it was attempted.
        assert_eq!((report.cached(), report.failed()), (0, 1));
    }

    /// Goal: pruning removes exactly the option directories under a NEF's directory in the
    /// store that are not configured, and nothing else. Method: a store with a configured
    /// option, two others (one an interrupted `.partial`), a stray file, a symlink to a
    /// directory outside the store, and a sibling of the store. Mutation: remove everything
    /// under the NEF directory: the configured and the stray file go; follow symlinks: the
    /// outside directory's file goes.
    #[test]
    fn pruning_removes_only_unconfigured_option_directories_of_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join(".rawpilot");
        let store = Store::new(&cache);
        let nef_dir = store.root().join("DSC_1.NEF");
        for d in ["Film", "Old", "Older.partial"] {
            fs::create_dir_all(nef_dir.join(d)).unwrap();
            fs::write(nef_dir.join(d).join("manifest.json"), "{}").unwrap();
        }
        fs::write(nef_dir.join("note.txt"), "x").unwrap();
        let outside = dir.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("keep.txt"), "x").unwrap();
        std::os::unix::fs::symlink(&outside, nef_dir.join("link")).unwrap();
        fs::create_dir_all(cache.join("tmp").join("Old")).unwrap();

        let removed = store.prune_unconfigured(&[option("Film", "")]).unwrap();

        assert_eq!(removed, 2);
        assert!(nef_dir.join("Film/manifest.json").is_file());
        assert!(!nef_dir.join("Old").exists());
        assert!(!nef_dir.join("Older.partial").exists());
        assert!(nef_dir.join("note.txt").is_file());
        assert!(nef_dir.join("link").is_symlink());
        assert!(outside.join("keep.txt").is_file());
        assert!(cache.join("tmp/Old").is_dir());
    }

    #[test]
    fn pruning_a_store_that_does_not_exist_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::new(&dir.path().join(".rawpilot"));
        assert_eq!(store.prune_unconfigured(&[]).unwrap(), 0);
    }

    /// Goal: the report's statuses line up with the jobs, whatever order the pool finishes
    /// them in. Method: a cached job and a failing one, given in both orders on two threads.
    /// Mutation: reverse the statuses in `render_all`: one order puts each status on the
    /// other job and the assertion goes red.
    #[test]
    fn the_report_lines_up_with_the_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join(".rawpilot");
        let shot = shot_in(dir.path(), None);
        let default = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
        let make = |name: &str, partial: &str| {
            Job::new(
                &shot,
                &option(name, partial),
                &default,
                &Rpl::default(),
                &params(),
            )
            .unwrap()
        };
        let cached = make("Film", "");
        install(
            &cached,
            &params(),
            &gradient(600, 400),
            &Store::new(&cache).dir(&shot.file, &cached.option).unwrap(),
        )
        .unwrap();
        // The failing job's NEF cannot be rendered by rawtherapee-cli.
        let failing = make("Gentle", "[Exposure]\nCurveMode=FilmLike\n");
        for jobs in [
            [cached.clone(), failing.clone()],
            [failing.clone(), cached.clone()],
        ] {
            let report = render_all(&jobs, &params(), 2, &cache, &|_| {}).unwrap();
            for (job, status) in jobs.iter().zip(&report.statuses) {
                assert_eq!(
                    matches!(status, Status::Cached),
                    job.option == cached.option,
                    "{}: {status:?}",
                    job.option
                );
            }
        }
    }
}
