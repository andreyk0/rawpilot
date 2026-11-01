//! Per-shot derived data, from the cache where possible and computed in parallel otherwise:
//! features for every shot, and the neutral development when the judge is shown one.

use anyhow::Result;
use image::RgbImage;
use rayon::prelude::*;
use tracing::{info, warn};

use crate::cache::{self, Cache, Kind};
use crate::config::{DevelopConfig, FeaturesConfig};
use crate::develop::{self, DevelopInfo};
use crate::features::{self, Features};
use crate::preview;
use crate::shot::Shot;

/// A decoded 45 MP preview holds ~140 MB of RGB plus ~45 MB of luma while it is measured.
/// Eight in flight keeps the peak near 1.5 GB whatever the core count.
const DECODE_THREADS_MAX: usize = 8;
/// dcraw_emu runs on ~2 cores and hands back ~70 MB of 16-bit half-size RGB per frame.
const DEVELOP_THREADS_MAX: usize = 4;

pub struct Analyzed {
    pub shot: Shot,
    pub features: Features,
    /// Present when the judge is shown developed previews and this frame developed.
    pub developed: Option<DevelopInfo>,
}

/// Shots whose preview cannot be read or decoded are logged and dropped.
pub fn analyze(shots: Vec<Shot>, cache: &Cache, cfg: &FeaturesConfig) -> Result<Vec<Analyzed>> {
    let total = shots.len();
    let results = cached_par(
        &shots,
        cache,
        cache::FEATURES,
        cfg,
        DECODE_THREADS_MAX,
        |shot| {
            let rgb = preview::decode(shot)?;
            let (f, thumb) = features::measure(shot, &rgb, cfg);
            Ok((f, thumb, cfg.thumb_jpeg_quality))
        },
    )?;
    let out: Vec<Analyzed> = shots
        .into_iter()
        .zip(results)
        .filter_map(|(shot, r)| match r {
            Ok(features) => Some(Analyzed {
                shot,
                features,
                developed: None,
            }),
            Err(e) => {
                warn!(file = %shot.file.display(), "skipping: {e:#}");
                None
            }
        })
        .collect();
    if out.len() < total {
        warn!("features: {} of {total} shots skipped", total - out.len());
    }
    Ok(out)
}

/// Fills `developed` for every shot. A frame that fails to develop is logged and keeps
/// `None`; its burst is then judged on camera previews.
pub fn develop_all(
    shots: &mut [Analyzed],
    cache: &Cache,
    cfg: &DevelopConfig,
    long_edge: u32,
    jpeg_quality: u8,
) -> Result<()> {
    let plain: Vec<Shot> = shots.iter().map(|a| a.shot.clone()).collect();
    let params = (cfg, long_edge);
    let results = cached_par(
        &plain,
        cache,
        cache::DEVELOPED,
        &params,
        DEVELOP_THREADS_MAX,
        |shot| {
            let (lin, raw_clipped) = develop::decode(&shot.file, long_edge)?;
            let (img, info) = develop::render(&lin, raw_clipped, cfg);
            Ok((info, img, jpeg_quality))
        },
    )?;
    for (a, r) in shots.iter_mut().zip(results) {
        match r {
            Ok(info) => a.developed = Some(info),
            Err(e) => {
                warn!(file = %a.shot.name(), "not developed, judged on camera preview: {e:#}")
            }
        }
    }
    Ok(())
}

/// `compute` for each shot whose cached entry is missing or stale, in a pool of at most
/// `threads`; results in shot order. Logs how many came from the cache.
fn cached_par<T, P, F>(
    shots: &[Shot],
    cache: &Cache,
    kind: Kind,
    params: &P,
    threads: usize,
    compute: F,
) -> Result<Vec<Result<T>>>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send,
    P: serde::Serialize + Sync,
    F: Fn(&Shot) -> Result<(T, RgbImage, u8)> + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.min(std::thread::available_parallelism()?.get()))
        .build()?;
    let results: Vec<Result<(T, bool)>> = pool.install(|| {
        shots
            .par_iter()
            .map(|shot| {
                let key = Cache::key(kind, shot, params)?;
                if let Some(v) = cache.load(kind, shot, &key) {
                    return Ok((v, true));
                }
                let (v, img, quality) = compute(shot)?;
                cache.store(kind, shot, &key, &v, &img, quality)?;
                Ok((v, false))
            })
            .collect()
    });
    let cached = results
        .iter()
        .filter(|r| matches!(r, Ok((_, true))))
        .count();
    let failed = results.iter().filter(|r| r.is_err()).count();
    info!(
        kind = kind.name(),
        "{} shots: {cached} cached, {} computed, {failed} failed",
        shots.len(),
        shots.len() - cached - failed
    );
    Ok(results.into_iter().map(|r| r.map(|(v, _)| v)).collect())
}
