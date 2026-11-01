//! `rawpilot cull` and `rawpilot process`: rate a folder of NEFs where RawTherapee reads
//! ratings, and for `process` also set the exposure a new pp3 opens with and pre-render the
//! look options for the shots worth reviewing.
//!
//! Scan → measure (and develop, for Gemini or the look) → group into scenes and bursts →
//! judge (burst, faces and scene passes, or local sharpness alone) → rate → report → write
//! ratings (`cull` creates a missing pp3 from the default profile; `process` also gives it the
//! look and keeps an owned pp3's look up to date) → `process` renders every look option of
//! each shot rated at least `review.prerender_min_rating`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use tracing::info;

use crate::analyze::{self, Analyzed};
use crate::cache::Cache;
use crate::config::{Config, Preview, ReviewConfig};
use crate::group::{self, Burst, Scene};
use crate::judge::{self, BurstJudgement};
use crate::lock::{FolderLock, Run};
use crate::looks::{self, RenderParams};
use crate::option::LookOption;
use crate::pp3::Look;
use crate::rate::{self, Decision};
use crate::rpl::{FaceBox, Rpl};
use crate::scene::{self, Role};
use crate::sidecar::{self, Mark, Store};
use crate::{faces, look, nef, report, rtcli, shortlist, writeback};

pub enum Judging {
    Gemini { api_key: String },
    LocalOnly,
}

/// What a run leaves behind besides ratings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outputs {
    /// `cull`: ratings only. A missing pp3 is created from the default profile without the
    /// look; an existing pp3's processing is never changed; nothing is rendered.
    RatingsOnly,
    /// `process`: ratings, the look in every pp3 rawpilot creates or still owns, and the
    /// look options rendered for the shots rated high enough.
    RatingsLooksAndRenders,
}

impl Outputs {
    /// Whether this run measures an exposure look for each shot: `process`, unless the
    /// configuration turns the look off. `cull` never does, so it leaves processing alone.
    fn measures_looks(self, cfg: &Config) -> bool {
        match self {
            Outputs::RatingsOnly => false,
            Outputs::RatingsLooksAndRenders => cfg.look.enabled,
        }
    }

    /// What the run tells the writeback about a shot whose measured look is `look`: `cull`
    /// says nothing, so the look recorded for the renders stays; `process` says what it
    /// found, none included.
    fn measure(self, look: Option<Look>) -> writeback::Measure {
        match self {
            Outputs::RatingsOnly => writeback::Measure::Skipped,
            Outputs::RatingsLooksAndRenders => writeback::Measure::Measured(look),
        }
    }
}

pub fn run(
    dir: &Path,
    cfg: &Config,
    judging: Judging,
    mode: writeback::Mode,
    outputs: Outputs,
) -> Result<()> {
    let folder = FolderLock::acquire(
        dir,
        match outputs {
            Outputs::RatingsOnly => Run::Cull,
            Outputs::RatingsLooksAndRenders => Run::Process,
        },
    )?;
    // Checked before any work is done, so a configuration rawpilot cannot write for fails the
    // run before Gemini is paid.
    let rt = sidecar::read_rt_options(&sidecar::default_rt_options_path()?)?;
    let store = rt.store();
    if store == Store::Pp3 || outputs == Outputs::RatingsLooksAndRenders {
        rtcli::check_profile_is_shared(&rt.config_dir, rt.custom_profile_builder)?;
    }
    let options = match outputs {
        Outputs::RatingsOnly => Vec::new(),
        Outputs::RatingsLooksAndRenders => LookOption::load_all(&cfg.review.options)?,
    };
    info!(
        ?store,
        "ratings go where RawTherapee reads them (ThumbnailRankColorMode)"
    );

    let cache = Cache::open(&folder)?;
    let mut shots = analyze::analyze(nef::scan(dir)?, &cache, &cfg.features)?;
    // The look is measured against the neutral development, which is made locally.
    let looking = outputs.measures_looks(cfg);
    let judged_on_development =
        matches!(judging, Judging::Gemini { .. }) && cfg.judge.preview == Preview::Developed;
    if judged_on_development || looking {
        analyze::develop_all(
            &mut shots,
            &cache,
            &cfg.develop,
            cfg.features.thumb_long_edge_px,
            cfg.features.thumb_jpeg_quality,
        )?;
    }
    let scenes = group_shots(&shots, cfg);
    let bursts: Vec<&Burst> = scenes.iter().flat_map(|s| s.bursts()).collect();
    let rel = rate::sharpness_rel_all(&shots, &bursts);

    let judged = match &judging {
        Judging::Gemini { api_key } => {
            gemini_passes(&shots, &scenes, &bursts, &rel, &cache, cfg, api_key)?
        }
        Judging::LocalOnly => Judged::local(&shots, &bursts, &rel),
    };
    let decisions = judged.rate(&bursts, &rel, cfg);
    let looks: Vec<Option<Look>> = if looking {
        shots
            .iter()
            .map(|a| look::for_shot(&cache, a, &cfg.look))
            .collect::<Result<_>>()?
    } else {
        vec![None; shots.len()]
    };

    let rows: Vec<report::Row> = (0..shots.len())
        .map(|i| report::Row {
            decision: &decisions[i],
            note: judged.notes[i].as_deref(),
            look: looks[i],
        })
        .collect();
    let report = report::write(&cache, &shots, &scenes, &rows)?;
    info!("report: {}", report.display());

    let analyses = analyses(&shots, &bursts, &judged, &decisions, &looks, cfg);
    let defaults = rtcli::DefaultProfiles::new(
        &rt,
        shots
            .first()
            .map(|a| a.shot.file.clone())
            .context("no readable shots")?,
        cache.root().join("rt-template"),
    );
    write_ratings(
        &targets(&shots, &decisions, &looks, &judged, analyses, outputs),
        &rt,
        &defaults,
        mode,
        &folder,
    )?;
    match (outputs, mode) {
        (Outputs::RatingsOnly, _) => Ok(()),
        (Outputs::RatingsLooksAndRenders, writeback::Mode::DryRun) => {
            info!("dry run: no look option is rendered");
            Ok(())
        }
        (Outputs::RatingsLooksAndRenders, writeback::Mode::Write) => {
            render_looks(&shots, &options, &cfg.review, &rt, &defaults, &cache)
        }
    }
}

/// Renders every option for the shots now rated at least `review.prerender_min_rating`.
/// Failed renders do not stop the batch; the run then ends in an error naming how many to
/// retry by running it again. When every render succeeded, the renders of options that are no
/// longer configured are removed.
fn render_looks(
    shots: &[Analyzed],
    options: &[LookOption],
    cfg: &ReviewConfig,
    rt: &sidecar::RtOptions,
    defaults: &rtcli::DefaultProfiles,
    cache: &Cache,
) -> Result<()> {
    let files: Vec<&Path> = shots.iter().map(|a| a.shot.file.as_path()).collect();
    let ratings = writeback::current_ratings(&files, rt)?;
    let default = defaults.get()?;
    let params = RenderParams::of(cfg);
    let mut jobs = Vec::new();
    for (a, rating) in shots.iter().zip(ratings) {
        if rating < i64::from(cfg.prerender_min_rating) {
            continue;
        }
        // Read back from the `.rpl` the write just saved, which is where `review` reads the
        // look and the faces too, including faces from earlier runs that this one did not
        // repeat.
        let record = Rpl::load(&a.shot.file);
        for option in options {
            jobs.push(looks::Job::new(
                &a.shot, option, &default, &record, &params,
            )?);
        }
    }
    info!(
        "rendering {} looks ({} options, rated at least {} stars), {} at a time",
        jobs.len(),
        options.len(),
        cfg.prerender_min_rating,
        cfg.render_jobs
    );
    let report = looks::render_all(&jobs, &params, cfg.render_jobs, cache.root(), &|p| match p
        .status
    {
        looks::Status::Cached => info!(
            "preprocessing {}/{}: {} {} (cached)",
            p.done, p.total, p.shot, p.option
        ),
        looks::Status::Rendered { took } => info!(
            "preprocessing {}/{}: {} {} ({:.1}s)",
            p.done,
            p.total,
            p.shot,
            p.option,
            took.as_secs_f64()
        ),
        looks::Status::Failed { .. } => info!(
            "preprocessing {}/{}: {} {} FAILED",
            p.done, p.total, p.shot, p.option
        ),
    })?;
    if report.failed() > 0 {
        bail!(
            "{} of {} renders failed (listed above); run process again to retry them",
            report.failed(),
            jobs.len()
        );
    }
    let pruned = looks::Store::new(cache.root()).prune_unconfigured(options)?;
    info!("removed {pruned} renders of options that are no longer configured");
    Ok(())
}

fn group_shots(shots: &[Analyzed], cfg: &Config) -> Vec<Scene> {
    let frames: Vec<group::Frame> = shots
        .iter()
        .map(|a| group::Frame {
            taken_at: a.shot.taken_at,
            dhash: a.features.dhash,
        })
        .collect();
    let scenes = group::group(&frames, &cfg.group);
    let bursts: usize = scenes.iter().map(|s| s.bursts().len()).sum();
    info!(
        "{} shots in {bursts} bursts, {} scenes",
        shots.len(),
        scenes.len()
    );
    scenes
}

/// What the judging passes decided. Per burst: the burst pass's judgement, the winner
/// (member index) and the scene pass's role for it. Per shot: a note for captions and the
/// report (the faces read), and the faces the pass located, `None` where it read no faces.
struct Judged {
    judgements: Vec<Option<BurstJudgement>>,
    winners: Vec<usize>,
    roles: Vec<Option<Role>>,
    notes: Vec<Option<String>>,
    face_boxes: Vec<Option<Vec<FaceBox>>>,
}

impl Judged {
    fn local(shots: &[Analyzed], bursts: &[&Burst], rel: &[f64]) -> Judged {
        Judged {
            judgements: vec![None; bursts.len()],
            winners: bursts
                .iter()
                .map(|b| rate::burst_winner(&b.of(rel), None))
                .collect(),
            roles: vec![None; bursts.len()],
            notes: vec![None; shots.len()],
            face_boxes: vec![None; shots.len()],
        }
    }

    /// One decision per shot.
    fn rate(&self, bursts: &[&Burst], rel: &[f64], cfg: &Config) -> Vec<Decision> {
        let outcomes: Vec<rate::BurstOutcome> = (0..bursts.len())
            .map(|b| rate::BurstOutcome {
                judgement: self.judgements[b].as_ref(),
                winner: self.winners[b],
                role: self.roles[b].as_ref(),
            })
            .collect();
        rate::rate_all(bursts, rel, &outcomes, &cfg.rate)
    }
}

/// Burst pass, then faces pass (which may move a people burst's winner), then scene pass
/// over the winners, then the shortlist pass over the scene pass's keepers.
fn gemini_passes(
    shots: &[Analyzed],
    scenes: &[Scene],
    bursts: &[&Burst],
    rel: &[f64],
    cache: &Cache,
    cfg: &Config,
    api_key: &str,
) -> Result<Judged> {
    let judgements = judge::judge_all(shots, bursts, rel, cache, &cfg.judge, api_key)?;
    let faces = faces::faces_all(
        shots,
        bursts,
        &judgements,
        rel,
        cfg.judge.faces_sharpness_rel_min,
        cache,
        &cfg.judge,
        api_key,
    )?;
    let mut notes: Vec<Option<String>> = vec![None; shots.len()];
    let mut face_boxes: Vec<Option<Vec<FaceBox>>> = vec![None; shots.len()];
    let winners: Vec<usize> = bursts
        .iter()
        .zip(&judgements)
        .zip(&faces)
        .map(|((b, j), f)| match f {
            Some(f) => {
                for (m, &i) in b.members().iter().enumerate() {
                    notes[i] = f.summary(m);
                    face_boxes[i] = f.located(m);
                }
                if let Some(j) = j
                    && j.best != f.winner
                {
                    info!(
                        from = %shots[b.members()[j.best]].shot.name(),
                        to = %shots[b.members()[f.winner]].shot.name(),
                        "faces pass moved the burst's winner"
                    );
                }
                f.winner
            }
            None => rate::burst_winner(&b.of(rel), j.as_ref()),
        })
        .collect();
    let winner_shots: Vec<usize> = bursts
        .iter()
        .zip(&winners)
        .map(|(b, &w)| b.members()[w])
        .collect();
    let mut roles = scene::roles_for_bursts(
        shots,
        scenes,
        &winner_shots,
        &notes,
        cache,
        &cfg.judge,
        api_key,
    )?;
    shortlist::calibrate(
        shots,
        &winner_shots,
        &mut roles,
        cache,
        &cfg.judge,
        &cfg.shortlist,
        api_key,
    )?;
    Ok(Judged {
        judgements,
        winners,
        roles,
        notes,
        face_boxes,
    })
}

/// Each shot's record of this run for its `.rpl`: what was measured, what each pass said
/// about it, and what was decided. Free-form JSON so it can grow; `format` is bumped when a
/// field changes meaning.
fn analyses(
    shots: &[Analyzed],
    bursts: &[&Burst],
    judged: &Judged,
    decisions: &[Decision],
    looks: &[Option<Look>],
    cfg: &Config,
) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = vec![serde_json::Value::Null; shots.len()];
    for (b, burst) in bursts.iter().enumerate() {
        let j = judged.judgements[b].as_ref();
        let winner_name = shots[burst.members()[judged.winners[b]]].shot.name();
        // The take the burst belongs to, named by the take's best file: bursts of one take
        // are attempts at the same shot.
        let take = match &judged.roles[b] {
            Some(Role::Alternate { take_best }) => take_best.clone(),
            Some(Role::Best { .. }) | None => winner_name,
        };
        for (m, &i) in burst.members().iter().enumerate() {
            let a = &shots[i];
            let winner = m == judged.winners[b];
            out[i] = serde_json::json!({
                "format": 1,
                "rawpilot": env!("CARGO_PKG_VERSION"),
                "model": cfg.judge.model,
                "taken_at": a.shot.taken_at,
                "exposure": a.shot.exposure,
                "af": a.shot.af,
                "features": a.features,
                "developed": a.developed,
                "burst": {
                    "first": shots[burst.members()[0]].shot.name(),
                    "size": burst.members().len(),
                    "member": m,
                    "winner": winner,
                },
                "take": take,
                "burst_pass": j.map(|j| serde_json::json!({
                    "scene_type": j.scene_type,
                    "judged_best": m == j.best,
                    "frame": j.frames[m],
                })),
                "faces": judged.notes[i],
                "scene_pass": if winner { judged.roles[b].as_ref() } else { None },
                "decision": decisions[i],
                "look": looks[i],
            });
        }
    }
    out
}

/// What the run wants written for each shot.
fn targets<'a>(
    shots: &'a [Analyzed],
    decisions: &[Decision],
    looks: &[Option<Look>],
    judged: &Judged,
    analyses: Vec<serde_json::Value>,
    outputs: Outputs,
) -> Vec<writeback::Target<'a>> {
    shots
        .iter()
        .zip(decisions)
        .zip(looks)
        .zip(&judged.face_boxes)
        .zip(analyses)
        .map(|((((a, d), look), faces), analysis)| {
            let (rating, label) = d.bucket.rating();
            writeback::Target {
                shot: &a.shot,
                mark: Mark { rating, label },
                look: outputs.measure(*look),
                analysis,
                faces: faces.clone(),
            }
        })
        .collect()
}

fn write_ratings(
    targets: &[writeback::Target],
    rt: &sidecar::RtOptions,
    defaults: &rtcli::DefaultProfiles,
    mode: writeback::Mode,
    folder: &FolderLock,
) -> Result<()> {
    let o = writeback::apply(targets, rt, defaults, mode, folder)?;
    info!(
        "ratings: {} {} ({} new pp3), {} unchanged, {} kept (rated by someone else)",
        o.written,
        match mode {
            writeback::Mode::Write => "written",
            writeback::Mode::DryRun => "would be written",
        },
        o.created,
        o.unchanged,
        o.kept_human
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cull` hands the writeback no look, so no existing pp3's processing can change (the
    /// writeback's own half of this is `a_batch_without_a_look_leaves_an_owned_pp3s_processing`).
    #[test]
    fn only_process_measures_looks() {
        let mut cfg = Config::load(None).unwrap();
        assert!(cfg.look.enabled);
        assert!(!Outputs::RatingsOnly.measures_looks(&cfg));
        assert!(Outputs::RatingsLooksAndRenders.measures_looks(&cfg));
        cfg.look.enabled = false;
        assert!(!Outputs::RatingsLooksAndRenders.measures_looks(&cfg));
    }
}
