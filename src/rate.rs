//! The rule table: burst measurements and Gemini's judgements (burst and scene passes, when
//! they ran) → a bucket per frame → the rating and colour label RawTherapee shows.
//!
//! Every burst keeps its winner at 2★ or above, whatever the judgements say. That makes the
//! scene-level coverage guarantee hold by construction: a scene is made of bursts.

use serde::{Deserialize, Serialize};

use crate::analyze::Analyzed;
use crate::config::RateConfig;
use crate::features::AfSharpness;
use crate::group::Burst;
use crate::judge::{BurstJudgement, FrameJudgement, Verdict};
use crate::scene::{Role, Tier};

/// RawTherapee's colour labels; the variant names are the `xmp:Label` strings it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Label {
    Red,
    Yellow,
    Green,
    Blue,
    Purple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stars(u8);

impl Stars {
    pub fn new(n: u8) -> Stars {
        assert!(n <= 5, "star rating {n} out of range");
        Stars(n)
    }

    pub fn get(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bucket {
    /// Burst winner judged good: a portfolio photo (`excellent`) or a keeper.
    Selected { excellent: bool },
    /// Burst winner that is ordinary, borderline or has a defect — or was not judged at all.
    Maybe,
    /// Burst winner judged a reject: kept visible because nothing replaces it.
    Coverage,
    /// Burst winner that lost to another burst's winner in the same take.
    Alternate,
    /// A usable alternative to its burst's winner.
    Review,
    /// Clearly worse than the burst's winner, or judged a reject.
    Reject,
}

impl Bucket {
    pub fn rating(self) -> (Stars, Option<Label>) {
        match self {
            Bucket::Selected { excellent: true } => (Stars::new(5), Some(Label::Green)),
            Bucket::Selected { excellent: false } => (Stars::new(4), Some(Label::Green)),
            Bucket::Maybe => (Stars::new(3), Some(Label::Yellow)),
            Bucket::Coverage => (Stars::new(2), Some(Label::Purple)),
            Bucket::Alternate => (Stars::new(2), Some(Label::Blue)),
            Bucket::Review => (Stars::new(1), None),
            Bucket::Reject => (Stars::new(0), None),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub bucket: Bucket,
    /// Sharpness relative to the burst's sharpest frame, 0–1.
    pub sharpness_rel: f64,
    pub why: String,
}

/// Which sharpness figure a burst is compared on. The AF window is the better measure, but
/// frames are only comparable on one figure, so it is used only when every member has it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharpnessBasis {
    AfWindow,
    Tiles,
}

pub fn sharpness_basis(shots: &[Analyzed], burst: &Burst) -> SharpnessBasis {
    let all_af = burst
        .members()
        .iter()
        .all(|&i| matches!(shots[i].features.af, AfSharpness::Measured { .. }));
    if all_af {
        SharpnessBasis::AfWindow
    } else {
        SharpnessBasis::Tiles
    }
}

fn sharpness(a: &Analyzed, basis: SharpnessBasis) -> f64 {
    match basis {
        SharpnessBasis::AfWindow => a
            .features
            .af
            .measured()
            .expect("AfWindow basis is chosen only when every member is measured"),
        SharpnessBasis::Tiles => a.features.tiles_sharpness,
    }
}

/// Every shot's sharpness relative to the sharpest frame of its burst (0–1), indexed like
/// `shots`. `bursts` must partition the shots.
pub fn sharpness_rel_all(shots: &[Analyzed], bursts: &[&Burst]) -> Vec<f64> {
    per_shot(shots.len(), bursts, |b| {
        let basis = sharpness_basis(shots, b);
        let sharp: Vec<f64> = b
            .members()
            .iter()
            .map(|&i| sharpness(&shots[i], basis))
            .collect();
        let max = sharp.iter().copied().fold(0.0, f64::max);
        sharp
            .iter()
            .map(|s| if max > 0.0 { s / max } else { 1.0 })
            .collect()
    })
}

/// What is known about one burst when it is rated. `winner` indexes its members.
pub struct BurstOutcome<'a> {
    pub judgement: Option<&'a BurstJudgement>,
    pub winner: usize,
    /// The scene pass's decision about the winner, when it ran.
    pub role: Option<&'a Role>,
}

/// The burst's winner by the burst pass alone: Gemini's best when judged, else the sharpest
/// frame. `rel` is the members' relative sharpness.
pub fn burst_winner(rel: &[f64], judgement: Option<&BurstJudgement>) -> usize {
    if let Some(j) = judgement {
        assert_eq!(j.frames.len(), rel.len(), "judgement/burst size mismatch");
        assert!(j.frames[j.best].is_some(), "judged best has no judgement");
        return j.best;
    }
    rel.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, _)| k)
        .expect("burst is non-empty")
}

/// One decision per shot, indexed like `shots`. `bursts` partition the shots, `outcomes`
/// runs parallel to `bursts`, and `rel` is `sharpness_rel_all`'s result.
pub fn rate_all(
    bursts: &[&Burst],
    rel: &[f64],
    outcomes: &[BurstOutcome],
    cfg: &RateConfig,
) -> Vec<Decision> {
    assert_eq!(bursts.len(), outcomes.len(), "one outcome per burst");
    let mut o = outcomes.iter();
    per_shot(rel.len(), bursts, |b| {
        let o = o.next().expect("checked: one outcome per burst");
        rate_burst(&b.of(rel), o, cfg)
    })
}

/// Runs `f` on each burst, in order, and scatters its per-member results back to shot order.
fn per_shot<T>(n: usize, bursts: &[&Burst], mut f: impl FnMut(&Burst) -> Vec<T>) -> Vec<T> {
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for b in bursts {
        let values = f(b);
        assert_eq!(values.len(), b.members().len());
        for (&i, v) in b.members().iter().zip(values) {
            assert!(out[i].is_none(), "shot {i} in two bursts");
            out[i] = Some(v);
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, v)| v.unwrap_or_else(|| panic!("shot {i} in no burst")))
        .collect()
}

/// Decisions for one burst's members, in member order; `rel` is their relative sharpness.
fn rate_burst(rel: &[f64], o: &BurstOutcome, cfg: &RateConfig) -> Vec<Decision> {
    assert!(o.winner < rel.len(), "winner out of range");
    if let Some(j) = o.judgement {
        assert_eq!(j.frames.len(), rel.len(), "judgement/burst size mismatch");
    }
    (0..rel.len())
        .map(|k| {
            let frame_j = o.judgement.and_then(|j| j.frames[k].as_ref());
            let (bucket, why) = if k == o.winner {
                match o.role {
                    Some(role) => role_bucket(role),
                    None => best_bucket(frame_j, cfg),
                }
            } else {
                other_bucket(rel[k], frame_j, cfg)
            };
            Decision {
                bucket,
                sharpness_rel: rel[k],
                why,
            }
        })
        .collect()
}

fn role_bucket(role: &Role) -> (Bucket, String) {
    match role {
        Role::Best { tier, reason } => {
            let bucket = match tier {
                Tier::Portfolio => Bucket::Selected { excellent: true },
                Tier::Keeper => Bucket::Selected { excellent: false },
                Tier::Ordinary => Bucket::Maybe,
                Tier::Reject => Bucket::Coverage,
            };
            (bucket, format!("best of take ({tier:?}): {reason}"))
        }
        Role::Alternate { take_best } => (
            Bucket::Alternate,
            format!("another attempt at the shot {take_best} is best of"),
        ),
    }
}

fn best_bucket(j: Option<&FrameJudgement>, cfg: &RateConfig) -> (Bucket, String) {
    let Some(j) = j else {
        return (Bucket::Maybe, "sharpest of burst (not judged)".into());
    };
    let bucket = match j.verdict {
        Verdict::Keep if j.defects.is_empty() => Bucket::Selected {
            excellent: j.aesthetic >= cfg.excellent_aesthetic_min,
        },
        Verdict::Keep | Verdict::Maybe => Bucket::Maybe,
        Verdict::Reject => Bucket::Coverage,
    };
    (bucket, format!("best of burst: {}", j.reason))
}

fn other_bucket(rel: f64, j: Option<&FrameJudgement>, cfg: &RateConfig) -> (Bucket, String) {
    if rel < cfg.reject_sharpness_rel_max {
        return (
            Bucket::Reject,
            format!("{:.0}% of burst's best sharpness", rel * 100.0),
        );
    }
    match j {
        Some(j) if j.verdict == Verdict::Reject => (Bucket::Reject, j.reason.clone()),
        Some(j) => (Bucket::Review, j.reason.clone()),
        None => (Bucket::Review, "alternative in burst".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{Clipping, DHash, Features};
    use crate::group::Frame;
    use crate::judge::{Defect, SceneType};
    use crate::shot::{Exposure, Orientation, PreviewSpan, Shot};

    fn cfg() -> RateConfig {
        RateConfig {
            reject_sharpness_rel_max: 0.5,
            excellent_aesthetic_min: 5,
        }
    }

    fn analyzed(af: AfSharpness, tiles: f64) -> Analyzed {
        Analyzed {
            shot: Shot {
                file: "x.NEF".into(),
                file_size: 0,
                taken_at: chrono::DateTime::parse_from_rfc3339("2026-10-18T14:00:00+00:00")
                    .unwrap(),
                preview: PreviewSpan { offset: 0, len: 0 },
                af: None,
                orientation: Orientation::Normal,
                exposure: Exposure {
                    shutter_s: 0.01,
                    focal_mm: 50.0,
                    iso: 100,
                    picture_control: None,
                    compensation_ev: None,
                },
            },
            features: Features {
                af,
                tiles_sharpness: tiles,
                clipping: Clipping {
                    highlights: 0.0,
                    shadows: 0.0,
                },
                dhash: DHash(0),
            },
            developed: None,
        }
    }

    fn af(s: f64) -> AfSharpness {
        AfSharpness::Measured {
            sharpness: s,
            texture: 50.0,
        }
    }

    fn burst_of(shots: &[Analyzed]) -> Burst {
        let frames: Vec<Frame> = shots
            .iter()
            .map(|a| Frame {
                taken_at: a.shot.taken_at,
                dhash: DHash(0),
            })
            .collect();
        let cfg = crate::config::GroupConfig {
            burst_gap_s: 1.0,
            burst_dhash_max: 0,
            scene_gap_s: 1.0,
        };
        crate::group::group(&frames, &cfg)[0].bursts()[0].clone()
    }

    fn buckets(d: &[Decision]) -> Vec<Bucket> {
        d.iter().map(|d| d.bucket).collect()
    }

    /// Rates one burst whose winner is the burst pass's.
    fn rate(rel: &[f64], judgement: Option<&BurstJudgement>, role: Option<&Role>) -> Vec<Decision> {
        let o = BurstOutcome {
            judgement,
            winner: burst_winner(rel, judgement),
            role,
        };
        rate_burst(rel, &o, &cfg())
    }

    fn best(tier: Tier) -> Role {
        Role::Best {
            tier,
            reason: "r".into(),
        }
    }

    /// The scene pass's tier replaces the burst pass's verdict for the winner; the other
    /// frames are rated as before.
    #[test]
    fn scene_tier_decides_the_winner() {
        let j = BurstJudgement {
            scene_type: SceneType::People,
            best: 0,
            frames: vec![fj(Verdict::Maybe, 2, &[]), fj(Verdict::Keep, 3, &[])],
        };
        let expect = [
            (Tier::Portfolio, Bucket::Selected { excellent: true }),
            (Tier::Keeper, Bucket::Selected { excellent: false }),
            (Tier::Ordinary, Bucket::Maybe),
            (Tier::Reject, Bucket::Coverage),
        ];
        for (tier, bucket) in expect {
            let d = rate(&[1.0, 0.9], Some(&j), Some(&best(tier)));
            assert_eq!(buckets(&d), [bucket, Bucket::Review], "{tier:?}");
        }
    }

    /// A winner that lost its take stays visible at 2★, so every burst keeps a frame.
    #[test]
    fn alternate_take_is_kept_at_two_stars() {
        let alt = Role::Alternate {
            take_best: "DSC_1.NEF".into(),
        };
        let d = rate(&[1.0], None, Some(&alt));
        assert_eq!(buckets(&d), [Bucket::Alternate]);
        assert_eq!(Bucket::Alternate.rating().0, Stars::new(2));
    }

    /// The faces pass can move the winner off the burst pass's choice.
    #[test]
    fn winner_can_differ_from_judged_best() {
        let j = BurstJudgement {
            scene_type: SceneType::People,
            best: 0,
            frames: vec![fj(Verdict::Keep, 3, &[]), fj(Verdict::Keep, 3, &[])],
        };
        let o = BurstOutcome {
            judgement: Some(&j),
            winner: 1,
            role: Some(&best(Tier::Keeper)),
        };
        let d = rate_burst(&[1.0, 0.9], &o, &cfg());
        assert_eq!(
            buckets(&d),
            [Bucket::Review, Bucket::Selected { excellent: false }]
        );
    }

    fn rel_of(shots: &[Analyzed]) -> Vec<f64> {
        sharpness_rel_all(shots, &[&burst_of(shots)])
    }

    #[test]
    fn relative_sharpness_is_against_the_burst_best() {
        let shots = [analyzed(af(100.0), 1.0), analyzed(af(400.0), 1.0)];
        assert_eq!(rel_of(&shots), [0.25, 1.0]);
    }

    #[test]
    fn unjudged_burst_keeps_sharpest_as_maybe_and_rejects_the_soft() {
        let d = rate(&[0.25, 1.0, 0.75], None, None);
        assert_eq!(buckets(&d), [Bucket::Reject, Bucket::Maybe, Bucket::Review]);
    }

    /// Frame 0 has the sharper AF window, but frame 1 has none: comparing an AF figure with a
    /// tiles figure would be meaningless, so the whole burst falls back to tiles.
    #[test]
    fn mixed_af_availability_compares_on_tiles() {
        let shots = [
            analyzed(af(900.0), 100.0),
            analyzed(AfSharpness::NoBox, 400.0),
        ];
        assert_eq!(
            sharpness_basis(&shots, &burst_of(&shots)),
            SharpnessBasis::Tiles
        );
        assert_eq!(rel_of(&shots), [0.25, 1.0]);
    }

    fn fj(verdict: Verdict, aesthetic: u8, defects: &[Defect]) -> Option<FrameJudgement> {
        Some(FrameJudgement {
            verdict,
            aesthetic,
            defects: defects.to_vec(),
            reason: "r".into(),
        })
    }

    #[test]
    fn judged_best_overrides_sharpest_and_maps_by_verdict() {
        let j = BurstJudgement {
            scene_type: SceneType::People,
            best: 1,
            frames: vec![fj(Verdict::Keep, 3, &[]), fj(Verdict::Keep, 5, &[])],
        };
        let d = rate(&[1.0, 0.95], Some(&j), None);
        assert_eq!(
            buckets(&d),
            [Bucket::Review, Bucket::Selected { excellent: true }]
        );
    }

    #[test]
    fn rejected_best_is_still_kept_as_coverage() {
        let j = BurstJudgement {
            scene_type: SceneType::Other,
            best: 0,
            frames: vec![fj(Verdict::Reject, 1, &[Defect::Misfire])],
        };
        let d = rate(&[1.0], Some(&j), None);
        assert_eq!(buckets(&d), [Bucket::Coverage]);
    }

    #[test]
    fn keep_with_defect_is_maybe() {
        let j = BurstJudgement {
            scene_type: SceneType::Landscape,
            best: 0,
            frames: vec![fj(Verdict::Keep, 4, &[Defect::TiltedHorizon])],
        };
        let d = rate(&[1.0], Some(&j), None);
        assert_eq!(buckets(&d), [Bucket::Maybe]);
    }

    /// Each frame is rated on its own slot: a judged frame by its verdict, a frame a long
    /// burst did not send by sharpness alone.
    #[test]
    fn each_frame_uses_its_own_judgement_slot() {
        let j = BurstJudgement {
            scene_type: SceneType::People,
            best: 1,
            frames: vec![
                fj(Verdict::Reject, 2, &[Defect::ClosedEyes]),
                fj(Verdict::Keep, 4, &[]),
                None,
                None,
            ],
        };
        let d = rate(&[0.9, 1.0, 0.9, 0.1], Some(&j), None);
        assert_eq!(
            buckets(&d),
            [
                Bucket::Reject,
                Bucket::Selected { excellent: false },
                Bucket::Review,
                Bucket::Reject
            ]
        );
    }
}
