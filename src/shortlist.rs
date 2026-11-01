//! The shortlist pass: one comparison across the whole shoot of every take judged keeper or
//! better, so that 5★ means the same thing in every part of the trip.
//!
//! The scene pass tiers takes relative to the up to two dozen photos in the same request;
//! two requests can disagree on what "portfolio" means. Here the candidates are ranked side
//! by side and the top `shortlist.portfolio_share` of them become portfolio, the rest keeper.
//! Ranking beats asking each request for absolute tiers for the reason the scene pass exists:
//! a language model's absolute scores are compressed, its comparisons are not.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tracing::{info, warn};

use crate::analyze::Analyzed;
use crate::cache::Cache;
use crate::config::{JudgeConfig, ShortlistConfig};
use crate::gemini;
use crate::judge::{self, Shown};
use crate::scene::{Role, Tier};

#[derive(Debug, Deserialize)]
struct RawAnswer {
    /// Candidate numbers, best first.
    ranking: Vec<usize>,
}

/// Re-tiers, in place, every `Role::Best` at keeper or portfolio. `winner_shots[b]` is burst
/// `b`'s winner (a `shots` index), parallel to `roles`. Leaves `roles` untouched when the
/// request fails.
pub fn calibrate(
    shots: &[Analyzed],
    winner_shots: &[usize],
    roles: &mut [Option<Role>],
    cache: &Cache,
    judge_cfg: &JudgeConfig,
    cfg: &ShortlistConfig,
    api_key: &str,
) -> Result<()> {
    let candidates: Vec<usize> = (0..roles.len())
        .filter(|&b| {
            matches!(
                roles[b],
                Some(Role::Best {
                    tier: Tier::Portfolio | Tier::Keeper,
                    ..
                })
            )
        })
        .collect();
    if candidates.len() < 2 {
        return Ok(());
    }
    let dir = cache.root().join(gemini::Pass::Shortlist.dir_name());
    std::fs::create_dir_all(&dir)?;
    let chunks = strided_chunks(candidates.len(), cfg.photos_max);
    let mut rankings = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let frames: Vec<&Analyzed> = chunk
            .iter()
            .map(|&k| &shots[winner_shots[candidates[k]]])
            .collect();
        match rank(&frames, cache, &dir, judge_cfg, api_key) {
            Ok(r) => rankings.push(r),
            Err(e) => {
                warn!("shortlist pass failed, keeping the scene pass's tiers: {e:#}");
                return Ok(());
            }
        }
    }
    let ranking = merge_rankings(&chunks, &rankings);
    let portfolio = portfolio_count(candidates.len(), cfg.portfolio_share);
    apply_ranking(roles, &candidates, &ranking, portfolio);
    info!(
        "shortlist: {} candidates, top {portfolio} become portfolio",
        candidates.len()
    );
    Ok(())
}

/// Positions `0..n` split into the fewest chunks of at most `max`, by stride: chunk `c` holds
/// `c, c + count, c + 2·count, …`. Each chunk then samples the whole shoot, so a place in one
/// chunk's ranking means about the same as the same place in another's.
fn strided_chunks(n: usize, max: usize) -> Vec<Vec<usize>> {
    assert!(max >= 2, "a chunk needs room for a comparison");
    let count = n.div_ceil(max).max(1);
    (0..count)
        .map(|c| (c..n).step_by(count).collect())
        .collect()
}

/// One ranking of all positions from per-chunk rankings (each a permutation of indices into
/// its chunk), ordered by relative place in the chunk; ties keep chunk order.
fn merge_rankings(chunks: &[Vec<usize>], rankings: &[Vec<usize>]) -> Vec<usize> {
    let mut scored: Vec<(f64, usize, usize)> = Vec::new();
    for (c, (chunk, ranking)) in chunks.iter().zip(rankings).enumerate() {
        for (place, &k) in ranking.iter().enumerate() {
            let relative = (place as f64 + 0.5) / ranking.len() as f64;
            scored.push((relative, c, chunk[k]));
        }
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, pos)| pos).collect()
}

/// `ranking` lists positions in `candidates` (burst indices), best first; the first
/// `portfolio` become portfolio and the rest keeper.
fn apply_ranking(
    roles: &mut [Option<Role>],
    candidates: &[usize],
    ranking: &[usize],
    portfolio: usize,
) {
    for (place, &k) in ranking.iter().enumerate() {
        let Some(Role::Best { tier, reason }) = &mut roles[candidates[k]] else {
            unreachable!("candidates are Best roles");
        };
        *tier = if place < portfolio {
            Tier::Portfolio
        } else {
            Tier::Keeper
        };
        *reason = format!(
            "#{} of {} on the shortlist; {reason}",
            place + 1,
            ranking.len()
        );
    }
}

/// How many of `n` candidates are portfolio: `share` of them rounded, at least one.
fn portfolio_count(n: usize, share: f64) -> usize {
    ((n as f64 * share).round() as usize).clamp(1, n)
}

/// Candidate indices, best first.
fn rank(
    frames: &[&Analyzed],
    cache: &Cache,
    dir: &Path,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<Vec<usize>> {
    let n = frames.len();
    let shown = judge::shown_for_frames(frames.iter().copied(), cfg);
    let mut parts = vec![gemini::Part::text(instructions(n, shown))];
    for (k, a) in frames.iter().enumerate() {
        parts.push(gemini::Part::text(format!("Photo {k}:")));
        parts.push(gemini::Part::jpeg(&judge::judging_jpeg(cache, a, shown)?));
    }
    let req = gemini::Request::new(cfg, parts, schema());
    let (ranking, usage) = gemini::ask(api_key, cfg, &req, dir, |text| {
        let a: RawAnswer = serde_json::from_str(text)
            .with_context(|| format!("model answer is not the requested JSON: {text}"))?;
        parse(a, n)
    })?;
    let mut tally = gemini::Tally::default();
    tally.record(&Ok::<_, anyhow::Error>(((), usage)));
    tally.log("shortlist pass");
    Ok(ranking)
}

/// The ranking must be a permutation of `0..n`.
fn parse(a: RawAnswer, n: usize) -> Result<Vec<usize>> {
    let mut seen = vec![false; n];
    for &k in &a.ranking {
        ensure!(k < n, "photo {k} out of range 0..{n}");
        ensure!(!seen[k], "photo {k} ranked twice");
        seen[k] = true;
    }
    ensure!(
        a.ranking.len() == n,
        "ranked {} of {n} photos",
        a.ranking.len()
    );
    Ok(a.ranking)
}

fn instructions(n: usize, shown: Shown) -> String {
    let rendering = match shown {
        Shown::Developed => {
            "They are neutral automatic developments of the raw files, standing in for a basic \
edit: judge them as near-final images, and neither reward nor penalise overall brightness."
        }
        Shown::Camera => {
            "They are the camera's JPEG previews; frames shot darker on purpose have been \
brightened back."
        }
    };
    format!(
        "These are the {n} strongest photos of a travel and hiking trip, numbered 0 to {last}, \
already chosen from hundreds. {rendering}

Rank all of them from best to worst as photographs someone would print or share from this \
trip: light, composition, the moment, the subject's expression, and how well the photo \
conveys the place. Return every photo number exactly once.",
        last = n - 1
    )
}

fn schema() -> serde_json::Value {
    serde_json::json!({
        "type": "OBJECT",
        "properties": {
            "ranking": {"type": "ARRAY", "items": {"type": "INTEGER"}}
        },
        "required": ["ranking"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranking_must_be_a_permutation() {
        assert_eq!(
            parse(
                RawAnswer {
                    ranking: vec![2, 0, 1]
                },
                3
            )
            .unwrap(),
            [2, 0, 1]
        );
        assert!(
            parse(
                RawAnswer {
                    ranking: vec![2, 0]
                },
                3
            )
            .is_err()
        );
        assert!(
            parse(
                RawAnswer {
                    ranking: vec![2, 2, 1]
                },
                3
            )
            .is_err()
        );
        assert!(
            parse(
                RawAnswer {
                    ranking: vec![3, 0, 1]
                },
                3
            )
            .is_err()
        );
    }

    /// 79 candidates at 48 per request: two chunks of 40 and 39, interleaved.
    #[test]
    fn strided_chunks_cover_everything_once() {
        let c = strided_chunks(79, 48);
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].len(), c[1].len()), (40, 39));
        assert_eq!(&c[0][..3], [0, 2, 4]);
        assert_eq!(&c[1][..3], [1, 3, 5]);
        let mut all: Vec<usize> = c.concat();
        all.sort_unstable();
        assert_eq!(all, (0..79).collect::<Vec<_>>());
        assert_eq!(strided_chunks(16, 48), [(0..16).collect::<Vec<_>>()]);
    }

    /// Chunk A = positions [0, 2], ranked [1, 0] (position 2 first); chunk B = [1, 3], ranked
    /// [0, 1]. Relative places: 2 → 0.25, 1 → 0.25, 0 → 0.75, 3 → 0.75; ties keep chunk order.
    #[test]
    fn rankings_merge_by_relative_place() {
        let merged = merge_rankings(&[vec![0, 2], vec![1, 3]], &[vec![1, 0], vec![0, 1]]);
        assert_eq!(merged, [2, 1, 0, 3]);
    }

    /// 20 candidates at 0.3: 6 portfolio; never none, never more than all.
    #[test]
    fn portfolio_share() {
        assert_eq!(portfolio_count(20, 0.3), 6);
        assert_eq!(portfolio_count(2, 0.1), 1);
        assert_eq!(portfolio_count(3, 1.5), 3);
    }

    fn best(tier: Tier) -> Option<Role> {
        Some(Role::Best {
            tier,
            reason: "r".into(),
        })
    }

    fn tier_of(r: &Option<Role>) -> Option<Tier> {
        match r {
            Some(Role::Best { tier, .. }) => Some(*tier),
            _ => None,
        }
    }

    /// Only keeper and portfolio takes are re-tiered, by their place in the ranking.
    #[test]
    fn ranking_assigns_tiers_to_candidates_only() {
        let mut roles = vec![
            best(Tier::Portfolio),
            best(Tier::Ordinary),
            best(Tier::Keeper),
            Some(Role::Alternate {
                take_best: "DSC_1.NEF".into(),
            }),
            best(Tier::Keeper),
        ];
        // Candidate 2 (burst 4) ranked best, then burst 0, then burst 2; one portfolio slot.
        apply_ranking(&mut roles, &[0, 2, 4], &[2, 0, 1], 1);
        let tiers: Vec<Option<Tier>> = roles.iter().map(tier_of).collect();
        assert_eq!(
            tiers,
            [
                Some(Tier::Keeper),
                Some(Tier::Ordinary),
                Some(Tier::Keeper),
                None,
                Some(Tier::Portfolio)
            ]
        );
    }
}
