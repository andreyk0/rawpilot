//! The scene pass: Gemini compares the winners of a scene's bursts side by side.
//!
//! Two things the burst pass cannot see. A photographer re-shoots the same subject and
//! framing seconds apart (to beat hand shake, closed eyes, a bad pose), and each attempt forms
//! its own burst; those attempts are one *take*, of which one photo should win. And absolute
//! scores from a language model are compressed (asked for a 1–5 score, it gives nearly every
//! burst winner a 2 or a 3), while side-by-side comparisons are not, so tiers are assigned
//! relative to the photos next to it.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::analyze::Analyzed;
use crate::cache::Cache;
use crate::config::JudgeConfig;
use crate::gemini::{self, Usage};
use crate::group::Scene;
use crate::judge::{self, Shown};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Among the best of the trip.
    Portfolio,
    /// A good photo worth editing.
    Keeper,
    /// A fine record of the moment, unremarkable as a photograph.
    Ordinary,
    /// Flawed, with nothing worth keeping beyond the record that it was taken.
    Reject,
}

const TIERS: &[&str] = &["portfolio", "keeper", "ordinary", "reject"];

/// What the scene pass decided about one burst winner.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The best photo of its take.
    Best { tier: Tier, reason: String },
    /// Another attempt at a take whose best photo, `take_best` (a file name), is a different
    /// burst's winner.
    Alternate { take_best: String },
}

/// One request's worth of burst winners: `shots[i]` indices, in capture order.
pub struct Chunk(pub Vec<usize>);

#[derive(Debug, Deserialize)]
struct RawAnswer {
    takes: Vec<RawTake>,
}

#[derive(Debug, Deserialize)]
struct RawTake {
    photos: Vec<usize>,
    best: usize,
    tier: Tier,
    reason: String,
}

/// One role per burst, in the order the scenes list their bursts; `None` where the pass did
/// not decide. `winners[b]` is the `shots` index of burst `b`'s winner; `notes` (indexed
/// like `shots`) are passed to Gemini in the winners' captions.
pub fn roles_for_bursts(
    shots: &[Analyzed],
    scenes: &[Scene],
    winners: &[usize],
    notes: &[Option<String>],
    cache: &Cache,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<Vec<Option<Role>>> {
    let mut burst_of_winner: HashMap<usize, usize> = HashMap::new();
    let mut per_scene: Vec<Vec<usize>> = Vec::with_capacity(scenes.len());
    let mut b = 0;
    for scene in scenes {
        let mut list = Vec::with_capacity(scene.bursts().len());
        for _ in scene.bursts() {
            burst_of_winner.insert(winners[b], b);
            list.push(winners[b]);
            b += 1;
        }
        per_scene.push(list);
    }
    assert_eq!(b, winners.len(), "one winner per burst");

    let chunks = chunk_scenes(per_scene, cfg.scene_photos_per_request_max);
    let ranked = rank_all(shots, notes, &chunks, cache, cfg, api_key)?;
    let mut roles: Vec<Option<Role>> = vec![None; winners.len()];
    for (chunk, r) in chunks.iter().zip(ranked) {
        for (&shot, role) in chunk.0.iter().zip(r.into_iter().flatten()) {
            roles[burst_of_winner[&shot]] = Some(role);
        }
    }
    Ok(roles)
}

/// Roles for each chunk's photos, in chunk order; `None` for a chunk whose request failed.
fn rank_all(
    shots: &[Analyzed],
    notes: &[Option<String>],
    chunks: &[Chunk],
    cache: &Cache,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<Vec<Option<Vec<Role>>>> {
    let dir = cache.root().join(gemini::Pass::Scene.dir_name());
    std::fs::create_dir_all(&dir)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.concurrency)
        .build()?;
    let results: Vec<Result<(Vec<Role>, Option<Usage>)>> = pool.install(|| {
        chunks
            .par_iter()
            .map(|c| rank_chunk(shots, notes, c, cache, &dir, cfg, api_key))
            .collect()
    });
    let mut tally = gemini::Tally::default();
    let out = results
        .into_iter()
        .zip(chunks)
        .map(|(r, c)| {
            tally.record(&r);
            match r {
                Ok((roles, _)) => Some(roles),
                Err(e) => {
                    let first = shots[c.0[0]].shot.name();
                    warn!(from = %first, "scene pass failed, keeping burst ratings: {e:#}");
                    None
                }
            }
        })
        .collect();
    tally.log("scene pass");
    Ok(out)
}

fn rank_chunk(
    shots: &[Analyzed],
    notes: &[Option<String>],
    chunk: &Chunk,
    cache: &Cache,
    dir: &Path,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<(Vec<Role>, Option<Usage>)> {
    let n = chunk.0.len();
    if n == 1 {
        // Only when the whole shoot has one burst: nothing to compare against.
        bail!("a single photo has nothing to be compared with");
    }
    let shown = judge::shown_for_frames(chunk.0.iter().map(|&i| &shots[i]), cfg);
    let t0 = shots[chunk.0[0]].shot.taken_at;
    let mut parts = vec![gemini::Part::text(instructions(n, shown))];
    for (k, &i) in chunk.0.iter().enumerate() {
        let a = &shots[i];
        let after_s = (a.shot.taken_at - t0).num_milliseconds() as f64 / 1000.0;
        let clipped = a
            .developed
            .filter(|_| shown == Shown::Developed)
            .map_or(String::new(), |d| {
                format!(", raw clipped {:.2}%", d.raw_clipped * 100.0)
            });
        let faces = notes[i].as_ref().map_or(String::new(), |n| {
            format!("; read from full-resolution crops: {n}")
        });
        parts.push(gemini::Part::text(format!(
            "Photo {k}: taken {after_s:.0} s after photo 0{clipped}{faces}"
        )));
        parts.push(gemini::Part::jpeg(&judge::judging_jpeg(cache, a, shown)?));
    }
    let request = gemini::Request::new(cfg, parts, response_schema());
    let names: Vec<String> = chunk.0.iter().map(|&i| shots[i].shot.name()).collect();
    gemini::ask(api_key, cfg, &request, dir, |text| {
        let answer: RawAnswer = serde_json::from_str(text)
            .with_context(|| format!("model answer is not the requested JSON: {text}"))?;
        parse_response(answer, &names)
    })
}

/// Checks that the takes partition the request's photos (`names`, in request order) and that
/// each take's best is one of its photos.
fn parse_response(a: RawAnswer, names: &[String]) -> Result<Vec<Role>> {
    let n = names.len();
    let mut roles: Vec<Option<Role>> = vec![None; n];
    for take in a.takes {
        ensure!(!take.photos.is_empty(), "empty take");
        ensure!(
            take.photos.contains(&take.best),
            "take best {} is not among its photos {:?}",
            take.best,
            take.photos
        );
        for &p in &take.photos {
            ensure!(p < n, "photo {p} out of range 0..{n}");
            ensure!(roles[p].is_none(), "photo {p} in two takes");
            roles[p] = Some(if p == take.best {
                Role::Best {
                    tier: take.tier,
                    reason: take.reason.clone(),
                }
            } else {
                Role::Alternate {
                    take_best: names[take.best].clone(),
                }
            });
        }
    }
    roles
        .into_iter()
        .enumerate()
        .map(|(p, r)| r.with_context(|| format!("photo {p} is in no take")))
        .collect()
}

fn instructions(n: usize, shown: Shown) -> String {
    let rendering = match shown {
        Shown::Developed => {
            "Images are neutral automatic developments of the raw files, standing in for a \
basic edit: judge them as near-final images, and neither reward nor penalise overall \
brightness."
        }
        Shown::Camera => {
            "Images are the camera's own JPEG previews; frames shot darker on purpose have been \
brightened back. Judge brightness as it will look after editing."
        }
    };
    format!(
        "You are culling photos from a travel and hiking trip, shot on a Nikon Z8. Below are \
{n} photos from one stretch of the trip, minutes apart, numbered 0 to {last} in the order \
taken. Each is already the best frame of its own burst. {rendering}

The photographer often takes the same shot several times in a row on purpose — hands shake, \
people blink and move, animals turn — so that one attempt can be chosen. Your tasks:

1. Group the photos into takes. A take is a set of attempts at the same picture: same main \
subject, same viewpoint and framing, differing only in timing, pose, expression, a small \
shift or a little zoom. A photo of a different subject, from a different viewpoint, or framed \
so that it makes a different picture is its own take. Most takes have one photo.
2. For each take choose the best photo. With people or animals, first eyes open — starting \
with the person most in focus, then as many others as possible — then expression and a pose \
that looks good in a final shot. Otherwise sharpness, a level horizon, the moment.
3. Give each take's best photo a tier, comparing the takes with each other: `portfolio` for \
the few that would be among the best photos of a trip (light, composition, moment — usually \
no more than one take in five), `keeper` for good photos worth editing, `ordinary` for fine \
records of the moment that are unremarkable as photographs, `reject` only for a technical \
failure that no edit can save: missed focus, motion blur, a misfire, exposure beyond recovery. \
A composition you would have framed differently, a distracting object or a less interesting \
subject is at worst `ordinary`, never `reject`. Judge the photograph, not how rare or \
spectacular the event in it is.

Return every photo in exactly one take, with a reason of at most 15 words for the take. The \
reason is shown to the photographer next to the photo, so describe the photo itself and \
never refer to photos by their number.",
        last = n - 1
    )
}

fn response_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "OBJECT",
        "properties": {
            "takes": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "photos": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                        "best": {"type": "INTEGER"},
                        "tier": {"type": "STRING", "enum": TIERS},
                        "reason": {"type": "STRING"}
                    },
                    "required": ["photos", "best", "tier", "reason"]
                }
            }
        },
        "required": ["takes"]
    })
}

/// Packs scenes' burst winners (`shots` indices in capture order, one list per scene) into
/// requests of at most `max` photos. Whole consecutive scenes share a request, so a scene of
/// one burst still has neighbours to be compared against; a scene larger than `max` is split
/// into near-equal parts, so no part is left with a single photo.
pub fn chunk_scenes(scenes: Vec<Vec<usize>>, max: usize) -> Vec<Chunk> {
    assert!(max >= 2, "a chunk needs room for a comparison");
    let mut out: Vec<Chunk> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for winners in scenes {
        if winners.len() > max {
            if !open.is_empty() {
                out.push(Chunk(std::mem::take(&mut open)));
            }
            out.extend(split_even(winners, max));
        } else if open.len() + winners.len() <= max {
            open.extend(winners);
        } else {
            out.push(Chunk(std::mem::replace(&mut open, winners)));
        }
    }
    if !open.is_empty() {
        out.push(Chunk(open));
    }
    out
}

fn split_even(winners: Vec<usize>, max: usize) -> Vec<Chunk> {
    let n = winners.len();
    let count = n.div_ceil(max);
    let (base, extra) = (n / count, n % count);
    let mut it = winners.into_iter();
    (0..count)
        .map(|c| Chunk(it.by_ref().take(base + usize::from(c < extra)).collect()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|k| format!("DSC_{k}.NEF")).collect()
    }

    fn take(photos: &[usize], best: usize, tier: Tier) -> RawTake {
        RawTake {
            photos: photos.to_vec(),
            best,
            tier,
            reason: "r".into(),
        }
    }

    #[test]
    fn takes_map_to_best_and_alternates() {
        let roles = parse_response(
            RawAnswer {
                takes: vec![
                    take(&[0, 2], 2, Tier::Keeper),
                    take(&[1], 1, Tier::Ordinary),
                ],
            },
            &names(3),
        )
        .unwrap();
        assert_eq!(
            roles[0],
            Role::Alternate {
                take_best: "DSC_2.NEF".into()
            }
        );
        assert!(matches!(
            roles[1],
            Role::Best {
                tier: Tier::Ordinary,
                ..
            }
        ));
        assert!(matches!(
            roles[2],
            Role::Best {
                tier: Tier::Keeper,
                ..
            }
        ));
    }

    #[test]
    fn rejects_answers_that_do_not_partition_the_photos() {
        let missing = RawAnswer {
            takes: vec![take(&[0], 0, Tier::Keeper)],
        };
        assert!(parse_response(missing, &names(2)).is_err());
        let twice = RawAnswer {
            takes: vec![take(&[0, 1], 0, Tier::Keeper), take(&[1], 1, Tier::Keeper)],
        };
        assert!(parse_response(twice, &names(2)).is_err());
        let out_of_range = RawAnswer {
            takes: vec![take(&[0, 1, 2], 0, Tier::Keeper)],
        };
        assert!(parse_response(out_of_range, &names(2)).is_err());
        let foreign_best = RawAnswer {
            takes: vec![take(&[0], 1, Tier::Keeper), take(&[1], 1, Tier::Keeper)],
        };
        assert!(parse_response(foreign_best, &names(2)).is_err());
    }

    #[test]
    fn schema_tiers_match_serde_names() {
        for t in TIERS {
            serde_json::from_value::<Tier>(serde_json::json!(t)).unwrap();
        }
    }

    fn sizes(scenes: &[usize], max: usize) -> Vec<usize> {
        let mut next = 0;
        let lists: Vec<Vec<usize>> = scenes
            .iter()
            .map(|&n| {
                let v: Vec<usize> = (next..next + n).collect();
                next += n;
                v
            })
            .collect();
        chunk_scenes(lists, max).iter().map(|c| c.0.len()).collect()
    }

    /// Small scenes share a request; one that would overflow starts the next.
    #[test]
    fn whole_scenes_are_packed_together() {
        assert_eq!(sizes(&[1, 3, 2], 24), [6]);
        assert_eq!(sizes(&[10, 10, 10], 24), [20, 10]);
    }

    /// A 25-winner scene at 24 per request: 13 + 12, not 24 + 1.
    #[test]
    fn large_scene_is_split_evenly() {
        assert_eq!(sizes(&[2, 25, 3], 24), [2, 13, 12, 3]);
    }

    #[test]
    fn order_is_preserved() {
        let chunks = chunk_scenes(vec![vec![0, 1], vec![2], vec![3, 4, 5]], 4);
        let flat: Vec<usize> = chunks.into_iter().flat_map(|c| c.0).collect();
        assert_eq!(flat, [0, 1, 2, 3, 4, 5]);
    }
}
