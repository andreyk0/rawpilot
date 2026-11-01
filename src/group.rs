//! Two levels of grouping over a capture-ordered shoot. A burst is a run of near-identical
//! frames from which one is picked; a scene is a run of bursts close in time, which must keep
//! at least one frame even when none is good.

use chrono::{DateTime, FixedOffset};

use crate::config::GroupConfig;
use crate::features::DHash;

/// What grouping needs to know about a frame.
pub struct Frame {
    pub taken_at: DateTime<FixedOffset>,
    pub dhash: DHash,
}

/// Indices into the frame slice, in capture order. Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Burst(Vec<usize>);

impl Burst {
    pub fn members(&self) -> &[usize] {
        &self.0
    }

    /// The members' entries of a slice indexed like the shots, in member order.
    pub fn of<T: Copy>(&self, per_shot: &[T]) -> Vec<T> {
        self.0.iter().map(|&i| per_shot[i]).collect()
    }
}

/// Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scene(Vec<Burst>);

impl Scene {
    pub fn bursts(&self) -> &[Burst] {
        &self.0
    }
}

fn gap_s(a: DateTime<FixedOffset>, b: DateTime<FixedOffset>) -> f64 {
    (b - a).num_milliseconds() as f64 / 1000.0
}

/// Which scene a frame is in: scenes are numbered from 0 in capture order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SceneNo(usize);

impl SceneNo {
    pub fn get(self) -> usize {
        self.0
    }
}

/// The scene of every frame: a new scene starts where the gap since the previous frame is
/// longer than `scene_gap_s`. This is the one place scenes are decided; [`group`] and the
/// review UI both read it. `taken_at` must be in capture order.
pub fn scene_numbers(taken_at: &[DateTime<FixedOffset>], scene_gap_s: f64) -> Vec<SceneNo> {
    assert!(
        taken_at.is_sorted(),
        "scene_numbers() needs capture times in order"
    );
    let mut scene = 0;
    let mut numbers = Vec::with_capacity(taken_at.len());
    for (i, &t) in taken_at.iter().enumerate() {
        if i > 0 && gap_s(taken_at[i - 1], t) > scene_gap_s {
            scene += 1;
        }
        numbers.push(SceneNo(scene));
    }
    numbers
}

/// `frames` must be in capture order.
pub fn group(frames: &[Frame], cfg: &GroupConfig) -> Vec<Scene> {
    assert!(
        frames.is_sorted_by_key(|f| f.taken_at),
        "group() needs frames in capture order"
    );
    let mut bursts: Vec<Vec<usize>> = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let continues = i > 0 && {
            let prev = &frames[i - 1];
            gap_s(prev.taken_at, f.taken_at) <= cfg.burst_gap_s
                && prev.dhash.distance(f.dhash) <= cfg.burst_dhash_max
        };
        match bursts.last_mut() {
            Some(b) if continues => b.push(i),
            _ => bursts.push(vec![i]),
        }
    }

    // A burst is consecutive frames, so a burst starts a scene exactly when its first frame
    // does.
    let times: Vec<_> = frames.iter().map(|f| f.taken_at).collect();
    let scene_of = scene_numbers(&times, cfg.scene_gap_s);
    let mut scenes: Vec<Vec<Burst>> = Vec::new();
    for b in bursts {
        let continues = scenes.is_empty() || scene_of[b[0]] == scene_of[b[0] - 1];
        match scenes.last_mut() {
            Some(s) if continues => s.push(Burst(b)),
            _ => scenes.push(vec![Burst(b)]),
        }
    }
    scenes.into_iter().map(Scene).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GroupConfig {
        GroupConfig {
            burst_gap_s: 1.5,
            burst_dhash_max: 10,
            scene_gap_s: 60.0,
        }
    }

    fn frame(t_s: f64, hash: u64) -> Frame {
        let base = DateTime::parse_from_rfc3339("2026-10-18T14:00:00+00:00").unwrap();
        Frame {
            taken_at: base + chrono::Duration::milliseconds((t_s * 1000.0) as i64),
            dhash: DHash(hash),
        }
    }

    fn shape(scenes: &[Scene]) -> Vec<Vec<Vec<usize>>> {
        scenes
            .iter()
            .map(|s| s.bursts().iter().map(|b| b.members().to_vec()).collect())
            .collect()
    }

    #[test]
    fn splits_bursts_on_time_gap_and_scenes_on_longer_gap() {
        let frames = [
            frame(0.0, 0),
            frame(0.13, 0),
            frame(0.26, 0),
            frame(5.0, 0),   // same scene, new burst
            frame(100.0, 0), // new scene
        ];
        assert_eq!(
            shape(&group(&frames, &cfg())),
            vec![vec![vec![0, 1, 2], vec![3]], vec![vec![4]]]
        );
    }

    #[test]
    fn recomposition_inside_time_gap_starts_new_burst() {
        let frames = [frame(0.0, 0), frame(0.13, 0), frame(0.26, u64::MAX)];
        assert_eq!(
            shape(&group(&frames, &cfg())),
            vec![vec![vec![0, 1], vec![2]]]
        );
    }

    #[test]
    fn gaps_at_exactly_the_threshold_continue() {
        let frames = [
            frame(0.0, 0),
            frame(1.5, 0b11_1111_1111),
            frame(61.5, 0b11_1111_1111),
        ];
        assert_eq!(
            shape(&group(&frames, &cfg())),
            vec![vec![vec![0, 1], vec![2]]]
        );
    }

    #[test]
    fn empty_input_gives_no_scenes() {
        assert!(group(&[], &cfg()).is_empty());
    }

    fn numbers(times_s: &[f64]) -> Vec<usize> {
        let times: Vec<_> = times_s.iter().map(|&t| frame(t, 0).taken_at).collect();
        scene_numbers(&times, 60.0)
            .into_iter()
            .map(SceneNo::get)
            .collect()
    }

    /// Hand-derived with a 60 s gap: 0, 60 (exactly the gap: continues), 121 (61 s after:
    /// splits), 122, 400 (splits). Mutation: compare with `>=` instead of `>`: the second
    /// frame starts a scene and the row goes red; drop the increment: all zeros.
    #[test]
    fn scene_numbers_split_on_gaps_longer_than_the_scene_gap() {
        assert_eq!(numbers(&[0.0, 60.0, 121.0, 122.0, 400.0]), [0, 0, 1, 1, 2]);
        assert!(numbers(&[]).is_empty());
    }

    /// `group` and `scene_numbers` decide scenes in one place: the frames of each scene
    /// `group` returns carry one number, and consecutive scenes carry consecutive numbers.
    /// Mutation: invert the `continues` comparison in `group`: the row goes red.
    #[test]
    fn group_scenes_are_the_scene_numbers() {
        let times_s = [0.0, 0.1, 5.0, 100.0, 100.1, 300.0, 359.0, 421.0];
        let frames: Vec<Frame> = times_s.iter().map(|&t| frame(t, 0)).collect();
        let by_number = numbers(&times_s);
        let by_group: Vec<usize> = {
            let mut of = vec![0; frames.len()];
            for (n, scene) in group(&frames, &cfg()).iter().enumerate() {
                for burst in scene.bursts() {
                    for &i in burst.members() {
                        of[i] = n;
                    }
                }
            }
            of
        };
        assert_eq!(by_group, by_number);
        assert_eq!(by_number, [0, 0, 0, 1, 1, 2, 2, 3]);
    }
}
