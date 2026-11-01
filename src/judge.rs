//! Gemini's judgement of a burst: which frame is best and how each frame stands on its own.
//!
//! One request per burst. Ranking near-duplicates side by side is the comparison a language
//! model is consistent at; absolute per-image scores drift between calls. The local
//! measurements go into the prompt so the model does not have to guess sharpness from a
//! 1024 px thumbnail.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use image::codecs::jpeg::JpegEncoder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::analyze::Analyzed;
use crate::cache::{self, Cache};
use crate::config::{JudgeConfig, Preview};
use crate::exposure;
use crate::gemini::{self, Usage};
use crate::group::Burst;

/// The thumbnail is re-encoded after brightening; high enough that the second generation of
/// JPEG loss is invisible at 1024 px.
const JUDGING_JPEG_QUALITY: u8 = 90;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Keep,
    Maybe,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SceneType {
    Landscape,
    People,
    Wildlife,
    Detail,
    Architecture,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Defect {
    MissedFocus,
    MotionBlur,
    ClosedEyes,
    AwkwardExpression,
    SubjectCutOff,
    TiltedHorizon,
    ClippedHighlights,
    Underexposed,
    Obstruction,
    Misfire,
}

const SCENE_TYPES: &[&str] = &[
    "landscape",
    "people",
    "wildlife",
    "detail",
    "architecture",
    "other",
];
const DEFECTS: &[&str] = &[
    "missed_focus",
    "motion_blur",
    "closed_eyes",
    "awkward_expression",
    "subject_cut_off",
    "tilted_horizon",
    "clipped_highlights",
    "underexposed",
    "obstruction",
    "misfire",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameJudgement {
    pub verdict: Verdict,
    /// 1–5.
    pub aesthetic: u8,
    pub defects: Vec<Defect>,
    pub reason: String,
}

/// `frames` has one slot per burst member, in member order; a slot is empty when the frame
/// was not sent (bursts longer than the per-request cap send their sharpest frames). `best`
/// indexes a filled slot. Both hold by construction in `parse_response`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BurstJudgement {
    pub scene_type: SceneType,
    pub best: usize,
    pub frames: Vec<Option<FrameJudgement>>,
}

/// The model's answer, before it is checked against the request.
#[derive(Debug, Deserialize)]
struct RawAnswer {
    scene_type: SceneType,
    best_frame: usize,
    frames: Vec<RawFrame>,
}

#[derive(Debug, Deserialize)]
struct RawFrame {
    frame: usize,
    verdict: Verdict,
    aesthetic: u8,
    defects: Vec<Defect>,
    reason: String,
}

/// One slot per burst, in the order given. A burst whose request fails after retries is
/// logged and left unjudged, so it is rated on local measurements alone.
pub fn judge_all(
    shots: &[Analyzed],
    bursts: &[&Burst],
    sharpness_rel: &[f64],
    cache: &Cache,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<Vec<Option<BurstJudgement>>> {
    let dir = cache.root().join(gemini::Pass::Judge.dir_name());
    std::fs::create_dir_all(&dir)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.concurrency)
        .build()?;
    let results: Vec<Result<(BurstJudgement, Option<Usage>)>> = pool.install(|| {
        bursts
            .par_iter()
            .map(|b| judge_burst(shots, b, sharpness_rel, cache, &dir, cfg, api_key))
            .collect()
    });

    let mut tally = gemini::Tally::default();
    let out = results
        .into_iter()
        .zip(bursts)
        .map(|(r, b)| {
            tally.record(&r);
            match r {
                Ok((j, _)) => Some(j),
                Err(e) => {
                    let first = shots[b.members()[0]].shot.name();
                    warn!(burst = %first, "not judged, rating on local measurements: {e:#}");
                    None
                }
            }
        })
        .collect();
    tally.log("burst judge");
    Ok(out)
}

/// The judgement and, when it came from the API rather than the cache, what it cost.
fn judge_burst(
    shots: &[Analyzed],
    burst: &Burst,
    sharpness_rel: &[f64],
    cache: &Cache,
    dir: &Path,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<(BurstJudgement, Option<Usage>)> {
    let sent = frames_to_send(burst, sharpness_rel, cfg.frames_per_request_max);
    let shown = shown_for(shots, burst, &sent, cfg);
    let mut parts = vec![gemini::Part::text(instructions(sent.len(), shown))];
    for (k, &m) in sent.iter().enumerate() {
        let a = &shots[burst.members()[m]];
        parts.push(gemini::Part::text(frame_caption(
            k,
            a,
            sharpness_rel[burst.members()[m]],
            shown,
        )));
        parts.push(gemini::Part::jpeg(&judging_jpeg(cache, a, shown)?));
    }
    let request = gemini::Request::new(cfg, parts, response_schema());
    gemini::ask(api_key, cfg, &request, dir, |text| {
        let answer: RawAnswer = serde_json::from_str(text)
            .with_context(|| format!("model answer is not the requested JSON: {text}"))?;
        parse_response(answer, &sent, burst.members().len())
    })
}

/// Member indices to send, in member order: all of them, or the sharpest `max`.
fn frames_to_send(burst: &Burst, sharpness_rel: &[f64], max: usize) -> Vec<usize> {
    let n = burst.members().len();
    let mut idx: Vec<usize> = (0..n).collect();
    if n > max {
        let rel = burst.of(sharpness_rel);
        idx.sort_by(|&a, &b| rel[b].total_cmp(&rel[a]));
        idx.truncate(max);
        idx.sort_unstable();
    }
    idx
}

/// `sent[k]` is the member index of request frame `k`.
fn parse_response(a: RawAnswer, sent: &[usize], members: usize) -> Result<BurstJudgement> {
    ensure!(
        a.best_frame < sent.len(),
        "best_frame {} out of range 0..{}",
        a.best_frame,
        sent.len()
    );
    let mut frames: Vec<Option<FrameJudgement>> = vec![None; members];
    for f in a.frames {
        ensure!(
            f.frame < sent.len(),
            "frame {} out of range 0..{}",
            f.frame,
            sent.len()
        );
        ensure!(
            (1..=5).contains(&f.aesthetic),
            "aesthetic {} out of 1..=5",
            f.aesthetic
        );
        let slot = &mut frames[sent[f.frame]];
        if slot.is_some() {
            bail!("frame {} judged twice", f.frame);
        }
        *slot = Some(FrameJudgement {
            verdict: f.verdict,
            aesthetic: f.aesthetic,
            defects: f.defects,
            reason: f.reason,
        });
    }
    for (k, &m) in sent.iter().enumerate() {
        ensure!(frames[m].is_some(), "frame {k} was sent but not judged");
    }
    Ok(BurstJudgement {
        scene_type: a.scene_type,
        best: sent[a.best_frame],
        frames,
    })
}

/// What the judge is shown. Every frame of one request is shown the same way, so frames are
/// never compared across renderings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Shown {
    Developed,
    Camera,
}

fn shown_for(shots: &[Analyzed], burst: &Burst, sent: &[usize], cfg: &JudgeConfig) -> Shown {
    shown_for_frames(sent.iter().map(|&m| &shots[burst.members()[m]]), cfg)
}

/// `Developed` when configured and every frame has a development, else `Camera`.
pub fn shown_for_frames<'a>(
    frames: impl Iterator<Item = &'a Analyzed>,
    cfg: &JudgeConfig,
) -> Shown {
    let undeveloped: Vec<String> = frames
        .filter(|a| a.developed.is_none())
        .map(|a| a.shot.name())
        .collect();
    shown_given(cfg.preview, &undeveloped)
}

/// One frame without a development puts the whole request on camera previews.
fn shown_given(preview: Preview, undeveloped: &[String]) -> Shown {
    match preview {
        Preview::Camera => Shown::Camera,
        Preview::Developed if undeveloped.is_empty() => Shown::Developed,
        Preview::Developed => {
            warn!(
                missing = ?undeveloped,
                "frames did not develop; this request uses camera previews"
            );
            Shown::Camera
        }
    }
}

fn instructions(n: usize, shown: Shown) -> String {
    let rendering = match shown {
        Shown::Developed => {
            "Images are neutral automatic developments of the raw files — exposure normalised \
to the scene, highlights compressed, mild contrast and colour — standing in for a basic edit. \
They show what each frame can become, not how the camera rendered it: judge them as \
near-final images, and neither reward nor penalise overall brightness, which an edit sets \
anyway. `raw clipped` is the share of the frame blown in the raw file itself, which no edit \
can recover: a real defect when it covers the subject or a large area."
        }
        Shown::Camera => {
            "Images are the camera's own JPEG previews. `clipped` is the share of blown \
highlights in that JPEG. In bright light the photographer often dials in negative exposure \
compensation to protect highlights and brightens the raw file in editing, accepting some \
shadow noise. Frames shot that way are shown already brightened back, and their caption says \
by how much. Judge brightness as it will look after editing: never penalise a frame for \
having been exposed darker, and use `underexposed` only when the subject would still be too \
dark after that."
        }
    };
    format!(
        "You are culling photos from a travel and hiking trip: landscapes, people, animals and \
details, shot on a Nikon Z8. Below are {n} frame(s) of ONE burst — near-identical shots taken \
seconds apart — numbered 0 to {last}, downscaled.

{rendering}

Each frame's caption gives measurements taken locally on the full-resolution image. \
`sharpness` is relative to the sharpest frame of this burst at the camera's focus point \
(100% = sharpest). Differences above about 70% are invisible in a print; below about 50% the \
frame has missed focus or motion blur, even if it looks fine at this size.

Decide:
- scene_type of the burst.
- best_frame: the one frame a photographer would keep from this burst, chosen on content. \
For people: eyes open, natural expression, gesture, nobody mid-blink or mid-word. For \
wildlife: eye visible, good pose. For landscapes: light, level horizon, composition, people or \
objects in the frame at a good moment. Use sharpness only to exclude soft frames and to break \
ties between frames whose content is equally good.
- For EVERY frame: verdict (keep = worth editing as a final image of the trip; maybe = \
usable but weak; reject = clearly flawed), aesthetic 1-5 judged against good travel \
photography (3 = a decent snapshot), defects (empty if none), and a reason of at most 12 words.

A one-frame burst is judged the same way. Intentional blur (panning) or shallow depth of \
field is not a defect.",
        last = n - 1
    )
}

pub fn judging_jpeg(cache: &Cache, a: &Analyzed, shown: Shown) -> Result<Vec<u8>> {
    match shown {
        Shown::Developed => {
            let path = cache.image_path(cache::DEVELOPED, &a.shot);
            std::fs::read(&path).with_context(|| format!("reading {}", path.display()))
        }
        Shown::Camera => {
            let path = cache.image_path(cache::FEATURES, &a.shot);
            let thumb = image::open(&path)
                .with_context(|| format!("reading thumbnail {}", path.display()))?
                .into_rgb8();
            let brightened =
                exposure::undo_negative_compensation(&thumb, a.shot.exposure.compensation_ev);
            let mut jpeg = Vec::new();
            JpegEncoder::new_with_quality(&mut jpeg, JUDGING_JPEG_QUALITY)
                .encode_image(&brightened)?;
            Ok(jpeg)
        }
    }
}

fn frame_caption(k: usize, a: &Analyzed, sharpness_rel: f64, shown: Shown) -> String {
    let e = &a.shot.exposure;
    let (clipping, compensation) = match (shown, a.developed) {
        (Shown::Developed, Some(d)) => (
            format!("raw clipped {:.2}%", d.raw_clipped * 100.0),
            e.compensation_ev
                .map_or(String::new(), |ev| format!(", shot at {ev:+.1} EV")),
        ),
        (Shown::Developed, None) => unreachable!("shown_for develops every sent frame"),
        (Shown::Camera, _) => (
            format!("clipped {:.1}%", a.features.clipping.highlights * 100.0),
            match e.compensation_ev {
                Some(ev) if ev < 0.0 => {
                    format!(", shot at {ev:+.1} EV, shown brightened by {:+.1} EV", -ev)
                }
                Some(ev) => format!(", shot at {ev:+.1} EV"),
                None => String::new(),
            },
        ),
    };
    format!(
        "Frame {k}: sharpness {:.0}%, {clipping}, 1/{:.0}s, {:.0}mm, ISO {}, Picture Control {}{compensation}",
        sharpness_rel * 100.0,
        1.0 / e.shutter_s,
        e.focal_mm,
        e.iso,
        e.picture_control.as_deref().unwrap_or("unknown"),
    )
}

fn response_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "OBJECT",
        "properties": {
            "scene_type": {"type": "STRING", "enum": SCENE_TYPES},
            "best_frame": {"type": "INTEGER"},
            "frames": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "frame": {"type": "INTEGER"},
                        "verdict": {"type": "STRING", "enum": ["keep", "maybe", "reject"]},
                        "aesthetic": {"type": "INTEGER"},
                        "defects": {"type": "ARRAY", "items": {"type": "STRING", "enum": DEFECTS}},
                        "reason": {"type": "STRING"}
                    },
                    "required": ["frame", "verdict", "aesthetic", "defects", "reason"]
                }
            }
        },
        "required": ["scene_type", "best_frame", "frames"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The source of a request's images is decided for the whole request: one undeveloped
    /// frame puts every frame on camera previews, and `Camera` never develops anything.
    #[test]
    fn one_undeveloped_frame_moves_the_whole_request_to_camera() {
        assert_eq!(shown_given(Preview::Developed, &[]), Shown::Developed);
        assert_eq!(
            shown_given(Preview::Developed, &["DSC_1.NEF".to_owned()]),
            Shown::Camera
        );
        assert_eq!(shown_given(Preview::Camera, &[]), Shown::Camera);
    }

    fn raw_frame(frame: usize) -> RawFrame {
        RawFrame {
            frame,
            verdict: Verdict::Keep,
            aesthetic: 3,
            defects: vec![],
            reason: String::new(),
        }
    }

    fn answer(best: usize, frames: &[usize]) -> RawAnswer {
        RawAnswer {
            scene_type: SceneType::People,
            best_frame: best,
            frames: frames.iter().map(|&f| raw_frame(f)).collect(),
        }
    }

    /// Request frames 0,1 were members 2 and 5 of a 6-frame burst.
    #[test]
    fn maps_request_frames_back_to_members() {
        let j = parse_response(answer(1, &[0, 1]), &[2, 5], 6).unwrap();
        assert_eq!(j.best, 5);
        let filled: Vec<usize> = (0..6).filter(|&m| j.frames[m].is_some()).collect();
        assert_eq!(filled, [2, 5]);
    }

    #[test]
    fn rejects_out_of_range_best() {
        assert!(parse_response(answer(2, &[0, 1]), &[0, 1], 2).is_err());
    }

    #[test]
    fn rejects_missing_or_duplicate_frames() {
        assert!(parse_response(answer(0, &[0]), &[0, 1], 2).is_err());
        assert!(parse_response(answer(0, &[0, 0, 1]), &[0, 1], 2).is_err());
    }

    #[test]
    fn rejects_aesthetic_out_of_range() {
        let mut a = answer(0, &[0]);
        a.frames[0].aesthetic = 0;
        assert!(parse_response(a, &[0], 1).is_err());
    }

    #[test]
    fn schema_enums_match_serde_names() {
        for s in SCENE_TYPES {
            serde_json::from_value::<SceneType>(serde_json::json!(s)).unwrap();
        }
        for d in DEFECTS {
            serde_json::from_value::<Defect>(serde_json::json!(d)).unwrap();
        }
    }
}
