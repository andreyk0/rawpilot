//! The faces pass: picking a people or wildlife burst's winner by eyes and expression.
//!
//! A series of near-identical frames of people is shot on purpose, so that one can be chosen
//! where the eyes are open — starting with the person most in focus — and the expressions and
//! poses work. At 1024 px a group's eyes are a few pixels, and behind glasses they are a few
//! shades of grey, so:
//!
//! 1. Gemini locates the faces in the burst's candidate frames (one request, consistent
//!    person ids across frames, the main subject marked).
//! 2. Each face is cropped at full resolution from the same source the locate request showed
//!    (`facecrop`): the raw when the request was developed, the camera preview otherwise. Two
//!    images per face: the band of it that holds the eyes, shadows lifted so lids and lashes
//!    behind tinted lenses show, and the whole face as it looks.
//! 3. Gemini locates the eye strip in each eye band (second request): a box from the brows to
//!    the middle of the nose across both eyes and the whole of any glasses. The strip is cut
//!    from the same source, exposed by its own pixels so that the inside of a lens gets the
//!    tonal range, lifted, and enlarged to 1024 px wide.
//! 4. Gemini reads the strips twice, each request asking one thing. The third reads each eye
//!    on its own against five anchored levels, noting what it sees before it answers. The
//!    fourth orders each person's frames from the most to the least open eyes. Neither is
//!    asked to rank or judge anything else.
//! 5. Gemini reads the whole faces (fifth request): each person's eyewear, each expression,
//!    and the frames ranked by how good the faces look together, with eyes expressly left out.
//! 6. The winner is chosen locally by a fixed order (`choose`), so the priorities are the
//!    photographer's and only the perception is the model's.
//!
//! What a closed eye costs depends on who can see it: eyes shut behind sunglasses are eyes
//! nobody will notice in the photograph, even when the model can tell. So the eyes of each
//! person are read as one fact and the eyewear as another, and how much a person's eyes weigh
//! is derived from the eyewear (`Person::eyewear`), never read per frame.

use std::cmp::Reverse;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::analyze::Analyzed;
use crate::cache::Cache;
use crate::config::JudgeConfig;
use crate::facecrop::{self, BandBox, Box2d, Crop, FaceFiles, Layout};
use crate::gemini::{self, Usage};
use crate::group::Burst;
use crate::judge::{self, BurstJudgement, SceneType, Shown, Verdict};
use crate::rpl::FaceBox;

/// More people than this are not read: in a crowd only the subjects matter, and Gemini is
/// asked to return the subjects first.
const PEOPLE_MAX: usize = 6;

/// What a person wears over their eyes. How much their eyes weigh is `Noticeability`'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Eyewear {
    None,
    Clear,
    Tinted,
    Dark,
}

impl Eyewear {
    fn describe(self) -> &'static str {
        match self {
            Eyewear::None => "bare",
            Eyewear::Clear => "behind clear lenses",
            Eyewear::Tinted => "behind tinted lenses",
            Eyewear::Dark => "behind dark lenses",
        }
    }
}

/// How open one eye is, on the five levels `LEVELS` describes to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "i64")]
pub enum EyeLevel {
    Shut = 0,
    Slit = 1,
    Narrowed = 2,
    Open = 3,
    Wide = 4,
}

impl EyeLevel {
    const ALL: [EyeLevel; 5] = [
        EyeLevel::Shut,
        EyeLevel::Slit,
        EyeLevel::Narrowed,
        EyeLevel::Open,
        EyeLevel::Wide,
    ];

    fn describe(self) -> &'static str {
        match self {
            EyeLevel::Shut => "shut",
            EyeLevel::Slit => "slit",
            EyeLevel::Narrowed => "narrowed",
            EyeLevel::Open => "open",
            EyeLevel::Wide => "wide open",
        }
    }

    /// How many levels this one is above `lower`.
    fn steps_above(self, lower: EyeLevel) -> usize {
        (self as usize)
            .checked_sub(lower as usize)
            .unwrap_or_else(|| panic!("{self:?} is below {lower:?}"))
    }
}

impl TryFrom<i64> for EyeLevel {
    type Error = String;

    fn try_from(n: i64) -> Result<EyeLevel, String> {
        EyeLevel::ALL
            .into_iter()
            .find(|&l| l as i64 == n)
            .ok_or_else(|| {
                format!(
                    "eye level {n} is outside {}..={}",
                    EyeLevel::Shut as i64,
                    EyeLevel::Wide as i64
                )
            })
    }
}

/// What the two eyes of one face show in one frame; `None` for an eye that cannot be seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EyePair {
    pub left: Option<EyeLevel>,
    pub right: Option<EyeLevel>,
}

impl EyePair {
    /// The level the face counts by: its worse visible eye. `None` when neither is visible.
    pub fn worse(self) -> Option<EyeLevel> {
        match (self.left, self.right) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        }
    }
}

/// Where a person's face in one frame stands in the model's order of that person's frames by
/// how open their eyes look.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compared {
    /// The index of the group of equally open frames the face is in, 0 = the most open.
    Group(usize),
    /// The model could not see this person's eyes in the frame.
    NotVisible,
}

impl std::fmt::Display for Compared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Compared::Group(g) => write!(f, "group {g}"),
            Compared::NotVisible => f.write_str("not visible"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expression {
    Awkward,
    Neutral,
    Good,
}

const EYEWEAR: &[&str] = &["none", "clear", "tinted", "dark"];
const EXPRESSIONS: &[&str] = &["awkward", "neutral", "good"];

/// A person who has a face in at least one candidate frame. Their eyewear is one fact for the
/// whole burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Person {
    pub id: usize,
    pub main: bool,
    pub eyewear: Eyewear,
    /// How many groups of equally open eyes the model made of this person's frames.
    pub eye_groups: usize,
}

/// The burst's people, main subject first, each once.
#[derive(Debug, Clone, PartialEq)]
pub struct People(Vec<Person>);

impl People {
    /// The people of `located` with a face, main first, each with their eyewear and their
    /// number of eye groups; both given in `Located::seen`'s order.
    fn new(located: &Located, eyewear: Vec<Eyewear>, eye_groups: Vec<usize>) -> People {
        let seen = located.seen();
        assert_eq!(eyewear.len(), seen.len(), "one eyewear per person seen");
        assert_eq!(
            eye_groups.len(),
            seen.len(),
            "one group count per person seen"
        );
        People(
            seen.iter()
                .zip(eyewear)
                .zip(eye_groups)
                .map(|((&(id, main), eyewear), eye_groups)| Person {
                    id,
                    main,
                    eyewear,
                    eye_groups,
                })
                .collect(),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceRead {
    pub person: usize,
    /// Where the face was located, in fractions of the displayed image.
    pub at: Box2d,
    pub eyes: EyePair,
    pub compared: Compared,
    pub expression: Expression,
}

/// The pass's result for one burst.
#[derive(Debug, Clone, PartialEq)]
pub struct BurstFaces {
    /// Member index of the chosen frame.
    pub winner: usize,
    pub people: People,
    /// Per member: what was read, or `None` for a frame that was not a candidate.
    pub members: Vec<Option<MemberFaces>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemberFaces {
    pub faces: Vec<FaceRead>,
    /// Place in Gemini's side-by-side ranking of the candidates' faces, 0 = best.
    pub place: usize,
    /// How many candidates were ranked.
    pub of: usize,
}

impl MemberFaces {
    fn face(&self, person: usize) -> Option<&FaceRead> {
        self.faces.iter().find(|f| f.person == person)
    }

    /// The level of `person`'s eyes in this frame; `None` when they have no face here or
    /// neither eye was visible.
    fn read_level(&self, person: usize) -> Option<EyeLevel> {
        self.face(person)?.eyes.worse()
    }

    /// Where `person` stands in the model's order of their frames; absent counts as not
    /// visible.
    fn compared(&self, person: usize) -> Compared {
        self.face(person)
            .map_or(Compared::NotVisible, |f| f.compared)
    }
}

/// The level counted for eyes nobody could read: both eyes unseen, or no face of that person
/// in the frame. It is below `Open`, because an open eye was seen to be fine, and above `Slit`
/// and `Shut`, because those were seen to be bad; unread eyes are neither known to be blinking
/// nor known to be open.
const UNREAD_LEVEL: EyeLevel = EyeLevel::Narrowed;

/// The level `choose` counts for eyes read as `read`, or unread.
fn counted_level(read: Option<EyeLevel>) -> EyeLevel {
    read.unwrap_or(UNREAD_LEVEL)
}

/// The group of a person's compare answer counted for a frame in which they are not visible
/// or absent: the group at index `groups / 2`, rounded down. That is the middle group of an
/// odd number, the less open of two, and group 0 for one group or none.
fn middle_group(groups: usize) -> usize {
    groups / 2
}

/// How likely a person's eyes are to be noticed in the photograph, most noticeable first: the
/// main subject's are what the picture is about, then the others' by how well their eyewear
/// shows them, and eyes behind dark lenses are seen by no one, whoever they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Noticeability {
    /// The main subject, bare or behind clear or tinted lenses.
    Main,
    Bare,
    Clear,
    Tinted,
    /// Anyone, the main subject included.
    Dark,
}

impl Noticeability {
    /// The tiers, counting from the last variant; `every_tier_is_reachable_and_counted` pins
    /// that nothing is declared after it.
    const COUNT: usize = Noticeability::Dark as usize + 1;

    fn of(person: Person) -> Noticeability {
        match (person.main, person.eyewear) {
            (_, Eyewear::Dark) => Noticeability::Dark,
            (true, Eyewear::None | Eyewear::Clear | Eyewear::Tinted) => Noticeability::Main,
            (false, Eyewear::None) => Noticeability::Bare,
            (false, Eyewear::Clear) => Noticeability::Clear,
            (false, Eyewear::Tinted) => Noticeability::Tinted,
        }
    }
}

/// How far one frame is from a person's best candidate frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Deficit {
    person: Person,
    /// The highest level the candidates count for the person.
    best: EyeLevel,
    /// The frame's level for the person; `None` when unread.
    level: Option<EyeLevel>,
    /// Where the frame stands in the model's order of the person's frames.
    compared: Compared,
}

impl Deficit {
    /// Level steps below the person's best level; unread eyes are counted at `UNREAD_LEVEL`.
    fn level_steps(&self) -> usize {
        self.best.steps_above(counted_level(self.level))
    }

    /// Compare groups behind the most open one; a frame in which the person is not visible
    /// is counted at `middle_group`.
    fn groups(&self) -> usize {
        match self.compared {
            Compared::Group(g) => g,
            Compared::NotVisible => middle_group(self.person.eye_groups),
        }
    }
}

/// The deficit of `member` for each person, against the `candidates` it is chosen among.
/// A person's best level is the highest the candidates count for them, unread frames
/// included, so someone read in no candidate has no deficit anywhere.
fn deficits(people: &People, candidates: &[&MemberFaces], member: &MemberFaces) -> Vec<Deficit> {
    people
        .0
        .iter()
        .map(|&person| Deficit {
            person,
            best: candidates
                .iter()
                .map(|c| counted_level(c.read_level(person.id)))
                .max()
                .expect("at least one candidate"),
            level: member.read_level(person.id),
            compared: member.compared(person.id),
        })
        .collect()
}

fn plural(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}

fn who(person: Person) -> String {
    if person.main {
        "main".to_string()
    } else {
        format!("person {}", person.id)
    }
}

impl BurstFaces {
    /// The faces located in a member, left to right; `None` for a member the pass did not
    /// read (not a candidate), `Some` of none for one it read and found no face in.
    pub fn located(&self, member: usize) -> Option<Vec<FaceBox>> {
        let m = self.members[member].as_ref()?;
        Some(
            m.faces
                .iter()
                .map(|f| FaceBox {
                    person: f.person,
                    at: f.at,
                })
                .collect(),
        )
    }

    /// "faces #2 of 6: 1 face, eyes: main 1 step below their best (behind dark lenses), 1
    /// awkward" for a member, for captions and the report: what the choice weighed.
    pub fn summary(&self, member: usize) -> Option<String> {
        let m = self.members[member].as_ref()?;
        let candidates: Vec<&MemberFaces> = self.members.iter().flatten().collect();
        let d = deficits(&self.people, &candidates, m);
        let mut parts = vec![plural(m.faces.len(), "face"), eyes_summary(&d)];
        parts.extend(compare_summary(&d));
        let awkward = m
            .faces
            .iter()
            .filter(|f| f.expression == Expression::Awkward)
            .count();
        if awkward > 0 {
            parts.push(format!("{awkward} awkward"));
        }
        Some(format!(
            "faces #{} of {}: {}",
            m.place + 1,
            m.of,
            parts.join(", ")
        ))
    }
}

/// Who is below their best level or unread, with their eyewear; "eyes at their best" when no
/// one is.
fn eyes_summary(deficits: &[Deficit]) -> String {
    let items: Vec<String> = deficits
        .iter()
        .filter_map(|d| {
            let what = match (d.level, d.level_steps()) {
                (Some(_), 0) => return None,
                (Some(_), n) => format!("{} below their best", plural(n, "step")),
                (None, 0) => "unread".to_string(),
                (None, n) => format!("unread, counted {} below their best", plural(n, "step")),
            };
            Some(format!(
                "{} {what} ({})",
                who(d.person),
                d.person.eyewear.describe()
            ))
        })
        .collect();
    if items.is_empty() {
        "eyes at their best".to_string()
    } else {
        format!("eyes: {}", items.join("; "))
    }
}

/// Who is behind the most open group of their frames, if anyone is.
fn compare_summary(deficits: &[Deficit]) -> Option<String> {
    let items: Vec<String> = deficits
        .iter()
        .filter(|d| d.groups() > 0)
        .map(|d| {
            let behind = plural(d.groups(), "group");
            match d.compared {
                Compared::Group(_) => format!("{} {behind} from the most open", who(d.person)),
                Compared::NotVisible => format!(
                    "{} not visible, counted {behind} from the most open",
                    who(d.person)
                ),
            }
        })
        .collect();
    (!items.is_empty()).then(|| format!("compare: {}", items.join("; ")))
}

#[derive(Debug, Deserialize)]
struct RawLocate {
    people: Vec<RawPerson>,
    frames: Vec<RawFrameFaces>,
}

#[derive(Debug, Deserialize)]
struct RawPerson {
    id: usize,
    main: bool,
}

#[derive(Debug, Deserialize)]
struct RawFrameFaces {
    frame: usize,
    faces: Vec<RawFace>,
}

#[derive(Debug, Deserialize)]
struct RawFace {
    person: usize,
    /// `[ymin, xmin, ymax, xmax]` on a 0–1000 grid, Gemini's native box format.
    box_2d: [i64; 4],
}

/// The people a locate answer lists: exactly one main subject, then the others in the
/// answer's priority order.
#[derive(Debug, Clone, PartialEq)]
struct Subjects {
    main: usize,
    others: Vec<usize>,
}

impl Subjects {
    /// The listed people (id, marked main), in the answer's order, as one main subject and
    /// the others; `None` for no one. The answer is meant to mark exactly one main. When it
    /// marks several, the first marked is main and the rest follow it as others; when it
    /// marks none, the first listed is main. Either is said, naming who moved.
    /// `main_cut`: a person marked main was listed beyond `PEOPLE_MAX` and so is not here.
    fn resolve(listed: &[(usize, bool)], main_cut: bool) -> Option<Subjects> {
        let marked: Vec<usize> = listed.iter().filter(|p| p.1).map(|p| p.0).collect();
        let unmarked = listed.iter().filter(|p| !p.1).map(|p| p.0);
        let mut ordered = marked.iter().copied().chain(unmarked);
        let main = ordered.next()?;
        let others: Vec<usize> = ordered.collect();
        match marked.as_slice() {
            [_] => {}
            [] => warn!(
                promoted = main,
                main_cut, "no main subject among the people read; the first listed is taken"
            ),
            [_, demoted @ ..] => warn!(
                main,
                demoted = ?demoted,
                "several people marked main; the first marked is kept"
            ),
        }
        Some(Subjects { main, others })
    }

    /// Main first, each with whether they are the main subject.
    fn ordered(&self) -> impl Iterator<Item = (usize, bool)> + '_ {
        std::iter::once((self.main, true)).chain(self.others.iter().map(|&id| (id, false)))
    }
}

/// Located faces: the `people` listed, and per request frame the boxes of those people
/// visible in it.
#[derive(Debug, Clone, PartialEq)]
struct Located {
    people: Option<Subjects>,
    frames: Vec<Vec<(usize, Box2d)>>,
}

impl Located {
    /// The people with a face in some frame, main subject first: the ones there is
    /// something to read of.
    fn seen(&self) -> Vec<(usize, bool)> {
        self.people
            .iter()
            .flat_map(Subjects::ordered)
            .filter(|&(id, _)| self.frames.iter().flatten().any(|&(p, _)| p == id))
            .collect()
    }

    /// Whether `id` is the main subject.
    fn is_main(&self, id: usize) -> bool {
        self.people.as_ref().is_some_and(|s| s.main == id)
    }

    /// Whether `person` has a face in request frame `frame`.
    fn has_face(&self, frame: usize, person: usize) -> bool {
        self.frames[frame].iter().any(|&(p, _)| p == person)
    }

    /// The main subject's id, when they have a face in some frame.
    fn main_seen(&self) -> Option<usize> {
        self.seen()
            .into_iter()
            .find(|&(_, main)| main)
            .map(|(id, _)| id)
    }

    /// Says so when there is no main subject to compare frames by: the choice then starts at
    /// the others' eyes.
    fn warn_if_main_unseen(&self) {
        if self.main_seen().is_none() {
            warn!("the main subject has no face in any frame; no frame is preferred for theirs");
        }
    }
}

/// One request frame in a model answer: its faces, each with the value read for it.
#[derive(Debug, Deserialize)]
struct RawFrame<F> {
    frame: usize,
    faces: Vec<F>,
}

#[derive(Debug, Deserialize)]
struct RawEyewear {
    person: usize,
    eyewear: Eyewear,
}

#[derive(Debug, Deserialize)]
struct RawStripBoxes {
    boxes: Vec<RawStripBox>,
}

#[derive(Debug, Deserialize)]
struct RawStripBox {
    /// The number in the image's caption, counting the faces in request order.
    image: usize,
    /// `[ymin, xmin, ymax, xmax]` on a 0–1000 grid of the eye band image.
    box_2d: [i64; 4],
}

/// One eye of a levels answer. The model's `seen` note, which it writes first so that it looks
/// before it answers, is not read.
#[derive(Debug, Deserialize)]
struct RawEye {
    /// Null for an eye that cannot be seen; a missing key is not null.
    #[serde(deserialize_with = "Option::deserialize")]
    level: Option<EyeLevel>,
}

#[derive(Debug, Deserialize)]
struct RawLevel {
    person: usize,
    left: RawEye,
    right: RawEye,
}

/// The levels request's answer.
#[derive(Debug, Deserialize)]
struct RawLevelsRead {
    frames: Vec<RawFrame<RawLevel>>,
}

#[derive(Debug, Deserialize)]
struct RawPersonOrder {
    person: usize,
    /// Groups of frame numbers, the most open eyes first; the frames of a group look alike.
    most_open_first: Vec<Vec<usize>>,
    not_visible: Vec<usize>,
}

/// The compare request's answer.
#[derive(Debug, Deserialize)]
struct RawCompareRead {
    people: Vec<RawPersonOrder>,
}

/// A validated compare answer: for each person with a face, in `Located::seen`'s order, how
/// many groups their frames fell into; and per request frame, for the faces in
/// `Located::frames`' order, where each stands.
#[derive(Debug, Clone, PartialEq)]
struct CompareRead {
    groups: Vec<usize>,
    compared: Vec<Vec<Compared>>,
}

#[derive(Debug, Deserialize)]
struct RawExpression {
    person: usize,
    expression: Expression,
}

/// The expression request's answer.
#[derive(Debug, Deserialize)]
struct RawExpressionRead {
    /// Each person's eyewear as the natural face crops show it.
    people: Vec<RawEyewear>,
    frames: Vec<RawFrame<RawExpression>>,
    /// Request frames ordered by how good the faces look, best first.
    ranking: Vec<usize>,
}

/// A validated expression answer: the eyewear of each person with a face, in
/// `Located::seen`'s order, per request frame the expressions of the faces cropped into it, in
/// `Located::frames`' order, and each frame's place in the ranking (0 = best faces).
#[derive(Debug, Clone, PartialEq)]
struct ExpressionRead {
    eyewear: Vec<Eyewear>,
    expressions: Vec<Vec<Expression>>,
    place: Vec<usize>,
}

/// The four read requests. Each asks about one thing, of the image that shows it best.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    /// Where the eye strip is in each eye band.
    StripBox,
    /// How open each eye is in each strip.
    Levels,
    /// How each person's frames order by how open their eyes look, from their strips.
    Compare,
    /// Expression and eyewear, from the whole faces.
    Expression,
}

impl Read {
    fn instructions(self, frames: usize) -> String {
        match self {
            Read::StripBox => STRIP_BOX_INSTRUCTIONS.to_string(),
            Read::Levels => levels_instructions(frames),
            Read::Compare => compare_instructions(frames),
            Read::Expression => expression_instructions(frames),
        }
    }

    fn schema(self) -> Value {
        match self {
            Read::StripBox => strip_box_schema(),
            Read::Levels => levels_schema(),
            Read::Compare => compare_schema(),
            Read::Expression => expression_schema(),
        }
    }

    /// The text before a face's image: `image` counts the faces in request order, `frame` is
    /// the request frame. Only the expression request weighs people, so only it is told who the
    /// main subject is.
    fn caption(self, image: usize, frame: usize, person: usize, main: bool) -> String {
        match (self, main) {
            (Read::StripBox, _) => format!("Image {image}:"),
            (Read::Levels | Read::Compare, _) => format!("Frame {frame}, person {person}:"),
            (Read::Expression, true) => format!("Frame {frame}, person {person} (main subject)"),
            (Read::Expression, false) => format!("Frame {frame}, person {person}"),
        }
    }

    /// The text standing in for a frame with no faces. The strip request numbers images, not
    /// frames, so it has none.
    fn empty_frame(self, frame: usize) -> Option<String> {
        match self {
            Read::StripBox => None,
            Read::Levels | Read::Compare | Read::Expression => {
                Some(format!("Frame {frame}: no faces"))
            }
        }
    }
}

/// One slot per burst: `Some` where the pass ran and read faces.
#[allow(clippy::too_many_arguments)]
pub fn faces_all(
    shots: &[Analyzed],
    bursts: &[&Burst],
    judgements: &[Option<BurstJudgement>],
    rel: &[f64],
    sharpness_rel_min: f64,
    cache: &Cache,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<Vec<Option<BurstFaces>>> {
    let dir = cache.root().join(gemini::Pass::Faces.dir_name());
    std::fs::create_dir_all(dir.join("crops"))?;
    std::fs::create_dir_all(dir.join("eye_strips"))?;
    std::fs::create_dir_all(dir.join("strips"))?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.concurrency)
        .build()?;
    let work: Vec<(usize, Vec<usize>)> = bursts
        .iter()
        .zip(judgements)
        .enumerate()
        .filter_map(|(b, (burst, j))| {
            let j = j.as_ref()?;
            let cands = candidates(burst, j, rel, sharpness_rel_min, cfg.frames_per_request_max);
            (has_faces(j.scene_type) && cands.len() >= 2).then_some((b, cands))
        })
        .collect();
    let results: Vec<Result<(Option<BurstFaces>, Option<Usage>)>> = pool.install(|| {
        work.par_iter()
            .map(|(b, cands)| {
                let j = judgements[*b].as_ref().expect("filtered on judged bursts");
                faces_burst(shots, bursts[*b], j, rel, cands, cache, &dir, cfg, api_key)
            })
            .collect()
    });
    let mut tally = gemini::Tally::default();
    let mut out: Vec<Option<BurstFaces>> = vec![None; bursts.len()];
    for ((b, _), r) in work.iter().zip(results) {
        tally.record(&r);
        match r {
            Ok((faces, _)) => out[*b] = faces,
            Err(e) => {
                let first = shots[bursts[*b].members()[0]].shot.name();
                warn!(burst = %first, "faces pass failed, keeping the burst pass's winner: {e:#}");
            }
        }
    }
    tally.log("faces pass");
    Ok(out)
}

fn has_faces(t: SceneType) -> bool {
    match t {
        SceneType::People | SceneType::Wildlife => true,
        SceneType::Landscape | SceneType::Detail | SceneType::Architecture | SceneType::Other => {
            false
        }
    }
}

/// Members worth choosing between: judged, not rejected, sharp enough that a viewer would not
/// notice the difference; at most `max`, sharpest first, returned in member order.
fn candidates(
    burst: &Burst,
    j: &BurstJudgement,
    rel: &[f64],
    sharpness_rel_min: f64,
    max: usize,
) -> Vec<usize> {
    let rel = burst.of(rel);
    let mut c: Vec<usize> = (0..rel.len())
        .filter(|&m| {
            j.frames[m]
                .as_ref()
                .is_some_and(|f| f.verdict != Verdict::Reject)
                && rel[m] >= sharpness_rel_min
        })
        .collect();
    c.sort_by(|&a, &b| rel[b].total_cmp(&rel[a]));
    c.truncate(max);
    c.sort_unstable();
    c
}

/// The requests' usage summed; `None` when every one was cached.
fn sum_usage(usages: impl IntoIterator<Item = Option<Usage>>) -> Option<Usage> {
    usages.into_iter().flatten().reduce(|mut sum, u| {
        sum += u;
        sum
    })
}

/// The answer text as the requested JSON.
fn from_json<T: DeserializeOwned>(text: &str) -> Result<T> {
    serde_json::from_str(text)
        .with_context(|| format!("model answer is not the requested JSON: {text}"))
}

/// Locate, then read the strips' boxes, levels and order, then the expressions; usage summed
/// over the requests made.
#[allow(clippy::too_many_arguments)]
fn faces_burst(
    shots: &[Analyzed],
    burst: &Burst,
    j: &BurstJudgement,
    rel: &[f64],
    cands: &[usize],
    cache: &Cache,
    dir: &Path,
    cfg: &JudgeConfig,
    api_key: &str,
) -> Result<(Option<BurstFaces>, Option<Usage>)> {
    // Names the burst in every warning below.
    let _burst =
        tracing::info_span!("faces", burst = %shots[burst.members()[0]].shot.name()).entered();
    let frames: Vec<&Analyzed> = cands.iter().map(|&m| &shots[burst.members()[m]]).collect();
    let shown = judge::shown_for_frames(frames.iter().copied(), cfg);

    let mut parts = vec![gemini::Part::text(locate_instructions(frames.len()))];
    for (k, a) in frames.iter().enumerate() {
        parts.push(gemini::Part::text(format!("Frame {k}:")));
        parts.push(gemini::Part::jpeg(&judge::judging_jpeg(cache, a, shown)?));
    }
    let req = gemini::Request::new(cfg, parts, locate_schema());
    let (located, locate_usage) = gemini::ask(api_key, cfg, &req, dir, |text| {
        parse_locate(from_json(text)?, frames.len())
    })?;
    if located.seen().is_empty() {
        return Ok((None, locate_usage));
    }
    located.warn_if_main_unseen();

    let layouts = face_layouts(&frames, &located, shown)?;
    let files = crop_faces(&frames, &layouts, shown, &dir.join("crops"))?;
    for (a, files) in frames.iter().zip(&files) {
        let contact = dir.join("strips").join(format!("{}.jpg", a.shot.name()));
        facecrop::sync_contact_strip(&contact, files)?;
    }

    let req = read_request(
        Read::StripBox,
        &located,
        &crop_paths(&files, Crop::EyeBand),
        cfg,
    )?;
    let (boxes, strip_usage) = gemini::ask(api_key, cfg, &req, dir, |text| {
        parse_strip_boxes(from_json(text)?, &located)
    })?;
    let strips = crop_eye_strips(&frames, &layouts, &boxes, shown, &dir.join("eye_strips"))?;

    let req = read_request(Read::Levels, &located, &strips, cfg)?;
    let (levels, levels_usage) = gemini::ask(api_key, cfg, &req, dir, |text| {
        parse_levels(from_json(text)?, &located)
    })?;
    let req = read_request(Read::Compare, &located, &strips, cfg)?;
    let (compare, compare_usage) = gemini::ask(api_key, cfg, &req, dir, |text| {
        parse_compare(from_json(text)?, &located)
    })?;
    let req = read_request(
        Read::Expression,
        &located,
        &crop_paths(&files, Crop::Face),
        cfg,
    )?;
    let (expression, expression_usage) = gemini::ask(api_key, cfg, &req, dir, |text| {
        parse_expression(from_json(text)?, &located)
    })?;

    let people = People::new(&located, expression.eyewear, compare.groups);
    let of = cands.len();
    let mut per_member: Vec<Option<MemberFaces>> = vec![None; burst.members().len()];
    let faces = faces_read(
        &located,
        &levels,
        &compare.compared,
        &expression.expressions,
    );
    for (k, (faces, place)) in faces.into_iter().zip(expression.place).enumerate() {
        for face in &faces {
            log_read(&frames[k].shot.name(), face);
        }
        per_member[cands[k]] = Some(MemberFaces { faces, place, of });
    }
    let burst_rel = burst.of(rel);
    let winner = choose(cands, &people, &per_member, j.best, |m| burst_rel[m]);
    Ok((
        Some(BurstFaces {
            winner,
            people,
            members: per_member,
        }),
        sum_usage([
            locate_usage,
            strip_usage,
            levels_usage,
            compare_usage,
            expression_usage,
        ]),
    ))
}

/// One debug line for a face read, so that a run can be scored against hand labels.
fn log_read(frame: &str, read: &FaceRead) {
    let eye = |e: Option<EyeLevel>| e.map_or("not visible", EyeLevel::describe);
    debug!(
        %frame,
        person = read.person,
        left = %eye(read.eyes.left),
        right = %eye(read.eyes.right),
        compare = %read.compared,
        "face read"
    );
}

/// Each frame's faces with the eyes, the place in the order of the person's frames and the
/// expression read for them.
fn faces_read(
    located: &Located,
    levels: &[Vec<EyePair>],
    compared: &[Vec<Compared>],
    expressions: &[Vec<Expression>],
) -> Vec<Vec<FaceRead>> {
    located
        .frames
        .iter()
        .zip(levels)
        .zip(compared)
        .zip(expressions)
        .map(|(((faces, levels), compared), expressions)| {
            faces
                .iter()
                .zip(levels)
                .zip(compared)
                .zip(expressions)
                .map(
                    |((((person, at), &eyes), &compared), &expression)| FaceRead {
                        person: *person,
                        at: *at,
                        eyes,
                        compared,
                        expression,
                    },
                )
                .collect()
        })
        .collect()
}

/// Where the faces of every request frame are cut from, in `located`'s order; `None` for a
/// frame with no faces. Every frame is cut from `shown`, the source its locate boxes refer
/// to; a frame that cannot be laid out fails the burst rather than being read from another
/// source.
fn face_layouts(
    frames: &[&Analyzed],
    located: &Located,
    shown: Shown,
) -> Result<Vec<Option<Layout>>> {
    frames
        .iter()
        .zip(&located.frames)
        .map(|(a, faces)| {
            if faces.is_empty() {
                return Ok(None);
            }
            let boxes: Vec<Box2d> = faces.iter().map(|(_, b)| *b).collect();
            Layout::new(&a.shot, shown, &boxes).map(Some)
        })
        .collect()
}

/// The files of every located face, per request frame, in `located`'s order.
fn crop_faces(
    frames: &[&Analyzed],
    layouts: &[Option<Layout>],
    shown: Shown,
    crops: &Path,
) -> Result<Vec<Vec<FaceFiles>>> {
    frames
        .iter()
        .zip(layouts)
        .map(|(a, layout)| match layout {
            None => Ok(Vec::new()),
            Some(layout) => facecrop::face_files(crops, &a.shot, shown, layout),
        })
        .collect()
}

/// The eye strip file of every located face, per request frame, in `located`'s order, cut
/// from `shown` like the face crops they lie in.
fn crop_eye_strips(
    frames: &[&Analyzed],
    layouts: &[Option<Layout>],
    boxes: &[Vec<BandBox>],
    shown: Shown,
    dir: &Path,
) -> Result<Vec<Vec<PathBuf>>> {
    frames
        .iter()
        .zip(layouts)
        .zip(boxes)
        .map(|((a, layout), strips)| match layout {
            None => Ok(Vec::new()),
            Some(layout) => facecrop::eye_strip_files(dir, &a.shot, shown, layout, strips),
        })
        .collect()
}

/// The `crop` file of every face, as `read_request` takes images.
fn crop_paths(files: &[Vec<FaceFiles>], crop: Crop) -> Vec<Vec<PathBuf>> {
    files
        .iter()
        .map(|frame| frame.iter().map(|f| f.get(crop).to_path_buf()).collect())
        .collect()
}

/// The request for `read`: its instructions, then each located face as one image captioned for
/// `read`, in `located`'s order. `images` holds one file per face. The images are the bytes of
/// the cached files, so a warm run sends what the cold run did.
fn read_request(
    read: Read,
    located: &Located,
    images: &[Vec<PathBuf>],
    cfg: &JudgeConfig,
) -> Result<gemini::Request> {
    let n = located.frames.len();
    let mut parts = vec![gemini::Part::text(read.instructions(n))];
    let mut image = 0;
    for (k, (faces, paths)) in located.frames.iter().zip(images).enumerate() {
        if faces.is_empty() {
            parts.extend(read.empty_frame(k).map(gemini::Part::text));
        }
        for ((person, _), path) in faces.iter().zip(paths) {
            let jpeg = std::fs::read(path)
                .with_context(|| format!("reading face image {}", path.display()))?;
            let main = located.is_main(*person);
            parts.push(gemini::Part::text(read.caption(image, k, *person, main)));
            parts.push(gemini::Part::jpeg(&jpeg));
            image += 1;
        }
    }
    Ok(gemini::Request::new(cfg, parts, read.schema()))
}

/// What a member's faces weigh in `choose`, best last: each field decides only when every
/// field before it ties.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Standing {
    /// Level deficits per `Noticeability`, most noticeable first: a smaller deficit on an
    /// earlier tier beats any amount on a later one.
    level_deficits: Reverse<[usize; Noticeability::COUNT]>,
    /// Compare deficits, the same way.
    compare_deficits: Reverse<[usize; Noticeability::COUNT]>,
    awkward: Reverse<usize>,
    place: Reverse<usize>,
    judged_best: bool,
}

impl Standing {
    fn of(
        member: &MemberFaces,
        people: &People,
        candidates: &[&MemberFaces],
        judged_best: bool,
    ) -> Standing {
        let d = deficits(people, candidates, member);
        let by_tier = |amount: fn(&Deficit) -> usize| {
            let mut tiers = [0; Noticeability::COUNT];
            for deficit in &d {
                tiers[Noticeability::of(deficit.person) as usize] += amount(deficit);
            }
            Reverse(tiers)
        };
        Standing {
            level_deficits: by_tier(Deficit::level_steps),
            compare_deficits: by_tier(Deficit::groups),
            awkward: Reverse(
                member
                    .faces
                    .iter()
                    .filter(|f| f.expression == Expression::Awkward)
                    .count(),
            ),
            place: Reverse(member.place),
            judged_best,
        }
    }
}

/// The photographer's rule, each step deciding only when all earlier ones tie. For each
/// person, a frame is measured against the best frame there is of them among the candidates:
///
/// - the level deficit is how many levels below the person's best frame their eyes are in
///   this one (`EyePair::worse`: a face counts by its worse visible eye; unread eyes count
///   as `UNREAD_LEVEL`);
/// - the compare deficit is the index of the person's group in the model's order of their
///   frames by how open their eyes look (0 = the most open; `middle_group` when they are not
///   visible or absent).
///
/// The steps are:
///
/// 1. level deficits, summed per `Noticeability` tier and compared tier by tier from the most
///    noticeable: the main subject bare or behind clear or tinted lenses; then others bare,
///    behind clear lenses, behind tinted lenses; last anyone's behind dark lenses, the main
///    subject's included, because nobody sees those;
/// 2. compare deficits, the same way;
/// 3. fewest awkward expressions;
/// 4. the expression request's ranking, 5. the burst pass's choice, 6. sharpness.
///
/// The anchored levels decide before the model's order of the frames because they are facts
/// about each eye, while the order separates frames the levels call equal. Deficits are
/// compared tier by tier, never summed across tiers: any deficit on a bare face loses to
/// every deficit behind tinted or dark lenses together.
fn choose(
    cands: &[usize],
    people: &People,
    members: &[Option<MemberFaces>],
    judged_best: usize,
    sharpness: impl Fn(usize) -> f64,
) -> usize {
    let faces = |m: usize| members[m].as_ref().expect("a candidate was read");
    let candidates: Vec<&MemberFaces> = cands.iter().map(|&m| faces(m)).collect();
    let standing = |m: usize| Standing::of(faces(m), people, &candidates, m == judged_best);
    *cands
        .iter()
        .max_by(|&&a, &&b| {
            standing(a)
                .cmp(&standing(b))
                .then(sharpness(a).total_cmp(&sharpness(b)))
        })
        .expect("at least two candidates")
}

fn parse_locate(raw: RawLocate, n: usize) -> Result<Located> {
    let mut kept: Vec<(usize, bool)> = Vec::new();
    // Listed beyond `PEOPLE_MAX`: not read, and their faces are dropped.
    let mut cut: Vec<usize> = Vec::new();
    let mut main_cut = false;
    for (listed, p) in raw.people.into_iter().enumerate() {
        ensure!(
            kept.iter().all(|(id, _)| *id != p.id) && !cut.contains(&p.id),
            "person {} listed twice",
            p.id
        );
        if listed < PEOPLE_MAX {
            kept.push((p.id, p.main));
        } else {
            main_cut |= p.main;
            warn!(
                person = p.id,
                main = p.main,
                "person listed beyond {PEOPLE_MAX}, not read"
            );
            cut.push(p.id);
        }
    }
    let people = Subjects::resolve(&kept, main_cut);
    let mut frames: Vec<Option<Vec<(usize, Box2d)>>> = vec![None; n];
    for f in raw.frames {
        ensure!(f.frame < n, "frame {} out of range 0..{n}", f.frame);
        ensure!(frames[f.frame].is_none(), "frame {} listed twice", f.frame);
        let mut boxes = Vec::new();
        for face in f.faces {
            if cut.contains(&face.person) {
                warn!(
                    frame = f.frame,
                    person = face.person,
                    "dropping the face of a person listed beyond {PEOPLE_MAX}"
                );
                continue;
            }
            ensure!(
                kept.iter().any(|(id, _)| *id == face.person),
                "frame {} has a face of person {}, who is not listed",
                f.frame,
                face.person
            );
            ensure!(
                boxes.iter().all(|(p, _)| *p != face.person),
                "frame {} has two faces of person {}",
                f.frame,
                face.person
            );
            boxes.push((face.person, Box2d::from_grid(face.box_2d)?));
        }
        // Left to right, the order they are attached in.
        boxes.sort_by(|a, b| a.1.x0.total_cmp(&b.1.x0));
        frames[f.frame] = Some(boxes);
    }
    let frames = frames
        .into_iter()
        .enumerate()
        .map(|(k, f)| f.with_context(|| format!("frame {k} missing")))
        .collect::<Result<_>>()?;
    Ok(Located { people, frames })
}

/// The strip boxes answer: every image answered once, each box inside the band. Arranged per
/// request frame, for its faces in `Located::frames`' order.
fn parse_strip_boxes(raw: RawStripBoxes, located: &Located) -> Result<Vec<Vec<BandBox>>> {
    let images: usize = located.frames.iter().map(Vec::len).sum();
    let mut boxes: Vec<Option<BandBox>> = vec![None; images];
    for b in raw.boxes {
        ensure!(
            b.image < images,
            "image {} out of range 0..{images}",
            b.image
        );
        ensure!(boxes[b.image].is_none(), "image {} answered twice", b.image);
        boxes[b.image] =
            Some(BandBox::from_grid(b.box_2d).with_context(|| format!("image {}", b.image))?);
    }
    let mut boxes = boxes
        .into_iter()
        .enumerate()
        .map(|(k, b)| b.with_context(|| format!("image {k} not answered")))
        .collect::<Result<Vec<_>>>()?
        .into_iter();
    Ok(located
        .frames
        .iter()
        .map(|faces| boxes.by_ref().take(faces.len()).collect())
        .collect())
}

/// The levels answer, per frame covering exactly the persons cropped into it; the eyes of the
/// faces in `Located::frames`' order.
fn parse_levels(raw: RawLevelsRead, located: &Located) -> Result<Vec<Vec<EyePair>>> {
    faces_per_frame(
        raw.frames
            .into_iter()
            .map(|f| {
                (
                    f.frame,
                    f.faces
                        .into_iter()
                        .map(|r| {
                            (
                                r.person,
                                EyePair {
                                    left: r.left.level,
                                    right: r.right.level,
                                },
                            )
                        })
                        .collect(),
                )
            })
            .collect(),
        located,
    )
}

/// The compare answer: every person with a face answered once, and every frame in which that
/// person has a face placed exactly once, in a non-empty group or as not visible. A frame in
/// which they have no face may be listed as not visible, and is then ignored; in a group, or
/// out of range, it is an error.
fn parse_compare(raw: RawCompareRead, located: &Located) -> Result<CompareRead> {
    let seen = located.seen();
    let orders = per_seen_person(
        "compare",
        raw.people.into_iter().map(|p| (p.person, p)).collect(),
        located,
    )?;
    let n = located.frames.len();
    let mut groups = Vec::new();
    // Per person, per request frame.
    let mut placed: Vec<Vec<Option<Compared>>> = Vec::new();
    for (&(id, _), order) in seen.iter().zip(orders) {
        ensure!(
            order.most_open_first.iter().all(|g| !g.is_empty()),
            "person {id}: an empty group"
        );
        let mut at: Vec<Option<Compared>> = vec![None; n];
        let grouped = order
            .most_open_first
            .iter()
            .enumerate()
            .flat_map(|(g, frames)| frames.iter().map(move |&f| (f, Compared::Group(g))));
        let hidden = order.not_visible.iter().map(|&f| (f, Compared::NotVisible));
        for (f, compared) in grouped.chain(hidden) {
            ensure!(f < n, "person {id}: frame {f} out of range 0..{n}");
            if !located.has_face(f, id) {
                // Absent and not visible count the same, so the model listing an absent person
                // as not visible says nothing wrong; placing them in a group does.
                ensure!(
                    compared == Compared::NotVisible,
                    "person {id}: frame {f} has no face of theirs"
                );
                warn!(
                    person = id,
                    frame = f,
                    "listed as not visible in a frame with no face of theirs, ignored"
                );
                continue;
            }
            ensure!(at[f].is_none(), "person {id}: frame {f} placed twice");
            at[f] = Some(compared);
        }
        if let Some(f) = (0..n).find(|&f| located.has_face(f, id) && at[f].is_none()) {
            bail!("person {id}: frame {f} not placed");
        }
        groups.push(order.most_open_first.len());
        placed.push(at);
    }
    let compared = located
        .frames
        .iter()
        .enumerate()
        .map(|(f, faces)| {
            faces
                .iter()
                .map(|&(person, _)| {
                    let i = seen
                        .iter()
                        .position(|&(id, _)| id == person)
                        .expect("a person with a face is seen");
                    placed[i][f].expect("every face was placed")
                })
                .collect()
        })
        .collect();
    Ok(CompareRead { groups, compared })
}

/// The expression answer, covering exactly the people with a face, exactly the persons
/// cropped into each frame, and ranking every frame once.
fn parse_expression(raw: RawExpressionRead, located: &Located) -> Result<ExpressionRead> {
    let eyewear = per_seen_person(
        "eyewear",
        raw.people
            .into_iter()
            .map(|w| (w.person, w.eyewear))
            .collect(),
        located,
    )?;
    let n = located.frames.len();
    let mut place: Vec<Option<usize>> = vec![None; n];
    for (pos, &f) in raw.ranking.iter().enumerate() {
        ensure!(f < n, "ranked frame {f} out of range 0..{n}");
        ensure!(place[f].is_none(), "frame {f} ranked twice");
        place[f] = Some(pos);
    }
    let place: Vec<usize> = place
        .into_iter()
        .enumerate()
        .map(|(k, p)| p.with_context(|| format!("frame {k} not ranked")))
        .collect::<Result<_>>()?;
    let expressions = faces_per_frame(
        raw.frames
            .into_iter()
            .map(|f| {
                (
                    f.frame,
                    f.faces
                        .into_iter()
                        .map(|r| (r.person, r.expression))
                        .collect(),
                )
            })
            .collect(),
        located,
    )?;
    Ok(ExpressionRead {
        eyewear,
        expressions,
        place,
    })
}

/// An answer's persons (`what` it is the answer for) must be exactly the expected ones, each
/// once, in any order.
fn ensure_same_persons(
    what: &str,
    got: impl Iterator<Item = usize>,
    want: impl Iterator<Item = usize>,
) -> Result<()> {
    let sorted = |mut ids: Vec<usize>| {
        ids.sort_unstable();
        ids
    };
    let (got, want) = (sorted(got.collect()), sorted(want.collect()));
    ensure!(
        got == want,
        "{what}: answer names persons {got:?}, expected {want:?}"
    );
    Ok(())
}

/// A value per person (`what` it is the answer for), which must cover exactly the people with
/// a face, each once; arranged in `Located::seen`'s order.
fn per_seen_person<T>(
    what: &str,
    mut answer: Vec<(usize, T)>,
    located: &Located,
) -> Result<Vec<T>> {
    let seen = located.seen();
    ensure_same_persons(
        what,
        answer.iter().map(|&(id, _)| id),
        seen.iter().map(|&(id, _)| id),
    )?;
    answer.sort_by_key(|&(id, _)| seen.iter().position(|&(s, _)| s == id));
    Ok(answer.into_iter().map(|(_, value)| value).collect())
}

/// A value read per face, per request frame, arranged in `located`'s order and covering
/// exactly the people cropped into each frame.
fn faces_per_frame<T>(
    frames: Vec<(usize, Vec<(usize, T)>)>,
    located: &Located,
) -> Result<Vec<Vec<T>>> {
    let n = located.frames.len();
    let mut out: Vec<Option<Vec<T>>> = (0..n).map(|_| None).collect();
    for (frame, mut faces) in frames {
        ensure!(frame < n, "frame {frame} out of range 0..{n}");
        ensure!(out[frame].is_none(), "frame {frame} listed twice");
        if located.frames[frame].is_empty() {
            // Nothing was cropped, so nothing was shown to read; an entry here is the model
            // describing a face it was never given.
            if !faces.is_empty() {
                warn!(frame, "ignoring faces read for a frame with no crops");
            }
            out[frame] = Some(Vec::new());
            continue;
        }
        let want: Vec<usize> = located.frames[frame].iter().map(|(p, _)| *p).collect();
        ensure_same_persons(
            &format!("frame {frame}"),
            faces.iter().map(|&(p, _)| p),
            want.iter().copied(),
        )?;
        faces.sort_by_key(|&(p, _)| want.iter().position(|&w| w == p));
        out[frame] = Some(faces.into_iter().map(|(_, value)| value).collect());
    }
    // A frame with no faces cropped needs no read.
    out.into_iter()
        .enumerate()
        .map(|(k, r)| match r {
            Some(r) => Ok(r),
            None if located.frames[k].is_empty() => Ok(Vec::new()),
            None => bail!("frame {k} not read"),
        })
        .collect()
}

fn locate_instructions(n: usize) -> String {
    format!(
        "Below are {n} frames of one burst from a travel photo shoot, numbered 0 to {last}: \
near-identical shots taken a fraction of a second apart.

List the people (or, for animals, the animals) that are the subject of the photo — at most \
{PEOPLE_MAX}, the most important first — ignoring strangers who are incidental. Give each a \
numeric id and keep the same id for the same individual in every frame. Mark exactly one as \
`main`: the subject most in focus or the one the photo is about.

For every frame, give the box of each listed subject's face (the head, for animals) visible \
in it, as box_2d [ymin, xmin, ymax, xmax] on a 0-1000 grid of that frame. Omit a subject \
whose face is not in the frame. Return every frame, even with no faces.",
        last = n - 1
    )
}

const STRIP_BOX_INSTRUCTIONS: &str = "\
Each image below is the middle of one person's face. For each image, give the box of the \
eye strip, like a carnival mask: from just above the eyebrows down to the middle of the \
nose, and across both eyes including the whole of any glasses' lenses and frames. Give it \
as box_2d [ymin, xmin, ymax, xmax] on a 0-1000 grid of that image. Answer every image, by \
its number.";

/// How open an eye is, level by level; `EyeLevel` names the same levels.
const LEVELS: &str = "\
- 4: wide open. The whole iris shows, with white of the eye above or below it.
- 3: open. Most of the iris shows; the upper lid touches or slightly covers its top.
- 2: narrowed. The lids cover the top and bottom of the iris, but the pupil still shows, \
as in a smile or squinting into the sun.
- 1: slit. Only a sliver of iris or a dark line between the lids; the pupil is hidden.
- 0: shut. The lids meet; no iris and no white of the eye shows.
- null: you cannot see this eye: a reflection, glare, darkness, the angle of the head, or the \
eye is outside the image.";

/// What the levels and compare requests both open with.
fn strip_intro(n: usize) -> String {
    format!(
        "Below are close-ups of the eyes of the people in {n} frames of one burst: \
near-identical shots taken a fraction of a second apart, so the photographer can choose the \
frame where the eyes look best. Each image is a strip across one person's eyes, from the \
brows to the nose, in one frame, captioned with its frame and person id. Its brightness was \
set from the strip itself and the shadows lifted, so lids and lashes behind tinted lenses \
and sunglasses show; it looks brighter than the photograph. Read the eyes through the \
lenses. Never infer an eye from the brows or from another frame: when you cannot make out \
the eye itself, it is null."
    )
}

fn levels_instructions(n: usize) -> String {
    format!(
        "{}

For each image, look at each eye on its own: the eye on the left of the picture and the eye \
on the right of the picture. First note in a few words what of the eye you can see (iris, \
pupil, white, lid line), then give its level:
{LEVELS}

Report every frame and, within it, exactly the persons shown for it.",
        strip_intro(n)
    )
}

fn compare_instructions(n: usize) -> String {
    format!(
        "{}

For each person, compare their eyes across the frames and order the frames from the most \
open eyes to the least open; a frame counts by its less open eye. Look closely: the frames \
are near-identical and the differences are small, a lid a little lower, less of the iris or \
pupil showing. Frames whose eyes look the same go in one group. Put frames where you cannot \
see this person's eyes in `not_visible` instead.",
        strip_intro(n)
    )
}

fn expression_instructions(n: usize) -> String {
    format!(
        "Below are face crops from {n} frames of one burst: near-identical shots taken a \
fraction of a second apart, so the photographer can choose the frame where the faces work. \
Each face is its own image, captioned with its frame and person id; the main subject's \
faces are captioned `(main subject)`.

For each person, once for the whole burst (it does not change between frames), report \
eyewear, judged by how the face looks in these images as the photograph is normally viewed:
- `none`: no glasses.
- `clear`: glasses with clear lenses; the eyes are plainly visible through them.
- `tinted`: lenses that dim the eyes but still let them be seen, including photochromic \
lenses and light sunglasses.
- `dark`: sunglasses or other lenses through which the eyes cannot be made out in the \
photograph as shown.

For every face in every frame, report expression: `good` (natural, engaged, flattering), \
`neutral`, or `awkward` (mid-word, grimace, mid-movement, unflattering).

Then rank all frames by how good the faces look together in the final photo — the most \
flattering expressions, head angles and gaze, the main subject mattering most — best \
first. Whether eyes are open or blinking is judged separately, from close-ups of the eyes: \
it must not drive this ranking or the expression labels. The frames are near-identical, so \
look closely at small differences in smile, mouth, head tilt and gaze; the ranking must \
separate frames even when your labels are the same.

Report eyewear for every person shown, and every frame with, within it, exactly the \
persons shown for it.",
    )
}

fn locate_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "people": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "main": {"type": "BOOLEAN"}
                    },
                    "required": ["id", "main"]
                }
            },
            "frames": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "frame": {"type": "INTEGER"},
                        "faces": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "person": {"type": "INTEGER"},
                                    "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                                },
                                "required": ["person", "box_2d"]
                            }
                        }
                    },
                    "required": ["frame", "faces"]
                }
            }
        },
        "required": ["people", "frames"]
    })
}

fn strip_box_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "boxes": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "image": {"type": "INTEGER"},
                        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                    },
                    "required": ["image", "box_2d"],
                    "propertyOrdering": ["image", "box_2d"]
                }
            }
        },
        "required": ["boxes"]
    })
}

/// The schema of an answer with, per request frame, the `face` object for each face.
fn frames_schema(face: Value) -> Value {
    json!({
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "frame": {"type": "INTEGER"},
                "faces": {"type": "ARRAY", "items": face}
            },
            "required": ["frame", "faces"]
        }
    })
}

/// One eye of the levels answer: what the model sees, then the level. The order is the
/// schema's `propertyOrdering`, because the note is written to be looked at before answering.
fn eye_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "seen": {"type": "STRING", "description": "a few words on what of this eye is visible"},
            "level": {
                "type": "INTEGER",
                "nullable": true,
                "minimum": EyeLevel::Shut as i64,
                "maximum": EyeLevel::Wide as i64
            }
        },
        "required": ["seen", "level"],
        "propertyOrdering": ["seen", "level"]
    })
}

fn levels_schema() -> Value {
    let mut frames = frames_schema(json!({
        "type": "OBJECT",
        "properties": {
            "person": {"type": "INTEGER"},
            "left": eye_schema(),
            "right": eye_schema()
        },
        "required": ["person", "left", "right"],
        "propertyOrdering": ["person", "left", "right"]
    }));
    frames["items"]["propertyOrdering"] = json!(["frame", "faces"]);
    json!({
        "type": "OBJECT",
        "properties": {"frames": frames},
        "required": ["frames"]
    })
}

fn compare_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "people": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "person": {"type": "INTEGER"},
                        "most_open_first": {
                            "type": "ARRAY",
                            "description": "groups of frame numbers",
                            "items": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                        },
                        "not_visible": {"type": "ARRAY", "items": {"type": "INTEGER"}}
                    },
                    "required": ["person", "most_open_first", "not_visible"],
                    "propertyOrdering": ["person", "most_open_first", "not_visible"]
                }
            }
        },
        "required": ["people"]
    })
}

fn expression_schema() -> Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "people": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "person": {"type": "INTEGER"},
                        "eyewear": {"type": "STRING", "enum": EYEWEAR}
                    },
                    "required": ["person", "eyewear"]
                }
            },
            "frames": frames_schema(json!({
                "type": "OBJECT",
                "properties": {
                    "person": {"type": "INTEGER"},
                    "expression": {"type": "STRING", "enum": EXPRESSIONS}
                },
                "required": ["person", "expression"]
            })),
            "ranking": {"type": "ARRAY", "items": {"type": "INTEGER"}}
        },
        "required": ["people", "frames", "ranking"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(id: usize, main: bool, eyewear: Eyewear) -> Person {
        Person {
            id,
            main,
            eyewear,
            eye_groups: 3,
        }
    }

    /// Both eyes at `level`.
    fn level(level: EyeLevel) -> EyePair {
        EyePair {
            left: Some(level),
            right: Some(level),
        }
    }

    const UNSEEN: EyePair = EyePair {
        left: None,
        right: None,
    };

    fn read(person: usize, eyes: EyePair, compared: Compared, expression: Expression) -> FaceRead {
        FaceRead {
            person,
            at: Box2d {
                x0: 0.4,
                y0: 0.4,
                x1: 0.6,
                y1: 0.6,
            },
            eyes,
            compared,
            expression,
        }
    }

    fn member(faces: Vec<FaceRead>, place: usize) -> Option<MemberFaces> {
        Some(MemberFaces {
            faces,
            place,
            of: 4,
        })
    }

    /// Faces in the most open group and good, except for the given persons' levels and a
    /// count of awkward expressions on the first persons.
    fn faces(levels: &[(usize, EyeLevel)], awkward: usize) -> Vec<FaceRead> {
        levels
            .iter()
            .enumerate()
            .map(|(k, &(p, l))| {
                let expression = if k < awkward {
                    Expression::Awkward
                } else {
                    Expression::Good
                };
                read(p, level(l), Compared::Group(0), expression)
            })
            .collect()
    }

    /// Person 1 is the main subject and bare; the others are bare unless said otherwise.
    fn people(others: &[(usize, Eyewear)]) -> People {
        let mut all = vec![person(1, true, Eyewear::None)];
        all.extend(others.iter().map(|&(id, w)| person(id, false, w)));
        People(all)
    }

    fn sharp(_: usize) -> f64 {
        1.0
    }

    fn chosen(people: &People, members: &[Option<MemberFaces>], best: usize) -> usize {
        let cands: Vec<usize> = (0..members.len()).collect();
        choose(&cands, people, members, best, sharp)
    }

    /// Each step of the order decides only when the steps before it tie. Every pair below
    /// is set up so that the step under test and every later step point opposite ways: the
    /// expected winner (listed second) is worse at all of them but the one under test.
    #[test]
    fn choice_follows_the_photographers_order() {
        use EyeLevel::*;
        let best_is_loser = 0;
        let p = people(&[(2, Eyewear::None)]);
        // Frames of the same eyes, differing in the step under test.
        let group = |g| {
            vec![
                read(1, level(Open), Compared::Group(g), Expression::Good),
                read(2, level(Open), Compared::Group(0), Expression::Good),
            ]
        };

        // 1. Level deficits, over the compare groups, awkwardness and ranking (see
        // `level_deficits_are_compared_by_noticeability`).
        let m = vec![
            member(faces(&[(1, Narrowed), (2, Open)], 0), 0),
            member(
                vec![
                    read(1, level(Open), Compared::Group(2), Expression::Awkward),
                    read(2, level(Open), Compared::Group(0), Expression::Good),
                ],
                3,
            ),
        ];
        assert_eq!(chosen(&p, &m, best_is_loser), 1, "level deficits");

        // 2. Compare deficits, over awkwardness and ranking.
        let mut worse = group(0);
        worse[0].expression = Expression::Awkward;
        let m = vec![member(group(2), 0), member(worse, 3)];
        assert_eq!(chosen(&p, &m, best_is_loser), 1, "compare deficits");

        // 3. Fewest awkward expressions, over the ranking.
        let m = vec![
            member(faces(&[(1, Open), (2, Open)], 1), 0),
            member(faces(&[(1, Open), (2, Open)], 0), 3),
        ];
        assert_eq!(chosen(&p, &m, best_is_loser), 1, "fewest awkward");

        // 4. The ranking, over the burst pass's pick.
        let m = vec![
            member(faces(&[(1, Open), (2, Open)], 0), 3),
            member(faces(&[(1, Open), (2, Open)], 0), 1),
        ];
        assert_eq!(chosen(&p, &m, best_is_loser), 1, "the ranking");

        // 5. The burst pass's pick, over sharpness; 6. sharpness.
        let m = vec![
            member(faces(&[(1, Open)], 0), 0),
            member(faces(&[(1, Open)], 0), 0),
        ];
        let p = people(&[]);
        let sharper0 = |k: usize| if k == 0 { 0.9 } else { 0.8 };
        let cands = [0, 1];
        assert_eq!(choose(&cands, &p, &m, 1, sharper0), 1, "the judged best");
        assert_eq!(choose(&cands, &p, &m, 5, sharper0), 0, "sharpness last");
    }

    /// Persons 1 (main, wearing `main`), 2 and 3 bare, 4 clear, 5 tinted, 6 dark.
    fn crowd(main: Eyewear) -> People {
        People(vec![
            person(1, true, main),
            person(2, false, Eyewear::None),
            person(3, false, Eyewear::None),
            person(4, false, Eyewear::Clear),
            person(5, false, Eyewear::Tinted),
            person(6, false, Eyewear::Dark),
        ])
    }

    /// Everyone of `crowd` open and in the most open group, except as `lowered` says.
    fn crowd_faces(lowered: &[(usize, EyeLevel)]) -> Vec<FaceRead> {
        (1..=6)
            .map(|id| {
                let l = lowered
                    .iter()
                    .find(|&&(p, _)| p == id)
                    .map_or(EyeLevel::Open, |&(_, l)| l);
                read(id, level(l), Compared::Group(0), Expression::Good)
            })
            .collect()
    }

    /// Two frames over `crowd(main)`, the one chosen of them. The first lowers the eyes of
    /// `loser` and is better at every later step (good expressions, most open groups, ranked
    /// first, the burst pass's pick); the second lowers those of `winner` and is worse at all
    /// of them, so it wins only on the level deficits it has. The lowered persons of the two
    /// frames differ, so everyone's best is `Open`.
    fn level_duel(
        main: Eyewear,
        loser: &[(usize, EyeLevel)],
        winner: &[(usize, EyeLevel)],
    ) -> usize {
        let mut worse = crowd_faces(winner);
        worse[0].expression = Expression::Awkward;
        for f in &mut worse {
            f.compared = Compared::Group(2);
        }
        let m = vec![member(crowd_faces(loser), 0), member(worse, 3)];
        chosen(&crowd(main), &m, 0)
    }

    /// Level deficits are summed per tier, in level steps, and compared tier by tier: each case
    /// below puts a small deficit on a more noticeable tier against a large one on the less
    /// noticeable ones.
    #[test]
    fn level_deficits_are_compared_by_noticeability() {
        use EyeLevel::*;
        use Eyewear::{Clear, Dark, None as Bare, Tinted};
        // The main subject's one step, bare or behind clear or tinted lenses, loses to two
        // bare others shut.
        for main in [Bare, Clear, Tinted] {
            assert_eq!(
                level_duel(main, &[(1, Narrowed)], &[(2, Shut), (3, Shut)]),
                1,
                "main behind {main:?}"
            );
        }
        // Behind dark lenses the main subject is in the Dark tier: shut, they lose to one
        // step of a bare other, not the other way round.
        assert_eq!(
            level_duel(Dark, &[(2, Narrowed)], &[(1, Shut)]),
            1,
            "dark main over one bare"
        );
        assert_eq!(
            level_duel(Dark, &[(2, Narrowed), (3, Narrowed)], &[(1, Shut)]),
            1,
            "dark main over two bare"
        );
        // An open main and a bare other shut lose to a dark other shut.
        assert_eq!(
            level_duel(Bare, &[(2, Shut)], &[(6, Shut)]),
            1,
            "bare other shut over dark other shut"
        );
        // One bare step loses to a step on each less noticeable tier.
        assert_eq!(
            level_duel(Bare, &[(2, Narrowed)], &[(4, Shut), (5, Shut), (6, Shut)]),
            1,
            "bare over clear+"
        );
        // One clear step loses to a tinted and a dark one; one tinted to two dark ones.
        assert_eq!(
            level_duel(Bare, &[(4, Narrowed)], &[(5, Shut), (6, Shut)]),
            1,
            "clear over tinted+"
        );
        assert_eq!(
            level_duel(Dark, &[(5, Narrowed)], &[(1, Shut), (6, Shut)]),
            1,
            "tinted over dark"
        );
        // Within a tier fewer steps is better, the main subject counting like anyone in Dark.
        assert_eq!(
            level_duel(Dark, &[(1, Shut)], &[(6, Slit)]),
            1,
            "fewer dark steps"
        );
        assert_eq!(
            level_duel(Bare, &[(2, Slit)], &[(3, Narrowed)]),
            1,
            "steps, not eyes"
        );
    }

    /// Two frames over `people(&[(2, none)])`, the first better than the second at every step
    /// after the level deficits; `main_a` and `main_b` are the main subject's face in each
    /// (`None` for no face). Returns the one chosen. Person 2 is open in both.
    fn main_duel(main_a: Option<EyePair>, main_b: Option<EyePair>) -> usize {
        let frame = |main: Option<EyePair>, compared, expression| {
            main.map(|eyes| read(1, eyes, compared, expression))
                .into_iter()
                .chain([read(2, level(EyeLevel::Open), compared, expression)])
                .collect::<Vec<_>>()
        };
        let m = vec![
            member(frame(main_a, Compared::Group(0), Expression::Good), 0),
            member(frame(main_b, Compared::Group(2), Expression::Awkward), 3),
        ];
        chosen(&people(&[(2, Eyewear::None)]), &m, 0)
    }

    /// Eyes nobody could read count as `Narrowed`: below an open face, above a shut one, and an
    /// absent face is as unread as an unseen one.
    #[test]
    fn unread_eyes_rank_between_slit_and_open() {
        use EyeLevel::*;
        for unread in [Some(UNSEEN), None] {
            assert_eq!(
                main_duel(unread, Some(level(Open))),
                1,
                "{unread:?} below open"
            );
            assert_eq!(
                main_duel(Some(level(Shut)), unread),
                1,
                "{unread:?} above shut"
            );
            assert_eq!(
                main_duel(Some(level(Slit)), unread),
                1,
                "{unread:?} above slit"
            );
            assert_eq!(
                main_duel(unread, Some(level(Wide))),
                1,
                "{unread:?} below wide"
            );
        }
        // The same constant, from the other side: Narrowed against unread is a tie that the
        // later steps decide, and the unread frame is the one they favour.
        assert_eq!(main_duel(Some(UNSEEN), Some(level(Narrowed))), 0);
    }

    /// A face counts by its worse visible eye, and by the one eye it has when the other is not
    /// visible.
    #[test]
    fn a_face_counts_by_its_worse_visible_eye() {
        use EyeLevel::*;
        let pair = |left, right| EyePair { left, right };
        assert_eq!(pair(Some(Wide), Some(Slit)).worse(), Some(Slit));
        assert_eq!(pair(Some(Slit), Some(Wide)).worse(), Some(Slit));
        assert_eq!(pair(Some(Open), None).worse(), Some(Open));
        assert_eq!(pair(None, Some(Shut)).worse(), Some(Shut));
        assert_eq!(pair(None, None).worse(), None);
        // Through the rule: a wide-open eye beside a slit is a slit (below unread), and an
        // open eye beside an unseen one is open (above unread).
        assert_eq!(
            main_duel(Some(pair(Some(Wide), Some(Slit))), Some(UNSEEN)),
            1
        );
        assert_eq!(
            main_duel(Some(level(Narrowed)), Some(pair(Some(Open), None))),
            1
        );
    }

    /// A single person over two or three frames, all open (or as said), whose compare group
    /// differs. People are `people(&[])` with the main subject's `eye_groups` set.
    fn compare_members(
        eye_groups: usize,
        frames: &[(Option<Compared>, EyeLevel)],
    ) -> (People, Vec<Option<MemberFaces>>) {
        let mut p = people(&[]);
        p.0[0].eye_groups = eye_groups;
        let m = frames
            .iter()
            .enumerate()
            .map(|(k, &(compared, l))| {
                // Earlier frames are better at the later steps.
                let faces = compared
                    .map(|c| read(1, level(l), c, Expression::Good))
                    .into_iter()
                    .collect();
                member(faces, k)
            })
            .collect();
        (p, m)
    }

    /// The order of a person's frames breaks a tie in the levels, ahead of the later steps, and
    /// never overrides a difference in them.
    #[test]
    fn compare_breaks_a_level_tie_and_never_overrides_a_level_difference() {
        use EyeLevel::*;
        let g = |i| Some(Compared::Group(i));
        // Equal levels: the frame in the more open group wins, though the first is better
        // ranked.
        let (p, m) = compare_members(2, &[(g(1), Open), (g(0), Open)]);
        assert_eq!(chosen(&p, &m, 0), 1, "compare breaks the tie");
        // A level difference wins over the order, even where the order points the other way
        // and the later steps do too.
        let (p, m) = compare_members(2, &[(g(1), Narrowed), (g(0), Open)]);
        assert_eq!(chosen(&p, &m, 0), 1, "the open frame wins");
        let (p, mut m) = compare_members(2, &[(g(0), Narrowed), (g(1), Open)]);
        m[0].as_mut().unwrap().place = 3;
        m[1].as_mut().unwrap().place = 0;
        assert_eq!(
            chosen(&p, &m, 0),
            1,
            "level over compare, though compare favours 0"
        );
    }

    /// A frame where the person is not visible, or absent, counts the middle group: behind the
    /// frames the model found more open and ahead of those it found less.
    #[test]
    fn frames_without_a_visible_person_count_the_middle_group() {
        use EyeLevel::*;
        let g = |i| Some(Compared::Group(i));
        // Three groups: middle is group 1. Levels are all Narrowed, so an absent face (unread,
        // Narrowed) ties with the read ones and only the compare step separates them.
        let (p, m) = compare_members(
            3,
            &[
                (g(2), Narrowed),
                (Some(Compared::NotVisible), Narrowed),
                (g(0), Narrowed),
                (None, Narrowed),
            ],
        );
        let pick = |a, b| choose(&[a, b], &p, &m, 0, sharp);
        assert_eq!(pick(0, 1), 1, "not visible over the least open group");
        assert_eq!(pick(1, 2), 2, "the most open group over not visible");
        assert_eq!(pick(0, 3), 3, "absent over the least open group");
        assert_eq!(pick(2, 3), 2, "the most open group over absent");
        // Not visible and absent are the same middle group: the ranking decides, in either order.
        assert_eq!(pick(1, 3), 1);
        assert_eq!(pick(3, 1), 1);
    }

    #[test]
    fn the_middle_group_rounds_down() {
        assert_eq!([0, 1, 2, 3, 4, 5].map(middle_group), [0, 0, 1, 1, 2, 2]);
    }

    /// Compare deficits are summed per tier like the level deficits and never across tiers.
    #[test]
    fn compare_deficits_are_compared_by_noticeability() {
        let p = crowd(Eyewear::None);
        let frame = |main: usize, dark: usize, place| {
            let faces = (1..=6)
                .map(|id| {
                    let g = match id {
                        1 => main,
                        6 => dark,
                        _ => 0,
                    };
                    read(
                        id,
                        level(EyeLevel::Open),
                        Compared::Group(g),
                        Expression::Good,
                    )
                })
                .collect();
            member(faces, place)
        };
        // The main subject one group behind loses to the dark person two behind, though the sum
        // is smaller, however the later steps favour the first.
        let m = vec![frame(1, 0, 0), frame(0, 2, 3)];
        assert_eq!(chosen(&p, &m, 0), 1);
        // Within the Dark tier fewer groups is better.
        let m = vec![frame(0, 2, 0), frame(0, 1, 3)];
        assert_eq!(chosen(&p, &m, 0), 1);
    }

    #[test]
    fn eyes_at_each_level_are_ordered() {
        use EyeLevel::*;
        let mut sorted = EyeLevel::ALL;
        sorted.sort();
        assert_eq!(sorted, [Shut, Slit, Narrowed, Open, Wide]);
        assert_eq!(EyeLevel::ALL.map(|l| l as i64), [0, 1, 2, 3, 4]);
        assert!(Slit < UNREAD_LEVEL && UNREAD_LEVEL < Open);
        assert_eq!(Wide.steps_above(Shut), 4);
    }

    /// Every (main, eyewear) pair lands on a tier below `COUNT`, and every tier is reached by
    /// some pair: a tier declared after `Dark` would be reachable yet outside `COUNT`, and a
    /// tier nobody reaches would be a dead counter.
    #[test]
    fn every_tier_is_reachable_and_counted() {
        let tiers: std::collections::BTreeSet<usize> = [true, false]
            .into_iter()
            .flat_map(|main| {
                [
                    Eyewear::None,
                    Eyewear::Clear,
                    Eyewear::Tinted,
                    Eyewear::Dark,
                ]
                .map(|w| Noticeability::of(person(1, main, w)) as usize)
            })
            .collect();
        assert_eq!(
            tiers.into_iter().collect::<Vec<_>>(),
            (0..Noticeability::COUNT).collect::<Vec<_>>()
        );
    }

    fn burst_faces(people: People, members: Vec<Vec<FaceRead>>) -> BurstFaces {
        BurstFaces {
            winner: 0,
            people,
            members: members
                .into_iter()
                .enumerate()
                .map(|(place, faces)| member(faces, place))
                .collect(),
        }
    }

    /// The review renders crop around the located faces: a read frame lists its faces' boxes
    /// (an empty list when none was found), a frame the pass did not read lists nothing.
    #[test]
    fn located_lists_the_faces_of_read_frames_only() {
        let mut bf = burst_faces(
            people(&[]),
            vec![
                vec![read(
                    1,
                    EyePair {
                        left: None,
                        right: None,
                    },
                    Compared::NotVisible,
                    Expression::Good,
                )],
                vec![],
            ],
        );
        bf.members.push(None);
        let boxes = bf.located(0).unwrap();
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].person, 1);
        assert_eq!(
            boxes[0].at,
            Box2d {
                x0: 0.4,
                y0: 0.4,
                x1: 0.6,
                y1: 0.6
            }
        );
        assert_eq!(bf.located(1), Some(vec![]));
        assert_eq!(bf.located(2), None);
    }

    #[test]
    fn summary_says_what_decided() {
        use EyeLevel::*;
        let p = People(vec![
            person(1, true, Eyewear::None),
            person(2, false, Eyewear::Dark),
            person(3, false, Eyewear::Clear),
        ]);
        let all_open = faces(&[(1, Open), (2, Open), (3, Open)], 0);
        let b = burst_faces(
            p.clone(),
            vec![
                faces(&[(1, Open), (2, Shut), (3, Open)], 0),
                all_open.clone(),
            ],
        );
        assert_eq!(
            b.summary(0).unwrap(),
            "faces #1 of 4: 3 faces, eyes: person 2 3 steps below their best (behind dark lenses)"
        );
        assert_eq!(
            b.summary(1).unwrap(),
            "faces #2 of 4: 3 faces, eyes at their best"
        );

        // The main subject behind dark lenses, one step down; and a compare group and an
        // awkward expression.
        let dark_main = People(vec![
            person(1, true, Eyewear::Dark),
            person(2, false, Eyewear::None),
        ]);
        let mut behind = faces(&[(1, Narrowed), (2, Open)], 1);
        behind[1].compared = Compared::Group(1);
        let b = burst_faces(dark_main, vec![behind, faces(&[(1, Open), (2, Open)], 0)]);
        assert_eq!(
            b.summary(0).unwrap(),
            "faces #1 of 4: 2 faces, eyes: main 1 step below their best (behind dark lenses), \
compare: person 2 1 group from the most open, 1 awkward"
        );

        // Unread eyes are said so: alone, and with the steps they were counted at.
        let mut unread = faces(&[(1, Open), (2, Open), (3, Open)], 0);
        unread[0].eyes = UNSEEN;
        unread[2].eyes = UNSEEN;
        let b = burst_faces(p, vec![unread, all_open]);
        assert_eq!(
            b.summary(0).unwrap(),
            "faces #1 of 4: 3 faces, eyes: main unread, counted 1 step below their best (bare); \
person 3 unread, counted 1 step below their best (behind clear lenses)"
        );

        // A person with no face in the frame is unread there, and counts as `Narrowed` in
        // everyone's best, so the frame that shows them slit is one step below it.
        let b = burst_faces(
            People(vec![
                person(1, true, Eyewear::None),
                person(3, false, Eyewear::Clear),
            ]),
            vec![faces(&[(1, Open)], 0), faces(&[(1, Open), (3, Slit)], 0)],
        );
        assert_eq!(
            b.summary(0).unwrap(),
            "faces #1 of 4: 1 face, eyes: person 3 unread (behind clear lenses), \
compare: person 3 not visible, counted 1 group from the most open"
        );
        assert_eq!(
            b.summary(1).unwrap(),
            "faces #2 of 4: 2 faces, eyes: person 3 1 step below their best (behind clear lenses)"
        );
    }

    /// A face as (person id, box_2d).
    type FaceSpec = (usize, [i64; 4]);

    fn raw_locate(people: &[(usize, bool)], frames: &[(usize, &[FaceSpec])]) -> RawLocate {
        RawLocate {
            people: people
                .iter()
                .map(|&(id, main)| RawPerson { id, main })
                .collect(),
            frames: frames
                .iter()
                .map(|(f, faces)| RawFrameFaces {
                    frame: *f,
                    faces: faces
                        .iter()
                        .map(|&(person, box_2d)| RawFace { person, box_2d })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn locate_orders_main_first_and_faces_left_to_right() {
        let l = parse_locate(
            raw_locate(
                &[(7, false), (3, true)],
                &[
                    (0, &[(7, [100, 600, 200, 700]), (3, [100, 100, 200, 200])]),
                    (1, &[]),
                ],
            ),
            2,
        )
        .unwrap();
        assert_eq!(
            l.people,
            Some(Subjects {
                main: 3,
                others: vec![7]
            })
        );
        let order: Vec<usize> = l.frames[0].iter().map(|(p, _)| *p).collect();
        assert_eq!(order, [3, 7]);
        assert!(l.frames[1].is_empty());
    }

    /// What `f` logged at any level up to debug, with `f`'s result.
    fn logged<T>(f: impl FnOnce() -> T) -> (T, String) {
        #[derive(Clone, Default)]
        struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Sink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl tracing_subscriber::fmt::MakeWriter<'_> for Sink {
            type Writer = Sink;
            fn make_writer(&self) -> Sink {
                self.clone()
            }
        }
        let sink = Sink::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(sink.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::DEBUG)
            .finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        let log = String::from_utf8(sink.0.lock().unwrap().clone()).unwrap();
        (out, log)
    }

    /// A face under an id that was never listed is the model contradicting itself: the answer
    /// is invalid, so `gemini::ask` asks again (twice at most) and the pass falls back to the
    /// burst pass's winner. The same person twice in a frame is as invalid.
    #[test]
    fn locate_rejects_faces_of_unlisted_persons_and_repeated_persons() {
        let unlisted = raw_locate(&[(1, true)], &[(0, &[(2, [100, 100, 200, 200])])]);
        let err = parse_locate(unlisted, 1).unwrap_err().to_string();
        assert!(err.contains("person 2, who is not listed"), "{err}");

        let twice = raw_locate(
            &[(1, true)],
            &[(0, &[(1, [100, 100, 200, 200]), (1, [100, 500, 200, 600])])],
        );
        let err = parse_locate(twice, 1).unwrap_err().to_string();
        assert!(err.contains("two faces of person 1"), "{err}");
    }

    /// People listed beyond `PEOPLE_MAX` are not read, and the pass says so, naming the
    /// main subject when that is who was cut; their faces are dropped, loudly.
    #[test]
    fn locate_warns_about_people_beyond_the_maximum() {
        let listed: Vec<(usize, bool)> = (1..=PEOPLE_MAX + 1)
            .map(|id| (id, id == PEOPLE_MAX + 1))
            .collect();
        let cut = PEOPLE_MAX + 1;
        let (l, log) = logged(|| {
            parse_locate(
                raw_locate(
                    &listed,
                    &[(0, &[(1, [100, 100, 200, 200]), (cut, [100, 500, 200, 600])])],
                ),
                1,
            )
            .unwrap()
        });
        let people: Vec<(usize, bool)> = l.people.as_ref().unwrap().ordered().collect();
        assert_eq!(people.len(), PEOPLE_MAX);
        assert_eq!(people[0], (1, true), "main falls to the first listed");
        assert_eq!(l.frames[0].len(), 1, "the cut person's face is dropped");
        assert!(log.contains("person=7 main=true"), "{log}");
        assert!(log.contains("dropping the face of a person"), "{log}");
        assert!(
            log.contains("no main subject among the people read") && log.contains("main_cut=true"),
            "{log}"
        );
    }

    /// "Exactly one main" is the answer's contract. When it is broken the first marked main
    /// (else the first listed) wins, and the pass says who moved.
    #[test]
    fn locate_says_when_the_main_subject_is_not_exactly_one() {
        let resolved = |listed: &[(usize, bool)]| {
            let first: [FaceSpec; 1] = [(listed[0].0, [100, 100, 200, 200])];
            logged(|| parse_locate(raw_locate(listed, &[(0, &first[..])]), 1).unwrap())
        };

        let (l, log) = resolved(&[(1, false), (2, true), (3, true), (4, false)]);
        assert_eq!(
            l.people,
            Some(Subjects {
                main: 2,
                others: vec![3, 1, 4]
            })
        );
        assert!(log.contains("several people marked main"), "{log}");
        assert!(log.contains("main=2 demoted=[3]"), "{log}");

        let (l, log) = resolved(&[(5, false), (6, false)]);
        assert_eq!(
            l.people,
            Some(Subjects {
                main: 5,
                others: vec![6]
            })
        );
        assert!(log.contains("promoted=5 main_cut=false"), "{log}");

        let (_, log) = resolved(&[(1, true), (2, false)]);
        assert_eq!(log, "", "one main: nothing to say");
    }

    #[test]
    fn locate_rejects_bad_boxes_and_missing_frames() {
        let bad = raw_locate(&[(1, true)], &[(0, &[(1, [500, 100, 400, 200])])]);
        assert!(parse_locate(bad, 1).is_err());
        let missing = raw_locate(&[(1, true)], &[(0, &[])]);
        assert!(parse_locate(missing, 2).is_err());
    }

    fn box_at(x0: f64) -> Box2d {
        Box2d {
            x0,
            y0: 0.1,
            x1: x0 + 0.1,
            y1: 0.2,
        }
    }

    /// People 1 (main) and 2 in frame 0, person 1 alone in frame 1, nobody in frame 2, and
    /// person 3 listed but never seen.
    fn located() -> Located {
        Located {
            people: Some(Subjects {
                main: 1,
                others: vec![2, 3],
            }),
            frames: vec![
                vec![(1, box_at(0.1)), (2, box_at(0.5))],
                vec![(1, box_at(0.1))],
                vec![],
            ],
        }
    }

    fn eyewear(person: usize, eyewear: Eyewear) -> RawEyewear {
        RawEyewear { person, eyewear }
    }

    fn levels_in(
        frame: usize,
        faces: &[(usize, Option<EyeLevel>, Option<EyeLevel>)],
    ) -> RawFrame<RawLevel> {
        RawFrame {
            frame,
            faces: faces
                .iter()
                .map(|&(person, left, right)| RawLevel {
                    person,
                    left: RawEye { level: left },
                    right: RawEye { level: right },
                })
                .collect(),
        }
    }

    fn raw_levels() -> RawLevelsRead {
        use EyeLevel::*;
        RawLevelsRead {
            frames: vec![
                levels_in(0, &[(2, None, None), (1, Some(Open), Some(Slit))]),
                levels_in(1, &[(1, Some(Shut), None)]),
                levels_in(2, &[]),
            ],
        }
    }

    /// Person 1 is the main subject in `located()`. When only others have faces, the pass
    /// warns, with the burst's name from the enclosing span.
    #[test]
    fn a_main_subject_nobody_saw_is_reported() {
        let (_, log) = logged(|| located().warn_if_main_unseen());
        assert_eq!(log, "", "main seen: nothing to say");
        let mut l = located();
        l.frames[0].remove(0);
        l.frames[1].clear();
        assert_eq!(l.seen(), [(2, false)]);
        assert_eq!(l.main_seen(), None);
        let (_, log) = logged(|| l.warn_if_main_unseen());
        assert!(
            log.contains("main subject has no face in any frame"),
            "{log}"
        );
    }

    #[test]
    fn levels_answer_is_arranged_by_face() {
        use EyeLevel::*;
        let eyes = |left, right| EyePair { left, right };
        assert_eq!(
            parse_levels(raw_levels(), &located()).unwrap(),
            [
                vec![eyes(Some(Open), Some(Slit)), eyes(None, None)],
                vec![eyes(Some(Shut), None)],
                vec![]
            ],
            "in the order the faces were attached"
        );
    }

    /// The model's integers become levels at the boundary: five are valid, an eye it cannot see
    /// is null, and anything else, or no answer for an eye, is not the requested JSON.
    #[test]
    fn levels_are_parsed_at_the_boundary() {
        let face = |left: &str, right: &str| {
            format!(
                r#"{{"frames": [{{"frame": 0, "faces": [{{"person": 1,
                "left": {left}, "right": {right}}}]}}]}}"#
            )
        };
        let eye = |level: &str| format!(r#"{{"seen": "iris", "level": {level}}}"#);
        let parsed: RawLevelsRead = from_json(&face(&eye("4"), &eye("null"))).unwrap();
        let RawLevel { left, right, .. } = &parsed.frames[0].faces[0];
        assert_eq!((left.level, right.level), (Some(EyeLevel::Wide), None));
        for bad in ["5", "-1", "2.5", r#""open""#] {
            assert!(
                from_json::<RawLevelsRead>(&face(&eye(bad), &eye("3"))).is_err(),
                "{bad}"
            );
        }
        let without_level = r#"{"seen": "iris"}"#;
        assert!(
            from_json::<RawLevelsRead>(&face(without_level, &eye("3"))).is_err(),
            "a missing level is not null"
        );
    }

    #[test]
    fn levels_answer_must_cover_exactly_the_cropped_persons() {
        use EyeLevel::*;
        let mut one_missing = raw_levels();
        one_missing.frames[0].faces.pop();
        assert!(parse_levels(one_missing, &located()).is_err());

        let mut stranger = raw_levels();
        stranger.frames[1].faces[0].person = 2;
        assert!(parse_levels(stranger, &located()).is_err());

        let mut frame_missing = raw_levels();
        frame_missing.frames.remove(1);
        assert!(parse_levels(frame_missing, &located()).is_err());

        let mut frame_twice = raw_levels();
        frame_twice
            .frames
            .push(levels_in(1, &[(1, Some(Open), Some(Open))]));
        assert!(parse_levels(frame_twice, &located()).is_err());

        let mut out_of_range = raw_levels();
        out_of_range.frames.push(levels_in(3, &[]));
        assert!(parse_levels(out_of_range, &located()).is_err());
    }

    /// The answer that failed DSC_2815's burst: a face reported for a frame with nothing
    /// cropped. That entry is dropped; the frame reads as faceless.
    #[test]
    fn faces_reported_for_an_uncropped_frame_are_ignored() {
        let mut raw = raw_levels();
        raw.frames[2] = levels_in(2, &[(1, Some(EyeLevel::Open), Some(EyeLevel::Open))]);
        assert_eq!(parse_levels(raw, &located()).unwrap()[2], []);
    }

    fn raw_strip_boxes(images: &[(usize, [i64; 4])]) -> RawStripBoxes {
        RawStripBoxes {
            boxes: images
                .iter()
                .map(|&(image, box_2d)| RawStripBox { image, box_2d })
                .collect(),
        }
    }

    /// `located()` has three faces: images 0 and 1 in frame 0, image 2 in frame 1.
    fn good_strip_boxes() -> Vec<(usize, [i64; 4])> {
        vec![
            (2, [100, 50, 900, 950]),
            (0, [200, 100, 800, 900]),
            (1, [250, 0, 750, 1000]),
        ]
    }

    #[test]
    fn strip_boxes_are_arranged_by_face_in_fractions_of_the_band() {
        let boxes = parse_strip_boxes(raw_strip_boxes(&good_strip_boxes()), &located()).unwrap();
        let grid = |g| BandBox::from_grid(g).unwrap();
        assert_eq!(
            boxes,
            [
                vec![grid([200, 100, 800, 900]), grid([250, 0, 750, 1000])],
                vec![grid([100, 50, 900, 950])],
                vec![]
            ]
        );
    }

    #[test]
    fn strip_boxes_must_answer_every_image_once_inside_the_band() {
        let ok = good_strip_boxes();
        let parse = |images: &[(usize, [i64; 4])]| {
            parse_strip_boxes(raw_strip_boxes(images), &located()).is_err()
        };
        assert!(parse(&ok[..2]), "image 1 missing");
        assert!(
            parse(&[ok.clone(), vec![(1, [0, 0, 500, 500])]].concat()),
            "image 1 twice"
        );
        assert!(
            parse(&[ok.clone(), vec![(3, [0, 0, 500, 500])]].concat()),
            "image 3 does not exist"
        );
        for bad in [
            [0, 0, 1001, 500],
            [-1, 0, 500, 500],
            [0, 0, 500, 1200],
            [500, 0, 400, 500],
            [0, 500, 500, 500],
            [100, 100, 100, 200],
        ] {
            let mut images = ok.clone();
            images[0].1 = bad;
            assert!(parse(&images), "{bad:?}");
        }
    }

    fn order(person: usize, groups: &[&[usize]], not_visible: &[usize]) -> RawPersonOrder {
        RawPersonOrder {
            person,
            most_open_first: groups.iter().map(|g| g.to_vec()).collect(),
            not_visible: not_visible.to_vec(),
        }
    }

    /// Person 1 has faces in frames 0 and 1, person 2 in frame 0 only.
    fn raw_compare() -> RawCompareRead {
        RawCompareRead {
            people: vec![order(2, &[&[0]], &[]), order(1, &[&[1], &[0]], &[])],
        }
    }

    #[test]
    fn compare_answer_is_arranged_by_face() {
        let c = parse_compare(raw_compare(), &located()).unwrap();
        assert_eq!(c.groups, [2, 1], "persons 1 and 2, main first");
        assert_eq!(
            c.compared,
            [
                vec![Compared::Group(1), Compared::Group(0)],
                vec![Compared::Group(0)],
                vec![]
            ]
        );
        let hidden = RawCompareRead {
            people: vec![order(2, &[&[0]], &[]), order(1, &[&[0]], &[1])],
        };
        let c = parse_compare(hidden, &located()).unwrap();
        assert_eq!(c.groups, [1, 1]);
        assert_eq!(c.compared[1], [Compared::NotVisible]);
        let alike = RawCompareRead {
            people: vec![order(2, &[&[0]], &[]), order(1, &[&[0, 1]], &[])],
        };
        let c = parse_compare(alike, &located()).unwrap();
        assert_eq!(c.compared[1], [Compared::Group(0)], "frames of one group");
    }

    /// The prompt asks for the frames where a person's eyes cannot be seen, so the model may
    /// list frames in which they are absent: accepted and ignored, unlike a group there.
    #[test]
    fn a_frame_without_the_person_may_be_listed_as_not_visible() {
        // Person 2 has a face in frame 0 only; frames 1 and 2 are absent for them.
        let listed = RawCompareRead {
            people: vec![order(2, &[&[0]], &[1, 2]), order(1, &[&[1], &[0]], &[])],
        };
        let c = parse_compare(listed, &located()).unwrap();
        assert_eq!(c, parse_compare(raw_compare(), &located()).unwrap());
        let grouped = RawCompareRead {
            people: vec![order(2, &[&[0], &[1]], &[]), order(1, &[&[1], &[0]], &[])],
        };
        assert!(parse_compare(grouped, &located()).is_err());
    }

    #[test]
    fn compare_answer_must_place_each_frame_of_each_person_once() {
        let fails = |people: Vec<RawPersonOrder>| {
            parse_compare(RawCompareRead { people }, &located()).is_err()
        };
        let p2 = || order(2, &[&[0]], &[]);
        assert!(
            fails(vec![order(1, &[&[1], &[0]], &[])]),
            "person 2 missing"
        );
        assert!(
            fails(vec![
                p2(),
                order(1, &[&[1], &[0]], &[]),
                order(3, &[&[0]], &[])
            ]),
            "person 3 has no face"
        );
        assert!(
            fails(vec![p2(), p2(), order(1, &[&[1], &[0]], &[])]),
            "person 2 twice"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0], &[0, 1]], &[])]),
            "frame 0 in two groups"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0], &[1]], &[1])]),
            "frame 1 in a group and not visible"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0], &[1]], &[0])]),
            "frame 0 in a group and not visible"
        );
        assert!(
            fails(vec![
                order(2, &[&[0], &[1]], &[]),
                order(1, &[&[0], &[1]], &[])
            ]),
            "person 2 has no face in frame 1"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0], &[1], &[2]], &[])]),
            "nobody has a face in frame 2"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0], &[1]], &[3])]),
            "frame 3 does not exist"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0]], &[])]),
            "frame 1 not placed"
        );
        assert!(
            fails(vec![p2(), order(1, &[&[0], &[], &[1]], &[])]),
            "an empty group"
        );
    }

    #[test]
    fn expression_answer_gives_each_seen_person_their_eyewear_once() {
        let r = parse_expression(raw_expression(vec![0, 1, 2]), &located()).unwrap();
        assert_eq!(
            r.eyewear,
            [Eyewear::None, Eyewear::Dark],
            "main first, and person 3, who has no face, left out"
        );
        let people = People::new(&located(), r.eyewear, vec![2, 1]);
        assert_eq!(
            people,
            People(vec![
                Person {
                    eye_groups: 2,
                    ..person(1, true, Eyewear::None)
                },
                Person {
                    eye_groups: 1,
                    ..person(2, false, Eyewear::Dark)
                }
            ]),
        );
    }

    #[test]
    fn expression_answer_missing_or_repeating_a_person_is_invalid() {
        let mut missing = raw_expression(vec![0, 1, 2]);
        missing.people.remove(0);
        assert!(
            parse_expression(missing, &located()).is_err(),
            "no eyewear for 2"
        );

        let mut twice = raw_expression(vec![0, 1, 2]);
        twice.people.push(eyewear(2, Eyewear::Clear));
        assert!(
            parse_expression(twice, &located()).is_err(),
            "two eyewear for 2"
        );

        let mut instead = raw_expression(vec![0, 1, 2]);
        instead.people[1] = eyewear(2, Eyewear::Clear);
        assert!(
            parse_expression(instead, &located()).is_err(),
            "2 twice, and none for 1"
        );

        let mut unknown = raw_expression(vec![0, 1, 2]);
        unknown.people.push(eyewear(3, Eyewear::Clear));
        assert!(
            parse_expression(unknown, &located()).is_err(),
            "person 3 was never shown"
        );
    }

    fn expression_in(frame: usize, faces: &[(usize, Expression)]) -> RawFrame<RawExpression> {
        RawFrame {
            frame,
            faces: faces
                .iter()
                .map(|&(person, expression)| RawExpression { person, expression })
                .collect(),
        }
    }

    fn raw_expression(ranking: Vec<usize>) -> RawExpressionRead {
        RawExpressionRead {
            people: vec![eyewear(2, Eyewear::Dark), eyewear(1, Eyewear::None)],
            frames: vec![
                expression_in(0, &[(2, Expression::Awkward), (1, Expression::Good)]),
                expression_in(1, &[(1, Expression::Neutral)]),
                expression_in(2, &[]),
            ],
            ranking,
        }
    }

    #[test]
    fn expression_answer_is_arranged_by_face_and_ranks_every_frame_once() {
        let r = parse_expression(raw_expression(vec![2, 0, 1]), &located()).unwrap();
        assert_eq!(
            r.expressions,
            [
                vec![Expression::Good, Expression::Awkward],
                vec![Expression::Neutral],
                vec![]
            ]
        );
        assert_eq!(r.place, [1, 2, 0]);
    }

    #[test]
    fn expression_answer_must_rank_every_frame_once_and_cover_the_cropped_persons() {
        let unranked = raw_expression(vec![0, 1]);
        assert!(parse_expression(unranked, &located()).is_err());
        let twice = raw_expression(vec![0, 1, 1, 2]);
        assert!(parse_expression(twice, &located()).is_err());
        let out_of_range = raw_expression(vec![0, 1, 2, 3]);
        assert!(parse_expression(out_of_range, &located()).is_err());

        let mut missing_person = raw_expression(vec![0, 1, 2]);
        missing_person.frames[0].faces.pop();
        assert!(parse_expression(missing_person, &located()).is_err());
        let mut missing_frame = raw_expression(vec![0, 1, 2]);
        missing_frame.frames.remove(0);
        assert!(parse_expression(missing_frame, &located()).is_err());
    }

    #[test]
    fn faces_pair_the_reads_per_person() {
        use EyeLevel::*;
        let levels = vec![vec![level(Open), UNSEEN], vec![level(Shut)], vec![]];
        let compared = vec![
            vec![Compared::Group(1), Compared::NotVisible],
            vec![Compared::Group(0)],
            vec![],
        ];
        let expressions = vec![
            vec![Expression::Good, Expression::Awkward],
            vec![Expression::Neutral],
            vec![],
        ];
        let f = faces_read(&located(), &levels, &compared, &expressions);
        // Each read carries the box its face was located at.
        let at = |read: FaceRead, x0: f64| FaceRead {
            at: box_at(x0),
            ..read
        };
        assert_eq!(
            f[0],
            [
                at(
                    read(1, level(Open), Compared::Group(1), Expression::Good),
                    0.1
                ),
                at(
                    read(2, UNSEEN, Compared::NotVisible, Expression::Awkward),
                    0.5
                )
            ]
        );
        assert_eq!(
            f[1],
            [at(
                read(1, level(Shut), Compared::Group(0), Expression::Neutral),
                0.1
            )]
        );
        assert!(f[2].is_empty());
    }

    /// One debug line per face read, naming the frame, the person, both eyes and the group, for
    /// scoring a run against hand labels.
    #[test]
    fn each_face_read_is_logged_for_scoring() {
        let face = EyePair {
            left: Some(EyeLevel::Wide),
            right: None,
        };
        let (_, log) = logged(|| {
            log_read(
                "DSC_0001.NEF",
                &read(2, face, Compared::Group(1), Expression::Good),
            );
            log_read(
                "DSC_0002.NEF",
                &read(1, UNSEEN, Compared::NotVisible, Expression::Good),
            );
        });
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2, "{log}");
        assert!(lines[0].contains("DEBUG"), "{log}");
        assert!(
            lines[0].contains(
                "frame=DSC_0001.NEF person=2 left=wide open right=not visible compare=group 1"
            ),
            "{log}"
        );
        assert!(
            lines[1].contains(
                "frame=DSC_0002.NEF person=1 left=not visible right=not visible compare=not visible"
            ),
            "{log}"
        );
    }

    /// The image parts of `req`, as sent.
    fn images(req: &gemini::Request) -> Vec<Value> {
        req.body["contents"][0]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p.get("inline_data").is_some())
            .cloned()
            .collect()
    }

    /// The captions of `req`, as sent.
    fn captions(req: &gemini::Request) -> Vec<String> {
        req.body["contents"][0]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|p| p["text"].as_str())
            .skip(1)
            .map(str::to_owned)
            .collect()
    }

    /// The captions and images of the four reads for the `located()` faces: each request's images
    /// are the bytes of the files it is given, in the order of the faces; captions number the
    /// images (strip boxes) or name frame and person (levels, compare, expression); only the
    /// expression request marks the main subject, and only the strip request has no caption for
    /// the frame without faces.
    #[test]
    fn each_read_captions_and_sends_the_files_it_is_given() {
        let dir = tempfile::tempdir().unwrap();
        let l = located();
        let cfg = crate::config::Config::load(None).unwrap().judge;
        let files: Vec<Vec<PathBuf>> = l
            .frames
            .iter()
            .enumerate()
            .map(|(k, faces)| {
                (0..faces.len())
                    .map(|i| {
                        let path = dir.path().join(format!("{k}.{i}.jpg"));
                        std::fs::write(&path, [k as u8, i as u8, 7]).unwrap();
                        path
                    })
                    .collect()
            })
            .collect();
        let sent: Vec<Value> = [[0u8, 0, 7], [0, 1, 7], [1, 0, 7]]
            .iter()
            .map(|b| serde_json::to_value(gemini::Part::jpeg(b)).unwrap())
            .collect();
        let req = |read| read_request(read, &l, &files, &cfg).unwrap();

        let strip = req(Read::StripBox);
        assert_eq!(captions(&strip), ["Image 0:", "Image 1:", "Image 2:"]);
        assert_eq!(images(&strip), sent);
        assert!(
            strip.body["contents"][0]["parts"][0]["text"]
                .as_str()
                .unwrap()
                .contains("like a carnival mask")
        );

        let by_frame = [
            "Frame 0, person 1:",
            "Frame 0, person 2:",
            "Frame 1, person 1:",
            "Frame 2: no faces",
        ];
        for read in [Read::Levels, Read::Compare] {
            assert_eq!(captions(&req(read)), by_frame, "{read:?}");
            assert_eq!(images(&req(read)), sent, "{read:?}");
        }

        let expression = req(Read::Expression);
        assert_eq!(
            captions(&expression),
            [
                "Frame 0, person 1 (main subject)",
                "Frame 0, person 2",
                "Frame 1, person 1 (main subject)",
                "Frame 2: no faces",
            ]
        );
        assert_eq!(images(&expression), sent);
        let text = |read: Read| read.instructions(3);
        assert!(text(Read::Expression).contains("the main subject mattering most"));
        for read in [Read::StripBox, Read::Levels, Read::Compare] {
            assert!(!text(read).contains("main subject"), "{read:?}");
        }
    }

    /// The levels and compare requests open alike, and each adds only its own question: the
    /// levels prompt carries the five anchored levels, the compare prompt orders frames by the
    /// less open eye. Neither asks about eyewear, and the levels the prompt lists are the levels
    /// parsed.
    #[test]
    fn the_strip_prompts_ask_one_thing_each() {
        let (levels, compare) = (levels_instructions(4), compare_instructions(4));
        let intro = strip_intro(4);
        assert!(intro.starts_with("Below are close-ups of the eyes of the people in 4 frames"));
        assert!(levels.starts_with(&intro) && compare.starts_with(&intro));
        assert!(levels.contains(LEVELS) && !compare.contains(LEVELS));
        assert!(compare.contains("a frame counts by its less open eye"));
        assert!(compare.contains("`not_visible`") && !levels.contains("not_visible"));
        assert!(!levels.contains("eyewear") && !compare.contains("eyewear"));
        for l in EyeLevel::ALL {
            let line = format!("- {}: {}.", l as i64, l.describe());
            assert!(LEVELS.contains(&line), "{line}");
        }
        assert!(LEVELS.contains("- null: you cannot see this eye"));
    }

    #[test]
    fn the_levels_schema_has_the_note_before_the_level() {
        let eye = &levels_schema()["properties"]["frames"]["items"]["properties"]["faces"]["items"]
            ["properties"]["left"];
        assert_eq!(eye["propertyOrdering"], json!(["seen", "level"]));
        assert_eq!(eye["properties"]["level"]["nullable"], json!(true));
        assert_eq!(eye["properties"]["level"]["minimum"], json!(0));
        assert_eq!(eye["properties"]["level"]["maximum"], json!(4));
        assert_eq!(
            levels_schema()["properties"]["frames"]["items"]["propertyOrdering"],
            json!(["frame", "faces"])
        );
    }

    /// Eyewear is judged on the natural crops, so only the expression request asks for it,
    /// in prompt and schema.
    #[test]
    fn only_the_expression_request_asks_for_eyewear() {
        assert!(expression_instructions(3).contains("eyewear"));
        assert_eq!(
            expression_schema()["properties"]["people"]["items"]["properties"]["eyewear"]["enum"],
            json!(EYEWEAR)
        );
    }

    /// The expression request is the one earlier runs asked, so their cached answers are found:
    /// its schema is pinned here as it was written out for them.
    #[test]
    fn the_expression_schema_is_unchanged() {
        let face = json!({
            "type": "OBJECT",
            "properties": {
                "person": {"type": "INTEGER"},
                "expression": {"type": "STRING", "enum": ["awkward", "neutral", "good"]}
            },
            "required": ["person", "expression"]
        });
        assert_eq!(
            expression_schema()["properties"]["frames"],
            json!({
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "frame": {"type": "INTEGER"},
                        "faces": {"type": "ARRAY", "items": face}
                    },
                    "required": ["frame", "faces"]
                }
            })
        );
    }

    #[test]
    fn schema_enums_match_serde_names() {
        for e in EYEWEAR {
            serde_json::from_value::<Eyewear>(json!(e)).unwrap();
        }
        for e in EXPRESSIONS {
            serde_json::from_value::<Expression>(json!(e)).unwrap();
        }
    }
}
