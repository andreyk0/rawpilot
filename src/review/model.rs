//! What the review session is: the shots and their renders, the filter, where the
//! photographer is, and what the server is waiting for. Every page is drawn from it and every
//! command is applied to it, so the page cannot hold a state the server does not.
//!
//! The model does no disk or network work itself. Writes (a look, a rating) go through
//! [`Disk`], and the renders arrive from the background worker (`work`) through
//! [`Model::next_batch`] and [`Model::batch_done`].
//!
//! # Stages
//!
//! The working view ([`Stage::Working`]) exists as soon as one shot the filter admits has
//! all its options rendered; the shots still to render are not in the strip yet and join it
//! as the worker finishes them. Only while the filter admits shots and none is rendered does
//! the page show the progress page ([`Stage::Waiting`]). The worker renders the shots
//! nearest the photographer first, and what it is doing is [`Model::activity`], which the
//! bar shows.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Result;

use super::command::{Action, Bounds, Command, FilterWay, Idx, Move, OptionIdx, ShotIdx};
use super::keys::Context;
use crate::choice::{Choose, OnHandEdited, Session};
use crate::config::OptionName;
use crate::group::{self, SceneNo};
use crate::looks::Fresh;
use crate::option::LookOption;
use crate::rate::Stars;
use crate::rpl::Pp3State;
use crate::shot::Shot;

/// The hint for a command that names a shot or option other than the ones on screen: it was
/// sent from a page that is no longer the current one.
const OUT_OF_DATE: &str = "That page was out of date; here is the current one.";

/// The hint for a choice whose render was found out of date and is being made again.
const NO_FRESH_RENDER: &str = "That option's render is out of date and is being made again.";

/// Writes to the photographer's files. [`Session`] is the real one.
pub trait Disk {
    fn choose(&self, nef: &Path, render: &Fresh, on: OnHandEdited) -> Result<Choose>;
    /// Sets the rating; the label stays as the file holds it.
    fn rate(&self, nef: &Path, stars: Stars) -> Result<()>;
    fn pp3_state(&self, nef: &Path) -> Result<Pp3State>;
}

impl Disk for Session {
    fn choose(&self, nef: &Path, render: &Fresh, on: OnHandEdited) -> Result<Choose> {
        Session::choose(self, nef, render, on)
    }

    fn rate(&self, nef: &Path, stars: Stars) -> Result<()> {
        Session::rate(self, nef, stars)
    }

    fn pp3_state(&self, nef: &Path) -> Result<Pp3State> {
        Session::pp3_state(self, nef)
    }
}

/// How a (shot, option) pair ended up after a batch: its render, or why it has none.
pub type Outcome = ((ShotIdx, OptionIdx), Result<Fresh, String>);

/// The lowest rating shown; 0 is "all".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Filter(Stars);

impl Filter {
    pub fn at_least(stars: Stars) -> Filter {
        Filter(stars)
    }

    pub fn admits(self, rating: Stars) -> bool {
        rating.get() >= self.0.get()
    }

    /// The minimums the steps visit, ascending. Not 1: "at least 1 star" and "all" differ
    /// only by unrated shots, which the filter chip would not tell apart.
    const STEPS: [u8; 5] = [0, 2, 3, 4, 5];

    /// Up: all, 2, 3, 4, 5; down is the reverse. A minimum the steps do not visit (1) moves
    /// to its neighbour in the given direction. `None` at the end: 5 going up, all going
    /// down. The ends do not wrap: one key press would otherwise turn the strictest filter
    /// into every shot of the card.
    pub fn step(self, way: FilterWay) -> Option<Filter> {
        let now = self.0.get();
        let stepped = match way {
            FilterWay::Up => Filter::STEPS.iter().find(|&&n| n > now),
            FilterWay::Down => Filter::STEPS.iter().rev().find(|&&n| n < now),
        };
        stepped.map(|&n| Filter(Stars::new(n)))
    }

    pub fn label(self) -> String {
        match self.0.get() {
            0 => "all".to_owned(),
            n => format!("≥{n}★"),
        }
    }
}

/// How far the worker has got, in renders.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Progress {
    pub done: usize,
    pub total: usize,
    /// The render that finished last.
    pub last: Option<String>,
}

/// Whether renders can be made, which needs RawTherapee's default profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Readiness {
    /// Resolving the default profile (about ten seconds). What is already rendered can be
    /// browsed meanwhile.
    Preparing,
    Ready,
    /// No render can be made, and why: the default profile cannot be resolved, or the
    /// worker died.
    Unavailable(String),
}

/// The background worker that renders what the filter needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Worker {
    Idle,
    Running(Progress),
}

/// What the page shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    /// The filter admits shots, none is rendered yet, and renders are on their way.
    Waiting(Wait),
    /// At least one shot is ready, or the filter admits nothing there is to wait for.
    Working,
    /// The filter admits shots, none is rendered, and none can be.
    Broken(String),
}

/// What the progress page is waiting for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wait {
    Profile,
    Renders(Progress),
}

/// What the renderer is doing about the shots the filter admits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Activity {
    /// Every shot the filter admits is rendered or has a recorded failure.
    Idle,
    /// Renders are owed and RawTherapee's default profile is not resolved yet.
    PreparingProfile,
    Rendering(Progress),
    /// Renders are owed and cannot be made.
    Stopped(String),
}

/// How the option on screen came to be on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The photographer arrived at this shot; the option shown is what was in effect or the
    /// last one looked at. `Accept` refuses to write it.
    Arrived,
    /// An option key or flip put it there on this shot: `Accept` writes it.
    Picked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewing {
    pub shown: OptionIdx,
    /// What was shown before `shown` on this shot, for flipping back.
    pub before: Option<OptionIdx>,
    pub origin: Origin,
}

/// A choice waiting for the photographer's yes: the shot's processing was edited in
/// RawTherapee and replacing it needs confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Confirmation {
    pub shot: ShotIdx,
    pub option: OptionIdx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Browse,
    Confirming(Confirmation),
}

/// A line the page shows after the last command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Notice {
    /// A write or a read failed.
    Error(String),
    /// The command was not applied, and why.
    Hint(String),
}

/// One shot as the session knows it.
pub struct Entry {
    pub shot: Shot,
    pub rating: Stars,
    pub pp3: Pp3State,
}

impl Entry {
    pub fn name(&self) -> String {
        self.shot.name()
    }
}

pub struct Model {
    entries: Vec<Entry>,
    /// The scene of each entry, as the pipeline decides scenes (`group::scene_numbers`).
    scenes: Vec<SceneNo>,
    /// The shot the resume record names, as far as the model knows: the one last handed to
    /// [`Model::unsaved_place`], or the one the session opened on.
    saved_place: Option<ShotIdx>,
    /// The configured look options: the one list an `OptionIdx` indexes, for the pages, the
    /// keys and the worker alike.
    options: Vec<LookOption>,
    /// Only renders compared with today's inputs (`looks::Fresh`) are held: a stale one is
    /// absent, so it is queued like a missing one and cannot be shown or chosen.
    renders: HashMap<(ShotIdx, OptionIdx), Fresh>,
    failed: BTreeMap<(ShotIdx, OptionIdx), String>,
    filter: Filter,
    /// The shot on screen; `None` while no shot the filter admits is rendered.
    current: Option<ShotIdx>,
    /// The last shot the photographer was on, which the worker renders around. Kept while
    /// `current` is `None`.
    anchor: Option<ShotIdx>,
    viewing: Viewing,
    mode: Mode,
    readiness: Readiness,
    worker: Worker,
    notice: Option<Notice>,
}

impl Model {
    /// A session over `entries`, with the `renders` already on disk. It starts
    /// [`Readiness::Preparing`] with the worker taken to be running. It shows `resume`, the
    /// shot the photographer was last on, when the filter shows it; else the first rendered
    /// shot of the filter after it; else the first one. Scenes split where the capture
    /// times, in order, have a gap longer than `scene_gap_s`.
    pub fn new(
        entries: Vec<Entry>,
        options: Vec<LookOption>,
        renders: HashMap<(ShotIdx, OptionIdx), Fresh>,
        filter: Filter,
        scene_gap_s: f64,
        resume: Option<ShotIdx>,
    ) -> Model {
        let first = Idx::new(0, options.len()).expect("the configuration has an option");
        let times: Vec<_> = entries.iter().map(|e| e.shot.taken_at).collect();
        let scenes = group::scene_numbers(&times, scene_gap_s);
        let mut model = Model {
            entries,
            scenes,
            saved_place: None,
            options,
            renders,
            failed: BTreeMap::new(),
            filter,
            current: None,
            anchor: resume,
            viewing: Viewing {
                shown: first,
                before: None,
                origin: Origin::Arrived,
            },
            mode: Mode::Browse,
            readiness: Readiness::Preparing,
            worker: Worker::Running(Progress::default()),
            notice: None,
        };
        model.snap(Beyond::First);
        // Opening on a shot is not a move: the record is rewritten when the photographer
        // goes elsewhere.
        model.saved_place = model.current;
        model
    }

    /// The shot on screen if it is not the one last handed out (or the one the session
    /// opened on), and marks it handed out: the caller writes it down, so the next launch
    /// opens there.
    pub fn unsaved_place(&mut self) -> Option<ShotIdx> {
        let current = self.current?;
        (self.saved_place != Some(current)).then(|| {
            self.saved_place = Some(current);
            current
        })
    }

    /// The scene `shot` is in, counted from 1, and how many scenes the shoot has.
    pub fn scene_of(&self, shot: ShotIdx) -> (usize, usize) {
        let count = self.scenes.last().map_or(0, |s| s.get() + 1);
        (self.scene(shot).get() + 1, count)
    }

    fn scene(&self, shot: ShotIdx) -> SceneNo {
        *self
            .scenes
            .get(shot.get())
            .expect("a ShotIdx is built against this list")
    }

    pub fn bounds(&self) -> Bounds {
        Bounds {
            shots: self.entries.len(),
            options: self.options.len(),
        }
    }

    pub fn entry(&self, shot: ShotIdx) -> &Entry {
        self.entries
            .get(shot.get())
            .expect("a ShotIdx is built against this list")
    }

    fn entry_mut(&mut self, shot: ShotIdx) -> &mut Entry {
        self.entries
            .get_mut(shot.get())
            .expect("a ShotIdx is built against this list")
    }

    pub fn option(&self, option: OptionIdx) -> &LookOption {
        self.options
            .get(option.get())
            .expect("an OptionIdx is built against this list")
    }

    pub fn option_indices(&self) -> impl Iterator<Item = OptionIdx> + use<> {
        Idx::all(self.options.len())
    }

    pub fn filter(&self) -> Filter {
        self.filter
    }

    /// What the renderer is doing about the shots the filter admits.
    pub fn activity(&self) -> Activity {
        if self.missing().is_empty() {
            return Activity::Idle;
        }
        match (&self.readiness, &self.worker) {
            (Readiness::Unavailable(why), _) => Activity::Stopped(why.clone()),
            (Readiness::Preparing, _) => Activity::PreparingProfile,
            (Readiness::Ready, Worker::Running(progress)) => Activity::Rendering(progress.clone()),
            // The caller that changed the filter claims the worker in the same locked step.
            (Readiness::Ready, Worker::Idle) => Activity::Rendering(Progress::default()),
        }
    }

    pub fn stage(&self) -> Stage {
        if self.has_visible() {
            return Stage::Working;
        }
        match self.activity() {
            Activity::Idle => Stage::Working,
            Activity::PreparingProfile => Stage::Waiting(Wait::Profile),
            Activity::Rendering(progress) => Stage::Waiting(Wait::Renders(progress)),
            Activity::Stopped(why) => Stage::Broken(why),
        }
    }

    pub fn current(&self) -> Option<ShotIdx> {
        self.current
    }

    pub fn viewing(&self) -> Viewing {
        self.viewing
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn notice(&self) -> Option<&Notice> {
        self.notice.as_ref()
    }

    pub fn failures(&self) -> impl Iterator<Item = (ShotIdx, OptionIdx, &str)> {
        self.failed.iter().map(|(&(s, o), e)| (s, o, e.as_str()))
    }

    pub fn render(&self, shot: ShotIdx, option: OptionIdx) -> Option<&Fresh> {
        self.renders.get(&(shot, option))
    }

    /// The keys in force.
    pub fn context(&self) -> Context {
        match (self.stage(), self.mode) {
            (Stage::Working, Mode::Browse) => Context::Browse,
            (Stage::Working, Mode::Confirming(_)) => Context::Answer,
            (Stage::Waiting(_) | Stage::Broken(_), _) => Context::Wait,
        }
    }

    /// The option the shot's pp3 holds, if the pp3 is the result of a choice.
    pub fn chosen(&self, shot: ShotIdx) -> Option<OptionIdx> {
        match &self.entry(shot).pp3 {
            Pp3State::Chosen { option } => self.option_named(option),
            Pp3State::Absent
            | Pp3State::Generated
            | Pp3State::Interrupted { .. }
            | Pp3State::HandEdited => None,
        }
    }

    fn option_named(&self, name: &OptionName) -> Option<OptionIdx> {
        self.option_indices()
            .find(|&o| self.option(o).name == *name)
    }

    fn rendered_fully(&self, shot: ShotIdx) -> bool {
        self.option_indices()
            .all(|o| self.renders.contains_key(&(shot, o)))
    }

    fn in_filter(&self, shot: ShotIdx) -> bool {
        self.filter.admits(self.entry(shot).rating)
    }

    /// The shots the filter admits that are fully rendered.
    fn shown_by_filter(&self) -> impl Iterator<Item = ShotIdx> {
        Idx::all(self.entries.len()).filter(|&s| self.in_filter(s) && self.rendered_fully(s))
    }

    fn has_visible(&self) -> bool {
        self.current.is_some() || self.shown_by_filter().next().is_some()
    }

    /// The shots the working view steps through: the filter's, as far as they are rendered,
    /// and the current one even when a new rating has taken it out of the filter.
    pub fn visible(&self) -> Vec<ShotIdx> {
        Idx::all(self.entries.len())
            .filter(|&s| Some(s) == self.current || (self.in_filter(s) && self.rendered_fully(s)))
            .collect()
    }

    /// Shots the filter admits that will not be shown: a render of theirs failed.
    pub fn skipped(&self) -> Vec<ShotIdx> {
        Idx::all(self.entries.len())
            .filter(|&s| self.in_filter(s) && !self.rendered_fully(s))
            .filter(|&s| {
                self.option_indices()
                    .any(|o| self.failed.contains_key(&(s, o)))
            })
            .collect()
    }

    /// The (shot, option) pairs the filter admits that have neither a render nor a failure,
    /// a shot at a time from the photographer's place outwards: the shots after it and
    /// before it alternately, nearest first, the one at the anchor first of all.
    fn missing(&self) -> Vec<(ShotIdx, OptionIdx)> {
        let admitted: Vec<ShotIdx> = Idx::all(self.entries.len())
            .filter(|&s| self.in_filter(s))
            .collect();
        let anchor = self.anchor.map_or(0, ShotIdx::get);
        // Where the anchor falls among the admitted shots: the first one at or after it.
        let at = admitted.partition_point(|s| s.get() < anchor);
        let mut order: Vec<(usize, usize, ShotIdx)> = admitted
            .iter()
            .enumerate()
            .map(|(i, &s)| match i.checked_sub(at) {
                Some(ahead) => (ahead, 0, s),
                None => (at - i, 1, s),
            })
            .collect();
        order.sort_by_key(|&(distance, side, _)| (distance, side));
        order
            .into_iter()
            .flat_map(|(_, _, s)| self.option_indices().map(move |o| (s, o)))
            .filter(|pair| !self.renders.contains_key(pair) && !self.failed.contains_key(pair))
            .collect()
    }

    // The worker's side.

    /// The default profile is resolved.
    pub fn profile_resolved(&mut self) {
        if self.readiness == Readiness::Preparing {
            self.readiness = Readiness::Ready;
        }
    }

    /// The key of every render held, for the worker to compare with the full keys once the
    /// default profile is known.
    pub fn render_keys(&self) -> Vec<((ShotIdx, OptionIdx), String)> {
        self.renders
            .iter()
            .map(|(&pair, r)| (pair, r.manifest.key.clone()))
            .collect()
    }

    /// Drops the renders the worker found stale under the default profile. A pair whose
    /// render has a different key now was rendered meanwhile and stays. The shot on screen
    /// moves off when it is no longer fully rendered, as when a filter change hides it.
    pub fn discard_stale(&mut self, stale: &[((ShotIdx, OptionIdx), String)]) {
        for (pair, key) in stale {
            if self
                .renders
                .get(pair)
                .is_some_and(|r| r.manifest.key == *key)
            {
                self.renders.remove(pair);
            }
        }
        if self.current.is_some_and(|c| !self.rendered_fully(c)) {
            self.current = None;
            self.snap(Beyond::Last);
        }
    }

    /// No render can be made: the default profile cannot be resolved, or the worker died.
    pub fn renders_unavailable(&mut self, why: String) {
        self.readiness = Readiness::Unavailable(why);
        self.worker = Worker::Idle;
    }

    /// What the worker renders next, at most `chunk` pairs, or `None` when the filter has
    /// nothing left to render: the worker is then idle. Called with the model locked, so a
    /// filter change cannot slip between "nothing left" and "idle".
    pub fn next_batch(&mut self, chunk: usize) -> Option<Vec<(ShotIdx, OptionIdx)>> {
        let mut missing = self.missing();
        if missing.is_empty() {
            self.worker = Worker::Idle;
            return None;
        }
        let (done, last) = match &self.worker {
            Worker::Running(p) => (p.done, p.last.clone()),
            Worker::Idle => (0, None),
        };
        self.worker = Worker::Running(Progress {
            done,
            total: done + missing.len(),
            last,
        });
        missing.truncate(chunk);
        Some(missing)
    }

    /// One render of the running batch finished.
    pub fn batch_progress(&mut self, finished: &str) {
        if let Worker::Running(p) = &mut self.worker {
            p.done = (p.done + 1).min(p.total);
            p.last = Some(finished.to_owned());
        }
    }

    /// The batch ended: what each pair now has on disk. The first shot to be fully rendered
    /// becomes the current one when none is; the photographer is never moved off a shot
    /// they are on.
    pub fn batch_done(&mut self, outcomes: Vec<Outcome>) {
        for (pair, outcome) in outcomes {
            match outcome {
                Ok(rendered) => {
                    self.failed.remove(&pair);
                    self.renders.insert(pair, rendered);
                }
                Err(why) => {
                    self.failed.insert(pair, why);
                }
            }
        }
        if self.current.is_none() {
            self.snap(Beyond::Last);
        }
    }

    /// Whether the caller must start the worker: the filter needs renders, renders can be
    /// made, and none is running. Marks it running.
    pub fn claim_worker(&mut self) -> bool {
        match (&self.readiness, &self.worker, self.missing().is_empty()) {
            (Readiness::Unavailable(_), _, _) | (_, Worker::Running(_), _) | (_, _, true) => false,
            (Readiness::Preparing | Readiness::Ready, Worker::Idle, false) => {
                self.worker = Worker::Running(Progress::default());
                true
            }
        }
    }

    // The photographer's side.

    /// Applies `command`. Nothing is applied when the model is not ready for it; the notice
    /// says why.
    pub fn apply(&mut self, command: Command, disk: &dyn Disk) {
        self.notice = None;
        let working = self.stage() == Stage::Working;
        match (working, self.mode, command) {
            (true, Mode::Browse, command) => self.browse(command, disk),
            (true, Mode::Confirming(c), Command::On(shot, Action::Yes(option)))
                if shot == c.shot && option == c.option =>
            {
                self.confirmed(c, disk);
            }
            (true, Mode::Confirming(c), Command::On(shot, Action::No(option)))
                if shot == c.shot && option == c.option =>
            {
                self.mode = Mode::Browse;
                self.hint("Kept the processing as edited in RawTherapee.");
            }
            (true, Mode::Confirming(_), Command::On(_, Action::Yes(_) | Action::No(_))) => {
                self.hint(OUT_OF_DATE);
            }
            (true, Mode::Confirming(_), _) => {
                self.hint("Answer the question first: y or n.");
            }
            (false, _, Command::StepFilter(way)) => self.step_filter(way),
            (false, _, _) => {
                self.hint("Still rendering; the view opens when the first shot is ready.");
            }
        }
    }

    fn browse(&mut self, command: Command, disk: &dyn Disk) {
        match command {
            Command::Move(m) => self.go(m),
            Command::Goto(shot) => {
                if self.visible().contains(&shot) {
                    self.arrive(shot);
                } else {
                    self.hint("That shot is not in the current filter.");
                }
            }
            Command::StepFilter(way) => self.step_filter(way),
            Command::On(shot, _) if self.current != Some(shot) => self.hint(OUT_OF_DATE),
            Command::On(shot, action) => self.on_screen(shot, action, disk),
        }
    }

    /// `action` on `shot`, which is the shot on screen.
    fn on_screen(&mut self, shot: ShotIdx, action: Action, disk: &dyn Disk) {
        match action {
            Action::View(option) => self.view(option),
            Action::Flip => self.flip(),
            Action::Accept(viewed) if viewed != self.viewing.shown => self.hint(OUT_OF_DATE),
            Action::Accept(viewed) => self.accept(shot, viewed, disk),
            Action::Rate(stars) => self.rate(shot, stars, disk),
            Action::Yes(_) | Action::No(_) => self.hint("There is no question to answer."),
        }
    }

    fn hint(&mut self, text: &str) {
        self.notice = Some(Notice::Hint(text.to_owned()));
    }

    fn fail(&mut self, error: &anyhow::Error) {
        self.notice = Some(Notice::Error(format!("{error:#}")));
    }

    /// Moves as `way` says, or says there is nowhere to go.
    fn go(&mut self, way: Move) {
        let visible = self.visible();
        let Some(at) = self
            .current
            .and_then(|c| visible.iter().position(|&s| s == c))
        else {
            return;
        };
        match self.destination(&visible, at, way) {
            Some(shot) => self.arrive(shot),
            None => self.hint(match way {
                Move::Next | Move::Last | Move::PageForward => "That is the last shot.",
                Move::Previous | Move::First | Move::PageBack => "That is the first shot.",
                Move::NextScene => "No later scene has a shot at this filter.",
                Move::PreviousScene => "No earlier scene has a shot at this filter.",
                Move::NextUndecided => "No shot after this one is undecided.",
                Move::PreviousUndecided => "No shot before this one is undecided.",
            }),
        }
    }

    /// Where `way` leads from `visible[at]`, among `visible`, or `None` when it leads
    /// nowhere other than where the photographer is.
    fn destination(&self, visible: &[ShotIdx], at: usize, way: Move) -> Option<ShotIdx> {
        let ahead = &visible[at + 1..];
        let behind = &visible[..at];
        let first_of = |scene: SceneNo| visible.iter().copied().find(|&s| self.scene(s) == scene);
        let moved = |to: usize| (to != at).then(|| visible[to]);
        match way {
            Move::Next => ahead.first().copied(),
            Move::Previous => behind.last().copied(),
            Move::NextScene => {
                let here = self.scene(visible[at]);
                ahead.iter().copied().find(|&s| self.scene(s) > here)
            }
            Move::PreviousScene => {
                let start = first_of(self.scene(visible[at]));
                if start == Some(visible[at]) {
                    behind.last().and_then(|&s| first_of(self.scene(s)))
                } else {
                    start
                }
            }
            Move::NextUndecided => ahead.iter().copied().find(|&s| self.chosen(s).is_none()),
            Move::PreviousUndecided => behind
                .iter()
                .rev()
                .copied()
                .find(|&s| self.chosen(s).is_none()),
            Move::First => moved(0),
            Move::Last => moved(visible.len() - 1),
            Move::PageForward => moved((at + Move::PAGE).min(visible.len() - 1)),
            Move::PageBack => moved(at.saturating_sub(Move::PAGE)),
        }
    }

    /// Moves to `shot`: it shows the option in effect there, else the one last looked at.
    fn arrive(&mut self, shot: ShotIdx) {
        self.current = Some(shot);
        self.anchor = Some(shot);
        self.viewing = Viewing {
            shown: self.chosen(shot).unwrap_or(self.viewing.shown),
            before: None,
            origin: Origin::Arrived,
        };
    }

    fn view(&mut self, option: OptionIdx) {
        let before = if option == self.viewing.shown {
            self.viewing.before
        } else {
            Some(self.viewing.shown)
        };
        self.viewing = Viewing {
            shown: option,
            before,
            origin: Origin::Picked,
        };
    }

    fn flip(&mut self) {
        match self.viewing.before {
            Some(before) => {
                self.viewing = Viewing {
                    shown: before,
                    before: Some(self.viewing.shown),
                    origin: Origin::Picked,
                };
            }
            None => self.hint("Nothing viewed before this one on this shot."),
        }
    }

    /// Writes `option`, which is on screen for `shot`.
    fn accept(&mut self, shot: ShotIdx, option: OptionIdx, disk: &dyn Disk) {
        match self.viewing.origin {
            Origin::Arrived => self.hint(
                "Press an option's key to view it first; Enter then chooses the option on screen.",
            ),
            Origin::Picked if self.chosen(shot) == Some(option) => self.go(Move::Next),
            Origin::Picked => {
                let Some(render) = self.render(shot, option).cloned() else {
                    self.hint(NO_FRESH_RENDER);
                    return;
                };
                let nef = self.entry(shot).shot.file.clone();
                match disk.choose(&nef, &render, OnHandEdited::Refuse) {
                    Ok(Choose::Applied) => self.chosen_now(shot, disk),
                    Ok(Choose::NeedsConfirmation) => {
                        self.mode = Mode::Confirming(Confirmation { shot, option });
                    }
                    Err(e) => self.fail(&e),
                }
            }
        }
    }

    fn confirmed(&mut self, c: Confirmation, disk: &dyn Disk) {
        let Some(render) = self.render(c.shot, c.option).cloned() else {
            self.mode = Mode::Browse;
            self.hint(NO_FRESH_RENDER);
            return;
        };
        let nef = self.entry(c.shot).shot.file.clone();
        match disk.choose(&nef, &render, OnHandEdited::Replace) {
            Ok(Choose::Applied) => {
                self.mode = Mode::Browse;
                self.chosen_now(c.shot, disk);
            }
            // A replacement was asked for and refused: leave the question open.
            Ok(Choose::NeedsConfirmation) => {
                self.hint("The pp3 still needs confirmation; nothing was written.");
            }
            Err(e) => {
                self.mode = Mode::Browse;
                self.fail(&e);
            }
        }
    }

    /// A choice was written for `shot`: read the pp3's state back and go on.
    fn chosen_now(&mut self, shot: ShotIdx, disk: &dyn Disk) {
        self.refresh_pp3(shot, disk);
        if self.notice.is_none() {
            self.go(Move::Next);
        }
    }

    fn refresh_pp3(&mut self, shot: ShotIdx, disk: &dyn Disk) {
        let nef = self.entry(shot).shot.file.clone();
        match disk.pp3_state(&nef) {
            Ok(state) => self.entry_mut(shot).pp3 = state,
            Err(e) => self.fail(&e),
        }
    }

    fn rate(&mut self, shot: ShotIdx, stars: Stars, disk: &dyn Disk) {
        let nef = self.entry(shot).shot.file.clone();
        match disk.rate(&nef, stars) {
            Ok(()) => {
                self.entry_mut(shot).rating = stars;
                self.refresh_pp3(shot, disk);
            }
            Err(e) => self.fail(&e),
        }
    }

    fn step_filter(&mut self, way: FilterWay) {
        match self.filter.step(way) {
            Some(filter) => {
                self.filter = filter;
                self.snap(Beyond::Last);
            }
            None => self.hint(match way {
                FilterWay::Up => "That is the strictest filter, 5 stars.",
                FilterWay::Down => "That filter already shows every shot.",
            }),
        }
    }

    /// Keeps `current` on a shot the filter shows: the current one if it still is, else the
    /// next one at or after the anchor, else the shot `beyond` says. `None` while no shot
    /// the filter admits is rendered; the anchor stays, so the worker renders around where
    /// the photographer was.
    fn snap(&mut self, beyond: Beyond) {
        let visible: Vec<ShotIdx> = self.shown_by_filter().collect();
        if self.current.is_some_and(|c| visible.contains(&c)) {
            return;
        }
        let after = self.anchor.map_or(0, ShotIdx::get);
        let target = visible
            .iter()
            .find(|s| s.get() >= after)
            .or_else(|| match beyond {
                Beyond::First => visible.first(),
                Beyond::Last => visible.last(),
            })
            .copied();
        match target {
            Some(shot) => self.arrive(shot),
            None => self.current = None,
        }
    }
}

/// Which shot `Model::snap` takes when none at or after the anchor is shown: the shoot's
/// first at launch, where the photographer has not seen the shots yet, the last when the
/// filter moves them off the end of the shoot.
#[derive(Debug, Clone, Copy)]
enum Beyond {
    First,
    Last,
}
