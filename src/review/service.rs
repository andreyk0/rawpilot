//! The session the server shares: the model behind a lock, the files it writes through, and
//! the background worker that makes the renders the filter needs.
//!
//! Request handlers only read the model, apply a command to it (a pp3 or a rating written,
//! a few milliseconds) and draw it. Everything slow runs on background threads: resolving the
//! default profile (the first rating of a shot with no pp3 waits for it, outside the model's
//! lock), and the worker's `looks::render_all`. The worker renders in chunks and asks the
//! model for the next chunk each time, so a filter change or a move to another shot takes
//! effect within one chunk.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use maud::Markup;
use tracing::{info, warn};

use super::command::{Action, Command, OptionIdx, ShotIdx};
use super::images::LooksRoot;
use super::keys::Keymap;
use super::model::{Model, Outcome};
use super::pages::{self, Layout};
use super::resume::Place;
use crate::choice::Session;
use crate::looks::{self, Against, Fresh, RenderParams, Standing};
use crate::pp3::DefaultProfile;
use crate::rpl::{Pp3State, Rpl};

/// What the worker needs to render besides the model's shots and options.
pub struct Renderer {
    pub params: RenderParams,
    pub jobs_max: usize,
    pub cache_root: PathBuf,
}

impl Renderer {
    /// Pairs per `render_all` call: four rounds of the pool, 16 renders at the default
    /// `render_jobs`, about a minute at the measured 15 s per NEF of four options, which is
    /// how long a filter change or a move waits to be noticed.
    fn chunk(&self) -> usize {
        4 * self.jobs_max
    }
}

pub struct Review {
    model: Mutex<Model>,
    session: Session,
    renderer: Renderer,
    pub keys: Keymap,
    pub layout: Layout,
    pub looks_root: LooksRoot,
    place: Place,
}

impl Review {
    pub fn new(
        model: Model,
        session: Session,
        renderer: Renderer,
        keys: Keymap,
        layout: Layout,
        looks_root: LooksRoot,
        place: Place,
    ) -> Review {
        Review {
            model: Mutex::new(model),
            session,
            renderer,
            keys,
            layout,
            looks_root,
            place,
        }
    }

    /// Runs `change` on the model, then writes down the shot on screen if it moved. The
    /// write happens under the lock, so two writers cannot interleave their temp files and
    /// the file names the shot the model last showed. A failed write is logged: losing the
    /// place costs the next launch a start at the first shot.
    fn change<R>(&self, change: impl FnOnce(&mut Model) -> R) -> R {
        let mut model = self.model();
        let result = change(&mut model);
        if let Some(shot) = model.unsaved_place()
            && let Err(e) = self.place.save(&model.entry(shot).name())
        {
            warn!("cannot remember the place: {e:#}");
        }
        result
    }

    /// A poisoned lock means a thread died holding it; the model is still the last state
    /// anyone wrote, and the page must keep drawing it.
    pub fn model(&self) -> MutexGuard<'_, Model> {
        self.model.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn draw(&self) -> Markup {
        pages::app(&self.model(), &self.keys, &self.layout)
    }

    /// Applies `command`, writing through the session, and starts the worker when the new
    /// filter needs renders and none is running.
    pub fn apply(self: &Arc<Self>, command: Command) {
        // A rating that must create a pp3 needs the default profile. Waiting here, with the
        // model unlocked, keeps the page drawing meanwhile; under the lock it would not.
        if self.needs_default_profile(&command) {
            let _ = self.session.defaults().get();
        }
        let start = self.change(|model| {
            model.apply(command, &self.session);
            model.claim_worker()
        });
        if start {
            self.spawn("the render worker", |review| review.work());
        }
    }

    /// Whether `command` writes a pp3 that does not exist yet, which starts from RawTherapee's
    /// default profile.
    fn needs_default_profile(&self, command: &Command) -> bool {
        match command {
            Command::On(shot, Action::Rate(_)) => {
                self.session.rating_creates_pp3()
                    && self.model().entry(*shot).pp3 == Pp3State::Absent
            }
            Command::Move(_)
            | Command::Goto(_)
            | Command::StepFilter(_)
            | Command::On(
                _,
                Action::View(_) | Action::Flip | Action::Accept(_) | Action::Yes(_) | Action::No(_),
            ) => false,
        }
    }

    /// Starts the background work of a new session: resolving the default profile, which
    /// takes seconds the page does not wait for, and rendering what the filter needs. The
    /// renders loaded at launch passed the check that does not need the profile; the resolver
    /// checks them against the full keys when it has it.
    pub fn launch(self: &Arc<Self>) {
        self.spawn("the default profile resolver", Review::resolve);
        self.spawn("the render worker", |review| review.work());
    }

    /// Resolves the default profile, then compares the renders shown so far with the full
    /// keys: the launch check could not see the default profile, so a render made under
    /// another one is dropped and queued here.
    fn resolve(self: &Arc<Self>) {
        let Some(default) = self.default_profile() else {
            return;
        };
        self.discard_stale(&default);
        let start = self.model().claim_worker();
        if start {
            self.spawn("the render worker", |review| review.work());
        }
    }

    pub(super) fn discard_stale(&self, default: &DefaultProfile) {
        // The model is unlocked while the disk is read.
        let held = self.model().render_keys();
        let stale: Vec<_> = held
            .into_iter()
            .filter(|(pair, _)| !self.is_fresh_under(*pair, default))
            .collect();
        if !stale.is_empty() {
            info!(
                "{} renders were made under another default profile; making them again",
                stale.len()
            );
        }
        self.change(|model| model.discard_stale(&stale));
    }

    /// Whether the render `pair` has on disk matches every input, the default profile
    /// included. A render that cannot be compared is not fresh.
    fn is_fresh_under(&self, pair: (ShotIdx, OptionIdx), default: &DefaultProfile) -> bool {
        let outcome = self.job(pair, default).and_then(|job| self.standing(&job));
        matches!(outcome, Ok(Standing::Fresh(_)))
    }

    fn standing(&self, job: &looks::Job) -> Result<Standing, String> {
        self.session
            .looks()
            .standing(Against::Job(job))
            .map_err(|e| format!("{e:#}"))
    }

    fn spawn(self: &Arc<Self>, what: &'static str, job: fn(&Arc<Review>)) {
        let review = Arc::clone(self);
        std::thread::spawn(move || {
            let outcome = catch_unwind(AssertUnwindSafe(|| job(&review)));
            if outcome.is_err() {
                warn!("{what} panicked");
                review
                    .model()
                    .renders_unavailable(format!("{what} stopped unexpectedly"));
            }
        });
    }

    /// The default profile, waiting for it to be resolved. The model hears how it went.
    fn default_profile(&self) -> Option<Arc<DefaultProfile>> {
        match self.session.defaults().get() {
            Ok(profile) => {
                self.model().profile_resolved();
                Some(profile)
            }
            Err(e) => {
                warn!("cannot resolve the default profile: {e:#}");
                self.model().renders_unavailable(format!("{e:#}"));
                None
            }
        }
    }

    fn work(&self) {
        loop {
            let Some(batch) = self.model().next_batch(self.renderer.chunk()) else {
                info!("renders ready");
                return;
            };
            let Some(default) = self.default_profile() else {
                return;
            };
            let outcomes = self.render(&batch, &default);
            self.change(|model| model.batch_done(outcomes));
        }
    }

    /// Renders `batch` and reports what each pair has on disk afterwards.
    fn render(&self, batch: &[(ShotIdx, OptionIdx)], default: &DefaultProfile) -> Vec<Outcome> {
        // The pairs that became jobs; the rest failed before rendering.
        let mut jobs = Vec::new();
        let mut outcomes = Vec::new();
        for &pair in batch {
            match self.job(pair, default) {
                Ok(job) => jobs.push((pair, job)),
                Err(why) => outcomes.push((pair, Err(why))),
            }
        }
        let planned: Vec<looks::Job> = jobs.iter().map(|(_, job)| job.clone()).collect();
        let report = looks::render_all(
            &planned,
            &self.renderer.params,
            self.renderer.jobs_max,
            &self.renderer.cache_root,
            // Each render is shown as soon as it is done, not when the whole batch is.
            &|p| {
                let (pair, job) = &jobs[p.job];
                let outcome = match p.status {
                    looks::Status::Failed { error } => Err(error.clone()),
                    looks::Status::Cached | looks::Status::Rendered { .. } => self.on_disk(job),
                };
                self.change(|model| {
                    model.batch_progress(&format!("{} {}", p.shot, p.option));
                    model.batch_done(vec![(*pair, outcome)]);
                });
            },
        );
        match report {
            Err(e) => {
                let why = format!("{e:#}");
                outcomes.extend(jobs.iter().map(|&(pair, _)| (pair, Err(why.clone()))));
            }
            Ok(report) => {
                assert_eq!(
                    report.statuses.len(),
                    jobs.len(),
                    "render_all reports on every job it was given"
                );
                for ((pair, job), status) in jobs.iter().zip(report.statuses) {
                    outcomes.push(match status {
                        looks::Status::Failed { error } => (*pair, Err(error)),
                        looks::Status::Cached | looks::Status::Rendered { .. } => {
                            (*pair, self.on_disk(job))
                        }
                    });
                }
            }
        }
        outcomes
    }

    fn job(
        &self,
        (shot, option): (ShotIdx, OptionIdx),
        default: &DefaultProfile,
    ) -> Result<looks::Job, String> {
        let (nef, look_option) = {
            let model = self.model();
            (model.entry(shot).shot.clone(), model.option(option).clone())
        };
        let record = Rpl::load(&nef.file);
        looks::Job::new(&nef, &look_option, default, &record, &self.renderer.params)
            .map_err(|e| format!("{e:#}"))
    }

    /// The render `job` has on disk, which must be the job's own.
    fn on_disk(&self, job: &looks::Job) -> Result<Fresh, String> {
        match self.standing(job)? {
            Standing::Fresh(fresh) => Ok(fresh),
            Standing::Missing => Err("no render was left on disk".to_owned()),
            Standing::Stale(why) => Err(format!("the render left on disk is out of date: {why}")),
        }
    }
}
