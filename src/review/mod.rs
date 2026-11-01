//! `rawpilot review DIR`: a local web page for going through a processed shoot, choosing a
//! look option per shot and setting ratings, with nothing slow while the photographer works.
//!
//! The photographer runs `rawpilot process DIR`, then this. At launch the shots in the
//! starting filter (rated at least `review.prerender_min_rating`) are checked for renders;
//! the missing ones, and the ones made from other inputs than today's (a retuned look
//! profile), are made in the background (`service`), nearest the shot on screen first. A
//! render is shown and written only after `looks::Fresh::check`; the part of the check that
//! needs RawTherapee's default profile runs when that is resolved. The working view opens as soon as one shot the filter admits is rendered and the
//! rest join it as they land, with the progress in the bar; the progress page shows only
//! while nothing the filter admits is rendered. RawTherapee's default profile is resolved
//! in the background too, and only a render or the first rating of a shot with no pp3
//! waits for it. Requests serve pictures from `<DIR>/.rawpilot/looks/` and write through
//! `choice::Session`; no request does image work.
//!
//! The server listens on 127.0.0.1 only and prints one line on stdout, the sign-in URL
//! `http://127.0.0.1:PORT/?t=TOKEN`; see `auth` for what admits a request.
//!
//! | Module | |
//! | --- | --- |
//! | `command` | the wire commands and the index witnesses they parse into |
//! | `keys` | the one key table: script, help overlay and tests read it |
//! | `model` | the session state and every transition of it |
//! | `open` | printing the sign-in URL, and opening it in the browser |
//! | `pages` | the markup, drawn from the model |
//! | `resume` | the last shot viewed, kept for the next launch |
//! | `service` | the model behind its lock, and the render worker |
//! | `server`, `auth`, `images` | HTTP, admission, and the picture routes |

mod auth;
mod command;
mod images;
mod keys;
mod model;
mod open;
mod pages;
mod resume;
mod server;
mod service;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use tracing::{info, warn};

use self::auth::Token;
use self::command::{Idx, OptionIdx, ShotIdx};
use self::images::LooksRoot;
use self::keys::{Keymap, OptionKey};
use self::model::{Entry, Filter, Model};
pub use self::open::Browser;
use self::pages::Layout;
use self::resume::Place;
use self::service::{Renderer, Review};
use crate::cache::Cache;
use crate::choice::Session;
use crate::config::Config;
use crate::lock::{FolderLock, Run};
use crate::looks::{Against, Fresh, Inputs, RenderParams, Standing, Store as LooksStore};
use crate::option::LookOption;
use crate::rate::Stars;
use crate::rpl::Rpl;
use crate::rtcli::{self, DefaultProfiles};
use crate::sidecar::{self, FoundMark};
use crate::{nef, writeback};

/// The letters a look option's key cannot be, because the UI binds them (lower case; case
/// does not matter).
pub fn reserved_option_letters() -> Vec<char> {
    keys::reserved_letters()
}

/// Opens the review of `dir` on 127.0.0.1:`port` (0 lets the system pick) and serves until
/// interrupted. `browser` says whether the sign-in URL is also opened in the default browser.
pub fn run(dir: &Path, cfg: &Config, port: u16, browser: Browser) -> Result<()> {
    // Held until the server stops.
    let folder = FolderLock::acquire(dir, Run::Review)?;
    let rt = sidecar::read_rt_options(&sidecar::default_rt_options_path()?)?;
    rtcli::check_profile_is_shared(&rt.config_dir, rt.custom_profile_builder)?;
    let options = LookOption::load_all(&cfg.review.options)?;
    let keys = Keymap::build(
        &options
            .iter()
            .map(|o| OptionKey {
                name: o.name.as_str(),
                key: o.key,
            })
            .collect::<Vec<_>>(),
    )?;
    let cache = Cache::open(&folder)?;
    let shots = nef::scan(dir)?;
    ensure!(!shots.is_empty(), "no readable NEFs in {}", dir.display());
    let files: Vec<&Path> = shots.iter().map(|s| s.file.as_path()).collect();
    let marks = writeback::current_marks(&files, &rt)?;

    let defaults =
        DefaultProfiles::new(&rt, shots[0].file.clone(), cache.root().join("rt-template"));
    let session = Session::new(LooksStore::new(cache.root()), rt, defaults, &folder);
    let entries = shots
        .into_iter()
        .zip(marks)
        .map(|(shot, mark)| {
            let pp3 = session.pp3_state(&shot.file)?;
            Ok(Entry {
                shot,
                rating: stars_of(&mark),
                pp3,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let params = RenderParams::of(&cfg.review);
    let renders = load_renders(&session, &entries, &options, &params);
    let place = Place::under(cache.root());
    let resume = place.load().and_then(|name| shot_named(&entries, &name));
    let model = Model::new(
        entries,
        options,
        renders,
        Filter::at_least(Stars::new(cfg.review.prerender_min_rating)),
        cfg.group.scene_gap_s,
        resume,
    );
    let looks_dir = session.looks().root().to_owned();
    std::fs::create_dir_all(&looks_dir)
        .with_context(|| format!("creating {}", looks_dir.display()))?;
    let review = Arc::new(Review::new(
        model,
        session,
        Renderer {
            params,
            jobs_max: cfg.review.render_jobs,
            cache_root: cache.root().to_owned(),
        },
        keys,
        Layout::of(&cfg.review),
        LooksRoot::open(&looks_dir).with_context(|| format!("opening {}", looks_dir.display()))?,
        place,
    ));
    serve(review, port, browser)
}

/// The shot the remembered place names, if the folder still has it.
fn shot_named(entries: &[Entry], name: &str) -> Option<ShotIdx> {
    let found = Idx::all(entries.len())
        .zip(entries)
        .find(|(_, e)| e.name() == name);
    if found.is_none() {
        info!(
            "{name}, where the last review stopped, is not in this folder; starting at the first shot"
        );
    }
    found.map(|(shot, _)| shot)
}

fn stars_of(mark: &FoundMark) -> Stars {
    mark.rating
        .and_then(|r| u8::try_from(r).ok())
        .filter(|&r| r <= 5)
        .map_or(Stars::new(0), Stars::new)
}

/// The renders on disk that match every input but the default profile, which is not resolved
/// yet. A render that is stale or cannot be read is left out, so it is made again; the
/// worker compares the others with the full keys once the profile is known
/// (`Review::launch`).
fn load_renders(
    session: &Session,
    entries: &[Entry],
    options: &[LookOption],
    params: &RenderParams,
) -> HashMap<(ShotIdx, OptionIdx), Fresh> {
    let mut renders = HashMap::new();
    for (shot, entry) in Idx::all(entries.len()).zip(entries) {
        let record = Rpl::load(&entry.shot.file);
        for (option, look_option) in Idx::all(options.len()).zip(options) {
            let standing = Inputs::new(&entry.shot, look_option, &record, params)
                .and_then(|inputs| session.looks().standing(Against::Inputs(&inputs)));
            match standing {
                Ok(Standing::Fresh(fresh)) => {
                    renders.insert((shot, option), fresh);
                }
                Ok(Standing::Missing) => {}
                Ok(Standing::Stale(why)) => info!(
                    "{}: {}: out of date ({why}); it will be rendered again",
                    entry.name(),
                    look_option.name
                ),
                Err(e) => warn!(
                    "{}: {}: {e:#}; it will be rendered again",
                    entry.name(),
                    look_option.name
                ),
            }
        }
    }
    renders
}

fn serve(review: Arc<Review>, port: u16, browser: Browser) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, port))
            .await
            .with_context(|| format!("cannot listen on 127.0.0.1:{port}"))?;
        let port = listener.local_addr()?.port();
        let token = Token::generate()?;
        let url = format!("http://127.0.0.1:{port}/?t={}", token.as_str());
        let app = server::App::new(Arc::clone(&review), token, port);
        review.launch();
        // The sign-in URL is the command's one line of output.
        open::announce(&url, &mut std::io::stdout(), browser, &open::xdg_open)
            .context("writing the sign-in URL")?;
        info!("review is serving on 127.0.0.1:{port}; open the URL above");
        axum::serve(listener, server::router(app))
            .with_graceful_shutdown(async {
                if let Err(error) = tokio::signal::ctrl_c().await {
                    warn!(%error, "cannot wait for an interrupt; serving until killed");
                    std::future::pending::<()>().await;
                }
            })
            .await
            .context("the server stopped")
    })
}
