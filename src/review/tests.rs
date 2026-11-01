//! Goal: the review server admits only the signed-in browser, parses every request into
//! witnesses before acting, serves pictures only from the looks directory, shows the working
//! view only when everything the filter needs is rendered, and cannot choose a look by a
//! stray key. Method: a temp shoot with NEF stand-ins and hand-made renders, a real
//! `choice::Session` for the writes, and the router driven with `tower::ServiceExt::oneshot`;
//! the model-level rules use a fake `Disk` that records what was asked of it.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use chrono::DateTime;
use tower::ServiceExt;

use super::auth::Token;
use super::command::{Action, Command, FilterWay, Idx, Move, OptionIdx, ShotIdx};
use super::images::LooksRoot;
use super::keys::{Keymap, OptionKey};
use super::model::{Activity, Disk, Entry, Filter, Mode, Model, Notice, Stage};
use super::pages::Layout;
use super::resume::Place;
use super::server::{App, router};
use super::service::{Renderer, Review};
use crate::choice::{Choose, OnHandEdited, Session};
use crate::config::{Config, OptionConfig, OptionName, PartialSource};
use crate::lock::{FolderLock, Run};
use crate::looks::{
    Against, Fresh, Inputs, Job, Manifest, RenderParams, Standing, Store as LooksStore, View,
};
use crate::option::LookOption;
use crate::pp3::{self, DefaultProfile};
use crate::rate::Stars;
use crate::rpl::{Pp3State, Rpl};
use crate::rtcli::DefaultProfiles;
use crate::shot::{Exposure, Orientation, PreviewSpan, Shot};
use crate::sidecar::{RankSource, RtOptions, SidecarStyle};

const PORT: u16 = 7357;
const HOST: &str = "127.0.0.1:7357";
const ORIGIN: &str = "http://127.0.0.1:7357";
const TOKEN: &str = "tok3n";
const RENDERED: &str = "[General]\nRank=0\nColorLabel=0\n\n[Exposure]\nCompensation=1.00\n";
const DEFAULT: &str = "[General]\nRank=1\nColorLabel=0\n\n[Exposure]\nCompensation=0\n";
/// The gap, in seconds, after which a shot starts a new scene in these tests.
const SCENE_GAP_S: f64 = 90.0;
const OPTIONS: [(&str, char); 3] = [("Camera", 'c'), ("Film", 'm'), ("Gentle", 'g')];

fn name(s: &str) -> OptionName {
    s.to_owned().try_into().unwrap()
}

fn shot_at(file: PathBuf) -> Shot {
    Shot {
        file,
        file_size: 1,
        taken_at: DateTime::parse_from_rfc3339("2026-01-01T10:00:00+00:00").unwrap(),
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
    }
}

fn look_options() -> Vec<LookOption> {
    LookOption::load_all(
        &OPTIONS
            .iter()
            .map(|(n, key)| OptionConfig {
                name: name(n),
                key: *key,
                partial: PartialSource::None,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn keymap() -> Keymap {
    Keymap::build(
        &OPTIONS
            .iter()
            .map(|(name, key)| OptionKey { name, key: *key })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn render_params() -> RenderParams {
    RenderParams::of(&Config::load(None).unwrap().review)
}

/// The inputs of option number `o` for the stand-in `nef`: what a render must have been made
/// from to be fresh.
fn inputs_of(nef: &Path, o: usize) -> Inputs {
    Inputs::new(
        &shot_at(nef.to_owned()),
        &look_options()[o],
        &Rpl::default(),
        &render_params(),
    )
    .unwrap()
}

/// Writes a render of option number `o` for `nef` under the store at `cache`, made from
/// today's inputs, with the manifest key `key` and `fit_file` as its fit view, and returns it
/// as the launch check finds it.
fn write_render(cache: &Path, nef: &Path, o: usize, key: &str, fit_file: &str) -> Fresh {
    let (option, _) = OPTIONS[o];
    let n = nef.file_name().unwrap();
    let render = cache.join("looks").join(n).join(option);
    fs::create_dir_all(&render).unwrap();
    fs::write(
        render.join("fit.jpg"),
        format!("fit of {} {option}", n.to_string_lossy()),
    )
    .unwrap();
    fs::write(render.join("profile.pp3"), RENDERED).unwrap();
    let inputs = inputs_of(nef, o);
    let manifest = Manifest {
        key: key.to_owned(),
        inputs_key: Some(inputs.key().to_owned()),
        option: name(option),
        pp3_blake3: pp3::digest(RENDERED),
        full_w: 30,
        full_h: 20,
        fit: View {
            file: fit_file.into(),
            w: 30,
            h: 20,
        },
        regions: vec![],
    };
    fs::write(
        render.join("manifest.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    match LooksStore::new(cache)
        .standing(Against::Inputs(&inputs))
        .unwrap()
    {
        Standing::Fresh(fresh) => fresh,
        other => panic!("a render made from today's inputs is not fresh: {other:?}"),
    }
}

/// A shoot of NEF stand-ins `DSC_<n>.NEF` with a render of every option for the first
/// `rendered` of them.
struct Shoot {
    dir: tempfile::TempDir,
    session: Session,
    entries: Vec<Entry>,
    renders: HashMap<(ShotIdx, OptionIdx), Fresh>,
}

/// What a shoot has on disk beyond renders and NEFs.
#[derive(Default, Clone, Copy)]
struct Quirks {
    /// The first shot's pp3, written before the session reads its state.
    first_pp3: Option<&'static str>,
    /// The first shot's Film render names a file outside the looks directory.
    film_escapes: bool,
    /// Render exactly these shots instead of the first `rendered`.
    only: Option<&'static [usize]>,
}

fn shoot(ratings: &[u8], rendered: usize) -> Shoot {
    shoot_with(ratings, rendered, Quirks::default())
}

fn shoot_with(ratings: &[u8], rendered: usize, quirks: Quirks) -> Shoot {
    shoot_resolving(ratings, rendered, quirks, |rt, sample, work| {
        DefaultProfiles::new(rt, sample, work)
    })
}

/// `shoot_with` over default profiles made by `defaults(rt, a NEF, a work directory)`.
fn shoot_resolving(
    ratings: &[u8],
    rendered: usize,
    quirks: Quirks,
    defaults: impl FnOnce(&RtOptions, PathBuf, PathBuf) -> DefaultProfiles,
) -> Shoot {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("secret.txt"), "secret").unwrap();
    let cache = dir.path().join(".rawpilot");
    let rt = RtOptions {
        sidecar_style: SidecarStyle::Ext,
        rank_source: RankSource::ProcParams,
        custom_profile_builder: false,
        config_dir: dir.path().to_owned(),
    };
    let nef_names: Vec<String> = (0..ratings.len()).map(|n| format!("DSC_{n}.NEF")).collect();
    for n in &nef_names {
        fs::write(dir.path().join(n), b"x").unwrap();
    }
    if let Some(text) = quirks.first_pp3 {
        fs::write(pp3::path_for(&dir.path().join(&nef_names[0])), text).unwrap();
    }
    let defaults = defaults(&rt, dir.path().join(&nef_names[0]), cache.join("t"));
    let folder = FolderLock::acquire(dir.path(), Run::Review).unwrap();
    let session = Session::new(LooksStore::new(&cache), rt, defaults, &folder);
    let mut entries = Vec::new();
    let mut renders = HashMap::new();
    for (i, (n, &rating)) in nef_names.iter().zip(ratings).enumerate() {
        let nef = dir.path().join(n);
        if quirks.only.map_or(i < rendered, |only| only.contains(&i)) {
            for (o, (option, _)) in OPTIONS.iter().enumerate() {
                // Four levels up from the option directory is the shoot's folder.
                let fit_file = if quirks.film_escapes && i == 0 && *option == "Film" {
                    "../../../../secret.txt"
                } else {
                    "fit.jpg"
                };
                let rendered = write_render(&cache, &nef, o, &format!("key{i}{o}"), fit_file);
                renders.insert(
                    (Idx::new(i, ratings.len()).unwrap(), Idx::new(o, 3).unwrap()),
                    rendered,
                );
            }
        }
        entries.push(Entry {
            shot: shot_at(nef.clone()),
            rating: Stars::new(rating),
            pp3: session.pp3_state(&nef).unwrap(),
        });
    }
    fs::create_dir_all(cache.join("looks")).unwrap();
    Shoot {
        dir,
        session,
        entries,
        renders,
    }
}

impl Shoot {
    fn model(self, filter: u8) -> (Model, Session, tempfile::TempDir) {
        self.model_resuming(filter, None)
    }

    /// The model as a launch that remembers `resume` makes it.
    fn model_resuming(
        self,
        filter: u8,
        resume: Option<usize>,
    ) -> (Model, Session, tempfile::TempDir) {
        let count = self.entries.len();
        let model = Model::new(
            self.entries,
            look_options(),
            self.renders,
            Filter::at_least(Stars::new(filter)),
            SCENE_GAP_S,
            resume.map(|n| shot(n, count)),
        );
        (model, self.session, self.dir)
    }

    /// Capture times, in seconds from the first shot, of the entries in order.
    fn taken_at(mut self, seconds: &[i64]) -> Shoot {
        assert_eq!(seconds.len(), self.entries.len());
        let base = DateTime::parse_from_rfc3339("2026-01-01T10:00:00+00:00").unwrap();
        for (entry, &s) in self.entries.iter_mut().zip(seconds) {
            entry.shot.taken_at = base + chrono::Duration::seconds(s);
        }
        self
    }
}

fn ready(model: &mut Model) {
    model.profile_resolved();
    assert!(model.next_batch(8).is_none(), "everything is rendered");
}

fn shot(n: usize, of: usize) -> ShotIdx {
    Idx::new(n, of).unwrap()
}

fn option(n: usize) -> OptionIdx {
    Idx::new(n, OPTIONS.len()).unwrap()
}

/// `action` on shot `n` of a shoot of `of` shots.
fn on(n: usize, of: usize, action: Action) -> Command {
    Command::On(shot(n, of), action)
}

// The server.

struct Site {
    app: axum::Router,
    review: Arc<Review>,
    dir: tempfile::TempDir,
}

fn site(ratings: &[u8], rendered: usize) -> Site {
    site_with(ratings, rendered, Quirks::default())
}

fn site_with(ratings: &[u8], rendered: usize, quirks: Quirks) -> Site {
    site_over(
        shoot_with(ratings, rendered, quirks),
        rendered == ratings.len(),
    )
}

/// A site over `shoot` at the filter "all"; `ready` is whether every render is on disk.
fn site_over(shoot: Shoot, ready_to_work: bool) -> Site {
    let (mut model, session, dir) = shoot.model(0);
    if ready_to_work {
        ready(&mut model);
    }
    let cfg = Config::load(None).unwrap();
    let looks_root = LooksRoot::open(&dir.path().join(".rawpilot/looks")).unwrap();
    let review = Arc::new(Review::new(
        model,
        session,
        Renderer {
            params: RenderParams::of(&cfg.review),
            jobs_max: 1,
            cache_root: dir.path().join(".rawpilot"),
        },
        keymap(),
        Layout::of(&cfg.review),
        looks_root,
        Place::under(&dir.path().join(".rawpilot")),
    ));
    let app = router(App::new(Arc::clone(&review), Token::from_text(TOKEN), PORT));
    Site { app, review, dir }
}

fn signed_in(method: &str, uri: &str) -> axum::http::request::Builder {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("host", HOST)
        .header("cookie", format!("rawpilot={TOKEN}"))
}

async fn send(site: &Site, request: Request<Body>) -> Response {
    site.app.clone().oneshot(request).await.unwrap()
}

async fn text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 22)
        .await
        .unwrap();
    String::from_utf8_lossy(&bytes).into_owned()
}

async fn get(site: &Site, uri: &str) -> Response {
    send(site, signed_in("GET", uri).body(Body::empty()).unwrap()).await
}

async fn post(site: &Site, cmd: &str) -> Response {
    send(
        site,
        signed_in("POST", "/act")
            .header("origin", ORIGIN)
            .header("content-type", "application/x-www-form-urlencoded")
            .body(Body::from(format!("cmd={cmd}")))
            .unwrap(),
    )
    .await
}

/// Mutation: let the guard pass a request without checking the cookie (`Verdict::Admitted`
/// for `NoSession`): every row of the first test goes red.
#[tokio::test]
async fn no_route_answers_without_the_cookie() {
    let site = site(&[3, 3], 2);
    for (method, uri) in [
        ("GET", "/"),
        ("GET", "/app"),
        ("GET", "/img/0/0/fit"),
        ("GET", "/static/app.js"),
        ("GET", "/static/app.css"),
        ("GET", "/static/layout.css"),
        ("POST", "/act"),
    ] {
        for cookie in [None, Some("rawpilot=wrong"), Some("rawpilot=")] {
            let mut request = Request::builder()
                .method(method)
                .uri(uri)
                .header("host", HOST)
                .header("origin", ORIGIN);
            if let Some(cookie) = cookie {
                request = request.header("cookie", cookie);
            }
            let response = send(&site, request.body(Body::empty()).unwrap()).await;
            assert_eq!(
                response.status(),
                StatusCode::UNAUTHORIZED,
                "{method} {uri} {cookie:?}"
            );
        }
    }
}

/// Mutation: make the sign-in compare only a prefix, or skip `Set-Cookie`.
#[tokio::test]
async fn the_sign_in_url_sets_the_cookie_and_a_wrong_token_does_not() {
    let site = site(&[3], 1);
    let host = |uri: &str| {
        Request::builder()
            .uri(uri)
            .header("host", HOST)
            .body(Body::empty())
            .unwrap()
    };
    let response = send(&site, host(&format!("/?t={TOKEN}"))).await;
    assert_eq!(response.status(), StatusCode::SEE_OTHER);
    assert_eq!(response.headers()["location"], "/");
    let cookie = response.headers()["set-cookie"]
        .to_str()
        .unwrap()
        .to_owned();
    assert!(
        cookie.starts_with(&format!("rawpilot={TOKEN};")) && cookie.contains("HttpOnly"),
        "{cookie}"
    );
    for wrong in ["/?t=tok3", "/?t=tok3nx", "/?t=", "/?t=TOK3N"] {
        assert_eq!(
            send(&site, host(wrong)).await.status(),
            StatusCode::UNAUTHORIZED,
            "{wrong}"
        );
    }
    // The cookie it set opens the page.
    let page = text(get(&site, "/").await).await;
    assert!(page.contains(r#"id="app""#));
}

/// Mutation: drop the Origin check for writes: the first row answers 200.
#[tokio::test]
async fn a_change_from_another_origin_is_refused() {
    let site = site(&[3, 3], 2);
    let request = signed_in("POST", "/act")
        .header("origin", "http://evil.example")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from("cmd=next"))
        .unwrap();
    assert_eq!(send(&site, request).await.status(), StatusCode::FORBIDDEN);
    let request = signed_in("POST", "/act")
        .header("content-type", "application/x-www-form-urlencoded")
        .body(Body::from("cmd=next"))
        .unwrap();
    assert_eq!(send(&site, request).await.status(), StatusCode::FORBIDDEN);
}

/// Mutation: build `Idx` with `n <= count` (or skip the digits check): the 2-shot, 3-option
/// bounds below accept `goto:2`, `view:3` or `rate:6`.
#[tokio::test]
async fn requests_with_bad_indices_options_or_ratings_are_rejected() {
    let site = site(&[3, 3], 2);
    for uri in [
        "/img/2/0/fit",
        "/img/0/3/fit",
        "/img/-1/0/fit",
        "/img/0/0/fit.jpg",
        "/img/0/0/..",
        "/img/x/0/fit",
        "/img/0%2F..%2F/0/fit",
    ] {
        let status = get(&site, uri).await.status();
        assert!(
            status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
            "{uri} -> {status}"
        );
        assert_ne!(status, StatusCode::OK, "{uri}");
    }
    assert_eq!(
        get(&site, "/img/0/0/r0").await.status(),
        StatusCode::NOT_FOUND
    );
    for cmd in [
        "goto:2",
        "view:0:3",
        "view:2:0",
        "rate:0:6",
        "rate:0:-1",
        "rate:0:",
        "rate:5",
        "accept",
        "nonsense",
        "goto:1x",
        "x",
    ] {
        assert_eq!(
            post(&site, cmd).await.status(),
            StatusCode::BAD_REQUEST,
            "{cmd}"
        );
    }
    for cmd in ["goto:1", "view:1:2", "rate:1:5", "next"] {
        assert_eq!(post(&site, cmd).await.status(), StatusCode::OK, "{cmd}");
    }
}

/// Mutation: drop the `starts_with` check in `LooksRoot::read`: the second request returns
/// the secret.
#[tokio::test]
async fn a_picture_is_served_only_from_below_the_looks_directory() {
    let site = site(&[3], 1);
    let fit = get(&site, "/img/0/0/fit").await;
    assert_eq!(fit.status(), StatusCode::OK);
    assert_eq!(fit.headers()["content-type"], "image/jpeg");
    assert_eq!(text(fit).await, "fit of DSC_0.NEF Camera");

    // A render whose manifest names a file outside the looks directory: the shoot's own
    // `secret.txt`, which exists, so the store accepts the manifest.
    let site = site_with(
        &[3],
        1,
        Quirks {
            film_escapes: true,
            ..Quirks::default()
        },
    );
    let response = get(&site, "/img/0/1/fit").await;
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert!(!text(response).await.contains("secret"));
}

/// Goal: the page carries the shot and option it shows (the script names them in what it
/// sends), and the server refuses a rating or choice aimed at another shot. Method: the
/// router, a second "tab" that still holds the page of shot 0 after the first moved to
/// shot 1. Mutation: drop `data-shot` from the page (first assertion), or the
/// `current != Some(shot)` arm of `browse` (the rating lands on shot 0).
#[tokio::test]
async fn the_page_names_its_shot_and_a_stale_page_cannot_rate_another() {
    let site = site(&[3, 3], 2);
    let page = text(get(&site, "/app").await).await;
    assert!(
        page.contains(r#"data-shot="0""#) && page.contains(r#"data-viewed="0""#),
        "{page}"
    );

    let page = text(post(&site, "view:0:2").await).await;
    assert!(page.contains(r#"data-viewed="2""#), "{page}");
    post(&site, "next").await;
    let page = text(get(&site, "/app").await).await;
    assert!(page.contains(r#"data-shot="1""#), "{page}");

    // The second tab, still on shot 0, rates it.
    let page = text(post(&site, "rate:0:5").await).await;
    assert!(
        page.contains("That page was out of date; here is the current one."),
        "{page}"
    );
    let model = site.review.model();
    assert_eq!(model.entry(shot(0, 2)).rating, Stars::new(3));
    assert_eq!(model.entry(shot(1, 2)).rating, Stars::new(3));
    assert!(!pp3::path_for(&site.dir.path().join("DSC_0.NEF")).exists());
}

/// Goal: pictures stay cacheable for an hour, and a render that replaces another is a new
/// URL, so the browser cannot show the old picture for it. Method: the page of a model whose
/// render is swapped for one with another key, and the picture route asked with the query.
/// Mutation: drop `?k=` from `images::url`: the first assertions go red.
#[tokio::test]
async fn a_new_render_is_a_new_picture_url_and_pictures_stay_cacheable() {
    let (mut model, _session, dir) = shoot(&[3], 1).model(3);
    assert!(draw(&model).contains(r#"src="/img/0/0/fit?k=key00""#));

    let again = write_render(
        &dir.path().join(".rawpilot"),
        &dir.path().join("DSC_0.NEF"),
        0,
        "key00b",
        "fit.jpg",
    );
    model.batch_done(vec![((shot(0, 1), option(0)), Ok(again))]);
    let page = draw(&model);
    assert!(page.contains(r#"src="/img/0/0/fit?k=key00b""#), "{page}");
    assert!(!page.contains(r#"src="/img/0/0/fit?k=key00""#), "{page}");

    let site = site(&[3], 1);
    let picture = get(&site, "/img/0/0/fit?k=key00").await;
    assert_eq!(picture.status(), StatusCode::OK);
    assert_eq!(picture.headers()["cache-control"], "private, max-age=3600");
}

// The phases.

fn draw(model: &Model) -> String {
    let layout = Layout::of(&Config::load(None).unwrap().review);
    super::pages::app(model, &keymap(), &layout).into_string()
}

/// The pairs of `batch` as the shot numbers they cover, once each, in order.
fn shots_of(batch: &[(ShotIdx, OptionIdx)]) -> Vec<usize> {
    let mut shots: Vec<usize> = batch.iter().map(|(s, _)| s.get()).collect();
    shots.dedup();
    shots
}

/// Goal: the working view opens as soon as one shot the filter admits is rendered, the rest
/// join the strip as their renders land, and nothing the photographer is looking at moves.
/// Method: a model of three shots at the filter ≥3 with only shot 0 on disk, driven through
/// the worker's calls (`next_batch`, `batch_done`) and drawn at each step. Mutation: make
/// `Model::stage` wait for `missing().is_empty()` (the view opened only when every render
/// was in): the first `working` assertion goes red.
#[test]
fn the_working_view_opens_with_the_first_rendered_shot_and_gains_the_rest() {
    let (mut model, session, _dir) = shoot(&[3, 3, 3], 1).model(3);
    let disk = FakeDisk::new(Choose::Applied);

    let page = draw(&model);
    assert!(page.contains(r#"data-stage="working""#), "{page}");
    assert!(page.contains(r#"data-view="working""#));
    assert!(
        page.contains("1 / 1"),
        "only shot 0 is in the strip: {page}"
    );
    assert!(
        page.contains("preparing RawTherapee&#39;s default profile")
            || page.contains("preparing RawTherapee's default profile"),
        "{page}"
    );
    assert!(page.contains("data-polling"), "{page}");

    model.profile_resolved();
    let batch = model.next_batch(8).expect("shots 1 and 2 have no render");
    assert_eq!(shots_of(&batch), [1, 2]);
    // The photographer is looking at Gentle on shot 0 when the renders land.
    model.apply(on(0, 3, Action::View(option(2))), &disk);
    let page = draw(&model);
    assert!(page.contains("rendering 0/6"), "{page}");
    assert!(page.contains(r#"data-stage="working""#));

    let donor = model.render(shot(0, 3), option(0)).unwrap().clone();
    let first_of_shot_1 = batch.iter().filter(|(s, _)| s.get() == 1).cloned();
    model.batch_done(first_of_shot_1.map(|p| (p, Ok(donor.clone()))).collect());
    let page = draw(&model);
    assert!(page.contains("1 / 2"), "shot 1 joined the strip: {page}");
    assert_eq!(model.current(), Some(shot(0, 3)), "still on the same shot");
    assert_eq!(model.viewing().shown, option(2), "still on the same option");

    let rest = model.next_batch(8).expect("shot 2 remains");
    model.batch_done(rest.into_iter().map(|p| (p, Ok(donor.clone()))).collect());
    assert!(model.next_batch(8).is_none());
    let page = draw(&model);
    assert!(
        page.contains("1 / 3") && page.contains(r#"id="activity""#),
        "{page}"
    );
    assert!(
        page.contains(">Ready<") && !page.contains("data-polling"),
        "{page}"
    );
    let _ = session;
}

/// Goal: the full progress page shows only while the filter admits shots and none is
/// rendered, and gives way to the working view with the first. Method: a model with nothing
/// on disk. Mutation: let `Model::stage` answer `Waiting` whenever renders are owed: the
/// last assertion goes red.
#[test]
fn the_progress_page_shows_only_until_the_first_shot_is_rendered() {
    let (mut model, _session, _dir) = shoot(&[3, 3], 0).model(3);
    let page = draw(&model);
    assert!(page.contains(r#"data-stage="waiting""#), "{page}");
    assert!(page.contains("Resolving RawTherapee"), "{page}");
    assert!(!page.contains(r#"data-view="working""#));

    model.profile_resolved();
    let batch = model.next_batch(3).unwrap();
    assert_eq!(shots_of(&batch), [0]);
    let page = draw(&model);
    assert!(page.contains("Rendering 0/6"), "{page}");
    assert!(page.contains("<progress"), "{page}");

    // The first shot's three options land: the view opens on it.
    let donor_dir = shoot(&[3], 1).model(0);
    let donor = donor_dir.0.render(shot(0, 1), option(0)).unwrap().clone();
    model.batch_done(batch.into_iter().map(|p| (p, Ok(donor.clone()))).collect());
    let page = draw(&model);
    assert!(page.contains(r#"data-stage="working""#), "{page}");
    assert!(
        page.contains("1 / 1") && page.contains("rendering 0/6"),
        "{page}"
    );
    assert_eq!(model.current(), Some(shot(0, 2)));
}

/// Goal: a failure to make renders is a page of its own only when there is nothing to show,
/// and that page still lets the photographer change the filter. Method: the worker reports
/// the profile unusable. Mutation: make `Stage` answer `Broken` whenever the renderer is
/// stopped: the second half goes red.
#[test]
fn unavailable_renders_stop_the_page_only_when_nothing_is_rendered() {
    let (mut model, session, _dir) = shoot(&[3, 5], 0).model(3);
    model.renders_unavailable("no rawtherapee-cli".to_owned());
    let page = draw(&model);
    assert!(page.contains(r#"data-stage="broken""#) && page.contains("no rawtherapee-cli"));
    assert!(
        page.contains(r#"data-cmd="filter:up""#),
        "the chip is there: {page}"
    );
    assert_eq!(model.context(), super::keys::Context::Wait);
    assert!(
        !model.claim_worker(),
        "no worker is started for renders that cannot be made"
    );
    model.apply(Command::StepFilter(FilterWay::Up), &session);
    assert_eq!(model.filter().label(), "≥4★");

    let (mut model, _session, _dir) = shoot(&[3, 3], 1).model(3);
    model.renders_unavailable("no rawtherapee-cli".to_owned());
    let page = draw(&model);
    assert!(page.contains(r#"data-stage="working""#), "{page}");
    assert!(
        page.contains("rendering stopped") && page.contains("no rawtherapee-cli"),
        "{page}"
    );
}

/// Goal: the worker renders the shots nearest the photographer first, after and before
/// alternately, and follows the photographer when they move. Method: seven shots, shots 1
/// and 5 on disk, the batch the worker would be given before and after a move to shot 5.
/// Mutation: order by shot number: the first row goes red.
#[test]
fn renders_go_outward_from_the_shot_on_screen() {
    let (mut model, session, _dir) = shoot_with(
        &[3; 7],
        0,
        Quirks {
            only: Some(&[1, 5]),
            ..Quirks::default()
        },
    )
    .model(3);
    model.profile_resolved();
    assert_eq!(model.current(), Some(shot(1, 7)));
    assert_eq!(shots_of(&model.next_batch(100).unwrap()), [2, 0, 3, 4, 6]);

    model.apply(on(1, 7, Action::View(option(0))), &session);
    model.apply(Command::Goto(shot(5, 7)), &session);
    assert_eq!(shots_of(&model.next_batch(100).unwrap()), [6, 4, 3, 2, 0]);
    // The options of one shot stay together, in the configured order.
    let batch = model.next_batch(3).unwrap();
    assert_eq!(
        batch.iter().map(|(_, o)| o.get()).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

/// Goal: `f` and `F` step one star and stop at the ends: a press at the strictest filter
/// does not turn into every shot of the card. Method: every step written out by hand from
/// the specification all -> 2 -> 3 -> 4 -> 5, and the model's answer at each end.
/// Mutation: wrap `Filter::step` at either end: the end rows go red.
#[test]
fn the_filter_steps_one_star_up_or_down_and_stops_at_the_ends() {
    let stepped = |from: u8, way| {
        Filter::at_least(Stars::new(from))
            .step(way)
            .map(Filter::label)
    };
    for (from, up, down) in [
        (0, Some("≥2★"), None),
        (2, Some("≥3★"), Some("all")),
        (3, Some("≥4★"), Some("≥2★")),
        (4, Some("≥5★"), Some("≥3★")),
        (5, None, Some("≥4★")),
        (1, Some("≥2★"), Some("all")),
    ] {
        assert_eq!(
            stepped(from, FilterWay::Up).as_deref(),
            up,
            "up from {from}"
        );
        assert_eq!(
            stepped(from, FilterWay::Down).as_deref(),
            down,
            "down from {from}"
        );
    }

    let (mut model, session, _dir) = shoot(&[5, 3], 2).model(5);
    model.apply(Command::StepFilter(FilterWay::Up), &session);
    assert_eq!(model.filter().label(), "≥5★");
    assert!(matches!(model.notice(), Some(Notice::Hint(_))));
    let (mut model, session, _dir) = shoot(&[5, 3], 2).model(0);
    model.apply(Command::StepFilter(FilterWay::Down), &session);
    assert_eq!(model.filter().label(), "all");
    assert!(matches!(model.notice(), Some(Notice::Hint(_))));
}

/// Goal: widening the filter to shots that are not rendered keeps the working view, with
/// the new shots joining as they render, and the worker is claimed once. Method: shot 1 is
/// rated one star and unrendered; the filter is widened to "all". Mutation: let
/// `step_filter` forget to `snap`, or drop the claim: the last two assertions go red.
#[test]
fn a_wider_filter_keeps_the_working_view_while_the_new_shots_render() {
    let (mut model, session, _dir) = shoot(&[3, 1], 1).model(3);
    ready(&mut model);
    model.apply(Command::StepFilter(FilterWay::Down), &session); // at least 2: shot 1 has one star
    assert_eq!(model.activity(), Activity::Idle);
    model.apply(Command::StepFilter(FilterWay::Down), &session); // all
    assert_eq!(model.stage(), Stage::Working);
    assert!(matches!(model.activity(), Activity::Rendering(_)));
    assert_eq!(model.current(), Some(shot(0, 2)));
    assert!(model.claim_worker(), "the worker is idle and has work");
    assert!(!model.claim_worker(), "one worker at a time");
}

/// Goal: while RawTherapee's default profile is being resolved the page stays live, and a
/// rating that must create a pp3 waits for it instead of holding the model. Method: the
/// profile's resolution is held on a channel by the test; the rating runs on its own thread,
/// and the page is drawn from another meanwhile. Mutation: drop the wait in `Review::apply`
/// (the rating then blocks inside the model's lock): the draw times out and the first
/// assertion goes red.
#[test]
fn a_rating_that_needs_the_default_profile_waits_for_it_with_the_page_live() {
    use std::sync::{Mutex, mpsc};
    use std::time::Duration;

    const DEFAULT_PROFILE: &str = "[General]\nRank=1\nColorLabel=0\n\n[Exposure]\nCompensation=0\n";
    let (release, gate) = mpsc::channel::<()>();
    let (started_tx, started) = mpsc::channel::<()>();
    let (gate, started_tx) = (Mutex::new(gate), Mutex::new(started_tx));
    let shoot = shoot_resolving(&[3, 3], 2, Quirks::default(), move |_, _, _| {
        DefaultProfiles::resolving_with(move || {
            started_tx.lock().unwrap().send(()).unwrap();
            gate.lock().unwrap().recv().unwrap();
            DefaultProfile::parse(DEFAULT_PROFILE.to_owned())
        })
    });
    let site = site_over(shoot, true);
    let nef = site.dir.path().join("DSC_0.NEF");
    let rating = std::thread::spawn({
        let review = Arc::clone(&site.review);
        move || review.apply(on(0, 2, Action::Rate(Stars::new(5))))
    });
    started.recv().unwrap();

    let (page_tx, page) = mpsc::channel();
    std::thread::spawn({
        let review = Arc::clone(&site.review);
        move || page_tx.send(review.draw().into_string()).unwrap()
    });
    let drawn = page.recv_timeout(Duration::from_secs(5));
    release.send(()).unwrap();
    rating.join().unwrap();

    let drawn = drawn.expect("the page was not drawn while the profile resolved");
    assert_eq!(
        drawn.matches(r#"class="star on""#).count(),
        3,
        "the rating waited: {drawn}"
    );
    assert_eq!(site.review.model().entry(shot(0, 2)).rating, Stars::new(5));
    assert!(
        fs::read_to_string(pp3::path_for(&nef))
            .unwrap()
            .contains("Rank=5")
    );
}

/// The page polls while renders are owed and says what is going on in the bar. Mutation:
/// drop `data-polling` from the page: the script would stop updating it.
#[tokio::test]
async fn the_page_polls_and_names_the_activity_while_renders_are_owed() {
    let site = site(&[3, 3], 1);
    let page = text(get(&site, "/app").await).await;
    assert!(page.contains(r#"data-stage="working""#), "{page}");
    assert!(page.contains("data-polling"), "{page}");
    assert!(
        page.contains("preparing RawTherapee&#39;s default profile")
            || page.contains("preparing RawTherapee's default profile"),
        "{page}"
    );
    let site = self::site(&[3, 3], 2);
    let page = text(get(&site, "/app").await).await;
    assert!(!page.contains("data-polling"), "{page}");
}

// Stale renders.

/// Rewrites the manifest of the render of `option` for `nef_name` under `cache`.
fn edit_manifest(
    cache: &Path,
    nef_name: &str,
    option: &str,
    edit: impl FnOnce(&mut serde_json::Value),
) {
    let path = cache
        .join("looks")
        .join(nef_name)
        .join(option)
        .join("manifest.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    edit(&mut json);
    fs::write(&path, json.to_string()).unwrap();
}

/// Goal: at launch a render counts only if it was made from today's inputs, as far as they
/// can be told without the default profile. Method: two shots with all three options
/// rendered, then one manifest records other inputs (a retuned look profile) and another has
/// none (written before they were recorded); `load_renders` is asked what it keeps.
/// Mutation: make `Fresh::check` pass an `Against::Inputs` check without comparing
/// `inputs_key`: the retuned pair is kept and the assertion goes red; treat a missing
/// `inputs_key` as fresh: the legacy pair is.
#[test]
fn launch_keeps_only_renders_made_from_todays_inputs() {
    let s = shoot(&[3, 3], 2);
    let cache = s.dir.path().join(".rawpilot");
    edit_manifest(&cache, "DSC_0.NEF", "Film", |m| {
        m["inputs_key"] = "retuned".into();
    });
    edit_manifest(&cache, "DSC_1.NEF", "Camera", |m| {
        m.as_object_mut().unwrap().remove("inputs_key");
    });

    let loaded = super::load_renders(&s.session, &s.entries, &look_options(), &render_params());

    let mut kept: Vec<(usize, usize)> = loaded.keys().map(|(s, o)| (s.get(), o.get())).collect();
    kept.sort_unstable();
    assert_eq!(kept, [(0, 0), (0, 2), (1, 1), (1, 2)]);
}

/// Goal: a render found stale is not shown, not chosen, and queued like a missing one, and
/// one that was replaced meanwhile is left alone. Method: a model of two rendered shots, the
/// photographer viewing Film on shot 0, whose Film render is then discarded. Mutation: make
/// `discard_stale` remove nothing: the first assertion goes red; drop its key comparison:
/// the replaced render is removed and the last one does.
#[test]
fn a_stale_render_is_neither_shown_nor_chosen_and_is_queued() {
    let (mut model, _session, _dir) = shoot(&[3, 3], 2).model(3);
    ready(&mut model);
    let disk = FakeDisk::new(Choose::Applied);
    model.apply(on(0, 2, Action::View(option(1))), &disk);
    assert_eq!(model.current(), Some(shot(0, 2)));

    model.discard_stale(&[((shot(0, 2), option(1)), "key01".to_owned())]);

    assert!(model.render(shot(0, 2), option(1)).is_none());
    assert_eq!(
        model.visible(),
        [shot(1, 2)],
        "shot 0 is not fully rendered any more"
    );
    assert_eq!(model.current(), Some(shot(1, 2)), "the view moved off it");
    model.apply(on(0, 2, Action::Accept(option(1))), &disk);
    assert!(disk.calls.borrow().is_empty(), "nothing was written");
    assert_eq!(
        model.next_batch(8),
        Some(vec![(shot(0, 2), option(1))]),
        "only the stale render is owed"
    );

    model.discard_stale(&[((shot(1, 2), option(0)), "not the key held".to_owned())]);
    assert!(model.render(shot(1, 2), option(0)).is_some());
}

/// Goal: once the default profile is known, the renders launch let through are compared with
/// the full keys, and one made under another default profile is dropped. Method: shot 0's
/// renders carry the keys of the profile the resolver returns, shot 1's those of another;
/// `Review::discard_stale` runs with the resolved one. Mutation: have `is_fresh_under`
/// answer `true`: shot 1 stays and the assertion goes red.
#[test]
fn renders_made_under_another_default_profile_are_dropped_once_it_is_known() {
    let current = DefaultProfile::parse(DEFAULT.to_owned()).unwrap();
    let retuned = DefaultProfile::parse(DEFAULT.replace("Rank=1", "Rank=2")).unwrap();
    let mut shoot = shoot(&[3, 3], 2);
    let cache = shoot.dir.path().join(".rawpilot");
    for (i, default) in [(0, &current), (1, &retuned)] {
        let nef = shoot.dir.path().join(format!("DSC_{i}.NEF"));
        for o in 0..OPTIONS.len() {
            let job = Job::of(inputs_of(&nef, o), default).unwrap();
            let fresh = write_render(&cache, &nef, o, job.key(), "fit.jpg");
            shoot
                .renders
                .insert((shot(i, 2), Idx::new(o, OPTIONS.len()).unwrap()), fresh);
        }
    }
    let site = site_over(shoot, false);

    site.review.discard_stale(&current);

    let mut model = site.review.model();
    for o in model.option_indices().collect::<Vec<_>>() {
        assert!(model.render(shot(0, 2), o).is_some());
        assert!(model.render(shot(1, 2), o).is_none());
    }
    model.profile_resolved();
    assert_eq!(shots_of(&model.next_batch(8).unwrap()), [1]);
}

// Choosing.

struct FakeDisk {
    calls: RefCell<Vec<String>>,
    on_choose: Cell<Choose>,
}

impl FakeDisk {
    fn new(on_choose: Choose) -> FakeDisk {
        FakeDisk {
            calls: RefCell::new(Vec::new()),
            on_choose: Cell::new(on_choose),
        }
    }
}

impl Disk for FakeDisk {
    fn choose(&self, nef: &Path, render: &Fresh, on: OnHandEdited) -> anyhow::Result<Choose> {
        let file = nef.file_name().unwrap().to_string_lossy();
        let option = &render.manifest.option;
        self.calls
            .borrow_mut()
            .push(format!("choose {file} {option} {on:?}"));
        Ok(self.on_choose.get())
    }

    fn rate(&self, nef: &Path, stars: Stars) -> anyhow::Result<()> {
        let file = nef.file_name().unwrap().to_string_lossy();
        self.calls
            .borrow_mut()
            .push(format!("rate {file} {}", stars.get()));
        Ok(())
    }

    fn pp3_state(&self, _: &Path) -> anyhow::Result<Pp3State> {
        Ok(Pp3State::Generated)
    }
}

/// Mutation: let `Accept` write when `origin` is `Arrived` (drop that arm): the first
/// assertion sees a `choose` call.
#[test]
fn enter_writes_only_an_option_that_was_picked_on_this_shot() {
    let (mut model, _session, _dir) = shoot(&[3, 3], 2).model(3);
    ready(&mut model);
    let disk = FakeDisk::new(Choose::Applied);

    model.apply(on(0, 2, Action::Accept(option(0))), &disk);
    assert!(disk.calls.borrow().is_empty(), "nothing was picked");
    assert_eq!(model.current(), Some(shot(0, 2)));

    model.apply(on(0, 2, Action::View(option(1))), &disk);
    assert!(disk.calls.borrow().is_empty(), "viewing writes nothing");
    model.apply(on(0, 2, Action::Accept(option(1))), &disk);
    assert_eq!(*disk.calls.borrow(), ["choose DSC_0.NEF Film Refuse"]);
    assert_eq!(model.current(), Some(shot(1, 2)), "and it moved on");

    // The next shot shows Film too, but arriving is not picking.
    assert_eq!(model.viewing().shown, option(1));
    model.apply(on(1, 2, Action::Accept(option(1))), &disk);
    assert_eq!(disk.calls.borrow().len(), 1);
}

const OUT_OF_DATE: &str = "That page was out of date; here is the current one.";

fn is_out_of_date(model: &Model) -> bool {
    matches!(model.notice(), Some(Notice::Hint(h)) if h == OUT_OF_DATE)
}

/// Goal: a command from a page that is not the current one (a second tab, a late reply) acts
/// on nothing. Method: the model is moved to shot 1 viewing Film, then every command that
/// writes or depends on the shot on screen is sent as the page of shot 0 or of another
/// viewed option would send it; the disk records every write asked of it.
/// Mutation: drop the `current != Some(shot)` arm of `browse`: the shot-0 rows write; drop
/// the `viewed != shown` guard of `Accept`: the Camera row writes.
#[test]
fn a_command_from_an_out_of_date_page_writes_nothing() {
    let (mut model, _session, _dir) = shoot(&[3, 3], 2).model(3);
    ready(&mut model);
    let disk = FakeDisk::new(Choose::Applied);
    model.apply(Command::Move(Move::Next), &disk);
    model.apply(on(1, 2, Action::View(option(1))), &disk);
    assert_eq!(model.current(), Some(shot(1, 2)));

    for stale in [
        // The page of shot 0.
        on(0, 2, Action::Accept(option(1))),
        on(0, 2, Action::Rate(Stars::new(5))),
        on(0, 2, Action::View(option(2))),
        on(0, 2, Action::Flip),
        // The page of shot 1 from before Film was viewed.
        on(1, 2, Action::Accept(option(0))),
    ] {
        model.apply(stale, &disk);
        assert!(is_out_of_date(&model), "{stale}");
        assert!(disk.calls.borrow().is_empty(), "{stale}");
        assert_eq!(model.current(), Some(shot(1, 2)), "{stale}");
        assert_eq!(model.viewing().shown, option(1), "{stale}");
    }

    model.apply(on(1, 2, Action::Accept(option(1))), &disk);
    assert!(!is_out_of_date(&model));
    assert_eq!(*disk.calls.borrow(), ["choose DSC_1.NEF Film Refuse"]);
}

/// Mutation: let `Flip` leave `before` unchanged (so a second flip does nothing), or let
/// `View` of the shown option forget `before`.
#[test]
fn flip_goes_back_to_the_option_viewed_before() {
    let (mut model, _session, _dir) = shoot(&[3], 1).model(3);
    ready(&mut model);
    let disk = FakeDisk::new(Choose::Applied);
    model.apply(on(0, 1, Action::Flip), &disk);
    assert_eq!(model.viewing().shown, option(0), "nothing to flip to");
    model.apply(on(0, 1, Action::View(option(2))), &disk);
    model.apply(on(0, 1, Action::Flip), &disk);
    assert_eq!(model.viewing().shown, option(0));
    model.apply(on(0, 1, Action::Flip), &disk);
    assert_eq!(model.viewing().shown, option(2));
}

/// Mutation: pass `OnHandEdited::Replace` on the first call, or drop the `Confirming`
/// guard that ignores other commands.
#[test]
fn a_hand_edited_pp3_asks_before_it_is_replaced() {
    let (mut model, _session, _dir) = shoot(&[3, 3], 2).model(3);
    ready(&mut model);
    let disk = FakeDisk::new(Choose::NeedsConfirmation);
    model.apply(on(0, 2, Action::View(option(2))), &disk);
    model.apply(on(0, 2, Action::Accept(option(2))), &disk);
    assert!(matches!(model.mode(), Mode::Confirming(c) if c.option == option(2)));
    assert_eq!(*disk.calls.borrow(), ["choose DSC_0.NEF Gentle Refuse"]);

    model.apply(Command::Move(Move::Next), &disk);
    assert_eq!(
        model.current(),
        Some(shot(0, 2)),
        "the question blocks navigation"
    );

    // An answer from a page showing another shot, or this shot's question about another
    // option (a second tab left on an older question), is not an answer to this question.
    disk.on_choose.set(Choose::Applied);
    model.apply(on(1, 2, Action::Yes(option(2))), &disk);
    assert!(is_out_of_date(&model));
    model.apply(on(0, 2, Action::Yes(option(1))), &disk);
    assert!(is_out_of_date(&model));
    model.apply(on(0, 2, Action::No(option(1))), &disk);
    assert!(is_out_of_date(&model));
    assert!(matches!(model.mode(), Mode::Confirming(_)));
    assert_eq!(disk.calls.borrow().len(), 1);

    model.apply(on(0, 2, Action::Yes(option(2))), &disk);
    assert_eq!(
        disk.calls.borrow().last().unwrap(),
        "choose DSC_0.NEF Gentle Replace"
    );
    assert_eq!(model.mode(), Mode::Browse);
    assert_eq!(model.current(), Some(shot(1, 2)));
}

/// The real session, over HTTP: the page after the refused choice carries the question, the
/// pp3 is untouched until yes, and no answers keep it. Mutation: write on the first
/// `accept`, or draw the dialog without the option's name.
#[tokio::test]
async fn the_page_asks_in_place_and_the_pp3_changes_only_on_yes() {
    // A pp3 rawpilot has no record of writing is the photographer's.
    let edited = "[General]\nRank=3\nColorLabel=0\n\n[Exposure]\nCompensation=0.00\nContrast=7\n";
    let site = site_with(
        &[3, 3],
        2,
        Quirks {
            first_pp3: Some(edited),
            ..Quirks::default()
        },
    );
    let nef = site.dir.path().join("DSC_0.NEF");
    assert_eq!(
        site.review.model().entry(shot(0, 2)).pp3,
        Pp3State::HandEdited
    );
    post(&site, "goto:0").await;
    post(&site, "view:0:2").await;
    let page = text(post(&site, "accept:0:2").await).await;
    assert!(
        page.contains("was edited in RawTherapee; replace it with <b>Gentle</b>?"),
        "{page}"
    );
    assert!(page.contains(r#"data-cmd="yes:0:2""#) && page.contains(r#"data-cmd="no:0:2""#));
    assert!(
        page.contains("&quot;y&quot;:&quot;yes:{shot}:{viewed}&quot;")
            && page.contains(r#"data-viewed="2""#)
            && !page.contains("&quot;j&quot;"),
        "{page}"
    );
    assert_eq!(fs::read_to_string(pp3::path_for(&nef)).unwrap(), edited);

    let page = text(post(&site, "no:0:2").await).await;
    assert!(!page.contains("alertdialog"));
    assert_eq!(fs::read_to_string(pp3::path_for(&nef)).unwrap(), edited);

    post(&site, "view:0:2").await;
    post(&site, "accept:0:2").await;
    let page = text(post(&site, "yes:0:2").await).await;
    assert!(!page.contains("alertdialog"));
    let written = fs::read_to_string(pp3::path_for(&nef)).unwrap();
    assert!(written.contains("Compensation=1.00") && !written.contains("Contrast=7"));
    assert!(written.contains("Rank=3"), "the rating is kept: {written}");
}

// Moving about.

/// Applies `way` and returns the shot on screen afterwards, as a number.
fn went(model: &mut Model, way: Move) -> usize {
    model.apply(Command::Move(way), &FakeDisk::new(Choose::Applied));
    model.current().unwrap().get()
}

fn hint_of(model: &Model) -> Option<&str> {
    match model.notice() {
        Some(Notice::Hint(h)) => Some(h),
        _ => None,
    }
}

/// Goal: `]` and `[` follow the pipeline's scenes, among the shots the filter shows.
/// Method: eight shots in four scenes (gaps over 90 s after the 3rd, 5th and 6th shot),
/// walked forward and back; then with ratings that hide a whole scene and the first shot of
/// another. Expected shots are derived by hand from the times.
/// Mutation: `>` to `>=` in `NextScene`: the first step lands on shot 1; make `PreviousScene`
/// always answer `start`: the step from shot 3 stays on 3.
#[test]
fn scene_keys_go_to_the_first_visible_shot_of_a_scene() {
    let times = [0, 10, 20, 200, 210, 400, 500, 510];
    let (mut model, _session, _dir) = shoot(&[3; 8], 8).taken_at(&times).model(0);
    ready(&mut model);
    assert_eq!(model.scene_of(shot(4, 8)), (2, 4));
    assert_eq!(model.scene_of(shot(7, 8)), (4, 4));

    let forward: Vec<usize> = (0..4).map(|_| went(&mut model, Move::NextScene)).collect();
    assert_eq!(forward, [3, 5, 6, 6]);
    assert_eq!(
        hint_of(&model),
        Some("No later scene has a shot at this filter.")
    );

    model.apply(Command::Goto(shot(7, 8)), &FakeDisk::new(Choose::Applied));
    let back: Vec<usize> = (0..5)
        .map(|_| went(&mut model, Move::PreviousScene))
        .collect();
    assert_eq!(back, [6, 5, 3, 0, 0]);
    assert_eq!(
        hint_of(&model),
        Some("No earlier scene has a shot at this filter.")
    );

    // Scene {3, 4} loses its first shot and scene {5} all of it.
    let ratings = [3, 3, 3, 1, 3, 1, 3, 3];
    let (mut model, _session, _dir) = shoot(&ratings, 8).taken_at(&times).model(2);
    ready(&mut model);
    let forward: Vec<usize> = (0..3).map(|_| went(&mut model, Move::NextScene)).collect();
    assert_eq!(forward, [4, 6, 6]);
    model.apply(Command::Goto(shot(7, 8)), &FakeDisk::new(Choose::Applied));
    let back: Vec<usize> = (0..4)
        .map(|_| went(&mut model, Move::PreviousScene))
        .collect();
    assert_eq!(back, [6, 4, 0, 0]);
}

fn chosen_film() -> Pp3State {
    Pp3State::Chosen {
        option: name("Film"),
    }
}

/// Goal: `J` and `K` find shots with no look chosen; an interrupted or hand-edited pp3 is
/// not a choice. Method: six shots whose pp3 states are set by hand.
/// Mutation: `is_none` to `is_some` in the undecided arms: every step goes red; count
/// `Pp3State::Interrupted` as chosen in `Model::chosen`: the walk skips shot 2.
#[test]
fn undecided_keys_skip_the_shots_with_a_look_chosen() {
    let mut shoot = shoot(&[3; 6], 6);
    shoot.entries[1].pp3 = chosen_film();
    shoot.entries[2].pp3 = Pp3State::Interrupted {
        option: name("Film"),
    };
    shoot.entries[3].pp3 = Pp3State::HandEdited;
    shoot.entries[4].pp3 = chosen_film();
    let (mut model, _session, _dir) = shoot.model(0);
    ready(&mut model);

    let forward: Vec<usize> = (0..4)
        .map(|_| went(&mut model, Move::NextUndecided))
        .collect();
    assert_eq!(forward, [2, 3, 5, 5]);
    assert_eq!(
        hint_of(&model),
        Some("No shot after this one is undecided.")
    );
    let back: Vec<usize> = (0..4)
        .map(|_| went(&mut model, Move::PreviousUndecided))
        .collect();
    assert_eq!(back, [3, 2, 0, 0]);
    assert_eq!(
        hint_of(&model),
        Some("No shot before this one is undecided.")
    );
}

/// Goal: Home, End, PageDown and PageUp count the shots the filter shows, not frame
/// numbers, and stop at the ends. Method: 25 shots of which the odd ones are hidden, so the
/// 13 visible are 0, 2, ..., 24.
/// Mutation: count from all shots (index `at + PAGE` of the frame list): 10 instead of 20;
/// drop the `.min(visible.len() - 1)`: the third step panics; `PAGE` 10 to 9: 18.
#[test]
fn home_end_and_pages_step_through_the_visible_shots() {
    let ratings: Vec<u8> = (0..25).map(|n| if n % 2 == 0 { 3 } else { 1 }).collect();
    let (mut model, _session, _dir) = shoot(&ratings, 25).model(2);
    ready(&mut model);

    let pages: Vec<usize> = (0..3)
        .map(|_| went(&mut model, Move::PageForward))
        .collect();
    assert_eq!(pages, [20, 24, 24]);
    assert_eq!(hint_of(&model), Some("That is the last shot."));
    let pages: Vec<usize> = (0..4).map(|_| went(&mut model, Move::PageBack)).collect();
    assert_eq!(pages, [4, 0, 0, 0]);
    assert_eq!(hint_of(&model), Some("That is the first shot."));

    assert_eq!(went(&mut model, Move::Last), 24);
    assert_eq!(went(&mut model, Move::Last), 24);
    assert_eq!(hint_of(&model), Some("That is the last shot."));
    assert_eq!(went(&mut model, Move::First), 0);
    assert_eq!(went(&mut model, Move::First), 0);
    assert_eq!(hint_of(&model), Some("That is the first shot."));
}

/// Moves are for browsing: with a question open, none of them moves. Mutation: let `apply`
/// route `Command::Move` to `browse` whatever the mode: the first row goes red.
#[test]
fn a_question_blocks_every_move() {
    let (mut model, _session, _dir) = shoot(&[3, 3, 3], 3).model(0);
    ready(&mut model);
    let disk = FakeDisk::new(Choose::NeedsConfirmation);
    model.apply(on(0, 3, Action::View(option(2))), &disk);
    model.apply(on(0, 3, Action::Accept(option(2))), &disk);
    assert!(matches!(model.mode(), Mode::Confirming(_)));
    for way in Move::ALL {
        model.apply(Command::Move(way), &disk);
        assert_eq!(model.current(), Some(shot(0, 3)), "{way:?}");
        assert_eq!(
            hint_of(&model),
            Some("Answer the question first: y or n."),
            "{way:?}"
        );
    }
}

// Resuming.

/// Goal: a launch opens where the last one stopped when the filter shows that shot, else on
/// the next shot it shows, else on the first. Method: five shots at filter 3, with the
/// remembered shot at each of those three kinds of place.
/// Mutation: ignore `resume` in `Model::new`: the first two rows go red; take `Beyond::Last`
/// for the launch: the third row opens on 3.
#[test]
fn a_launch_opens_where_the_last_one_stopped() {
    let open = |ratings: &[u8], resume: Option<usize>| {
        let (model, _session, _dir) = shoot(ratings, ratings.len()).model_resuming(3, resume);
        model.current().map(ShotIdx::get)
    };
    assert_eq!(open(&[3, 3, 3, 3, 3], Some(2)), Some(2), "shown");
    assert_eq!(
        open(&[3, 3, 1, 3, 3], Some(2)),
        Some(3),
        "hidden: the next one"
    );
    assert_eq!(
        open(&[3, 3, 3, 3, 1], Some(4)),
        Some(0),
        "hidden and last: the first"
    );
    assert_eq!(open(&[3, 3, 3, 3, 3], None), Some(0), "nothing remembered");
}

/// Goal: the shot to write down is the one on screen once it differs from what the record
/// holds; opening on a shot writes nothing. Method: moves on a model that resumed at 1.
/// Mutation: report `current` always: the first assertion goes red; never update
/// `saved_place`: the third.
#[test]
fn the_place_to_write_down_is_the_shot_on_screen_when_it_moves() {
    let (mut model, _session, _dir) = shoot(&[3, 3, 3], 3).model_resuming(0, Some(1));
    assert_eq!(model.unsaved_place(), None, "opened on it");
    went(&mut model, Move::Next);
    assert_eq!(model.unsaved_place(), Some(shot(2, 3)));
    assert_eq!(model.unsaved_place(), None, "once");
    went(&mut model, Move::Previous);
    assert_eq!(model.unsaved_place(), Some(shot(1, 3)));
}

#[test]
fn the_remembered_file_name_finds_its_shot() {
    let shoot = shoot(&[3, 3, 3], 3);
    assert_eq!(
        super::shot_named(&shoot.entries, "DSC_2.NEF"),
        Some(shot(2, 3))
    );
    assert_eq!(super::shot_named(&shoot.entries, "DSC_9.NEF"), None);
    assert_eq!(super::shot_named(&shoot.entries, "../DSC_2.NEF"), None);
}

/// Goal: the bar names the scene of the shot on screen, out of all scenes of the shoot.
/// Mutation: count scenes among the visible shots only, or from 0: the rows go red.
#[tokio::test]
async fn the_bar_shows_the_scene_on_screen() {
    let site = site_over(shoot(&[3, 3, 3], 3).taken_at(&[0, 10, 500]), true);
    let page = text(get(&site, "/app").await).await;
    assert!(
        page.contains(r#"<span class="scene""#) && page.contains("scene 1/2"),
        "{page}"
    );
    let page = text(post(&site, "scene:next").await).await;
    assert!(page.contains("scene 2/2"), "{page}");
    assert!(page.contains("DSC_2.NEF"));
}

fn place_of(site: &Site) -> Option<String> {
    Place::under(&site.dir.path().join(".rawpilot")).load()
}

/// Goal: moving in the page leaves the shot on screen in `review-state.json`; looking does
/// not; a record that cannot be written costs a log line, not the command. Method: the
/// router, then the record read back (and, for the failure, a directory in its place).
/// Mutation: skip the save in `Review::change`: the first assertion after `next` goes red.
/// The unwritable record is checked to leave the command's answer intact; no mutation of it
/// was tried.
#[tokio::test]
async fn moving_writes_the_place_down() {
    let site = site(&[3, 3, 3], 3);
    assert_eq!(
        place_of(&site),
        None,
        "opening on the first shot writes nothing"
    );
    get(&site, "/app").await;
    assert_eq!(place_of(&site), None, "looking writes nothing");

    post(&site, "next").await;
    assert_eq!(place_of(&site).as_deref(), Some("DSC_1.NEF"));
    post(&site, "last").await;
    assert_eq!(place_of(&site).as_deref(), Some("DSC_2.NEF"));
    post(&site, "view:2:1").await;
    assert_eq!(place_of(&site).as_deref(), Some("DSC_2.NEF"));

    let record = site.dir.path().join(".rawpilot/review-state.json");
    fs::remove_file(&record).unwrap();
    fs::create_dir(&record).unwrap();
    let response = post(&site, "first").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(text(response).await.contains("DSC_0.NEF"));
}
