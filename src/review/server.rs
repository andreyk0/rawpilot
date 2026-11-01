//! The HTTP side: admission first, then the routes.
//!
//! | Route | |
//! | --- | --- |
//! | `GET /?t=TOKEN` | sign-in: sets the cookie, redirects to `/` |
//! | `GET /` | the page |
//! | `GET /app` | the page's `#app` element, which the script polls and swaps |
//! | `POST /act` | form field `cmd`: a [`Command`]; answers with the new `#app` |
//! | `GET /img/<shot>/<option>/<fit\|rN>` | a render's picture, by index |
//! | `GET /static/{app.js,app.css,layout.css}` | the embedded assets |
//!
//! Every response carries a content-security policy that admits only this origin's own files.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Form, Path, Query, Request, State};
use axum::http::header::{
    CACHE_CONTROL, CONTENT_SECURITY_POLICY, CONTENT_TYPE, LOCATION, SET_COOKIE,
};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use tracing::warn;

use super::auth::{self, Intent, Names, Token, Verdict};
use super::command::{Command, Idx};
use super::images::{ImageError, ImgRef};
use super::pages;
use super::service::Review;

/// A form is one short field; this bounds what a request can make the server hold.
const BODY_BYTES_MAX: usize = 4 * 1024;

const APP_JS: &str = include_str!("app.js");
const APP_CSS: &str = include_str!("app.css");

#[derive(Clone)]
pub struct App {
    review: Arc<Review>,
    token: Arc<Token>,
    names: Arc<Names>,
}

impl App {
    pub fn new(review: Arc<Review>, token: Token, port: u16) -> App {
        App {
            review,
            token: Arc::new(token),
            names: Arc::new(Names::on_port(port)),
        }
    }
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app", get(fragment))
        .route("/act", post(act))
        .route("/img/{shot}/{option}/{img}", get(image))
        .route(
            "/static/app.js",
            get(|| async { asset("text/javascript", APP_JS) }),
        )
        .route(
            "/static/app.css",
            get(|| async { asset("text/css", APP_CSS) }),
        )
        .route("/static/layout.css", get(layout_css))
        .layer(axum::extract::DefaultBodyLimit::max(BODY_BYTES_MAX))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .with_state(app)
}

#[derive(Deserialize)]
struct SignIn {
    t: Option<String>,
}

/// Admission for every request, the sign-in included.
async fn guard(State(app): State<App>, request: Request, next: Next) -> Response {
    let url_token = (request.uri().path() == "/")
        .then(|| Query::<SignIn>::try_from_uri(request.uri()).ok()?.0.t)
        .flatten();
    let verdict = auth::admit(
        &app.names,
        &app.token,
        Intent::of(request.method()),
        url_token.as_deref(),
        request.headers(),
    );
    let response = match verdict {
        Verdict::Refused(refusal) => plain(refusal.status(), refusal.message()),
        Verdict::SignIn => sign_in(&app.token),
        Verdict::Admitted => next.run(request).await,
    };
    harden(response)
}

fn sign_in(token: &Token) -> Response {
    let mut response = StatusCode::SEE_OTHER.into_response();
    response
        .headers_mut()
        .insert(LOCATION, HeaderValue::from_static("/"));
    if let Ok(cookie) = HeaderValue::from_str(&auth::session_cookie(token)) {
        response.headers_mut().insert(SET_COOKIE, cookie);
    }
    response
}

/// The headers every response carries: pages are never cached, nothing is framed or sniffed,
/// and the page may load only its own files and talk only to its own origin. A response that
/// set its own `Cache-Control` (the pictures) keeps it.
fn harden(mut response: Response) -> Response {
    const CSP: &str = "default-src 'none'; img-src 'self'; style-src 'self'; script-src 'self'; \
        connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
    let headers = response.headers_mut();
    headers
        .entry(CACHE_CONTROL)
        .or_insert(HeaderValue::from_static("no-store"));
    headers.insert(CONTENT_SECURITY_POLICY, HeaderValue::from_static(CSP));
    headers.insert(
        HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("no-referrer"),
    );
    response
}

fn plain(status: StatusCode, line: &str) -> Response {
    (
        status,
        [(CONTENT_TYPE, "text/plain; charset=utf-8")],
        format!("{line}\n"),
    )
        .into_response()
}

fn asset(kind: &'static str, text: impl IntoResponse) -> Response {
    ([(CONTENT_TYPE, format!("{kind}; charset=utf-8"))], text).into_response()
}

async fn layout_css(State(app): State<App>) -> Response {
    asset("text/css", app.review.layout.css())
}

async fn index(State(app): State<App>) -> Response {
    pages::shell(&app.review.draw()).into_response()
}

async fn fragment(State(app): State<App>) -> Response {
    app.review.draw().into_response()
}

#[derive(Deserialize)]
struct Act {
    cmd: String,
}

/// Parses the command, applies it off the request thread, and answers with the page as it
/// now is. A text that is not a command is a 400.
async fn act(State(app): State<App>, Form(form): Form<Act>) -> Response {
    let bounds = app.review.model().bounds();
    let command = match Command::parse(&form.cmd, bounds) {
        Ok(command) => command,
        Err(error) => return plain(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let review = Arc::clone(&app.review);
    match tokio::task::spawn_blocking(move || review.apply(command)).await {
        Ok(()) => app.review.draw().into_response(),
        Err(error) => {
            warn!(%error, "a command stopped before it finished");
            plain(StatusCode::INTERNAL_SERVER_ERROR, "the command stopped")
        }
    }
}

async fn image(
    State(app): State<App>,
    Path((shot, option, img)): Path<(String, String, String)>,
) -> Response {
    let wanted = {
        let model = app.review.model();
        let bounds = model.bounds();
        let parsed = (
            Idx::parse(&shot, bounds.shots),
            Idx::parse(&option, bounds.options),
            ImgRef::parse(&img),
        );
        match parsed {
            (Ok(shot), Ok(option), Ok(img)) => model
                .render(shot, option)
                .and_then(|r| img.view(r).map(|view| r.path(view))),
            (Err(e), _, _) => return plain(StatusCode::BAD_REQUEST, &e.to_string()),
            (_, Err(e), _) => return plain(StatusCode::BAD_REQUEST, &e.to_string()),
            (_, _, Err(e)) => return plain(StatusCode::BAD_REQUEST, &e.to_string()),
        }
    };
    let Some(path) = wanted else {
        return plain(StatusCode::NOT_FOUND, "no such picture");
    };
    let root = app.review.looks_root.clone();
    match tokio::task::spawn_blocking(move || root.read(&path)).await {
        Ok(Ok(bytes)) => (
            [
                (CONTENT_TYPE, "image/jpeg"),
                (CACHE_CONTROL, "private, max-age=3600"),
            ],
            bytes,
        )
            .into_response(),
        Ok(Err(error @ ImageError::Outside(_))) => {
            warn!(%error, "a render names a file outside the looks directory");
            plain(StatusCode::FORBIDDEN, "refused")
        }
        Ok(Err(error @ ImageError::Unreadable { .. })) => {
            warn!(%error, "a picture cannot be read");
            plain(StatusCode::NOT_FOUND, "no such picture")
        }
        Err(error) => {
            warn!(%error, "a picture read stopped");
            plain(StatusCode::INTERNAL_SERVER_ERROR, "the read stopped")
        }
    }
}
