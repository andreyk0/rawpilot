//! Who may talk to the server.
//!
//! The server listens on loopback only, so the one thing a hostile page in the
//! photographer's own browser can do is send requests to it. Each launch has a token from the
//! operating system's random source. It is printed in the sign-in URL (`/?t=TOKEN`), which
//! answers with the `rawpilot` cookie; every other request, images included, needs that cookie.
//! In this order:
//!
//! 1. `Host` is one of the two loopback names with the server's port: a page reached
//!    through a rebinding DNS name carries its own name there.
//! 2. The sign-in token, or the cookie, equals the token; `SameSite=Strict` keeps the cookie
//!    off cross-site requests.
//! 3. A POST also carries an `Origin` that is the server's own.
//!
//! [`admit`] is the one function that decides, and says why it refused.

use std::fmt::Write as _;

use axum::http::header::COOKIE;
use axum::http::{HeaderMap, Method, StatusCode};
use subtle::ConstantTimeEq;

pub const COOKIE_NAME: &str = "rawpilot";

/// Bytes of randomness in a token: 256 bits.
const TOKEN_BYTES: usize = 32;

/// A per-launch secret. Equality is not offered: the only comparison is [`Token::matches`],
/// which takes the same time wherever the texts differ.
#[derive(Clone)]
pub struct Token(String);

impl Token {
    pub fn generate() -> Result<Token, TokenError> {
        let mut bytes = [0_u8; TOKEN_BYTES];
        getrandom::fill(&mut bytes).map_err(TokenError)?;
        let hex = bytes.iter().fold(String::new(), |mut hex, byte| {
            // Writing to a `String` cannot fail.
            let _ = write!(hex, "{byte:02x}");
            hex
        });
        Ok(Token(hex))
    }

    /// A token with a known value, for tests that must send it.
    #[cfg(test)]
    pub fn from_text(text: &str) -> Token {
        Token(text.to_owned())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn matches(&self, offered: &str) -> bool {
        self.0.as_bytes().ct_eq(offered.as_bytes()).into()
    }
}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Token(..)")
    }
}

#[derive(Debug, thiserror::Error)]
#[error("cannot read the system's random source: {0}")]
pub struct TokenError(getrandom::Error);

/// What the request is, as far as admission goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Read,
    Write,
}

impl Intent {
    pub fn of(method: &Method) -> Intent {
        if *method == Method::GET || *method == Method::HEAD {
            Intent::Read
        } else {
            Intent::Write
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The cookie admitted it.
    Admitted,
    /// The token in the URL admitted it: the sign-in, which the caller answers with the
    /// cookie.
    SignIn,
    Refused(Refusal),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// `Host` is missing or not a loopback name with this port.
    WrongHost,
    /// No cookie, or one that is not the token.
    NoSession,
    /// A POST without this server's own `Origin`.
    WrongOrigin,
}

impl Refusal {
    pub const fn status(self) -> StatusCode {
        match self {
            Refusal::WrongHost => StatusCode::BAD_REQUEST,
            Refusal::NoSession => StatusCode::UNAUTHORIZED,
            Refusal::WrongOrigin => StatusCode::FORBIDDEN,
        }
    }

    pub const fn message(self) -> &'static str {
        match self {
            Refusal::WrongHost => "Wrong Host header: use http://127.0.0.1 with the printed port.",
            Refusal::NoSession => "Not signed in: open the URL that `rawpilot review` printed.",
            Refusal::WrongOrigin => "Refused: a change must come from this site's own page.",
        }
    }
}

/// The names this server answers to, built from its port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Names {
    hosts: [String; 2],
    origins: [String; 2],
}

impl Names {
    pub fn on_port(port: u16) -> Names {
        let host = |name: &str| format!("{name}:{port}");
        Names {
            hosts: [host("127.0.0.1"), host("localhost")],
            origins: [
                format!("http://{}", host("127.0.0.1")),
                format!("http://{}", host("localhost")),
            ],
        }
    }
}

/// Decides whether the request may proceed. `url_token` is the `t` of the sign-in URL, given
/// only for a request to `/`.
pub fn admit(
    names: &Names,
    token: &Token,
    intent: Intent,
    url_token: Option<&str>,
    headers: &HeaderMap,
) -> Verdict {
    let text = |name: &str| headers.get(name).and_then(|value| value.to_str().ok());
    if !text("host").is_some_and(|host| names.hosts.iter().any(|ours| ours == host)) {
        return Verdict::Refused(Refusal::WrongHost);
    }
    if intent == Intent::Read && url_token.is_some_and(|offered| token.matches(offered)) {
        return Verdict::SignIn;
    }
    let has_cookie = headers
        .get_all(COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|line| line.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .any(|(name, value)| name == COOKIE_NAME && token.matches(value));
    if !has_cookie {
        return Verdict::Refused(Refusal::NoSession);
    }
    match intent {
        Intent::Read => Verdict::Admitted,
        Intent::Write => {
            if text("origin").is_some_and(|o| names.origins.iter().any(|ours| ours == o)) {
                Verdict::Admitted
            } else {
                Verdict::Refused(Refusal::WrongOrigin)
            }
        }
    }
}

/// The `Set-Cookie` value that signs the browser in.
pub fn session_cookie(token: &Token) -> String {
    format!(
        "{COOKIE_NAME}={}; HttpOnly; SameSite=Strict; Path=/",
        token.as_str()
    )
}

#[cfg(test)]
mod tests {
    //! Goal: the admission matrix. Method: hand-written headers against a fixed token and
    //! port; every refusal reason and the one combination that admits each intent.

    use axum::http::{HeaderName, HeaderValue};

    use super::*;

    const HOST_OK: (&str, &str) = ("host", "127.0.0.1:7357");
    const COOKIE_OK: (&str, &str) = ("cookie", "other=1; rawpilot=abc123");
    const ORIGIN_OK: (&str, &str) = ("origin", "http://127.0.0.1:7357");

    fn verdict(intent: Intent, url_token: Option<&str>, pairs: &[(&str, &str)]) -> Verdict {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.append(
                HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        admit(
            &Names::on_port(7357),
            &Token::from_text("abc123"),
            intent,
            url_token,
            &headers,
        )
    }

    /// Mutation: skip the cookie check (`has_cookie = true`) and the first two rows go red.
    #[test]
    fn a_request_without_the_cookie_is_refused() {
        for pairs in [vec![HOST_OK], vec![HOST_OK, ("cookie", "other=1")]] {
            assert_eq!(
                verdict(Intent::Read, None, &pairs),
                Verdict::Refused(Refusal::NoSession)
            );
        }
    }

    /// Mutation: compare the cookie by prefix, or ignore the cookie name.
    #[test]
    fn a_cookie_that_is_not_the_token_is_refused() {
        for cookie in [
            "rawpilot=abc12",
            "rawpilot=abc1234",
            "rawpilot=",
            "rawpilot=ABC123",
            "other=abc123",
        ] {
            assert_eq!(
                verdict(Intent::Read, None, &[HOST_OK, ("cookie", cookie)]),
                Verdict::Refused(Refusal::NoSession),
                "{cookie}"
            );
        }
        assert_eq!(
            verdict(Intent::Read, None, &[HOST_OK, COOKIE_OK]),
            Verdict::Admitted
        );
    }

    /// Mutation: accept any wrong token in the URL, or admit a sign-in on a POST.
    #[test]
    fn only_the_right_url_token_signs_in_and_only_on_a_read() {
        assert_eq!(
            verdict(Intent::Read, Some("abc123"), &[HOST_OK]),
            Verdict::SignIn
        );
        assert_eq!(
            verdict(Intent::Read, Some("abc124"), &[HOST_OK]),
            Verdict::Refused(Refusal::NoSession)
        );
        assert_eq!(
            verdict(Intent::Write, Some("abc123"), &[HOST_OK, ORIGIN_OK]),
            Verdict::Refused(Refusal::NoSession)
        );
    }

    /// Mutation: accept any `Host`, or check the cookie before the host.
    #[test]
    fn the_host_must_be_a_loopback_name_with_the_port() {
        for host in ["127.0.0.1:7357", "localhost:7357"] {
            assert_eq!(
                verdict(Intent::Read, None, &[("host", host), COOKIE_OK]),
                Verdict::Admitted
            );
        }
        for host in [
            "127.0.0.1",
            "127.0.0.1:7358",
            "evil.example:7357",
            "[::1]:7357",
        ] {
            assert_eq!(
                verdict(Intent::Read, None, &[("host", host), COOKIE_OK]),
                Verdict::Refused(Refusal::WrongHost),
                "{host}"
            );
        }
    }

    /// Mutation: drop the origin check on writes.
    #[test]
    fn a_change_needs_this_servers_origin() {
        assert_eq!(
            verdict(Intent::Write, None, &[HOST_OK, COOKIE_OK, ORIGIN_OK]),
            Verdict::Admitted
        );
        for pairs in [
            vec![HOST_OK, COOKIE_OK],
            vec![HOST_OK, COOKIE_OK, ("origin", "http://evil.example")],
            vec![HOST_OK, COOKIE_OK, ("origin", "http://127.0.0.1:7358")],
        ] {
            assert_eq!(
                verdict(Intent::Write, None, &pairs),
                Verdict::Refused(Refusal::WrongOrigin)
            );
        }
    }

    #[test]
    fn tokens_are_random_hex_of_the_stated_length() {
        let (a, b) = (Token::generate().unwrap(), Token::generate().unwrap());
        assert_eq!(a.as_str().len(), 2 * TOKEN_BYTES);
        assert!(a.as_str().bytes().all(|c| c.is_ascii_hexdigit()));
        assert!(!a.matches(b.as_str()));
    }
}
