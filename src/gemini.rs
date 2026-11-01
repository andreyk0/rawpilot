//! Minimal Gemini API (generativelanguage.googleapis.com) client: one `generateContent` call
//! with inline JPEGs and a JSON response schema.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::cache::write_atomic;
use crate::config::JudgeConfig;

const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/models";

#[derive(Serialize)]
#[serde(untagged)]
pub enum Part {
    Text { text: String },
    Image { inline_data: InlineData },
}

#[derive(Serialize)]
pub struct InlineData {
    mime_type: &'static str,
    data: String,
}

impl Part {
    pub fn text(s: impl Into<String>) -> Part {
        Part::Text { text: s.into() }
    }

    pub fn jpeg(bytes: &[u8]) -> Part {
        Part::Image {
            inline_data: InlineData {
                mime_type: "image/jpeg",
                data: base64::engine::general_purpose::STANDARD.encode(bytes),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub output_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, o: Usage) {
        self.prompt_tokens += o.prompt_tokens;
        self.output_tokens += o.output_tokens;
    }
}

pub struct Reply {
    pub text: String,
    /// The concrete model that answered, as the API reports it.
    pub model_version: String,
    pub usage: Usage,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Response {
    candidates: Vec<Candidate>,
    model_version: String,
    usage_metadata: UsageMetadata,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Candidate {
    content: Option<Content>,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Content {
    parts: Vec<TextPart>,
}

#[derive(Deserialize)]
struct TextPart {
    text: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
    prompt_token_count: u64,
    #[serde(default)]
    candidates_token_count: u64,
    /// Thinking tokens are billed as output.
    #[serde(default)]
    thoughts_token_count: u64,
}

/// A complete generateContent request: everything that determines the answer, apart from
/// the model, which is part of the URL.
pub struct Request {
    pub model: String,
    pub body: serde_json::Value,
}

impl Request {
    pub fn new(cfg: &JudgeConfig, parts: Vec<Part>, schema: serde_json::Value) -> Request {
        Request {
            model: cfg.model.clone(),
            body: serde_json::json!({
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": schema,
                    "mediaResolution": cfg.media_resolution,
                    "thinkingConfig": {"thinkingLevel": cfg.thinking_level},
                }
            }),
        }
    }

    /// Identifies the request exactly: two requests with the same digest are the same
    /// question, so a cached answer to one answers the other.
    pub fn digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(self.model.as_bytes());
        h.update(&[0]);
        h.update(self.body.to_string().as_bytes());
        h.finalize().to_hex().to_string()
    }
}

/// An answer on disk, keyed by the digest of the request that produced it.
#[derive(Serialize, Deserialize)]
struct CachedAnswer {
    model_version: String,
    usage: Usage,
    text: String,
}

/// A pass that asks Gemini, and so the directory under `<dir>/.rawpilot/` holding its answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Pass {
    Judge,
    Scene,
    Faces,
    Shortlist,
}

impl Pass {
    pub const ALL: [Pass; 4] = [Pass::Judge, Pass::Scene, Pass::Faces, Pass::Shortlist];

    pub fn dir_name(self) -> &'static str {
        match self {
            Pass::Judge => "judge",
            Pass::Scene => "scene",
            Pass::Faces => "faces",
            Pass::Shortlist => "shortlist",
        }
    }
}

/// One answer found in the answer cache, as `finish` archives it.
#[derive(Debug, Serialize)]
pub struct StoredAnswer {
    pub pass: Pass,
    /// The request digest the answer is cached under (`Request::digest`).
    pub digest: String,
    pub model_version: String,
    pub usage: Usage,
    /// The model's reply, verbatim.
    pub text: String,
}

/// Every answer cached under `cache_root` (`<dir>/.rawpilot`), in pass order and then digest
/// order. A pass that was never run has no directory and no answers. An entry that cannot be
/// read is logged and left out: the archive is then missing it, which the log says.
pub fn stored_answers(cache_root: &Path) -> Result<Vec<StoredAnswer>> {
    let mut answers = Vec::new();
    for pass in Pass::ALL {
        let dir = cache_root.join(pass.dir_name());
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        };
        let mut found = Vec::new();
        for entry in entries {
            let path = entry?.path();
            let Some(digest) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".json"))
            else {
                continue;
            };
            match std::fs::read_to_string(&path)
                .map_err(anyhow::Error::from)
                .and_then(|t| Ok(serde_json::from_str::<CachedAnswer>(&t)?))
            {
                Ok(c) => found.push(StoredAnswer {
                    pass,
                    digest: digest.to_owned(),
                    model_version: c.model_version,
                    usage: c.usage,
                    text: c.text,
                }),
                Err(e) => warn!(file = %path.display(), "unreadable cached answer left out: {e:#}"),
            }
        }
        found.sort_by(|a, b| a.digest.cmp(&b.digest));
        answers.extend(found);
    }
    Ok(answers)
}

/// How many times a model answer that fails `parse` is asked for again. Models occasionally
/// return an out-of-range index or skip an item; a second answer usually does not.
const INVALID_ANSWER_ATTEMPTS: u32 = 2;

/// `req`'s answer parsed by `parse`, from `cache_dir` when this exact request was answered
/// before. The usage is `Some` only when the API was called. Only answers that parse are
/// stored, so a cache hit is always a valid answer under the current `parse`, or is asked
/// again.
pub fn ask<T>(
    api_key: &str,
    cfg: &JudgeConfig,
    req: &Request,
    cache_dir: &Path,
    parse: impl Fn(&str) -> Result<T>,
) -> Result<(T, Option<Usage>)> {
    let path = cache_dir.join(format!("{}.json", req.digest()));
    if let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(cached) = serde_json::from_str::<CachedAnswer>(&text)
    {
        match parse(&cached.text) {
            Ok(v) => return Ok((v, None)),
            Err(e) => warn!("cached answer no longer parses, asking again: {e:#}"),
        }
    }
    let mut usage = Usage::default();
    let mut last_err = None;
    for _ in 0..INVALID_ANSWER_ATTEMPTS {
        let reply = generate(api_key, cfg, req)?;
        usage += reply.usage;
        match parse(&reply.text) {
            Ok(v) => {
                let cached = CachedAnswer {
                    model_version: reply.model_version,
                    usage: reply.usage,
                    text: reply.text,
                };
                write_atomic(&path, serde_json::to_string_pretty(&cached)?.as_bytes())?;
                return Ok((v, Some(usage)));
            }
            Err(e) => {
                warn!("model answer invalid: {e:#}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.expect("loop ran at least once")).context("no valid answer")
}

/// Totals for one pass over many requests, logged at the end.
#[derive(Debug, Default)]
pub struct Tally {
    pub requests: usize,
    pub cached: usize,
    pub failed: usize,
    pub usage: Usage,
}

impl Tally {
    pub fn record<T>(&mut self, r: &Result<(T, Option<Usage>)>) {
        self.requests += 1;
        match r {
            Ok((_, Some(u))) => self.usage += *u,
            Ok((_, None)) => self.cached += 1,
            Err(_) => self.failed += 1,
        }
    }

    pub fn log(&self, pass: &str) {
        tracing::info!(
            "{pass}: {} requests, {} cached, {} failed; used {} prompt + {} output tokens",
            self.requests,
            self.cached,
            self.failed,
            self.usage.prompt_tokens,
            self.usage.output_tokens
        );
    }
}

/// Retries 429 and 5xx with exponential backoff; any other failure is returned at once,
/// since it would fail identically again. A generateContent call has no side effects, so
/// repeating it is safe.
pub fn generate(api_key: &str, cfg: &JudgeConfig, req: &Request) -> Result<Reply> {
    let body = &req.body;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(cfg.request_timeout_s))
        .build()?;
    let url = format!("{ENDPOINT}/{}:generateContent", req.model);
    let mut delay = Duration::from_secs(2);
    for attempt in 1..=cfg.attempts_max {
        let sent = client
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&body)
            .send();
        let retryable = match sent {
            Ok(r) if r.status().is_success() => {
                return parse(r.json().context("reading Gemini response")?);
            }
            Ok(r) => {
                let status = r.status();
                let text = r.text().unwrap_or_default();
                if !(status.as_u16() == 429 || status.is_server_error()) {
                    bail!("Gemini returned {status}: {text}");
                }
                format!("{status}: {text}")
            }
            Err(e) if e.is_timeout() || e.is_connect() => e.to_string(),
            Err(e) => return Err(e).context("calling Gemini"),
        };
        if attempt == cfg.attempts_max {
            bail!("Gemini failed after {attempt} attempts: {retryable}");
        }
        warn!(
            attempt,
            "Gemini call failed, retrying in {delay:?}: {retryable}"
        );
        std::thread::sleep(delay + jitter(delay));
        delay *= 2;
    }
    unreachable!("loop returns or bails on the last attempt");
}

/// Up to a quarter of `d`, from the clock's sub-second noise; enough to spread concurrent
/// retries without a random-number dependency.
fn jitter(d: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|t| t.subsec_nanos())
        .unwrap_or(0);
    d.mul_f64(f64::from(nanos % 1000) / 4000.0)
}

fn parse(r: Response) -> Result<Reply> {
    let c = r
        .candidates
        .into_iter()
        .next()
        .context("Gemini returned no candidate")?;
    let text: String = c
        .content
        .map(|c| c.parts.into_iter().filter_map(|p| p.text).collect())
        .unwrap_or_default();
    if text.is_empty() {
        bail!(
            "Gemini returned no text (finishReason {:?})",
            c.finish_reason
        );
    }
    debug!(model = %r.model_version, "Gemini answer: {text}");
    Ok(Reply {
        text,
        model_version: r.model_version,
        usage: Usage {
            prompt_tokens: r.usage_metadata.prompt_token_count,
            output_tokens: r.usage_metadata.candidates_token_count
                + r.usage_metadata.thoughts_token_count,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(model: &str, image: &[u8]) -> Request {
        let mut cfg = crate::config::Config::load(None).unwrap().judge;
        cfg.model = model.to_owned();
        Request::new(
            &cfg,
            vec![Part::text("judge"), Part::jpeg(image)],
            serde_json::json!({"type": "OBJECT"}),
        )
    }

    /// The digest is the answer cache's key: identical requests share it, and a change to
    /// any image byte or to the model makes a new question.
    #[test]
    fn digest_covers_images_and_model() {
        let base = request("m1", &[1, 2, 3]).digest();
        assert_eq!(base, request("m1", &[1, 2, 3]).digest());
        assert_ne!(base, request("m1", &[1, 2, 4]).digest());
        assert_ne!(base, request("m2", &[1, 2, 3]).digest());
    }

    fn cache_answer(root: &Path, pass: Pass, digest: &str, text: &str) {
        let dir = root.join(pass.dir_name());
        std::fs::create_dir_all(&dir).unwrap();
        let answer = CachedAnswer {
            model_version: "m-1".into(),
            usage: Usage {
                prompt_tokens: 7,
                output_tokens: 3,
            },
            text: text.into(),
        };
        std::fs::write(
            dir.join(format!("{digest}.json")),
            serde_json::to_string_pretty(&answer).unwrap(),
        )
        .unwrap();
    }

    /// Goal: the answers `ask` cached are all found, in pass order then digest order, and
    /// only they. Method: answers written through `CachedAnswer`'s serialisation into a temp
    /// cache with one stray temp file, one unreadable entry and a pass never run.
    /// Mutation: drop the sort, skip a pass in `Pass::ALL`, or stop filtering on `.json`: red.
    #[test]
    fn stored_answers_are_every_cached_answer_in_a_fixed_order() {
        let root = tempfile::tempdir().unwrap();
        cache_answer(root.path(), Pass::Scene, "bb", r#"{"roles":[1]}"#);
        cache_answer(root.path(), Pass::Judge, "mm", r#"{"a":3}"#);
        cache_answer(root.path(), Pass::Judge, "zz", r#"{"a":1}"#);
        cache_answer(root.path(), Pass::Judge, "aa", r#"{"a":2}"#);
        cache_answer(root.path(), Pass::Judge, "qq", r#"{"a":4}"#);
        cache_answer(root.path(), Pass::Shortlist, "cc", "[]");
        cache_answer(root.path(), Pass::Faces, "dd", "{}");
        std::fs::write(root.path().join("judge/ee.json.tmp"), "half").unwrap();
        std::fs::write(root.path().join("judge/ff.json"), "not json").unwrap();

        let got: Vec<(Pass, String, String)> = stored_answers(root.path())
            .unwrap()
            .into_iter()
            .map(|a| (a.pass, a.digest, a.text))
            .collect();
        let want = |p: Pass, d: &str, t: &str| (p, d.to_owned(), t.to_owned());
        assert_eq!(
            got,
            vec![
                want(Pass::Judge, "aa", r#"{"a":2}"#),
                want(Pass::Judge, "mm", r#"{"a":3}"#),
                want(Pass::Judge, "qq", r#"{"a":4}"#),
                want(Pass::Judge, "zz", r#"{"a":1}"#),
                want(Pass::Scene, "bb", r#"{"roles":[1]}"#),
                want(Pass::Faces, "dd", "{}"),
                want(Pass::Shortlist, "cc", "[]"),
            ]
        );
        assert!(
            stored_answers(&root.path().join("missing"))
                .unwrap()
                .is_empty()
        );
    }
}
