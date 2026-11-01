mod analyze;
mod cache;
mod choice;
mod config;
mod develop;
mod exposure;
mod facecrop;
mod faces;
mod features;
mod finish;
mod gemini;
mod group;
mod judge;
mod lock;
mod look;
mod looks;
mod nef;
mod option;
mod pipeline;
mod pp3;
mod preview;
mod rate;
mod regions;
mod report;
mod review;
mod rpl;
mod rtcli;
mod scene;
mod shortlist;
mod shot;
mod sidecar;
mod tone;
mod writeback;
mod xmp;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::config::Config;
#[derive(Parser)]
#[command(about = "Assistant for a Nikon NEF + RawTherapee workflow")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List the shots rawpilot can read in DIR, with capture time and AF box.
    Scan { dir: PathBuf },
    /// Rate every shot in DIR where RawTherapee reads ratings (pp3 or XMP, per its
    /// ThumbnailRankColorMode), and write .rawpilot/report.html. A pp3 that is missing is
    /// created from the default profile; the processing of an existing pp3 is never changed,
    /// and nothing is rendered.
    Cull(RunArgs),
    /// Cull DIR, then also open fresh images at a corrected exposure (the look) and render
    /// every look option of the shots rated high enough for `rawpilot review`.
    Process(RunArgs),
    /// Serve a local page for going through DIR after `process`: view each shot's look
    /// options, choose one, set ratings. Listens on 127.0.0.1 only and prints the sign-in URL.
    Review(ReviewArgs),
    /// After `review`: export a JPEG of every keeper (rated at least --min-rating where
    /// RawTherapee reads ratings), archive its NEF, pp3, rpl and xmp under
    /// `<media-root>/photos-RAW/<event>/` without replacing anything there, then file the JPEGs
    /// with `import-media`. Close RawTherapee first.
    Finish(FinishArgs),
}

#[derive(Args)]
struct FinishArgs {
    dir: PathBuf,
    /// Name of the folder under photos-RAW, e.g. 2026-11-01-autumn.
    #[arg(long)]
    event: finish::EventName,
    /// Shots rated at least this many stars are kept.
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..=5))]
    min_rating: u8,
    /// The media drive's root, holding photos-RAW (which must exist) and photos; defaults to
    /// the MEDIA_ROOT environment variable.
    #[arg(long)]
    media_root: Option<PathBuf>,
    /// Print what would be exported, archived and refused; write nothing anywhere.
    #[arg(long)]
    dry_run: bool,
}

impl FinishArgs {
    /// The command as `finish::run` takes it. `media_root_env` is the `MEDIA_ROOT` variable.
    fn resolve(self, media_root_env: Option<std::ffi::OsString>) -> Result<finish::Args> {
        let media =
            finish::MediaRoot::open(&finish::MediaRoot::choose(self.media_root, media_root_env)?)?;
        Ok(finish::Args {
            // Absolute, so that no path handed to a program can read as an option.
            dir: std::fs::canonicalize(&self.dir)
                .with_context(|| format!("resolving {}", self.dir.display()))?,
            event: self.event,
            min_rating: rate::Stars::new(self.min_rating),
            media,
            mode: if self.dry_run {
                writeback::Mode::DryRun
            } else {
                writeback::Mode::Write
            },
        })
    }
}

#[derive(Args)]
struct ReviewArgs {
    dir: PathBuf,
    /// Complete replacement for the built-in config/default.toml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Port on 127.0.0.1; 0 lets the system pick one.
    #[arg(long, default_value_t = 0)]
    port: u16,
    /// Also open the sign-in URL in the default browser (`xdg-open`).
    #[arg(long)]
    open: bool,
}

impl ReviewArgs {
    fn browser(&self) -> review::Browser {
        if self.open {
            review::Browser::Open
        } else {
            review::Browser::Leave
        }
    }
}

#[derive(Args)]
struct RunArgs {
    dir: PathBuf,
    /// Complete replacement for the built-in config/default.toml.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Decide and report, but write no rating, pp3 or render.
    #[arg(long)]
    dry_run: bool,
    /// Skip Gemini: pick each burst's sharpest frame. No image leaves the machine.
    #[arg(long)]
    local_only: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Cmd::Scan { dir } => scan(&dir),
        Cmd::Cull(args) => run(args, pipeline::Outputs::RatingsOnly),
        Cmd::Process(args) => run(args, pipeline::Outputs::RatingsLooksAndRenders),
        Cmd::Review(args) => {
            let cfg = Config::load(args.config.as_deref())?;
            review::run(&args.dir, &cfg, args.port, args.browser())
        }
        Cmd::Finish(args) => {
            let args = args.resolve(std::env::var_os("MEDIA_ROOT"))?;
            let rt = sidecar::read_rt_options(&sidecar::default_rt_options_path()?)?;
            finish::run(&args, &rt, &finish::Installed)
        }
    }
}

fn run(args: RunArgs, outputs: pipeline::Outputs) -> Result<()> {
    let mode = if args.dry_run {
        writeback::Mode::DryRun
    } else {
        writeback::Mode::Write
    };
    let judging = if args.local_only {
        pipeline::Judging::LocalOnly
    } else {
        pipeline::Judging::Gemini {
            api_key: std::env::var("GEMINI_API_KEY")
                .context("GEMINI_API_KEY is not set (or pass --local-only)")?,
        }
    };
    pipeline::run(
        &args.dir,
        &Config::load(args.config.as_deref())?,
        judging,
        mode,
        outputs,
    )
}

fn scan(dir: &Path) -> Result<()> {
    let shots = nef::scan(dir)?;
    for s in &shots {
        let af =
            s.af.map(|b| format!("{}x{}@{},{}", b.rect.w, b.rect.h, b.rect.x, b.rect.y))
                .unwrap_or_else(|| "-".into());
        println!(
            "{}\t{}\t1/{:.0}s\t{}mm\tISO{}\t{af}",
            s.name(),
            s.taken_at.format("%H:%M:%S%.3f"),
            1.0 / s.exposure.shutter_s,
            s.exposure.focal_mm,
            s.exposure.iso,
        );
    }
    info!("{} shots", shots.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Goal: `review --open` asks for the browser and nothing else does. Method: the command
    //! lines parsed by clap.

    use super::*;

    fn browser_of(line: &[&str]) -> review::Browser {
        match Cli::try_parse_from(line).unwrap().command {
            Cmd::Review(args) => args.browser(),
            _ => panic!("not a review command line"),
        }
    }

    fn finish_of(line: &[&str]) -> Result<FinishArgs, clap::Error> {
        match Cli::try_parse_from(line)?.command {
            Cmd::Finish(args) => Ok(args),
            _ => panic!("not a finish command line"),
        }
    }

    /// Mutation: make `--event` optional or unchecked, change the default or the range of
    /// `--min-rating`: a row goes red.
    #[test]
    fn finish_needs_a_safe_event_and_a_rating_from_one_to_five() {
        let ok = finish_of(&["rawpilot", "finish", "d", "--event", "2026-11-01-x"]).unwrap();
        assert_eq!(ok.min_rating, 3);
        assert!(!ok.dry_run);
        assert_eq!(ok.media_root, None);
        let ok = finish_of(&[
            "rawpilot",
            "finish",
            "d",
            "--event",
            "x",
            "--min-rating",
            "5",
            "--dry-run",
        ])
        .unwrap();
        assert_eq!((ok.min_rating, ok.dry_run), (5, true));

        for bad in [
            &["rawpilot", "finish", "d"][..],
            &["rawpilot", "finish", "d", "--event", "a/b"],
            &["rawpilot", "finish", "d", "--event", ".."],
            &["rawpilot", "finish", "d", "--event", ""],
            &[
                "rawpilot",
                "finish",
                "d",
                "--event",
                "x",
                "--min-rating",
                "0",
            ],
            &[
                "rawpilot",
                "finish",
                "d",
                "--event",
                "x",
                "--min-rating",
                "6",
            ],
        ] {
            assert!(finish_of(bad).is_err(), "{bad:?}");
        }
    }

    /// The media root is the flag's, else the environment's (passed in, never read here),
    /// and must hold `photos-RAW`. Mutation: ignore the flag, or skip the `photos-RAW`
    /// check: red.
    #[test]
    fn finish_resolves_the_media_root_from_flag_or_environment() {
        let dir = tempfile::tempdir().unwrap();
        let drive = tempfile::tempdir().unwrap();
        let line = |extra: &[&str]| {
            let mut v = vec![
                "rawpilot",
                "finish",
                dir.path().to_str().unwrap(),
                "--event",
                "e",
            ];
            v.extend_from_slice(extra);
            finish_of(&v).unwrap()
        };
        let env = |p: &Path| Some(p.as_os_str().to_owned());

        assert!(line(&[]).resolve(None).is_err());
        assert!(
            line(&[]).resolve(env(drive.path())).is_err(),
            "no photos-RAW yet"
        );
        std::fs::create_dir(drive.path().join("photos-RAW")).unwrap();
        let args = line(&[]).resolve(env(drive.path())).unwrap();
        assert_eq!(args.media.path(), drive.path());
        let args = line(&["--media-root", drive.path().to_str().unwrap(), "--dry-run"])
            .resolve(env(Path::new("/nonexistent")))
            .unwrap();
        assert_eq!(args.media.path(), drive.path());
        assert_eq!(args.mode, writeback::Mode::DryRun);
    }

    /// Mutation: make `browser` always answer `Leave` (or `Open`): one of the rows goes red.
    #[test]
    fn the_open_flag_asks_for_the_browser() {
        assert_eq!(
            browser_of(&["rawpilot", "review", "shoot", "--open"]),
            review::Browser::Open
        );
        assert_eq!(
            browser_of(&["rawpilot", "review", "shoot"]),
            review::Browser::Leave
        );
        assert_eq!(
            browser_of(&["rawpilot", "review", "--port", "7", "--open", "shoot"]),
            review::Browser::Open
        );
    }
}
