//! Telling the photographer where the page is: the sign-in URL on stdout, and with
//! `--open` also in the default browser.

use std::io::{self, Write};
use std::process::{Command, Stdio};

use tracing::warn;

/// Whether the sign-in URL is also opened in the default browser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Browser {
    Leave,
    Open,
}

/// Hands a URL to something that shows it. The real one is [`xdg_open`].
pub type Launcher<'a> = dyn Fn(&str) -> io::Result<()> + 'a;

/// Prints `url` as the command's one line of output, then, for [`Browser::Open`], hands the
/// very same text to `launch`. A launcher that fails costs a warning: the URL is printed
/// either way.
pub fn announce(
    url: &str,
    out: &mut dyn Write,
    browser: Browser,
    launch: &Launcher<'_>,
) -> io::Result<()> {
    writeln!(out, "{url}")?;
    out.flush()?;
    match browser {
        Browser::Leave => {}
        Browser::Open => {
            if let Err(error) = launch(url) {
                warn!(%error, "cannot open the browser; open the URL above yourself");
            }
        }
    }
    Ok(())
}

/// `xdg-open URL`, the freedesktop way to open the default browser, not waited for: the
/// command's output is not ours, so all three standard streams are closed. A thread reaps
/// the child and reports a failed exit.
pub fn xdg_open(url: &str) -> io::Result<()> {
    let mut child = Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    std::thread::spawn(move || match child.wait() {
        Ok(status) if status.success() => {}
        Ok(status) => warn!(%status, "xdg-open could not open the browser"),
        Err(error) => warn!(%error, "cannot wait for xdg-open"),
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Goal: `--open` hands the browser the URL that was printed, and only with the flag; a
    //! launcher that fails does not stop the URL being printed. Method: a recording launcher
    //! and an in-memory stdout.

    use std::cell::RefCell;

    use super::*;

    const URL: &str = "http://127.0.0.1:4242/?t=abc";

    fn run(browser: Browser, fail: bool) -> (String, Vec<String>) {
        let launched = RefCell::new(Vec::new());
        let mut out = Vec::new();
        announce(URL, &mut out, browser, &|url| {
            launched.borrow_mut().push(url.to_owned());
            if fail {
                Err(io::Error::new(io::ErrorKind::NotFound, "no xdg-open"))
            } else {
                Ok(())
            }
        })
        .unwrap();
        (String::from_utf8(out).unwrap(), launched.into_inner())
    }

    /// Mutation: launch with anything but `url` (e.g. the bare `http://127.0.0.1:4242/`):
    /// the first assertion goes red.
    #[test]
    fn open_hands_over_the_printed_url() {
        let (printed, launched) = run(Browser::Open, false);
        assert_eq!(printed, format!("{URL}\n"));
        assert_eq!(launched, [URL]);
    }

    /// Mutation: launch for `Leave` too: goes red.
    #[test]
    fn without_the_flag_nothing_is_launched() {
        let (printed, launched) = run(Browser::Leave, false);
        assert_eq!(printed, format!("{URL}\n"));
        assert!(launched.is_empty());
    }

    /// Mutation: propagate the launcher's error with `?`: goes red.
    #[test]
    fn a_launcher_that_fails_leaves_the_url_printed() {
        let (printed, launched) = run(Browser::Open, true);
        assert_eq!(printed, format!("{URL}\n"));
        assert_eq!(launched, [URL]);
    }
}
