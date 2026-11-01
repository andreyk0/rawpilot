//! The name of an event, which becomes the one folder `finish` archives into.

use std::fmt;
use std::str::FromStr;

/// The longest event name accepted. Folder names are limited to 255 bytes by the common file
/// systems; 100 characters leave room for multi-byte ones and for the temporary names
/// `archive` puts beside the files inside.
const CHARS_MAX: usize = 100;

/// A name that is safe as a single folder name under `photos-RAW/`: not empty, no `/`, not
/// `.` or `..`, no leading `-` (a program given it as an argument would read an option), no
/// control characters, at most [`CHARS_MAX`] characters. The only way to build one is
/// [`EventName::parse`], so everything that joins one onto a path can rely on this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventName(String);

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum BadEventName {
    #[error("the event name is empty")]
    Empty,
    #[error("the event name is longer than {CHARS_MAX} characters")]
    TooLong,
    #[error("the event name {0:?} is not a single folder name")]
    NotAFolderName(String),
    #[error("the event name {0:?} starts with `-`")]
    LeadingDash(String),
    #[error("the event name {0:?} has a control character")]
    ControlCharacter(String),
}

impl EventName {
    pub fn parse(s: &str) -> Result<EventName, BadEventName> {
        if s.is_empty() {
            return Err(BadEventName::Empty);
        }
        if s.chars().count() > CHARS_MAX {
            return Err(BadEventName::TooLong);
        }
        if s.chars().any(char::is_control) {
            return Err(BadEventName::ControlCharacter(s.to_owned()));
        }
        if s.contains('/') || s == "." || s == ".." {
            return Err(BadEventName::NotAFolderName(s.to_owned()));
        }
        if s.starts_with('-') {
            return Err(BadEventName::LeadingDash(s.to_owned()));
        }
        Ok(EventName(s.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for EventName {
    type Err = BadEventName;

    fn from_str(s: &str) -> Result<EventName, BadEventName> {
        EventName::parse(s)
    }
}

impl fmt::Display for EventName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    //! Goal: only a single, plain folder name becomes an `EventName`. Method: one row per rule,
    //! each with a name that breaks that rule alone, and the names the photographer types.

    use super::*;

    /// Mutation: drop any one rule from `parse`: its row goes red.
    #[test]
    fn only_a_single_plain_folder_name_is_an_event() {
        for ok in ["2026-11-01-autumn", "Kuća na moru", "a.b", "...", "x-"] {
            assert_eq!(EventName::parse(ok).unwrap().as_str(), ok, "{ok:?}");
        }
        let long = "x".repeat(CHARS_MAX);
        assert!(EventName::parse(&long).is_ok());
        let too_long = format!("{long}x");
        let rows = [
            ("", BadEventName::Empty),
            (too_long.as_str(), BadEventName::TooLong),
            ("a/b", BadEventName::NotAFolderName("a/b".into())),
            ("/abs", BadEventName::NotAFolderName("/abs".into())),
            (".", BadEventName::NotAFolderName(".".into())),
            ("..", BadEventName::NotAFolderName("..".into())),
            ("-rf", BadEventName::LeadingDash("-rf".into())),
            ("a\nb", BadEventName::ControlCharacter("a\nb".into())),
            ("a\0b", BadEventName::ControlCharacter("a\0b".into())),
            ("a\tb", BadEventName::ControlCharacter("a\tb".into())),
        ];
        for (bad, why) in rows {
            assert_eq!(EventName::parse(bad), Err(why), "{bad:?}");
        }
    }
}
