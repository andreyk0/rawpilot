//! Where the photographer was: the file name of the last shot viewed, kept in
//! `<DIR>/.rawpilot/review-state.json` so the next launch opens there.
//!
//! The record is `{"shot": "DSC_0001.NEF"}`, written atomically and parsed strictly. A
//! missing, unreadable or malformed record means "no place": review starts at the first shot.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::cache::write_atomic;

const FILE: &str = "review-state.json";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    shot: String,
}

/// A JSON object with exactly the record's fields. (Deserializing straight into `Record`
/// would also take a JSON array of the field values.)
fn parse(text: &str) -> Result<Record, serde_json::Error> {
    let value: serde_json::Value = serde_json::from_str(text)?;
    if !value.is_object() {
        return Err(serde::de::Error::custom("not a JSON object"));
    }
    serde_json::from_value(value)
}

/// The resume record of one folder.
#[derive(Debug, Clone)]
pub struct Place {
    path: PathBuf,
}

impl Place {
    /// The record under the folder's cache root (`<DIR>/.rawpilot`).
    pub fn under(cache_root: &Path) -> Place {
        Place {
            path: cache_root.join(FILE),
        }
    }

    /// The file name of the shot last viewed. `None` when there is no record, or it cannot
    /// be read or understood, which is logged: it does not stop the review.
    pub fn load(&self) -> Option<String> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return None,
            Err(e) => {
                warn!(
                    "cannot read {}: {e}; starting at the first shot",
                    self.path.display()
                );
                return None;
            }
        };
        match parse(&text) {
            Ok(record) => Some(record.shot),
            Err(e) => {
                warn!(
                    "{} is not a place record: {e}; starting at the first shot",
                    self.path.display()
                );
                None
            }
        }
    }

    pub fn save(&self, shot: &str) -> Result<()> {
        let record = Record {
            shot: shot.to_owned(),
        };
        let text = serde_json::to_string(&record)?;
        write_atomic(&self.path, text.as_bytes())
            .with_context(|| format!("writing {}", self.path.display()))
    }
}

#[cfg(test)]
mod tests {
    //! Goal: a place is read back exactly as written, and anything else on disk is no place.
    //! Method: a temp cache root; the records a damaged or foreign file could hold are
    //! written by hand.

    use super::*;

    fn place() -> (tempfile::TempDir, Place) {
        let dir = tempfile::tempdir().unwrap();
        let place = Place::under(dir.path());
        (dir, place)
    }

    /// Mutation: save the wrong field, or load `None` always.
    #[test]
    fn a_saved_place_is_read_back() {
        let (dir, place) = place();
        place.save("DSC_0042.NEF").unwrap();
        assert_eq!(place.load().as_deref(), Some("DSC_0042.NEF"));
        place.save("DSC_0007.NEF").unwrap();
        assert_eq!(place.load().as_deref(), Some("DSC_0007.NEF"));
        // The record's bytes are part of the contract with the next version.
        assert_eq!(
            std::fs::read_to_string(dir.path().join("review-state.json")).unwrap(),
            r#"{"shot":"DSC_0007.NEF"}"#
        );
        assert!(!dir.path().join("review-state.json.tmp").exists());
    }

    /// Mutation: drop `deny_unknown_fields`, or fall back to a default on a parse error:
    /// the unknown-field and wrong-type rows go red.
    #[test]
    fn no_record_or_one_not_understood_is_no_place() {
        let (dir, place) = place();
        assert_eq!(place.load(), None, "missing");
        for text in [
            "",
            "not json",
            "{}",
            r#"{"shot":3}"#,
            r#"{"shot":null}"#,
            r#"{"shot":"a.NEF","extra":1}"#,
            r#"["a.NEF"]"#,
        ] {
            std::fs::write(dir.path().join("review-state.json"), text).unwrap();
            assert_eq!(place.load(), None, "{text:?}");
        }
    }

    #[test]
    fn an_unreadable_record_is_no_place() {
        let (dir, place) = place();
        // A directory where the file should be cannot be read as one.
        std::fs::create_dir(dir.path().join("review-state.json")).unwrap();
        assert_eq!(place.load(), None);
    }
}
