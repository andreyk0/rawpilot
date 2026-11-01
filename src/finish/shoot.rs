//! What `finish` knows about each shot of the folder, and the whole-shoot summary it archives.

use chrono::{DateTime, FixedOffset};
use serde::Serialize;

use super::event::EventName;
use crate::config::OptionName;
use crate::rate::Stars;
use crate::rpl::{Chosen, Rpl};
use crate::shot::Shot;
use crate::sidecar::FoundMark;

/// Whether `finish` archives and exports a shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keeping {
    Keeper,
    Dropped,
}

impl Keeping {
    /// A shot is kept when the rating RawTherapee reads for it now is at least `min`. A
    /// rejected shot (`-1`) and an unrated one (`0`) are below any `min` a photographer can
    /// give (`1..=5`).
    pub fn of(current_rating: i64, min: Stars) -> Keeping {
        if current_rating >= i64::from(min.get()) {
            Keeping::Keeper
        } else {
            Keeping::Dropped
        }
    }
}

/// One shot of the folder.
#[derive(Debug, Clone)]
pub struct Frame {
    pub name: String,
    pub taken_at: DateTime<FixedOffset>,
    /// The rating rawpilot last wrote; `None` when it never wrote one for this shot.
    pub rawpilot_rating: Option<Stars>,
    /// The rating in the store RawTherapee reads (0 for none).
    pub current_rating: i64,
    /// The look option chosen in the review UI, once its pp3 was written.
    pub look: Option<OptionName>,
    pub keeping: Keeping,
}

impl Frame {
    pub fn new(shot: &Shot, record: &Rpl, current: &FoundMark, min: Stars) -> Frame {
        let current_rating = current.rating.unwrap_or(0);
        Frame {
            name: shot.name(),
            taken_at: shot.taken_at,
            rawpilot_rating: record.written.committed.as_ref().map(|s| s.mark.rating),
            current_rating,
            look: match &record.chosen {
                Some(Chosen::Applied(choice)) => Some(choice.option.clone()),
                Some(Chosen::Pending(_)) | None => None,
            },
            keeping: Keeping::of(current_rating, min),
        }
    }
}

#[derive(Serialize)]
struct SummaryFile<'a> {
    event: &'a str,
    min_rating: u8,
    shots: Vec<SummaryShot<'a>>,
}

#[derive(Serialize)]
struct SummaryShot<'a> {
    file: &'a str,
    taken_at: DateTime<FixedOffset>,
    rawpilot_rating: Option<Stars>,
    current_rating: i64,
    look: Option<&'a str>,
    keeper: bool,
}

/// The summary of the shoot as the JSON `rawpilot/summary.json` holds. The same frames give
/// the same bytes, which is how a re-run finds the archived copy identical.
pub fn summary_json(event: &EventName, min: Stars, frames: &[Frame]) -> String {
    let file = SummaryFile {
        event: event.as_str(),
        min_rating: min.get(),
        shots: frames
            .iter()
            .map(|f| SummaryShot {
                file: &f.name,
                taken_at: f.taken_at,
                rawpilot_rating: f.rawpilot_rating,
                current_rating: f.current_rating,
                look: f.look.as_ref().map(OptionName::as_str),
                keeper: matches!(f.keeping, Keeping::Keeper),
            })
            .collect(),
    };
    let mut json = serde_json::to_string_pretty(&file).expect("a summary is plain data");
    json.push('\n');
    json
}

#[cfg(test)]
mod tests {
    //! Goal: the rating threshold and the summary's wire shape. Method: expected values
    //! written out.

    use super::*;

    /// Mutation: `>` for `>=`, or compare against a different threshold: red.
    #[test]
    fn a_shot_is_kept_from_the_threshold_up() {
        let min = Stars::new(3);
        let kept: Vec<Keeping> = [-1, 0, 2, 3, 4, 5]
            .into_iter()
            .map(|r| Keeping::of(r, min))
            .collect();
        use Keeping::{Dropped, Keeper};
        assert_eq!(kept, [Dropped, Dropped, Dropped, Keeper, Keeper, Keeper]);
    }
}
