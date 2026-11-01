//! What the page asks of the server, and the indices it names things by.
//!
//! A request carries a command as text (`goto:3`, `rate:4`). It is parsed against the sizes
//! of the shoot into a [`Command`] whose indices are witnesses: an [`Idx`] exists only for a
//! position inside its list, so nothing downstream bounds-checks or turns client text into a
//! path.

use std::fmt;
use std::marker::PhantomData;

use crate::rate::Stars;

/// Marks an [`Idx`] into the shots of the shoot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OfShot {}

/// Marks an [`Idx`] into the configured look options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OfOption {}

/// A position inside the list `K` marks, built only against that list's length.
pub struct Idx<K>(usize, PhantomData<K>);

pub type ShotIdx = Idx<OfShot>;
pub type OptionIdx = Idx<OfOption>;

impl<K> Idx<K> {
    /// `n` as an index into a list of `count` items.
    pub fn new(n: usize, count: usize) -> Option<Idx<K>> {
        (n < count).then_some(Idx(n, PhantomData))
    }

    /// Every index of a list of `count` items, in order.
    pub fn all(count: usize) -> impl Iterator<Item = Idx<K>> {
        (0..count).map(|n| Idx(n, PhantomData))
    }

    pub fn get(self) -> usize {
        self.0
    }

    /// Decimal digits and nothing else, naming a position inside the list.
    pub fn parse(text: &str, count: usize) -> Result<Idx<K>, IndexError> {
        let n = if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) {
            text.parse::<usize>().ok()
        } else {
            None
        };
        n.and_then(|n| Idx::new(n, count))
            .ok_or_else(|| IndexError {
                text: text.to_owned(),
                count,
            })
    }
}

// Derived impls would demand the same of `K`, which is only a marker.
impl<K> Clone for Idx<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K> Copy for Idx<K> {}
impl<K> PartialEq for Idx<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<K> Eq for Idx<K> {}
impl<K> std::hash::Hash for Idx<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl<K> PartialOrd for Idx<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<K> Ord for Idx<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl<K> fmt::Debug for Idx<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Idx({})", self.0)
    }
}
impl<K> fmt::Display for Idx<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{text:?} is not a position in a list of {count}")]
pub struct IndexError {
    text: String,
    count: usize,
}

/// The sizes a command's indices are checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub shots: usize,
    pub options: usize,
}

/// One thing the photographer can ask for. [`fmt::Display`] is the wire text and
/// [`Command::parse`] reads it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Move to another shot. It names no shot of the page: it moves from wherever the
    /// server is, and writes nothing.
    Move(Move),
    Goto(ShotIdx),
    /// Move the rating filter one star stricter or looser.
    StepFilter(FilterWay),
    /// Something done to the shot the page shows. The command names that shot, so a page
    /// that is out of date (a second tab, a reply that arrived late) cannot act on another.
    On(ShotIdx, Action),
}

/// Where to go from the shot on screen, among the shots the working view steps through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Next,
    Previous,
    /// The first shot of the next scene that has one.
    NextScene,
    /// The first shot of the scene on screen, or of the scene before it when already there.
    PreviousScene,
    /// The next shot with no look chosen.
    NextUndecided,
    PreviousUndecided,
    First,
    Last,
    /// [`Move::PAGE`] shots on.
    PageForward,
    PageBack,
}

impl Move {
    /// How many shots a page moves.
    pub const PAGE: usize = 10;

    pub const ALL: [Move; 10] = [
        Move::Next,
        Move::Previous,
        Move::NextScene,
        Move::PreviousScene,
        Move::NextUndecided,
        Move::PreviousUndecided,
        Move::First,
        Move::Last,
        Move::PageForward,
        Move::PageBack,
    ];

    /// The wire text, the one place it is written.
    fn wire(self) -> &'static str {
        match self {
            Move::Next => "next",
            Move::Previous => "prev",
            Move::NextScene => "scene:next",
            Move::PreviousScene => "scene:prev",
            Move::NextUndecided => "undecided:next",
            Move::PreviousUndecided => "undecided:prev",
            Move::First => "first",
            Move::Last => "last",
            Move::PageForward => "page:forward",
            Move::PageBack => "page:back",
        }
    }
}

/// What is done to the shot on screen, with everything on the page it depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Show an option's render in place of the current one; writes nothing.
    View(OptionIdx),
    /// Show the option viewed before this one.
    Flip,
    /// Write the option the page shows as the shot's look, then move to the next shot.
    Accept(OptionIdx),
    Rate(Stars),
    /// Answer yes to the question on screen, which asks about replacing with this option.
    Yes(OptionIdx),
    /// Answer no to the question on screen, which asks about this option.
    No(OptionIdx),
}

/// An [`Action`] as a key means it, before the page says which shot and which option it
/// shows. The page's script fills the two slots of [`Gesture::template`] from the `#app`
/// element it holds when the command is sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gesture {
    View(OptionIdx),
    Flip,
    Accept,
    Rate(Stars),
    Yes,
    No,
}

/// Where the script puts the shot on screen (`data-shot`) in a key's command.
pub const SHOT_SLOT: &str = "{shot}";
/// Where the script puts the option on screen (`data-viewed`) in a key's command.
pub const VIEWED_SLOT: &str = "{viewed}";

impl Gesture {
    /// The wire text with the shot and viewed option left as slots.
    pub fn template(self) -> String {
        self.text(&SHOT_SLOT, &VIEWED_SLOT)
    }

    /// The one place the wire form of these commands is written.
    fn text(self, shot: &dyn fmt::Display, viewed: &dyn fmt::Display) -> String {
        match self {
            Gesture::View(o) => format!("view:{shot}:{o}"),
            Gesture::Flip => format!("flip:{shot}"),
            Gesture::Accept => format!("accept:{shot}:{viewed}"),
            Gesture::Rate(s) => format!("rate:{shot}:{}", s.get()),
            Gesture::Yes => format!("yes:{shot}:{viewed}"),
            Gesture::No => format!("no:{shot}:{viewed}"),
        }
    }
}

impl Action {
    /// What a key would say to ask for this, without the page's two facts.
    pub fn gesture(self) -> Gesture {
        match self {
            Action::View(o) => Gesture::View(o),
            Action::Flip => Gesture::Flip,
            Action::Accept(_) => Gesture::Accept,
            Action::Rate(s) => Gesture::Rate(s),
            Action::Yes(_) => Gesture::Yes,
            Action::No(_) => Gesture::No,
        }
    }

    fn text(self, shot: ShotIdx) -> String {
        let viewed = match self {
            Action::Accept(viewed) | Action::Yes(viewed) | Action::No(viewed) => viewed.to_string(),
            Action::View(_) | Action::Flip | Action::Rate(_) => String::new(),
        };
        self.gesture().text(&shot, &viewed)
    }
}

/// Which way the rating filter moves: `Up` asks for more stars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterWay {
    Up,
    Down,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    #[error("unknown command {0:?}")]
    Unknown(String),
    #[error("command {command}: {source}")]
    Index {
        command: &'static str,
        source: IndexError,
    },
    #[error("rating {0:?} is not 0 to 5")]
    Rating(String),
}

impl Command {
    pub fn parse(text: &str, bounds: Bounds) -> Result<Command, CommandError> {
        fn index<K>(
            command: &'static str,
            count: usize,
            argument: &str,
        ) -> Result<Idx<K>, CommandError> {
            Idx::parse(argument, count).map_err(|source| CommandError::Index { command, source })
        }
        let shot = |command, a| index(command, bounds.shots, a);
        let option = |command, a| index(command, bounds.options, a);
        if let Some(m) = Move::ALL.into_iter().find(|m| m.wire() == text) {
            return Ok(Command::Move(m));
        }
        let parts: Vec<&str> = text.split(':').collect();
        match parts.as_slice() {
            ["filter", "up"] => Ok(Command::StepFilter(FilterWay::Up)),
            ["filter", "down"] => Ok(Command::StepFilter(FilterWay::Down)),
            ["goto", s] => Ok(Command::Goto(shot("goto", s)?)),
            ["view", s, o] => Ok(Command::On(
                shot("view", s)?,
                Action::View(option("view", o)?),
            )),
            ["flip", s] => Ok(Command::On(shot("flip", s)?, Action::Flip)),
            ["accept", s, o] => Ok(Command::On(
                shot("accept", s)?,
                Action::Accept(option("accept", o)?),
            )),
            ["rate", s, n] => Ok(Command::On(shot("rate", s)?, Action::Rate(parse_stars(n)?))),
            ["yes", s, o] => Ok(Command::On(shot("yes", s)?, Action::Yes(option("yes", o)?))),
            ["no", s, o] => Ok(Command::On(shot("no", s)?, Action::No(option("no", o)?))),
            _ => Err(CommandError::Unknown(text.to_owned())),
        }
    }
}

/// A single digit 0 to 5.
pub fn parse_stars(text: &str) -> Result<Stars, CommandError> {
    match text.as_bytes() {
        [d @ b'0'..=b'5'] => Ok(Stars::new(d - b'0')),
        _ => Err(CommandError::Rating(text.to_owned())),
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Command::Move(m) => f.write_str(m.wire()),
            Command::Goto(i) => write!(f, "goto:{i}"),
            Command::StepFilter(FilterWay::Up) => f.write_str("filter:up"),
            Command::StepFilter(FilterWay::Down) => f.write_str("filter:down"),
            Command::On(shot, action) => f.write_str(&action.text(*shot)),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Goal: client text becomes a command only when every index names something that
    //! exists. Method: the wire texts the page sends, good and bad, against a shoot of 5
    //! shots and 3 options; `Display` is checked against the text written out by hand.

    use super::*;

    const BOUNDS: Bounds = Bounds {
        shots: 5,
        options: 3,
    };

    fn shot(n: usize) -> ShotIdx {
        Idx::new(n, BOUNDS.shots).unwrap()
    }

    fn option(n: usize) -> OptionIdx {
        Idx::new(n, BOUNDS.options).unwrap()
    }

    #[test]
    fn every_command_parses_from_its_wire_text() {
        for (text, command) in [
            ("next", Command::Move(Move::Next)),
            ("prev", Command::Move(Move::Previous)),
            ("scene:next", Command::Move(Move::NextScene)),
            ("scene:prev", Command::Move(Move::PreviousScene)),
            ("undecided:next", Command::Move(Move::NextUndecided)),
            ("undecided:prev", Command::Move(Move::PreviousUndecided)),
            ("first", Command::Move(Move::First)),
            ("last", Command::Move(Move::Last)),
            ("page:forward", Command::Move(Move::PageForward)),
            ("page:back", Command::Move(Move::PageBack)),
            ("goto:4", Command::Goto(shot(4))),
            ("view:3:2", Command::On(shot(3), Action::View(option(2)))),
            ("flip:1", Command::On(shot(1), Action::Flip)),
            (
                "accept:4:1",
                Command::On(shot(4), Action::Accept(option(1))),
            ),
            (
                "rate:2:0",
                Command::On(shot(2), Action::Rate(Stars::new(0))),
            ),
            (
                "rate:0:5",
                Command::On(shot(0), Action::Rate(Stars::new(5))),
            ),
            ("filter:up", Command::StepFilter(FilterWay::Up)),
            ("filter:down", Command::StepFilter(FilterWay::Down)),
            ("yes:3:1", Command::On(shot(3), Action::Yes(option(1)))),
            ("no:0:2", Command::On(shot(0), Action::No(option(2)))),
        ] {
            assert_eq!(Command::parse(text, BOUNDS), Ok(command), "{text}");
            assert_eq!(command.to_string(), text);
        }
    }

    /// `Move::ALL` is what `parse` searches, so a move missing from it cannot be asked for.
    /// The match has no wildcard, so a new `Move` fails to compile here until it is counted;
    /// the count must then match `ALL`. Mutation: drop a move from `ALL` or repeat one.
    #[test]
    fn every_move_is_in_the_table_once() {
        let counted = |m: Move| match m {
            Move::Next
            | Move::Previous
            | Move::NextScene
            | Move::PreviousScene
            | Move::NextUndecided
            | Move::PreviousUndecided
            | Move::First
            | Move::Last
            | Move::PageForward
            | Move::PageBack => 1,
        };
        assert_eq!(Move::ALL.into_iter().map(counted).sum::<usize>(), 10);
        let mut texts: Vec<String> = Move::ALL
            .into_iter()
            .map(|m| Command::Move(m).to_string())
            .collect();
        texts.sort();
        texts.dedup();
        assert_eq!(texts.len(), Move::ALL.len());
    }

    /// A key's command is the same text with two slots the page fills; once filled it is
    /// the command a click on the page would send. Mutation: swap the slot order in
    /// `Gesture::text` for accept: the accept row goes red.
    #[test]
    fn a_keys_template_filled_in_is_the_command() {
        // On shot 4 with option 2 shown.
        for (gesture, template, command) in [
            (
                Gesture::View(option(1)),
                "view:{shot}:1",
                Action::View(option(1)),
            ),
            (Gesture::Flip, "flip:{shot}", Action::Flip),
            (
                Gesture::Accept,
                "accept:{shot}:{viewed}",
                Action::Accept(option(2)),
            ),
            (
                Gesture::Rate(Stars::new(3)),
                "rate:{shot}:3",
                Action::Rate(Stars::new(3)),
            ),
            (Gesture::Yes, "yes:{shot}:{viewed}", Action::Yes(option(2))),
            (Gesture::No, "no:{shot}:{viewed}", Action::No(option(2))),
        ] {
            assert_eq!(gesture.template(), template);
            let filled = template.replace(SHOT_SLOT, "4").replace(VIEWED_SLOT, "2");
            assert_eq!(
                Command::parse(&filled, BOUNDS),
                Ok(Command::On(shot(4), command)),
                "{filled}"
            );
            assert_eq!(command.gesture(), gesture);
        }
    }

    /// Mutation: `Idx::new` accepts `n <= count`, or `parse` drops the digits-only check
    /// (`+1`, ` 1`, `0x1` would then parse via `usize::from_str` or be rejected for the
    /// wrong reason): the rows below go red.
    #[test]
    fn indices_outside_the_lists_are_refused() {
        for text in [
            "goto:5",
            "goto:-1",
            "goto:",
            "goto:+1",
            "goto: 1",
            "goto:1.0",
            "goto:../x",
            "view:0:3",
            "view:5:0",
            "view:0:99999999999999999999999999",
            "rate:0:6",
            "rate:0:10",
            "rate:0:",
            "rate:0:x",
            "rate:0:-1",
            "rate:5:1",
            "accept:0:3",
            "accept:5:0",
            "yes:5:0",
            "no:-1:0",
            "yes:0:3",
            "yes:0",
            "no:0",
            // The shot is part of every command that acts on one.
            "view:1",
            "rate:3",
            "accept",
            "accept:1",
            "flip",
            "yes",
            "no",
            "flip:1:2",
            "accept:1:2:3",
            "goto",
            "filter",
            "filter:",
            "filter:sideways",
            "filter:UP",
            "next:1",
            "scene",
            "scene:",
            "scene:sideways",
            "scene:next:1",
            "undecided",
            "undecided:UP",
            "page",
            "page:down",
            "page:forward:2",
            "first:1",
            "LAST",
            "",
            "nonsense",
            "NEXT",
        ] {
            assert!(
                Command::parse(text, BOUNDS).is_err(),
                "{text:?} was accepted"
            );
        }
    }

    #[test]
    fn an_empty_list_has_no_index() {
        assert_eq!(
            Command::parse(
                "goto:0",
                Bounds {
                    shots: 0,
                    options: 3
                }
            )
            .unwrap_err(),
            CommandError::Index {
                command: "goto",
                source: IndexError {
                    text: "0".into(),
                    count: 0
                }
            }
        );
    }
}
