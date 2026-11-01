//! Which key asks for what.
//!
//! One table, built here, serves three readers: the page's script (the table travels in a
//! data attribute, and the script only looks a key up and sends the command), the help
//! overlay, and the tests. The script holds no key knowledge of its own.
//!
//! Option keys only *view* an option. Writing a look is `Enter`/`Space`, on the option on
//! screen and only after an option key was pressed for this shot (`model::Origin`), so a
//! stray keystroke, a held key or Caps Lock cannot choose anything.

use std::collections::BTreeMap;

use super::command::{Command, FilterWay, Gesture, Idx, Move, OfOption, OptionIdx};
use crate::rate::Stars;

/// What a key does once pressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Ask the server.
    Send(Command),
    /// Ask the server about the shot on screen; the script names the shot and the viewed
    /// option from the page it holds when it sends.
    OnShot(Gesture),
    /// Show or hide the help overlay; the page's own state, nothing is sent.
    Help,
}

impl KeyAction {
    fn wire(self) -> String {
        match self {
            KeyAction::Send(c) => c.to_string(),
            KeyAction::OnShot(g) => g.template(),
            KeyAction::Help => "help".to_owned(),
        }
    }
}

/// Which table applies: browsing the shoot, answering a question that blocks everything
/// else, or waiting for renders, when only the filter can change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Context {
    Browse,
    Answer,
    Wait,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Binding {
    /// `KeyboardEvent.key`.
    key: String,
    action: KeyAction,
}

/// The keys of one launch.
#[derive(Debug, Clone)]
pub struct Keymap {
    browse: Vec<Binding>,
    answer: Vec<Binding>,
    wait: Vec<Binding>,
    option_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KeyError {
    #[error("look option {option} is bound to {key:?}, which {taken_by}")]
    Taken {
        option: String,
        key: char,
        taken_by: String,
    },
}

/// A look option as the keys see it.
pub struct OptionKey<'a> {
    pub name: &'a str,
    pub key: char,
}

/// One line of the help overlay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpRow {
    pub keys: Vec<String>,
    pub what: String,
}

fn bind(keys: &[&str], action: KeyAction) -> Vec<Binding> {
    keys.iter()
        .map(|k| Binding {
            key: (*k).to_owned(),
            action,
        })
        .collect()
}

/// The browsing table before any option is added: what the UI binds whatever the
/// configuration says.
fn fixed_browse() -> Vec<Binding> {
    let send = KeyAction::Send;
    let on_shot = KeyAction::OnShot;
    let mut browse = Vec::new();
    let go = |m| send(Command::Move(m));
    browse.extend(bind(&["j", "ArrowRight"], go(Move::Next)));
    browse.extend(bind(&["k", "ArrowLeft"], go(Move::Previous)));
    browse.extend(bind(&["]"], go(Move::NextScene)));
    browse.extend(bind(&["["], go(Move::PreviousScene)));
    browse.extend(bind(&["J"], go(Move::NextUndecided)));
    browse.extend(bind(&["K"], go(Move::PreviousUndecided)));
    browse.extend(bind(&["Home"], go(Move::First)));
    browse.extend(bind(&["End"], go(Move::Last)));
    browse.extend(bind(&["PageDown"], go(Move::PageForward)));
    browse.extend(bind(&["PageUp"], go(Move::PageBack)));
    browse.extend(bind(&["Enter", " "], on_shot(Gesture::Accept)));
    browse.extend(bind(&["Tab"], on_shot(Gesture::Flip)));
    browse.extend(bind(&["f"], send(Command::StepFilter(FilterWay::Up))));
    browse.extend(bind(&["F"], send(Command::StepFilter(FilterWay::Down))));
    browse.extend(bind(&["?"], KeyAction::Help));
    for (digit, n) in ["0", "1", "2", "3", "4", "5"].into_iter().zip(0..) {
        browse.extend(bind(&[digit], on_shot(Gesture::Rate(Stars::new(n)))));
    }
    browse
}

/// The letters the UI binds itself, lower case: a look option's key cannot be one of them
/// (case does not matter, so `F` is `f`). Configuration is checked against this list
/// (`ReviewConfig::validate`), and so is the keymap.
pub fn reserved_letters() -> Vec<char> {
    let mut letters: Vec<char> = fixed_browse()
        .iter()
        .filter_map(|b| {
            let mut chars = b.key.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphabetic() => Some(c.to_ascii_lowercase()),
                _ => None,
            }
        })
        .collect();
    letters.sort_unstable();
    letters.dedup();
    letters
}

impl Keymap {
    /// The table for `options`, in the order of the configured options. An option whose
    /// letter another binding already has is refused.
    pub fn build(options: &[OptionKey<'_>]) -> Result<Keymap, KeyError> {
        let send = KeyAction::Send;
        let on_shot = KeyAction::OnShot;
        let mut browse = fixed_browse();
        for (i, option) in Idx::<OfOption>::all(options.len()).zip(options) {
            let key = option.key.to_ascii_lowercase().to_string();
            if let Some(taken) = browse.iter().find(|b| b.key == key) {
                return Err(KeyError::Taken {
                    option: option.name.to_owned(),
                    key: option.key,
                    taken_by: format!("already means {}", describe(taken.action, &[])),
                });
            }
            browse.extend(bind(&[&key], on_shot(Gesture::View(i))));
        }
        let mut answer = bind(&["y"], on_shot(Gesture::Yes));
        answer.extend(bind(&["n", "Escape"], on_shot(Gesture::No)));
        answer.extend(bind(&["?"], KeyAction::Help));
        let mut wait = bind(&["f"], send(Command::StepFilter(FilterWay::Up)));
        wait.extend(bind(&["F"], send(Command::StepFilter(FilterWay::Down))));
        wait.extend(bind(&["?"], KeyAction::Help));
        Ok(Keymap {
            browse,
            answer,
            wait,
            option_names: options.iter().map(|o| o.name.to_owned()).collect(),
        })
    }

    fn table(&self, context: Context) -> &[Binding] {
        match context {
            Context::Browse => &self.browse,
            Context::Answer => &self.answer,
            Context::Wait => &self.wait,
        }
    }

    /// The table as the script reads it: a JSON object from key to wire text.
    pub fn json(&self, context: Context) -> String {
        let map: BTreeMap<&str, String> = self
            .table(context)
            .iter()
            .map(|b| (b.key.as_str(), b.action.wire()))
            .collect();
        // A map of strings always serializes.
        serde_json::to_string(&map).expect("a string map serializes")
    }

    /// The overlay's lines: keys that do the same thing share one.
    pub fn help(&self, context: Context) -> Vec<HelpRow> {
        let mut rows: Vec<HelpRow> = Vec::new();
        for b in self.table(context) {
            let what = describe(b.action, &self.option_names);
            let shown = key_label(&b.key);
            match rows.iter_mut().find(|r| r.what == what) {
                Some(row) => row.keys.push(shown),
                None => rows.push(HelpRow {
                    keys: vec![shown],
                    what,
                }),
            }
        }
        rows
    }
}

fn key_label(key: &str) -> String {
    match key {
        " " => "Space".to_owned(),
        "ArrowRight" => "→".to_owned(),
        "ArrowLeft" => "←".to_owned(),
        other => other.to_owned(),
    }
}

fn describe(action: KeyAction, option_names: &[String]) -> String {
    let option = |i: OptionIdx| {
        option_names
            .get(i.get())
            .map_or_else(|| format!("option {i}"), Clone::clone)
    };
    let gesture = |g: Gesture| match g {
        Gesture::View(i) => format!("view {} (writes nothing)", option(i)),
        Gesture::Flip => "flip back to the option viewed before".to_owned(),
        Gesture::Accept => {
            "choose the viewed option for this shot, then next shot (after viewing one)".to_owned()
        }
        Gesture::Rate(_) => "set the rating, 0 to 5 stars".to_owned(),
        Gesture::Yes => "yes, replace the processing".to_owned(),
        Gesture::No => "no, keep it".to_owned(),
    };
    match action {
        KeyAction::Help => "show or hide this help".to_owned(),
        KeyAction::OnShot(g) => gesture(g),
        KeyAction::Send(command) => match command {
            Command::Move(m) => match m {
                Move::Next => "next shot".to_owned(),
                Move::Previous => "previous shot".to_owned(),
                Move::NextScene => "next scene (a pause in shooting starts one)".to_owned(),
                Move::PreviousScene => "start of this scene, then of the previous one".to_owned(),
                Move::NextUndecided => "next shot with no look chosen".to_owned(),
                Move::PreviousUndecided => "previous shot with no look chosen".to_owned(),
                Move::First => "first shot".to_owned(),
                Move::Last => "last shot".to_owned(),
                Move::PageForward => format!("{} shots on", Move::PAGE),
                Move::PageBack => format!("{} shots back", Move::PAGE),
            },
            Command::Goto(_) => "go to a shot".to_owned(),
            Command::On(_, action) => gesture(action.gesture()),
            Command::StepFilter(FilterWay::Up) => {
                "rating filter up one star: all, 2, 3, 4, 5 (stops at 5)".to_owned()
            }
            Command::StepFilter(FilterWay::Down) => {
                "rating filter down one star: 5, 4, 3, 2, all (stops at all)".to_owned()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    //! Goal: the key-to-command mapping, the part of the page that decides what a keystroke
    //! does. Method: the table built for three options, looked up with the `KeyboardEvent.key`
    //! values a browser sends; expectations are written out by hand.

    use super::*;
    use crate::review::command::{Bounds, Command, SHOT_SLOT, VIEWED_SLOT};

    fn options() -> [OptionKey<'static>; 3] {
        [
            OptionKey {
                name: "Camera",
                key: 'c',
            },
            OptionKey {
                name: "Film",
                key: 'm',
            },
            OptionKey {
                name: "Gentle",
                key: 'g',
            },
        ]
    }

    fn keymap() -> Keymap {
        Keymap::build(&options()).unwrap()
    }

    /// What the script does for `key`: it looks the key up in the table the page carries.
    fn sent(context: Context, key: &str) -> Option<String> {
        let table: BTreeMap<String, String> =
            serde_json::from_str(&keymap().json(context)).unwrap();
        table.get(key).cloned()
    }

    /// Mutation: bind `j` and `ArrowRight` to `Previous`: the `j` and `ArrowRight` rows go
    /// red.
    #[test]
    fn browsing_keys_ask_for_what_the_help_says() {
        for (key, wire) in [
            ("j", "next"),
            ("ArrowRight", "next"),
            ("k", "prev"),
            ("ArrowLeft", "prev"),
            ("]", "scene:next"),
            ("[", "scene:prev"),
            ("J", "undecided:next"),
            ("K", "undecided:prev"),
            ("Home", "first"),
            ("End", "last"),
            ("PageDown", "page:forward"),
            ("PageUp", "page:back"),
            ("Tab", "flip:{shot}"),
            ("f", "filter:up"),
            ("F", "filter:down"),
            ("?", "help"),
            ("Enter", "accept:{shot}:{viewed}"),
            (" ", "accept:{shot}:{viewed}"),
            ("0", "rate:{shot}:0"),
            ("3", "rate:{shot}:3"),
            ("5", "rate:{shot}:5"),
            ("c", "view:{shot}:0"),
            ("m", "view:{shot}:1"),
            ("g", "view:{shot}:2"),
        ] {
            assert_eq!(sent(Context::Browse, key).as_deref(), Some(wire), "{key}");
        }
    }

    /// Mutation: bind each option's uppercase letter too (Caps Lock, Shift): the `C`, `M`
    /// rows go red.
    #[test]
    fn uppercase_and_unbound_keys_do_nothing() {
        for key in ["C", "M", "Y", "6", "a", "v", "Escape", "y", "n", "x"] {
            // `y`, `n` and Escape answer a question, which browsing is not.
            assert_eq!(sent(Context::Browse, key), None, "{key}");
        }
    }

    /// Mutation: add `j` to the answer table: its row goes red.
    #[test]
    fn a_question_leaves_only_its_answers() {
        assert_eq!(
            sent(Context::Answer, "y").as_deref(),
            Some("yes:{shot}:{viewed}")
        );
        assert_eq!(
            sent(Context::Answer, "n").as_deref(),
            Some("no:{shot}:{viewed}")
        );
        assert_eq!(
            sent(Context::Answer, "Escape").as_deref(),
            Some("no:{shot}:{viewed}")
        );
        for key in ["j", "ArrowRight", "Enter", " ", "c", "3", "f", "F", "Tab"] {
            assert_eq!(sent(Context::Answer, key), None, "{key}");
        }
    }

    /// Moving is for browsing only: not while a question is open, not while the page waits
    /// for renders. Mutation: bind `J` or `PageDown` in the answer or wait table: its row
    /// goes red.
    #[test]
    fn the_new_moves_work_only_while_browsing() {
        for key in ["]", "[", "J", "K", "Home", "End", "PageDown", "PageUp"] {
            assert_eq!(sent(Context::Answer, key), None, "answer {key}");
            assert_eq!(sent(Context::Wait, key), None, "wait {key}");
            assert!(sent(Context::Browse, key).is_some(), "browse {key}");
        }
    }

    /// Mutation: skip the collision check in `build`.
    #[test]
    fn an_option_cannot_take_a_key_the_ui_uses() {
        for taken in ['j', 'k', 'f'] {
            let clash = [OptionKey {
                name: "Film",
                key: taken,
            }];
            assert!(
                matches!(Keymap::build(&clash), Err(KeyError::Taken { .. })),
                "{taken}"
            );
        }
        // Case does not make a different key: config rejects two options on one letter.
        assert!(
            Keymap::build(&[OptionKey {
                name: "Film",
                key: 'F'
            }])
            .is_err()
        );
    }

    /// The letters the configuration refuses are exactly the ones the keymap does: neither
    /// can drift from the other. Mutation: bind another letter in `fixed_browse` without
    /// listing it (or list one it does not bind): the row for that letter goes red.
    #[test]
    fn the_reserved_letters_are_the_ones_the_keymap_refuses() {
        let reserved = reserved_letters();
        assert_eq!(reserved, ['f', 'j', 'k']);
        for letter in 'a'..='z' {
            let taken = Keymap::build(&[OptionKey {
                name: "Film",
                key: letter,
            }])
            .is_err();
            assert_eq!(taken, reserved.contains(&letter), "{letter}");
        }
    }

    #[test]
    fn every_wire_text_in_the_table_parses_back() {
        let bounds = Bounds {
            shots: 1,
            options: 3,
        };
        let on_screen = Idx::new(0, bounds.shots).unwrap();
        for context in [Context::Browse, Context::Answer, Context::Wait] {
            for b in keymap().table(context) {
                match b.action {
                    KeyAction::Send(c) => {
                        assert_eq!(Command::parse(&c.to_string(), bounds), Ok(c));
                    }
                    KeyAction::OnShot(g) => {
                        // What the script does: fill the slots from the page.
                        let sent = g
                            .template()
                            .replace(SHOT_SLOT, "0")
                            .replace(VIEWED_SLOT, "2");
                        let parsed = Command::parse(&sent, bounds).unwrap();
                        assert_eq!(parsed.to_string(), sent);
                        assert!(matches!(parsed, Command::On(s, _) if s == on_screen));
                    }
                    KeyAction::Help => {}
                }
            }
        }
    }

    #[test]
    fn the_help_groups_keys_that_do_the_same_thing() {
        let rows = keymap().help(Context::Browse);
        let row = |what: &str| rows.iter().find(|r| r.what == what).unwrap().keys.clone();
        assert_eq!(row("next shot"), ["j", "→"]);
        assert_eq!(row("next scene (a pause in shooting starts one)"), ["]"]);
        assert_eq!(row("start of this scene, then of the previous one"), ["["]);
        assert_eq!(row("next shot with no look chosen"), ["J"]);
        assert_eq!(row("previous shot with no look chosen"), ["K"]);
        assert_eq!(row("first shot"), ["Home"]);
        assert_eq!(row("last shot"), ["End"]);
        assert_eq!(row("10 shots on"), ["PageDown"]);
        assert_eq!(row("10 shots back"), ["PageUp"]);
        assert_eq!(
            row("choose the viewed option for this shot, then next shot (after viewing one)"),
            ["Enter", "Space"]
        );
        assert_eq!(
            row("set the rating, 0 to 5 stars"),
            ["0", "1", "2", "3", "4", "5"]
        );
        assert_eq!(row("view Film (writes nothing)"), ["m"]);
        assert_eq!(
            row("rating filter up one star: all, 2, 3, 4, 5 (stops at 5)"),
            ["f"]
        );
        assert_eq!(
            row("rating filter down one star: 5, 4, 3, 2, all (stops at all)"),
            ["F"]
        );
    }
}
