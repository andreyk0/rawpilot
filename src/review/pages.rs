//! The pages, drawn from the [`Model`] alone: the same model always draws the same markup.
//!
//! `app` is the one element the script replaces after every command and every poll; `shell`
//! wraps it once. The script holds no state of its own beyond whether the help overlay is
//! open.

use std::collections::VecDeque;

use maud::{DOCTYPE, Markup, html};

use super::command::{Action, Command, OptionIdx, ShotIdx};
use super::images::{self, ImgRef};
use super::keys::{Context, Keymap};
use super::model::{
    Activity, Confirmation, Filter, Mode, Model, Notice, Origin, Progress, Stage, Wait,
};
use crate::config::{CentreWidthPct, ReviewConfig};
use crate::looks::RegionView;
use crate::rate::Stars;
use crate::regions::Region;
use crate::rpl::Pp3State;

/// The sizes `[review]` makes tunable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    pub region_zoom: f64,
    pub centre_width_pct: CentreWidthPct,
    pub thumb_height_px: u32,
    pub filmstrip_reach: usize,
}

impl Layout {
    pub fn of(cfg: &ReviewConfig) -> Layout {
        Layout {
            region_zoom: cfg.region_zoom,
            centre_width_pct: cfg.centre_width_pct,
            thumb_height_px: cfg.thumb_height_px,
            filmstrip_reach: cfg.filmstrip_reach,
        }
    }

    /// The custom properties `app.css` reads. Served as a stylesheet of its own because the
    /// page's content-security policy admits no inline style.
    pub fn css(&self) -> String {
        format!(
            ":root {{ --centre-width: {}vw; --thumb-height: {}px; }}\n",
            self.centre_width_pct.get(),
            self.thumb_height_px
        )
    }
}

pub const NAME: &str = "rawpilot review";

pub fn shell(app: &Markup) -> Markup {
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                title { (NAME) }
                link rel="stylesheet" href="/static/layout.css";
                link rel="stylesheet" href="/static/app.css";
            }
            body {
                (app)
                script src="/static/app.js" {}
            }
        }
    }
}

pub fn app(model: &Model, keys: &Keymap, layout: &Layout) -> Markup {
    let context = model.context();
    // What the script names in the commands it sends, so that a page which is no longer the
    // current one cannot act on the shot it shows.
    let stage = model.stage();
    let on_screen = match stage {
        Stage::Working => model.current(),
        Stage::Waiting(_) | Stage::Broken(_) => None,
    };
    // The option the page's commands are about: the one in the question while it is open.
    let viewed = match model.mode() {
        Mode::Confirming(c) => c.option,
        Mode::Browse => model.viewing().shown,
    };
    // The script polls while renders are owed, which is also while the page is waiting.
    let polling = model.activity() != Activity::Idle;
    html! {
        div #app data-stage=(stage_name(&stage)) data-polling=[polling.then_some("")]
            data-keys=(keys.json(context))
            data-shot=[on_screen] data-viewed=[on_screen.map(|_| viewed)] {
            @match &stage {
                Stage::Working => (working(model, layout)),
                Stage::Waiting(Wait::Profile) => (waiting(model, "Preparing", "Resolving RawTherapee's default profile (about ten seconds).", None)),
                Stage::Waiting(Wait::Renders(p)) => (rendering(model, p)),
                Stage::Broken(why) => (broken(model, why)),
            }
            @if let Some(notice) = model.notice() { (notice_line(notice)) }
            @if let Mode::Confirming(c) = model.mode() { (confirmation(model, c)) }
            (help(keys, context))
        }
    }
}

fn stage_name(stage: &Stage) -> &'static str {
    match stage {
        Stage::Waiting(_) => "waiting",
        Stage::Working => "working",
        Stage::Broken(_) => "broken",
    }
}

fn rt_banner() -> Markup {
    html! {
        span.rt-banner title="RawTherapee may write back the pp3 files it holds in memory over what is chosen here." {
            "Close RawTherapee while reviewing"
        }
    }
}

/// The top bar: what the photographer reads sits in the middle, level with the picture; the
/// minor items hold the edges.
fn bar(left: Markup, centre: Markup, right: Markup) -> Markup {
    html! {
        header.bar {
            div.bar-side { (left) }
            div.bar-centre { (centre) }
            div.bar-side.end { (right) }
        }
    }
}

/// The bar of a page with no shot to show.
fn title_bar(filter: Filter) -> Markup {
    bar(
        html! { (filter_chip(filter)) },
        html! { span.name { (NAME) } },
        rt_banner(),
    )
}

fn filter_chip(filter: Filter) -> Markup {
    html! {
        button.chip type="button" data-cmd="filter:up" title="Rating filter: f one star up, F one star down" {
            "show " b { (filter.label()) }
        }
    }
}

fn waiting(model: &Model, title: &str, detail: &str, progress: Option<&Progress>) -> Markup {
    html! {
        (title_bar(model.filter()))
        main.waiting data-view="preprocessing" {
            h1 { (title) }
            p { (detail) }
            @if let Some(p) = progress {
                progress max=(p.total) value=(p.done) {}
            }
        }
    }
}

/// The page while no shot the filter admits is rendered yet. It gives way to the working
/// view when the first one is.
fn rendering(model: &Model, p: &Progress) -> Markup {
    let title = format!("Rendering {}/{}", p.done, p.total);
    let detail = p.last.as_deref().map_or_else(
        || {
            format!(
                "Rendering the look options of the shots the filter ({}) admits, around the first.",
                model.filter().label()
            )
        },
        |last| format!("Last finished: {last}"),
    );
    waiting(model, &title, &detail, Some(p))
}

fn broken(model: &Model, why: &str) -> Markup {
    html! {
        (title_bar(model.filter()))
        main.waiting data-view="broken" {
            h1 { "Cannot render" }
            p.error { (why) }
            p { "Press " kbd { "f" } " or " kbd { "F" } " to look at shots that are already rendered." }
        }
    }
}

/// What the renderer is doing, for the bar of the working view; the script updates this
/// element alone when only it changes between two polls.
fn activity_note(activity: &Activity) -> Markup {
    html! {
        @match activity {
            Activity::Idle => span #activity.ready { "Ready" },
            Activity::PreparingProfile => span #activity.busy { "preparing RawTherapee's default profile" },
            Activity::Rendering(p) => span #activity.busy { "rendering " (p.done) "/" (p.total) },
            Activity::Stopped(why) => span #activity.stopped title=(why) { "rendering stopped" },
        }
    }
}

fn notice_line(notice: &Notice) -> Markup {
    let (class, text) = match notice {
        Notice::Error(e) => ("notice error", e),
        Notice::Hint(h) => ("notice hint", h),
    };
    html! { div class=(class) role="status" { (text) } }
}

fn confirmation(model: &Model, c: Confirmation) -> Markup {
    let answer = |action| Command::On(c.shot, action).to_string();
    html! {
        div.confirm role="alertdialog" aria-label="Replace processing?" {
            div.card {
                p {
                    "This image's processing was edited in RawTherapee; replace it with "
                    b { (model.option(c.option).name) } "?"
                }
                button type="button" data-cmd=(answer(Action::Yes(c.option))) { kbd { "y" } " replace" }
                button type="button" data-cmd=(answer(Action::No(c.option))) { kbd { "n" } " keep" }
            }
        }
    }
}

fn help(keys: &Keymap, context: Context) -> Markup {
    html! {
        div.help-overlay role="dialog" aria-label="Keys" {
            div.card {
                h2 { "Keys" }
                table {
                    @for row in keys.help(context) {
                        tr {
                            td { @for k in &row.keys { kbd { (k) } " " } }
                            td { (row.what) }
                        }
                    }
                }
                p.dim { "Option keys only view; Enter or Space writes the viewed option." }
            }
        }
    }
}

fn stars(rating: u8) -> String {
    (1..=5)
        .map(|n| if n <= rating { '★' } else { '☆' })
        .collect()
}

/// What the shot's pp3 says about its look, as a short tag.
fn look_tag(model: &Model, shot: ShotIdx) -> Option<String> {
    match &model.entry(shot).pp3 {
        Pp3State::Chosen { option } => Some(format!("✓ {option}")),
        Pp3State::Interrupted { option } => Some(format!("interrupted: {option}")),
        Pp3State::HandEdited => Some("edited in RT".to_owned()),
        Pp3State::Absent | Pp3State::Generated => None,
    }
}

/// The shots the filter admits that cannot be shown because a render of theirs failed.
fn failure_note(model: &Model) -> Markup {
    let skipped = model.skipped();
    html! {
        @if !skipped.is_empty() {
            div.notice.hint {
                (skipped.len()) " shot(s) at this filter are not shown because a render failed: "
                @for (i, (s, o, why)) in model.failures().filter(|(s, _, _)| skipped.contains(s)).enumerate() {
                    @if i > 0 { "; " }
                    (model.entry(s).name()) " " (model.option(o).name) ": " (why)
                }
            }
        }
    }
}

fn working(model: &Model, layout: &Layout) -> Markup {
    let visible = model.visible();
    let Some(shot) = model.current() else {
        return html! {
            (title_bar(model.filter()))
            (failure_note(model))
            main.waiting data-view="empty" {
                h1 { "No shots at " (model.filter().label()) }
                p { "Press " kbd { "f" } " or " kbd { "F" } " to change the rating filter." }
            }
        };
    };
    let position = visible.iter().position(|&s| s == shot).map_or(0, |i| i + 1);
    let (scene, scenes) = model.scene_of(shot);
    let entry = model.entry(shot);
    let viewing = model.viewing();
    let chosen = model.chosen(shot);
    html! {
        (bar(
            html! {
                span.pos { (position) " / " (visible.len()) }
                span.scene title="Scenes split where shooting paused; [ and ] move between them" { "scene " (scene) "/" (scenes) }
                (filter_chip(model.filter()))
                span.keys-hint { kbd { "?" } " keys" }
            },
            html! {
                div.bar-main {
                    span.name { (entry.name()) }
                    span.rating { @for n in 1..=5_u8 {
                        button.star.on[n <= entry.rating.get()] type="button" data-cmd=(Command::On(shot, Action::Rate(Stars::new(n)))) { (if n <= entry.rating.get() { "★" } else { "☆" }) }
                    } }
                    nav.options {
                        @for option in model.option_indices() {
                            @let info = model.option(option);
                            button.opt.shown[option == viewing.shown].chosen[Some(option) == chosen]
                                type="button" data-cmd=(Command::On(shot, Action::View(option))) {
                                kbd { (info.key.to_ascii_lowercase()) } " " (info.name)
                                @if Some(option) == chosen { span.tag { "chosen" } }
                            }
                        }
                    }
                }
                span.state { (state_text(model, shot)) }
            },
            html! {
                (activity_note(&model.activity()))
                (rt_banner())
            },
        ))
        (failure_note(model))
        main data-view="working" {
            @if let Some(render) = model.render(shot, viewing.shown) {
                @let key = &render.manifest.key;
                @let crops = CropColumns::of(&render.manifest.regions);
                @let names = region_names(&render.manifest.regions);
                aside.regions.left { @for &c in &crops.left { (crop(shot, viewing.shown, key, c, &names[c.0], layout)) } }
                section.fit {
                    img src=(images::url(shot, viewing.shown, ImgRef::Fit, key))
                        width=(render.manifest.fit.w) height=(render.manifest.fit.h)
                        alt=(format!("{} through {}", entry.name(), model.option(viewing.shown).name));
                }
                aside.regions.right { @for &c in &crops.right { (crop(shot, viewing.shown, key, c, &names[c.0], layout)) } }
            }
        }
        (filmstrip(model, &visible, shot, layout))
        (prefetch(model, &visible, shot))
    }
}

/// Which of a shot's region crops sit left and right of the fit view, each column listed
/// top to bottom. Crop 0 is the most important. Crops go right, left, right, ... in order of
/// importance, so the two sides stay level, and within a column the most important sits at
/// the middle line, where the eyes rest, with the next ones alternately below and above it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CropColumns<'a, T> {
    left: Vec<(usize, &'a T)>,
    right: Vec<(usize, &'a T)>,
}

impl<'a, T> CropColumns<'a, T> {
    /// Each crop comes with its position in `crops`, which names its picture.
    fn of(crops: &'a [T]) -> CropColumns<'a, T> {
        let ranked = |first: usize| crops.iter().enumerate().skip(first).step_by(2);
        CropColumns {
            left: CropColumns::middle_out(ranked(1)),
            right: CropColumns::middle_out(ranked(0)),
        }
    }

    /// `ranked`, most important first, arranged so the first is in the middle of the
    /// column's centred stack, the second just below it, the third just above, and so on.
    fn middle_out(ranked: impl Iterator<Item = (usize, &'a T)>) -> Vec<(usize, &'a T)> {
        let mut column = VecDeque::new();
        for (rank, crop) in ranked.enumerate() {
            if rank % 2 == 1 {
                column.push_back(crop);
            } else {
                column.push_front(crop);
            }
        }
        column.into()
    }
}

fn crop(
    shot: ShotIdx,
    option: OptionIdx,
    key: &str,
    (n, region): (usize, &RegionView),
    name: &str,
    layout: &Layout,
) -> Markup {
    html! {
        figure {
            img src=(images::url(shot, option, ImgRef::Region(n), key))
                width=(zoomed(region.view.w, layout)) height=(zoomed(region.view.h, layout))
                alt=(name);
            figcaption { (name) }
        }
    }
}

fn zoomed(px: u32, layout: &Layout) -> u32 {
    // Rounded to a whole CSS pixel; the product of two small positive numbers.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let scaled = (f64::from(px) * layout.region_zoom).round() as u32;
    scaled
}

/// The caption of each of a shot's regions, in their order. Faces are numbered from the left
/// of the frame, so the numbers read in the order the photographer sees the faces.
fn region_names(regions: &[RegionView]) -> Vec<String> {
    let face_left = |r: &RegionView| match r.region {
        Region::Face { at, .. } => Some(at.x0),
        Region::Af { .. } | Region::Centre => None,
    };
    regions
        .iter()
        .map(|r| match r.region {
            Region::Af { .. } => "Focus (AF)".to_owned(),
            Region::Face { at, focus, .. } => {
                let nth = 1 + regions
                    .iter()
                    .filter_map(face_left)
                    .filter(|&x0| x0 < at.x0)
                    .count();
                match focus {
                    Some(_) => format!("Face {nth} · focus"),
                    None => format!("Face {nth}"),
                }
            }
            Region::Centre => "Centre".to_owned(),
        })
        .collect()
}

/// Says which option is on screen and whether it is what is written.
fn state_text(model: &Model, shot: ShotIdx) -> String {
    let viewing = model.viewing();
    let shown = &model.option(viewing.shown).name;
    match (model.chosen(shot), viewing.origin) {
        (Some(c), _) if c == viewing.shown => format!("Viewing {shown}, which is chosen."),
        (chosen, Origin::Picked) => {
            let was = chosen.map_or_else(
                || "nothing is chosen".to_owned(),
                |c| format!("{} is chosen", model.option(c).name),
            );
            format!("Previewing {shown}; {was}. Enter chooses {shown}.")
        }
        (Some(c), Origin::Arrived) => {
            format!("Viewing {shown}; {} is chosen.", model.option(c).name)
        }
        (None, Origin::Arrived) => format!("Viewing {shown}; nothing is chosen."),
    }
}

fn filmstrip(model: &Model, visible: &[ShotIdx], current: ShotIdx, layout: &Layout) -> Markup {
    let at = visible.iter().position(|&s| s == current).unwrap_or(0);
    let from = at.saturating_sub(layout.filmstrip_reach);
    let to = (at + layout.filmstrip_reach + 1).min(visible.len());
    let window = visible.get(from..to).unwrap_or_default();
    let first_option = model.option_indices().next();
    html! {
        footer.strip {
            @for &shot in window {
                @let entry = model.entry(shot);
                @let option = model.chosen(shot).or(first_option);
                a.thumb.current[shot == current] data-cmd=(format!("goto:{shot}")) title=(entry.name()) {
                    @if let Some(render) = option.and_then(|o| model.render(shot, o).map(|r| (o, r))) {
                        @let (o, r) = render;
                        img src=(images::url(shot, o, ImgRef::Fit, &r.manifest.key))
                            height=(layout.thumb_height_px)
                            width=(thumb_width(r.manifest.fit.w, r.manifest.fit.h, layout))
                            loading="lazy" decoding="async" alt=(entry.name());
                    }
                    span.meta {
                        (stars(entry.rating.get()))
                        @if let Some(tag) = look_tag(model, shot) { " " span.look { (tag) } }
                    }
                }
            }
        }
    }
}

fn thumb_width(w: u32, h: u32, layout: &Layout) -> u32 {
    (u64::from(layout.thumb_height_px) * u64::from(w))
        .checked_div(u64::from(h.max(1)))
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(layout.thumb_height_px)
}

/// The neighbours' pictures for the option on screen, fetched ahead so stepping shows them
/// without a wait.
fn prefetch(model: &Model, visible: &[ShotIdx], current: ShotIdx) -> Markup {
    let shown = model.viewing().shown;
    let at = visible.iter().position(|&s| s == current).unwrap_or(0);
    let neighbours = [at.checked_sub(1), Some(at + 1)]
        .into_iter()
        .flatten()
        .filter_map(|i| visible.get(i).copied());
    html! {
        div.prefetch hidden {
            @for shot in neighbours {
                @if let Some(r) = model.render(shot, shown) {
                    img src=(images::url(shot, shown, ImgRef::Fit, &r.manifest.key)) alt="";
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Goal: the crops of a shot go to both sides of the fit view, the most important
    //! nearest the middle, and the layout stylesheet carries the configured centre share.
    //! Method: hand-written expectations from the rule "crop 0 right, crop 1 left, ...;
    //! within a column the first is at the middle, the second below it, the third above".

    use super::*;
    use crate::facecrop::Box2d;
    use crate::looks::View;
    use crate::shot::PixelRect;

    /// Faces are numbered by where they stand in the frame, not by their place in the list
    /// (the focused one comes first) or by the faces pass's person ids.
    #[test]
    fn faces_are_numbered_from_the_left_and_the_focused_one_says_so() {
        let at = |x0: f64| Box2d {
            x0,
            y0: 0.3,
            x1: x0 + 0.1,
            y1: 0.5,
        };
        let view = |region| RegionView {
            region,
            crop: at(0.0),
            rect: PixelRect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            view: View {
                file: String::new(),
                w: 1,
                h: 1,
            },
        };
        let face = |person, x0, focus| {
            view(Region::Face {
                person,
                at: at(x0),
                focus,
            })
        };
        let regions = [
            face(1, 0.55, Some(at(0.58))),
            face(0, 0.1, None),
            face(3, 0.3, None),
        ];
        assert_eq!(
            region_names(&regions),
            ["Face 3 · focus", "Face 1", "Face 2"]
        );
        assert_eq!(
            region_names(&[view(Region::Af { at: at(0.4) }), face(2, 0.2, None)]),
            ["Focus (AF)", "Face 1"]
        );
    }

    /// The crop numbers of each column, top to bottom, of `count` crops.
    fn columns(count: usize) -> (Vec<usize>, Vec<usize>) {
        let crops = vec![(); count];
        let c = CropColumns::of(&crops);
        let numbers = |col: Vec<(usize, &())>| col.into_iter().map(|(n, ())| n).collect();
        (numbers(c.left), numbers(c.right))
    }

    /// Mutation: send even crops left, or push the second crop to the front of its column:
    /// the 1-, 2- and 3-crop rows go red.
    #[test]
    fn crops_go_to_both_sides_most_important_at_the_middle() {
        for (count, left, right) in [
            (0, vec![], vec![]),
            (1, vec![], vec![0]),
            (2, vec![1], vec![0]),
            (3, vec![1], vec![0, 2]),
            (4, vec![1, 3], vec![0, 2]),
            (5, vec![1, 3], vec![4, 0, 2]),
        ] {
            assert_eq!(columns(count), (left, right), "{count} crops");
        }
    }

    /// Mutation: drop the `vw` unit or format the share from another field: red.
    #[test]
    fn the_stylesheet_carries_the_centre_share_and_thumbnail_height() {
        let cfg = crate::config::Config::load(None).unwrap().review;
        assert_eq!(
            Layout::of(&cfg).css(),
            ":root { --centre-width: 60vw; --thumb-height: 84px; }\n"
        );
    }
}
