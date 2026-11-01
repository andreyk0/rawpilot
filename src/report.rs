//! `.rawpilot/report.html`: the shoot as scenes → bursts → frames, with the measurements that
//! drove each decision and the camera's AF box drawn over the image. It is the tuning
//! instrument: a threshold is judged by looking at what it did here. Each frame is shown as
//! the judge was shown it: the neutral development when there is one, else the camera JPEG.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Result;

use crate::analyze::Analyzed;
use crate::cache::{self, Cache, Kind, write_atomic};
use crate::features::AfSharpness;
use crate::group::Scene;
use crate::pp3::Look;
use crate::rate::{Decision, Label};
use crate::shot::{Orientation, Shot};

/// `decisions` is indexed like `shots`.
pub fn write(cache: &Cache, shots: &[Analyzed], scenes: &[Scene], rows: &[Row]) -> Result<PathBuf> {
    assert_eq!(shots.len(), rows.len(), "one row per shot");
    let mut html = String::from(HEAD);
    for (si, scene) in scenes.iter().enumerate() {
        let first = &shots[scene.bursts()[0].members()[0]].shot;
        writeln!(
            html,
            "<h2>Scene {} <small>{} · {} bursts</small></h2>",
            si + 1,
            first.taken_at.format("%H:%M:%S"),
            scene.bursts().len()
        )?;
        for burst in scene.bursts() {
            html.push_str("<div class=burst>");
            for &i in burst.members() {
                card(&mut html, cache, &shots[i], &rows[i])?;
            }
            html.push_str("</div>\n");
        }
    }
    html.push_str("</body></html>\n");
    let path = cache.root().join("report.html");
    write_atomic(&path, html.as_bytes())?;
    Ok(path)
}

fn shown_kind(a: &Analyzed) -> Kind {
    if a.developed.is_some() {
        cache::DEVELOPED
    } else {
        cache::FEATURES
    }
}

/// What was decided about one shot, for its card.
pub struct Row<'a> {
    pub decision: &'a Decision,
    /// The faces pass's summary.
    pub note: Option<&'a str>,
    /// The exposure a pp3 created for it opens with.
    pub look: Option<Look>,
}

fn card(html: &mut String, cache: &Cache, a: &Analyzed, row: &Row) -> Result<()> {
    let d = row.decision;
    let look = row.look.map_or(String::new(), |l| {
        format!(
            "<br>opens in RawTherapee at {:+.2} EV, highlight compression {}",
            l.compensation_ev, l.highlight_compr
        )
    });
    // The faces pass's crops for this frame, when it made them: what the eyes were read from.
    // The pass leaves a strip exactly for the frames it found faces in (`sync_contact_strip`),
    // and a note exists only for frames it read, so a strip beside a note is this run's.
    let strip_rel = format!("faces/strips/{}.jpg", a.shot.name());
    let faces = match row.note {
        Some(n) if cache.root().join(&strip_rel).is_file() => format!(
            "<br><b>{}</b><img class=strip loading=lazy src=\"{}\">",
            escape(n),
            escape(&strip_rel)
        ),
        Some(n) => format!("<br><b>{}</b>", escape(n)),
        None => String::new(),
    };
    let f = &a.features;
    let af = match f.af {
        AfSharpness::NoBox => "AF —".to_owned(),
        AfSharpness::Featureless { texture } => format!("AF flat (tex {texture:.1})"),
        AfSharpness::Measured { sharpness, texture } => {
            format!("AF {sharpness:.0} (tex {texture:.1})")
        }
    };
    let ev = a
        .shot
        .exposure
        .compensation_ev
        .filter(|&ev| ev != 0.0)
        .map_or(String::new(), |ev| format!(" · {ev:+.1} EV"));
    let developed = a.developed.map_or(String::new(), |i| {
        format!(
            "<br>developed {:+.1} EV · raw clipped {:.2}%",
            i.push_ev,
            i.raw_clipped * 100.0
        )
    });
    let (stars, label) = d.bucket.rating();
    let colour = label.map_or("#555", |l| match l {
        Label::Red => "#d33",
        Label::Yellow => "#db3",
        Label::Green => "#3b5",
        Label::Blue => "#39e",
        Label::Purple => "#a5d",
    });
    write!(
        html,
        "<figure style=\"border-color:{colour}\"><div class=img><img loading=lazy src=\"{}\">{}</div>\
         <figcaption><b>{}</b> {}<br><span class=stars>{}</span> {:?} · sharp {:.0}%<br>\
         <i>{}</i><br>{af} · tiles {:.0}<br>JPEG clip hi {:.1}% lo {:.1}%{ev}{developed}{look}{faces}</figcaption></figure>",
        escape(&Cache::image_rel(shown_kind(a), &a.shot).to_string_lossy()),
        af_overlay(&a.shot),
        escape(&a.shot.name()),
        a.shot.taken_at.format("%H:%M:%S%.3f"),
        "★".repeat(stars.get().into()) + &"☆".repeat((5 - stars.get()).into()),
        d.bucket,
        d.sharpness_rel * 100.0,
        escape(&d.why),
        f.tiles_sharpness,
        f.clipping.highlights * 100.0,
        f.clipping.shadows * 100.0,
    )?;
    Ok(())
}

/// The AF box as a percentage-positioned div over the display-oriented thumbnail.
fn af_overlay(shot: &Shot) -> String {
    let Some(af) = shot.af else {
        return String::new();
    };
    let (sw, sh) = (f64::from(af.space_w), f64::from(af.space_h));
    let r = af.rect;
    let (x, y, w, h) = (
        f64::from(r.x) / sw,
        f64::from(r.y) / sh,
        f64::from(r.w) / sw,
        f64::from(r.h) / sh,
    );
    let (x, y, w, h) = match shot.orientation {
        Orientation::Normal => (x, y, w, h),
        Orientation::Rotate90Cw => (1.0 - (y + h), x, h, w),
        Orientation::Rotate180 => (1.0 - (x + w), 1.0 - (y + h), w, h),
        Orientation::Rotate270Cw => (y, 1.0 - (x + w), h, w),
    };
    format!(
        "<span class=af style=\"left:{:.2}%;top:{:.2}%;width:{:.2}%;height:{:.2}%\"></span>",
        x * 100.0,
        y * 100.0,
        w * 100.0,
        h * 100.0
    )
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

const HEAD: &str = r#"<!doctype html><html><head><meta charset=utf-8><title>rawpilot report</title>
<style>
body{font:13px system-ui,sans-serif;background:#1d1f21;color:#ddd;margin:16px}
h2{margin:24px 0 8px;font-weight:600}h2 small{color:#999;font-weight:400}
.burst{display:flex;flex-wrap:wrap;gap:6px;padding:6px;margin:6px 0;border-left:3px solid #555}
figure{margin:0;width:240px;border-top:4px solid #555}.stars{color:#fc4;letter-spacing:1px}
.img{position:relative}.img img{width:100%;display:block}
.af{position:absolute;border:2px solid #f33;box-sizing:border-box;min-width:6px;min-height:6px}
figcaption{padding:3px 0;line-height:1.35;color:#bbb}figcaption b{color:#eee}
.strip{display:block;max-width:100%;margin-top:3px}
</style></head><body><h1>rawpilot</h1>
"#;
