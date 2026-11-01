//! The 1:1 regions cut from a full-size render for the review UI: where the camera focused,
//! where the faces pass found faces, or the centre when there is neither.
//!
//! Regions are decided in fractions of the displayed (oriented) image, which is what
//! `rawtherapee-cli` writes: it applies the NEF's EXIF orientation itself, so a render of an
//! orientation-8 frame is portrait. The camera's AF box is in the sensor's orientation and is
//! turned into the displayed frame here (`Orientation::displayed_rect`); Gemini's face boxes
//! already are displayed fractions. The pixels of a region are worked out from the size of the
//! render once it exists (`Cut::of`), so the regions that go into a cache key do not depend
//! on a size that is only known afterwards.

use serde::{Deserialize, Serialize};

use crate::config::ReviewConfig;
use crate::facecrop::Box2d;
use crate::rpl::FaceBox;
use crate::shot::{PixelRect, Shot};

/// A subject worth looking at 1:1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Region {
    /// The camera's AF box, when its centre is on none of the located faces.
    Af { at: Box2d },
    /// A face the faces pass located. `focus` is the AF box when its centre is on this face,
    /// which then has no region of its own (the face's region already shows it 1:1).
    Face {
        person: usize,
        at: Box2d,
        #[serde(default)]
        focus: Option<Box2d>,
    },
    /// Neither of the above exists: the middle of the frame.
    Centre,
}

/// The settings that size a region; part of every render's cache key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Sizing {
    pub af_min_px: u32,
    pub face_pad_fraction: f64,
    pub centre_px: u32,
    pub max_px: u32,
}

impl Sizing {
    pub fn of(cfg: &ReviewConfig) -> Sizing {
        Sizing {
            af_min_px: cfg.af_region_min_px,
            face_pad_fraction: cfg.face_pad_fraction,
            centre_px: cfg.centre_region_px,
            max_px: cfg.region_max_px,
        }
    }
}

/// The regions of `shot`, most important first: where the camera focused (the face its AF
/// box is on, or the AF box alone), then the other located faces, left to right; the centre
/// when that leaves none. `faces` is `None` when the faces pass never read the frame, which is
/// not the same as reading it and finding none, but plans the same regions.
pub fn plan(shot: &Shot, faces: Option<&[FaceBox]>) -> Vec<Region> {
    let af = shot.af.map(|af| {
        let (sw, sh) = (af.space_w, af.space_h);
        let shown = shot.orientation.displayed_rect(af.in_frame(sw, sh), sw, sh);
        let (dw, dh) = shot.orientation.displayed_size(sw, sh);
        fractions(shown, dw, dh)
    });
    let mut faces: Vec<&FaceBox> = faces.unwrap_or_default().iter().collect();
    faces.sort_by(|a, b| a.at.x0.total_cmp(&b.at.x0));
    let focused = af.and_then(|af| faces.iter().position(|f| holds_centre_of(&f.at, &af)));
    let mut regions: Vec<Region> = faces
        .iter()
        .enumerate()
        .map(|(i, f)| Region::Face {
            person: f.person,
            at: f.at,
            focus: af.filter(|_| focused == Some(i)),
        })
        .collect();
    match (focused, af) {
        (Some(i), _) => regions[..=i].rotate_right(1),
        (None, Some(at)) => regions.insert(0, Region::Af { at }),
        (None, None) => {}
    }
    if regions.is_empty() {
        vec![Region::Centre]
    } else {
        regions
    }
}

fn holds_centre_of(face: &Box2d, af: &Box2d) -> bool {
    let (cx, cy) = ((af.x0 + af.x1) / 2.0, (af.y0 + af.y1) / 2.0);
    (face.x0..=face.x1).contains(&cx) && (face.y0..=face.y1).contains(&cy)
}

fn fractions(r: PixelRect, w: u32, h: u32) -> Box2d {
    let f = |v: u32, of: u32| f64::from(v) / f64::from(of);
    Box2d {
        x0: f(r.x, w),
        y0: f(r.y, h),
        x1: f(r.x + r.w, w),
        y1: f(r.y + r.h, h),
    }
}

/// Where a region lies in a render of a known size, and how big its file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cut {
    /// The pixels of the render the region covers.
    pub rect: PixelRect,
    /// The size of the region's file: `rect`'s, scaled down when it exceeds `Sizing::max_px`.
    pub out_w: u32,
    pub out_h: u32,
}

impl Cut {
    /// `region` in a `w×h` render. The rectangle always lies inside the render, moved rather
    /// than shrunk when it would cross an edge, and shrunk only when the render is smaller.
    pub fn of(region: &Region, (w, h): (u32, u32), s: &Sizing) -> Cut {
        let (cx, cy, want_w, want_h) = match region {
            Region::Af { at } => {
                let (bw, bh) = (extent(at.x0, at.x1, w), extent(at.y0, at.y1, h));
                (
                    centre(at.x0, at.x1, w),
                    centre(at.y0, at.y1, h),
                    bw.max(s.af_min_px),
                    bh.max(s.af_min_px),
                )
            }
            Region::Face { at, .. } => {
                let grow = 1.0 + 2.0 * s.face_pad_fraction;
                (
                    centre(at.x0, at.x1, w),
                    centre(at.y0, at.y1, h),
                    (f64::from(extent(at.x0, at.x1, w)) * grow).ceil() as u32,
                    (f64::from(extent(at.y0, at.y1, h)) * grow).ceil() as u32,
                )
            }
            Region::Centre => (
                f64::from(w) / 2.0,
                f64::from(h) / 2.0,
                s.centre_px,
                s.centre_px,
            ),
        };
        let (rw, rh) = (want_w.clamp(1, w), want_h.clamp(1, h));
        let place = |c: f64, len: u32, limit: u32| {
            let start = (c - f64::from(len) / 2.0).round().max(0.0) as u32;
            start.min(limit - len)
        };
        let rect = PixelRect {
            x: place(cx, rw, w),
            y: place(cy, rh, h),
            w: rw,
            h: rh,
        };
        let (out_w, out_h) = scaled_down(rw, rh, s.max_px);
        Cut { rect, out_w, out_h }
    }

    /// The region as a box of the displayed image of `w×h` pixels.
    pub fn as_box(&self, (w, h): (u32, u32)) -> Box2d {
        fractions(self.rect, w, h)
    }
}

/// Pixels spanned by `[a, b]` fractions of `len` pixels, at least one.
fn extent(a: f64, b: f64, len: u32) -> u32 {
    (((b - a) * f64::from(len)).round() as u32).max(1)
}

fn centre(a: f64, b: f64, len: u32) -> f64 {
    (a + b) / 2.0 * f64::from(len)
}

/// `w×h` scaled, keeping its proportions, so that neither side exceeds `max`; unchanged when
/// it already fits.
pub fn scaled_down(w: u32, h: u32, max: u32) -> (u32, u32) {
    let longest = w.max(h);
    if longest <= max {
        return (w, h);
    }
    let s = f64::from(max) / f64::from(longest);
    let side = |v: u32| ((f64::from(v) * s).round() as u32).max(1);
    (side(w), side(h))
}

#[cfg(test)]
mod tests {
    //! Goal: the regions land on the subject in the displayed image whatever the frame's
    //! orientation, are padded, capped and clamped as configured, and fall back to the
    //! centre. Method: expected rectangles are derived by hand from the geometry (comments
    //! show the arithmetic); orientation is checked against `image`'s own rotation.

    use super::*;
    use crate::shot::{AfBox, Exposure, Orientation};

    fn sizing() -> Sizing {
        Sizing {
            af_min_px: 512,
            face_pad_fraction: 0.25,
            centre_px: 1024,
            max_px: 1200,
        }
    }

    fn shot(af: Option<AfBox>, orientation: Orientation) -> Shot {
        Shot {
            file: "x.NEF".into(),
            file_size: 0,
            taken_at: chrono::DateTime::parse_from_rfc3339("2026-10-18T14:00:00+00:00").unwrap(),
            preview: crate::shot::PreviewSpan { offset: 0, len: 0 },
            af,
            orientation,
            exposure: Exposure {
                shutter_s: 0.01,
                focal_mm: 50.0,
                iso: 100,
                picture_control: None,
                compensation_ev: None,
            },
        }
    }

    fn af(x: u32, y: u32, w: u32, h: u32) -> AfBox {
        AfBox {
            rect: PixelRect { x, y, w, h },
            space_w: 8256,
            space_h: 5504,
        }
    }

    fn only_af(r: &[Region]) -> Box2d {
        match r {
            [Region::Af { at }] => *at,
            other => panic!("expected one AF region, got {other:?}"),
        }
    }

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn af_box_of_an_upright_frame_is_its_fractions() {
        // 158 px at x=4895 of 8256, 156 px at y=2303 of 5504.
        let at = only_af(&plan(
            &shot(Some(af(4895, 2303, 158, 156)), Orientation::Normal),
            None,
        ));
        near(at.x0, 4895.0 / 8256.0);
        near(at.x1, 5053.0 / 8256.0);
        near(at.y0, 2303.0 / 5504.0);
        near(at.y1, 2459.0 / 5504.0);
    }

    /// Orientation 8 (Rotate270Cw): the render is 5504×8256. A stored point (x, y) shows at
    /// (y, 8256 - x), so the stored box x 1000..1100, y 200..250 shows at x 200..250 and
    /// y 8256-1100..8256-1000 = 7156..7256.
    #[test]
    fn af_box_of_an_orientation_8_frame_is_turned_into_the_displayed_frame() {
        let at = only_af(&plan(
            &shot(Some(af(1000, 200, 100, 50)), Orientation::Rotate270Cw),
            None,
        ));
        near(at.x0, 200.0 / 5504.0);
        near(at.x1, 250.0 / 5504.0);
        near(at.y0, 7156.0 / 8256.0);
        near(at.y1, 7256.0 / 8256.0);
    }

    /// The pixels `image` rotates for display carry the AF mark to where `plan` puts the
    /// region, for every orientation: an independent oracle for the mapping.
    #[test]
    fn af_region_covers_the_marked_pixels_in_every_orientation() {
        let (w, h) = (80u32, 40u32);
        let marked = PixelRect {
            x: 50,
            y: 5,
            w: 10,
            h: 6,
        };
        let mut stored = image::RgbImage::new(w, h);
        for y in marked.y..marked.y + marked.h {
            for x in marked.x..marked.x + marked.w {
                stored.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }
        for o in [
            Orientation::Normal,
            Orientation::Rotate90Cw,
            Orientation::Rotate180,
            Orientation::Rotate270Cw,
        ] {
            let shown = o.display(&stored);
            let b = AfBox {
                rect: marked,
                space_w: w,
                space_h: h,
            };
            let at = only_af(&plan(&shot(Some(b), o), None));
            let (dw, dh) = shown.dimensions();
            let (x0, x1) = (
                (at.x0 * f64::from(dw)).round() as u32,
                (at.x1 * f64::from(dw)).round() as u32,
            );
            let (y0, y1) = (
                (at.y0 * f64::from(dh)).round() as u32,
                (at.y1 * f64::from(dh)).round() as u32,
            );
            for (x, y, p) in shown.enumerate_pixels() {
                let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
                assert_eq!(p.0 == [255, 0, 0], inside, "{o:?} at {x},{y}");
            }
        }
    }

    #[test]
    fn an_af_box_is_padded_to_the_minimum_around_its_centre() {
        // 158×156 centred at (4974, 2381) in a 8256×5504 render: 512×512 from
        // (4974-256, 2381-256) = (4718, 2125).
        let r = plan(
            &shot(Some(af(4895, 2303, 158, 156)), Orientation::Normal),
            None,
        );
        let cut = Cut::of(&r[0], (8256, 5504), &sizing());
        assert_eq!(
            cut.rect,
            PixelRect {
                x: 4718,
                y: 2125,
                w: 512,
                h: 512
            }
        );
        assert_eq!((cut.out_w, cut.out_h), (512, 512));
    }

    #[test]
    fn a_big_af_box_is_kept_and_a_region_at_an_edge_moves_inside() {
        // 700 px wide stays 700; at the left edge the 512-high part is shifted, not shrunk.
        let r = plan(&shot(Some(af(0, 0, 700, 100)), Orientation::Normal), None);
        let cut = Cut::of(&r[0], (8256, 5504), &sizing());
        assert_eq!(
            cut.rect,
            PixelRect {
                x: 0,
                y: 0,
                w: 700,
                h: 512
            }
        );
    }

    /// 400×300 px face: 25 % of each side added on both sides is 600×450.
    #[test]
    fn a_face_is_padded_by_a_quarter_of_its_size_on_every_side() {
        let w = 4000u32;
        let h = 3000u32;
        let face = Region::Face {
            person: 1,
            at: Box2d {
                x0: 1000.0 / 4000.0,
                y0: 1000.0 / 3000.0,
                x1: 1400.0 / 4000.0,
                y1: 1300.0 / 3000.0,
            },
            focus: None,
        };
        let cut = Cut::of(&face, (w, h), &sizing());
        assert_eq!(
            cut.rect,
            PixelRect {
                x: 900,
                y: 925,
                w: 600,
                h: 450
            }
        );
    }

    /// A 3000×2000 region is scaled to fit 1200, keeping its proportions; a 600×450 one is
    /// not enlarged.
    #[test]
    fn regions_are_scaled_down_to_the_cap_and_never_up() {
        let big = Region::Face {
            person: 0,
            at: Box2d {
                x0: 0.0,
                y0: 0.0,
                x1: 2400.0 / 8000.0,
                y1: 1600.0 / 5000.0,
            },
            focus: None,
        };
        // 2400×1600 padded by 25 % per side: 3600×2400, scaled by 1200/3600.
        let cut = Cut::of(&big, (8000, 5000), &sizing());
        assert_eq!(cut.rect.w, 3600);
        assert_eq!((cut.out_w, cut.out_h), (1200, 800));
        let small = Cut::of(
            &Region::Centre,
            (8000, 5000),
            &Sizing {
                centre_px: 600,
                ..sizing()
            },
        );
        assert_eq!((small.out_w, small.out_h), (600, 600));
    }

    #[test]
    fn with_no_af_box_and_no_faces_the_centre_is_cut() {
        for faces in [None, Some(&[][..])] {
            let r = plan(&shot(None, Orientation::Normal), faces);
            assert_eq!(r, [Region::Centre]);
        }
        // 1024 square about (4128, 2752): from (3616, 2240).
        let cut = Cut::of(&Region::Centre, (8256, 5504), &sizing());
        assert_eq!(
            cut.rect,
            PixelRect {
                x: 3616,
                y: 2240,
                w: 1024,
                h: 1024
            }
        );
    }

    #[test]
    fn af_and_faces_are_both_planned_and_the_centre_only_when_nothing_else() {
        let face = FaceBox {
            person: 2,
            at: Box2d {
                x0: 0.1,
                y0: 0.1,
                x1: 0.2,
                y1: 0.2,
            },
        };
        let r = plan(
            &shot(Some(af(4895, 2303, 158, 156)), Orientation::Normal),
            Some(&[face]),
        );
        assert!(matches!(r[0], Region::Af { .. }));
        assert_eq!(
            r[1],
            Region::Face {
                person: 2,
                at: face.at,
                focus: None
            }
        );
        assert_eq!(r.len(), 2);
        assert_eq!(
            plan(&shot(None, Orientation::Normal), Some(&[face])).len(),
            1
        );
    }

    /// The AF box (centre about 0.6, 0.43 of the frame) on the right-hand face: that face
    /// comes first, carrying the focus, and the AF box gets no region of its own; the other
    /// faces follow left to right, whatever order the faces pass listed them in.
    #[test]
    fn the_face_the_camera_focused_on_comes_first_and_takes_the_af_box() {
        let face = |person, x0: f64| FaceBox {
            person,
            at: Box2d {
                x0,
                y0: 0.35,
                x1: x0 + 0.1,
                y1: 0.5,
            },
        };
        let (left, middle, right) = (face(0, 0.1), face(3, 0.3), face(1, 0.55));
        let r = plan(
            &shot(Some(af(4895, 2303, 158, 156)), Orientation::Normal),
            Some(&[right, left, middle]),
        );
        let people: Vec<(usize, bool)> = r
            .iter()
            .map(|r| match r {
                Region::Face { person, focus, .. } => (*person, focus.is_some()),
                other => panic!("expected faces only, got {other:?}"),
            })
            .collect();
        assert_eq!(people, [(1, true), (0, false), (3, false)]);
    }

    #[test]
    fn a_render_smaller_than_the_region_is_cut_whole() {
        let cut = Cut::of(&Region::Centre, (800, 600), &sizing());
        assert_eq!(
            cut.rect,
            PixelRect {
                x: 0,
                y: 0,
                w: 800,
                h: 600
            }
        );
    }
}
