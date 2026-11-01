//! The domain record for one NEF, read from the NEF header by `nef`.

use std::path::PathBuf;

use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};

/// Byte range of the embedded full-size JPEG (`JpgFromRaw`) inside the NEF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreviewSpan {
    pub offset: u64,
    pub len: u64,
}

/// Rectangle in preview pixel coordinates, top-left origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Where the camera reported focus, in the camera's AF coordinate space (`space_w×space_h`),
/// which `in_frame` maps onto a decoded image of any size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AfBox {
    pub rect: PixelRect,
    pub space_w: u32,
    pub space_h: u32,
}

impl AfBox {
    /// The box scaled into a `w×h` frame and clipped to it.
    pub fn in_frame(&self, w: u32, h: u32) -> PixelRect {
        let sx = f64::from(w) / f64::from(self.space_w);
        let sy = f64::from(h) / f64::from(self.space_h);
        let scale = |v: u32, s: f64| (f64::from(v) * s).round() as u32;
        let x = scale(self.rect.x, sx).min(w.saturating_sub(1));
        let y = scale(self.rect.y, sy).min(h.saturating_sub(1));
        PixelRect {
            x,
            y,
            w: scale(self.rect.w, sx).clamp(1, w - x),
            h: scale(self.rect.h, sy).clamp(1, h - y),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Exposure {
    pub shutter_s: f64,
    pub focal_mm: f64,
    pub iso: u32,
    pub picture_control: Option<String>,
    /// Exposure compensation dialled in at capture, in EV. Negative values are usually
    /// deliberate in bright sun: the preview looks dark but the raw keeps its highlights.
    pub compensation_ev: Option<f64>,
}

/// How the preview must be rotated for display. The preview pixels are stored in sensor
/// orientation; mirrored EXIF orientations do not occur on a camera body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    Normal,
    Rotate90Cw,
    Rotate180,
    Rotate270Cw,
}

impl Orientation {
    pub fn from_exif(v: u8) -> anyhow::Result<Orientation> {
        match v {
            1 => Ok(Orientation::Normal),
            6 => Ok(Orientation::Rotate90Cw),
            3 => Ok(Orientation::Rotate180),
            8 => Ok(Orientation::Rotate270Cw),
            other => anyhow::bail!("unsupported EXIF orientation {other}"),
        }
    }

    /// `img`, stored in sensor orientation, turned the way it is displayed.
    pub fn display(self, img: &image::RgbImage) -> image::RgbImage {
        use image::imageops;
        match self {
            Orientation::Normal => img.clone(),
            Orientation::Rotate90Cw => imageops::rotate90(img),
            Orientation::Rotate180 => imageops::rotate180(img),
            Orientation::Rotate270Cw => imageops::rotate270(img),
        }
    }

    /// Size of the displayed image whose stored size is `w×h`.
    pub fn displayed_size(self, w: u32, h: u32) -> (u32, u32) {
        match self {
            Orientation::Normal | Orientation::Rotate180 => (w, h),
            Orientation::Rotate90Cw | Orientation::Rotate270Cw => (h, w),
        }
    }

    /// A rectangle of the `stored_w×stored_h` stored image as the rectangle of the displayed
    /// image that shows the same pixels: the inverse of `stored_rect`. `r` must lie inside the
    /// stored image.
    pub fn displayed_rect(self, r: PixelRect, stored_w: u32, stored_h: u32) -> PixelRect {
        assert!(
            u64::from(r.x) + u64::from(r.w) <= u64::from(stored_w)
                && u64::from(r.y) + u64::from(r.h) <= u64::from(stored_h),
            "{r:?} lies outside the {stored_w}×{stored_h} stored image"
        );
        let (w, h) = (r.w, r.h);
        match self {
            Orientation::Normal => r,
            Orientation::Rotate90Cw => PixelRect {
                x: stored_h - r.y - h,
                y: r.x,
                w: h,
                h: w,
            },
            Orientation::Rotate180 => PixelRect {
                x: stored_w - r.x - w,
                y: stored_h - r.y - h,
                w,
                h,
            },
            Orientation::Rotate270Cw => PixelRect {
                x: r.y,
                y: stored_w - r.x - w,
                w: h,
                h: w,
            },
        }
    }

    /// A rectangle of the displayed image as the rectangle of the `stored_w×stored_h` stored
    /// image that shows the same pixels. Exact on whole pixels. `r` must lie inside the
    /// displayed image (callers clamp to it); a rectangle that does not is a programmer error.
    pub fn stored_rect(self, r: PixelRect, stored_w: u32, stored_h: u32) -> PixelRect {
        let (shown_w, shown_h) = self.displayed_size(stored_w, stored_h);
        assert!(
            u64::from(r.x) + u64::from(r.w) <= u64::from(shown_w)
                && u64::from(r.y) + u64::from(r.h) <= u64::from(shown_h),
            "{r:?} lies outside the {shown_w}×{shown_h} displayed image ({self:?} of {stored_w}×{stored_h} stored)"
        );
        let (w, h) = (r.w, r.h);
        match self {
            Orientation::Normal => r,
            Orientation::Rotate90Cw => PixelRect {
                x: r.y,
                y: stored_h - r.x - w,
                w: h,
                h: w,
            },
            Orientation::Rotate180 => PixelRect {
                x: stored_w - r.x - w,
                y: stored_h - r.y - h,
                w,
                h,
            },
            Orientation::Rotate270Cw => PixelRect {
                x: stored_w - r.y - h,
                y: r.x,
                w: h,
                h: w,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shot {
    pub file: PathBuf,
    pub file_size: u64,
    pub taken_at: DateTime<FixedOffset>,
    pub preview: PreviewSpan,
    pub af: Option<AfBox>,
    pub orientation: Orientation,
    pub exposure: Exposure,
}

impl Shot {
    pub fn name(&self) -> String {
        self.file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Marks a rectangle of a 10×6 stored image, turns the image for display, finds the mark's
    /// bounding box there, and maps that displayed rectangle back: it must be the marked
    /// rectangle. `image`'s own rotations are the oracle.
    #[test]
    fn stored_rect_inverts_display_on_whole_pixels() {
        let (w, h) = (10u32, 6u32);
        let marked = PixelRect {
            x: 5,
            y: 1,
            w: 3,
            h: 2,
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
            assert_eq!(shown.dimensions(), o.displayed_size(w, h), "{o:?}");
            let hits: Vec<(u32, u32)> = shown
                .enumerate_pixels()
                .filter(|(_, _, p)| p.0 == [255, 0, 0])
                .map(|(x, y, _)| (x, y))
                .collect();
            let x0 = hits.iter().map(|p| p.0).min().unwrap();
            let y0 = hits.iter().map(|p| p.1).min().unwrap();
            let x1 = hits.iter().map(|p| p.0).max().unwrap();
            let y1 = hits.iter().map(|p| p.1).max().unwrap();
            let displayed = PixelRect {
                x: x0,
                y: y0,
                w: x1 - x0 + 1,
                h: y1 - y0 + 1,
            };
            assert_eq!(o.stored_rect(displayed, w, h), marked, "{o:?}");
        }
    }

    /// `displayed_rect` undoes `stored_rect` for every orientation (the 10×6 stored image
    /// shows as 10×6 or 6×10).
    #[test]
    fn displayed_rect_inverts_stored_rect() {
        for o in [
            Orientation::Normal,
            Orientation::Rotate90Cw,
            Orientation::Rotate180,
            Orientation::Rotate270Cw,
        ] {
            let (dw, dh) = o.displayed_size(10, 6);
            for (x, y, w, h) in [(0, 0, 1, 1), (1, 2, 3, 2), (dw - 2, dh - 3, 2, 3)] {
                let shown = PixelRect { x, y, w, h };
                let stored = o.stored_rect(shown, 10, 6);
                assert_eq!(o.displayed_rect(stored, 10, 6), shown, "{o:?} {shown:?}");
            }
        }
    }

    #[test]
    fn stored_rect_accepts_the_whole_displayed_image() {
        for o in [Orientation::Normal, Orientation::Rotate90Cw] {
            let (dw, dh) = o.displayed_size(10, 6);
            let whole = PixelRect {
                x: 0,
                y: 0,
                w: dw,
                h: dh,
            };
            let stored = o.stored_rect(whole, 10, 6);
            assert_eq!((stored.x, stored.y, stored.w, stored.h), (0, 0, 10, 6));
        }
    }

    #[test]
    #[should_panic(expected = "lies outside the 6×10 displayed image")]
    fn stored_rect_refuses_a_rectangle_beyond_the_displayed_image() {
        // 10×6 stored, turned 90°, shows 6×10: x + w = 7 overruns.
        let r = PixelRect {
            x: 4,
            y: 0,
            w: 3,
            h: 1,
        };
        Orientation::Rotate90Cw.stored_rect(r, 10, 6);
    }
}
