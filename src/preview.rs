//! The in-camera JPEG embedded in each NEF. Culling works on it rather than on a demosaic:
//! it is full resolution on the Z8, already has the Picture Control applied, and decodes in a
//! fraction of the time a raw would take.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Context, Result, ensure};
use image::RgbImage;

use crate::shot::Shot;

/// Z8 JpgFromRaw previews measured at 4–7 MB; anything far larger means a bad offset.
const PREVIEW_LEN_MAX: u64 = 64 << 20;

fn read_jpeg_bytes(shot: &Shot) -> Result<Vec<u8>> {
    let span = shot.preview;
    ensure!(
        span.len <= PREVIEW_LEN_MAX,
        "{}: preview length {} exceeds {PREVIEW_LEN_MAX}",
        shot.file.display(),
        span.len
    );
    ensure!(
        span.offset + span.len <= shot.file_size,
        "{}: preview span runs past end of file",
        shot.file.display()
    );
    let mut f =
        File::open(&shot.file).with_context(|| format!("opening {}", shot.file.display()))?;
    f.seek(SeekFrom::Start(span.offset))?;
    let mut buf = vec![0u8; usize::try_from(span.len)?];
    f.read_exact(&mut buf)
        .with_context(|| format!("reading preview of {}", shot.file.display()))?;
    ensure!(
        buf.starts_with(&[0xFF, 0xD8]),
        "{}: preview is not a JPEG (no SOI marker)",
        shot.file.display()
    );
    Ok(buf)
}

/// Size of the preview in stored orientation, read from the JPEG header alone.
pub fn dimensions(shot: &Shot) -> Result<(u32, u32)> {
    let bytes = read_jpeg_bytes(shot)?;
    image::ImageReader::with_format(std::io::Cursor::new(bytes), image::ImageFormat::Jpeg)
        .into_dimensions()
        .with_context(|| format!("reading preview header of {}", shot.file.display()))
}

pub fn decode(shot: &Shot) -> Result<RgbImage> {
    let bytes = read_jpeg_bytes(shot)?;
    Ok(
        image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg)
            .with_context(|| format!("decoding preview of {}", shot.file.display()))?
            .into_rgb8(),
    )
}
