//! `<dir>/.rawpilot/`: everything rawpilot derives from a shoot, so a re-run with unchanged
//! inputs reads instead of recomputing. Each entry records the key it was computed under; an
//! entry with a different key is stale and is recomputed, never patched.
//!
//! Each kind of entry is a JSON record plus a JPEG per shot. Files are written to a temporary
//! name and renamed, so an interrupted run leaves either the old entry or the new one. The
//! JSON record is renamed last: its presence vouches for the image next to it.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use image::RgbImage;
use image::codecs::jpeg::JpegEncoder;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::lock::FolderLock;
use crate::shot::Shot;

/// A kind of per-shot entry: where its record and image live, and the version to bump when
/// the code computing it changes.
#[derive(Debug, Clone, Copy)]
pub struct Kind {
    records: &'static str,
    images: &'static str,
    version: u32,
}

impl Kind {
    pub fn name(self) -> &'static str {
        self.records
    }
}

/// `features::measure` and the display thumbnail.
pub const FEATURES: Kind = Kind {
    records: "features",
    images: "thumbs",
    version: 1,
};

/// `develop::render` and its rendering.
pub const DEVELOPED: Kind = Kind {
    records: "developed",
    images: "developed",
    version: 2,
};

pub struct Cache {
    root: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct Entry<T> {
    key: String,
    value: T,
}

impl Cache {
    /// The cache of the folder `folder` holds: nothing is derived into a folder another
    /// rawpilot may be working in.
    pub fn open(folder: &FolderLock) -> Result<Cache> {
        let root = folder.dir().join(".rawpilot");
        for kind in [FEATURES, DEVELOPED] {
            for sub in [kind.records, kind.images] {
                fs::create_dir_all(root.join(sub))
                    .with_context(|| format!("creating {}", root.join(sub).display()))?;
            }
        }
        Ok(Cache { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Image path relative to the cache root, as the report links it.
    pub fn image_rel(kind: Kind, shot: &Shot) -> PathBuf {
        PathBuf::from(kind.images).join(format!("{}.jpg", shot.name()))
    }

    pub fn image_path(&self, kind: Kind, shot: &Shot) -> PathBuf {
        self.root.join(Cache::image_rel(kind, shot))
    }

    fn record_path(&self, kind: Kind, shot: &Shot) -> PathBuf {
        self.root
            .join(kind.records)
            .join(format!("{}.json", shot.name()))
    }

    /// The file's identity (name, size, mtime), the kind's version and `params`: everything
    /// that changes the result.
    pub fn key(kind: Kind, shot: &Shot, params: &impl Serialize) -> Result<String> {
        Cache::key_under(kind.records, kind.version, shot, params)
    }

    /// `key` for derived files that are not a `Kind` entry: `label` names what is derived and
    /// `version` is bumped when the code deriving it changes.
    pub fn key_under(
        label: &str,
        version: u32,
        shot: &Shot,
        params: &impl Serialize,
    ) -> Result<String> {
        let mtime_ns = fs::metadata(&shot.file)
            .and_then(|m| m.modified())
            .with_context(|| format!("stat {}", shot.file.display()))?
            .duration_since(UNIX_EPOCH)?
            .as_nanos();
        let mut h = blake3::Hasher::new();
        h.update(shot.name().as_bytes());
        h.update(&shot.file_size.to_le_bytes());
        h.update(&mtime_ns.to_le_bytes());
        h.update(label.as_bytes());
        h.update(&version.to_le_bytes());
        h.update(serde_json::to_string(params)?.as_bytes());
        Ok(h.finalize().to_hex().to_string())
    }

    /// The cached value if it was computed under `key` and its image is present.
    pub fn load<T: DeserializeOwned>(&self, kind: Kind, shot: &Shot, key: &str) -> Option<T> {
        let text = fs::read_to_string(self.record_path(kind, shot)).ok()?;
        let entry: Entry<T> = serde_json::from_str(&text).ok()?;
        (entry.key == key && self.image_path(kind, shot).is_file()).then_some(entry.value)
    }

    pub fn store<T: Serialize>(
        &self,
        kind: Kind,
        shot: &Shot,
        key: &str,
        value: &T,
        image: &RgbImage,
        jpeg_quality: u8,
    ) -> Result<()> {
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, jpeg_quality).encode_image(image)?;
        write_atomic(&self.image_path(kind, shot), &jpeg)?;
        let entry = Entry {
            key: key.to_owned(),
            value,
        };
        write_atomic(
            &self.record_path(kind, shot),
            serde_json::to_string_pretty(&entry)?.as_bytes(),
        )
    }
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut f = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))
}
