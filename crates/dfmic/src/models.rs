//! Where the model lives, and how it gets there.
//!
//! The three OpenVINO graphs are ~20 MB of converted IR. They are not compiled
//! into the binary: they live in the user's data directory and are downloaded
//! once, on request. Nothing is fetched behind the user's back — a program that
//! silently reaches for the network on first start is not something to spring on
//! someone — so a missing model is a clear error naming the two ways to fix it,
//! and both the CLI (`dfmic models download`) and the window offer the download
//! explicitly.
//!
//! Resolution order, most explicit first:
//!   1. `--models DIR` — never touched, never verified: the caller owns it.
//!   2. `DFMIC_MODELS` — same, for packagers and for the dev shell.
//!   3. `$XDG_DATA_HOME/dfmic/models` — what `download` fills in.
//!
//! The download is a plain `curl` of a release asset plus `tar` to unpack it.
//! That is deliberately not an HTTP client in the dependency tree: the project
//! already shells out to `pw-metadata`, `systemctl` and `pw-play`, and three
//! external commands are a smaller surface than a TLS stack. The Nix package
//! puts them on the wrapper's `PATH`; in a dev shell they are already there.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Version of the converted graphs. Part of the path, so bumping it fetches the
/// new model instead of reusing an old one.
pub const VERSION: &str = "DeepFilterNet3_ll-0.5.6";

/// The converted IR as a `.tar.gz`, hosted on the Hugging Face Hub.
///
/// It has to be the *converted* graphs, not upstream's ONNX: only the IR has the
/// recurrence made explicit, which is what the NPU plugin can run. The archive
/// holds the six required files inside a single top-level directory.
///
/// `/resolve/main/` rather than `/resolve/<revision>/` because the model is
/// versioned by `VERSION` below, and the file name carries it.
const MODEL_URL: &str = "https://huggingface.co/jsjxjjfrkznmeme/dfmic-models-DeepFilterNet3_ll/resolve/main/dfmic-models-DeepFilterNet3_ll-0.5.6.tar.gz";

/// SHA-256 of that archive, as 64 lowercase hex digits.
///
/// Verified against the uploaded file, which was built reproducibly from the Nix
/// derivation:
///
/// ```sh
/// tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='@0' -cf - models \
///   | gzip -9n > dfmic-models-DeepFilterNet3_ll-0.5.6.tar.gz
/// ```
///
/// Those flags are what make it stable; a plain `tar -czf` embeds timestamps and
/// would change the hash on every rebuild. If the archive is ever re-uploaded,
/// this must change with it — the download is refused otherwise.
const MODEL_SHA256: &str = "fbb58a47bd55ffe1857300182143055c6332fbd072c51efbc3008d625a0279b5";

/// Files the graphs need. `config.ini` and `manifest.json` are informational and
/// not required.
const REQUIRED: [&str; 6] = [
    "enc.xml",
    "enc.bin",
    "erb_dec.xml",
    "erb_dec.bin",
    "df_dec.xml",
    "df_dec.bin",
];

/// Where the model is, without downloading anything.
///
/// `Ok` when the directory exists and holds every file the graphs need. When it
/// does not, the error says how to get it, because that is the only thing the
/// caller can act on.
pub fn resolve(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let dir = path(explicit);
    if is_complete(&dir) {
        Ok(dir)
    } else {
        Err(format!(
            "the model is not installed ({} holds no enc/erb_dec/df_dec files).\n\
             Run `dfmic models download` to fetch it, or pass --models DIR / set DFMIC_MODELS.",
            dir.display()
        ))
    }
}

/// Where the model would be, without checking that it is there.
///
/// The daemon uses this: it has to start and run without a model, and only needs
/// the directory to look in once the files appear.
pub fn path(explicit: Option<&Path>) -> PathBuf {
    match explicit {
        Some(dir) => dir.to_path_buf(),
        None => match std::env::var("DFMIC_MODELS") {
            Ok(dir) => PathBuf::from(dir),
            Err(_) => data_dir(),
        },
    }
}

/// Whether a model is available without downloading one.
///
/// `explicit` is `--models`; without it the search follows [`path`]. The daemon
/// passes the directory it was started with, so a `--models DIR` run is not
/// reported as "no model" while the worker happily loads from that same
/// directory.
pub fn is_ready(explicit: Option<&Path>) -> bool {
    resolve(explicit).is_ok()
}

/// Where the downloaded model lives, whether or not it exists yet.
pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from(".local/share"));
    base.join("dfmic").join("models").join(VERSION)
}

/// Whether this build knows where to fetch the model from.
pub fn downloadable() -> bool {
    !MODEL_URL.contains("/owner/") && !MODEL_SHA256.starts_with("0000")
}

/// The directory a download would fill in: exactly what [`resolve`] looks at.
///
/// Downloading anywhere else would be a button that does nothing — the files
/// would land somewhere the daemon never reads. `explicit` is `--models`; without
/// it the same rule as [`path`] applies (`DFMIC_MODELS`, then the data
/// directory).
pub fn download_target(explicit: Option<&Path>) -> PathBuf {
    path(explicit)
}

/// Whether downloading could actually help.
///
/// False when the build has no URL, and false when the target cannot be written
/// — a packaged service pointing `DFMIC_MODELS` into the Nix store, for
/// instance. Offering a button that is bound to fail with `Permission denied`
/// would be worse than saying so up front.
pub fn can_download(explicit: Option<&Path>) -> bool {
    downloadable() && is_writable(&download_target(explicit))
}

/// Download the model into [`download_target`], replacing an incomplete copy.
pub fn download_default(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let dir = download_target(explicit);
    if is_complete(&dir) {
        return Ok(dir);
    }
    download(&dir)?;
    Ok(dir)
}

/// Whether `dir` could be created or written to.
///
/// Walks up to the nearest existing ancestor, because the target itself usually
/// does not exist yet, and asks the kernel rather than the mode bits: a
/// root-owned `0755` directory reports itself writable to `permissions()` while
/// `open` returns `EACCES`, which is how the window ended up offering a download
/// button that could only fail. A probe file is created and removed instead, so
/// ownership, ACLs and a read-only mount are all accounted for.
fn is_writable(dir: &Path) -> bool {
    let mut probe = dir.to_path_buf();
    loop {
        if probe.exists() {
            if !probe.is_dir() {
                return false;
            }
            // The target itself is where the files land if it exists; otherwise
            // the nearest ancestor is where the directory would be created.
            break;
        }
        match probe.parent() {
            Some(parent) => probe = parent.to_path_buf(),
            None => return false,
        }
    }
    let test = probe.join(format!(".dfmic-write-test-{}", std::process::id()));
    match std::fs::File::create(&test) {
        Ok(_) => {
            let _ = std::fs::remove_file(&test);
            true
        }
        Err(_) => false,
    }
}

/// Whether `dir` holds every file the graphs need, and each is non-empty.
///
/// The size check matters: six zero-byte files used to count as an installed
/// model, so the download button refused to run and the worker retried the load
/// in a loop instead of downloading over the wreckage. `manifest.json` is the
/// converter's own list of graphs; its presence is what distinguishes a real
/// model from a touched set of names.
fn is_complete(dir: &Path) -> bool {
    REQUIRED
        .iter()
        .all(|name| dir.join(name).metadata().is_ok_and(|meta| meta.len() > 0))
        && dir.join("manifest.json").is_file()
}

/// Fetch and unpack the model into `dir`.
///
/// Everything happens in a sibling temporary directory and is moved into place
/// only once the archive has been verified and every required file is present,
/// so an interrupted download can never leave a half-model that later looks
/// usable.
fn download(dir: &Path) -> Result<(), String> {
    if !downloadable() {
        return Err(format!(
            "this build has no download URL configured, so it cannot fetch the model.\n\
             Expected it in {}, or pass --models DIR / set DFMIC_MODELS.\n\
             (Set MODEL_URL and MODEL_SHA256 in src/models.rs to enable the download.)",
            dir.display()
        ));
    }
    let parent = dir
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", dir.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;

    let staging = parent.join(format!(".download-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)
        .map_err(|error| format!("could not create {}: {error}", staging.display()))?;

    let result = fetch_into(&staging, dir);
    let _ = std::fs::remove_dir_all(&staging);
    result
}

fn fetch_into(staging: &Path, dir: &Path) -> Result<(), String> {
    let archive = staging.join("model.tar.gz");
    eprintln!("dfmic: downloading the model ({MODEL_URL})");
    run(
        Command::new("curl")
            // `--max-time` bounds the whole transfer and `--connect-timeout` the
            // handshake: without them a hung connection blocks the download
            // thread — and the CLI that waits on it — for ever.
            .args([
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--connect-timeout",
                "30",
                "--max-time",
                "900",
                "--retry",
                "3",
                "--retry-delay",
                "2",
            ])
            .arg("--output")
            .arg(&archive)
            .arg(MODEL_URL),
        "curl",
    )?;

    eprintln!("dfmic: verifying the model");
    let digest = run_capture(Command::new("sha256sum").arg(&archive), "sha256sum")?;
    let digest = digest
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    if digest != MODEL_SHA256 {
        return Err(format!(
            "the downloaded model is not the expected file.\n  expected {MODEL_SHA256}\n  got      {digest}"
        ));
    }

    eprintln!("dfmic: unpacking the model");
    let unpacked = staging.join("unpacked");
    std::fs::create_dir_all(&unpacked)
        .map_err(|error| format!("could not create {}: {error}", unpacked.display()))?;
    run(
        Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("-C")
            .arg(&unpacked),
        "tar",
    )?;

    // The archive may hold the files directly or wrap them in one directory.
    let source =
        if is_complete(&unpacked) {
            unpacked.clone()
        } else {
            let nested = std::fs::read_dir(&unpacked)
                .map_err(|error| format!("could not read {}: {error}", unpacked.display()))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|path| path.is_dir() && is_complete(path));
            match nested {
                Some(path) => path,
                None => return Err(
                    "the downloaded model does not contain enc/erb_dec/df_dec .xml and .bin files"
                        .into(),
                ),
            }
        };

    install(&source, dir)?;
    if !is_complete(dir) {
        return Err(format!(
            "{} is missing model files after unpacking",
            dir.display()
        ));
    }
    eprintln!("dfmic: model ready in {}", dir.display());
    Ok(())
}

/// Move a finished model from `source` into `dir`, keeping whatever else lives
/// there.
///
/// The new files are merged over the old directory rather than replacing it: `dir`
/// is whatever `--models`/`DFMIC_MODELS` pointed at, which may be a place the user
/// keeps other things, and `remove_dir_all` on it would destroy them. The old
/// contents are read from where they were moved *to* — renaming the directory
/// aside and merging into the now-empty path would drop everything that was
/// there, which is precisely the bug this exists to avoid.
fn install(source: &Path, dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        // Nothing to preserve, and a rename is atomic.
        return std::fs::rename(source, dir)
            .map_err(|error| format!("could not move the model into {}: {error}", dir.display()));
    }
    let parent = dir
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", dir.display()))?;
    let retired = parent.join(format!(".replaced-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&retired);
    std::fs::rename(dir, &retired)
        .map_err(|error| format!("could not move {} aside: {error}", dir.display()))?;
    // Everything that was there first, then the new model over it, so an
    // unrelated file keeps its place and a model file is replaced.
    match merge_into(&retired, dir).and_then(|()| merge_into(source, dir)) {
        Ok(()) => {
            let _ = std::fs::remove_dir_all(&retired);
            Ok(())
        }
        Err(error) => {
            // Put the old directory back rather than leaving the user with
            // neither the old files nor the new model.
            let _ = std::fs::remove_dir_all(dir);
            let _ = std::fs::rename(&retired, dir);
            Err(error)
        }
    }
}

/// Create `dest` and copy everything from `source` into it, leaving any other
/// files that already live in `dest` alone.
fn merge_into(source: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest)
        .map_err(|error| format!("could not create {}: {error}", dest.display()))?;
    let entries = std::fs::read_dir(source)
        .map_err(|error| format!("could not read {}: {error}", source.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("could not read {}: {error}", source.display()))?;
        let target = dest.join(entry.file_name());
        let kind = entry
            .file_type()
            .map_err(|error| format!("could not read {}: {error}", entry.path().display()))?;
        if kind.is_dir() {
            merge_into(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target).map_err(|error| {
                format!(
                    "could not copy {} to {}: {error}",
                    entry.path().display(),
                    target.display()
                )
            })?;
        }
    }
    Ok(())
}

fn run(command: &mut Command, name: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("could not run {name}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let message = String::from_utf8_lossy(&output.stderr);
        Err(format!(
            "{name} failed: {}",
            if message.trim().is_empty() {
                output.status.to_string()
            } else {
                message.trim().to_string()
            }
        ))
    }
}

fn run_capture(command: &mut Command, name: &str) -> Result<String, String> {
    let output = command
        .output()
        .map_err(|error| format!("could not run {name}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{name} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, name: &str, contents: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), contents).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dfmic-models-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A model that is not complete is not a model: six touched filenames used to
    /// count as one, which made the download refuse to run and the worker retry
    /// the load for ever.
    #[test]
    fn zero_byte_files_are_not_a_model() {
        let dir = scratch("empty");
        for name in REQUIRED {
            touch(&dir, name, "");
        }
        touch(&dir, "manifest.json", "{}");
        assert!(!is_complete(&dir), "empty files were accepted as a model");

        for name in REQUIRED {
            touch(&dir, name, "x");
        }
        assert!(is_complete(&dir), "files with content were rejected");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The converter's manifest is what distinguishes a real model from a set of
    /// correctly named files.
    #[test]
    fn the_manifest_is_required() {
        let dir = scratch("nomanifest");
        for name in REQUIRED {
            touch(&dir, name, "x");
        }
        assert!(!is_complete(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Downloading over an incomplete model must not delete the user's other
    /// files: `--models`/`DFMIC_MODELS` may point anywhere.
    #[test]
    fn replacing_a_model_keeps_unrelated_files() {
        let dir = scratch("replace");
        touch(&dir, "keepme.txt", "user data");
        touch(&dir, "enc.xml", "partial");
        let source = scratch("replace-source");
        for name in REQUIRED {
            touch(&source, name, "x");
        }
        touch(&source, "manifest.json", "{}");

        install(&source, &dir).unwrap();

        assert!(
            dir.join("keepme.txt").is_file(),
            "merging a download destroyed an unrelated file"
        );
        assert!(is_complete(&dir), "the new model was not installed");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&source);
    }

    /// Writability is decided by the kernel, not by the mode bits: a directory
    /// owned by someone else can be `0755` and still refuse a write.
    #[test]
    fn writability_is_probed_not_inferred() {
        let dir = scratch("writable");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(is_writable(&dir), "a writable directory was refused");
        // A path under a regular file cannot be written no matter who owns it.
        let file = dir.join("file");
        touch(&dir, "file", "x");
        assert!(!is_writable(&file));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
