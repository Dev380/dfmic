//! Minimal WAV reader/writer.
//!
//! DeepFilterNet works at 48 kHz mono, and the daemon always gets 48 kHz float
//! audio from PipeWire, so this deliberately supports exactly what the offline
//! path needs: PCM 16/24/32-bit and IEEE float 32, any channel count (downmixed
//! to mono on read).

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct Wav {
    pub sample_rate: u32,
    pub channels: u16,
    /// Mono, at `sample_rate`.
    pub samples: Vec<f32>,
}

pub fn read(path: &Path) -> Result<Wav, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF/WAVE file", path.display()));
    }

    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // format, channels, rate, bits
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= buf.len() {
        let id = &buf[pos..pos + 4];
        let size =
            u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(buf.len());
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err("truncated fmt chunk".into());
                }
                let b = &buf[body_start..body_end];
                let format = u16::from_le_bytes([b[0], b[1]]);
                let channels = u16::from_le_bytes([b[2], b[3]]);
                let rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let bits = u16::from_le_bytes([b[14], b[15]]);
                fmt = Some((format, channels, rate, bits));
            }
            b"data" => data = Some(&buf[body_start..body_end]),
            _ => {}
        }
        pos = body_end + (size & 1);
    }

    let (format, channels, rate, bits) = fmt.ok_or("missing fmt chunk")?;
    let data = data.ok_or("missing data chunk")?;
    if channels == 0 {
        return Err("zero channels".into());
    }

    let mut mono = Vec::new();
    let frames = match (format, bits) {
        (1, 16) => {
            for frame in data.chunks_exact(2 * channels as usize) {
                let mut acc = 0.0f32;
                for c in frame.chunks_exact(2) {
                    acc += i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0;
                }
                mono.push(acc / channels as f32);
            }
            data.len() / (2 * channels as usize)
        }
        (1, 24) => {
            for frame in data.chunks_exact(3 * channels as usize) {
                let mut acc = 0.0f32;
                for c in frame.chunks_exact(3) {
                    let v = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                    acc += v as f32 / 8_388_608.0;
                }
                mono.push(acc / channels as f32);
            }
            data.len() / (3 * channels as usize)
        }
        (1, 32) => {
            for frame in data.chunks_exact(4 * channels as usize) {
                let mut acc = 0.0f32;
                for c in frame.chunks_exact(4) {
                    acc += i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / 2_147_483_648.0;
                }
                mono.push(acc / channels as f32);
            }
            data.len() / (4 * channels as usize)
        }
        (3, 32) => {
            for frame in data.chunks_exact(4 * channels as usize) {
                let mut acc = 0.0f32;
                for c in frame.chunks_exact(4) {
                    acc += f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                mono.push(acc / channels as f32);
            }
            data.len() / (4 * channels as usize)
        }
        (f, b) => {
            return Err(format!(
                "unsupported WAV format {f} with {b} bits per sample"
            ))
        }
    };
    let _ = frames;

    Ok(Wav {
        sample_rate: rate,
        channels,
        samples: mono,
    })
}

/// Write mono float samples as 32-bit float WAV.
pub fn write_f32(path: &Path, sample_rate: u32, samples: &[f32]) -> Result<(), String> {
    let data_len = samples.len() * 4;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
    out.extend_from_slice(&4u16.to_le_bytes()); // block align
    out.extend_from_slice(&32u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    File::create(path)
        .and_then(|mut f| f.write_all(&out))
        .map_err(|e| format!("{}: {e}", path.display()))
}
