// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use image::{DynamicImage, RgbImage};
use llm_multimodal::{PreProcessorConfig, VideoClip, VideoSource};
use serde_json::Value;
use tokio::io::AsyncReadExt as _;

use crate::error::{Result, bail_multimodal, multimodal};
use crate::multimodal::video::FetchedVideo;

const MAX_INPUT: usize = 256 * 1024 * 1024;
const MAX_DECODED: usize = 1024 * 1024 * 1024;

#[derive(Clone)]
pub(crate) struct VideoSampling {
    fps: f64,
    max_frames: usize,
    max_duration: f64,
    temporal: usize,
}

impl VideoSampling {
    pub(crate) fn new(config: &PreProcessorConfig) -> Result<Self> {
        let number = |key: &str, default: f64| {
            config.extra.get(key).and_then(Value::as_f64).unwrap_or(default)
        };
        let sampling = Self {
            fps: number("fps", 2.0),
            max_frames: number("max_frames", 2048.0) as usize,
            max_duration: number("max_duration", 0.0),
            temporal: config.temporal_patch_size.unwrap_or(2),
        };
        if !sampling.fps.is_finite()
            || sampling.fps <= 0.0
            || sampling.max_frames == 0
            || sampling.temporal == 0
        {
            bail_multimodal!("invalid GLM video sampling configuration");
        }
        Ok(sampling)
    }

    pub(super) fn indices(
        &self,
        total: usize,
        fps: f64,
        duration: Option<f64>,
    ) -> Result<Vec<usize>> {
        if total == 0 || !fps.is_finite() || fps <= 0.0 {
            bail_multimodal!("GLM video requires a frame count and positive source FPS");
        }
        let duration = duration
            .filter(|v| *v > 0.0)
            .unwrap_or(((total - 1) as f64 / fps).round_ties_even() + 1.0);
        let max_seconds = duration as usize;
        let capped = if self.max_duration > 0.0 {
            duration.min(self.max_duration)
        } else {
            duration
        };
        let count = ((capped * self.fps) as usize).min(self.max_frames);
        if count == 0 {
            return Ok(Vec::new());
        }
        let linspace = |start: usize, end: usize| -> Vec<usize> {
            if count == 1 {
                return vec![start];
            }
            let step = (end - start) as f64 / (count - 1) as f64;
            (0..count)
                .map(|i| {
                    if i == count - 1 {
                        end
                    } else {
                        (start as f64 + i as f64 * step) as usize
                    }
                })
                .collect()
        };
        let mut indices = if total < count {
            linspace(0, total - 1)
        } else {
            let mut indices = Vec::new();
            let mut second = 0.0;
            for index in 0..total {
                if index as f64 * (1.0 / fps) >= second {
                    second += 1.0 / self.fps;
                    indices.push(index);
                    if second >= max_seconds as f64 {
                        break;
                    }
                }
            }
            indices
        };
        if indices.len() < count {
            indices = linspace(
                *indices.first().unwrap_or(&0),
                *indices.last().unwrap_or(&(total - 1)),
            );
        } else if indices.len() > count {
            indices = linspace(0, total - 1);
        }
        indices.dedup();
        if indices.len() % 2 != 0 {
            indices.push(*indices.last().unwrap());
        }
        Ok(indices)
    }

    pub(crate) async fn fetch(&self, url: String) -> Result<FetchedVideo> {
        let (bytes, source) = if let Some(data) = url.strip_prefix("data:") {
            let (header, encoded) =
                data.split_once(',').ok_or_else(|| multimodal!("invalid video data URL"))?;
            if !header.ends_with(";base64") {
                bail_multimodal!("video data URL requires base64 encoding");
            }
            if encoded.len() > MAX_INPUT.div_ceil(3) * 4 {
                bail_multimodal!("video exceeds input byte limit");
            }
            let bytes =
                STANDARD.decode(encoded).map_err(|e| multimodal!("invalid video base64: {e}"))?;
            (Bytes::from(bytes), VideoSource::DataUrl)
        } else {
            let parsed = reqwest_0_13::Url::parse(&url)
                .map_err(|e| multimodal!("invalid video URL: {e}"))?;
            if !matches!(parsed.scheme(), "http" | "https") {
                bail_multimodal!("unsupported video URL scheme");
            }
            let mut response = reqwest_0_13::Client::new()
                .get(parsed)
                .timeout(Duration::from_secs(10))
                .send()
                .await
                .map_err(|e| multimodal!("video fetch failed: {e}"))?
                .error_for_status()
                .map_err(|e| multimodal!("video fetch failed: {e}"))?;
            if response.content_length().is_some_and(|n| n > MAX_INPUT as u64) {
                bail_multimodal!("video exceeds input byte limit");
            }
            let mut bytes = Vec::new();
            while let Some(chunk) =
                response.chunk().await.map_err(|e| multimodal!("video fetch failed: {e}"))?
            {
                if bytes.len() + chunk.len() > MAX_INPUT {
                    bail_multimodal!("video exceeds input byte limit");
                }
                bytes.extend_from_slice(&chunk);
            }
            (Bytes::from(bytes), VideoSource::Url { url })
        };
        self.decode(bytes, source).await
    }

    pub(crate) async fn decode(&self, bytes: Bytes, source: VideoSource) -> Result<FetchedVideo> {
        if bytes.len() > MAX_INPUT {
            bail_multimodal!("video exceeds input byte limit");
        }
        let file =
            tempfile::NamedTempFile::new().map_err(|e| multimodal!("video temporary file: {e}"))?;
        tokio::fs::write(file.path(), &bytes)
            .await
            .map_err(|e| multimodal!("video temporary file: {e}"))?;
        let path = file
            .path()
            .to_str()
            .ok_or_else(|| multimodal!("invalid video temporary path"))?;
        let probe = run(
            "ffprobe",
            &[
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration",
                "-of",
                "json",
                path,
            ],
            65536,
        )
        .await?;
        let metadata: Value = serde_json::from_slice(&probe)
            .map_err(|e| multimodal!("invalid video metadata: {e}"))?;
        let stream = &metadata["streams"][0];
        let integer = |key: &str| {
            stream[key]
                .as_u64()
                .or_else(|| stream[key].as_str()?.parse().ok())
                .map(|n| n as usize)
        };
        let width = integer("width").ok_or_else(|| multimodal!("video has no width"))?;
        let height = integer("height").ok_or_else(|| multimodal!("video has no height"))?;
        let total = integer("nb_frames")
            .or_else(|| integer("nb_read_frames"))
            .ok_or_else(|| multimodal!("video has no frame count"))?;
        let rate = stream["avg_frame_rate"]
            .as_str()
            .and_then(|v| v.split_once('/'))
            .ok_or_else(|| multimodal!("video has no FPS"))?;
        let fps = rate.0.parse::<f64>().unwrap_or(0.0) / rate.1.parse::<f64>().unwrap_or(0.0);
        let duration = stream["duration"]
            .as_str()
            .and_then(|v| v.parse::<f64>().ok())
            .or_else(|| metadata["format"]["duration"].as_str()?.parse::<f64>().ok());
        let indices = self.indices(total, fps, duration)?;
        if indices.is_empty() {
            bail_multimodal!("GLM video sampling selected no frames");
        }
        let mut unique = indices.clone();
        unique.dedup();
        let frame_bytes = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| multimodal!("video dimensions overflow"))?;
        let expected = frame_bytes
            .checked_mul(unique.len())
            .filter(|n| *n <= MAX_DECODED)
            .ok_or_else(|| multimodal!("video exceeds decoded byte limit"))?;
        let select = format!(
            "select={}",
            unique.iter().map(|i| format!("eq(n\\,{i})")).collect::<Vec<_>>().join("+")
        );
        let decoded = run(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-threads",
                "2",
                "-noautorotate",
                "-i",
                path,
                "-map",
                "0:v:0",
                "-vf",
                &select,
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            expected,
        )
        .await?;
        if decoded.len() != expected {
            bail_multimodal!(
                "GLM video decoded {} bytes, expected {expected}",
                decoded.len()
            );
        }
        let mut frames = Vec::with_capacity(indices.len());
        for index in &indices {
            let pos = unique.binary_search(index).unwrap();
            let data = decoded[pos * frame_bytes..(pos + 1) * frame_bytes].to_vec();
            let image = RgbImage::from_raw(width as u32, height as u32, data)
                .ok_or_else(|| multimodal!("invalid decoded GLM frame"))?;
            frames.push(DynamicImage::ImageRgb8(image));
        }
        let timestamps = indices.chunks(self.temporal).map(|chunk| chunk[0] as f64 / fps).collect();
        let hash = llm_multimodal::hasher::hash_video(&bytes);
        Ok(FetchedVideo {
            clip: Arc::new(VideoClip::new_with_sample_fps(
                frames,
                bytes,
                source,
                hash,
                self.fps as f32,
            )),
            timestamps: Some(timestamps),
        })
    }
}

async fn run(program: &str, args: &[&str], limit: usize) -> Result<Vec<u8>> {
    let result = tokio::time::timeout(Duration::from_secs(60), async {
        let mut child = tokio::process::Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| multimodal!("failed to start {program}: {e}"))?;
        let mut stdout = child.stdout.take().unwrap().take(limit as u64 + 1);
        let mut stderr = child.stderr.take().unwrap().take(65536);
        let mut output = Vec::new();
        let mut error = Vec::new();
        tokio::try_join!(
            stdout.read_to_end(&mut output),
            stderr.read_to_end(&mut error)
        )
        .map_err(|e| multimodal!("{program} output failed: {e}"))?;
        if output.len() > limit {
            bail_multimodal!("{program} exceeded the output byte limit");
        }
        let status = child.wait().await.map_err(|e| multimodal!("{program} failed: {e}"))?;
        if !status.success() {
            bail_multimodal!("{program} failed: {}", String::from_utf8_lossy(&error));
        }
        Ok(output)
    })
    .await;
    result.map_err(|_| multimodal!("{program} timed out"))?
}
