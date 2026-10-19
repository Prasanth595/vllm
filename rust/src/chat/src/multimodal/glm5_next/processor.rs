// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use image::{DynamicImage, RgbImage};
use llm_multimodal::{
    Modality, ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, RgbFrameRef,
    TransformError, VisionPreProcessor,
};
use ndarray::Array2;

use super::resize;

type Result<T> = std::result::Result<T, TransformError>;

pub(super) struct Glm5NextProcessor {
    patch: usize,
    temporal: usize,
    merge: usize,
    factor: usize,
    min_tokens: usize,
    max_tokens: usize,
    resize: bool,
    mean: [f64; 3],
    std: [f64; 3],
    normalized: [[f32; 256]; 3],
}

impl Glm5NextProcessor {
    pub(super) fn new(config: &PreProcessorConfig, modality: Modality) -> Result<Self> {
        let patch = config.get_patch_size(14);
        let temporal = config.temporal_patch_size.unwrap_or(2);
        let merge = config.merge_size.unwrap_or(2);
        let extra = |name: &str, default: usize| {
            config.extra.get(name).and_then(|v| v.as_u64()).map_or(default, |v| v as usize)
        };
        let factor = patch.saturating_mul(merge).saturating_mul(extra("patch_expand_factor", 1));
        let min_tokens = extra("min_image_tokens", 16);
        let max_tokens = if modality == Modality::Video {
            extra("max_image_tokens", 240_000).min(30_000)
        } else {
            extra("max_image_tokens", 8000)
        };
        if factor == 0 || temporal == 0 || min_tokens == 0 || max_tokens < min_tokens {
            return Err(TransformError::ShapeError(
                "invalid GLM patch size or token budget".into(),
            ));
        }
        if config.resampling.is_some_and(|v| v != 3) {
            return Err(TransformError::ShapeError(
                "GLM requires bicubic resampling".into(),
            ));
        }
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let rescale = config.do_rescale.unwrap_or(true);
        let normalize = config.should_normalize();
        let scale = config.rescale_factor.unwrap_or(1.0 / 255.0);
        if !scale.is_finite() || scale <= 0.0 || std.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(TransformError::ShapeError(
                "invalid GLM normalization parameters".into(),
            ));
        }
        let normalized = std::array::from_fn(|c| {
            let multiplier = if rescale { (1.0 / scale) as f32 } else { 1.0 };
            let m = mean[c] as f32 * multiplier;
            let s = std[c] as f32 * multiplier;
            std::array::from_fn(|pixel| {
                if normalize {
                    (pixel as f32 - m) / s
                } else if rescale {
                    pixel as f32 * scale as f32
                } else {
                    pixel as f32
                }
            })
        });
        Ok(Self {
            patch,
            temporal,
            merge,
            factor,
            min_tokens,
            max_tokens,
            resize: config.do_resize.unwrap_or(true),
            mean,
            std,
            normalized,
        })
    }

    pub(super) fn smart_resize(&self, frames: usize, h: usize, w: usize) -> Result<(usize, usize)> {
        if frames == 0 || h == 0 || w == 0 {
            return Err(TransformError::EmptyBatch);
        }
        let align = |n: usize| n.div_ceil(self.factor) * self.factor;
        let pixels_per_token = self.temporal * self.factor * self.factor;
        let min_pixels = self.min_tokens * pixels_per_token;
        let max_pixels = self.max_tokens * pixels_per_token;
        let aligned_frames = ((frames as f64 / self.temporal as f64).round_ties_even() as usize
            * self.temporal)
            .max(self.temporal);
        let (mut rh, mut rw) = (align(h), align(w));
        if aligned_frames * rh * rw < min_pixels {
            let scale = (min_pixels as f64 / (frames * h * w) as f64).sqrt();
            rh = align((h as f64 * scale).ceil() as usize);
            rw = align((w as f64 * scale).ceil() as usize);
        }
        if aligned_frames * rh * rw > max_pixels {
            if max_pixels < aligned_frames * self.factor * self.factor {
                return Err(TransformError::ShapeError(
                    "GLM video exceeds the pixel budget".into(),
                ));
            }
            let (mut low, mut high) = (1, h);
            (rh, rw) = (self.factor, self.factor);
            while low <= high {
                let mid = (low + high) / 2;
                let ch = align(mid);
                let cw = align((w * mid / h).max(1));
                if aligned_frames * ch * cw <= max_pixels {
                    (rh, rw) = (ch, cw);
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }
        Ok((rh, rw))
    }

    fn patchify(&self, frames: &[RgbImage], image: bool) -> Result<(Vec<f32>, [usize; 3])> {
        let first = frames.first().ok_or(TransformError::EmptyBatch)?;
        let (h, w) = (first.height() as usize, first.width() as usize);
        if frames.iter().any(|f| f.dimensions() != first.dimensions()) {
            return Err(TransformError::InconsistentShapes);
        }
        let count = if image { self.temporal } else { frames.len() };
        let (rh, rw) = if self.resize {
            self.smart_resize(count, h, w)?
        } else {
            (h, w)
        };
        if !rh.is_multiple_of(self.patch * self.merge)
            || !rw.is_multiple_of(self.patch * self.merge)
        {
            return Err(TransformError::ShapeError(
                "GLM image dimensions must align with merged patches".into(),
            ));
        }
        let mut scale = (rh as f64 / h as f64).min(rw as f64 / w as f64);
        if count * h * w >= self.min_tokens * self.temporal * self.factor * self.factor {
            scale = scale.min(1.0);
        }
        let ch = ((h as f64 * scale).floor() as usize).clamp(1, rh);
        let cw = ((w as f64 * scale).floor() as usize).clamp(1, rw);
        let resized: Vec<_> =
            frames.iter().map(|f| resize::bicubic(f, cw as u32, ch as u32)).collect();
        let grid = [
            count.div_ceil(self.temporal),
            rh / self.patch,
            rw / self.patch,
        ];
        let len = grid.iter().product::<usize>() * 3 * self.temporal * self.patch * self.patch;
        let mut values = Vec::with_capacity(len);
        for t in 0..grid[0] {
            for by in 0..grid[1] / self.merge {
                for bx in 0..grid[2] / self.merge {
                    for my in 0..self.merge {
                        for mx in 0..self.merge {
                            for c in 0..3 {
                                for dt in 0..self.temporal {
                                    let frame =
                                        &resized[(t * self.temporal + dt).min(resized.len() - 1)];
                                    for py in 0..self.patch {
                                        let y = (by * self.merge + my) * self.patch + py;
                                        for px in 0..self.patch {
                                            let x = (bx * self.merge + mx) * self.patch + px;
                                            let pixel = if y < ch && x < cw {
                                                frame.get_pixel(x as u32, y as u32)[c]
                                            } else {
                                                0
                                            };
                                            values.push(self.normalized[c][pixel as usize]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((values, grid))
    }

    fn process(&self, items: Vec<Vec<RgbImage>>, video: bool) -> Result<PreprocessedEncoderInputs> {
        if items.is_empty() {
            return Err(TransformError::EmptyBatch);
        }
        let mut values = Vec::new();
        let mut grids = Vec::new();
        let mut sizes = Vec::new();
        let mut patches = Vec::new();
        for frames in &items {
            let (data, grid) = self.patchify(frames, !video)?;
            values.extend(data);
            grids.extend(grid.map(|n| n as i64));
            patches.push(grid.iter().product::<usize>() as i64);
            sizes.push((frames[0].height(), frames[0].width()));
        }
        let modality = if video { "video" } else { "image" };
        let width = 3 * self.temporal * self.patch * self.patch;
        let encoder_input = Array2::from_shape_vec((values.len() / width, width), values)
            .map_err(|e| TransformError::ShapeError(format!("{e}")))?
            .into_dyn();
        Ok(PreprocessedEncoderInputs {
            encoder_input,
            feature_token_counts: patches.iter().map(|n| *n as usize / self.merge.pow(2)).collect(),
            item_sizes: sizes,
            model_specific: HashMap::from([
                (
                    format!("{modality}_grid_thw"),
                    ModelSpecificValue::int_2d(grids, items.len(), 3),
                ),
                (
                    format!("patches_per_{modality}"),
                    ModelSpecificValue::int_1d(patches),
                ),
            ]),
        })
    }
}

impl VisionPreProcessor for Glm5NextProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.mean
    }
    fn default_std(&self) -> [f64; 3] {
        self.std
    }
    fn model_name(&self) -> &'static str {
        "glm5_next"
    }
    fn calculate_num_tokens(&self, width: u32, height: u32) -> usize {
        width as usize / self.patch * (height as usize / self.patch) / self.merge.pow(2)
    }
    fn preprocess(&self, images: &[DynamicImage]) -> Result<PreprocessedEncoderInputs> {
        self.process(images.iter().map(|f| vec![f.to_rgb8()]).collect(), false)
    }
    fn preprocess_video(&self, frames: &[DynamicImage]) -> Result<PreprocessedEncoderInputs> {
        self.process(
            vec![frames.iter().map(DynamicImage::to_rgb8).collect()],
            true,
        )
    }
    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
    ) -> Result<PreprocessedEncoderInputs> {
        let frames = frames
            .iter()
            .map(|f| {
                RgbImage::from_raw(f.width, f.height, f.data.to_vec())
                    .ok_or_else(|| TransformError::ShapeError("invalid GLM RGB frame".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.process(vec![frames], true)
    }
}
