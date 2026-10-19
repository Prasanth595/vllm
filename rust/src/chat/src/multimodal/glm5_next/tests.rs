// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use image::{DynamicImage, RgbImage};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::*;

#[derive(Deserialize)]
struct Case {
    name: String,
    width: u32,
    height: u32,
    frames: usize,
    config: Value,
    grid: Vec<i64>,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
    normalization: Vec<Vec<f32>>,
}

fn pattern(width: u32, height: u32, frame: usize) -> DynamicImage {
    let data = (0..width as usize * height as usize * 3)
        .map(|i| ((i * 37 + i / 13 + frame * 41) % 256) as u8)
        .collect();
    DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, data).unwrap())
}

#[test]
fn pixels_and_grids_match_transformers() {
    let fixtures: Fixtures = serde_json::from_str(include_str!("fixtures/processor.json")).unwrap();
    for case in fixtures.cases {
        let config = PreProcessorConfig::from_value(case.config).unwrap();
        let modality = if case.frames == 0 {
            Modality::Image
        } else {
            Modality::Video
        };
        let processor = Glm5NextProcessor::new(&config, modality).unwrap();
        let frames: Vec<_> =
            (0..case.frames.max(1)).map(|f| pattern(case.width, case.height, f)).collect();
        let result = if case.frames == 0 {
            processor.preprocess(&frames)
        } else {
            processor.preprocess_video(&frames)
        }
        .unwrap();
        let key = if case.frames == 0 {
            "image_grid_thw"
        } else {
            "video_grid_thw"
        };
        let ModelSpecificValue::IntTensor { data, .. } = &result.model_specific[key] else {
            panic!("grid is not an integer tensor");
        };
        assert_eq!(data, &case.grid, "{} grid", case.name);
        assert_eq!(
            result.encoder_input.shape(),
            case.shape,
            "{} shape",
            case.name
        );
        let mut hash = Sha256::new();
        let means = PreProcessorConfig::default().get_image_mean().map(|x| x as f32 * 255.0);
        let stds = PreProcessorConfig::default().get_image_std().map(|x| x as f32 * 255.0);
        for (i, &value) in result.encoder_input.iter().enumerate() {
            let c = (i / (2 * 14 * 14)) % 3;
            let pixel = (value * stds[c] + means[c]).round().clamp(0.0, 255.0) as u8;
            let expected = fixtures.normalization[c][pixel as usize];
            assert!(
                (value - expected).abs() < 5e-7,
                "{} normalization: {value} != {expected}",
                case.name
            );
            hash.update([pixel]);
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            case.sha256,
            "{} pixels",
            case.name
        );
    }
}

#[test]
fn image_batch_preserves_order_and_padding() {
    let processor =
        Glm5NextProcessor::new(&PreProcessorConfig::default(), Modality::Image).unwrap();
    let images = [pattern(93, 121, 0), pattern(31, 19, 1)];
    let batch = processor.preprocess(&images).unwrap();
    let individual: Vec<_> = images
        .iter()
        .flat_map(|i| {
            processor.preprocess(std::slice::from_ref(i)).unwrap().encoder_input.into_iter()
        })
        .collect();
    assert_eq!(batch.encoder_input.as_slice().unwrap(), individual);
    assert_eq!(batch.feature_token_counts.len(), 2);
}

#[test]
fn invalid_inputs_are_rejected() {
    let processor =
        Glm5NextProcessor::new(&PreProcessorConfig::default(), Modality::Video).unwrap();
    assert!(processor.preprocess(&[]).is_err());
    assert!(processor.preprocess_video(&[pattern(28, 28, 0), pattern(56, 28, 1)]).is_err());
    assert!(
        Glm5NextProcessor::new(
            &PreProcessorConfig::from_value(json!({"patch_size": 0})).unwrap(),
            Modality::Image
        )
        .is_err()
    );
}

#[test]
fn sampling_matches_transformers_for_short_long_and_fractional_fps_videos() {
    #[derive(Deserialize)]
    struct Sample {
        total: usize,
        fps: f64,
        duration: Option<f64>,
        count: usize,
        sha256: String,
    }
    let cases: Vec<Sample> = serde_json::from_str(include_str!("fixtures/sampling.json")).unwrap();
    let sampler = video::VideoSampling::new(&PreProcessorConfig::default()).unwrap();
    for case in cases {
        let indices = sampler.indices(case.total, case.fps, case.duration).unwrap();
        assert_eq!(indices.len(), case.count);
        let mut hash = Sha256::new();
        for index in indices {
            hash.update((index as i64).to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            case.sha256,
            "{} frames at {} fps",
            case.total,
            case.fps
        );
    }
}

#[test]
fn video_tokens_keep_fractional_timestamps_and_select_image_embeddings() {
    use crate::multimodal::{
        MultimodalModelContext, PreparedMedia, ResolvedPlaceholder, TokenizerResolver,
    };
    use std::sync::Arc;
    use vllm_text::tokenizer::Tokenizer as _;
    use vllm_tokenizer::test_utils::TestTokenizer;
    let tokenizer = Arc::new(
        TestTokenizer::new()
            .with_regular_token("<|image|>", 256)
            .with_regular_token("<|video|>", 257)
            .with_regular_token("<|begin_of_image|>", 258)
            .with_regular_token("<|end_of_image|>", 259)
            .with_regular_token("<|begin_of_video|>", 260)
            .with_regular_token("<|end_of_video|>", 261),
    );
    let context = MultimodalModelContext {
        model_id: "glm".into(),
        model_type: Some("glm5_next".into()),
        config: json!({"model_type":"glm5_next", "vision_config": {}}),
        tokenizer: TokenizerResolver(tokenizer.clone()),
    };
    let inputs = PreprocessedEncoderInputs {
        encoder_input: ndarray::ArrayD::zeros(vec![8, 1176]),
        feature_token_counts: vec![2],
        item_sizes: vec![(28, 28)],
        model_specific: HashMap::from([(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 2], 1, 3),
        )]),
    };
    let replacement = video_replacements(&context.metadata(), &inputs, &[0.0, 1.125]).unwrap();
    let prepared = PreparedMedia {
        modality: Modality::Video,
        placeholder: ResolvedPlaceholder::resolve(&Glm5NextSpec, &context, Modality::Video)
            .unwrap(),
        replacements: replacement,
        items: vec![],
    };
    let mut tokens = tokenizer
        .encode(
            "before<|begin_of_video|><|video|><|end_of_video|>after",
            false,
        )
        .unwrap();
    let ranges =
        crate::multimodal::expand::expand_prompt_token_ids(&mut tokens, &[prepared]).unwrap();
    assert_eq!(
        tokenizer.decode(&tokens, false).unwrap(),
        "before<|begin_of_video|><|begin_of_image|><|image|><|end_of_image|>0.0 seconds<|begin_of_image|><|image|><|end_of_image|>1.1 seconds<|end_of_video|>after"
    );
    let range = &ranges[&Modality::Video][0];
    assert_eq!(range.offset, 6);
    assert_eq!(tokens.iter().filter(|&&id| id == 256).count(), 2);
    assert!(range.is_embed.is_some());
}

#[tokio::test]
#[ignore = "requires ffmpeg and ffprobe"]
async fn video_decode_preserves_sampled_rgb_frames_and_timestamps() {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt as _;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.mkv");
    let mut command = tokio::process::Command::new("ffmpeg");
    command
        .args([
            "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "28x28", "-r", "4", "-i",
            "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0",
        ])
        .arg(&path)
        .stdin(Stdio::piped())
        .kill_on_drop(true);
    let mut child = command.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for index in 0..8 {
        stdin.write_all(pattern(28, 28, index).to_rgb8().as_raw()).await.unwrap();
    }
    drop(stdin);
    assert!(child.wait().await.unwrap().success());
    let sampler = video::VideoSampling::new(&PreProcessorConfig::default()).unwrap();
    let decoded = sampler
        .decode(
            std::fs::read(path).unwrap().into(),
            llm_multimodal::VideoSource::InlineBytes,
        )
        .await
        .unwrap();
    assert_eq!(decoded.timestamps.unwrap(), vec![0.0, 1.0]);
    assert_eq!(decoded.clip.frames.len(), 4);
    for (frame, index) in decoded.clip.frames.iter().zip([0, 2, 4, 6]) {
        assert_eq!(frame.to_rgb8(), pattern(28, 28, index).to_rgb8());
    }
}
