// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod processor;
mod resize;
pub(super) mod video;

use std::collections::HashMap;

use llm_multimodal::registry::{ModelRegistryError, RegistryResult};
use llm_multimodal::{
    EncoderFieldLayouts, FieldLayout, Modality, ModelMetadata, ModelProcessorSpec,
    ModelSpecificValue, PreProcessorConfig, PreprocessedEncoderInputs, PromptReplacement, TokenId,
    VisionPreProcessor,
};
use serde_json::{Value, json};

use processor::Glm5NextProcessor;

pub(super) struct Glm5NextSpec;

impl ModelProcessorSpec for Glm5NextSpec {
    fn name(&self) -> &'static str {
        "glm5_next"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some("glm5_next")
            && metadata.config.get("vision_config").is_some_and(Value::is_object)
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".into())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id("<|image|>")
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Ok("<|video|>".into()),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image | Modality::Video => self.placeholder_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([
            (Modality::Image, usize::MAX),
            (Modality::Video, usize::MAX),
        ]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn vision_processor(
        &self,
        _metadata: &ModelMetadata,
        config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image | Modality::Video => Glm5NextProcessor::new(config, modality)
                .map(|p| Box::new(p) as Box<dyn VisionPreProcessor>)
                .map_err(|e| ModelRegistryError::InvalidConfig {
                    message: format!("{e}"),
                }),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        inputs: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_id = self.placeholder_token_id(metadata)?;
        Ok(inputs
            .feature_token_counts
            .iter()
            .map(|n| PromptReplacement::sequence(Modality::Image, "<|image|>", vec![image_id; *n]))
            .collect())
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        let name = if modality == Modality::Video {
            "video"
        } else {
            "image"
        };
        EncoderFieldLayouts {
            encoder_input: FieldLayout::flat(format!("patches_per_{name}")),
            model_specific: HashMap::from([
                (format!("{name}_grid_thw"), FieldLayout::Batched),
                (format!("patches_per_{name}"), FieldLayout::Batched),
            ]),
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".into(), "video_grid_thw".into()]
    }
}

pub(super) fn video_replacements(
    metadata: &ModelMetadata,
    inputs: &PreprocessedEncoderInputs,
    timestamps: &[f64],
) -> RegistryResult<Vec<PromptReplacement>> {
    let invalid = || ModelRegistryError::InvalidConfig {
        message: "missing GLM video grid or timestamps".into(),
    };
    let Some(ModelSpecificValue::IntTensor { data: grid, shape }) =
        inputs.model_specific.get("video_grid_thw")
    else {
        return Err(invalid());
    };
    if shape != &[1, 3]
        || grid.len() != 3
        || grid[0] <= 0
        || timestamps.len() != grid[0] as usize
        || inputs.feature_token_counts.len() != 1
    {
        return Err(invalid());
    }
    let image_id = metadata.token_id("<|image|>")?;
    let begin_image = metadata.token_id("<|begin_of_image|>")?;
    let end_image = metadata.token_id("<|end_of_image|>")?;
    let count = inputs.feature_token_counts[0] / timestamps.len();
    let mut tokens = Vec::new();
    for seconds in timestamps {
        tokens.push(begin_image);
        tokens.extend(std::iter::repeat_n(image_id, count));
        tokens.push(end_image);
        let stamp = format!("{seconds:.1} seconds");
        let ids = metadata.tokenizer.encode_text(&stamp).ok_or_else(|| {
            ModelRegistryError::InvalidConfig {
                message: "failed to tokenize GLM video timestamp".into(),
            }
        })?;
        tokens.extend(ids.into_iter().map(|id| id as TokenId));
    }
    Ok(vec![
        PromptReplacement::sequence(Modality::Video, "<|video|>", tokens).with_structural_prefix(1),
    ])
}

#[cfg(test)]
mod tests;
