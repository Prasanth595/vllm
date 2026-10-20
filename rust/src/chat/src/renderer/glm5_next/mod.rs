// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod encoding;
mod tokenization;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;
use vllm_text::Prompt;
use vllm_text::tokenizer::DynTokenizer;

use super::{ChatRenderer, RenderedPrompt};
use crate::error::{Result, bail_chat_template_thrown};
use crate::request::ChatRequest;

#[derive(Clone, Default)]
pub struct Glm5NextChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
    multimodal: bool,
    tokenizer: Option<Arc<tokenization::SegmentTokenizer>>,
}

impl Glm5NextChatRenderer {
    pub fn new(default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            default_template_kwargs,
            multimodal: false,
            tokenizer: None,
        }
    }

    /// Enable bounded text-segment caching for compatible GLM tokenizer files.
    pub fn with_tokenizer(mut self, tokenizer: DynTokenizer, path: &Path) -> Result<Self> {
        self.tokenizer = tokenization::SegmentTokenizer::from_file(tokenizer, path)?.map(Arc::new);
        Ok(self)
    }

    pub fn with_multimodal(mut self, enabled: bool) -> Self {
        self.multimodal = enabled;
        self
    }
}

impl ChatRenderer for Glm5NextChatRenderer {
    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt> {
        request.validate()?;
        if request.chat_options.chat_template.is_some() {
            bail_chat_template_thrown!(
                "glm5_next does not accept a chat_template override; use tokenizer-mode hf for Jinja"
            );
        }
        let mut kwargs = self.default_template_kwargs.clone();
        kwargs.extend(request.chat_options.template_kwargs.clone());
        if let Some(effort) = &request.chat_options.reasoning_effort {
            kwargs.insert("reasoning_effort".into(), serde_json::json!(effort));
        }
        let messages = encoding::messages(&request.messages);
        let tools = if request.tool_parsing_enabled() {
            encoding::tools(request.tools())
        } else {
            Vec::new()
        };
        let (prompt, media_order) = if self.multimodal {
            encoding::render_with_media(
                &messages,
                &tools,
                request.chat_options.generation_prompt_mode,
                &kwargs,
            )?
        } else {
            (
                encoding::render(
                    &messages,
                    &tools,
                    request.chat_options.generation_prompt_mode,
                    &kwargs,
                )?,
                Vec::new(),
            )
        };
        let prompt = match &self.tokenizer {
            Some(tokenizer) => Prompt::TokenIds(tokenizer.encode(&prompt)?),
            None => Prompt::Text(prompt),
        };
        Ok(RenderedPrompt {
            prompt,
            media_order: Some(media_order),
            effective_template_kwargs: kwargs,
        })
    }
}

#[cfg(test)]
mod tests;
