// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod encoding;

use std::collections::HashMap;

use serde_json::Value;
use vllm_text::Prompt;

use super::{ChatRenderer, RenderedPrompt};
use crate::error::{Result, bail_chat_template_thrown};
use crate::request::ChatRequest;

#[derive(Clone, Default)]
pub struct Glm5NextChatRenderer {
    default_template_kwargs: HashMap<String, Value>,
    multimodal: bool,
}

impl Glm5NextChatRenderer {
    pub fn new(default_template_kwargs: HashMap<String, Value>) -> Self {
        Self {
            default_template_kwargs,
            multimodal: false,
        }
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
        Ok(RenderedPrompt {
            prompt: Prompt::Text(prompt),
            media_order: Some(media_order),
            effective_template_kwargs: kwargs,
        })
    }
}

#[cfg(test)]
mod tests;
