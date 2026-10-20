// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;

use aho_corasick::{AhoCorasick, MatchKind};
use parking_lot::Mutex;
use serde_json::Value;
use thiserror_ext::AsReport as _;
use vllm_text::tokenizer::DynTokenizer;

use crate::error::{Result, chat_template_thrown};

/// Split at added tokens, preserving each intervening BPE text span intact.
pub(super) struct SegmentTokenizer {
    tokenizer: DynTokenizer,
    matcher: AhoCorasick,
    token_ids: Vec<u32>,
    added_ids: HashSet<u32>,
    cache: Mutex<SegmentCache>,
}

const MAX_CACHE_BYTES: usize = 8 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 4096;
const MAX_SEGMENT_BYTES: usize = 16 * 1024;

// FIFO bounds both retained token data and per-entry allocation overhead.
#[derive(Default)]
struct SegmentCache {
    entries: HashMap<Arc<str>, Arc<[u32]>>,
    order: VecDeque<Arc<str>>,
    bytes: usize,
}

impl SegmentCache {
    fn insert(&mut self, text: &str, ids: Arc<[u32]>) {
        if text.len() > MAX_SEGMENT_BYTES || self.entries.contains_key(text) {
            return;
        }
        let bytes = text.len() + std::mem::size_of_val(ids.as_ref());
        while self.bytes + bytes > MAX_CACHE_BYTES || self.entries.len() >= MAX_CACHE_ENTRIES {
            let Some(key) = self.order.pop_front() else {
                return;
            };
            if let Some(ids) = self.entries.remove(&key) {
                self.bytes -= key.len() + std::mem::size_of_val(ids.as_ref());
            }
        }
        let key: Arc<str> = Arc::from(text);
        self.order.push_back(key.clone());
        self.entries.insert(key, ids);
        self.bytes += bytes;
    }
}

impl SegmentTokenizer {
    pub(super) fn from_file(tokenizer: DynTokenizer, path: &Path) -> Result<Option<Self>> {
        let data = std::fs::read(path).map_err(|error| {
            chat_template_thrown!("failed to read GLM tokenizer: {}", error.as_report())
        })?;
        let config: Value = serde_json::from_slice(&data).map_err(|error| {
            chat_template_thrown!("failed to parse GLM tokenizer: {}", error.as_report())
        })?;
        Self::from_config(tokenizer, &config)
    }

    pub(super) fn from_config(tokenizer: DynTokenizer, config: &Value) -> Result<Option<Self>> {
        // Prefix insertion, normalization and token whitespace rules can make
        // independently encoded spans differ from whole-prompt encoding.
        let steps = config["pre_tokenizer"]["pretokenizers"].as_array();
        if !config["normalizer"].is_null()
            || !config["truncation"].is_null()
            || !config["padding"].is_null()
            || !config["model"]["dropout"].is_null()
            || config["model"]["type"] != "BPE"
            || config["pre_tokenizer"]["type"] != "Sequence"
            || !steps.is_some_and(|steps| {
                steps.len() == 2
                    && steps[0]["type"] == "Split"
                    && steps[1]["type"] == "ByteLevel"
                    && steps[1]["add_prefix_space"] == false
            })
            || !(config["post_processor"].is_null()
                || config["post_processor"]["type"] == "ByteLevel")
        {
            return Ok(None);
        }
        let Some(tokens) = config["added_tokens"].as_array().filter(|tokens| !tokens.is_empty())
        else {
            return Ok(None);
        };
        let mut patterns = Vec::with_capacity(tokens.len());
        let mut token_ids = Vec::with_capacity(tokens.len());
        for token in tokens {
            let Some(content) = token["content"].as_str().filter(|text| !text.is_empty()) else {
                return Ok(None);
            };
            let Some(id) = token["id"].as_u64().and_then(|id| u32::try_from(id).ok()) else {
                return Ok(None);
            };
            if ["single_word", "lstrip", "rstrip", "normalized"]
                .iter()
                .any(|flag| token[*flag] != false)
                || tokenizer.token_to_id(content) != Some(id)
            {
                return Ok(None);
            }
            patterns.push(content);
            token_ids.push(id);
        }
        if tokenizer.added_vocab().len() != patterns.len()
            || !tokenizer.added_vocab().iter().all(|(text, id)| {
                patterns
                    .iter()
                    .zip(&token_ids)
                    .any(|(pattern, token_id)| text == pattern && id == token_id)
            })
        {
            return Ok(None);
        }
        let added_ids: HashSet<u32> = token_ids.iter().copied().collect();
        // Whole-prompt results can be partitioned at added-token IDs only if
        // ordinary BPE tokens cannot produce those same IDs.
        let Some(vocab) = config["model"]["vocab"].as_object() else {
            return Ok(None);
        };
        if vocab.values().any(|id| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .is_some_and(|id| added_ids.contains(&id))
        }) {
            return Ok(None);
        }
        let matcher = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns)
            .map_err(|error| {
                chat_template_thrown!("failed to build GLM token matcher: {}", error.as_report())
            })?;
        Ok(Some(Self {
            tokenizer,
            matcher,
            token_ids,
            added_ids,
            cache: Mutex::new(SegmentCache::default()),
        }))
    }

    fn text_tokens(&self, text: &str) -> Result<Arc<[u32]>> {
        if text.len() > MAX_SEGMENT_BYTES {
            return Ok(self.tokenizer.encode_ordinary(text)?.into());
        }
        if let Some(ids) = self.cache.lock().entries.get(text).cloned() {
            return Ok(ids);
        }
        let ids: Arc<[u32]> = self.tokenizer.encode_ordinary(text)?.into();
        self.cache.lock().insert(text, ids.clone());
        Ok(ids)
    }

    pub(super) fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let mut missing_bytes = 0;
        let mut text_bytes = 0;
        let mut cursor = 0;
        {
            let cache = self.cache.lock();
            for end in self
                .matcher
                .find_iter(prompt)
                .map(|matched| (matched.start(), matched.end()))
                .chain(std::iter::once((prompt.len(), prompt.len())))
            {
                let text = &prompt[cursor..end.0];
                text_bytes += text.len();
                if !cache.entries.contains_key(text) {
                    missing_bytes += text.len();
                }
                cursor = end.1;
            }
        }
        // Cold requests avoid per-segment tokenizer calls and seed the cache.
        if missing_bytes > text_bytes / 2 {
            return self.encode_whole_and_cache(prompt);
        }
        let mut ids = Vec::new();
        let mut cursor = 0;
        for matched in self.matcher.find_iter(prompt) {
            if matched.start() > cursor {
                ids.extend_from_slice(&self.text_tokens(&prompt[cursor..matched.start()])?);
            }
            ids.push(self.token_ids[matched.pattern().as_usize()]);
            cursor = matched.end();
        }
        if cursor < prompt.len() {
            ids.extend_from_slice(&self.text_tokens(&prompt[cursor..])?);
        }
        Ok(ids)
    }

    fn encode_whole_and_cache(&self, prompt: &str) -> Result<Vec<u32>> {
        let ids = self.tokenizer.encode(prompt, false)?;
        let mut text_cursor = 0;
        let mut token_cursor = 0;
        let mut markers = ids.iter().enumerate().filter(|(_, id)| self.added_ids.contains(id));
        let mut cache = self.cache.lock();
        for matched in self.matcher.find_iter(prompt) {
            let Some((position, id)) = markers.next() else {
                return Ok(ids);
            };
            if *id != self.token_ids[matched.pattern().as_usize()] {
                return Ok(ids);
            }
            let text = &prompt[text_cursor..matched.start()];
            if !text.is_empty() && text.len() <= MAX_SEGMENT_BYTES {
                cache.insert(text, Arc::from(&ids[token_cursor..position]));
            }
            text_cursor = matched.end();
            token_cursor = position + 1;
        }
        if text_cursor < prompt.len()
            && prompt.len() - text_cursor <= MAX_SEGMENT_BYTES
            && markers.next().is_none()
        {
            cache.insert(&prompt[text_cursor..], Arc::from(&ids[token_cursor..]));
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_cache_evicts_old_entries_and_rejects_oversized_text() {
        let mut cache = SegmentCache::default();
        for index in 0..=MAX_CACHE_ENTRIES {
            cache.insert(&index.to_string(), Arc::from([index as u32]));
        }
        assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
        assert!(!cache.entries.contains_key("0"));
        let ids: Arc<[u32]> = vec![0; MAX_SEGMENT_BYTES].into();
        for index in 0..MAX_CACHE_ENTRIES {
            cache.insert(&format!("large-{index}"), ids.clone());
        }
        assert!(cache.bytes <= MAX_CACHE_BYTES);
        assert!(cache.entries.len() < MAX_CACHE_ENTRIES);
        let size = cache.entries.len();
        cache.insert(&"x".repeat(MAX_SEGMENT_BYTES + 1), Arc::from([1]));
        assert_eq!(cache.entries.len(), size);
    }
}
