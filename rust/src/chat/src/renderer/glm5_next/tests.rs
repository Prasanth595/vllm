// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Value, json};

use super::{Glm5NextChatRenderer, encoding};
use crate::renderer::hf::{ChatTemplateContentFormatOption, HfChatRenderer, MultimodalRenderInfo};
use crate::request::{
    ChatMessage, ChatRequest, ChatTool, GenerationPromptMode, ResolvedToolContext,
};
use crate::{AssistantContentBlock, AssistantToolCall, ChatRenderer, EffortValue};

#[derive(Deserialize)]
struct TestCase {
    name: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    add_generation_prompt: bool,
    kwargs: HashMap<String, Value>,
    expected: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct TestCases {
    cases: Vec<TestCase>,
}

#[test]
fn renders_expected_prompts_and_rejects_invalid_input() {
    let fixtures: TestCases =
        serde_json::from_str(include_str!("fixtures/test_cases.json")).unwrap();
    for case in fixtures.cases {
        let result = encoding::render(
            &case.messages,
            &case.tools,
            case.add_generation_prompt,
            &case.kwargs,
        );
        if case.error.is_some() {
            assert!(
                result.is_err(),
                "{}: expected rejection, got {result:?}",
                case.name
            );
        } else {
            assert_eq!(result.unwrap(), case.expected.unwrap(), "{}", case.name);
        }
    }
}

fn tool_request(count: usize) -> ChatRequest {
    let calls = (0..count)
        .map(|index| {
            AssistantContentBlock::ToolCall(AssistantToolCall {
                id: format!("call_{index}"),
                name: "lookup".into(),
                arguments: json!({"q": format!("城市 {index}"), "values": [true, null, 1.5]})
                    .to_string(),
            })
        })
        .collect();
    let mut messages = vec![
        ChatMessage::system("助手"),
        ChatMessage::user("查询天气"),
        ChatMessage::assistant_blocks(calls),
    ];
    messages.extend(
        (0..count).rev().map(|index| {
            ChatMessage::tool_response(format!("结果 {index}"), format!("call_{index}"))
        }),
    );
    let tools = vec![ChatTool {
        name: "lookup".into(),
        description: Some("查询天气".into()),
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: Some(true),
        defer_loading: None,
    }];
    let tool_context = ResolvedToolContext::new(&messages, tools, None, true).unwrap();
    ChatRequest {
        messages,
        tool_context,
        ..ChatRequest::for_test()
    }
}

const CHECKPOINT_TEMPLATES: [(&str, &str); 2] = [
    ("GLM-5.3", include_str!("fixtures/glm_5_3.jinja")),
    (
        "GLM-5.3-Flash",
        include_str!("fixtures/glm_5_3_flash.jinja"),
    ),
];

fn hf_renderer(template: &str) -> HfChatRenderer {
    HfChatRenderer::new(
        Some(template.into()),
        HashMap::new(),
        ChatTemplateContentFormatOption::OpenAi,
    )
    .unwrap()
}

#[test]
fn typed_requests_match_existing_hf_renderer() {
    let native = Glm5NextChatRenderer::default();
    for (name, template) in CHECKPOINT_TEMPLATES {
        let hf = hf_renderer(template);
        for count in [1, 2, 8] {
            let mut request = tool_request(count);
            for clear in [false, true] {
                request
                    .chat_options
                    .template_kwargs
                    .insert("clear_thinking".into(), json!(clear));
                assert_eq!(
                    native.render(&request).unwrap().prompt,
                    hf.render(&request).unwrap().prompt,
                    "{name}"
                );
            }
            request.tool_context.tool_choice = crate::request::ChatToolChoice::None;
            assert_eq!(
                native.render(&request).unwrap().prompt,
                hf.render(&request).unwrap().prompt
            );
        }
    }
}

#[test]
fn flash_media_matches_checkpoint_template() {
    let hf = hf_renderer(CHECKPOINT_TEMPLATES[1].1).with_multimodal(Some(MultimodalRenderInfo {
        image_token: Some("<|image|>".into()),
        video_token: Some("<|video|>".into()),
        audio_token: Some("<|begin_of_audio|>".into()),
    }));
    let mut request = tool_request(2);
    request.messages.push(ChatMessage::user(vec![
        crate::request::ChatContentPart::image_url("unused"),
    ]));
    let expected = hf.render(&request).unwrap();
    let renderer = Glm5NextChatRenderer::default().with_multimodal(true);
    let actual = renderer.render(&request).unwrap();
    assert_eq!(actual.prompt, expected.prompt);
    assert_eq!(
        actual.media_order.as_ref().unwrap(),
        &[crate::renderer::MediaPartSource {
            message_index: 5,
            content_part_index: 0
        }]
    );
    assert!(
        actual
            .prompt
            .into_text()
            .unwrap()
            .contains("<|begin_of_image|><|image|><|end_of_image|>")
    );
}

#[test]
fn media_order_tracks_reordered_tool_responses() {
    use crate::renderer::MediaPartSource;
    use crate::request::ChatContentPart;
    let mut request = tool_request(2);
    request.messages[3] =
        ChatMessage::tool_response(vec![ChatContentPart::image_url("second")], "call_1");
    request.messages[4] =
        ChatMessage::tool_response(vec![ChatContentPart::image_url("first")], "call_0");
    let native = Glm5NextChatRenderer::default().with_multimodal(true).render(&request).unwrap();
    let hf = hf_renderer(CHECKPOINT_TEMPLATES[1].1)
        .with_multimodal(Some(MultimodalRenderInfo {
            image_token: Some("<|image|>".into()),
            video_token: Some("<|video|>".into()),
            audio_token: None,
        }))
        .render(&request)
        .unwrap();
    assert_eq!(native.prompt, hf.prompt);
    assert_eq!(
        native.media_order.unwrap(),
        vec![
            MediaPartSource {
                message_index: 4,
                content_part_index: 0
            },
            MediaPartSource {
                message_index: 3,
                content_part_index: 0
            }
        ]
    );
}

#[test]
fn defaults_request_kwargs_and_typed_effort_have_defined_precedence() {
    let renderer = Glm5NextChatRenderer::new(HashMap::from([
        ("reasoning_effort".into(), json!("low")),
        ("clear_thinking".into(), json!(true)),
    ]));
    let mut request = ChatRequest::for_test();
    request
        .chat_options
        .template_kwargs
        .insert("reasoning_effort".into(), json!("high"));
    let rendered = renderer.render(&request).unwrap();
    assert!(rendered.prompt.into_text().unwrap().contains("Reasoning Effort: High"));
    request.chat_options.reasoning_effort = Some(EffortValue::from("low"));
    let rendered = renderer.render(&request).unwrap();
    assert!(rendered.prompt.into_text().unwrap().contains("Reasoning Effort: Low"));
    assert_eq!(
        rendered.effective_template_kwargs["reasoning_effort"],
        "low"
    );
    assert_eq!(rendered.effective_template_kwargs["clear_thinking"], true);
}

#[test]
fn empty_explicit_reasoning_suppresses_embedded_tag_parsing() {
    let request = ChatRequest {
        messages: vec![ChatMessage::assistant_blocks(vec![
            AssistantContentBlock::Reasoning {
                text: String::new(),
            },
            AssistantContentBlock::Text {
                text: "<think>literal</think>answer".into(),
            },
        ])],
        ..ChatRequest::for_test()
    };
    let rendered = Glm5NextChatRenderer::default().render(&request).unwrap();
    assert_eq!(
        rendered.prompt.into_text().unwrap(),
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|assistant|><think></think><think>literal</think>answer<|assistant|><think>"
    );
}

#[test]
fn unsupported_overrides_and_continuations_are_client_errors() {
    let renderer = Glm5NextChatRenderer::default();
    let mut request = ChatRequest::for_test();
    request.chat_options.chat_template = Some("override".into());
    assert!(renderer.render(&request).unwrap_err().is_request_validation_error());
    request.chat_options.chat_template = None;
    request.messages = vec![ChatMessage::assistant_text("prefix")];
    request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
    assert!(renderer.render(&request).unwrap_err().is_request_validation_error());
}

#[test]
fn media_parts_render_reminders_without_multimodal_inputs() {
    let request = ChatRequest {
        messages: vec![ChatMessage::user(vec![
            crate::request::ChatContentPart::image_url("unused"),
        ])],
        ..ChatRequest::for_test()
    };
    let rendered = Glm5NextChatRenderer::default().render(&request).unwrap();
    assert!(rendered.prompt.into_text().unwrap().contains("unable to process this image"));
    assert_eq!(rendered.media_order, Some(Vec::new()));
}
