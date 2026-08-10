use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<InputToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InputToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: InputFunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InputFunctionCall {
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OutputFunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OutputFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedAssistant {
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<OutputToolCall>,
    pub malformed_tool_markup: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolChoiceMode {
    None,
    Auto,
    Required,
    Specific,
}

#[derive(Clone, Debug)]
pub struct SelectedTools {
    pub tools: Vec<Value>,
    pub choice: ToolChoiceMode,
    pub required_name: Option<String>,
}

#[derive(Debug, Error)]
pub enum ChatTemplateError {
    #[error("messages must contain at least one item")]
    NoMessages,
    #[error("system message must be the first message")]
    SystemMessageOrder,
    #[error("unsupported message role: {0}")]
    UnsupportedRole(String),
    #[error("message content must be a string, null, or an array of text parts")]
    InvalidContent,
    #[error("image and video content are not supported by this text runtime")]
    UnsupportedMedia,
    #[error("tool at index {0} must be an OpenAI function tool")]
    InvalidTool(usize),
    #[error("function tool at index {0} has no valid name")]
    InvalidToolName(usize),
    #[error("tool_choice requires a tools array")]
    ToolChoiceWithoutTools,
    #[error("unsupported tool_choice value")]
    InvalidToolChoice,
    #[error("tool_choice names an unknown function: {0}")]
    UnknownTool(String),
    #[error("assistant tool call arguments must be a JSON object or an object-encoded string")]
    InvalidToolArguments,
    #[error("no user query was found in messages")]
    NoUserQuery,
}

pub fn select_tools(
    tools: &[Value],
    tool_choice: Option<&Value>,
) -> Result<SelectedTools, ChatTemplateError> {
    let mut names = Vec::with_capacity(tools.len());
    for (index, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ChatTemplateError::InvalidTool(index));
        }
        let Some(name) = tool
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
        else {
            return Err(ChatTemplateError::InvalidToolName(index));
        };
        names.push(name.to_owned());
    }

    let Some(choice) = tool_choice else {
        return Ok(SelectedTools {
            tools: tools.to_vec(),
            choice: ToolChoiceMode::Auto,
            required_name: None,
        });
    };
    if let Some(choice) = choice.as_str() {
        return match choice {
            "none" => Ok(SelectedTools {
                tools: Vec::new(),
                choice: ToolChoiceMode::None,
                required_name: None,
            }),
            "auto" => Ok(SelectedTools {
                tools: tools.to_vec(),
                choice: ToolChoiceMode::Auto,
                required_name: None,
            }),
            "required" if tools.is_empty() => Err(ChatTemplateError::ToolChoiceWithoutTools),
            "required" => Ok(SelectedTools {
                tools: tools.to_vec(),
                choice: ToolChoiceMode::Required,
                required_name: None,
            }),
            _ => Err(ChatTemplateError::InvalidToolChoice),
        };
    }

    let name = choice
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| *kind == "function")
        .and_then(|_| choice.get("function"))
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .ok_or(ChatTemplateError::InvalidToolChoice)?;
    let Some(index) = names.iter().position(|candidate| candidate == name) else {
        return Err(ChatTemplateError::UnknownTool(name.to_owned()));
    };
    Ok(SelectedTools {
        tools: vec![tools[index].clone()],
        choice: ToolChoiceMode::Specific,
        required_name: Some(name.to_owned()),
    })
}

pub fn render_qwen_chat(
    messages: &[ChatMessage],
    tools: &[Value],
    enable_thinking: bool,
    preserve_thinking: bool,
) -> Result<String, ChatTemplateError> {
    if messages.is_empty() {
        return Err(ChatTemplateError::NoMessages);
    }
    let rendered_contents: Vec<String> = messages
        .iter()
        .map(|message| render_content(&message.content))
        .collect::<Result<_, _>>()?;
    for (index, message) in messages.iter().enumerate() {
        if message.role == "system" && index != 0 {
            return Err(ChatTemplateError::SystemMessageOrder);
        }
    }

    let mut output = String::new();
    if !tools.is_empty() {
        output.push_str("<|im_start|>system\n");
        output.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools {
            output.push('\n');
            output.push_str(&jinja_tojson(tool));
        }
        output.push_str("\n</tools>");
        output.push_str(
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>",
        );
        if matches!(messages[0].role.as_str(), "system" | "developer") {
            let system = rendered_contents[0].trim();
            if !system.is_empty() {
                output.push_str("\n\n");
                output.push_str(system);
            }
        }
        output.push_str("<|im_end|>\n");
    } else if matches!(messages[0].role.as_str(), "system" | "developer") {
        output.push_str("<|im_start|>system\n");
        output.push_str(rendered_contents[0].trim());
        output.push_str("<|im_end|>\n");
    }

    let last_query_index = messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            if message.role != "user" {
                return None;
            }
            let content = rendered_contents[index].trim();
            (!content.starts_with("<tool_response>") || !content.ends_with("</tool_response>"))
                .then_some(index)
        })
        .ok_or(ChatTemplateError::NoUserQuery)?;

    for (index, message) in messages.iter().enumerate() {
        let mut content = rendered_contents[index].trim().to_owned();
        match message.role.as_str() {
            "system" => continue,
            "developer" => {
                if index == 0 {
                    continue;
                }
                output.push_str("<|im_start|>system\n");
                output.push_str(&content);
                output.push_str("<|im_end|>\n");
            }
            "user" => {
                output.push_str("<|im_start|>user\n");
                output.push_str(&content);
                output.push_str("<|im_end|>\n");
            }
            "assistant" => {
                let mut reasoning = message.reasoning_content.clone().unwrap_or_default();
                if message.reasoning_content.is_none() {
                    if let Some(close_index) = content.find("</think>") {
                        let prefix = &content[..close_index];
                        reasoning = prefix
                            .rsplit_once("<think>")
                            .map(|(_, value)| value)
                            .unwrap_or(prefix)
                            .trim()
                            .to_owned();
                        content = content[close_index + "</think>".len()..]
                            .trim_start_matches('\n')
                            .to_owned();
                    }
                }
                output.push_str("<|im_start|>assistant\n");
                if preserve_thinking || index > last_query_index {
                    output.push_str("<think>\n");
                    output.push_str(reasoning.trim());
                    output.push_str("\n</think>\n\n");
                }
                output.push_str(&content);
                if let Some(tool_calls) = &message.tool_calls {
                    for (tool_index, tool_call) in tool_calls.iter().enumerate() {
                        if tool_call.kind != "function" {
                            return Err(ChatTemplateError::InvalidToolArguments);
                        }
                        if tool_index == 0 && !content.trim().is_empty() {
                            output.push_str("\n\n");
                        } else if tool_index != 0 {
                            output.push('\n');
                        }
                        output.push_str("<tool_call>\n<function=");
                        output.push_str(&tool_call.function.name);
                        output.push_str(">\n");
                        let arguments = arguments_object(&tool_call.function.arguments)?;
                        for (name, value) in arguments {
                            output.push_str("<parameter=");
                            output.push_str(name);
                            output.push_str(">\n");
                            if let Some(value) = value.as_str() {
                                output.push_str(value);
                            } else {
                                output.push_str(&jinja_tojson(value));
                            }
                            output.push_str("\n</parameter>\n");
                        }
                        output.push_str("</function>\n</tool_call>");
                    }
                }
                output.push_str("<|im_end|>\n");
            }
            "tool" => {
                if index > 0 && messages[index - 1].role != "tool" {
                    output.push_str("<|im_start|>user");
                }
                output.push_str("\n<tool_response>\n");
                output.push_str(&content);
                output.push_str("\n</tool_response>");
                if index + 1 == messages.len() || messages[index + 1].role != "tool" {
                    output.push_str("<|im_end|>\n");
                }
            }
            role => return Err(ChatTemplateError::UnsupportedRole(role.to_owned())),
        }
    }

    output.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        output.push_str("<think>\n");
    } else {
        output.push_str("<think>\n\n</think>\n\n");
    }
    Ok(output)
}

// Hugging Face's Jinja environment uses Jinja's default `tojson` policy for
// this model template: recursively sorted object keys, one space after each
// comma and colon, ASCII JSON escapes, and HTML-safe escapes.  Tokenization is
// sensitive to every one of those bytes, so serde_json's compact formatter is
// not interchangeable here.
fn jinja_tojson(value: &Value) -> String {
    fn push_string(output: &mut String, value: &str) {
        output.push('"');
        for character in value.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\u{0008}' => output.push_str("\\b"),
                '\u{000c}' => output.push_str("\\f"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                '<' => output.push_str("\\u003c"),
                '>' => output.push_str("\\u003e"),
                '&' => output.push_str("\\u0026"),
                '\'' => output.push_str("\\u0027"),
                character if (' '..='~').contains(&character) => output.push(character),
                character => {
                    let scalar = character as u32;
                    if scalar <= 0xffff {
                        output.push_str(&format!("\\u{scalar:04x}"));
                    } else {
                        let surrogate = scalar - 0x10000;
                        let high = 0xd800 + (surrogate >> 10);
                        let low = 0xdc00 + (surrogate & 0x03ff);
                        output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                    }
                }
            }
        }
        output.push('"');
    }

    fn push_value(output: &mut String, value: &Value) {
        match value {
            Value::Null => output.push_str("null"),
            Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Value::Number(value) => output.push_str(&value.to_string()),
            Value::String(value) => push_string(output, value),
            Value::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    push_value(output, value);
                }
                output.push(']');
            }
            Value::Object(values) => {
                let mut entries = values.iter().collect::<Vec<_>>();
                entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
                output.push('{');
                for (index, (key, value)) in entries.into_iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    push_string(output, key);
                    output.push_str(": ");
                    push_value(output, value);
                }
                output.push('}');
            }
        }
    }

    let mut output = String::new();
    push_value(&mut output, value);
    output
}

fn render_content(content: &Value) -> Result<String, ChatTemplateError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(value) => Ok(value.clone()),
        Value::Array(parts) => {
            let mut output = String::new();
            for part in parts {
                let Some(object) = part.as_object() else {
                    return Err(ChatTemplateError::InvalidContent);
                };
                let kind = object.get("type").and_then(Value::as_str);
                if matches!(kind, Some("image" | "image_url" | "video"))
                    || object.contains_key("image")
                    || object.contains_key("image_url")
                    || object.contains_key("video")
                {
                    return Err(ChatTemplateError::UnsupportedMedia);
                }
                let Some(text) = object.get("text").and_then(Value::as_str) else {
                    return Err(ChatTemplateError::InvalidContent);
                };
                output.push_str(text);
            }
            Ok(output)
        }
        _ => Err(ChatTemplateError::InvalidContent),
    }
}

fn arguments_object(value: &Value) -> Result<&Map<String, Value>, ChatTemplateError> {
    if let Some(arguments) = value.as_object() {
        return Ok(arguments);
    }
    Err(ChatTemplateError::InvalidToolArguments)
}

pub fn normalize_input_tool_arguments(
    messages: &mut [ChatMessage],
) -> Result<(), ChatTemplateError> {
    for message in messages {
        let Some(tool_calls) = &mut message.tool_calls else {
            continue;
        };
        for tool_call in tool_calls {
            if let Some(encoded) = tool_call.function.arguments.as_str() {
                let parsed: Value = serde_json::from_str(encoded)
                    .map_err(|_| ChatTemplateError::InvalidToolArguments)?;
                if !parsed.is_object() {
                    return Err(ChatTemplateError::InvalidToolArguments);
                }
                tool_call.function.arguments = parsed;
            }
            if !tool_call.function.arguments.is_object() {
                return Err(ChatTemplateError::InvalidToolArguments);
            }
        }
    }
    Ok(())
}

pub fn parse_assistant_output(
    raw: &str,
    thinking_enabled: bool,
    request_id: &str,
) -> ParsedAssistant {
    let (reasoning_content, visible) = split_reasoning(raw, thinking_enabled);
    let mut parsed = ParsedAssistant {
        reasoning_content,
        ..ParsedAssistant::default()
    };
    let Some(first_tool) = visible.find("<tool_call>") else {
        parsed.content = nonempty(visible.trim().to_owned());
        return parsed;
    };

    let prefix = visible[..first_tool].trim_end();
    parsed.content = nonempty(prefix.to_owned());
    let mut cursor = first_tool;
    while let Some(relative_start) = visible[cursor..].find("<tool_call>") {
        let start = cursor + relative_start;
        let body_start = start + "<tool_call>".len();
        let Some(relative_end) = visible[body_start..].find("</tool_call>") else {
            parsed.malformed_tool_markup = true;
            parsed.tool_calls.clear();
            parsed.content = nonempty(visible.trim().to_owned());
            return parsed;
        };
        let body_end = body_start + relative_end;
        match parse_tool_call(
            &visible[body_start..body_end],
            request_id,
            parsed.tool_calls.len(),
        ) {
            Some(call) => parsed.tool_calls.push(call),
            None => {
                parsed.malformed_tool_markup = true;
                parsed.tool_calls.clear();
                parsed.content = nonempty(visible.trim().to_owned());
                return parsed;
            }
        }
        cursor = body_end + "</tool_call>".len();
        if !visible[cursor..].trim().is_empty()
            && !visible[cursor..].trim_start().starts_with("<tool_call>")
        {
            parsed.malformed_tool_markup = true;
            parsed.tool_calls.clear();
            parsed.content = nonempty(visible.trim().to_owned());
            return parsed;
        }
    }
    parsed
}

fn split_reasoning(raw: &str, thinking_enabled: bool) -> (Option<String>, &str) {
    if !thinking_enabled {
        return (None, raw);
    }
    if let Some(close) = raw.find("</think>") {
        let prefix = &raw[..close];
        let reasoning = prefix.strip_prefix("<think>").unwrap_or(prefix).trim();
        let visible = raw[close + "</think>".len()..].trim_start_matches('\n');
        return (nonempty(reasoning.to_owned()), visible);
    }
    (nonempty(raw.trim().to_owned()), "")
}

fn parse_tool_call(body: &str, request_id: &str, index: usize) -> Option<OutputToolCall> {
    let body = body.trim();
    if body.starts_with('{') {
        let value: Value = serde_json::from_str(body).ok()?;
        let name = value.get("name")?.as_str()?.to_owned();
        let arguments = value
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(Map::new()));
        arguments.as_object()?;
        let encoded_arguments = serde_json::to_string(&arguments).ok()?;
        return Some(OutputToolCall {
            id: format!("call_{request_id}_{index}"),
            kind: "function".to_owned(),
            function: OutputFunctionCall {
                name,
                arguments: encoded_arguments,
            },
        });
    }

    let function_start = body.find("<function=")? + "<function=".len();
    let name_end = body[function_start..].find('>')? + function_start;
    let name = body[function_start..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let function_close = body.rfind("</function>")?;
    if function_close < name_end {
        return None;
    }
    let parameter_body = &body[name_end + 1..function_close];
    let mut arguments = Map::new();
    let mut cursor = 0;
    while let Some(relative_start) = parameter_body[cursor..].find("<parameter=") {
        let start = cursor + relative_start + "<parameter=".len();
        let key_end = parameter_body[start..].find('>')? + start;
        let key = parameter_body[start..key_end].trim();
        if key.is_empty() {
            return None;
        }
        let value_start = key_end + 1;
        let value_end = parameter_body[value_start..].find("</parameter>")? + value_start;
        let raw_value = parameter_body[value_start..value_end]
            .strip_prefix('\n')
            .unwrap_or(&parameter_body[value_start..value_end])
            .strip_suffix('\n')
            .unwrap_or_else(|| {
                parameter_body[value_start..value_end]
                    .strip_prefix('\n')
                    .unwrap_or(&parameter_body[value_start..value_end])
            });
        let value =
            serde_json::from_str(raw_value).unwrap_or_else(|_| Value::String(raw_value.to_owned()));
        arguments.insert(key.to_owned(), value);
        cursor = value_end + "</parameter>".len();
    }
    Some(OutputToolCall {
        id: format!("call_{request_id}_{index}"),
        kind: "function".to_owned(),
        function: OutputFunctionCall {
            name: name.to_owned(),
            arguments: serde_json::to_string(&arguments).ok()?,
        },
    })
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn function_type() -> String {
    "function".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn basic_non_thinking_prompt_matches_qwen_template() {
        let messages = vec![ChatMessage {
            role: "user".to_owned(),
            content: json!("Reply with exactly OK."),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }];
        let prompt = render_qwen_chat(&messages, &[], false, false).unwrap();
        assert_eq!(
            prompt,
            "<|im_start|>user\nReply with exactly OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn tool_json_matches_huggingface_jinja_tojson() {
        let tool = json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "查找 <item>",
                "parameters": {
                    "type": "object",
                    "required": ["q"],
                    "properties": {"q": {"type": "string"}}
                }
            }
        });
        assert_eq!(
            jinja_tojson(&tool),
            r#"{"function": {"description": "\u67e5\u627e \u003citem\u003e", "name": "lookup", "parameters": {"properties": {"q": {"type": "string"}}, "required": ["q"], "type": "object"}}, "type": "function"}"#
        );
    }

    #[test]
    fn developer_message_uses_qwen_system_slot() {
        let messages = vec![
            ChatMessage {
                role: "developer".to_owned(),
                content: json!("Follow the integration contract."),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".to_owned(),
                content: json!("Reply with OK."),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let prompt = render_qwen_chat(&messages, &[], false, false).unwrap();
        assert!(
            prompt.starts_with("<|im_start|>system\nFollow the integration contract.<|im_end|>\n")
        );
        assert!(prompt.contains("<|im_start|>user\nReply with OK.<|im_end|>"));
    }

    #[test]
    fn named_tool_choice_requires_function_type() {
        let tools = vec![json!({
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        })];
        let invalid = json!({"type": "custom", "function": {"name": "lookup"}});
        assert!(matches!(
            select_tools(&tools, Some(&invalid)),
            Err(ChatTemplateError::InvalidToolChoice)
        ));
    }

    #[test]
    fn parses_xml_tool_call_and_reasoning() {
        let parsed = parse_assistant_output(
            "I should check.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nShanghai\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            true,
            "abc",
        );
        assert_eq!(parsed.reasoning_content.as_deref(), Some("I should check."));
        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Shanghai","days":2}"#
        );
    }

    #[test]
    fn renders_grouped_tool_responses() {
        let messages = vec![
            ChatMessage {
                role: "user".to_owned(),
                content: json!("lookup"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".to_owned(),
                content: Value::Null,
                reasoning_content: Some("".to_owned()),
                tool_calls: Some(vec![InputToolCall {
                    id: Some("call_1".to_owned()),
                    kind: "function".to_owned(),
                    function: InputFunctionCall {
                        name: "lookup".to_owned(),
                        arguments: json!({"q": "x"}),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".to_owned(),
                content: json!("one"),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some("call_1".to_owned()),
                name: None,
            },
        ];
        let prompt = render_qwen_chat(&messages, &[], false, false).unwrap();
        assert!(
            prompt.contains("<|im_start|>user\n<tool_response>\none\n</tool_response><|im_end|>")
        );
    }
}
