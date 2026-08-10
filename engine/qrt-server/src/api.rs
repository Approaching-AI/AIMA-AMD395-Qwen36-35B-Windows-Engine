use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Path as AxumPath, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore, TryAcquireError};

use crate::backend::{BackendError, GenerationResult, InferenceBackend, LoadMetrics};
use crate::chat::{
    normalize_input_tool_arguments, parse_assistant_output, render_qwen_chat, select_tools,
    ChatMessage, ParsedAssistant, SelectedTools, ToolChoiceMode,
};
use crate::tokenizer::{TokenCodec, TokenizerError};

const MAX_HTTP_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_STOP_SEQUENCES: usize = 4;
pub const DEFAULT_MAX_QUEUE_DEPTH: usize = 64;
pub const DEFAULT_QUEUE_TIMEOUT_SECONDS: u64 = 600;

#[derive(Clone, Debug)]
pub struct QueueConfig {
    pub max_waiting_requests: usize,
    pub wait_timeout: Duration,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_waiting_requests: DEFAULT_MAX_QUEUE_DEPTH,
            wait_timeout: Duration::from_secs(DEFAULT_QUEUE_TIMEOUT_SECONDS),
        }
    }
}

#[derive(Clone)]
pub struct ServerState {
    inner: Arc<ServerStateInner>,
}

struct ServerStateInner {
    backend: Arc<dyn InferenceBackend>,
    tokenizer: Arc<dyn TokenCodec>,
    model_id: String,
    max_context_tokens: usize,
    api_key: Option<String>,
    request_queue: Arc<RequestQueue>,
    request_sequence: AtomicU64,
    created: u64,
    started_at: String,
    load_metrics: LoadMetrics,
    shutdown_sender: Mutex<Option<oneshot::Sender<()>>>,
}

impl ServerState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: Arc<dyn InferenceBackend>,
        tokenizer: Arc<dyn TokenCodec>,
        model_id: String,
        max_context_tokens: usize,
        api_key: Option<String>,
        load_metrics: LoadMetrics,
        shutdown_sender: oneshot::Sender<()>,
    ) -> Self {
        Self::new_with_queue(
            backend,
            tokenizer,
            model_id,
            max_context_tokens,
            api_key,
            load_metrics,
            QueueConfig::default(),
            shutdown_sender,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_queue(
        backend: Arc<dyn InferenceBackend>,
        tokenizer: Arc<dyn TokenCodec>,
        model_id: String,
        max_context_tokens: usize,
        api_key: Option<String>,
        load_metrics: LoadMetrics,
        queue_config: QueueConfig,
        shutdown_sender: oneshot::Sender<()>,
    ) -> Self {
        let created = epoch_seconds();
        Self {
            inner: Arc::new(ServerStateInner {
                backend,
                tokenizer,
                model_id,
                max_context_tokens,
                api_key,
                request_queue: Arc::new(RequestQueue::new(queue_config)),
                request_sequence: AtomicU64::new(1),
                created,
                started_at: created.to_string(),
                load_metrics,
                shutdown_sender: Mutex::new(Some(shutdown_sender)),
            }),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    pub fn max_context_tokens(&self) -> usize {
        self.inner.max_context_tokens
    }

    pub fn begin_shutdown(&self) {
        self.inner.request_queue.stop_accepting();
    }

    fn next_id(&self, prefix: &str) -> String {
        let sequence = self.inner.request_sequence.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{:x}-{sequence:x}", epoch_nanos())
    }
}

#[derive(Debug)]
struct RequestQueue {
    gate: Arc<Semaphore>,
    accepting: AtomicBool,
    max_waiting_requests: usize,
    wait_timeout: Duration,
    active: AtomicUsize,
    waiting: AtomicUsize,
    started_total: AtomicU64,
    completed_total: AtomicU64,
    queued_total: AtomicU64,
    rejected_total: AtomicU64,
    timed_out_total: AtomicU64,
    shutdown_rejected_total: AtomicU64,
    wait_ns_total: AtomicU64,
    wait_ns_max: AtomicU64,
}

impl RequestQueue {
    fn new(config: QueueConfig) -> Self {
        Self {
            gate: Arc::new(Semaphore::new(1)),
            accepting: AtomicBool::new(true),
            max_waiting_requests: config.max_waiting_requests,
            wait_timeout: config.wait_timeout,
            active: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
            started_total: AtomicU64::new(0),
            completed_total: AtomicU64::new(0),
            queued_total: AtomicU64::new(0),
            rejected_total: AtomicU64::new(0),
            timed_out_total: AtomicU64::new(0),
            shutdown_rejected_total: AtomicU64::new(0),
            wait_ns_total: AtomicU64::new(0),
            wait_ns_max: AtomicU64::new(0),
        }
    }

    fn stop_accepting(&self) {
        self.accepting.store(false, Ordering::Release);
        self.gate.close();
    }

    fn reserve_waiter(self: &Arc<Self>) -> Option<WaitingRequest> {
        if !self.accepting.load(Ordering::Acquire) {
            self.shutdown_rejected_total.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let reserved = self
            .waiting
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.max_waiting_requests).then_some(current + 1)
            })
            .is_ok();
        if !reserved {
            self.rejected_total.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        self.queued_total.fetch_add(1, Ordering::Relaxed);
        Some(WaitingRequest {
            queue: self.clone(),
        })
    }

    fn start_request(
        self: &Arc<Self>,
        gate_permit: OwnedSemaphorePermit,
        queue_wait_ns: u64,
    ) -> RequestPermit {
        self.active.fetch_add(1, Ordering::AcqRel);
        self.started_total.fetch_add(1, Ordering::Relaxed);
        self.wait_ns_total
            .fetch_add(queue_wait_ns, Ordering::Relaxed);
        self.wait_ns_max.fetch_max(queue_wait_ns, Ordering::Relaxed);
        RequestPermit {
            _gate_permit: gate_permit,
            queue: self.clone(),
            queue_wait_ns,
        }
    }

    fn snapshot(&self) -> Value {
        let wait_ns_total = self.wait_ns_total.load(Ordering::Relaxed);
        let started_total = self.started_total.load(Ordering::Relaxed);
        json!({
            "accepting_requests": self.accepting.load(Ordering::Acquire),
            "active_requests": self.active.load(Ordering::Acquire),
            "waiting_requests": self.waiting.load(Ordering::Acquire),
            "max_waiting_requests": self.max_waiting_requests,
            "wait_timeout_seconds": self.wait_timeout.as_secs(),
            "started_total": started_total,
            "completed_total": self.completed_total.load(Ordering::Relaxed),
            "queued_total": self.queued_total.load(Ordering::Relaxed),
            "rejected_total": self.rejected_total.load(Ordering::Relaxed),
            "timed_out_total": self.timed_out_total.load(Ordering::Relaxed),
            "shutdown_rejected_total": self.shutdown_rejected_total.load(Ordering::Relaxed),
            "wait_mean_ms": if started_total == 0 {
                0.0
            } else {
                wait_ns_total as f64 / started_total as f64 / 1_000_000.0
            },
            "wait_max_ms": ns_to_ms(self.wait_ns_max.load(Ordering::Relaxed)),
        })
    }
}

#[derive(Debug)]
struct WaitingRequest {
    queue: Arc<RequestQueue>,
}

impl Drop for WaitingRequest {
    fn drop(&mut self) {
        self.queue.waiting.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug)]
struct RequestPermit {
    _gate_permit: OwnedSemaphorePermit,
    queue: Arc<RequestQueue>,
    queue_wait_ns: u64,
}

impl RequestPermit {
    fn queue_wait_ns(&self) -> u64 {
        self.queue_wait_ns
    }
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        self.queue.active.fetch_sub(1, Ordering::AcqRel);
        self.queue.completed_total.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(health))
        .route("/v1/models", get(models))
        .route("/v1/models/{model}", get(model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/admin/shutdown", post(shutdown))
        .layer(DefaultBodyLimit::max(MAX_HTTP_BODY_BYTES))
        .with_state(state)
}

async fn health(State(state): State<ServerState>) -> Json<Value> {
    let accepting_requests = state.inner.request_queue.accepting.load(Ordering::Acquire);
    Json(json!({
        "status": if accepting_requests { "ok" } else { "draining" },
        "ready": accepting_requests,
        "model": state.inner.model_id,
        "max_model_len": state.inner.max_context_tokens,
        "max_output_tokens": state.inner.backend.max_output_tokens(),
        "pid": std::process::id(),
        "started_at": state.inner.started_at,
        "capabilities": {
            "batch_size": 1,
            "continuous_prompt_lengths": true,
            "streaming": true,
            "tool_calls": true,
            "prefix_cache": true,
            "bounded_fifo_queue": true,
        },
        "queue": state.inner.request_queue.snapshot(),
        "load": load_metrics_json(&state.inner.load_metrics),
    }))
}

async fn models(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    Json(json!({
        "object": "list",
        "data": [model_json(&state)]
    }))
    .into_response()
}

async fn model(
    AxumPath(model): AxumPath<String>,
    State(state): State<ServerState>,
    headers: HeaderMap,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    if let Err(error) = validate_model(&state, &model) {
        return error.into_response();
    }
    Json(model_json(&state)).into_response()
}

fn model_json(state: &ServerState) -> Value {
    json!({
        "id": state.inner.model_id,
        "object": "model",
        "created": state.inner.created,
        "owned_by": "qrt",
        "root": state.inner.model_id,
        "parent": null,
        "max_model_len": state.inner.max_context_tokens,
        "permission": []
    })
}

async fn shutdown(State(state): State<ServerState>, headers: HeaderMap) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    state.begin_shutdown();
    let sender = state
        .inner
        .shutdown_sender
        .lock()
        .ok()
        .and_then(|mut sender| sender.take());
    match sender {
        Some(sender) => {
            let _ = sender.send(());
            Json(json!({"status": "stopping", "pid": std::process::id()})).into_response()
        }
        None => Json(json!({"status": "stopping", "pid": std::process::id()})).into_response(),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    chat_template_kwargs: ChatTemplateKwargs,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    #[serde(rename = "seed")]
    _seed: Option<i64>,
    #[serde(default, rename = "user")]
    _user: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ChatTemplateKwargs {
    #[serde(default = "default_true")]
    enable_thinking: bool,
    #[serde(default)]
    preserve_thinking: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    fn into_vec(self) -> Result<Vec<String>, ApiError> {
        let values = match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        };
        if values.len() > MAX_STOP_SEQUENCES {
            return Err(ApiError::invalid(
                format!("stop supports at most {MAX_STOP_SEQUENCES} sequences"),
                Some("stop"),
                Some("invalid_stop"),
            ));
        }
        if values.iter().any(String::is_empty) {
            return Err(ApiError::invalid(
                "stop sequences must not be empty",
                Some("stop"),
                Some("invalid_stop"),
            ));
        }
        Ok(values)
    }
}

async fn chat_completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    let Json(mut request) = match payload {
        Ok(payload) => payload,
        Err(error) => return ApiError::json(error).into_response(),
    };
    if let Err(error) = validate_model(&state, &request.model) {
        return error.into_response();
    }
    if let Err(error) = validate_sampling(
        request.temperature,
        request.top_p,
        request.n,
        request.presence_penalty,
        request.frequency_penalty,
    ) {
        return error.into_response();
    }
    if request.logprobs.unwrap_or(false) || request.top_logprobs.unwrap_or(0) != 0 {
        return ApiError::unsupported(
            "logprobs are not yet exposed by the native token ABI",
            "logprobs",
        )
        .into_response();
    }
    if let Some(format) = &request.response_format {
        if format.get("type").and_then(Value::as_str) != Some("text") {
            return ApiError::unsupported(
                "only response_format.type=text is supported",
                "response_format",
            )
            .into_response();
        }
    }
    if let Err(error) = normalize_input_tool_arguments(&mut request.messages) {
        return ApiError::invalid(
            error.to_string(),
            Some("messages"),
            Some("invalid_messages"),
        )
        .into_response();
    }
    let selected_tools = match select_tools(&request.tools, request.tool_choice.as_ref()) {
        Ok(tools) => tools,
        Err(error) => {
            return ApiError::invalid(
                error.to_string(),
                Some("tool_choice"),
                Some("invalid_tools"),
            )
            .into_response()
        }
    };
    let prompt = match render_qwen_chat(
        &request.messages,
        &selected_tools.tools,
        request.chat_template_kwargs.enable_thinking,
        request.chat_template_kwargs.preserve_thinking,
    ) {
        Ok(prompt) => prompt,
        Err(error) => {
            return ApiError::invalid(
                error.to_string(),
                Some("messages"),
                Some("invalid_messages"),
            )
            .into_response()
        }
    };
    let prompt_token_ids = match state.inner.tokenizer.encode(&prompt) {
        Ok(tokens) => tokens,
        Err(error) => return tokenizer_error(error).into_response(),
    };
    let max_output_tokens = match resolve_max_output(
        &state,
        request.max_completion_tokens.or(request.max_tokens),
        256,
        prompt_token_ids.len(),
    ) {
        Ok(tokens) => tokens,
        Err(error) => return error.into_response(),
    };
    let stops = match request.stop.take().map(StopInput::into_vec).transpose() {
        Ok(stops) => stops.unwrap_or_default(),
        Err(error) => return error.into_response(),
    };
    let request_id = state.next_id("chatcmpl");
    let allow_parallel_tools = request.parallel_tool_calls.unwrap_or(true);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    if request.stream {
        return chat_stream_response(
            state,
            request_id,
            prompt_token_ids,
            max_output_tokens,
            stops,
            request.ignore_eos,
            request.chat_template_kwargs.enable_thinking,
            selected_tools,
            allow_parallel_tools,
            include_usage,
        )
        .await;
    }

    match run_generation(
        &state,
        prompt_token_ids.clone(),
        max_output_tokens,
        None,
        false,
    )
    .await
    {
        Ok(result) => {
            let finalized = match finalize_tokens(
                state.inner.tokenizer.as_ref(),
                &result.token_ids,
                &stops,
                request.ignore_eos,
                max_output_tokens,
            ) {
                Ok(output) => output,
                Err(error) => return error.into_response(),
            };
            let parsed = parse_assistant_output(
                &finalized.text,
                request.chat_template_kwargs.enable_thinking,
                &request_id,
            );
            if let Err(error) = enforce_tool_choice(&selected_tools, &parsed, allow_parallel_tools)
            {
                return error.into_response();
            }
            let finish_reason = if parsed.tool_calls.is_empty() {
                finalized.finish_reason
            } else {
                "tool_calls"
            };
            let usage = usage_json(prompt_token_ids.len(), finalized.generated_token_count);
            let metrics = request_metrics_json(&result);
            let body = json!({
                "id": request_id,
                "object": "chat.completion",
                "created": epoch_seconds(),
                "model": state.inner.model_id,
                "system_fingerprint": system_fingerprint(),
                "choices": [{
                    "index": 0,
                    "message": assistant_message_json(&parsed),
                    "logprobs": null,
                    "finish_reason": finish_reason,
                    "stop_reason": finalized.stop_reason,
                }],
                "usage": usage,
                "qrt_metrics": metrics,
            });
            response_with_request_metadata(
                Json(body).into_response(),
                &request_id,
                result.metrics.queue_wait_ns,
            )
        }
        Err(error) => response_with_request_id(error.into_response(), &request_id),
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: CompletionPrompt,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    best_of: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<StopInput>,
    #[serde(default)]
    echo: bool,
    #[serde(default)]
    logprobs: Option<usize>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    logit_bias: Option<Value>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    #[serde(rename = "seed")]
    _seed: Option<i64>,
    #[serde(default, rename = "user")]
    _user: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum CompletionPrompt {
    Text(String),
    Tokens(Vec<u32>),
    TextBatch(Vec<String>),
    TokenBatch(Vec<Vec<u32>>),
}

async fn completions(
    State(state): State<ServerState>,
    headers: HeaderMap,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    let Json(mut request) = match payload {
        Ok(payload) => payload,
        Err(error) => return ApiError::json(error).into_response(),
    };
    if let Err(error) = validate_model(&state, &request.model) {
        return error.into_response();
    }
    if let Err(error) = validate_sampling(
        request.temperature,
        request.top_p,
        request.n,
        request.presence_penalty,
        request.frequency_penalty,
    ) {
        return error.into_response();
    }
    if request.best_of.unwrap_or(1) != 1 {
        return ApiError::unsupported("best_of values other than 1 are unsupported", "best_of")
            .into_response();
    }
    if request.logprobs.unwrap_or(0) != 0 {
        return ApiError::unsupported(
            "logprobs are not yet exposed by the native token ABI",
            "logprobs",
        )
        .into_response();
    }
    if request.suffix.is_some() {
        return ApiError::unsupported("suffix insertion is unsupported", "suffix").into_response();
    }
    if request
        .logit_bias
        .as_ref()
        .is_some_and(|bias| bias.as_object().is_none_or(|values| !values.is_empty()))
    {
        return ApiError::unsupported("logit_bias is unsupported", "logit_bias").into_response();
    }
    let raw_token_prompt = matches!(&request.prompt, CompletionPrompt::Tokens(_));
    let (prompt_token_ids, prompt_text) =
        match completion_prompt(state.inner.tokenizer.as_ref(), request.prompt) {
            Ok(prompt) => prompt,
            Err(error) => return error.into_response(),
        };
    let max_output_tokens =
        match resolve_max_output(&state, request.max_tokens, 16, prompt_token_ids.len()) {
            Ok(tokens) => tokens,
            Err(error) => return error.into_response(),
        };
    let exact_first_token = raw_token_prompt && max_output_tokens == 1;
    let stops = match request.stop.take().map(StopInput::into_vec).transpose() {
        Ok(stops) => stops.unwrap_or_default(),
        Err(error) => return error.into_response(),
    };
    let request_id = state.next_id("cmpl");
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    if request.stream {
        return completion_stream_response(
            state,
            request_id,
            prompt_token_ids,
            prompt_text,
            max_output_tokens,
            stops,
            request.ignore_eos,
            request.echo,
            include_usage,
            exact_first_token,
        )
        .await;
    }

    match run_generation(
        &state,
        prompt_token_ids.clone(),
        max_output_tokens,
        None,
        exact_first_token,
    )
    .await
    {
        Ok(result) => {
            let finalized = match finalize_tokens(
                state.inner.tokenizer.as_ref(),
                &result.token_ids,
                &stops,
                request.ignore_eos,
                max_output_tokens,
            ) {
                Ok(output) => output,
                Err(error) => return error.into_response(),
            };
            let text = if request.echo {
                format!("{prompt_text}{}", finalized.text)
            } else {
                finalized.text.clone()
            };
            let body = json!({
                "id": request_id,
                "object": "text_completion",
                "created": epoch_seconds(),
                "model": state.inner.model_id,
                "system_fingerprint": system_fingerprint(),
                "choices": [{
                    "index": 0,
                    "text": text,
                    "logprobs": null,
                    "finish_reason": finalized.finish_reason,
                    "stop_reason": finalized.stop_reason,
                    "token_ids": finalized.generated_token_ids,
                }],
                "usage": usage_json(prompt_token_ids.len(), finalized.generated_token_count),
                "qrt_metrics": request_metrics_json(&result),
            });
            response_with_request_metadata(
                Json(body).into_response(),
                &request_id,
                result.metrics.queue_wait_ns,
            )
        }
        Err(error) => response_with_request_id(error.into_response(), &request_id),
    }
}

#[derive(Debug, Deserialize)]
struct TokenizeRequest {
    model: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    chat_template_kwargs: ChatTemplateKwargs,
    #[serde(default)]
    add_special_tokens: bool,
    #[serde(default)]
    return_token_strs: bool,
}

async fn tokenize(
    State(state): State<ServerState>,
    headers: HeaderMap,
    payload: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    let Json(mut request) = match payload {
        Ok(payload) => payload,
        Err(error) => return ApiError::json(error).into_response(),
    };
    if let Err(error) = validate_model(&state, &request.model) {
        return error.into_response();
    }
    let prompt = match (request.prompt.take(), request.messages.as_mut()) {
        (Some(prompt), None) => prompt,
        (None, Some(messages)) => {
            if let Err(error) = normalize_input_tool_arguments(messages) {
                return ApiError::invalid(
                    error.to_string(),
                    Some("messages"),
                    Some("invalid_messages"),
                )
                .into_response();
            }
            match render_qwen_chat(
                messages,
                &request.tools,
                request.chat_template_kwargs.enable_thinking,
                request.chat_template_kwargs.preserve_thinking,
            ) {
                Ok(prompt) => prompt,
                Err(error) => {
                    return ApiError::invalid(
                        error.to_string(),
                        Some("messages"),
                        Some("invalid_messages"),
                    )
                    .into_response()
                }
            }
        }
        _ => {
            return ApiError::invalid(
                "provide exactly one of prompt or messages",
                Some("prompt"),
                Some("invalid_request"),
            )
            .into_response()
        }
    };
    let tokens = match state.inner.tokenizer.encode(&prompt) {
        Ok(tokens) => tokens,
        Err(error) => return tokenizer_error(error).into_response(),
    };
    if tokens.len() > state.inner.max_context_tokens {
        return context_error(tokens.len(), 0, state.inner.max_context_tokens).into_response();
    }
    let token_strs = request.return_token_strs.then(|| {
        tokens
            .iter()
            .map(|token| state.inner.tokenizer.token_string(*token))
            .collect::<Vec<_>>()
    });
    let _ = request.add_special_tokens;
    Json(json!({
        "count": tokens.len(),
        "max_model_len": state.inner.max_context_tokens,
        "tokens": tokens,
        "token_strs": token_strs,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
struct DetokenizeRequest {
    model: String,
    tokens: Vec<u32>,
    #[serde(default)]
    skip_special_tokens: bool,
}

async fn detokenize(
    State(state): State<ServerState>,
    headers: HeaderMap,
    payload: Result<Json<DetokenizeRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = authorize(&state, &headers) {
        return error.into_response();
    }
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return ApiError::json(error).into_response(),
    };
    if let Err(error) = validate_model(&state, &request.model) {
        return error.into_response();
    }
    if request.tokens.len() > state.inner.max_context_tokens {
        return context_error(request.tokens.len(), 0, state.inner.max_context_tokens)
            .into_response();
    }
    if let Some(token) = request
        .tokens
        .iter()
        .find(|token| **token as usize >= state.inner.tokenizer.vocab_size())
    {
        return ApiError::invalid(
            format!("token id {token} is outside the vocabulary"),
            Some("tokens"),
            Some("invalid_token"),
        )
        .into_response();
    }
    match state
        .inner
        .tokenizer
        .decode(&request.tokens, request.skip_special_tokens)
    {
        Ok(prompt) => Json(json!({"prompt": prompt})).into_response(),
        Err(error) => tokenizer_error(error).into_response(),
    }
}

#[allow(clippy::too_many_arguments)]
async fn chat_stream_response(
    state: ServerState,
    request_id: String,
    prompt_token_ids: Vec<u32>,
    max_output_tokens: usize,
    stops: Vec<String>,
    ignore_eos: bool,
    thinking_enabled: bool,
    selected_tools: SelectedTools,
    allow_parallel_tools: bool,
    include_usage: bool,
) -> Response {
    let permit = match acquire_request_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => {
            return response_with_request_id(error.into_response(), &request_id);
        }
    };
    let queue_wait_ns = permit.queue_wait_ns();
    let prompt_tokens = prompt_token_ids.len();
    let model = state.inner.model_id.clone();
    let codec = state.inner.tokenizer.clone();
    let (events_tx, mut events_rx) = mpsc::unbounded_channel();
    let handle = spawn_generation(
        state.inner.backend.clone(),
        prompt_token_ids,
        max_output_tokens,
        Some(events_tx),
        permit,
        false,
    );
    let stream_id = request_id.clone();
    let direct_content_stream = selected_tools.tools.is_empty() && !thinking_enabled;
    let output = stream! {
        yield Ok::<Event, Infallible>(sse_json(chat_chunk(
            &stream_id,
            &model,
            json!({"role": "assistant", "content": ""}),
            Value::Null,
        )));
        let mut observed_ids = Vec::new();
        let mut text_stream = TextStreamState::new(stops.clone());
        let mut eos_observed = false;
        while let Some(event) = events_rx.recv().await {
            observed_ids.push(event.token_id);
            if !ignore_eos && codec.is_eos(event.token_id) {
                eos_observed = true;
                continue;
            }
            if direct_content_stream && !text_stream.stopped && !eos_observed {
                if let Ok(full) = codec.decode(&observed_ids, true) {
                    if let Some(delta) = text_stream.update(&full, false) {
                        yield Ok(sse_json(chat_chunk(
                            &stream_id,
                            &model,
                            json!({"content": delta}),
                            Value::Null,
                        )));
                    }
                }
            }
        }
        match handle.await {
            Ok(Ok(result)) => match finalize_tokens(codec.as_ref(), &result.token_ids, &stops, ignore_eos, max_output_tokens) {
                Ok(finalized) => {
                    let mut finish_reason = finalized.finish_reason;
                    if direct_content_stream {
                        if let Some(delta) = text_stream.update(&finalized.text, true) {
                            yield Ok(sse_json(chat_chunk(
                                &stream_id,
                                &model,
                                json!({"content": delta}),
                                Value::Null,
                            )));
                        }
                    } else {
                        let parsed = parse_assistant_output(&finalized.text, thinking_enabled, &stream_id);
                        if let Err(error) = enforce_tool_choice(
                            &selected_tools,
                            &parsed,
                            allow_parallel_tools,
                        ) {
                            yield Ok(sse_json(error.body()));
                            yield Ok(Event::default().data("[DONE]"));
                            return;
                        }
                        if let Some(reasoning) = parsed.reasoning_content {
                            yield Ok(sse_json(chat_chunk(
                                &stream_id,
                                &model,
                                json!({"reasoning_content": reasoning}),
                                Value::Null,
                            )));
                        }
                        if let Some(content) = parsed.content {
                            yield Ok(sse_json(chat_chunk(
                                &stream_id,
                                &model,
                                json!({"content": content}),
                                Value::Null,
                            )));
                        }
                        if !parsed.tool_calls.is_empty() {
                            finish_reason = "tool_calls";
                            for (index, tool_call) in parsed.tool_calls.into_iter().enumerate() {
                                yield Ok(sse_json(chat_chunk(
                                    &stream_id,
                                    &model,
                                    json!({"tool_calls": [{
                                        "index": index,
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        }
                                    }]}),
                                    Value::Null,
                                )));
                            }
                        }
                    }
                    yield Ok(sse_json(chat_chunk(
                        &stream_id,
                        &model,
                        json!({}),
                        json!(finish_reason),
                    )));
                    if include_usage {
                        yield Ok(sse_json(json!({
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": epoch_seconds(),
                            "model": model,
                            "system_fingerprint": system_fingerprint(),
                            "choices": [],
                            "usage": usage_json(prompt_tokens, finalized.generated_token_count),
                            "qrt_metrics": request_metrics_json(&result),
                        })));
                    }
                }
                Err(error) => yield Ok(sse_json(error.body())),
            },
            Ok(Err(error)) => yield Ok(sse_json(error.body())),
            Err(error) => yield Ok(sse_json(ApiError::internal(format!("inference worker failed: {error}")).body())),
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    let response = Sse::new(output)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response();
    response_with_request_metadata(response, &request_id, queue_wait_ns)
}

#[allow(clippy::too_many_arguments)]
async fn completion_stream_response(
    state: ServerState,
    request_id: String,
    prompt_token_ids: Vec<u32>,
    prompt_text: String,
    max_output_tokens: usize,
    stops: Vec<String>,
    ignore_eos: bool,
    echo: bool,
    include_usage: bool,
    exact_first_token: bool,
) -> Response {
    let permit = match acquire_request_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => {
            return response_with_request_id(error.into_response(), &request_id);
        }
    };
    let queue_wait_ns = permit.queue_wait_ns();
    let prompt_tokens = prompt_token_ids.len();
    let model = state.inner.model_id.clone();
    let codec = state.inner.tokenizer.clone();
    let (events_tx, mut events_rx) = mpsc::unbounded_channel();
    let handle = spawn_generation(
        state.inner.backend.clone(),
        prompt_token_ids,
        max_output_tokens,
        Some(events_tx),
        permit,
        exact_first_token,
    );
    let stream_id = request_id.clone();
    let output = stream! {
        if echo && !prompt_text.is_empty() {
            yield Ok::<Event, Infallible>(sse_json(completion_chunk(
                &stream_id,
                &model,
                &prompt_text,
                Value::Null,
            )));
        }
        let mut observed_ids = Vec::new();
        let mut text_stream = TextStreamState::new(stops.clone());
        let mut eos_observed = false;
        while let Some(event) = events_rx.recv().await {
            observed_ids.push(event.token_id);
            if !ignore_eos && codec.is_eos(event.token_id) {
                eos_observed = true;
                continue;
            }
            if !text_stream.stopped && !eos_observed {
                if let Ok(full) = codec.decode(&observed_ids, true) {
                    if let Some(delta) = text_stream.update(&full, false) {
                        yield Ok(sse_json(completion_chunk(
                            &stream_id,
                            &model,
                            &delta,
                            Value::Null,
                        )));
                    }
                }
            }
        }
        match handle.await {
            Ok(Ok(result)) => match finalize_tokens(codec.as_ref(), &result.token_ids, &stops, ignore_eos, max_output_tokens) {
                Ok(finalized) => {
                    if let Some(delta) = text_stream.update(&finalized.text, true) {
                        yield Ok(sse_json(completion_chunk(
                            &stream_id,
                            &model,
                            &delta,
                            Value::Null,
                        )));
                    }
                    yield Ok(sse_json(completion_chunk(
                        &stream_id,
                        &model,
                        "",
                        json!(finalized.finish_reason),
                    )));
                    if include_usage {
                        yield Ok(sse_json(json!({
                            "id": stream_id,
                            "object": "text_completion",
                            "created": epoch_seconds(),
                            "model": model,
                            "system_fingerprint": system_fingerprint(),
                            "choices": [],
                            "usage": usage_json(prompt_tokens, finalized.generated_token_count),
                            "qrt_metrics": request_metrics_json(&result),
                        })));
                    }
                }
                Err(error) => yield Ok(sse_json(error.body())),
            },
            Ok(Err(error)) => yield Ok(sse_json(error.body())),
            Err(error) => yield Ok(sse_json(ApiError::internal(format!("inference worker failed: {error}")).body())),
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    let response = Sse::new(output)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response();
    response_with_request_metadata(response, &request_id, queue_wait_ns)
}

fn chat_chunk(id: &str, model: &str, delta: Value, finish_reason: Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": epoch_seconds(),
        "model": model,
        "system_fingerprint": system_fingerprint(),
        "usage": null,
        "choices": [{
            "index": 0,
            "delta": delta,
            "logprobs": null,
            "finish_reason": finish_reason,
        }]
    })
}

fn completion_chunk(id: &str, model: &str, text: &str, finish_reason: Value) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": epoch_seconds(),
        "model": model,
        "system_fingerprint": system_fingerprint(),
        "usage": null,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": finish_reason,
        }]
    })
}

fn sse_json(value: Value) -> Event {
    Event::default()
        .data(serde_json::to_string(&value).expect("serde_json::Value serialization cannot fail"))
}

async fn run_generation(
    state: &ServerState,
    input_tokens: Vec<u32>,
    max_output_tokens: usize,
    events: Option<mpsc::UnboundedSender<crate::backend::BackendToken>>,
    exact_first_token: bool,
) -> Result<GenerationResult, ApiError> {
    let permit = acquire_request_permit(state).await?;
    let queue_wait_ns = permit.queue_wait_ns();
    let mut result = spawn_generation(
        state.inner.backend.clone(),
        input_tokens,
        max_output_tokens,
        events,
        permit,
        exact_first_token,
    )
    .await
    .map_err(|error| ApiError::internal(format!("inference worker failed: {error}")))??;
    result.metrics.queue_wait_ns = queue_wait_ns;
    Ok(result)
}

fn spawn_generation(
    backend: Arc<dyn InferenceBackend>,
    input_tokens: Vec<u32>,
    max_output_tokens: usize,
    events: Option<mpsc::UnboundedSender<crate::backend::BackendToken>>,
    permit: RequestPermit,
    exact_first_token: bool,
) -> tokio::task::JoinHandle<Result<GenerationResult, ApiError>> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let result = if exact_first_token {
            backend.generate_exact_first_token(&input_tokens, max_output_tokens, events)
        } else {
            backend.generate(&input_tokens, max_output_tokens, events)
        };
        result.map_err(ApiError::backend)
    })
}

async fn acquire_request_permit(state: &ServerState) -> Result<RequestPermit, ApiError> {
    let queue = state.inner.request_queue.clone();
    if !queue.accepting.load(Ordering::Acquire) {
        queue
            .shutdown_rejected_total
            .fetch_add(1, Ordering::Relaxed);
        return Err(ApiError::queue_unavailable());
    }
    match queue.gate.clone().try_acquire_owned() {
        Ok(permit) => {
            if !queue.accepting.load(Ordering::Acquire) {
                drop(permit);
                queue
                    .shutdown_rejected_total
                    .fetch_add(1, Ordering::Relaxed);
                return Err(ApiError::queue_unavailable());
            }
            return Ok(queue.start_request(permit, 0));
        }
        Err(TryAcquireError::Closed) => {
            queue
                .shutdown_rejected_total
                .fetch_add(1, Ordering::Relaxed);
            return Err(ApiError::queue_unavailable());
        }
        Err(TryAcquireError::NoPermits) => {}
    }

    let Some(waiting) = queue.reserve_waiter() else {
        if queue.accepting.load(Ordering::Acquire) {
            return Err(ApiError::queue_full(queue.max_waiting_requests));
        }
        return Err(ApiError::queue_unavailable());
    };
    let wait_started = Instant::now();
    let acquired =
        tokio::time::timeout(queue.wait_timeout, queue.gate.clone().acquire_owned()).await;
    drop(waiting);
    match acquired {
        Ok(Ok(permit)) => {
            let queue_wait_ns = duration_ns(wait_started.elapsed());
            Ok(queue.start_request(permit, queue_wait_ns))
        }
        Ok(Err(_)) => {
            queue
                .shutdown_rejected_total
                .fetch_add(1, Ordering::Relaxed);
            Err(ApiError::queue_unavailable())
        }
        Err(_) => {
            queue.timed_out_total.fetch_add(1, Ordering::Relaxed);
            Err(ApiError::queue_timeout(queue.wait_timeout))
        }
    }
}

struct FinalizedOutput {
    text: String,
    generated_token_ids: Vec<u32>,
    generated_token_count: usize,
    finish_reason: &'static str,
    stop_reason: Option<String>,
}

fn finalize_tokens(
    codec: &dyn TokenCodec,
    generated: &[u32],
    stops: &[String],
    ignore_eos: bool,
    requested_tokens: usize,
) -> Result<FinalizedOutput, ApiError> {
    let mut decode_count = generated.len();
    let mut generated_count = generated.len();
    let mut finish_reason = if generated.len() >= requested_tokens {
        "length"
    } else {
        "stop"
    };
    let mut stop_reason = None;
    if !ignore_eos {
        if let Some(index) = generated.iter().position(|token| codec.is_eos(*token)) {
            decode_count = index;
            generated_count = index + 1;
            finish_reason = "stop";
        }
    }
    let mut text = codec
        .decode(&generated[..decode_count], true)
        .map_err(tokenizer_error)?;
    if let Some((position, stop)) = earliest_stop(&text, stops) {
        text.truncate(position);
        finish_reason = "stop";
        stop_reason = Some(stop.to_owned());
        for count in 1..=decode_count {
            let prefix = codec
                .decode(&generated[..count], true)
                .map_err(tokenizer_error)?;
            if prefix.contains(stop) {
                generated_count = count;
                break;
            }
        }
    }
    Ok(FinalizedOutput {
        text,
        generated_token_ids: generated[..generated_count].to_vec(),
        generated_token_count: generated_count,
        finish_reason,
        stop_reason,
    })
}

struct TextStreamState {
    emitted: String,
    stops: Vec<String>,
    stopped: bool,
}

impl TextStreamState {
    fn new(stops: Vec<String>) -> Self {
        Self {
            emitted: String::new(),
            stops,
            stopped: false,
        }
    }

    fn update(&mut self, full: &str, final_update: bool) -> Option<String> {
        if self.stopped {
            return None;
        }
        let mut target_end = full.len();
        if let Some((position, _)) = earliest_stop(full, &self.stops) {
            target_end = position;
            self.stopped = true;
        } else if !final_update {
            let hold_bytes = self
                .stops
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(1)
                .saturating_sub(1)
                .max(if full.ends_with('\u{fffd}') { 3 } else { 0 });
            target_end = target_end.saturating_sub(hold_bytes);
            while target_end > 0 && !full.is_char_boundary(target_end) {
                target_end -= 1;
            }
        }
        let target = &full[..target_end];
        if !target.starts_with(&self.emitted) {
            return None;
        }
        let delta = target[self.emitted.len()..].to_owned();
        self.emitted.push_str(&delta);
        (!delta.is_empty()).then_some(delta)
    }
}

fn earliest_stop<'a>(text: &str, stops: &'a [String]) -> Option<(usize, &'a str)> {
    stops
        .iter()
        .filter_map(|stop| text.find(stop).map(|position| (position, stop.as_str())))
        .min_by_key(|(position, _)| *position)
}

fn assistant_message_json(parsed: &ParsedAssistant) -> Value {
    let mut message = serde_json::Map::new();
    message.insert("role".to_owned(), json!("assistant"));
    message.insert("content".to_owned(), json!(parsed.content));
    if let Some(reasoning) = &parsed.reasoning_content {
        message.insert("reasoning_content".to_owned(), json!(reasoning));
    }
    if !parsed.tool_calls.is_empty() {
        message.insert("tool_calls".to_owned(), json!(parsed.tool_calls));
    }
    Value::Object(message)
}

fn enforce_tool_choice(
    selected_tools: &SelectedTools,
    parsed: &ParsedAssistant,
    allow_parallel_tools: bool,
) -> Result<(), ApiError> {
    if !allow_parallel_tools && parsed.tool_calls.len() > 1 {
        return Err(ApiError::internal(
            "the model produced multiple calls while parallel_tool_calls=false",
        ));
    }
    if selected_tools.choice == ToolChoiceMode::Required && parsed.tool_calls.is_empty() {
        return Err(ApiError::internal(
            "the model did not produce a tool call for tool_choice=required",
        ));
    }
    if let Some(required_name) = &selected_tools.required_name {
        if parsed.tool_calls.is_empty()
            || parsed
                .tool_calls
                .iter()
                .any(|call| call.function.name != *required_name)
        {
            return Err(ApiError::internal(format!(
                "the model did not honor the required tool {required_name}"
            )));
        }
    }
    Ok(())
}

fn completion_prompt(
    codec: &dyn TokenCodec,
    prompt: CompletionPrompt,
) -> Result<(Vec<u32>, String), ApiError> {
    match prompt {
        CompletionPrompt::Text(text) => {
            let tokens = codec.encode(&text).map_err(tokenizer_error)?;
            Ok((tokens, text))
        }
        CompletionPrompt::Tokens(tokens) => {
            let text = codec.decode(&tokens, false).map_err(tokenizer_error)?;
            Ok((tokens, text))
        }
        CompletionPrompt::TextBatch(mut prompts) if prompts.len() == 1 => {
            let text = prompts.remove(0);
            let tokens = codec.encode(&text).map_err(tokenizer_error)?;
            Ok((tokens, text))
        }
        CompletionPrompt::TokenBatch(mut prompts) if prompts.len() == 1 => {
            let tokens = prompts.remove(0);
            let text = codec.decode(&tokens, false).map_err(tokenizer_error)?;
            Ok((tokens, text))
        }
        CompletionPrompt::TextBatch(_) | CompletionPrompt::TokenBatch(_) => Err(ApiError::invalid(
            "batch size 1 is required",
            Some("prompt"),
            Some("unsupported_batch_size"),
        )),
    }
}

fn validate_model(state: &ServerState, requested: &str) -> Result<(), ApiError> {
    if requested == state.inner.model_id {
        Ok(())
    } else {
        Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("The model `{requested}` does not exist"),
            "invalid_request_error",
            Some("model"),
            Some("model_not_found"),
        ))
    }
}

fn validate_sampling(
    temperature: Option<f64>,
    top_p: Option<f64>,
    n: Option<usize>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
) -> Result<(), ApiError> {
    if temperature.is_some_and(|value| value != 0.0) {
        return Err(ApiError::unsupported(
            "the native engine currently supports greedy decoding only; use temperature=0",
            "temperature",
        ));
    }
    if top_p.is_some_and(|value| value != 1.0) {
        return Err(ApiError::unsupported(
            "the native engine currently supports top_p=1 only",
            "top_p",
        ));
    }
    if n.unwrap_or(1) != 1 {
        return Err(ApiError::unsupported(
            "n values other than 1 are unsupported",
            "n",
        ));
    }
    if presence_penalty.unwrap_or(0.0) != 0.0 {
        return Err(ApiError::unsupported(
            "presence_penalty is unsupported",
            "presence_penalty",
        ));
    }
    if frequency_penalty.unwrap_or(0.0) != 0.0 {
        return Err(ApiError::unsupported(
            "frequency_penalty is unsupported",
            "frequency_penalty",
        ));
    }
    Ok(())
}

fn resolve_max_output(
    state: &ServerState,
    requested: Option<usize>,
    default: usize,
    prompt_tokens: usize,
) -> Result<usize, ApiError> {
    let output_tokens = requested.unwrap_or(default);
    if output_tokens == 0 {
        return Err(ApiError::invalid(
            "max_tokens must be at least 1",
            Some("max_tokens"),
            Some("invalid_max_tokens"),
        ));
    }
    if output_tokens > state.inner.backend.max_output_tokens() {
        return Err(ApiError::invalid(
            format!(
                "max_tokens={} exceeds the native output limit {}",
                output_tokens,
                state.inner.backend.max_output_tokens()
            ),
            Some("max_tokens"),
            Some("max_tokens_exceeded"),
        ));
    }
    if prompt_tokens == 0 {
        return Err(ApiError::invalid(
            "the prompt tokenized to an empty sequence",
            Some("prompt"),
            Some("empty_prompt"),
        ));
    }
    if prompt_tokens.saturating_add(output_tokens) > state.inner.max_context_tokens {
        return Err(context_error(
            prompt_tokens,
            output_tokens,
            state.inner.max_context_tokens,
        ));
    }
    Ok(output_tokens)
}

fn context_error(prompt: usize, output: usize, maximum: usize) -> ApiError {
    ApiError::invalid(
        format!(
            "This model's maximum context length is {maximum} tokens. The request has {prompt} prompt tokens and requests {output} completion tokens."
        ),
        Some("messages"),
        Some("context_length_exceeded"),
    )
}

fn authorize(state: &ServerState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(expected) = &state.inner.api_key else {
        return Ok(());
    };
    let provided = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if provided
        .is_some_and(|provided| constant_time_equal(provided.as_bytes(), expected.as_bytes()))
    {
        Ok(())
    } else {
        Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "Incorrect API key provided",
            "invalid_request_error",
            None,
            Some("invalid_api_key"),
        ))
    }
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

fn usage_json(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn request_metrics_json(result: &GenerationResult) -> Value {
    json!({
        "queue_wait_ms": ns_to_ms(result.metrics.queue_wait_ns),
        "request_ms": ns_to_ms(result.metrics.wall_ns),
        "ttft_ms": ns_to_ms(result.metrics.ttft_ns),
        "tpot_ms": ns_to_ms(result.metrics.tpot_ns),
        "tpot_samples": result.metrics.tpot_samples,
    })
}

fn load_metrics_json(metrics: &LoadMetrics) -> Value {
    json!({
        "provider_dll_ms": ns_to_ms(metrics.provider_dll_load_ns),
        "provider_preload_wall_ms": ns_to_ms(metrics.provider_preload_wall_ns),
        "provider_preload_reported_ms": ns_to_ms(metrics.provider_preload_reported_ns),
        "provider_preload_stored_entries": metrics.provider_preload_stored_entries,
        "engine_create_ms": ns_to_ms(metrics.engine_create_ns),
        "total_ms": ns_to_ms(metrics.total_load_ns),
    })
}

fn ns_to_ms(value: u64) -> f64 {
    value as f64 / 1_000_000.0
}

fn system_fingerprint() -> &'static str {
    "fp_qrt_native_qwen36"
}

fn response_with_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

fn response_with_request_metadata(
    mut response: Response,
    request_id: &str,
    queue_wait_ns: u64,
) -> Response {
    response = response_with_request_id(response, request_id);
    if let Ok(value) = HeaderValue::from_str(&format!("{:.3}", ns_to_ms(queue_wait_ns))) {
        response.headers_mut().insert("x-qrt-queue-wait-ms", value);
    }
    response
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn default_true() -> bool {
    true
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn epoch_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn tokenizer_error(error: TokenizerError) -> ApiError {
    ApiError::internal(error.to_string())
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
    param: Option<&'static str>,
    code: Option<&'static str>,
    retry_after_seconds: Option<u64>,
}

impl ApiError {
    fn new(
        status: StatusCode,
        message: impl Into<String>,
        kind: &'static str,
        param: Option<&'static str>,
        code: Option<&'static str>,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            kind,
            param,
            code,
            retry_after_seconds: None,
        }
    }

    fn with_retry_after(mut self, seconds: u64) -> Self {
        self.retry_after_seconds = Some(seconds.max(1));
        self
    }

    fn queue_full(max_waiting_requests: usize) -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "The inference queue is full (maximum {max_waiting_requests} waiting requests)."
            ),
            "server_error",
            None,
            Some("queue_full"),
        )
        .with_retry_after(1)
    }

    fn queue_timeout(timeout: Duration) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "The request exceeded the inference queue wait limit of {} seconds.",
                timeout.as_secs()
            ),
            "server_error",
            None,
            Some("queue_timeout"),
        )
        .with_retry_after(1)
    }

    fn queue_unavailable() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "The inference service is shutting down and is not accepting requests.",
            "server_error",
            None,
            Some("service_shutting_down"),
        )
        .with_retry_after(1)
    }

    fn invalid(
        message: impl Into<String>,
        param: Option<&'static str>,
        code: Option<&'static str>,
    ) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            param,
            code,
        )
    }

    fn unsupported(message: impl Into<String>, param: &'static str) -> Self {
        Self::invalid(message, Some(param), Some("unsupported_parameter"))
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
            "server_error",
            None,
            Some("engine_error"),
        )
    }

    fn backend(error: BackendError) -> Self {
        Self::internal(error.to_string())
    }

    fn json(error: JsonRejection) -> Self {
        Self::invalid(
            format!("Invalid JSON request: {error}"),
            None,
            Some("invalid_json"),
        )
    }

    fn body(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let retry_after_seconds = self.retry_after_seconds;
        let mut response = (self.status, Json(self.body())).into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                axum::http::header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer"),
            );
        }
        if let Some(seconds) = retry_after_seconds {
            if let Ok(value) = HeaderValue::from_str(&seconds.to_string()) {
                response
                    .headers_mut()
                    .insert(axum::http::header::RETRY_AFTER, value);
            }
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ScriptedBackend;
    use crate::tokenizer::ByteCodec;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_app_with_options(
        output: &str,
        max_context_tokens: usize,
        api_key: Option<&str>,
    ) -> (Router, oneshot::Receiver<()>) {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let app = router(ServerState::new(
            Arc::new(ScriptedBackend::new(
                output.bytes().map(u32::from).collect(),
            )),
            Arc::new(ByteCodec),
            "test-model".to_owned(),
            max_context_tokens,
            api_key.map(str::to_owned),
            LoadMetrics::default(),
            shutdown_tx,
        ));
        (app, shutdown_rx)
    }

    fn test_app_with_context(output: &str, max_context_tokens: usize) -> Router {
        test_app_with_options(output, max_context_tokens, None).0
    }

    fn test_app(output: &str) -> Router {
        test_app_with_context(output, 4096)
    }

    fn test_state_with_queue(max_waiting_requests: usize, wait_timeout: Duration) -> ServerState {
        let (shutdown_tx, _shutdown_rx) = oneshot::channel();
        ServerState::new_with_queue(
            Arc::new(ScriptedBackend::new(
                b"OK".iter().copied().map(u32::from).collect(),
            )),
            Arc::new(ByteCodec),
            "test-model".to_owned(),
            4096,
            None,
            LoadMetrics::default(),
            QueueConfig {
                max_waiting_requests,
                wait_timeout,
            },
            shutdown_tx,
        )
    }

    async fn wait_for_waiting(state: &ServerState, expected: usize) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if state.inner.request_queue.waiting.load(Ordering::Acquire) == expected {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("queue did not reach the expected waiting count");
    }

    #[tokio::test]
    async fn inference_queue_is_bounded_and_fifo() {
        let state = test_state_with_queue(2, Duration::from_secs(1));
        let active = acquire_request_permit(&state).await.unwrap();
        let (order_tx, mut order_rx) = mpsc::unbounded_channel();

        let second_state = state.clone();
        let second_tx = order_tx.clone();
        tokio::spawn(async move {
            let permit = acquire_request_permit(&second_state).await.unwrap();
            second_tx.send((2_u8, permit)).unwrap();
        });
        wait_for_waiting(&state, 1).await;

        let third_state = state.clone();
        tokio::spawn(async move {
            let permit = acquire_request_permit(&third_state).await.unwrap();
            order_tx.send((3_u8, permit)).unwrap();
        });
        wait_for_waiting(&state, 2).await;

        let overflow = acquire_request_permit(&state).await.unwrap_err();
        assert_eq!(overflow.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(overflow.code, Some("queue_full"));
        let overflow_response = overflow.into_response();
        assert_eq!(
            overflow_response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .unwrap(),
            "1"
        );

        drop(active);
        let (ticket, second) = tokio::time::timeout(Duration::from_secs(1), order_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ticket, 2);
        assert!(second.queue_wait_ns() > 0);
        drop(second);
        let (ticket, third) = tokio::time::timeout(Duration::from_secs(1), order_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ticket, 3);
        assert!(third.queue_wait_ns() > 0);
        drop(third);

        let queue = state.inner.request_queue.snapshot();
        assert_eq!(queue["active_requests"], 0);
        assert_eq!(queue["waiting_requests"], 0);
        assert_eq!(queue["started_total"], 3);
        assert_eq!(queue["completed_total"], 3);
        assert_eq!(queue["queued_total"], 2);
        assert_eq!(queue["rejected_total"], 1);
    }

    #[tokio::test]
    async fn inference_queue_timeout_is_structured_and_recoverable() {
        let state = test_state_with_queue(1, Duration::from_millis(10));
        let active = acquire_request_permit(&state).await.unwrap();
        let timeout = acquire_request_permit(&state).await.unwrap_err();
        assert_eq!(timeout.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(timeout.code, Some("queue_timeout"));
        assert_eq!(state.inner.request_queue.waiting.load(Ordering::Acquire), 0);
        assert_eq!(
            state
                .inner
                .request_queue
                .timed_out_total
                .load(Ordering::Relaxed),
            1
        );
        drop(active);
        let recovered = acquire_request_permit(&state).await.unwrap();
        drop(recovered);
    }

    #[tokio::test]
    async fn shutdown_releases_waiters_and_rejects_new_requests() {
        let state = test_state_with_queue(1, Duration::from_secs(30));
        let active = acquire_request_permit(&state).await.unwrap();
        let waiting_state = state.clone();
        let waiter = tokio::spawn(async move { acquire_request_permit(&waiting_state).await });
        wait_for_waiting(&state, 1).await;

        state.begin_shutdown();
        let queued_error = waiter.await.unwrap().unwrap_err();
        assert_eq!(queued_error.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(queued_error.code, Some("service_shutting_down"));
        let new_error = acquire_request_permit(&state).await.unwrap_err();
        assert_eq!(new_error.code, Some("service_shutting_down"));
        assert_eq!(state.inner.request_queue.waiting.load(Ordering::Acquire), 0);

        let health = health(State(state.clone())).await.0;
        assert_eq!(health["ready"], false);
        assert_eq!(health["status"], "draining");
        assert_eq!(health["queue"]["accepting_requests"], false);
        drop(active);
    }

    #[tokio::test]
    async fn model_retrieve_is_openai_shaped() {
        let response = test_app("")
            .oneshot(
                Request::get("/v1/models/test-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["id"], "test-model");
        assert_eq!(value["object"], "model");
        assert_eq!(value["owned_by"], "qrt");
        assert_eq!(value["max_model_len"], 4096);
    }

    #[tokio::test]
    async fn model_retrieve_unknown_model_returns_openai_error() {
        let response = test_app("")
            .oneshot(
                Request::get("/v1/models/missing-model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["param"], "model");
        assert_eq!(value["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn chat_completion_is_openai_shaped() {
        let response = test_app("OK.")
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":3,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["content"], "OK.");
        assert_eq!(value["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completion_stream_ends_with_done() {
        let response = test_app("abc")
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","prompt":"x","max_tokens":3,"temperature":0,"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"object\":\"text_completion\""));
        assert!(text.contains("\"usage\":null"));
        assert!(text.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn chat_completion_returns_structured_named_tool_call() {
        let output = concat!(
            "<tool_call>\n",
            "<function=get_weather>\n",
            "<parameter=city>\n",
            "Shanghai\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>"
        );
        let request = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Use the weather tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            },
            "parallel_tool_calls": false,
            "max_completion_tokens": output.len(),
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        let response = test_app(output)
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        assert!(value["choices"][0]["message"]["content"].is_null());
        let call = &value["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "get_weather");
        let arguments: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"city": "Shanghai"}));
    }

    #[tokio::test]
    async fn chat_stream_usage_chunk_precedes_done() {
        let response = test_app("abc")
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"x"}],"max_tokens":3,"temperature":0,"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let data_lines = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect::<Vec<_>>();
        assert_eq!(data_lines.last(), Some(&"[DONE]"));
        let usage: Value = serde_json::from_str(data_lines[data_lines.len() - 2]).unwrap();
        assert_eq!(usage["object"], "chat.completion.chunk");
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["completion_tokens"], 3);
        assert_eq!(
            usage["usage"]["total_tokens"].as_u64().unwrap(),
            usage["usage"]["prompt_tokens"].as_u64().unwrap() + 3
        );
    }

    #[tokio::test]
    async fn chat_stream_does_not_publish_tokens_after_eos() {
        let response = test_app("A\0B")
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","messages":[{"role":"user","content":"x"}],"max_tokens":3,"temperature":0,"stream":true,"stream_options":{"include_usage":true},"chat_template_kwargs":{"enable_thinking":false}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let chunks = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str::<Value>(data).unwrap())
            .collect::<Vec<_>>();
        let content = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"].get(0))
            .filter_map(|choice| choice["delta"]["content"].as_str())
            .collect::<String>();
        assert_eq!(content, "A");
        let finish = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"].get(0))
            .find_map(|choice| choice["finish_reason"].as_str());
        assert_eq!(finish, Some("stop"));
        let usage = chunks
            .iter()
            .find(|chunk| chunk["choices"] == json!([]))
            .unwrap();
        assert_eq!(usage["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn completion_stream_does_not_publish_tokens_after_eos() {
        let response = test_app("A\0B")
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","prompt":"x","max_tokens":3,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        let chunks = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str::<Value>(data).unwrap())
            .collect::<Vec<_>>();
        let content = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"].get(0))
            .filter_map(|choice| choice["text"].as_str())
            .collect::<String>();
        assert_eq!(content, "A");
        let finish = chunks
            .iter()
            .filter_map(|chunk| chunk["choices"].get(0))
            .find_map(|choice| choice["finish_reason"].as_str());
        assert_eq!(finish, Some("stop"));
        let usage = chunks
            .iter()
            .find(|chunk| chunk["choices"] == json!([]))
            .unwrap();
        assert_eq!(usage["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn completion_context_accepts_exact_limit_and_rejects_one_token_over() {
        let app = test_app_with_context("WXYZ", 12);
        let accepted = app
            .clone()
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","prompt":"12345678","max_tokens":4,"temperature":0}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);

        let rejected = app
            .oneshot(
                Request::post("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"test-model","prompt":"12345678","max_tokens":5,"temperature":0}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
        let body = rejected.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "context_length_exceeded");
    }

    #[tokio::test]
    async fn admin_shutdown_requires_bearer_key_and_signals_once() {
        let (app, mut shutdown_rx) = test_app_with_options("", 4096, Some("test-secret"));
        let unauthorized = app
            .clone()
            .oneshot(
                Request::post("/admin/shutdown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            unauthorized
                .headers()
                .get(axum::http::header::WWW_AUTHENTICATE)
                .unwrap(),
            "Bearer"
        );
        assert!(matches!(
            shutdown_rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));

        let authorized = app
            .oneshot(
                Request::post("/admin/shutdown")
                    .header(axum::http::header::AUTHORIZATION, "Bearer test-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
        let body = authorized.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["status"], "stopping");
        shutdown_rx.await.unwrap();
    }

    #[test]
    fn stop_is_not_streamed() {
        let mut stream = TextStreamState::new(vec!["STOP".to_owned()]);
        assert_eq!(stream.update("hello ST", false), Some("hello".to_owned()));
        assert_eq!(
            stream.update("hello STOP trailing", false),
            Some(" ".to_owned())
        );
        assert!(stream.stopped);
    }

    #[test]
    fn eos_is_counted_but_not_decoded() {
        let finalized =
            finalize_tokens(&ByteCodec, &[b'A' as u32, 0, b'B' as u32], &[], false, 8).unwrap();
        assert_eq!(finalized.text, "A");
        assert_eq!(finalized.generated_token_ids, vec![b'A' as u32, 0]);
        assert_eq!(finalized.generated_token_count, 2);
        assert_eq!(finalized.finish_reason, "stop");
        assert_eq!(finalized.stop_reason, None);
    }
}
