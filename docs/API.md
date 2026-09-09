# OpenAI API and service operation

The resident server implements the OpenAI-compatible subset needed by common
chat and completion clients while keeping unsupported behavior explicit.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness, capabilities, load timing, and queue counters |
| `GET` | `/ready` | Same readiness contract as `/health` |
| `GET` | `/v1/models` | List the configured served model |
| `GET` | `/v1/models/{model}` | Retrieve the configured model |
| `POST` | `/v1/completions` | Text or raw-token completion |
| `POST` | `/v1/chat/completions` | Qwen chat template, tools, and continuation |
| `POST` | `/tokenize` | Tokenizer utility |
| `POST` | `/detokenize` | Detokenizer utility |
| `POST` | `/admin/shutdown` | Authenticated graceful shutdown used by `qrt stop` |

The base URL for OpenAI SDKs is `http://127.0.0.1:8000/v1` unless the operator
chooses another address or port.

## Streaming

Set `stream: true` on either completion endpoint. Responses use
`text/event-stream`, preserve the non-stream token output, finish with
`data: [DONE]`, and can include a final usage chunk with:

```json
{"stream_options":{"include_usage":true}}
```

Chat tool calls are emitted as OpenAI-shaped `delta.tool_calls` fragments and
finish with `finish_reason: "tool_calls"`.

## Thinking

The chat and `/tokenize` endpoints accept
`"thinking": {"type": "enabled", "budget_tokens": 128}` or
`"thinking": {"type": "disabled"}`. Type is required; the optional budget must
be a positive integer no greater than the resolved generation limit. The
budget is validated advisory metadata, not a separate reasoning-token cutoff:
`max_tokens` / `max_completion_tokens` bounds the entire generation (currently
at most 512 tokens). Unknown thinking fields and invalid types are rejected.

The older `chat_template_kwargs.enable_thinking` alias remains supported.
Conflicting explicit aliases are rejected. Omission and an empty kwargs object
both disable thinking; an empty object no longer accidentally enables it.
`preserve_thinking` continues to control reasoning in message history.

Enabled reasoning is returned in `message.reasoning_content` or
`delta.reasoning_content`; the answer is `content`. For text without tools,
reasoning and answer deltas arrive during generation, with split think markers,
UTF-8 boundaries and stop sequences held until safe. Structured tool output is
validated as a complete result before publishing executable calls. No GPU
correctness or packaged-runtime acceptance is implied by protocol unit tests.

## Tool calling

The chat endpoint accepts function tools, `tool_choice` values `auto`, `none`,
`required`, or a named function, plus `parallel_tool_calls`. Tool arguments are
normalized to a JSON object. A normal OpenAI continuation appends the assistant
tool-call message followed by a `role: "tool"` result message. The engine then
generates the assistant continuation with `tool_choice: "none"` if requested.

Tool output is untrusted model text. Applications must validate the function
name and JSON schema before executing any external action.

Identical function/argument calls within one response are suppressed after
canonical JSON key ordering (including nested objects); different arguments or
function names remain distinct. Only declared functions may be returned as
calls. `tool_choice: "none"` leaves generated markup as text, never executable
calls. `parallel_tool_calls: false` admits only the first valid unique call.

`qrt_tool_progress` in JSON and the SSE metadata chunk reports duplicate and
parallel suppression, exhausted-history suppression, and `no_progress`.
Two explicit failed/empty tool results for the same function/arguments exhaust
one permitted retry; that call is then suppressed. Tool results are matched by
call ID and a duplicate result ID is counted once. Classification is deliberately
conservative: arbitrary useful text mentioning an error is not failure evidence.
When all generated calls are exhausted, the response ends with `stop` and
`no_progress: true`, even for `tool_choice: "required"`. A distinct fallback
call may still be returned. The caller owns semantic retry strategy, idempotency
across HTTP retries, side-effect authorization, and the final blocked/best-effort
user response; the engine does not execute tools or infer progress from the world.

## Sampling and limits

- Batch size is exactly 1. Single-element text/token batches are accepted.
- Decode is deterministic greedy: `temperature=0`, `top_p=1`, and `n=1`.
- `presence_penalty`, `frequency_penalty`, logprobs, non-empty `logit_bias`,
  suffix insertion, and non-text response formats are rejected as unsupported.
- One native request emits at most 512 tokens.
- `prompt_tokens + requested_output_tokens` must not exceed
  `--max-model-len` (qualified at 262,144).
- There is no prompt-length allowlist. Every positive token length satisfying
  the total-context constraint follows the continuous arbitrary-context path.

Unsupported inputs return a structured OpenAI error with
`code: "unsupported_parameter"`; they are never silently approximated.

## Queue behavior

The target runtime executes one batch-one request at a time. Additional
requests enter a fair FIFO semaphore queue. Defaults are 64 waiting requests
and a 600-second wait timeout; configure them with:

```text
--max-queue-depth N
--queue-timeout-seconds S
```

Each successful response includes `x-qrt-queue-wait-ms` and
`qrt_metrics.queue_wait_ms`. `/health.queue` reports active/waiting counts,
totals, rejections, timeouts, and mean/maximum waits.

| HTTP | Error code | Meaning |
|---:|---|---|
| `429` | `queue_full` | Waiting capacity is exhausted |
| `503` | `queue_timeout` | A queued request exceeded its wait limit |
| `503` | `service_shutting_down` | Admission is closed during shutdown |
| `400` | `context_length_exceeded` | Prompt plus requested output exceeds the limit |

The three overload responses include `Retry-After`. Shutdown closes admission,
releases queued requests with `503`, allows the active request to finish, and
then releases native resources.

## Timing and qualification observations

The unreleased timing contract is `qrt_metrics.timing_contract_version: 2`.
`decode_total_ms` is the native aggregate decode duration; `tpot_ms` is that
total divided by `tpot_samples` (zero when there are no samples). Older
candidate responses without this version incorrectly exposed the aggregate
under `tpot_ms`; their saved numbers must not be interpreted as per-token
latency or silently rewritten. `ttft_ms` is the native report duration, not
an independently measured client's first-token arrival time.

For explicit q8192 qualification, the diagnostic override
`QRT_SERVER_FIRST_TOKEN_LOGIT_DIAGNOSTIC=1` emits a structured stderr
observation after each ordinary q8192 request. It binds the actual prompt
token digest, first emitted token and raw logit from the existing engine
report. A prefix seed, stale/nonmatching report or nonfinite logit is marked
unavailable. This does not change the v1 native ABI or expose OpenAI logprobs,
and it does not modify model computation or sample a replacement token.
It is off by default; normal logging is unchanged.

Observation v1 uses FNV offset `14695981039346656037`. Existing frozen QRT
oracle/CLI digests use the historical offset `1469598103934665603`. The
verifier computes both from the same SHA-256-verified little-endian token
array; it does not compare the differently seeded strings directly, change
the frozen oracle or treat this metadata difference as a numerical failure.

`scripts/verify_baiying_q8192_http.py` provides bounded two-request
qualification under the Windows Job guard. Its offline replay mode validates
saved JSON/SSE and shutdown evidence without loading a model. Regular SSE
chunks may have `usage: null`; only the final non-null usage object is the
aggregate. An offline parser repair retains the original failed controller
record and cannot invent missing logit or queue observations.

## Authentication and binding

`--api-key VALUE` requires `Authorization: Bearer VALUE` for model and admin
endpoints; comparison is constant-time. A non-loopback bind without an API key
is rejected unless `--allow-unauthenticated` is explicitly supplied. TLS is
outside this process and should be terminated by a trusted local proxy.

## Prefix cache

The provider records reusable prefill snapshots and uses copy-on-write for
extensions. Acceptance requires ordered log evidence for a seed, a COW hit,
and a later resident hit with no reseed, followed by an unrelated prompt and an
identical-output A-B-A isolation check. Cache reuse is an optimization; it does
not alter token output or the context-limit contract.
