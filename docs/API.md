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

## Tool calling

The chat endpoint accepts function tools, `tool_choice` values `auto`, `none`,
`required`, or a named function, plus `parallel_tool_calls`. Tool arguments are
normalized to a JSON object. A normal OpenAI continuation appends the assistant
tool-call message followed by a `role: "tool"` result message. The engine then
generates the assistant continuation with `tool_choice: "none"` if requested.

Tool output is untrusted model text. Applications must validate the function
name and JSON schema before executing any external action.

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
