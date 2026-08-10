# Changelog

## 1.0.0 - 2026-08-10

- Published the Apache-2.0 native Windows engine source and `gfx1151` AOT
  inventory for Qwen3.6-35B-A3B BF16 on Ryzen AI Max+ 395.
- Added the resident `qrt` lifecycle, OpenAI-compatible completions/chat API,
  SSE streaming, structured tool calls, and tool-result continuation.
- Added continuous prompt lengths through a 262,144-token total-context limit,
  prefix caching, and a bounded FIFO batch-one request queue.
- Added clean Windows build orchestration and numerical provider smoke tests,
  including selected-MoE event-ring backpressure coverage.
- Published correctness-attached performance, full MMLU-Pro parity data, and
  redacted OpenAI/lifecycle acceptance evidence.

## 0.1.0-dev - 2026-08-09

- Created the public release-preparation repository and governance files.
