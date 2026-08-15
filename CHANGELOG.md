# Changelog

## 1.0.1 - 2026-08-15

- Removed the q8191/q8193 shape cliff around the retained q8192 route while
  keeping q8192 below its confirmed 4,187.416 ms TTFT target.
- Corrected exact q8192 terminal MoE execution to use the GB10-valid resident
  raw-BF16 matrix path, with a q1-at-KV8192 terminal plan and no retained-only
  packed alias or device corridor.
- Added dynamic CK-FMHA and fused-GDN neighbor surfaces, a BF16 router endpoint,
  position-scoped decode tie policies, and inherited HIP launch-status cleanup.
- Added a deterministic 18-case q8191/q8192/q8193 continuation gate. Every
  one- and two-token output matched the GB10 BF16 oracle across three cold
  repetitions per shape; the worst neighbor ratio was 1.423x.
- Kept the causal padded verifier as an opt-in diagnostic; production inference
  uses the corrected native path directly.

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
