# Changelog

## Unreleased

- Replaced the selected-MoE combine boundary with the sorted BF16 arithmetic
  used by the external authority and added exact sync/async v2/v3 parity gates.
- Published a reproducible, private-path-free nine-object q8192 AOT set with
  source generators, metadata hashes, real layer-3 GB10 component comparison,
  and PowerShell 5.1 transport verification.
- Added a strict cold random-length product verifier: each random length is
  bracketed by its interval's upper anchor, every response must match GB10,
  and route-log verification rejects fallbacks or prefix-cache contamination.
- Restored the FLA GDN generator and arbitrary-tail/Hawkeye numerical contract
  tests needed to reproduce and review the retained provider source.
- Promotion remains pending a clean native Windows build and
  correctness-attached real-model product qualification on `baiying` from the
  exact candidate commit.

## 1.0.1 - 2026-08-15

- Removed the q8191/q8193 shape cliff around the retained q8192 route while
  keeping q8192 below its confirmed 4,187.416 ms TTFT target.
- Corrected exact q8192 terminal MoE execution to use the GB10-valid resident
  raw-BF16 matrix path, with a q1-at-KV8192 terminal plan and no retained-only
  packed alias or device corridor.
- Added dynamic CK-FMHA and fused-GDN neighbor surfaces, a BF16 router endpoint,
  position-scoped decode tie policies, and inherited HIP launch-status cleanup.
- Promoted the accepted exact-arbitrary convolution arithmetic and layer-zero
  fused-GDN settings into the shipped runtime profile.
- Added a deterministic 18-case q8191/q8192/q8193 continuation gate. Every
  one- and two-token output matched the GB10 BF16 oracle across three cold
  repetitions per shape; the worst neighbor ratio was 1.036619x and the worst
  positive residual was 142.288 ms.
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
