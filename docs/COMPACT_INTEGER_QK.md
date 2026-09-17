# Compact exact integer QK experiment

The previous scalar integer QK row occupies 156 bytes. This isolated retry
uses a 52-byte row: sixteen signed16 coefficients, the original biased
exponents and one control word. Supported BF16 values are represented exactly
under a common power-of-two scale. Unsupported ranges, exceptional encodings
and negative zero keep the original sixteen BF16 words and use original
arithmetic. Every original bit can be reconstructed; this is not quantization.

Four byte-dot products reconstruct signed16 products. The exact common K16
exponent and each signed discarded remainder reproduce the original
toward-zero alignment. Carries retain width26 and their original order.
Unsupported shifts or ranges fall back to the original group operation.
The host test checks all 65536 encodings in four scale contexts, 4194304
roundtrip words, 109075 independent signed16 products and 500000 raw carries
against `group_sum<26,-133>`. UBSan reports no errors. Its fallback/exact/
remainder path counts are 418314/4076/77610; generated coverage does not
predict captured-model coverage.

The native candidate reuses the current four-score 32x32 tile schedule, with
K64 and K128 windows. Shared row storage is 13312 or 26624 bytes. Register
allocation, spills and occupancy require native inspection. All query
encoding, one-time key encoding and fallback must be included in comparisons
against the current decoded four-score control.

The standalone native harness checks 131072 raw carry states, generated
partial/causal tiles, original score bits, independent CPU dots, complete
row encodings, immutable inputs and memory guards. Its captured q8192 shape
extends the original q7169 layer3 Q/K by repeating the first 1023 rows.
That extension is component arithmetic evidence, not a new GB10 prompt.
Every warmup and timed score is checked outside its timer. Captured encoding
coverage and 65536 independent host carry comparisons are diagnostics.

Native compilation, correctness and speed are pending. No provider dispatch,
runtime defaults or package changes. The current model baseline remains
23902.4417 ms TTFT; the 10000 ms boundary, retained 4187.415605 ms target
and full GB10 continuation requirements remain unchanged.
