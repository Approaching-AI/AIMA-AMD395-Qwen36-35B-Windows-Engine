#pragma once
#include "../src/qrt_context_limits.h"

namespace qrt_sm121_attention_capacity {
// Workspace capacity, not a claim that every context has passed model acceptance.
// Keep the existing allocation extent for ordinary calls. Extended storage is
// allocated only when a call crosses it; q8192 does not pay for the larger cap.
constexpr unsigned kInitialTokens = 131072u;
// A 256k owner, 1024 real suffix inputs and the complete resident decode tail,
// rounded to the exact attention tile width. Numerical acceptance is separate.
constexpr unsigned kTokens = QRT_QWEN36_ATTENTION_CAPACITY_TOKENS;
static_assert(kInitialTokens < kTokens && kInitialTokens % 32u == 0u);
static_assert(kTokens >= 262144u + 1024u + 1536u + 1u);
static_assert(kTokens - (262144u + 1024u + 1536u + 1u) < 32u);
static_assert(kTokens % 32u == 0u);
}
