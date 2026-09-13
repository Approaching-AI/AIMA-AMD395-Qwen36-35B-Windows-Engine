#pragma once

namespace qrt_sm121_attention_capacity {
// Workspace capacity, not a claim that every context has passed model acceptance.
// Includes the 65536-token owner, 1024 suffix inputs and the resident decode
// tail. The 128k/256k prefix-plus-suffix targets need larger capacities and
// separate acceptance; this bound does not claim either target is complete.
constexpr unsigned kTokens = 131072u;
static_assert(kTokens >= 65536u + 1024u + 1536u + 1u);
static_assert(kTokens % 32u == 0u);
}
