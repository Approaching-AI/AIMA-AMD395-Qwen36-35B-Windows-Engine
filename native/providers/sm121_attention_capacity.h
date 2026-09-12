#pragma once

namespace qrt_sm121_attention_capacity {
// Workspace capacity, not a claim that every context has passed model acceptance.
// Includes the 32768-token owner, 1024 suffix inputs and subsequent decode.
constexpr unsigned kTokens = 65536u;
}
