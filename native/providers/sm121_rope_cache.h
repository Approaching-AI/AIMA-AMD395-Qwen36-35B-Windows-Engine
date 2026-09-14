#ifndef QRT_SM121_ROPE_CACHE_H
#define QRT_SM121_ROPE_CACHE_H

#include "../src/qrt_context_limits.h"
#include <cstddef>

namespace qrt_sm121_rope_cache {
constexpr size_t kColumns = 64u;
constexpr size_t kOriginalRows = 262144u;
constexpr size_t kRuntimeRows = QRT_QWEN36_ATTENTION_CAPACITY_TOKENS;
static_assert(kRuntimeRows == 264736u, "recapture RoPE when extending the runtime capacity");

struct Layout {
    size_t rows;
    size_t bytes;
    unsigned char sha256[32];
};
constexpr Layout kOriginal = {kOriginalRows, kOriginalRows * kColumns * 2u, {
    0xba,0x12,0xce,0x21,0x83,0x27,0xd4,0xcf,0x23,0xaa,0xc7,0xdf,0xac,0xd8,0xe9,0xef,
    0xbc,0x99,0xfd,0x20,0x76,0x11,0xa8,0x46,0x62,0x27,0x08,0x98,0x38,0xef,0x0e,0x80}};
// The original MRoPE constructor already owns 4 * 262144 rows. This longer
// prefix preserves every byte of kOriginal and adds the resident suffix/tail.
constexpr Layout kRuntime = {kRuntimeRows, kRuntimeRows * kColumns * 2u, {
    0x1c,0x4d,0x86,0xd4,0x92,0xb5,0x87,0xa5,0xf4,0xf4,0x33,0xda,0xd8,0x4c,0x99,0x72,
    0x2d,0x2f,0xee,0x70,0x2b,0xcf,0x27,0xd7,0xaf,0x77,0xc7,0xfe,0x6a,0x5d,0x2d,0x6a}};

constexpr const Layout *layout_for_bytes(size_t bytes) {
    return bytes == kOriginal.bytes ? &kOriginal
        : bytes == kRuntime.bytes ? &kRuntime : nullptr;
}

constexpr bool position_supported(size_t rows, size_t position) {
    return (rows == kOriginalRows || rows == kRuntimeRows) && position < rows;
}
}  // namespace qrt_sm121_rope_cache
#endif
