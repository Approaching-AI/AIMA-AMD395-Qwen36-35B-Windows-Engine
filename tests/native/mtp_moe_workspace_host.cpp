#include "native/providers/gdn/sm121_mtp_moe_layout.h"
#include <array>
#include <cassert>
#include <limits>

using namespace qrt_sm121_mtp;
int main() {
    alignas(256) std::array<unsigned char, 160000> storage{};
    for (unsigned rows : {1u, 2u}) {
        const size_t needed = moe_workspace_bytes(rows);
        assert(needed <= storage.size() && needed % 256u == 0u);
        MoeBuffers buffers;
        assert(bind_moe_buffers(storage.data(), needed, rows, &buffers));
        struct Span { void* pointer; size_t bytes; };
        const Span spans[] = {
            {buffers.router, rows * 256u * 2u}, {buffers.shared_gate, rows * 2u},
            {buffers.shared_gate_up, rows * 1024u * 2u}, {buffers.shared_activated, rows * 512u * 2u},
            {buffers.shared_down, rows * 2048u * 2u}, {buffers.shared, rows * 2048u * 2u},
            {buffers.routed_gate_up, rows * 8192u * 2u}, {buffers.routed_activated, rows * 4096u * 2u},
            {buffers.routed_weighted, rows * 16384u * 2u}, {buffers.routed, rows * 2048u * 2u},
            {buffers.output, rows * 2048u * 2u}, {buffers.topk_ids, rows * 8u * 4u},
            {buffers.topk_weights, rows * 8u * 4u}, {buffers.invalid, 4u}
        };
        for (unsigned i = 0; i < 14u; ++i) {
            const auto* begin = static_cast<unsigned char*>(spans[i].pointer);
            assert(begin >= storage.data() && begin + spans[i].bytes <= storage.data() + needed);
            assert(reinterpret_cast<uintptr_t>(begin) % 256u == 0u);
            for (unsigned j = i + 1; j < 14u; ++j)
                assert(moe_disjoint(spans[i].pointer, spans[i].bytes, spans[j].pointer, spans[j].bytes));
        }
        auto* published = buffers.output;
        assert(!bind_moe_buffers(storage.data(), needed - 1u, rows, &buffers));
        assert(buffers.output == published);
        assert(!bind_moe_buffers(storage.data() + 1u, needed, rows, &buffers));
        assert(!bind_moe_buffers(nullptr, needed, rows, &buffers));
        assert(!bind_moe_buffers(storage.data(), needed, rows, nullptr));
        const uintptr_t last_page = std::numeric_limits<uintptr_t>::max() & ~uintptr_t(255u);
        assert(!bind_moe_buffers(reinterpret_cast<void*>(last_page), needed, rows, &buffers));
    }
    for (unsigned rows : {0u, 3u, ~0u}) {
        MoeBuffers buffers;
        assert(!moe_workspace_bytes(rows));
        assert(!bind_moe_buffers(storage.data(), storage.size(), rows, &buffers));
    }
    assert(!moe_disjoint(storage.data(), 32u, storage.data() + 16u, 32u));
    assert(moe_disjoint(storage.data(), 32u, storage.data() + 32u, 32u));
    assert(!moe_disjoint(nullptr, 32u, storage.data(), 32u));
    assert(!moe_disjoint(storage.data(), 0u, storage.data(), 32u));
    assert(!moe_disjoint(reinterpret_cast<void*>(~uintptr_t(0) - 3u), 8u, storage.data(), 32u));
}
