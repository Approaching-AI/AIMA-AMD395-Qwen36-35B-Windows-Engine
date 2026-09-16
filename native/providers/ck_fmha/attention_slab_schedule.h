#pragma once

// Isolated execution experiment. Each slot owns a stream and scratch arena;
// submissions in that slot are ordered, while output query spans are disjoint.
// The caller completes input preparation before entering this scheduler.
namespace qrt_attention_slab_schedule {
struct Config { unsigned queries, batch, slots, window_slabs; };
struct Result {
    // 1: configuration; 2: submission; 3: completion; 4: deadline.
    unsigned error = 0u, submitted = 0u, completion_calls = 0u, windows = 0u;
    bool drained = true;
};
inline bool valid(Config c) {
    return c.queries && c.queries <= 8192u && c.batch && c.batch <= 128u &&
        (c.slots == 1u || c.slots == 2u || c.slots == 4u) &&
        (c.window_slabs == 1u || c.window_slabs == 8u) && c.slots <= c.window_slabs;
}
template<class Submit, class Complete, class Deadline>
Result run(Config c, Submit submit, Complete complete, Deadline deadline) {
    Result r;
    if (!valid(c)) { r.error = 1u; return r; }
    bool pending[4]{};
    auto drain = [&] {
        for (unsigned slot = 0u; slot < c.slots; ++slot) if (pending[slot]) {
            ++r.completion_calls;
            if (!complete(slot)) { if (!r.error) r.error = 3u; r.drained = false; }
            pending[slot] = false;
        }
    };
    unsigned in_window = 0u;
    for (unsigned offset = 0u; offset < c.queries; offset += c.batch) {
        if (!deadline()) { r.error = 4u; break; }
        const unsigned slot = r.submitted % c.slots;
        const unsigned count = c.queries - offset < c.batch ? c.queries - offset : c.batch;
        // Even a failed submission may have queued a producer before failing
        // on a consumer. Always attempt to drain that slot and every peer.
        pending[slot] = true;
        if (!submit(slot, offset, count)) { r.error = 2u; break; }
        ++r.submitted; ++in_window;
        if (in_window == c.window_slabs || offset + count == c.queries) {
            drain(); ++r.windows; in_window = 0u;
            if (r.error) break;
            if (!deadline()) { r.error = 4u; break; }
        }
    }
    drain();
    return r;
}
} // namespace qrt_attention_slab_schedule
