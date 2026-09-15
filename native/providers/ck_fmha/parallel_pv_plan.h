#ifndef QRT_PARALLEL_PV_PLAN_H
#define QRT_PARALLEL_PV_PLAN_H
#include <cstddef>

namespace qrt_parallel_pv {
constexpr unsigned maximum_tokens = 8192u, maximum_queries = 128u;
constexpr unsigned tile_queries = 16u, heads = 16u, dimensions = 256u;
struct Partial { float value, absolute; };
static_assert(sizeof(Partial) == 8u);

// One sixteen-query slab is reused only after its ordered reduction finishes
// on the same stream. A q8192 call needs at most256 MiB, independent of its
// query batch. The final causal K32 tile always contains both K16 groups.
constexpr unsigned groups(unsigned end) { return (end + 31u) / 32u * 2u; }
constexpr bool valid(unsigned start, unsigned count, unsigned output_start,
                     unsigned stride) {
    return stride && stride <= maximum_tokens && count && count <= maximum_queries &&
        start < stride && count <= stride - start &&
        output_start < maximum_tokens && count <= maximum_tokens - output_start;
}
constexpr size_t partial_count(unsigned start, unsigned count) {
    return start < maximum_tokens && count && count <= maximum_queries &&
        count <= maximum_tokens - start
        ? size_t(groups(start + count)) * tile_queries * heads * dimensions : 0u;
}
static_assert(partial_count(8064u, 128u) * sizeof(Partial) == 268435456u);
} // namespace qrt_parallel_pv
#endif
