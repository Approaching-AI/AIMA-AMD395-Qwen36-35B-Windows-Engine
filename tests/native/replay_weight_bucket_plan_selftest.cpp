#include "../../native/providers/moe_accumulator/replay_weight_bucket_plan.h"
#include <cassert>
#include <cstdio>
#include <initializer_list>
namespace buckets = qrt_replay_weight_buckets;
int main() {
    size_t plans = 0u, cells = 0u;
    for (unsigned rows = 1u; rows <= 16384u; ++rows) for (unsigned tokens : {1u,7u,64u,65u,255u,256u,257u,8192u})
        for (unsigned policy = 0u; policy < 3u; ++policy) {
            const buckets::Plan plan{rows,tokens,policy == 0u ? 1u : policy == 1u ? 16u : 64u,policy == 0u ? 8192u : policy == 1u ? 256u : 64u};
            assert(buckets::valid(plan)); const unsigned bins = buckets::bucket_count(plan);
            const uint64_t columns = (uint64_t(rows) + plan.weight_rows - 1u) / plan.weight_rows;
            const uint64_t expected_bins = columns * ((uint64_t(tokens) + plan.token_rows - 1u) / plan.token_rows);
            assert(bins == expected_bins && bins <= 32768u && buckets::workspace_words(plan) == 3u * expected_bins + 2u);
            for (unsigned token : {0u,tokens / 2u,tokens - 1u}) for (unsigned row : {0u,rows / 2u,rows - 1u}) {
                const uint64_t cell = uint64_t(token) * rows + row;
                const uint64_t expected = (uint64_t(token) / plan.token_rows) * columns + row / plan.weight_rows;
                assert(buckets::bucket(plan,unsigned(cell)) == expected && expected < bins); ++cells;
            }
            assert(buckets::bucket(plan,rows*tokens) == UINT32_MAX && buckets::bucket(plan,UINT32_MAX) == UINT32_MAX); ++plans;
        }
    for (auto plan : {buckets::Plan{0,8192,1,8192},buckets::Plan{16385,8192,1,8192},buckets::Plan{8192,0,16,256},buckets::Plan{8192,8193,16,256},buckets::Plan{8192,8192,32,256},buckets::Plan{8192,8192,16,0}}) {
        assert(!buckets::valid(plan) && !buckets::bucket_count(plan) && !buckets::workspace_words(plan) && buckets::bucket(plan,0u) == UINT32_MAX);
    }
    std::printf("{\"kind\":\"replay_weight_bucket_host\",\"plans\":%zu,\"independent_cell_mappings\":%zu,\"maximum_bins\":32768,\"maximum_workspace_bytes\":393224,\"raw_mismatches\":0,\"native_execution\":false}\n",plans,cells);
}
