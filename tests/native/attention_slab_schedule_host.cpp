#include "../../native/providers/ck_fmha/attention_slab_schedule.h"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace schedule = qrt_attention_slab_schedule;
void require(bool value) { if (!value) throw std::runtime_error("slab ownership check"); }
int main() {
    unsigned cases = 0u, faults = 0u;
    for (unsigned n : {1u, 17u, 127u, 128u, 129u, 1023u, 1024u, 1025u, 7169u, 8192u})
        for (auto geometry : {std::array<unsigned,2>{1u,1u}, {1u,8u}, {2u,8u}, {4u,8u}}) {
            const schedule::Config config{n, 128u, geometry[0], geometry[1]};
            const unsigned slabs = (n + 127u) / 128u;
            for (unsigned kind = 0u; kind < 4u; ++kind) {
                const unsigned failure_count = kind ? slabs + 2u : 1u;
                for (unsigned fail = 1u; fail <= failure_count; ++fail) {
                    unsigned submits = 0u, completes = 0u, deadlines = 0u;
                    bool injected = false;
                    std::vector<unsigned> coverage(n);
                    std::array<bool,4> pending{};
                    const auto result = schedule::run(config,
                        [&](unsigned slot, unsigned offset, unsigned count) {
                            require(slot == submits % config.slots && offset == submits * 128u);
                            require(count && count <= 128u && offset + count <= n);
                            pending[slot] = true; ++submits;
                            for (unsigned i = offset; i < offset + count; ++i) require(++coverage[i] == 1u);
                            if (kind == 1u && submits == fail) { injected = true; return false; }
                            return true;
                        },
                        [&](unsigned slot) {
                            require(pending[slot]); pending[slot] = false; ++completes;
                            if (kind == 2u && completes == fail) { injected = true; return false; }
                            return true;
                        },
                        [&] {
                            ++deadlines;
                            if (kind == 3u && deadlines == fail) { injected = true; return false; }
                            return true;
                        });
                    for (bool active : pending) require(!active);
                    require(result.completion_calls == completes);
                    require(result.error == (injected ? kind + 1u : 0u));
                    require(result.drained == !(injected && kind == 2u));
                    if (!injected) {
                        require(submits == slabs && result.submitted == slabs);
                        for (unsigned count : coverage) require(count == 1u);
                    }
                    ++cases; faults += injected;
                }
            }
        }
    unsigned callbacks = 0u;
    for (auto bad : {schedule::Config{0,128,1,1}, {8193,128,1,1}, {1,0,1,1},
                     {129,129,1,1}, {128,128,0,8}, {128,128,3,8}, {128,128,2,1},
                     {128,128,1,7}, {128,128,8,8}}) {
        const auto r = schedule::run(bad, [&](unsigned,unsigned,unsigned){++callbacks;return true;},
            [&](unsigned){++callbacks;return true;}, [&]{++callbacks;return true;});
        require(r.error == 1u && r.drained && !callbacks);
    }
    std::printf("{\"kind\":\"attention_slab_schedule_host\",\"cases\":%u,\"injected_failures\":%u,\"invalid_configs\":9,\"exact_query_coverage\":true,\"all_pending_slots_drained\":true,\"completion_failure_reported\":true}\n", cases, faults);
}
