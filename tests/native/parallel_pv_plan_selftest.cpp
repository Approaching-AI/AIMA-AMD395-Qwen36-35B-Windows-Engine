#include "../../native/providers/ck_fmha/parallel_pv_plan.h"
#include "../../native/providers/moe_accumulator/sm121_pv_final_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace plan = qrt_parallel_pv;
namespace b = qrt_sm121_pv_bound;
namespace f = qrt_sm121_pv_final_bound;
uint32_t random_word(uint32_t& seed) {
    seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed;
}
int main() {
    size_t shapes = 0u;
    for (unsigned start = 0u; start < 8192u; ++start) {
        for (unsigned count : {1u, 2u, 15u, 16u, 17u, 31u, 32u, 64u, 127u, 128u}) {
            if (count > 8192u - start) continue;
            assert(plan::valid(start, count, 0u, start + count));
            assert(plan::valid(start, count, 8192u - count, 8192u));
            assert(!plan::valid(start, count, 8193u - count, 8192u));
            const size_t capacity = plan::partial_count(start, count);
            assert(capacity && capacity * sizeof(plan::Partial) <= 268435456u);
            for (unsigned offset = 0u; offset < count; offset += 16u) {
                const unsigned active = count - offset < 16u ? count - offset : 16u;
                const size_t last = ((size_t(plan::groups(start + offset + active) - 1u) * 16u + active - 1u) * 16u + 15u) * 256u + 255u;
                assert(last < capacity);
            }
            ++shapes;
        }
    }
    for (unsigned count : {0u, 129u, 0xffffffffu}) {
        assert(!plan::valid(0u, count, 0u, 8192u));
        assert(plan::partial_count(0u, count) == 0u);
    }
    assert(!plan::valid(0xffffffffu, 1u, 0u, 8192u));
    assert(!plan::valid(0u, 1u, 0u, 8193u));
    assert(plan::partial_count(8192u, 1u) == 0u);

    // Independent wide26-bit reference versus a zero-carry FP32 dot followed
    // by one FP32 add. Vary both dot trees, cancellation, exponent range, alpha
    // resets and complete512-group sequences. This is a host envelope test;
    // actual WMMA results require the separate native test.
    unsigned groups = 0u, admissions = 0u, differing = 0u;
    for (unsigned trial = 0u; trial < 1024u; ++trial) {
        uint32_t seed = 0x3958192u ^ (977u * trial + 1u);
        float proxy = 0.0f, reference = 0.0f, state = 0.0f;
        for (unsigned group = 0u; group < 512u; ++group) {
            if (!(group & 1u)) {
                const float alpha = group % 64u == 0u ? (trial % 3u ? 0.625f : 0.0f) :
                    group % 8u == 0u ? 0.99609375f : 1.0f;
                state = f::rescale(state, proxy, alpha);
                proxy = f::multiply(proxy, alpha); reference = f::multiply(reference, alpha);
            }
            uint16_t p[16], v[16]; float terms[16], absolute = 0.0f;
            for (unsigned i = 0u; i < 16u; ++i) {
                const unsigned pe = trial % 3u ? 118u + random_word(seed) % 10u : 64u + random_word(seed) % 64u;
                const unsigned ve = trial % 3u ? 119u + random_word(seed) % 16u : 64u + random_word(seed) % 127u;
                p[i] = uint16_t(pe << 7u | (random_word(seed) & 127u));
                v[i] = uint16_t(ve << 7u | (random_word(seed) & 0x807fu));
                if (trial % 5u == 0u && (i & 1u)) { p[i] = p[i - 1u]; v[i] = v[i - 1u] ^ 0x8000u; }
                if (trial % 7u == 0u && i % 4u == 0u) p[i] = 0u;
                terms[i] = f::multiply(b::value(uint32_t(p[i]) << 16u), b::value(uint32_t(v[i]) << 16u));
                absolute = f::add(absolute, b::absolute(terms[i]));
            }
            state = f::group(state, proxy, absolute);
            reference = qrt_q1_moe_hawkeye::accumulate_bf16_impl<26, 16, -133>(reference, p, v, 16u);
            float dot = 0.0f;
            if (trial & 1u) {
                for (unsigned i = 0u; i < 16u; ++i) dot = f::add(dot, terms[15u - i]);
            } else {
                for (unsigned step = 8u; step; step >>= 1u)
                    for (unsigned i = 0u; i < step; ++i) terms[i] = f::add(terms[i], terms[i + step]);
                dot = terms[0];
            }
            proxy = f::add(proxy, dot); ++groups;
            differing += unsigned(b::bits(proxy) != b::bits(reference));
            if (group & 1u) {
                const float error = f::finalize(state, group + 1u);
                assert(std::fabs(double(proxy) - reference) <= double(error));
                const float reciprocal = 0.00390625f + float(trial % 11u) * 0.00003125f;
                const float output = f::multiply(proxy, reciprocal), exact = f::multiply(reference, reciprocal);
                const float final_error = b::finish(error, proxy, reciprocal);
                assert(std::fabs(double(output) - exact) <= double(final_error));
                if (b::same_bf16(output, final_error)) {
                    assert(b::bf16(output) == b::bf16(exact)); ++admissions;
                }
            }
        }
    }
    assert(groups == 524288u && admissions && differing);
    std::printf("{\"kind\":\"parallel_pv_host_plan_and_envelope\",\"shapes\":%zu,\"groups\":%u,\"different_fp32\":%u,\"admitted_endpoints\":%u,\"underestimates\":0,\"false_admissions\":0,\"maximum_workspace_bytes\":268435456,\"native_wmma_checked\":false,\"inference_acceptance\":false}\n", shapes, groups, differing, admissions);
}
