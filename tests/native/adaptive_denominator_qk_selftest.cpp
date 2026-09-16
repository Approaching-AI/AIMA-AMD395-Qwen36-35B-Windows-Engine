#include "../../native/providers/ck_fmha/adaptive_denominator_tile_qk.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
namespace route = qrt_selective_qk;
namespace adaptive = qrt_adaptive_denominator_qk;
namespace prepared = qrt_prepared_decoded_qk;
constexpr unsigned batch = 128u, guard = 64u;
enum Metric : unsigned {
    ScoreCells, ScoreOutside, NativeScoreDifferent, ScoreMaximumRatio,
    ProbabilityCells, ProbabilityDifferent, AlphaDifferent, DenominatorRows, DenominatorOutside,
    OutputCells, OutputDifferent, ExternalDifferent, ExternalCells, Nonfinite,
    RepairCells, RepairDifferent, CertifiedWrong, FinalRows, TileCells, TileMembershipErrors, TileListErrors, MetricCount
};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void finish() {
    check(hipGetLastError());
    hipEvent_t event{}; check(hipEventCreateWithFlags(&event, hipEventDisableTiming)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        check(status == hipErrorNotReady ? hipSuccess : status);
        require(std::chrono::steady_clock::now() < deadline, "selective QK completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class Function> double timed(Function function) {
    const auto start = std::chrono::steady_clock::now(); function(); finish();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
template<class T> struct Buffer {
    T* allocation = nullptr; size_t size;
    explicit Buffer(size_t n) : size(n) {
        check(hipMalloc(reinterpret_cast<void**>(&allocation), (n + 2u * guard) * sizeof(T)));
        check(hipMemset(allocation, 0xa5, (n + 2u * guard) * sizeof(T)));
    }
    Buffer(const Buffer&) = delete;
    ~Buffer() { if (allocation) (void)hipFree(allocation); }
    T* data() { return allocation + guard; }
    void upload(const std::vector<T>& values) {
        require(values.size() == size, "upload length");
        check(hipMemcpy(data(), values.data(), size * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> download() {
        std::vector<T> values(size); check(hipMemcpy(values.data(), data(), size * sizeof(T), hipMemcpyDeviceToHost)); return values;
    }
    void guards() {
        std::array<unsigned char, guard * sizeof(T)> bytes{};
        for (size_t offset : {size_t(0u), size + guard}) {
            check(hipMemcpy(bytes.data(), allocation + offset, bytes.size(), hipMemcpyDeviceToHost));
            for (auto b : bytes) require(b == 0xa5u, "buffer redzone changed");
        }
    }
    void immutable(const std::vector<T>& values) {
        const auto actual = download();
        require(values.size() == actual.size() && !std::memcmp(actual.data(), values.data(), size * sizeof(T)), "input changed"); guards();
    }
};
template<class T> std::vector<T> read(const char* path, size_t n) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file && file.tellg() == std::streamoff(n * sizeof(T)), "input file length");
    std::vector<T> out(n); file.seekg(0);
    require(bool(file.read(reinterpret_cast<char*>(out.data()), n * sizeof(T))), "input read"); return out;
}
__device__ void count_metric(unsigned* stats, unsigned index, unsigned count) {
    for (unsigned offset = 16u; offset; offset >>= 1u) count += __shfl_down(count, offset, 32u);
    if (threadIdx.x % 32u == 0u && count) atomicAdd(stats + index, count);
}
__device__ void maximum_metric(unsigned* stats, unsigned index, float value) {
    for (unsigned offset = 16u; offset; offset >>= 1u) value = fmaxf(value, __shfl_down(value, offset, 32u));
    if (threadIdx.x % 32u == 0u) atomicMax(stats + index, __float_as_uint(value));
}
__global__ void inspect_scores(const float* canonical, const float* native, const float* errors,
    unsigned start, unsigned rows, unsigned stride, unsigned* stats) {
    unsigned cells = 0u, outside = 0u, different = 0u; float ratio = 0.0f;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * stride; i += gridDim.x * blockDim.x) {
        if (i % stride > start + i / stride / kQueryHeads) continue;
        ++cells; const auto interval = route::score_interval(native[i], errors[i]);
        outside += !(canonical[i] >= interval.low && canonical[i] <= interval.high);
        different += __float_as_uint(canonical[i]) != __float_as_uint(native[i]);
        if (errors[i] > 0.0f && isfinite(errors[i])) ratio = fmaxf(ratio, fabsf(canonical[i] - native[i]) / errors[i]);
    }
    count_metric(stats, ScoreCells, cells); count_metric(stats, ScoreOutside, outside);
    count_metric(stats, NativeScoreDifferent, different); maximum_metric(stats, ScoreMaximumRatio, ratio);
}
__global__ void inspect_probabilities(const uint16_t* canonical, const uint16_t* actual,
    unsigned start, unsigned rows, unsigned stride, unsigned* stats) {
    unsigned cells = 0u, different = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * stride; i += gridDim.x * blockDim.x) {
        if (i % stride > start + i / stride / kQueryHeads) continue;
        ++cells; different += canonical[i] != actual[i];
    }
    count_metric(stats, ProbabilityCells, cells); count_metric(stats, ProbabilityDifferent, different);
}
__global__ void inspect_scales(const float* canonical, const float* actual, const float* den,
    unsigned start, unsigned rows, unsigned stride, unsigned* stats) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x, tiles = (stride + 31u) / 32u;
    unsigned alphas = 0u, outside = 0u;
    if (row < rows) {
        const size_t base = size_t(row) * (tiles + 1u);
        for (unsigned tile = 0u; tile < (start + row / kQueryHeads + 32u) / 32u; ++tile)
            alphas += __float_as_uint(canonical[base + tile]) != __float_as_uint(actual[base + tile]);
        const float value = canonical[base + tiles];
        outside = !(value >= den[size_t(row) * 2u] && value <= den[size_t(row) * 2u + 1u]);
    }
    count_metric(stats, AlphaDifferent, alphas); count_metric(stats, DenominatorRows, unsigned(row < rows));
    count_metric(stats, DenominatorOutside, outside);
}
__global__ void inspect_repaired(const float* canonical, const float* actual,
    const unsigned* indices, const unsigned* count, unsigned* stats) {
    unsigned cells = 0u, different = 0u;
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count; slot += gridDim.x * blockDim.x) {
        ++cells; const unsigned cell = indices[slot];
        different += __float_as_uint(canonical[cell]) != __float_as_uint(actual[cell]);
    }
    count_metric(stats, RepairCells, cells); count_metric(stats, RepairDifferent, different);
}
__global__ void inspect_tile_membership(const unsigned* indices, const unsigned* count,
    unsigned stride, const unsigned* masks, unsigned* stats) {
    const unsigned key_tiles = (stride + 15u) / 16u;
    unsigned missing = 0u;
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count; slot += gridDim.x * blockDim.x) {
        const unsigned cell = indices[slot], row = cell / stride, key = cell % stride;
        const unsigned head = row % kQueryHeads, query = row / kQueryHeads;
        const unsigned tile = (query / 16u * kQueryHeads + head) * key_tiles + key / 16u;
        const unsigned position = query % 16u * 16u + key % 16u;
        missing += !(masks[size_t(tile) * 8u + position / 32u] & (1u << (position % 32u)));
    }
    count_metric(stats, TileMembershipErrors, missing);
}
__global__ void inspect_tile_list(const unsigned* tiles, const unsigned* count, unsigned* owners,
    const unsigned* masks, unsigned start, unsigned queries, unsigned stride, unsigned capacity, unsigned* stats) {
    const unsigned key_tiles = (stride + 15u) / 16u;
    unsigned cells = 0u, invalid = 0u;
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count; slot += gridDim.x * blockDim.x) {
        const unsigned tile = tiles[slot];
        if (tile >= capacity) { ++invalid; continue; }
        invalid += atomicAdd(owners + tile, 1u) != 1u;
        const unsigned query_tile = tile / key_tiles / kQueryHeads * 16u;
        const unsigned key_tile = tile % key_tiles * 16u;
        for (unsigned word = 0u; word < 8u; ++word) {
            const unsigned mask = masks[size_t(tile) * 8u + word];
            cells += unsigned(__popc(mask));
            for (unsigned bit = 0u; bit < 32u; ++bit) {
                if (!(mask & (1u << bit))) continue;
                const unsigned position = word * 32u + bit;
                const unsigned query = query_tile + position / 16u, key = key_tile + position % 16u;
                invalid += query >= queries || key >= stride || key > start + query;
            }
        }
    }
    count_metric(stats, TileCells, cells); count_metric(stats, TileListErrors, invalid);
}
__global__ void inspect_certificate(const float* actual, const float* canonical,
    const unsigned* pending, unsigned cells, bool final, unsigned* stats) {
    unsigned wrong = 0u, unresolved = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < cells; i += gridDim.x * blockDim.x) {
        const bool ready = !pending[i / kHeadDim];
        wrong += ready && f32_to_bf16(actual[i]) != f32_to_bf16(canonical[i]);
        unresolved += final && i % kHeadDim == 0u && !ready;
    }
    count_metric(stats, CertifiedWrong, wrong); count_metric(stats, FinalRows, unresolved);
}
__global__ void inspect_final(const float* output, const float* canonical, const uint16_t* reference,
    unsigned start, unsigned cells, unsigned reference_tokens, unsigned* stats) {
    unsigned count = 0u, different = 0u, external_different = 0u, external_cells = 0u, nonfinite = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < cells; i += gridDim.x * blockDim.x) {
        ++count; const auto actual = f32_to_bf16(output[i]), expected = f32_to_bf16(canonical[i]);
        different += actual != expected; nonfinite += !isfinite(output[i]) || !isfinite(canonical[i]);
        const size_t index = size_t(start) * kQueryHeads * kHeadDim + i;
        if (index < size_t(reference_tokens) * kQueryHeads * kHeadDim) {
            ++external_cells; external_different += actual != reference[index] || expected != reference[index];
        }
    }
    count_metric(stats, OutputCells, count); count_metric(stats, OutputDifferent, different);
    count_metric(stats, ExternalDifferent, external_different); count_metric(stats, ExternalCells, external_cells);
    count_metric(stats, Nonfinite, nonfinite);
}
float as_float(unsigned word) { float x; std::memcpy(&x, &word, 4u); return x; }
void run(unsigned tokens, unsigned mode, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& v, const std::vector<uint16_t>& reference,
    const std::vector<unsigned char>& exp2, const std::vector<unsigned char>& reciprocal, unsigned reference_tokens, unsigned variant) {
    require(variant < 4u, "invalid component variant");
    const bool tile_repair = (variant & 1u) != 0u;
    const unsigned step = variant & 2u ? 4u : 1u, actual_rounds = 16u / step + 2u;
    Buffer<uint16_t> dq(q.size()), dk(k.size()), dv(v.size()), dt(k.size()), dr(reference.size());
    Buffer<unsigned char> de(exp2.size()), drecip(reciprocal.size());
    dq.upload(q); dk.upload(k); dv.upload(v); dr.upload(reference); de.upload(exp2); drecip.upload(reciprocal);
    const size_t score_capacity = size_t(batch) * kQueryHeads * tokens;
    const size_t scale_capacity = size_t(batch) * kQueryHeads * ((tokens + 31u) / 32u + 1u);
    const size_t output_capacity = size_t(batch) * kQueryHeads * kHeadDim;
    Buffer<float> es(score_capacity), ns(score_capacity), errors(score_capacity), ez(scale_capacity), nz(scale_capacity), maxima(scale_capacity);
    Buffer<uint16_t> ep(score_capacity), np(score_capacity);
    Buffer<float> canonical(output_capacity), accumulator(output_capacity), output(output_capacity), den(size_t(batch) * kQueryHeads * 2u);
    Buffer<float> lower(score_capacity), upper(score_capacity), maximum_cost(size_t(batch) * kQueryHeads);
    Buffer<unsigned> indices(score_capacity), count(1u), pending(size_t(batch) * kQueryHeads), stats(MetricCount), adaptive_stats(adaptive::rounds + 2u);
    const unsigned tile_capacity = ((batch + 15u) / 16u) * kQueryHeads * ((tokens + 15u) / 16u);
    Buffer<unsigned> tile_masks(size_t(tile_capacity) * 8u), tile_owners(tile_capacity), tile_indices(tile_capacity), tile_count(1u);
    Buffer<uint32_t> packed(prepared::workspace_words);
    prepared::Workspace workspace{packed.data(), tokens};
    auto* pq = packed.data(); auto* pk = pq + prepared::query_words;
    auto* qflags = pk + prepared::key_words; auto* kflags = qflags + prepared::query_flag_words;
    check(hipMemset(stats.data(), 0, MetricCount * 4u));
    check(hipMemset(adaptive_stats.data(), 0, adaptive_stats.size * 4u));
    const double preparation_ms = timed([&] {
        check(hipError_t(prepared::prepare_workspace(dq.data(), dk.data(), dt.data(), workspace, nullptr)));
    });
    double original_qk_ms = 0.0, original_probability_ms = 0.0, native_bound_ms = 0.0, repair_pipeline_ms = 0.0, canonical_pv_ms = 0.0;
    uint64_t selected[2u + adaptive::rounds - 1u] = {}, selected_tiles = 0u; unsigned cpu_dots = 0u;
    double collection_ms = 0.0, replay_ms = 0.0, initialization_ms = 0.0, certificate_ms = 0.0;
    for (unsigned start = 0u; start < tokens; start += batch) {
        const unsigned queries = std::min(batch, tokens - start), stride = start + queries;
        const unsigned rows = queries * kQueryHeads, cells = rows * kHeadDim, tiles = (stride + 31u) / 32u;
        original_qk_ms += timed([&] {
            check(hipError_t(prepared::launch_workspace(&workspace, dq.data(), dt.data(), es.data(), nullptr, start, queries, stride, tokens)));
        });
        for (unsigned sample = 0u; sample < 2u; ++sample) {
            const unsigned query = sample ? queries - 1u : 0u, head = sample ? 15u : 0u;
            const unsigned key = sample ? (start + query) / 2u : start + query, kv_head = head / 8u;
            const float expected = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                q.data() + (size_t(start + query) * kQueryHeads + head) * kHeadDim,
                k.data() + (size_t(key) * kKvHeads + kv_head) * kHeadDim, kHeadDim) * kExactScale;
            float actual; check(hipMemcpy(&actual, es.data() + (size_t(query) * kQueryHeads + head) * stride + key, 4u, hipMemcpyDeviceToHost));
            require(!std::memcmp(&expected, &actual, 4u), "original score differs from independent wide CPU accumulator"); ++cpu_dots;
        }
        native_bound_ms += timed([&] {
            hipLaunchKernelGGL(route::native_scores, dim3((stride + kIntegerMatrixColumns - 1u) / kIntegerMatrixColumns,
                kQueryHeads, (queries + 15u) / 16u), dim3(kThreads), 0u, nullptr,
                dq.data(), dt.data(), ns.data(), errors.data(), start, queries, stride, tokens);
        });
        hipLaunchKernelGGL(inspect_scores, dim3(128u), dim3(256u), 0u, nullptr,
            es.data(), ns.data(), errors.data(), start, rows, stride, stats.data()); finish();
        const auto repair = [&](bool update) {
            if (tile_repair) {
                check(hipMemset(tile_masks.data(), 0, tile_masks.size * sizeof(unsigned)));
                check(hipMemset(tile_owners.data(), 0, tile_owners.size * sizeof(unsigned)));
                check(hipMemset(tile_count.data(), 0, sizeof(unsigned)));
                hipLaunchKernelGGL(adaptive::mark_tiles, dim3(256u), dim3(256u), 0u, nullptr,
                    indices.data(), count.data(), stride, tile_masks.data(), tile_owners.data(), tile_indices.data(), tile_count.data());
                check(hipGetLastError());
                if (update) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(adaptive::repair_tiles<true>), dim3(4096u), dim3(256u), 0u, nullptr,
                        dq.data(), dt.data(), pq, pk, qflags, kflags, ns.data(), errors.data(), start, queries, stride, tokens,
                        tile_masks.data(), tile_indices.data(), tile_count.data(), maxima.data(), lower.data(), upper.data(), de.data());
                } else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(adaptive::repair_tiles<false>), dim3(4096u), dim3(256u), 0u, nullptr,
                        dq.data(), dt.data(), pq, pk, qflags, kflags, ns.data(), errors.data(), start, queries, stride, tokens,
                        tile_masks.data(), tile_indices.data(), tile_count.data(), nullptr, nullptr, nullptr, nullptr);
                }
                check(hipGetLastError()); return;
            }
            if (update) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(adaptive::repair<true>), dim3(256u), dim3(256u), 0u, nullptr,
                    dq.data(), dt.data(), pq, pk, qflags, kflags, ns.data(), errors.data(), start, stride, tokens,
                    indices.data(), count.data(), maxima.data(), lower.data(), upper.data(), de.data());
            } else {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(adaptive::repair<false>), dim3(256u), dim3(256u), 0u, nullptr,
                    dq.data(), dt.data(), pq, pk, qflags, kflags, ns.data(), errors.data(), start, stride, tokens,
                    indices.data(), count.data(), nullptr, nullptr, nullptr, nullptr);
            }
            check(hipGetLastError());
        };
        const auto inspect_replay = [&] {
            hipLaunchKernelGGL(inspect_repaired, dim3(128u), dim3(256u), 0u, nullptr,
                es.data(), ns.data(), indices.data(), count.data(), stats.data()); finish();
            if (tile_repair) {
                selected_tiles += tile_count.download()[0];
                hipLaunchKernelGGL(inspect_tile_membership, dim3(128u), dim3(256u), 0u, nullptr,
                    indices.data(), count.data(), stride, tile_masks.data(), stats.data());
                hipLaunchKernelGGL(inspect_tile_list, dim3(128u), dim3(256u), 0u, nullptr,
                    tile_indices.data(), tile_count.data(), tile_owners.data(), tile_masks.data(), start, queries, stride, tile_capacity, stats.data()); finish();
            }
        };
        for (unsigned pass = 0u; pass < 2u; ++pass) {
            const double collect_time = timed([&] {
                check(hipMemset(count.data(), 0, 4u));
                if (pass == 0u) {
                    hipLaunchKernelGGL(route::collect_maxima, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), start, stride, indices.data(), count.data());
                } else {
                    hipLaunchKernelGGL(route::collect_probabilities, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), maxima.data(), start, stride, de.data(), indices.data(), count.data());
                }
                check(hipGetLastError());
            });
            collection_ms += collect_time; repair_pipeline_ms += collect_time;
            const double replay_time = timed([&] { repair(false); });
            replay_ms += replay_time; repair_pipeline_ms += replay_time;
            selected[pass] += count.download()[0]; inspect_replay();
        }
        const double initialization_time = timed([&] {
            hipLaunchKernelGGL(adaptive::initialize, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                ns.data(), errors.data(), maxima.data(), np.data(), nz.data(), lower.data(), upper.data(),
                maximum_cost.data(), pending.data(), start, stride, de.data());
        });
        initialization_ms += initialization_time; repair_pipeline_ms += initialization_time;
        original_probability_ms += timed([&] {
            hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                es.data(), ep.data(), ez.data(), start, stride, de.data(), true);
        });
        hipLaunchKernelGGL(inspect_probabilities, dim3(128u), dim3(256u), 0u, nullptr,
            ep.data(), np.data(), start, rows, stride, stats.data()); finish();
        // Candidate numerator uses only its own certified probabilities and
        // exact alphas. It never uses baseline scores, numerator or GB10 values.
        hipLaunchKernelGGL(blackwell_probability_value_kernel, dim3(kQueryHeads, queries), dim3(kHeadDim), 0u, nullptr,
            dv.data(), ep.data(), ez.data(), canonical.data(), start, 0u, stride, drecip.data(), nullptr, nullptr, nullptr); finish();
        canonical_pv_ms += timed([&] {
            hipLaunchKernelGGL(blackwell_probability_value_kernel, dim3(kQueryHeads, queries), dim3(kHeadDim), 0u, nullptr,
                dv.data(), np.data(), nz.data(), output.data(), start, 0u, stride, drecip.data(), accumulator.data(), nullptr, nullptr);
        });
        for (unsigned round = 0u; round < actual_rounds; ++round) {
            const double certificate_time = timed([&] {
                check(hipMemset(count.data(), 0, 4u));
                hipLaunchKernelGGL(adaptive::certify_and_collect, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                    errors.data(), lower.data(), upper.data(), accumulator.data(), nz.data(), den.data(), output.data(), maximum_cost.data(),
                    pending.data(), start, stride, drecip.data(), round, step, indices.data(), count.data(), adaptive_stats.data());
                check(hipGetLastError());
            });
            certificate_ms += certificate_time; repair_pipeline_ms += certificate_time;
            hipLaunchKernelGGL(inspect_certificate, dim3(128u), dim3(256u), 0u, nullptr,
                output.data(), canonical.data(), pending.data(), cells, round == actual_rounds - 1u, stats.data());
            hipLaunchKernelGGL(inspect_scales, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr,
                ez.data(), nz.data(), den.data(), start, rows, stride, stats.data()); finish();
            if (round == actual_rounds - 1u) break;
            selected[round + 2u] += count.download()[0];
            const double replay_time = timed([&] { repair(true); });
            replay_ms += replay_time; repair_pipeline_ms += replay_time; inspect_replay();
        }
        hipLaunchKernelGGL(inspect_final, dim3(128u), dim3(256u), 0u, nullptr,
            output.data(), canonical.data(), dr.data(), start, cells, reference_tokens, stats.data()); finish();
        const auto partial = stats.download();
        for (unsigned index : {ScoreOutside, ProbabilityDifferent, AlphaDifferent, DenominatorOutside,
             OutputDifferent, ExternalDifferent, Nonfinite, RepairDifferent, CertifiedWrong, FinalRows, TileMembershipErrors, TileListErrors}) {
            if (partial[index]) std::fprintf(stderr, "ADAPTIVE_QK_FAILURE metric=%u count=%u start=%u\n", index, partial[index], start);
            require(partial[index] == 0u, "adaptive denominator QK numerical check failed");
        }
        if (reference_tokens && ((start / batch) % 16u == 0u || start + queries == tokens)) {
            std::fprintf(stderr, "ADAPTIVE_QK_PROGRESS completed_queries=%u total_queries=%u\n", start + queries, tokens); std::fflush(stderr);
        }
    }
    const auto result = stats.download(), refinement = adaptive_stats.download();
    for (auto* b : {&es, &ns, &errors, &ez, &nz, &maxima, &canonical, &accumulator, &output, &den, &lower, &upper, &maximum_cost}) b->guards();
    for (auto* b : {&ep, &np, &dt}) b->guards();
    for (auto* b : {&indices, &count, &pending, &stats, &adaptive_stats, &packed, &tile_masks, &tile_owners, &tile_indices, &tile_count}) b->guards();
    dq.immutable(q); dk.immutable(k); dv.immutable(v); dr.immutable(reference); de.immutable(exp2); drecip.immutable(reciprocal);
    require(refinement[adaptive::rounds] == tokens * kQueryHeads, "every token/head must certify");
    uint64_t total_selected = 0u; for (auto n : selected) total_selected += n;
    require(result[RepairCells] == total_selected, "complete replay score inspection");
    require(result[TileCells] == (tile_repair ? total_selected : 0u), "complete tile membership inspection");
    const double phase_total = collection_ms + replay_ms + initialization_ms + certificate_ms;
    require(std::abs(phase_total - repair_pipeline_ms) < 0.001, "complete phase clocks");
    std::printf("{\"kind\":\"adaptive_denominator_qk_component\",\"tokens\":%u,\"query_batch\":128,\"mode\":%u,\"reference_tokens\":%u,\"variant\":%u,\"priority_step\":%u,\"certificate_rounds\":%u,"
        "\"score_cells\":%u,\"score_bound_violations\":%u,\"native_score_bit_differences\":%u,\"maximum_score_error_to_bound\":%.9g,"
        "\"cpu_dots\":%u,\"probability_cells\":%u,\"probability_bf16_differences\":%u,\"alpha_bit_differences\":%u,\"denominator_bound_violations\":%u,"
        "\"replayed_scores\":%u,\"replay_score_bit_differences\":%u,\"certified_wrong\":%u,\"final_uncertain_rows\":%u,"
        "\"output_cells\":%u,\"output_bf16_differences\":%u,\"external_bf16_differences\":%u,\"external_cells\":%u,\"nonfinite\":%u,"
        "\"original_prepared_qk_ms\":%.6f,\"original_probability_ms\":%.6f,\"preparation_ms\":%.6f,\"native_bounded_qk_ms\":%.6f,\"repair_and_probability_pipeline_ms\":%.6f,"
        "\"candidate_canonical_pv_ms\":%.6f,\"redzones_pass\":true,\"immutable_inputs\":true,"
        "\"strict_final_interval_and_row_fallback\":true,\"cached_probability_intervals\":true,"
        "\"baseline_scores_are_compute_input\":false,\"external_reference_is_compute_input\":false,"
        "\"inference_acceptance\":false,\"performance_acceptance\":false,\"certified_rows\":%u,\"fallback_rows\":%u,\"collection_ms\":%.6f,\"replay_ms\":%.6f,\"initialization_ms\":%.6f,\"certificate_ms\":%.6f,\"selected_tiles\":%llu,\"tile_cells\":%u,\"tile_membership_errors\":%u,\"tile_list_errors\":%u,\"exact_score_counts\":[",
        tokens, mode, reference_tokens, variant, step, actual_rounds, result[ScoreCells], result[ScoreOutside], result[NativeScoreDifferent], as_float(result[ScoreMaximumRatio]),
        cpu_dots, result[ProbabilityCells], result[ProbabilityDifferent], result[AlphaDifferent], result[DenominatorOutside],
        result[RepairCells], result[RepairDifferent], result[CertifiedWrong], result[FinalRows],
        result[OutputCells], result[OutputDifferent], result[ExternalDifferent], result[ExternalCells], result[Nonfinite],
        original_qk_ms, original_probability_ms, preparation_ms, native_bound_ms, repair_pipeline_ms, canonical_pv_ms,
        refinement[adaptive::rounds], refinement[adaptive::rounds + 1u], collection_ms, replay_ms, initialization_ms, certificate_ms,
        (unsigned long long)selected_tiles, result[TileCells], result[TileMembershipErrors], result[TileListErrors]);
    for (unsigned i = 0u; i < 2u + actual_rounds - 1u; ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)selected[i]);
    std::printf("],\"pending_rows_per_round\":[");
    for (unsigned i = 0u; i < actual_rounds; ++i) std::printf("%s%u", i ? "," : "", refinement[i]);
    std::printf("]}\n"); std::fflush(stdout);
}
}

int main(int argc, char** argv) try {
    const bool selftest = argc == 4 && std::string(argv[1]) == "--selftest";
    const bool product = argc == 8 && std::string(argv[1]) == "--q8192";
    require(selftest || product || argc == 7, "Q K V GB10-reference exp2 reciprocal, --q8192 followed by those files, or --selftest exp2 reciprocal");
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    require(!std::strncmp(device.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    const unsigned offset = product ? 1u : 0u;
    const auto exp2 = read<unsigned char>(argv[selftest ? 2u : 5u + offset], qrt_sm121_exp2_interpolated::table_bytes);
    const auto reciprocal = read<unsigned char>(argv[selftest ? 3u : 6u + offset], qrt_sm121_attention_rcp::table_bytes);
    require(qrt_sm121_exp2_interpolated::valid_layout(exp2.data(), exp2.size()), "exp2 layout");
    require(qrt_sm121_attention_rcp::valid_layout(reciprocal.data(), reciprocal.size()), "reciprocal layout");
    if (!selftest) {
        constexpr unsigned captured = 7169u;
        const unsigned tokens = product ? 8192u : captured;
        auto q = read<uint16_t>(argv[1u + offset], size_t(captured) * kQueryHeads * kHeadDim);
        auto k = read<uint16_t>(argv[2u + offset], size_t(captured) * kKvHeads * kHeadDim);
        auto v = read<uint16_t>(argv[3u + offset], size_t(captured) * kKvHeads * kHeadDim);
        const auto reference = read<uint16_t>(argv[4u + offset], size_t(captured) * kQueryHeads * kHeadDim);
        const auto extend = [&](std::vector<uint16_t>& values, unsigned heads) {
            const size_t width = size_t(heads) * kHeadDim;
            values.resize(size_t(tokens) * width);
            for (unsigned token = captured; token < tokens; ++token)
                std::copy_n(values.data() + size_t(token - captured) * width, width, values.data() + size_t(token) * width);
        };
        extend(q, kQueryHeads); extend(k, kKvHeads); extend(v, kKvHeads);
        for (unsigned variant = 0u; variant < 4u; ++variant)
            run(tokens, product ? 100u : 99u, q, k, v, reference, exp2, reciprocal, captured, variant);
    } else {
        for (unsigned tokens : {1u, 17u, 31u, 32u, 33u, 65u, 129u}) {
            for (unsigned mode = 0u; mode < 5u; ++mode) {
                uint32_t random = 0x716933u + tokens * 137u + mode;
                auto next = [&] { random ^= random << 13u; random ^= random >> 17u; random ^= random << 5u; return random; };
                std::vector<uint16_t> q(size_t(tokens) * kQueryHeads * kHeadDim), k(size_t(tokens) * kKvHeads * kHeadDim), v(k.size());
                for (auto* data : {&q, &k}) {
                    for (size_t i = 0u; i < data->size(); ++i) {
                        const uint32_t word = next();
                        uint16_t value = uint16_t((word & 0x807fu) | ((123u + (word >> 8u) % 7u) << 7u));
                        if (mode == 1u) value = 0u;
                        if (mode == 2u) value = uint16_t(0x3f80u | ((i & 1u) ? 0x8000u : 0u));
                        if (mode == 3u) value = uint16_t(word & 0x807fu);
                        if (mode == 4u) value = uint16_t((word & 0x807fu) | ((110u + (word >> 8u) % 35u) << 7u));
                        (*data)[i] = value;
                    }
                }
                for (auto& value : v) value = uint16_t((next() & 0x807fu) | 0x3e80u);
                for (unsigned variant = 0u; variant < 4u; ++variant)
                    run(tokens, mode, q, k, v, std::vector<uint16_t>(q.size()), exp2, reciprocal, 0u, variant);
            }
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "adaptive_denominator_qk_error=%s\n", error.what()); return 2;
}
