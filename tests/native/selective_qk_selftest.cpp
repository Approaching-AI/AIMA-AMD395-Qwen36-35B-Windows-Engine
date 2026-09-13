#include "../../native/providers/ck_fmha/selective_qk.h"
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
constexpr unsigned batch = 32u, guard = 64u;
enum Metric : unsigned {
    ScoreCells, ScoreOutside, NativeScoreDifferent, ScoreMaximumRatio,
    ProbabilityCells, ProbabilityDifferent, AlphaDifferent, DenominatorRows, DenominatorOutside,
    InitialUncertainCells, InitialRows, InitialAdmittedWrong,
    RefinedUncertainCells, RefinedRows, RefinedAdmittedWrong,
    FinalUncertainCells, FinalRows, FinalAdmittedWrong,
    OutputCells, OutputDifferent, ExternalDifferent, Nonfinite, MetricCount
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
__global__ void denominator_guard(const float* accumulator, const float* scales, const float* den,
    const float* canonical_output, float* output, unsigned* needed_rows, unsigned rows,
    unsigned tiles, const unsigned char* table, unsigned pass, unsigned* stats) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned uncertain = 0u, admitted_wrong = 0u;
    if (i < rows * kHeadDim) {
        const unsigned row = i / kHeadDim;
        const float carry = accumulator[i];
        const auto reciprocal = route::reciprocal_interval({den[size_t(row) * 2u], den[size_t(row) * 2u + 1u]}, table);
        const float a = carry * reciprocal.low, b = carry * reciprocal.high;
        const float center = carry * qrt_sm121_attention_rcp::evaluate(table, scales[size_t(row) * (tiles + 1u) + tiles]);
        const bool stable = isfinite(a) && isfinite(b) && f32_to_bf16(a) == f32_to_bf16(b);
        output[i] = center;
        if (!stable) { uncertain = 1u; atomicOr(needed_rows + row, 1u); }
        // Canonical comparison is diagnostic-only and never feeds selection.
        admitted_wrong = stable && f32_to_bf16(center) != f32_to_bf16(canonical_output[i]);
    }
    count_metric(stats, InitialUncertainCells + pass * 3u, uncertain);
    count_metric(stats, InitialAdmittedWrong + pass * 3u, admitted_wrong);
}
__global__ void inspect_rows(const unsigned* needed, unsigned rows, unsigned pass, unsigned* stats) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    count_metric(stats, InitialRows + pass * 3u, unsigned(row < rows && needed[row]));
}
__global__ void inspect_final(const float* output, const float* canonical, const uint16_t* reference,
    unsigned start, unsigned cells, bool external, unsigned* stats) {
    unsigned count = 0u, different = 0u, external_different = 0u, nonfinite = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < cells; i += gridDim.x * blockDim.x) {
        ++count; const auto actual = f32_to_bf16(output[i]), expected = f32_to_bf16(canonical[i]);
        different += actual != expected; nonfinite += !isfinite(output[i]) || !isfinite(canonical[i]);
        if (external) external_different += actual != reference[size_t(start) * kQueryHeads * kHeadDim + i]
            || expected != reference[size_t(start) * kQueryHeads * kHeadDim + i];
    }
    count_metric(stats, OutputCells, count); count_metric(stats, OutputDifferent, different);
    count_metric(stats, ExternalDifferent, external_different); count_metric(stats, Nonfinite, nonfinite);
}
float as_float(unsigned word) { float x; std::memcpy(&x, &word, 4u); return x; }
void run(unsigned tokens, unsigned mode, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& v, const std::vector<uint16_t>& reference,
    const std::vector<unsigned char>& exp2, const std::vector<unsigned char>& reciprocal, bool external) {
    Buffer<uint16_t> dq(q.size()), dk(k.size()), dv(v.size()), dt(k.size()), dr(reference.size());
    Buffer<unsigned char> de(exp2.size()), drecip(reciprocal.size());
    dq.upload(q); dk.upload(k); dv.upload(v); dr.upload(reference); de.upload(exp2); drecip.upload(reciprocal);
    const size_t score_capacity = size_t(batch) * kQueryHeads * tokens;
    const size_t scale_capacity = size_t(batch) * kQueryHeads * ((tokens + 31u) / 32u + 1u);
    const size_t output_capacity = size_t(batch) * kQueryHeads * kHeadDim;
    Buffer<float> es(score_capacity), ns(score_capacity), errors(score_capacity), ez(scale_capacity), nz(scale_capacity), maxima(scale_capacity);
    Buffer<uint16_t> ep(score_capacity), np(score_capacity);
    Buffer<float> canonical(output_capacity), accumulator(output_capacity), output(output_capacity), den(size_t(batch) * kQueryHeads * 2u);
    Buffer<unsigned> indices(score_capacity), count(1u), needed(size_t(batch) * kQueryHeads), stats(MetricCount);
    check(hipMemset(stats.data(), 0, MetricCount * 4u));
    double transpose_ms = timed([&] { check(hipError_t(transpose_keys(dk.data(), dt.data(), k.size(), tokens, nullptr))); });
    double original_qk_ms = 0.0, original_probability_ms = 0.0, native_bound_ms = 0.0, repair_pipeline_ms = 0.0, canonical_pv_ms = 0.0;
    uint64_t selected[4] = {}; unsigned cpu_dots = 0u;
    for (unsigned start = 0u; start < tokens; start += batch) {
        const unsigned queries = std::min(batch, tokens - start), stride = start + queries;
        const unsigned rows = queries * kQueryHeads, cells = rows * kHeadDim, tiles = (stride + 31u) / 32u;
        original_qk_ms += timed([&] {
            hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
                dim3((stride + 31u) / 32u, kQueryHeads, (queries + 7u) / 8u), dim3(kThreads), 0u, nullptr,
                dq.data(), dt.data(), es.data(), start, queries, stride, tokens);
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
        const auto repair = [&] {
            hipLaunchKernelGGL(route::repair_scores, dim3(256u), dim3(256u), 0u, nullptr,
                dq.data(), dk.data(), ns.data(), errors.data(), start, stride, indices.data(), count.data()); check(hipGetLastError());
        };
        for (unsigned pass = 0u; pass < 2u; ++pass) {
            repair_pipeline_ms += timed([&] {
                check(hipMemset(count.data(), 0, 4u));
                if (pass == 0u) {
                    hipLaunchKernelGGL(route::collect_maxima, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), start, stride, indices.data(), count.data());
                } else {
                    hipLaunchKernelGGL(route::collect_probabilities, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), maxima.data(), start, stride, de.data(), indices.data(), count.data());
                }
                check(hipGetLastError()); repair();
            });
            selected[pass] += count.download()[0];
        }
        const auto probabilities = [&] {
            hipLaunchKernelGGL(route::probabilities_and_denominators, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                ns.data(), errors.data(), maxima.data(), np.data(), nz.data(), den.data(), start, stride, de.data()); check(hipGetLastError());
        };
        repair_pipeline_ms += timed(probabilities);
        original_probability_ms += timed([&] {
            hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                es.data(), ep.data(), ez.data(), start, stride, de.data(), true);
        });
        hipLaunchKernelGGL(inspect_probabilities, dim3(128u), dim3(256u), 0u, nullptr,
            ep.data(), np.data(), start, rows, stride, stats.data());
        hipLaunchKernelGGL(inspect_scales, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr,
            ez.data(), nz.data(), den.data(), start, rows, stride, stats.data()); finish();
        // Each side computes its own canonical numerator. The candidate reads
        // only its repaired probabilities and maxima, never baseline scores.
        hipLaunchKernelGGL(blackwell_probability_value_kernel, dim3(kQueryHeads, queries), dim3(kHeadDim), 0u, nullptr,
            dv.data(), ep.data(), ez.data(), canonical.data(), start, 0u, stride, drecip.data(), nullptr, nullptr, nullptr); finish();
        canonical_pv_ms += timed([&] {
            hipLaunchKernelGGL(blackwell_probability_value_kernel, dim3(kQueryHeads, queries), dim3(kHeadDim), 0u, nullptr,
                dv.data(), np.data(), nz.data(), output.data(), start, 0u, stride, drecip.data(), accumulator.data(), nullptr, nullptr);
        });
        for (unsigned pass = 0u; pass < 3u; ++pass) {
            check(hipMemset(needed.data(), 0, rows * 4u));
            repair_pipeline_ms += timed([&] {
                hipLaunchKernelGGL(denominator_guard, dim3((cells + 255u) / 256u), dim3(256u), 0u, nullptr,
                    accumulator.data(), nz.data(), den.data(), canonical.data(), output.data(), needed.data(), rows,
                    tiles, drecip.data(), pass, stats.data());
            });
            hipLaunchKernelGGL(inspect_rows, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr,
                needed.data(), rows, pass, stats.data()); finish();
            if (pass == 2u) break;
            repair_pipeline_ms += timed([&] {
                check(hipMemset(count.data(), 0, 4u));
                if (pass == 0u) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(route::collect_denominator_refinement<false>), dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), maxima.data(), nz.data(), needed.data(), start, stride, de.data(), indices.data(), count.data());
                } else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(route::collect_denominator_refinement<true>), dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), maxima.data(), nz.data(), needed.data(), start, stride, de.data(), indices.data(), count.data());
                }
                check(hipGetLastError()); repair(); probabilities();
            });
            selected[pass + 2u] += count.download()[0];
        }
        hipLaunchKernelGGL(inspect_probabilities, dim3(128u), dim3(256u), 0u, nullptr,
            ep.data(), np.data(), start, rows, stride, stats.data());
        hipLaunchKernelGGL(inspect_scales, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr,
            ez.data(), nz.data(), den.data(), start, rows, stride, stats.data());
        hipLaunchKernelGGL(inspect_final, dim3(128u), dim3(256u), 0u, nullptr,
            output.data(), canonical.data(), dr.data(), start, cells, external, stats.data()); finish();
        const auto partial = stats.download();
        for (unsigned index : {ScoreOutside, ProbabilityDifferent, AlphaDifferent, DenominatorOutside,
            InitialAdmittedWrong, RefinedAdmittedWrong, FinalAdmittedWrong, FinalUncertainCells, OutputDifferent, ExternalDifferent, Nonfinite}) {
            if (partial[index]) std::fprintf(stderr, "SELECTIVE_QK_FAILURE metric=%u count=%u start=%u\n", index, partial[index], start);
            require(partial[index] == 0u, "selective QK numerical check failed");
        }
        if (external && ((start / batch) % 32u == 0u || start + queries == tokens)) {
            std::fprintf(stderr, "SELECTIVE_QK_PROGRESS completed_queries=%u total_queries=%u\n", start + queries, tokens); std::fflush(stderr);
        }
    }
    const auto result = stats.download();
    for (auto* b : {&es, &ns, &errors, &ez, &nz, &maxima, &canonical, &accumulator, &output, &den}) b->guards();
    for (auto* b : {&ep, &np, &dt}) b->guards();
    for (auto* b : {&indices, &count, &needed, &stats}) b->guards();
    dq.immutable(q); dk.immutable(k); dv.immutable(v); dr.immutable(reference); de.immutable(exp2); drecip.immutable(reciprocal);
    std::printf("{\"kind\":\"selective_qk_component\",\"tokens\":%u,\"query_batch\":32,\"mode\":%u,\"external_reference\":%s,"
        "\"score_cells\":%u,\"score_bound_violations\":%u,\"native_score_bit_differences\":%u,\"maximum_score_error_to_bound\":%.9g,"
        "\"cpu_dots\":%u,\"probability_cells\":%u,\"probability_bf16_differences\":%u,\"alpha_bit_differences\":%u,\"denominator_bound_violations\":%u,"
        "\"exact_score_counts\":[%llu,%llu,%llu,%llu],\"initial_uncertain_cells\":%u,\"initial_uncertain_heads\":%u,"
        "\"refined_uncertain_cells\":%u,\"refined_uncertain_heads\":%u,\"final_uncertain_cells\":%u,\"admitted_wrong\":%u,"
        "\"output_cells\":%u,\"output_bf16_differences\":%u,\"external_bf16_differences\":%u,\"nonfinite\":%u,"
        "\"original_qk_ms\":%.6f,\"original_probability_ms\":%.6f,\"key_transpose_ms\":%.6f,\"native_bounded_qk_ms\":%.6f,\"repair_and_probability_pipeline_ms\":%.6f,"
        "\"candidate_canonical_pv_ms\":%.6f,\"redzones_pass\":true,\"immutable_inputs\":true,"
        "\"baseline_scores_are_compute_input\":false,\"external_reference_is_compute_input\":false,"
        "\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        tokens, mode, external ? "true" : "false", result[ScoreCells], result[ScoreOutside], result[NativeScoreDifferent], as_float(result[ScoreMaximumRatio]),
        cpu_dots, result[ProbabilityCells], result[ProbabilityDifferent], result[AlphaDifferent], result[DenominatorOutside],
        (unsigned long long)selected[0], (unsigned long long)selected[1], (unsigned long long)selected[2], (unsigned long long)selected[3],
        result[InitialUncertainCells], result[InitialRows], result[RefinedUncertainCells], result[RefinedRows], result[FinalUncertainCells],
        result[InitialAdmittedWrong] + result[RefinedAdmittedWrong] + result[FinalAdmittedWrong], result[OutputCells], result[OutputDifferent], result[ExternalDifferent], result[Nonfinite],
        original_qk_ms, original_probability_ms, transpose_ms, native_bound_ms, repair_pipeline_ms, canonical_pv_ms);
    std::fflush(stdout);
}
}

int main(int argc, char** argv) try {
    const bool selftest = argc == 4 && std::string(argv[1]) == "--selftest";
    require(selftest || argc == 7, "supply Q K V GB10-reference compact-exp2 reciprocal files, or --selftest exp2 reciprocal");
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    require(!std::strncmp(device.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    const auto exp2 = read<unsigned char>(argv[selftest ? 2 : 5], qrt_sm121_exp2_interpolated::table_bytes);
    const auto reciprocal = read<unsigned char>(argv[selftest ? 3 : 6], qrt_sm121_attention_rcp::table_bytes);
    require(qrt_sm121_exp2_interpolated::valid_layout(exp2.data(), exp2.size()), "exp2 layout");
    require(qrt_sm121_attention_rcp::valid_layout(reciprocal.data(), reciprocal.size()), "reciprocal layout");
    if (!selftest) {
        constexpr unsigned tokens = 7169u;
        run(tokens, 99u, read<uint16_t>(argv[1], size_t(tokens) * kQueryHeads * kHeadDim),
            read<uint16_t>(argv[2], size_t(tokens) * kKvHeads * kHeadDim),
            read<uint16_t>(argv[3], size_t(tokens) * kKvHeads * kHeadDim),
            read<uint16_t>(argv[4], size_t(tokens) * kQueryHeads * kHeadDim), exp2, reciprocal, true);
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
                run(tokens, mode, q, k, v, std::vector<uint16_t>(q.size()), exp2, reciprocal, false);
            }
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "selective_qk_error=%s\n", error.what()); return 2;
}
