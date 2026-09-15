#include "../../native/providers/ck_fmha/coarse_selective_qk.h"
#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
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
#include <utility>

namespace {
using namespace qrt_blackwell_attention;
namespace route = qrt_selective_qk;
constexpr unsigned batch = 128u, guard = 64u;
enum Metric : unsigned {
    ScoreCells, ScoreOutside, NativeScoreDifferent, ScoreMaximumRatio,
    ProbabilityCells, ProbabilityDifferent, AlphaDifferent, DenominatorRows, DenominatorOutside,
    InitialUncertainCells, InitialRows, InitialAdmittedWrong,
    RefinedUncertainCells, RefinedRows, RefinedAdmittedWrong,
    FinalUncertainCells, FinalRows, FinalAdmittedWrong,
    OutputCells, OutputDifferent, ExternalDifferent, Nonfinite,
    ExternalCells, TailChanged, MetricCount
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
__global__ void inspect_tail(const unsigned char* data, size_t used, size_t capacity, unsigned* stats) {
    unsigned changed = 0u;
    for (size_t i = used + size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < capacity;
         i += size_t(gridDim.x) * blockDim.x) changed += data[i] != 0xa5u;
    count_metric(stats, TailChanged, changed);
}
template<class T> void reset(Buffer<T>& buffer) {
    check(hipMemset(buffer.data(), 0xa5, buffer.size * sizeof(T)));
}
template<class T> void tail(Buffer<T>& buffer, size_t used, unsigned* stats) {
    require(used <= buffer.size, "live region exceeds buffer");
    if (used < buffer.size) hipLaunchKernelGGL(inspect_tail, dim3(128u), dim3(256u), 0u, nullptr,
        reinterpret_cast<const unsigned char*>(buffer.data()), used * sizeof(T), buffer.size * sizeof(T), stats);
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
    const float* canonical_output, float* output, unsigned* needed_rows, float* row_budgets, unsigned rows,
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
        if (carry != 0.0f) {
            // Work selection only. Estimate how much denominator refinement
            // this row needs from its nearest output BF16 midpoint. Reserve
            // FP32/RCP headroom and use the tightest of all256 dimensions.
            // The later interval certificate and exact-row fallback retain
            // sole authority to admit an output; this estimate never does.
            const float magnitude = fabsf(center);
            const uint32_t rounded = uint32_t(f32_to_bf16(magnitude)) << 16u;
            float budget = 0.0f;
            if (magnitude > 0.0f && rounded < 0x7f800000u) {
                const float lower = rounded ? __uint_as_float(rounded - 0x8000u) : 0.0f;
                const float upper = __uint_as_float(rounded + 0x8000u);
                const float distance = fminf(magnitude - lower, upper - magnitude);
                const float relative = fmaxf(distance / magnitude - 0x1p-20f, 0.0f);
                budget = (scales[size_t(row) * (tiles + 1u) + tiles] * relative) * 0.5f;
            }
            atomicMin(reinterpret_cast<unsigned*>(row_budgets) + row, __float_as_uint(budget));
        }
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
    unsigned start, unsigned cells, unsigned reference_tokens, bool external, unsigned* stats) {
    unsigned count = 0u, different = 0u, external_different = 0u, nonfinite = 0u, external_cells = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < cells; i += gridDim.x * blockDim.x) {
        ++count; const auto actual = f32_to_bf16(output[i]), expected = f32_to_bf16(canonical[i]);
        different += actual != expected; nonfinite += !isfinite(output[i]) || !isfinite(canonical[i]);
        if (external && start + i / (kQueryHeads * kHeadDim) < reference_tokens) {
            ++external_cells; external_different += actual != reference[size_t(start) * kQueryHeads * kHeadDim + i]
            || expected != reference[size_t(start) * kQueryHeads * kHeadDim + i];
        }
    }
    count_metric(stats, OutputCells, count); count_metric(stats, OutputDifferent, different);
    count_metric(stats, ExternalDifferent, external_different); count_metric(stats, Nonfinite, nonfinite);
    count_metric(stats, ExternalCells, external_cells);
}
float as_float(unsigned word) { float x; std::memcpy(&x, &word, 4u); return x; }
void run(unsigned tokens, unsigned mode, unsigned chunk, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
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
    Buffer<float> canonical(output_capacity), accumulator(output_capacity), output(output_capacity), den(size_t(batch) * kQueryHeads * 2u), row_budgets(size_t(batch) * kQueryHeads);
    Buffer<unsigned> indices(score_capacity), count(1u), needed(size_t(batch) * kQueryHeads), stats(MetricCount);
    Buffer<uint32_t> pq(q.size()), pk(k.size());
    Buffer<unsigned> qflags(size_t(tokens) * kQueryHeads), kflags(size_t(tokens) * kKvHeads);
    Buffer<unsigned> qok(size_t(tokens) * kQueryHeads), kok(size_t(tokens) * kKvHeads);
    check(hipMemset(stats.data(), 0, MetricCount * 4u));
    const double selected_preparation_ms = timed([&] {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_prepared_decoded_qk::prepare<false>),
            dim3(tokens * kQueryHeads), dim3(kHeadDim), 0u, nullptr,
            dq.data(), pq.data(), qflags.data(), nullptr, tokens);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_prepared_decoded_qk::prepare<true>),
            dim3(tokens * kKvHeads), dim3(kHeadDim), 0u, nullptr,
            dk.data(), pk.data(), kflags.data(), dt.data(), tokens);
    });
    const double transpose_ms = timed([&] { check(hipError_t(transpose_keys(dk.data(), dt.data(), k.size(), tokens, nullptr))); });
    const double eligibility_ms = timed([&] {
        hipLaunchKernelGGL(qrt_sm121_coarse_projection_matrix::eligibility,
            dim3(tokens * kQueryHeads), dim3(256u), 0u, nullptr, dq.data(), qok.data(), tokens * kQueryHeads, kHeadDim);
        hipLaunchKernelGGL(qrt_sm121_coarse_projection_matrix::eligibility,
            dim3(tokens * kKvHeads), dim3(256u), 0u, nullptr, dk.data(), kok.data(), tokens * kKvHeads, kHeadDim);
    });
    const auto original_qok = qok.download(), original_kok = kok.download();
    for (const auto pair : {std::make_pair(&q, &original_qok), std::make_pair(&k, &original_kok)}) {
        for (size_t row = 0u; row < pair.second->size(); ++row) {
            unsigned valid = 1u;
            for (unsigned i = 0u; i < kHeadDim; ++i) {
                const uint16_t value = (*pair.first)[row * kHeadDim + i];
                const unsigned exponent = (value >> 7u) & 255u;
                valid &= unsigned(!(value & 0x7fffu) || (exponent >= 80u && exponent <= 174u));
            }
            require((*pair.second)[row] == valid, "coarse eligibility differs from CPU row check");
        }
    }
    const auto original_pq = pq.download(), original_pk = pk.download();
    const auto original_qflags = qflags.download(), original_kflags = kflags.download();
    const auto original_dt = dt.download();
    double original_qk_ms = 0.0, original_probability_ms = 0.0, native_bound_ms = 0.0, repair_pipeline_ms = 0.0, canonical_pv_ms = 0.0;
    uint64_t selected[4] = {}; unsigned cpu_dots = 0u;
    for (unsigned start = 0u; start < tokens; start += batch) {
        const unsigned queries = std::min(batch, tokens - start), stride = start + queries;
        const unsigned rows = queries * kQueryHeads, cells = rows * kHeadDim, tiles = (stride + 31u) / 32u;
        for (auto* b : {&es, &ns, &errors, &ez, &nz, &maxima, &canonical, &accumulator, &output, &den}) reset(*b);
        reset(ep); reset(np);
        original_qk_ms += timed([&] {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_prepared_decoded_qk::scores<128u, true>),
                dim3((stride + 15u) / 16u, kQueryHeads, (queries + 15u) / 16u), dim3(kThreads), 0u, nullptr,
                dq.data(), dt.data(), pq.data(), pk.data(), qflags.data(), kflags.data(),
                es.data(), start, queries, stride, tokens);
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
            const dim3 grid((stride + kIntegerMatrixColumns - 1u) / kIntegerMatrixColumns,
                kQueryHeads, (queries + 15u) / 16u);
            if (!chunk) hipLaunchKernelGGL(route::native_scores, grid, dim3(kThreads), 0u, nullptr,
                dq.data(), dt.data(), ns.data(), errors.data(), start, queries, stride, tokens);
#define COARSE_SELECTIVE_CASE(c) if (chunk == c) hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_coarse_selective_qk::native_scores<c>), grid, dim3(kThreads), 0u, nullptr, dq.data(), dt.data(), qok.data(), kok.data(), ns.data(), errors.data(), start, queries, stride, tokens)
            COARSE_SELECTIVE_CASE(64u); COARSE_SELECTIVE_CASE(128u); COARSE_SELECTIVE_CASE(256u);
#undef COARSE_SELECTIVE_CASE
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
            check(hipMemset(row_budgets.data(), 0x7f, rows * 4u));
            repair_pipeline_ms += timed([&] {
                hipLaunchKernelGGL(denominator_guard, dim3((cells + 255u) / 256u), dim3(256u), 0u, nullptr,
                    accumulator.data(), nz.data(), den.data(), canonical.data(), output.data(), needed.data(), row_budgets.data(), rows,
                    tiles, drecip.data(), pass, stats.data());
            });
            hipLaunchKernelGGL(inspect_rows, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr,
                needed.data(), rows, pass, stats.data()); finish();
            if (pass == 2u) break;
            repair_pipeline_ms += timed([&] {
                check(hipMemset(count.data(), 0, 4u));
                if (pass == 0u) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(route::collect_denominator_refinement<false>), dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), maxima.data(), nz.data(), needed.data(), row_budgets.data(), start, stride, de.data(), indices.data(), count.data());
                } else {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(route::collect_denominator_refinement<true>), dim3(kQueryHeads, queries), dim3(32u), 0u, nullptr,
                        ns.data(), errors.data(), maxima.data(), nz.data(), needed.data(), row_budgets.data(), start, stride, de.data(), indices.data(), count.data());
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
            output.data(), canonical.data(), dr.data(), start, cells, unsigned(reference.size() / (kQueryHeads * kHeadDim)), external, stats.data()); finish();
        for (auto* b : {&es, &ns, &errors}) tail(*b, size_t(rows) * stride, stats.data());
        for (auto* b : {&ep, &np}) tail(*b, size_t(rows) * stride, stats.data());
        for (auto* b : {&ez, &nz}) tail(*b, size_t(rows) * (tiles + 1u), stats.data());
        tail(maxima, size_t(rows) * tiles, stats.data());
        for (auto* b : {&canonical, &accumulator, &output}) tail(*b, cells, stats.data());
        tail(den, size_t(rows) * 2u, stats.data()); finish();
        const auto partial = stats.download();
        for (unsigned index : {ScoreOutside, ProbabilityDifferent, AlphaDifferent, DenominatorOutside,
            InitialAdmittedWrong, RefinedAdmittedWrong, FinalAdmittedWrong, FinalUncertainCells, OutputDifferent, ExternalDifferent, Nonfinite, TailChanged}) {
            if (partial[index]) std::fprintf(stderr, "SELECTIVE_QK_FAILURE metric=%u count=%u start=%u\n", index, partial[index], start);
            require(partial[index] == 0u, "selective QK numerical check failed");
        }
        if (external && ((start / batch) % 32u == 0u || start + queries == tokens)) {
            std::fprintf(stderr, "SELECTIVE_QK_PROGRESS completed_queries=%u total_queries=%u\n", start + queries, tokens); std::fflush(stderr);
        }
    }
    const auto result = stats.download();
    for (auto* b : {&es, &ns, &errors, &ez, &nz, &maxima, &canonical, &accumulator, &output, &den, &row_budgets}) b->guards();
    for (auto* b : {&ep, &np, &dt}) b->guards();
    for (auto* b : {&indices, &count, &needed, &stats}) b->guards();
    pq.immutable(original_pq); pk.immutable(original_pk); qflags.immutable(original_qflags); kflags.immutable(original_kflags);
    qok.immutable(original_qok); kok.immutable(original_kok); dt.immutable(original_dt);
    dq.immutable(q); dk.immutable(k); dv.immutable(v); dr.immutable(reference); de.immutable(exp2); drecip.immutable(reciprocal);
    std::printf("{\"kind\":\"coarse_selective_qk_component\",\"tokens\":%u,\"query_batch\":128,\"mode\":%u,\"coarse_chunk\":%u,\"external_reference\":%s,"
        "\"score_cells\":%u,\"score_bound_violations\":%u,\"native_score_bit_differences\":%u,\"maximum_score_error_to_bound\":%.9g,"
        "\"cpu_dots\":%u,\"probability_cells\":%u,\"probability_bf16_differences\":%u,\"alpha_bit_differences\":%u,\"denominator_bound_violations\":%u,"
        "\"exact_score_counts\":[%llu,%llu,%llu,%llu],\"initial_uncertain_cells\":%u,\"initial_uncertain_heads\":%u,"
        "\"refined_uncertain_cells\":%u,\"refined_uncertain_heads\":%u,\"final_uncertain_cells\":%u,\"admitted_wrong\":%u,"
        "\"output_cells\":%u,\"output_bf16_differences\":%u,\"external_bf16_differences\":%u,\"nonfinite\":%u,"
        "\"original_qk_ms\":%.6f,\"original_probability_ms\":%.6f,\"key_transpose_ms\":%.6f,\"native_bounded_qk_ms\":%.6f,\"repair_and_probability_pipeline_ms\":%.6f,"
        "\"candidate_canonical_pv_ms\":%.6f,\"redzones_pass\":true,\"immutable_inputs\":true,"
        "\"external_compared_cells\":%u,\"selected_preparation_ms\":%.6f,\"coarse_eligibility_ms\":%.6f,\"all_eligibility_rows_cpu_checked\":true,\"prepared_inputs_immutable\":true,\"unused_slab_tails_pass\":true,"
        "\"denominator_refinement_budget\":\"output_BF16_margin_work_threshold\",\"strict_final_interval_and_row_fallback\":true,"
        "\"baseline_scores_are_compute_input\":false,\"external_reference_is_compute_input\":false,\"real_model_prompt\":false,\"hardware_error_bound_proven\":false,"
        "\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        tokens, mode, chunk, external ? "true" : "false", result[ScoreCells], result[ScoreOutside], result[NativeScoreDifferent], as_float(result[ScoreMaximumRatio]),
        cpu_dots, result[ProbabilityCells], result[ProbabilityDifferent], result[AlphaDifferent], result[DenominatorOutside],
        (unsigned long long)selected[0], (unsigned long long)selected[1], (unsigned long long)selected[2], (unsigned long long)selected[3],
        result[InitialUncertainCells], result[InitialRows], result[RefinedUncertainCells], result[RefinedRows], result[FinalUncertainCells],
        result[InitialAdmittedWrong] + result[RefinedAdmittedWrong] + result[FinalAdmittedWrong], result[OutputCells], result[OutputDifferent], result[ExternalDifferent], result[Nonfinite],
        original_qk_ms, original_probability_ms, transpose_ms, native_bound_ms, repair_pipeline_ms, canonical_pv_ms,
        result[ExternalCells], selected_preparation_ms, eligibility_ms);
    std::fflush(stdout);
}
}

int main(int argc, char** argv) try {
    const bool selftest = argc == 5 && std::string(argv[1]) == "--selftest";
    require(selftest || argc == 9, "supply --q7169/--q8192 Q K V GB10-reference exp2 reciprocal chunk, or --selftest exp2 reciprocal chunk");
    const unsigned chunk = unsigned(std::stoul(argv[argc - 1]));
    require(chunk == 0u || chunk == 64u || chunk == 128u || chunk == 256u, "unsupported coarse chunk");
    require(selftest || std::string(argv[1]) == "--q7169" || std::string(argv[1]) == "--q8192", "unsupported capture shape");
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    require(!std::strncmp(device.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    const auto exp2 = read<unsigned char>(argv[selftest ? 2 : 6], qrt_sm121_exp2_interpolated::table_bytes);
    const auto reciprocal = read<unsigned char>(argv[selftest ? 3 : 7], qrt_sm121_attention_rcp::table_bytes);
    require(qrt_sm121_exp2_interpolated::valid_layout(exp2.data(), exp2.size()), "exp2 layout");
    require(qrt_sm121_attention_rcp::valid_layout(reciprocal.data(), reciprocal.size()), "reciprocal layout");
    if (!selftest) {
        const unsigned tokens = std::string(argv[1]) == "--q8192" ? 8192u : 7169u;
        const auto extend = [&](unsigned argument, unsigned heads) {
            auto values = read<uint16_t>(argv[argument], size_t(7169u) * heads * kHeadDim);
            values.reserve(size_t(tokens) * heads * kHeadDim);
            for (size_t i = values.size(); i < size_t(tokens) * heads * kHeadDim; ++i)
                values.push_back(values[i - size_t(7169u) * heads * kHeadDim]);
            return values;
        };
        run(tokens, 99u, chunk, extend(2u, kQueryHeads), extend(3u, kKvHeads), extend(4u, kKvHeads),
            read<uint16_t>(argv[5], size_t(7169u) * kQueryHeads * kHeadDim), exp2, reciprocal, true);
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
                run(tokens, mode, chunk, q, k, v, std::vector<uint16_t>(q.size()), exp2, reciprocal, false);
            }
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "selective_qk_error=%s\n", error.what()); return 2;
}
