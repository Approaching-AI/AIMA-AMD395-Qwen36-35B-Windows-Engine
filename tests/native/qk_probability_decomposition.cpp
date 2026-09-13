#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

// Numerical decomposition only. Exact QK is deliberately used to construct
// diagnostic hybrids. The external GB10 BF16 file is comparison-only. Nothing
// here establishes a route that avoids exact QK, or a product performance gain.
namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned tokens = 7169u, batch = 32u, guard = 64u;
constexpr uint32_t sentinel = 0xa5a5a5a5u;
enum Metric : unsigned {
    ScoreCells, ScoreDifferent, ScoreNonfinite, ScoreMaximumError,
    ProbabilityCells, NativeProbabilityDifferent, CenteredProbabilityDifferent,
    AlphaCells, NativeAlphaDifferent, CenteredAlphaDifferent,
    DenominatorRows, NativeDenominatorDifferent, CenteredDenominatorDifferent,
    NativeDenominatorMaximumRelative, CenteredDenominatorMaximumRelative, CenteredScoreClamps,
    BaseOutput = 16u, VariantFields = 3u, VariantCount = 6u,
    MetricCount = BaseOutput + VariantFields * VariantCount
};
const char* names[VariantCount] = {"canonical", "native_qk", "centered_capped_native_qk",
    "centered_probability_only", "centered_denominator_only", "native_alpha_only"};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void finish() {
    hipEvent_t event{}; check(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        check(status == hipErrorNotReady ? hipSuccess : status);
        require(std::chrono::steady_clock::now() < deadline, "decomposition completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* allocation = nullptr;
    size_t size;
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
        std::vector<T> values(size);
        check(hipMemcpy(values.data(), data(), size * sizeof(T), hipMemcpyDeviceToHost));
        return values;
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
        require(values.size() == actual.size() && !std::memcmp(actual.data(), values.data(), size * sizeof(T)), "input changed");
        guards();
    }
};
template<class T> std::vector<T> read(const char* path, size_t n) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file && file.tellg() == std::streamoff(n * sizeof(T)), "input file length");
    std::vector<T> out(n); file.seekg(0);
    require(bool(file.read(reinterpret_cast<char*>(out.data()), n * sizeof(T))), "input file read");
    return out;
}
__device__ void count_metric(unsigned* stats, unsigned index, unsigned count) {
    for (unsigned offset = 16u; offset; offset >>= 1u) count += __shfl_down(count, offset, 32u);
    if ((threadIdx.x % 32u) == 0u && count) atomicAdd(stats + index, count);
}
__device__ void maximum_metric(unsigned* stats, unsigned index, float value) {
    for (unsigned offset = 16u; offset; offset >>= 1u) value = fmaxf(value, __shfl_down(value, offset, 32u));
    if ((threadIdx.x % 32u) == 0u) atomicMax(stats + index, __float_as_uint(value));
}
__global__ void inspect_scores(const float* original, const float* native,
    unsigned start, unsigned queries, unsigned stride, unsigned* stats) {
    unsigned cells = 0u, different = 0u, nonfinite = 0u; float maximum = 0.0f;
    const size_t total = size_t(queries) * kQueryHeads * stride;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        const unsigned row = unsigned(i / stride), key = unsigned(i % stride);
        if (key > start + row / kQueryHeads) continue;
        const float a = original[i], b = native[i]; ++cells;
        different += __float_as_uint(a) != __float_as_uint(b);
        if (!isfinite(a) || !isfinite(b)) ++nonfinite;
        else maximum = fmaxf(maximum, fabsf(a - b));
    }
    count_metric(stats, ScoreCells, cells); count_metric(stats, ScoreDifferent, different);
    count_metric(stats, ScoreNonfinite, nonfinite); maximum_metric(stats, ScoreMaximumError, maximum);
}
__global__ void centered_probability(const float* exact_scores, const float* native_scores,
    uint16_t* probabilities, float* scales, unsigned start, unsigned stride,
    const unsigned char* exp2_table, unsigned* stats) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned live_tokens = start + blockIdx.y + 1u;
    const unsigned tile_stride = (stride + 31u) / 32u;
    float running_max = -INFINITY, running_sum = 1.0f;
    unsigned clamps = 0u;
    for (unsigned tile = 0u; tile < (live_tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane;
        const float exact = key < live_tokens ? exact_scores[size_t(row) * stride + key] : -INFINITY;
        float maximum = fmaxf(running_max, exact);
        for (unsigned mask = 16u; mask; mask >>= 1u) maximum = fmaxf(maximum, __shfl_xor(maximum, mask, 32u));
        const float alpha = blackwell_attention_exp(running_max - maximum, exp2_table);
        // The actual canonical score cannot exceed this exact prefix maximum.
        // Project a native overestimate onto that valid score range before
        // evaluating the nonpositive exp2 table. This is part of the diagnostic
        // hybrid, not a new exp2 approximation or a runtime arithmetic change.
        const float raw = key < live_tokens ? native_scores[size_t(row) * stride + key] : -INFINITY;
        clamps += unsigned(key < live_tokens && raw > maximum);
        const float probability = key < live_tokens
            ? blackwell_attention_exp(fminf(raw, maximum) - maximum, exp2_table) : 0.0f;
        if (key < stride) probabilities[size_t(row) * stride + key] = f32_to_bf16(probability);
        float sum = probability;
        constexpr unsigned order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
        for (unsigned step = 0u; step < 5u; ++step) sum += __shfl_xor(sum, order[step], 32u);
        running_sum = running_sum * alpha + sum; running_max = maximum;
        if (lane == 0u) scales[size_t(row) * (tile_stride + 1u) + tile] = alpha;
    }
    if (lane == 0u) scales[size_t(row) * (tile_stride + 1u) + tile_stride] = running_sum;
    count_metric(stats, CenteredScoreClamps, clamps);
}
__global__ void inspect_probabilities(const uint16_t* exact, const uint16_t* native,
    const uint16_t* centered, unsigned start, unsigned queries, unsigned stride, unsigned* stats) {
    unsigned cells = 0u, n = 0u, c = 0u;
    const size_t total = size_t(queries) * kQueryHeads * stride;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
        if (i % stride > start + unsigned(i / stride) / kQueryHeads) continue;
        ++cells; n += exact[i] != native[i]; c += exact[i] != centered[i];
    }
    count_metric(stats, ProbabilityCells, cells); count_metric(stats, NativeProbabilityDifferent, n);
    count_metric(stats, CenteredProbabilityDifferent, c);
}
__global__ void inspect_scales(const float* exact, const float* native, const float* centered,
    unsigned start, unsigned rows, unsigned stride, unsigned* stats) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x, tiles = (stride + 31u) / 32u;
    unsigned count = 0u, n = 0u, c = 0u, dn = 0u, dc = 0u; float en = 0.0f, ec = 0.0f;
    if (row < rows) {
        const size_t base = size_t(row) * (tiles + 1u);
        count = (start + row / kQueryHeads + 32u) / 32u;
        for (unsigned tile = 0u; tile < count; ++tile) {
            n += __float_as_uint(exact[base + tile]) != __float_as_uint(native[base + tile]);
            c += __float_as_uint(exact[base + tile]) != __float_as_uint(centered[base + tile]);
        }
        const float d = exact[base + tiles];
        dn = __float_as_uint(d) != __float_as_uint(native[base + tiles]);
        dc = __float_as_uint(d) != __float_as_uint(centered[base + tiles]);
        en = fabsf(native[base + tiles] - d) / fabsf(d);
        ec = fabsf(centered[base + tiles] - d) / fabsf(d);
    }
    count_metric(stats, AlphaCells, count); count_metric(stats, NativeAlphaDifferent, n);
    count_metric(stats, CenteredAlphaDifferent, c); count_metric(stats, DenominatorRows, unsigned(row < rows));
    count_metric(stats, NativeDenominatorDifferent, dn); count_metric(stats, CenteredDenominatorDifferent, dc);
    maximum_metric(stats, NativeDenominatorMaximumRelative, en); maximum_metric(stats, CenteredDenominatorMaximumRelative, ec);
}
__global__ void replace_denominator(float* scales, const float* canonical, unsigned rows, unsigned tiles) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) scales[size_t(row) * (tiles + 1u) + tiles] = canonical[size_t(row) * (tiles + 1u) + tiles];
}
__global__ void denominator_only(const float* accumulator, const float* scales, float* output,
    unsigned cells, unsigned tiles, const unsigned char* rcp) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cells) output[i] = accumulator[i] * qrt_sm121_attention_rcp::evaluate(rcp,
        scales[size_t(i / kHeadDim) * (tiles + 1u) + tiles]);
}
__global__ void inspect_output(const float* output, const float* canonical, const uint16_t* reference,
    unsigned start, unsigned cells, unsigned variant, unsigned* stats, unsigned* affected) {
    unsigned different = 0u, nonfinite = 0u; float maximum = 0.0f;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < cells; i += gridDim.x * blockDim.x) {
        const float x = output[i]; const size_t global = size_t(start) * kQueryHeads * kHeadDim + i;
        if (f32_to_bf16(x) != reference[global]) {
            ++different; atomicOr(affected + global / kHeadDim, 1u << variant);
        }
        if (!isfinite(x)) ++nonfinite;
        else maximum = fmaxf(maximum, fabsf(x - canonical[i]));
    }
    count_metric(stats, BaseOutput + variant * VariantFields, different);
    count_metric(stats, BaseOutput + variant * VariantFields + 1u, nonfinite);
    maximum_metric(stats, BaseOutput + variant * VariantFields + 2u, maximum);
}
void pv(const uint16_t* v, const uint16_t* p, const float* s, float* output, unsigned start,
    unsigned count, unsigned stride, const unsigned char* rcp, float* accumulator = nullptr) {
    hipLaunchKernelGGL(blackwell_probability_value_kernel, dim3(kQueryHeads, count), dim3(kHeadDim), 0u, nullptr,
        v, p, s, output, start, 0u, stride, rcp, accumulator, nullptr, nullptr);
    check(hipGetLastError()); finish();
}
float as_float(unsigned word) { float x; std::memcpy(&x, &word, 4u); return x; }
}

int main(int argc, char** argv) try {
    require(argc == 7, "supply Q K V GB10-reference compact-exp2 reciprocal files");
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    require(!std::strncmp(device.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    const auto q = read<uint16_t>(argv[1], size_t(tokens) * kQueryHeads * kHeadDim);
    const auto k = read<uint16_t>(argv[2], size_t(tokens) * kKvHeads * kHeadDim);
    const auto v = read<uint16_t>(argv[3], k.size());
    const auto reference = read<uint16_t>(argv[4], q.size());
    const auto exp2 = read<unsigned char>(argv[5], 38909480u);
    const auto reciprocal = read<unsigned char>(argv[6], qrt_sm121_attention_rcp::table_bytes);
    require(qrt_sm121_attention_rcp::valid_layout(reciprocal.data(), reciprocal.size()), "reciprocal layout");
    Buffer<uint16_t> dq(q.size()), dk(k.size()), dv(v.size()), dt(k.size()), dr(reference.size());
    Buffer<unsigned char> de(exp2.size()), drecip(reciprocal.size());
    dq.upload(q); dk.upload(k); dv.upload(v); dr.upload(reference); de.upload(exp2); drecip.upload(reciprocal);
    const size_t score_capacity = size_t(batch) * kQueryHeads * tokens;
    const size_t scale_capacity = size_t(batch) * kQueryHeads * ((tokens + 31u) / 32u + 1u);
    const size_t output_capacity = size_t(batch) * kQueryHeads * kHeadDim;
    Buffer<float> es(score_capacity), ns(score_capacity), ez(scale_capacity), nz(scale_capacity), cz(scale_capacity), work(scale_capacity);
    Buffer<uint16_t> ep(score_capacity), np(score_capacity), cp(score_capacity);
    Buffer<float> canonical(output_capacity), accumulator(output_capacity), candidate(output_capacity);
    Buffer<unsigned> stats(MetricCount), affected(size_t(tokens) * kQueryHeads);
    check(hipMemset(stats.data(), 0, MetricCount * 4u)); check(hipMemset(affected.data(), 0, affected.size * 4u));
    check(hipError_t(transpose_keys(dk.data(), dt.data(), k.size(), tokens, nullptr))); finish();
    for (unsigned start = 0u; start < tokens; start += batch) {
        const unsigned count = std::min(batch, tokens - start), stride = start + count;
        const unsigned rows = count * kQueryHeads, cells = rows * kHeadDim, tiles = (stride + 31u) / 32u;
        hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
            dim3((stride + 31u) / 32u, kQueryHeads, (count + 7u) / 8u), dim3(kThreads), 0u, nullptr,
            dq.data(), dt.data(), es.data(), start, count, stride, tokens);
        check(hipGetLastError()); finish();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_scores_kernel<true>),
            dim3((stride + kIntegerMatrixColumns - 1u) / kIntegerMatrixColumns, kQueryHeads, (count + 15u) / 16u),
            dim3(kThreads), 0u, nullptr, dq.data(), dt.data(), ns.data(), start, count, stride, tokens, nullptr, nullptr);
        check(hipGetLastError()); finish();
        hipLaunchKernelGGL(inspect_scores, dim3(128u), dim3(256u), 0u, nullptr, es.data(), ns.data(), start, count, stride, stats.data());
        check(hipGetLastError());
        hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(kQueryHeads, count), dim3(32u), 0u, nullptr,
            es.data(), ep.data(), ez.data(), start, stride, de.data(), true);
        check(hipGetLastError());
        hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(kQueryHeads, count), dim3(32u), 0u, nullptr,
            ns.data(), np.data(), nz.data(), start, stride, de.data(), true);
        check(hipGetLastError());
        hipLaunchKernelGGL(centered_probability, dim3(kQueryHeads, count), dim3(32u), 0u, nullptr,
            es.data(), ns.data(), cp.data(), cz.data(), start, stride, de.data(), stats.data());
        check(hipGetLastError()); finish();
        hipLaunchKernelGGL(inspect_probabilities, dim3(128u), dim3(256u), 0u, nullptr,
            ep.data(), np.data(), cp.data(), start, count, stride, stats.data()); check(hipGetLastError());
        hipLaunchKernelGGL(inspect_scales, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr,
            ez.data(), nz.data(), cz.data(), start, rows, stride, stats.data()); check(hipGetLastError()); finish();
        for (unsigned variant = 0u; variant < VariantCount; ++variant) {
            float* output = variant ? candidate.data() : canonical.data();
            if (variant == 0u) pv(dv.data(), ep.data(), ez.data(), output, start, count, stride, drecip.data(), accumulator.data());
            if (variant == 1u) pv(dv.data(), np.data(), nz.data(), output, start, count, stride, drecip.data());
            if (variant == 2u) pv(dv.data(), cp.data(), cz.data(), output, start, count, stride, drecip.data());
            if (variant == 3u || variant == 5u) {
                check(hipMemcpy(work.data(), variant == 3u ? cz.data() : nz.data(), size_t(rows) * (tiles + 1u) * 4u, hipMemcpyDeviceToDevice));
                hipLaunchKernelGGL(replace_denominator, dim3((rows + 255u) / 256u), dim3(256u), 0u, nullptr, work.data(), ez.data(), rows, tiles);
                check(hipGetLastError());
                pv(dv.data(), variant == 3u ? cp.data() : ep.data(), work.data(), output, start, count, stride, drecip.data());
            }
            if (variant == 4u) {
                hipLaunchKernelGGL(denominator_only, dim3((cells + 255u) / 256u), dim3(256u), 0u, nullptr,
                    accumulator.data(), cz.data(), output, cells, tiles, drecip.data()); check(hipGetLastError()); finish();
            }
            hipLaunchKernelGGL(inspect_output, dim3(128u), dim3(256u), 0u, nullptr,
                output, canonical.data(), dr.data(), start, cells, variant, stats.data(), affected.data());
            check(hipGetLastError()); finish();
        }
        if ((start / batch) % 32u == 0u || start + count == tokens) {
            const auto partial = stats.download();
            require(partial[BaseOutput] == 0u && partial[BaseOutput + 1u] == 0u, "canonical output failed external GB10 boundary");
            require(partial[CenteredAlphaDifferent] == 0u, "centered diagnostic changed canonical alpha");
            for (unsigned variant = 0u; variant < VariantCount; ++variant)
                require(partial[BaseOutput + variant * VariantFields + 1u] == 0u, "diagnostic produced nonfinite output");
            std::fprintf(stderr, "QK_DECOMPOSITION_PROGRESS completed_queries=%u total_queries=%u\n", start + count, tokens);
            std::fflush(stderr);
        }
    }
    const auto result = stats.download(), heads = affected.download();
    require(result[BaseOutput] == 0u && result[BaseOutput + 1u] == 0u && result[ScoreNonfinite] == 0u && result[CenteredAlphaDifferent] == 0u, "baseline or recentering failed");
    require(std::isfinite(as_float(result[NativeDenominatorMaximumRelative])) &&
        std::isfinite(as_float(result[CenteredDenominatorMaximumRelative])), "nonfinite denominator metric");
    for (unsigned variant = 0u; variant < VariantCount; ++variant)
        require(result[BaseOutput + variant * VariantFields + 1u] == 0u &&
            std::isfinite(as_float(result[BaseOutput + variant * VariantFields + 2u])), "nonfinite output or metric");
    for (auto* b : {&es, &ns, &ez, &nz, &cz, &work, &canonical, &accumulator, &candidate}) b->guards();
    for (auto* b : {&ep, &np, &cp, &dt}) b->guards(); stats.guards(); affected.guards();
    dq.immutable(q); dk.immutable(k); dv.immutable(v); dr.immutable(reference); de.immutable(exp2); drecip.immutable(reciprocal);
    std::printf("{\"kind\":\"qk_probability_decomposition\",\"tokens\":7169,\"query_batch\":32,\"score_cells\":%u,\"score_bit_differences\":%u,\"score_maximum_absolute_error\":%.9g,\"probability_cells\":%u,\"native_probability_bf16_differences\":%u,\"centered_probability_bf16_differences\":%u,\"centered_native_scores_capped\":%u,\"alpha_cells\":%u,\"native_alpha_bit_differences\":%u,\"centered_alpha_bit_differences\":%u,\"denominator_rows\":%u,\"native_denominator_bit_differences\":%u,\"centered_denominator_bit_differences\":%u,\"native_denominator_maximum_relative_error\":%.9g,\"centered_denominator_maximum_relative_error\":%.9g,\"redzones_pass\":true,\"immutable_inputs\":true,\"external_reference_is_compute_input\":false,\"exact_qk_constructs_diagnostic_hybrids\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        result[ScoreCells], result[ScoreDifferent], as_float(result[ScoreMaximumError]), result[ProbabilityCells],
        result[NativeProbabilityDifferent], result[CenteredProbabilityDifferent], result[CenteredScoreClamps], result[AlphaCells], result[NativeAlphaDifferent],
        result[CenteredAlphaDifferent], result[DenominatorRows], result[NativeDenominatorDifferent], result[CenteredDenominatorDifferent],
        as_float(result[NativeDenominatorMaximumRelative]), as_float(result[CenteredDenominatorMaximumRelative]));
    for (unsigned variant = 0u; variant < VariantCount; ++variant) {
        unsigned changed_heads = 0u, changed_tokens = 0u;
        for (unsigned token = 0u; token < tokens; ++token) {
            bool any = false;
            for (unsigned head = 0u; head < kQueryHeads; ++head) {
                const bool changed = (heads[size_t(token) * kQueryHeads + head] & (1u << variant)) != 0u;
                changed_heads += unsigned(changed); any = any || changed;
            }
            changed_tokens += unsigned(any);
        }
        std::printf("{\"kind\":\"qk_probability_decomposition_output\",\"variant\":\"%s\",\"elements\":%zu,\"bf16_differences\":%u,\"nonfinite\":%u,\"maximum_float_error_from_canonical\":%.9g,\"affected_heads\":%u,\"affected_tokens\":%u,\"inference_acceptance\":false}\n",
            names[variant], reference.size(), result[BaseOutput + variant * VariantFields], result[BaseOutput + variant * VariantFields + 1u],
            as_float(result[BaseOutput + variant * VariantFields + 2u]), changed_heads, changed_tokens);
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "qk_probability_decomposition_error=%s\n", e.what()); return 2; }
