#ifndef QRT_SM121_BOUNDED_PROJECTION_H
#define QRT_SM121_BOUNDED_PROJECTION_H
#include "sm121_scalar_projection.h"
#include "sm121_float_row_bounds.h"

namespace qrt_sm121_bounded_projection {
namespace bounds = qrt_sm121_float_row_bounds;
using Value = qrt_q1_moe_hawkeye::Value;
constexpr unsigned threads = 256u, lanes = 4u;

__global__ void prepare_kernel(const uint16_t* input, uint32_t* output, unsigned groups) {
    const unsigned group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    uint16_t row[16];
    __builtin_memcpy(row, input + size_t(group) * 16u, sizeof(row));
    output[group] = bounds::prepare(row);
}

template<bool Audit = false>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    const uint32_t* left_bounds, const uint32_t* right_bounds, unsigned width,
    uint32_t* trace = nullptr, uint32_t* stats = nullptr) {
    const unsigned lane = threadIdx.x & 3u;
    Value carry{0u, -133, false};
    unsigned counts[4]{};
#pragma unroll 1
    for (unsigned base = 0u; base < width; base += 16u) {
        uint64_t a, b;
        __builtin_memcpy(&a, left + base + lane * 4u, sizeof(a));
        __builtin_memcpy(&b, right + base + lane * 4u, sizeof(b));
        const uint32_t lb = left_bounds[base / 16u], rb = right_bounds[base / 16u];
        int maximum;
        if (bounds::exponent(carry, lb, rb, &maximum)) {
            if constexpr (Audit) ++counts[maximum == carry.exponent ? 0u : 1u];
            const float scale = qrt_sm121_float_alignment::from_bits(uint32_t(152 - maximum) << 23u);
            uint32_t modulo = 0u;
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
                const float product = qrt_sm121_float_alignment::from_bits(uint32_t(x) << 16u) *
                    qrt_sm121_float_alignment::from_bits(uint32_t(y) << 16u);
                modulo += uint32_t(int32_t(product * scale));
            }
            modulo = qrt_sm121_lane_reduce::sum<lanes>(modulo);
            const unsigned shift = unsigned(maximum - carry.exponent);
            const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
            modulo += carry.negative ? 0u - aligned : aligned;
            const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, ((a ^ b) & 0x8000u) != 0u);
            carry = qrt_sm121_wave16::normalize(sum.magnitude, sum.negative, maximum);
        } else {
            const bool eligible = (lb & rb & bounds::valid_bit) != 0u;
            if constexpr (Audit) ++counts[eligible ? 2u : 3u];
            qrt_sm121_float_subgroup::Product products[4];
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                const uint16_t x = uint16_t(a >> (i * 16u)), y = uint16_t(b >> (i * 16u));
                const uint32_t original = uint32_t(x) | (uint32_t(y) << 16u);
                if (!eligible) products[i] = {0.0f, original, 512};
                else {
                    const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
                    products[i] = {
                        qrt_sm121_float_alignment::from_bits(uint32_t(x) << 16u) *
                        qrt_sm121_float_alignment::from_bits(uint32_t(y) << 16u), original,
                        zero ? -133 : int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254};
                }
            }
            carry = qrt_sm121_float_subgroup::accumulate<lanes>(carry, products);
        }
        if constexpr (Audit) if (!lane && trace) {
            trace[base / 16u * 3u] = carry.significand;
            trace[base / 16u * 3u + 1u] = uint32_t(int32_t(carry.exponent));
            trace[base / 16u * 3u + 2u] = unsigned(carry.negative);
        }
    }
    if constexpr (Audit) if (!lane && stats) {
#pragma unroll
        for (unsigned i = 0u; i < 4u; ++i) stats[i] = counts[i];
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}

template<bool Audit = false>
__global__ void replay_kernel(const uint16_t* weights, const uint16_t* inputs,
    const uint32_t* weight_bounds, const uint32_t* input_bounds,
    const unsigned* indices, unsigned count, float* output, unsigned rows,
    unsigned tokens, unsigned width, uint32_t* stats = nullptr) {
    const unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (slot >= count) return;
    const unsigned cell = indices[slot];
    if (size_t(cell) >= size_t(rows) * tokens) return;
    const unsigned row = cell % rows, token = cell / rows, groups = width / 16u;
    const float value = dot<Audit>(inputs + size_t(token) * width, weights + size_t(row) * width,
        input_bounds + size_t(token) * groups, weight_bounds + size_t(row) * groups,
        width, nullptr, stats ? stats + size_t(slot) * 4u : nullptr);
    if (!(threadIdx.x & 3u)) {
        const uint32_t bits = __float_as_uint(value);
        output[cell] = qrt_sm121_float_alignment::from_bits((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
    }
}

inline hipError_t prepare(const uint16_t* input, uint32_t* output, size_t capacity,
    unsigned rows, unsigned width, hipStream_t stream) {
    if (!input || !output || !rows || rows > 16384u || !width || width > 4096u || width % 16u)
        return hipErrorInvalidValue;
    const unsigned groups = rows * (width / 16u);
    if (capacity < groups) return hipErrorInvalidValue;
    hipLaunchKernelGGL(prepare_kernel, dim3((groups + threads - 1u) / threads), dim3(threads), 0u, stream,
        input, output, groups);
    return hipGetLastError();
}

inline hipError_t launch(const uint16_t* weights, const uint16_t* inputs,
    const uint32_t* weight_bounds, size_t weight_capacity, const uint32_t* input_bounds,
    size_t input_capacity, const unsigned* indices, unsigned count, float* output,
    unsigned rows, unsigned tokens, unsigned width, hipStream_t stream) {
    if (!weights || !inputs || !weight_bounds || !input_bounds || !indices || !output ||
        !rows || rows > 16384u || !tokens || tokens > 8192u || !width || width > 4096u || width % 16u ||
        count > size_t(rows) * tokens || weight_capacity < size_t(rows) * (width / 16u) ||
        input_capacity < size_t(tokens) * (width / 16u)) return hipErrorInvalidValue;
    if (!count) return hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<false>), dim3((count * lanes + threads - 1u) / threads),
        dim3(threads), 0u, stream, weights, inputs, weight_bounds, input_bounds, indices, count, output,
        rows, tokens, width, nullptr);
    return hipGetLastError();
}
} // namespace qrt_sm121_bounded_projection
#endif
