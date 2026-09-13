#ifndef QRT_SM121_ROW_MAX_PROJECTION_H
#define QRT_SM121_ROW_MAX_PROJECTION_H
#include "sm121_scalar_projection.h"
#include "sm121_row_maximum.h"

// Component candidate; the product dispatcher is unchanged. Complete-row
// maxima stay in registers throughout a dot, with no per-K16 metadata reads.
namespace qrt_sm121_row_max_projection {
namespace row_max = qrt_sm121_row_maximum;
using Value = qrt_q1_moe_hawkeye::Value;
constexpr unsigned threads = 256u, lanes = 4u;

__global__ void maximum_rows_kernel(const uint16_t* input, uint32_t* output,
    unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    __shared__ uint32_t maximum;
    if (!threadIdx.x) maximum = 0u;
    __syncthreads();
    uint32_t local = 0u;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x)
        local = row_max::append(local, input[size_t(row) * width + k]);
    if (local) atomicMax(&maximum, local);
    __syncthreads();
    if (!threadIdx.x) output[row] = maximum;
}

template<bool Audit = false>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned width, uint32_t left_maximum, uint32_t right_maximum,
    uint32_t* trace = nullptr, uint32_t* stats = nullptr) {
    const unsigned lane = threadIdx.x & 3u;
    const bool valid = row_max::eligible(left_maximum, right_maximum);
    if constexpr (!Audit) if (!valid)
        return qrt_sm121_subgroup::dot<lanes>(left, right, width);
    const int upper = row_max::product_upper(left_maximum, right_maximum);
    Value carry{0u, -133, false};
    unsigned counts[4]{};
#pragma unroll 1
    for (unsigned base = 0u; base < width; base += 16u) {
        uint64_t a, b;
        __builtin_memcpy(&a, left + base + lane * 4u, sizeof(a));
        __builtin_memcpy(&b, right + base + lane * 4u, sizeof(b));
        if (valid && row_max::carry_dominates(carry.exponent, upper)) {
            // Each actual nonzero product exponent is <= the sum of row
            // maxima, so the original K16 maximum is exactly carry.exponent.
            // Carry alignment is the identity; the products and their RTZ
            // conversion retain the validated scalar arithmetic.
            const float scale = qrt_sm121_float_alignment::from_bits(uint32_t(152 - carry.exponent) << 23u);
            uint32_t modulo = 0u;
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                const uint16_t x = uint16_t(a >> (16u * i)), y = uint16_t(b >> (16u * i));
                const float product = qrt_sm121_float_alignment::from_bits(uint32_t(x) << 16u) *
                    qrt_sm121_float_alignment::from_bits(uint32_t(y) << 16u);
                modulo += uint32_t(int32_t(product * scale));
            }
            modulo = qrt_sm121_lane_reduce::sum<lanes>(modulo);
            const uint32_t aligned = carry.significand << 2u;
            modulo += carry.negative ? 0u - aligned : aligned;
            const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, ((a ^ b) & 0x8000u) != 0u);
            carry = qrt_sm121_wave16::normalize(sum.magnitude, sum.negative, carry.exponent);
            if constexpr (Audit) ++counts[0];
        } else if (valid) {
            qrt_sm121_float_subgroup::Product products[4];
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i) {
                const uint16_t x = uint16_t(a >> (16u * i)), y = uint16_t(b >> (16u * i));
                const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
                products[i] = {
                    qrt_sm121_float_alignment::from_bits(uint32_t(x) << 16u) *
                        qrt_sm121_float_alignment::from_bits(uint32_t(y) << 16u),
                    uint32_t(x) | (uint32_t(y) << 16u),
                    zero ? -133 : int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254};
            }
            carry = qrt_sm121_float_subgroup::accumulate<lanes>(carry, products);
            if constexpr (Audit) ++counts[1];
        } else {
            uint32_t products[4];
#pragma unroll
            for (unsigned i = 0u; i < 4u; ++i)
                products[i] = qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
                    uint16_t(a >> (16u * i)), uint16_t(b >> (16u * i)), -133));
            carry = qrt_sm121_subgroup::accumulate_products<lanes>(carry, products);
            if constexpr (Audit) ++counts[2];
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
    const uint32_t* weight_maxima, const uint32_t* input_maxima,
    const unsigned* indices, unsigned count, float* output, unsigned rows,
    unsigned tokens, unsigned width, uint32_t* stats = nullptr) {
    const unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (slot >= count) return;
    const unsigned cell = indices[slot];
    if (size_t(cell) >= size_t(rows) * tokens) return;
    const unsigned row = cell % rows, token = cell / rows;
    const float value = dot<Audit>(inputs + size_t(token) * width, weights + size_t(row) * width,
        width, input_maxima[token], weight_maxima[row], nullptr, stats ? stats + size_t(slot) * 4u : nullptr);
    if (!(threadIdx.x & 3u)) {
        const uint32_t bits = __float_as_uint(value);
        output[cell] = qrt_sm121_float_alignment::from_bits((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
    }
}

inline hipError_t prepare(const uint16_t* input, uint32_t* output, size_t capacity,
    unsigned rows, unsigned width, hipStream_t stream) {
    if (!input || !output || !rows || rows > 16384u || !width || width > 4096u || width % 16u || capacity < rows)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(maximum_rows_kernel, dim3(rows), dim3(threads), 0u, stream, input, output, rows, width);
    return hipGetLastError();
}

inline hipError_t launch(const uint16_t* weights, const uint16_t* inputs,
    const uint32_t* weight_maxima, size_t weight_capacity, const uint32_t* input_maxima,
    size_t input_capacity, const unsigned* indices, unsigned count, float* output,
    unsigned rows, unsigned tokens, unsigned width, hipStream_t stream) {
    if (!weights || !inputs || !weight_maxima || !input_maxima || !indices || !output ||
        !rows || rows > 16384u || !tokens || tokens > 8192u || !width || width > 4096u || width % 16u ||
        count > size_t(rows) * tokens || weight_capacity < rows || input_capacity < tokens)
        return hipErrorInvalidValue;
    if (!count) return hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<false>), dim3((count * lanes + threads - 1u) / threads),
        dim3(threads), 0u, stream, weights, inputs, weight_maxima, input_maxima, indices, count, output,
        rows, tokens, width, nullptr);
    return hipGetLastError();
}
} // namespace qrt_sm121_row_max_projection
#endif
