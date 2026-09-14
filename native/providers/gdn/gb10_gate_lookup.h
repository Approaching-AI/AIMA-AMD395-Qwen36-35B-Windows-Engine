#ifndef QRT_GB10_GATE_LOOKUP_H
#define QRT_GB10_GATE_LOOKUP_H

#include <hip/hip_runtime.h>
#include <cstdint>

namespace qrt_gb10_gate_lookup {
constexpr unsigned heads = 32u;
constexpr unsigned values = 65536u;

// The tables enumerate the complete BF16 input domain. Preserve the existing
// CPU gate handoff's RNE conversion, head-major G table, and token-major
// [G32, beta32] output ABI. No transcendental operation is recomputed here.
__global__ void rows(const float* a, const float* b, const uint32_t* g,
                    const uint16_t* beta, float* output, unsigned tokens) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(tokens) * heads) return;
    const uint32_t a_bits = __float_as_uint(a[index]);
    const uint32_t b_bits = __float_as_uint(b[index]);
    const uint16_t ai = uint16_t((a_bits + 0x7fffu + ((a_bits >> 16u) & 1u)) >> 16u);
    const uint16_t bi = uint16_t((b_bits + 0x7fffu + ((b_bits >> 16u) & 1u)) >> 16u);
    const unsigned head = unsigned(index % heads);
    const size_t destination = (index / heads) * (2u * heads) + head;
    output[destination] = __uint_as_float(g[size_t(head) * values + ai]);
    output[destination + heads] = __uint_as_float(uint32_t(beta[bi]) << 16u);
}

inline hipError_t launch(const float* a, const float* b, const uint32_t* g,
                         const uint16_t* beta, float* output, unsigned tokens,
                         hipStream_t stream = nullptr) {
    if (!a || !b || !g || !beta || !output || !tokens || tokens > 65536u)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(rows, dim3((size_t(tokens) * heads + 255u) / 256u),
                       dim3(256), 0, stream, a, b, g, beta, output, tokens);
    return hipGetLastError();
}
} // namespace qrt_gb10_gate_lookup
#endif
