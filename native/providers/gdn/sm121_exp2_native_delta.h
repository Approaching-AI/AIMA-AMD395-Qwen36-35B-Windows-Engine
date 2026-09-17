#pragma once
#include "sm121_exp2_interpolated.h"

#if defined(__HIPCC__)
#define QRT_EXP_DELTA_INLINE __host__ __device__ __forceinline__
#else
#define QRT_EXP_DELTA_INLINE inline
#endif

// Each original negative FP32 input owns
// a two-bit correction to the actual gfx1151 EXP instruction. Code3 escapes
// to the complete SHA-bound source table. No bound on native SFU error is
// assumed. Derived bytes must be built and verified on the execution device;
// they are neither model outputs nor a portable replacement source artifact.
namespace qrt_sm121_exp2_native_delta {
namespace source = qrt_sm121_exp2_interpolated;
constexpr uint32_t cells = source::end - source::begin;
constexpr size_t packed_bytes = (size_t(cells) + 3u) / 4u;
static_assert(cells % 4u == 0u);

QRT_EXP_DELTA_INLINE unsigned encode(uint32_t native, uint32_t expected) {
    const int64_t delta = int64_t(expected) - int64_t(native);
    return delta >= -1 && delta <= 1 ? unsigned(delta + 1) : 3u;
}
QRT_EXP_DELTA_INLINE unsigned code(const unsigned char* packed, uint32_t relative) {
    return (packed[relative >> 2u] >> (2u * (relative & 3u))) & 3u;
}
QRT_EXP_DELTA_INLINE uint32_t apply(uint32_t native, unsigned correction) {
    return native + uint32_t(correction) - 1u;
}

#if defined(__HIPCC__)
__device__ __forceinline__ float native_exp(float argument) {
    float result;
    asm("v_exp_f32 %0, %1" : "=v"(result) : "v"(argument));
    return result;
}
__device__ __forceinline__ float evaluate(const unsigned char* original,
    const unsigned char* packed, float argument) {
    const uint32_t input = qrt_sm121_exp2::bits(argument), magnitude = input & 0x7fffffffu;
    if (magnitude < source::begin || (!(input >> 31u) && magnitude < qrt_sm121_exp2::positive_one_end))
        return 1.0f;
    if (magnitude > 0x7f800000u || !(input >> 31u)) return qrt_sm121_exp2::value(0x7fc00000u);
    if (magnitude >= source::end) return 0.0f;
    const unsigned correction = code(packed, magnitude - source::begin);
    if (correction == 3u) return source::evaluate(original, argument);
    return qrt_sm121_exp2::value(apply(qrt_sm121_exp2::bits(native_exp(argument)), correction));
}
__global__ void build(const unsigned char* original, unsigned char* packed) {
    for (size_t byte = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         byte < packed_bytes; byte += size_t(gridDim.x) * blockDim.x) {
        unsigned result = 0u;
#pragma unroll
        for (unsigned part = 0u; part < 4u; ++part) {
            const uint32_t relative = uint32_t(byte * 4u + part);
            const float input = qrt_sm121_exp2::value(0x80000000u | (source::begin + relative));
            result |= encode(qrt_sm121_exp2::bits(native_exp(input)), source::decode(original, relative)) << (2u * part);
        }
        packed[byte] = static_cast<unsigned char>(result);
    }
}
__global__ void verify(const unsigned char* original,const unsigned char* packed,unsigned* bad){
    for(uint32_t cell=blockIdx.x*blockDim.x+threadIdx.x;cell<cells;cell+=gridDim.x*blockDim.x){
        const float input=qrt_sm121_exp2::value(0x80000000u|(source::begin+cell));
        if(qrt_sm121_exp2::bits(evaluate(original,packed,input))!=source::decode(original,cell))
            atomicAdd(bad,1u);
    }
}
#endif
} // namespace qrt_sm121_exp2_native_delta
#undef QRT_EXP_DELTA_INLINE
