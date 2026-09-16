#include "wave_qk_probe.h"
#include "../moe_accumulator/sm121_decoded_bf16.h"
#include "../moe_accumulator/sm121_f32_carry.h"

#if QRT_QK_WAVE_BITS == 32
#define QRT_WAVE_NAMESPACE qrt_wave_qk32
#define QRT_WAVE_LAUNCH qrt_wave_qk_32
#define QRT_WAVE_PROBE qrt_wave_qk_probe_32
#elif QRT_QK_WAVE_BITS == 64
#define QRT_WAVE_NAMESPACE qrt_wave_qk64
#define QRT_WAVE_LAUNCH qrt_wave_qk_64
#define QRT_WAVE_PROBE qrt_wave_qk_probe_64
#else
#error Requires QRT_QK_WAVE_BITS=32 or64
#endif

// This translation unit contains scalar kernels only. It deliberately has no
// dependency on the wave32 matrix kernels in the main attention provider.
namespace QRT_WAVE_NAMESPACE {
using namespace qrt_wave_qk_probe;
namespace decoded=qrt_sm121_decoded_bf16;
constexpr uint32_t deferred_bits=0x7ffffffeu;
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    constexpr unsigned Window = 128u, Rows = 16u, Keys = 16u, prepared_query_start = 0u;
    static_assert(Window && Window % 16u == 0u && head_dim % Window == 0u);
    static_assert(Rows * Keys == threads);
    constexpr unsigned rows = Rows, keys = Keys;
    __shared__ uint32_t qvalues[rows][Window], kvalues[Window][keys];
    const unsigned head = blockIdx.y, kv_head = head / (query_heads / kv_heads);
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * keys;
    const unsigned qr = threadIdx.x / keys, kc = threadIdx.x % keys;
    const unsigned row = query_tile + qr, key = key_tile + kc;
    const unsigned packed_start = query_start - prepared_query_start;
    const bool live = row < query_count && key < stride;
    const bool active = live && key <= query_start + row;
    const size_t output_cell = (size_t(row) * query_heads + head) * stride + key;
    const unsigned last_query = query_start + min(query_tile + rows, query_count) - 1u;
    if (key_tile > last_query) {
        if (live) output[output_cell] = -INFINITY;
        return;
    }
    bool fallback = active && (!query_flags[(packed_start + row) * query_heads + head] ||
        !key_flags[key * kv_heads + kv_head]);
    float float_carry = 0.0f;
    for (unsigned window = 0u; window < head_dim; window += Window) {
        for (unsigned cell = threadIdx.x; cell < rows * Window; cell += threads) {
            const unsigned r = cell / Window, c = cell % Window;
            qvalues[r][c] = query_tile + r < query_count
                ? packed_query[(size_t(packed_start + query_tile + r) * query_heads + head) * head_dim + window + c]
                : decoded::pack(0u);
        }
        for (unsigned cell = threadIdx.x; cell < Window * keys; cell += threads) {
            const unsigned r = cell / keys, c = cell % keys;
            kvalues[r][c] = key_tile + c < stride
                ? packed_key[(size_t(kv_head) * head_dim + window + r) * key_stride + key_tile + c]
                : decoded::pack(0u);
        }
        __syncthreads();
        if (active && !fallback) {
            for (unsigned base = 0u; base < Window; base += 16u) {
                qrt_sm121_float_alignment::Group group;
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    decoded::set_packed(group, i, qvalues[qr][base + i], kvalues[base + i][kc]);
                float next;
                if (!qrt_sm121_f32_carry::accumulate<0u>(float_carry, group, &next)) { fallback = true; break; }
                float_carry = next;
            }
        }
        __syncthreads();
    }
    const bool deferred = active && fallback;
    if (live) output[output_cell] = !active ? -INFINITY : deferred
        ? qrt_sm121_float_alignment::from_bits(deferred_bits) : float_carry * scale;
}

__global__ void report_wave(unsigned* output) {
    const unsigned long long active=__ballot(true);
    if(threadIdx.x==0u){output[0]=warpSize;output[1]=unsigned(__popcll(active));}
}
}
extern "C" hipError_t QRT_WAVE_LAUNCH(const uint32_t* q,const uint32_t* k,
    const unsigned* qf,const unsigned* kf,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride) {
    hipLaunchKernelGGL(QRT_WAVE_NAMESPACE::scores,
        dim3((stride+15u)/16u,qrt_wave_qk_probe::query_heads,(count+15u)/16u),
        dim3(qrt_wave_qk_probe::threads),0u,nullptr,q,k,qf,kf,output,start,count,stride,key_stride);
    return hipGetLastError();
}
extern "C" hipError_t QRT_WAVE_PROBE(unsigned* output) {
    hipLaunchKernelGGL(QRT_WAVE_NAMESPACE::report_wave,dim3(1u),dim3(256u),0u,nullptr,output);
    return hipGetLastError();
}
