#ifndef QRT_FLA_BLACKWELL_STATE_H
#define QRT_FLA_BLACKWELL_STATE_H
#include <hip/hip_runtime.h>
#include <cstdint>

// Fixed-size, one-chunk state operator. The HIP compiler owns the launch ABI;
// callers never construct a hand-written kernarg array for these kernels.
namespace qrt_fla_blackwell_state {
hipError_t prepare_exp2_table();
void release_exp2_table();
uint64_t exp2_table_storage_bytes();
const unsigned char* exp2_table_device();
inline bool valid_project(const uint16_t* w, const uint16_t* u, const float* g,
                          const float* initial, uint16_t* h, uint16_t* v_new,
                          uint16_t* residual, unsigned count) {
    if (!w || !u || !g || !initial || !h || !v_new || !residual || !count || count > 64) return false;
    const void* inputs[] = {w, u, g, initial};
    const void* outputs[] = {h, v_new, residual};
    for (auto output : outputs) for (auto input : inputs) if (output == input) return false;
    return h != v_new && h != residual && v_new != residual;
}
inline bool valid_update(const uint16_t* k, const uint16_t* residual, const float* g,
                         const float* initial, float* final, unsigned count) {
    return k && residual && g && initial && final && count && count <= 64 &&
           static_cast<const void*>(final) != k && static_cast<const void*>(final) != residual &&
           final != g && final != initial;
}
hipError_t project(const uint16_t* w, const uint16_t* u, const float* g,
                   const float* initial, uint16_t* h, uint16_t* v_new,
                   uint16_t* residual, unsigned count, hipStream_t stream);
hipError_t update(const uint16_t* k, const uint16_t* residual, const float* g,
                  const float* initial, float* final, unsigned count, hipStream_t stream);
}
#endif
