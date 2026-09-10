// CPU-only ABI/range test double, never a numerical implementation.
#include "../../native/providers/gdn/blackwell_state.h"
namespace qrt_fla_blackwell_state {
hipError_t prepare_exp2_table() { return hipSuccess; }
void release_exp2_table() {}
uint64_t exp2_table_storage_bytes() { return 0; }
hipError_t project(const uint16_t* w, const uint16_t* u, const float* g,
                   const float* initial, uint16_t* h, uint16_t* v_new,
                   uint16_t* residual, unsigned count, hipStream_t) {
    if (!valid_project(w, u, g, initial, h, v_new, residual, count)) return 1;
    const size_t state = 32u * 128u * 128u;
    const void* pointers[] = {w, u, g, initial, h, v_new, residual};
    const size_t bytes[] = {count * 4096u * 2u, count * 4096u * 2u, count * 32u * 4u,
                           state * 4u, state * 2u, count * 4096u * 2u, 64u * 4096u * 2u};
    for (unsigned i = 0; i < 7; ++i) if (!fake_range(const_cast<void*>(pointers[i]), bytes[i])) return 1;
    std::cerr << "FAKE_HIP blackwell_project tokens=" << count << '\n';
    return 0;
}
hipError_t update(const uint16_t* k, const uint16_t* residual, const float* g,
                  const float* initial, float* final, unsigned count, hipStream_t) {
    if (!valid_update(k, residual, g, initial, final, count)) return 1;
    const size_t state = 32u * 128u * 128u;
    const void* pointers[] = {k, residual, g, initial, final};
    const size_t bytes[] = {count * 2048u * 2u, 64u * 4096u * 2u, count * 32u * 4u, state * 4u, state * 4u};
    for (unsigned i = 0; i < 5; ++i) if (!fake_range(const_cast<void*>(pointers[i]), bytes[i])) return 1;
    std::cerr << "FAKE_HIP blackwell_update tokens=" << count << '\n';
    return 0;
}
}
