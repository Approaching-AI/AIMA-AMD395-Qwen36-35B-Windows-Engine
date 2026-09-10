// CPU-only ABI/range test double, never a numerical implementation.
#include "../../native/providers/gdn/blackwell_state.h"
#include "../../native/providers/gdn/blackwell_wu_output.h"
#include "../../native/providers/gdn/blackwell_l2norm.h"
namespace qrt_fla_blackwell_norm {
hipError_t prepare_table() { return hipSuccess; }
void release_table() {}
uint64_t table_storage_bytes() { return 0; }
const unsigned char* table_device() { return reinterpret_cast<const unsigned char*>(1); }
hipError_t normalize(const float* raw, uint16_t* q, uint16_t* k, unsigned tokens, hipStream_t) {
    if (!valid_normalize(raw, q, k, tokens) || !fake_range(const_cast<float*>(raw), size_t(tokens) * 8192u * 4u) ||
        !fake_range(q, size_t(tokens) * 2048u * 2u) || !fake_range(k, size_t(tokens) * 2048u * 2u)) return 1;
    std::cerr << "FAKE_HIP blackwell_norm tokens=" << tokens << '\n'; return 0;
}
}
namespace qrt_fla_blackwell_state {
hipError_t prepare_exp2_table() { return hipSuccess; }
void release_exp2_table() {}
uint64_t exp2_table_storage_bytes() { return 0; }
const unsigned char* exp2_table_device() { return reinterpret_cast<const unsigned char*>(1); }
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
namespace qrt_fla_blackwell_aux {
hipError_t output_scores(const uint16_t* q, const uint16_t* k, const float* g,
                          uint16_t* scores, unsigned count, hipStream_t) {
    if (!count || count > 64) return 1;
    const void* pointers[] = {q, k, g, scores};
    const size_t bytes[] = {count * 2048u * 2u, count * 2048u * 2u, count * 32u * 4u, 64u * 32u * 64u * 2u};
    for (unsigned i = 0; i < 4; ++i) if (!fake_range(const_cast<void*>(pointers[i]), bytes[i])) return 1;
    std::cerr << "FAKE_HIP blackwell_scores tokens=" << count << '\n'; return 0;
}
hipError_t output_values(const uint16_t* q, const uint16_t* v, const uint16_t* h,
                          const float* g, const uint16_t* scores, float* output,
                          unsigned count, hipStream_t) {
    if (!count || count > 64) return 1;
    const void* pointers[] = {q, v, h, g, scores, output};
    const size_t bytes[] = {count * 2048u * 2u, count * 4096u * 2u, 32u * 128u * 128u * 2u,
        count * 32u * 4u, 64u * 32u * 64u * 2u, count * 4096u * 4u};
    for (unsigned i = 0; i < 6; ++i) if (!fake_range(const_cast<void*>(pointers[i]), bytes[i])) return 1;
    std::cerr << "FAKE_HIP blackwell_output tokens=" << count << '\n'; return 0;
}
hipError_t recompute_wu(const uint16_t* k, const uint16_t* v, const uint16_t* beta,
                         const uint16_t* inverse, const float* g, uint16_t* w,
                         uint16_t* u, unsigned count, hipStream_t) {
    if (!valid_wu(k, v, beta, inverse, g, w, u, count) || u != v) return 1;
    const void* pointers[] = {k, v, beta, inverse, g, w, u};
    const size_t bytes[] = {count * 2048u * 2u, count * 4096u * 2u, count * 32u * 2u,
        count * 2048u * 2u, count * 32u * 4u, count * 4096u * 2u, count * 4096u * 2u};
    for (unsigned i = 0; i < 7; ++i) if (!fake_range(const_cast<void*>(pointers[i]), bytes[i])) return 1;
    std::cerr << "FAKE_HIP blackwell_wu_inplace tokens=" << count << '\n'; return 0;
}
}
