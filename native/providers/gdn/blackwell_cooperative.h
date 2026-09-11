#ifndef QRT_FLA_BLACKWELL_COOPERATIVE_H
#define QRT_FLA_BLACKWELL_COOPERATIVE_H
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Optional batched FLA schedule. Public entry points validate the same spans
// and aliases before dispatch. Reference arithmetic and endpoint types remain
// unchanged; the original schedule stays available in the same DLL.
namespace qrt_fla_blackwell_cooperative {
inline bool enabled() {
    const char* value = std::getenv("QRT_FLA_GDN_COOPERATIVE_EXACT");
    return value && std::strcmp(value, "1") == 0;
}
hipError_t wu(const uint16_t*, const uint16_t*, const uint16_t*, const uint16_t*,
              const float*, uint16_t*, uint16_t*, unsigned, const unsigned char*, hipStream_t);
hipError_t scores(const uint16_t*, const uint16_t*, const float*, uint16_t*,
                  unsigned, const unsigned char*, hipStream_t);
hipError_t output(const uint16_t*, const uint16_t*, const uint16_t*, const float*,
                  const uint16_t*, float*, unsigned, const unsigned char*, hipStream_t);
hipError_t state(const uint16_t*, const uint16_t*, const uint16_t*, const float*,
                 uint16_t*, uint16_t*, float*, unsigned, const unsigned char*, hipStream_t);
}
#endif
