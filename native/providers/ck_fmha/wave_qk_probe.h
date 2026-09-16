#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// Isolated host launch ABI. Each implementation is built from the same source
// with a different device wave size. Main-fixture kernels retain wave32.
namespace qrt_wave_qk_probe {
constexpr unsigned query_heads=16u,kv_heads=2u,head_dim=256u,threads=256u;
constexpr float scale=0.0625f;
using Launch=hipError_t (*)(const uint32_t*,const uint32_t*,const unsigned*,
    const unsigned*,float*,unsigned,unsigned,unsigned,unsigned);
}
extern "C" hipError_t qrt_wave_qk_32(const uint32_t*,const uint32_t*,const unsigned*,
    const unsigned*,float*,unsigned,unsigned,unsigned,unsigned);
extern "C" hipError_t qrt_wave_qk_64(const uint32_t*,const uint32_t*,const unsigned*,
    const unsigned*,float*,unsigned,unsigned,unsigned,unsigned);
extern "C" hipError_t qrt_wave_qk_probe_32(unsigned*);
extern "C" hipError_t qrt_wave_qk_probe_64(unsigned*);
