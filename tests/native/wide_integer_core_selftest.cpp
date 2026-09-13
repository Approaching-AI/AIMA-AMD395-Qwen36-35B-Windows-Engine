#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_wide_integer_core.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace wide = qrt_sm121_wide_core;
constexpr unsigned tiles = 4096u, guard = 64u;
struct Cell { float high, low, combined; };
__global__ void matrix(const wide::Row* input, Cell* output) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32u;
    const auto value = wide::products(input[base + lane % 16u], input[base + 16u + lane % 16u]);
    for (unsigned i = 0u; i < 8u; ++i)
        output[blockIdx.x * 256u + (2u * i + lane / 16u) * 16u + lane % 16u] = {value.high[i], value.low[i], value.combined[i]};
}
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
template<class T> struct Guarded {
    T* base = nullptr; size_t size;
    explicit Guarded(size_t n): size(n) {
        check(hipMalloc(reinterpret_cast<void**>(&base), (n + 2u * guard) * sizeof(T)));
        check(hipMemset(base, 0xa5, (n + 2u * guard) * sizeof(T)));
    }
    ~Guarded() { if (base && hipFree(base) != hipSuccess) std::abort(); }
    T* data() { return base + guard; }
    std::vector<T> read() { std::vector<T> out(size); check(hipMemcpy(out.data(), data(), size * sizeof(T), hipMemcpyDeviceToHost)); return out; }
    bool guards() {
        std::vector<unsigned char> a(guard * sizeof(T)), b(a.size());
        check(hipMemcpy(a.data(), base, a.size(), hipMemcpyDeviceToHost));
        check(hipMemcpy(b.data(), data() + size, b.size(), hipMemcpyDeviceToHost));
        return std::all_of(a.begin(), a.end(), [](unsigned char x) { return x == 0xa5; }) &&
            std::all_of(b.begin(), b.end(), [](unsigned char x) { return x == 0xa5; });
    }
};
uint32_t seed = 0x3958192u;
uint32_t next() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    std::vector<uint32_t> input(tiles * 512u); std::vector<wide::Row> rows(tiles * 32u);
    for (unsigned i = 0u; i < input.size(); ++i) {
        const unsigned mode = i / 512u % 7u;
        input[i] = i < 262144u ? i : mode == 0u ? 0u : mode == 1u ? wide::mask :
            mode == 2u ? wide::center : mode == 3u ? wide::center - 1u : next() & wide::mask;
        const uint32_t positive = input[i] ^ wide::center; auto& row = rows[i / 16u];
        row.high[i % 16u] = qrt_sm121_positive_karatsuba::positive_half_bits(positive >> 9u);
        row.low[i % 16u] = qrt_sm121_positive_karatsuba::positive_half_bits(positive & 511u);
        const uint32_t sum = wide::unsigned_row_sum(row) + positive;
        row.original[16] = uint16_t(sum); row.original[17] = uint16_t(sum >> 16u);
    }
    Guarded<wide::Row> device(rows.size()); Guarded<Cell> output(tiles * 256u);
    check(hipMemcpy(device.data(), rows.data(), rows.size() * sizeof(wide::Row), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(matrix, dim3(tiles), dim3(32u), 0u, nullptr, device.data(), output.data());
    check(hipGetLastError()); check(hipDeviceSynchronize()); const auto result = output.read();
    unsigned partial_bad = 0u, reconstructed_bad = 0u;
    for (unsigned index = 0u; index < result.size(); ++index) {
        const unsigned base = index / 256u * 512u, row = index / 16u % 16u, column = index % 16u;
        int parts[3]{}; int64_t expected = 0;
        for (unsigned i = 0u; i < 16u; ++i) {
            const uint32_t a = input[base + row * 16u + i], b = input[base + 256u + column * 16u + i];
            const uint32_t ap = a ^ wide::center, bp = b ^ wide::center;
            const int ah = int(ap >> 9u), al = int(ap & 511u), bh = int(bp >> 9u), bl = int(bp & 511u);
            parts[0] += ah * bh; parts[1] += al * bl; parts[2] += (ah + al) * (bh + bl);
            const int ac = a & wide::center ? int(a) - 262144 : int(a), bc = b & wide::center ? int(b) - 262144 : int(b);
            expected += int64_t(ac) * bc;
        }
        const float actual[] = {result[index].high, result[index].low, result[index].combined};
        bool convertible = true;
        for (unsigned p = 0u; p < 3u; ++p) {
            convertible = convertible && std::isfinite(actual[p]) && actual[p] >= 0.0f && actual[p] <= 16711744.0f;
            if (actual[p] != float(parts[p])) {
                if (partial_bad < 4u) std::printf("DIFF cell=%u partial=%u expected=%d actual=%.9g\n", index, p, parts[p], double(actual[p]));
                ++partial_bad;
            }
        }
        const auto& a = rows[index / 256u * 32u + row]; const auto& b = rows[index / 256u * 32u + 16u + column];
        reconstructed_bad += unsigned(!convertible || expected != wide::reconstruct(int(actual[0]), int(actual[1]), int(actual[2]), wide::unsigned_row_sum(a), wide::unsigned_row_sum(b)));
    }
    const auto after = device.read(); const bool immutable = std::memcmp(after.data(), rows.data(), rows.size() * sizeof(wide::Row)) == 0;
    const bool safe = device.guards() && output.guards();
    std::printf("{\"kind\":\"wide_positive_integer_core\",\"tiles\":4096,\"cells\":1048576,\"core_encodings_enumerated\":262144,\"row_bytes\":132,\"partial_mismatches\":%u,\"reconstruction_mismatches\":%u,\"redzones_pass\":%s,\"immutable_inputs\":%s,\"inference_acceptance\":false}\n", partial_bad, reconstructed_bad, safe ? "true" : "false", immutable ? "true" : "false");
    return !partial_bad && !reconstructed_bad && safe && immutable ? 0 : 2;
} catch (const std::exception& error) { std::fprintf(stderr, "wide_core_error=%s\n", error.what()); return 1; }
