#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include "../../native/providers/moe_accumulator/sm121_positive_karatsuba.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace k3 = qrt_sm121_positive_karatsuba;
namespace attention = qrt_blackwell_attention;
struct CoreRow { int high[4], low[4]; unsigned positive_sum; };
struct Cell { int integer[4]; float candidate[3]; };
constexpr unsigned kGuard = 64u;

__global__ void compare_cores(const CoreRow* rows, Cell* output) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32u;
    const auto original = attention::blackwell_integer_prepared_products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
    const auto candidate = k3::products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned index = blockIdx.x * 256u + (i * 2u + lane / 16u) * 16u + lane % 16u;
        Cell result{};
        for (unsigned p = 0u; p < 4u; ++p) result.integer[p] = original.value[p][i];
        result.candidate[0] = candidate.high[i]; result.candidate[1] = candidate.low[i];
        result.candidate[2] = candidate.combined[i]; output[index] = result;
    }
}

// Both paths consume the same packed original cores. Fresh rows are loaded
// in each K16 iteration, and each result contributes to the final checksum.
// This is a primitive cost diagnostic, without canonical carry/exception work.
template<bool Candidate>
__global__ void sequence_cores(const CoreRow* rows, uint64_t* output, unsigned tiles, unsigned groups) {
    const unsigned lane = threadIdx.x;
    uint64_t checksum[8]{};
    for (unsigned group = 0u; group < groups; ++group) {
        const unsigned base = ((blockIdx.x * groups + group) % tiles) * 32u;
        if constexpr (Candidate) {
            const auto result = k3::products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
            for (unsigned i = 0u; i < 8u; ++i)
                checksum[i] += uint64_t(k3::reconstruct(int(result.high[i]), int(result.low[i]), int(result.combined[i]),
                    rows[base + i * 2u + lane / 16u].positive_sum, rows[base + 16u + lane % 16u].positive_sum));
        } else {
            const auto result = attention::blackwell_integer_prepared_products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
            for (unsigned i = 0u; i < 8u; ++i)
                checksum[i] += uint64_t(int64_t(result.value[0][i]) * 65536 +
                    (int64_t(result.value[1][i]) + result.value[2][i]) * 256 + result.value[3][i]);
        }
    }
    for (unsigned i = 0u; i < 8u; ++i)
        output[blockIdx.x * 256u + (i * 2u + lane / 16u) * 16u + lane % 16u] = checksum[i];
}

void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
template<class T> struct Guarded {
    T* base = nullptr; size_t size;
    explicit Guarded(size_t n): size(n) {
        check(hipMalloc(reinterpret_cast<void**>(&base), (n + 2u * kGuard) * sizeof(T)));
        check(hipMemset(base, 0xa5, (n + 2u * kGuard) * sizeof(T)));
    }
    ~Guarded() { if (base && hipFree(base) != hipSuccess) std::abort(); }
    T* data() { return base + kGuard; }
    bool guards() {
        std::vector<unsigned char> low(kGuard * sizeof(T)), high(low.size());
        check(hipMemcpy(low.data(), base, low.size(), hipMemcpyDeviceToHost));
        check(hipMemcpy(high.data(), data() + size, high.size(), hipMemcpyDeviceToHost));
        for (auto byte : low) if (byte != 0xa5) return false;
        for (auto byte : high) if (byte != 0xa5) return false;
        return true;
    }
    std::vector<T> read() { std::vector<T> out(size); check(hipMemcpy(out.data(), data(), size * sizeof(T), hipMemcpyDeviceToHost)); return out; }
};
uint32_t seed = 0x3958192u;
uint32_t random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
int decode(uint16_t core) { return core & 0x8000u ? int(core) - 65536 : int(core); }

int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    constexpr unsigned tiles = 4096u;
    std::vector<uint16_t> cores(tiles * 512u);
    std::vector<CoreRow> rows(tiles * 32u);
    const uint16_t edges[] = {0u, 1u, 127u, 128u, 255u, 256u, 32640u, 32767u, 32768u, 32896u, 65408u, 65535u};
    for (unsigned i = 0u; i < cores.size(); ++i) {
        const unsigned mode = i / 512u % 8u;
        cores[i] = mode == 0u ? uint16_t(i / 4096u + (i % 512u) * 128u) : mode == 1u ? edges[random_word() % 12u] :
            mode == 2u ? uint16_t((i & 1u) ? 32767u : 32768u) : mode == 3u ? uint16_t((i % 16u) ? 0u : random_word()) :
            mode == 4u ? uint16_t(random_word() & 0xff80u) : uint16_t(random_word());
        const unsigned row = i / 16u, word = i % 16u / 4u, shift = i % 4u * 8u;
        rows[row].high[word] = int(uint32_t(rows[row].high[word]) | uint32_t(cores[i] >> 8u) << shift);
        rows[row].low[word] = int(uint32_t(rows[row].low[word]) | uint32_t(cores[i] & 255u) << shift);
    }
    for (unsigned i = 0u; i < cores.size(); ++i) rows[i / 16u].positive_sum += k3::sum_digit(cores[i]);
    Guarded<CoreRow> input(rows.size()); Guarded<Cell> output(tiles * 256u);
    check(hipMemcpy(input.data(), rows.data(), rows.size() * sizeof(CoreRow), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(compare_cores, dim3(tiles), dim3(32u), 0u, nullptr, input.data(), output.data());
    check(hipGetLastError()); check(hipDeviceSynchronize()); const auto cells = output.read();
    unsigned integer_bad = 0u, candidate_bad = 0u, reconstruction_bad = 0u;
    for (unsigned index = 0u; index < cells.size(); ++index) {
        const unsigned tile = index / 256u, row = index / 16u % 16u, column = index % 16u;
        int integer[4]{}, balanced[3]{}; int64_t mathematical = 0;
        for (unsigned i = 0u; i < 16u; ++i) {
            const uint16_t a = cores[tile * 512u + row * 16u + i], b = cores[tile * 512u + 256u + column * 16u + i];
            const int ah = int(a >> 8u) - ((a & 0x8000u) ? 256 : 0), al = a & 255u;
            const int bh = int(b >> 8u) - ((b & 0x8000u) ? 256 : 0), bl = b & 255u;
            integer[0] += ah * bh; integer[1] += ah * bl; integer[2] += al * bh; integer[3] += al * bl;
            const auto x = k3::split(a), y = k3::split(b);
            balanced[0] += x.high * y.high; balanced[1] += x.low * y.low;
            balanced[2] += (x.high + x.low + 128) * (y.high + y.low + 128);
            mathematical += int64_t(decode(a)) * decode(b);
        }
        for (unsigned p = 0u; p < 4u; ++p) integer_bad += unsigned(integer[p] != cells[index].integer[p]);
        for (unsigned p = 0u; p < 3u; ++p) {
            if (cells[index].candidate[p] != float(balanced[p])) {
                if (candidate_bad < 4u) std::printf("DIFF cell=%u partial=%u expected=%d actual=%.9g\n", index, p, balanced[p], double(cells[index].candidate[p]));
                ++candidate_bad;
            }
        }
        bool convertible = true;
        for (float part : cells[index].candidate)
            convertible = convertible && std::isfinite(part) && part >= -4161600.0f && part <= 4161600.0f;
        reconstruction_bad += unsigned(!convertible || mathematical != k3::reconstruct(
            int(cells[index].candidate[0]), int(cells[index].candidate[1]), int(cells[index].candidate[2]), rows[tile * 32u + row].positive_sum, rows[tile * 32u + 16u + column].positive_sum));
    }
    const auto unchanged = input.read();
    const bool immutable = std::memcmp(unchanged.data(), rows.data(), rows.size() * sizeof(CoreRow)) == 0;
    const bool guards = input.guards() && output.guards();
    std::printf("{\"kind\":\"positive_karatsuba_integer_core\",\"tiles\":%u,\"cells\":%u,\"integer_partial_mismatches\":%u,\"candidate_raw_partial_mismatches\":%u,\"reconstruction_mismatches\":%u,\"redzones_pass\":%s,\"immutable_inputs\":%s,\"inference_acceptance\":false}\n", tiles, unsigned(cells.size()), integer_bad, candidate_bad, reconstruction_bad, guards ? "true" : "false", immutable ? "true" : "false");
    if (integer_bad || candidate_bad || reconstruction_bad || !guards || !immutable) return 2;
    for (unsigned groups : {16u, 64u, 256u}) {
        constexpr unsigned blocks = 1024u;
        Guarded<uint64_t> original(blocks * 256u), candidate(blocks * 256u);
        auto run = [&](bool use_candidate) {
            check(hipDeviceSynchronize()); const auto start = std::chrono::steady_clock::now();
            if (use_candidate) hipLaunchKernelGGL(HIP_KERNEL_NAME(sequence_cores<true>), dim3(blocks), dim3(32u), 0u, nullptr, input.data(), candidate.data(), tiles, groups);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(sequence_cores<false>), dim3(blocks), dim3(32u), 0u, nullptr, input.data(), original.data(), tiles, groups);
            check(hipGetLastError()); check(hipDeviceSynchronize());
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        };
        run(false); run(true); const double original_ms = run(false), candidate_ms = run(true);
        const auto a = original.read(), b = candidate.read(); unsigned bad = 0u;
        for (unsigned i = 0u; i < a.size(); ++i) bad += unsigned(a[i] != b[i]);
        const auto after = input.read();
        const bool stable = std::memcmp(after.data(), rows.data(), rows.size() * sizeof(CoreRow)) == 0;
        const bool ok = original.guards() && candidate.guards() && input.guards();
        std::printf("{\"kind\":\"positive_karatsuba_core_sequence\",\"groups\":%u,\"blocks\":%u,\"original_ms\":%.6f,\"candidate_ms\":%.6f,\"checksum_mismatches\":%u,\"redzones_pass\":%s,\"immutable_inputs\":%s,\"diagnostic_only\":true,\"inference_acceptance\":false}\n", groups, blocks, original_ms, candidate_ms, bad, ok ? "true" : "false", stable ? "true" : "false");
        if (bad || !ok || !stable) return 2;
    }
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "positive_karatsuba_core_error=%s\n", error.what()); return 1; }
