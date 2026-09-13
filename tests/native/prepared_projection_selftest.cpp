#include "../../native/providers/moe_accumulator/sm121_prepared_projection.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr unsigned guard = 65u; // Deliberately only halfword-aligned operands.
constexpr uint16_t operand_guard = 0x5a5au;
constexpr unsigned flag_guard = 0xa5a5a5a5u;
struct Result { float original, prepared; };
constexpr Result result_guard{12345.25f, -12345.25f};
uint32_t seed = 0x3958192u;
uint32_t random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
uint32_t bits(float x) { uint32_t word; std::memcpy(&word, &x, 4u); return word; }
uint16_t bf16(float x) { const uint32_t word = bits(x); return uint16_t((word + 0x7fffu + ((word >> 16u) & 1u)) >> 16u); }
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void check(hipError_t error) { if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error)); }
void complete() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const hipError_t status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now() < deadline, "GPU completion timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* base = nullptr;
    explicit Buffer(const std::vector<T>& host) {
        check(hipMalloc(reinterpret_cast<void**>(&base), host.size() * sizeof(T)));
        check(hipMemcpy(base, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Buffer() { if (base) (void)hipFree(base); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() { return base + guard; }
    void read(std::vector<T>& host) { check(hipMemcpy(host.data(), base, host.size() * sizeof(T), hipMemcpyDeviceToHost)); }
    void write(const std::vector<T>& host) { check(hipMemcpy(base, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice)); }
};

template<unsigned Lanes>
__global__ void compare(const uint16_t* a, const uint16_t* b, const uint16_t* pa,
    const uint16_t* pb, const unsigned* fa, const unsigned* fb, Result* output,
    unsigned rows, unsigned k) {
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / Lanes;
    if (row >= rows) return;
    const size_t offset = size_t(row) * k;
    const float original = qrt_sm121_subgroup::dot<Lanes>(a + offset, b + offset, k);
    const float prepared = fa[row] && fb[row]
        ? qrt_sm121_prepared_projection::dot<Lanes>(pa + offset, pb + offset, k)
        : qrt_sm121_subgroup::dot<Lanes>(a + offset, b + offset, k);
    if (!(threadIdx.x & (Lanes - 1u))) output[row] = {original, prepared};
}

unsigned check_preparation(const std::vector<uint16_t>& source,
    const std::vector<uint16_t>& encoded, const std::vector<unsigned>& flags,
    unsigned rows, unsigned k) {
    unsigned eligible_rows = 0u;
    for (unsigned i = 0u; i < guard; ++i) {
        require(encoded[i] == operand_guard && encoded[guard + size_t(rows) * k + i] == operand_guard,
                "prepared operand redzone");
        require(flags[i] == flag_guard && flags[guard + rows + i] == flag_guard, "prepared flag redzone");
    }
    for (unsigned row = 0u; row < rows; ++row) {
        bool valid_row = true;
        for (unsigned column = 0u; column < k; ++column) {
            const size_t index = guard + size_t(row) * k + column;
            const uint16_t value = source[index];
            const unsigned exponent = (value >> 7u) & 255u;
            const bool zero = (value & 0x7fffu) == 0u;
            const bool valid = zero || (exponent >= 64u && exponent <= 191u);
            const uint16_t expected = !valid ? 0u : zero ? value : uint16_t(
                (value & 0x8000u) | ((exponent - 64u) << 8u) | 128u | (value & 127u));
            require(encoded[index] == expected, "prepared cell differs from CPU encoding");
            valid_row &= valid;
        }
        require(flags[guard + row] == unsigned(valid_row), "mixed row eligibility mismatch");
        eligible_rows += valid_row;
    }
    return eligible_rows;
}

void run(unsigned rows, unsigned k) {
    const size_t cells = size_t(rows) * k;
    std::vector<uint16_t> a(cells + 2u * guard, operand_guard), b = a;
    for (unsigned row = 0u; row < rows; ++row) for (unsigned column = 0u; column < k; ++column) {
        const size_t index = guard + size_t(row) * k + column;
        const unsigned exponent = row % 13u ? 119u + random_word() % 20u : 64u + random_word() % 128u;
        a[index] = uint16_t((random_word() & 0x807fu) | (exponent << 7u));
        b[index] = uint16_t((random_word() & 0x807fu) | ((254u - exponent) << 7u));
        if (row % 17u == 0u) a[index] = uint16_t((row & 1u) << 15u);
        if (row % 19u == 0u) b[index] = uint16_t((column & 1u) << 15u);
        if (row % 23u == 0u) { a[index] = uint16_t(0x3f80u | ((column & 1u) << 15u)); b[index] = 0x3f80u; }
    }
    for (unsigned row = 0u; row < rows; ++row) {
        const size_t index = guard + size_t(row) * k;
        if (row % 7u == 1u) a[index + 3u] = 0x8001u; // Signed subnormal fallback.
        if (row % 11u == 2u) b[index + 5u] = 0x1fffu; // Just below encoded exponent range.
        if (row % 29u == 3u) a[index + 7u] = 0x6000u; // Just above encoded range.
        if (row % 31u == 4u) { a[index + 9u] = 0x7fc1u; b[index + 9u] = 0u; }
    }
    std::vector<uint16_t> pa(a.size(), operand_guard), pb = pa;
    std::vector<unsigned> fa(rows + 2u * guard, flag_guard), fb = fa;
    Buffer<uint16_t> da(a), db(b), dpa(pa), dpb(pb);
    Buffer<unsigned> dfa(fa), dfb(fb);
    hipLaunchKernelGGL(qrt_sm121_prepared_projection::prepare_rows_kernel,
        dim3(rows + 3u), dim3(256u), 0u, nullptr, da.data(), dpa.data(), dfa.data(), rows, k);
    check(hipGetLastError());
    hipLaunchKernelGGL(qrt_sm121_prepared_projection::prepare_rows_kernel,
        dim3(rows + 3u), dim3(256u), 0u, nullptr, db.data(), dpb.data(), dfb.data(), rows, k);
    check(hipGetLastError()); complete();
    dpa.read(pa); dpb.read(pb); dfa.read(fa); dfb.read(fb);
    const unsigned eligible_a = check_preparation(a, pa, fa, rows, k);
    const unsigned eligible_b = check_preparation(b, pb, fb, rows, k);
    unsigned prepared_dots = 0u;
    std::vector<float> expected(rows);
    for (unsigned row = 0u; row < rows; ++row) {
        expected[row] = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(
            0.0f, a.data() + guard + size_t(row) * k, b.data() + guard + size_t(row) * k, k);
        prepared_dots += fa[guard + row] && fb[guard + row];
    }
    std::vector<Result> result(rows + 2u * guard, result_guard);
    Buffer<Result> output(result);
    for (unsigned lanes : {4u, 8u, 16u}) {
        std::fill(result.begin(), result.end(), result_guard); output.write(result);
        const dim3 grid((rows * lanes + 255u) / 256u + 1u);
        if (lanes == 4u) hipLaunchKernelGGL(compare<4u>, grid, dim3(256u), 0u, nullptr,
            da.data(), db.data(), dpa.data(), dpb.data(), dfa.data(), dfb.data(), output.data(), rows, k);
        else if (lanes == 8u) hipLaunchKernelGGL(compare<8u>, grid, dim3(256u), 0u, nullptr,
            da.data(), db.data(), dpa.data(), dpb.data(), dfa.data(), dfb.data(), output.data(), rows, k);
        else hipLaunchKernelGGL(compare<16u>, grid, dim3(256u), 0u, nullptr,
            da.data(), db.data(), dpa.data(), dpb.data(), dfa.data(), dfb.data(), output.data(), rows, k);
        check(hipGetLastError()); complete(); output.read(result);
        unsigned raw_bad = 0u, cpu_bad = 0u, bf16_bad = 0u;
        for (unsigned row = 0u; row < rows; ++row) {
            const Result actual = result[guard + row];
            raw_bad += bits(actual.original) != bits(actual.prepared);
            cpu_bad += bits(actual.prepared) != bits(expected[row]);
            bf16_bad += bf16(actual.prepared) != bf16(expected[row]);
        }
        for (unsigned i = 0u; i < guard; ++i) require(
            !std::memcmp(&result[i], &result_guard, sizeof(Result)) &&
            !std::memcmp(&result[guard + rows + i], &result_guard, sizeof(Result)), "dot output redzone");
        std::printf("{\"kind\":\"prepared_projection_dot\",\"rows\":%u,\"k\":%u,\"lanes\":%u,\"encoded_cells\":%zu,\"eligible_left_rows\":%u,\"eligible_right_rows\":%u,\"prepared_dots\":%u,\"fallback_dots\":%u,\"raw_bit_mismatches\":%u,\"cpu_bit_mismatches\":%u,\"bf16_mismatches\":%u,\"redzones_pass\":true,\"inference_acceptance\":false}\n",
            rows, k, lanes, cells * 2u, eligible_a, eligible_b, prepared_dots, rows - prepared_dots, raw_bad, cpu_bad, bf16_bad);
        require(!raw_bad && !cpu_bad && !bf16_bad, "prepared dot numerical mismatch");
    }
    auto after = a; da.read(after); require(after == a, "original left changed");
    db.read(after); require(after == b, "original right changed");
    dpa.read(after); require(after == pa, "prepared left changed during dot");
    dpb.read(after); require(after == pb, "prepared right changed during dot");
    auto flags_after = fa; dfa.read(flags_after); require(flags_after == fa, "left flags changed");
    dfb.read(flags_after); require(flags_after == fb, "right flags changed");
    std::printf("{\"kind\":\"prepared_projection_immutable\",\"rows\":%u,\"k\":%u,\"original_and_prepared_inputs_immutable\":true,\"redzones_pass\":true}\n", rows, k);
}
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    require(!std::strncmp(properties.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    for (const auto shape : {std::pair{1u, 16u}, {17u, 16u}, {257u, 512u}, {1027u, 2048u}, {1027u, 4096u}})
        run(shape.first, shape.second);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "prepared_projection_selftest_error=%s\n", error.what()); return 2;
}
