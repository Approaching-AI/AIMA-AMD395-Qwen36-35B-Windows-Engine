#include "../../native/providers/moe_accumulator/sm121_f32_carry_projection.h"
#include "float_alignment_cases.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace scalar = qrt_sm121_scalar_projection;
namespace fast = qrt_sm121_f32_carry_projection;
namespace cases = qrt_float_alignment_cases;
constexpr unsigned guard = 65u;
void check(hipError_t x) { if (x != hipSuccess) throw std::runtime_error(hipGetErrorString(x)); }
void complete() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= end) throw std::runtime_error("scalar projection completion timeout");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* base = nullptr; size_t size;
    explicit Buffer(const std::vector<T>& values):size(values.size()) {
        check(hipMalloc(reinterpret_cast<void**>(&base), size * sizeof(T)));
        check(hipMemcpy(base, values.data(), size * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Buffer() { if (base) (void)hipFree(base); }
    T* data() { return base + guard; }
    std::vector<T> read() {
        std::vector<T> result(size); check(hipMemcpy(result.data(), base, size * sizeof(T), hipMemcpyDeviceToHost)); return result;
    }
};
template<unsigned Variant>
__global__ void execute(const uint16_t* a, const uint16_t* b, const uint16_t* transposed,
    const unsigned* fa, const unsigned* fb, uint32_t* output, unsigned rows, unsigned width) {
    constexpr unsigned lanes = Variant < 3u ? 4u : 16u;
    constexpr unsigned staging = Variant < 3u ? 1u : 4u;
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (row >= rows) return;
    const bool eligible = fa[row] && fb[row];
    float value;
    if constexpr (Variant % 3u == 0u)
        value = scalar::validated_dot<lanes, staging>(a + size_t(row) * width, b + size_t(row) * width, width, eligible);
    else
        value = fast::dot<Variant % 3u - 1u, lanes, staging>(a + size_t(row) * width, b + size_t(row) * width, width, eligible);
    if (!(threadIdx.x & (lanes - 1u))) __builtin_memcpy(output + row, &value, 4u);
}
void run(unsigned rows, unsigned width) {
    const size_t words = size_t(rows) * width;
    std::vector<uint16_t> a(words + 2u * guard, 0x5a5au), b = a, transpose = a;
    std::vector<uint32_t> expected(rows); unsigned eligible_dots = 0u;
    std::vector<unsigned> af(rows + 2u * guard, 0xa5a5a5a5u), bf = af;
    for (unsigned row = 0u; row < rows; ++row) {
        bool valid_a = true, valid_b = true; cases::original::Value carry{0u, -133, false};
        for (unsigned group = 0u; group < width / 16u; ++group) {
            cases::original::Value terms[17]; terms[0] = carry;
            for (unsigned i = 0u; i < 16u; ++i) {
                auto pair = cases::input(row, group, i);
                if (!(row & 1u)) {
                    pair.left = uint16_t(((row * 97u + group * 47u + i * 31u) & 0x807fu) | ((110u + i % 20u) << 7u));
                    pair.right = uint16_t(((row * 61u + group * 53u + i * 127u) & 0x807fu) | ((119u + group % 10u) << 7u));
                }
                const size_t cell = size_t(row) * width + group * 16u + i;
                a[guard + cell] = pair.left; b[guard + cell] = pair.right;
                transpose[guard + size_t(group * 16u + i) * rows + row] = pair.right;
                const auto allowed = [](uint16_t x) { const unsigned e = (x >> 7u) & 255u; return !(x & 0x7fffu) || (e >= 64u && e <= 190u); };
                valid_a &= allowed(pair.left); valid_b &= allowed(pair.right);
                terms[i + 1u] = cases::original::multiply_bf16(pair.left, pair.right, -133);
            }
            carry = cases::original::group_sum<26, -133>(terms, 17u);
        }
        expected[row] = cases::output_bits(carry); af[guard + row] = valid_a; bf[guard + row] = valid_b;
        eligible_dots += valid_a && valid_b;
    }
    Buffer<uint16_t> da(a), db(b), dt(transpose);
    const std::vector<unsigned> flags(rows + 2u * guard, 0xa5a5a5a5u);
    Buffer<unsigned> dfa(flags), dfb(flags);
    hipLaunchKernelGGL(scalar::eligible_rows_kernel, dim3(rows + 3u), dim3(256u), 0u, nullptr, da.data(), dfa.data(), rows, width);
    check(hipGetLastError());
    hipLaunchKernelGGL(scalar::eligible_rows_kernel, dim3(rows + 3u), dim3(256u), 0u, nullptr, db.data(), dfb.data(), rows, width);
    check(hipGetLastError()); complete();
    if (dfa.read() != af || dfb.read() != bf) throw std::runtime_error("row eligibility or flag guards differ");
    for (unsigned variant : {0u, 1u, 2u, 3u, 4u, 5u}) {
        std::vector<uint32_t> initial(rows + 2u * guard, 0xa5a5a5a5u); Buffer<uint32_t> out(initial);
        const dim3 grid((rows * (variant < 3u ? 4u : 16u) + 255u) / 256u + 1u);
        if (variant == 1u) hipLaunchKernelGGL(execute<1u>, grid, dim3(256u), 0u, nullptr, da.data(), db.data(), dt.data(), dfa.data(), dfb.data(), out.data(), rows, width);
        if (variant == 2u) hipLaunchKernelGGL(execute<2u>, grid, dim3(256u), 0u, nullptr, da.data(), db.data(), dt.data(), dfa.data(), dfb.data(), out.data(), rows, width);
        if (variant == 3u) hipLaunchKernelGGL(execute<3u>, grid, dim3(256u), 0u, nullptr, da.data(), db.data(), dt.data(), dfa.data(), dfb.data(), out.data(), rows, width);
        if (variant == 4u) hipLaunchKernelGGL(execute<4u>, grid, dim3(256u), 0u, nullptr, da.data(), db.data(), dt.data(), dfa.data(), dfb.data(), out.data(), rows, width);
        if (variant == 5u) hipLaunchKernelGGL(execute<5u>, grid, dim3(256u), 0u, nullptr, da.data(), db.data(), dt.data(), dfa.data(), dfb.data(), out.data(), rows, width);
        if (variant == 0u) hipLaunchKernelGGL(execute<0u>, grid, dim3(256u), 0u, nullptr, da.data(), db.data(), dt.data(), dfa.data(), dfb.data(), out.data(), rows, width);
        check(hipGetLastError()); complete(); auto actual = out.read(); unsigned bad = 0u;
        for (unsigned row = 0u; row < rows; ++row) bad += actual[guard + row] != expected[row];
        for (unsigned i = 0u; i < guard; ++i)
            if (actual[i] != initial[i] || actual[guard + rows + i] != initial[guard + rows + i]) throw std::runtime_error("output guard changed");
        if (da.read() != a || db.read() != b || dt.read() != transpose || dfa.read() != af || dfb.read() != bf)
            throw std::runtime_error("immutable input changed");
        std::printf("{\"kind\":\"f32_carry_projection_safety\",\"variant\":%u,\"rows\":%u,\"width\":%u,\"eligible_dots\":%u,\"fallback_dots\":%u,\"raw_bit_mismatches\":%u,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n", variant, rows, width, eligible_dots, rows - eligible_dots, bad);
        if (bad || !eligible_dots || eligible_dots == rows) throw std::runtime_error("scalar projection arithmetic or coverage failure");
    }
}
int main() try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    run(17u, 16u); run(257u, 272u); run(513u, 2048u); run(257u, 4096u); return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "f32_carry_projection_error=%s\n", error.what()); return 2; }
