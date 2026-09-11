#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <cstdio>
#include <cstring>
#include <vector>

constexpr unsigned kCases = 64u, kCells = 256u;
struct Cell { float partials[4]; uint32_t magnitude; int exponent; unsigned negative, accepted; };

__global__ void matrix_probe(const uint16_t* input, const float* carries, Cell* output) {
    __shared__ uint16_t left[16][18], right[16][18];
    const unsigned lane = threadIdx.x, base = blockIdx.x * 512u;
    if (lane < 16u) {
        for (unsigned i = 0; i < 16u; ++i) {
            left[lane][i] = input[base + lane * 16u + i];
            right[lane][i] = input[base + 256u + lane * 16u + i];
        }
    }
    __syncthreads();
    const auto matrix = qrt_blackwell_attention::blackwell_mantissa_products(left, right, lane);
    for (unsigned element = 0; element < 8u; ++element) {
        const unsigned row = 2u * element + lane / 16u, column = lane % 16u;
        const unsigned index = blockIdx.x * 256u + row * 16u + column;
        Cell result{};
        uint32_t pairs[16]; float partials[4];
        for (unsigned i = 0; i < 16u; ++i)
            pairs[i] = uint32_t(left[row][i]) | (uint32_t(right[column][i]) << 16u);
        for (unsigned i = 0; i < 4u; ++i) result.partials[i] = partials[i] = matrix.value[i][element];
        qrt_sm121_group16::AlignedSum sum{};
        const auto carry = qrt_q1_moe_hawkeye::value_from_float(carries[index], -133);
        result.accepted = qrt_sm121_mantissa_parts::sum(carry, pairs, partials, &sum);
        result.magnitude = sum.value.magnitude; result.negative = sum.value.negative; result.exponent = sum.max_exponent;
        output[index] = result;
    }
}

unsigned seed = 0x3958192u;
unsigned random_word() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
float bf16(uint16_t v) { return qrt_sm121_native_product::from_bits(uint32_t(v) << 16u); }
bool same(const qrt_sm121_group16::AlignedSum& a, const qrt_sm121_group16::AlignedSum& b) {
    return a.max_exponent == b.max_exponent && a.value.magnitude == b.value.magnitude && a.value.negative == b.value.negative;
}
int main() {
    std::vector<uint16_t> input(kCases * 512u);
    std::vector<float> carries(kCases * kCells);
    std::vector<Cell> output(kCases * kCells);
    for (unsigned test = 0; test < kCases; ++test) {
        const unsigned spread = test % 4u == 0u ? 10u : 4u;
        for (unsigned i = 0; i < 512u; ++i) {
            const unsigned exponent = 120u + random_word() % spread;
            input[test * 512u + i] = uint16_t((random_word() & 0x807fu) | (exponent << 7u));
            if (test % 11u == 0u && i % 5u == 0u) input[test * 512u + i] = 0u;
        }
        for (unsigned i = 0; i < kCells; ++i) {
            const uint32_t bits = (random_word() & 0x807fffffu) | ((123u + test % 24u) << 23u);
            carries[test * kCells + i] = test % 3u == 0u ? 0.0f : qrt_sm121_native_product::from_bits(bits);
        }
    }
    uint16_t* device_input = nullptr; float* device_carry = nullptr; Cell* device_output = nullptr;
    auto check = [](hipError_t error) { if (error != hipSuccess) { std::fprintf(stderr, "HIP=%s\n", hipGetErrorString(error)); std::exit(1); } };
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7)) return 1;
    check(hipMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(uint16_t)));
    check(hipMalloc(reinterpret_cast<void**>(&device_carry), carries.size() * sizeof(float)));
    check(hipMalloc(reinterpret_cast<void**>(&device_output), output.size() * sizeof(Cell)));
    check(hipMemcpy(device_input, input.data(), input.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    check(hipMemcpy(device_carry, carries.data(), carries.size() * sizeof(float), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(matrix_probe, dim3(kCases), dim3(32u), 0u, nullptr, device_input, device_carry, device_output);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    check(hipMemcpy(output.data(), device_output, output.size() * sizeof(Cell), hipMemcpyDeviceToHost));
    check(hipFree(device_output)); check(hipFree(device_carry)); check(hipFree(device_input));
    unsigned accepted = 0, partial_bad = 0, host_bad = 0, device_bad = 0, status_bad = 0;
    for (unsigned index = 0; index < output.size(); ++index) {
        const unsigned test = index / kCells, row = index / 16u % 16u, column = index % 16u;
        uint32_t pairs[16], products[16]; float expected_parts[4]{};
        for (unsigned i = 0; i < 16u; ++i) {
            const uint16_t a = input[test * 512u + row * 16u + i], b = input[test * 512u + 256u + column * 16u + i];
            pairs[i] = uint32_t(a) | (uint32_t(b) << 16u);
            products[i] = qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a, b, -133));
            const uint16_t left[2] = {qrt_sm121_mantissa_parts::high(a), qrt_sm121_mantissa_parts::low(a)};
            const uint16_t right[2] = {qrt_sm121_mantissa_parts::high(b), qrt_sm121_mantissa_parts::low(b)};
            for (unsigned h = 0; h < 2u; ++h) for (unsigned l = 0; l < 2u; ++l) {
                volatile float product = bf16(left[h]) * bf16(right[l]);
                volatile float rounded = expected_parts[h * 2u + l] + product;
                expected_parts[h * 2u + l] = rounded;
            }
        }
        const auto carry = qrt_q1_moe_hawkeye::value_from_float(carries[index], -133);
        const auto expected = qrt_sm121_group16::sum_packed(carry, products);
        qrt_sm121_group16::AlignedSum host{};
        const bool valid = qrt_sm121_mantissa_parts::sum(carry, pairs, output[index].partials, &host);
        if (valid != bool(output[index].accepted)) ++status_bad;
        if (!valid) continue;
        ++accepted;
        bool bad_parts = false;
        for (unsigned i = 0; i < 4u; ++i) if (expected_parts[i] != output[index].partials[i]) bad_parts = true;
        partial_bad += unsigned(bad_parts);
        host_bad += unsigned(!same(host, expected));
        qrt_sm121_group16::AlignedSum device{{output[index].magnitude, bool(output[index].negative)}, output[index].exponent};
        const bool bad_device = !same(device, expected);
        device_bad += unsigned(bad_device);
        if ((bad_parts || bad_device) && partial_bad + device_bad <= 4u) {
            std::printf("DIFF index=%u carry=%08x accepted=%u host=%u device=%u expected=%u exponent=%d\n", index,
                qrt_sm121_native_product::float_bits(carries[index]), unsigned(valid), host.value.magnitude, device.value.magnitude, expected.value.magnitude, expected.max_exponent);
            for (unsigned i = 0; i < 4u; ++i) std::printf("PART %u expected=%08x native=%08x\n", i,
                qrt_sm121_native_product::float_bits(expected_parts[i]), qrt_sm121_native_product::float_bits(output[index].partials[i]));
            for (unsigned pair : pairs) std::printf("%08x ", pair);
            std::printf("\n");
        }
    }
    std::printf("{\"kind\":\"mantissa_wmma_selftest\",\"cells\":%u,\"accepted\":%u,\"partial_mismatches\":%u,\"host_sum_mismatches\":%u,\"device_sum_mismatches\":%u,\"eligibility_mismatches\":%u,\"inference_acceptance\":false}\n",
        unsigned(output.size()), accepted, partial_bad, host_bad, device_bad, status_bad);
    return accepted && !partial_bad && !host_bad && !device_bad && !status_bad ? 0 : 2;
}
