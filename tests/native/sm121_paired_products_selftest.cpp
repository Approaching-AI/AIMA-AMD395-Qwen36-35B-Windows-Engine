#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_paired_products.h"
#include "../../native/providers/moe_accumulator/sm121_group16_modulo.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

constexpr unsigned kCount = 65536u * 16u + 7u, kGuard = 32u;
using Products = qrt_sm121_paired_products::Products;
__global__ void paired_probe(const uint32_t* left, const uint32_t* right, Products* output) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kCount) output[index] = qrt_sm121_paired_products::multiply(left[index], right[index]);
}

int main() {
    auto check = [](hipError_t error) { if (error != hipSuccess) {
        std::fprintf(stderr, "HIP=%s\n", hipGetErrorString(error)); std::exit(1);
    } };
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    if (std::strncmp(device.gcnArchName, "gfx1151", 7)) return 1;
    std::vector<uint32_t> left(kCount + 2u * kGuard, 0x19239581u), right(left.size(), 0x61925128u);
    std::vector<Products> output(left.size());
    unsigned seed = 0x3952026u;
    auto random = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    const uint16_t controls[] = {0,0x8000,1,0x7f,0x80,0x807f,0x3f80,0xbf80,0x3fff,0xbfff,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0xffff};
    for (unsigned i = 0; i < kCount; ++i) {
        left[i + kGuard] = (i / 16u % 65536u) | (random() & 0xffff0000u);
        right[i + kGuard] = uint32_t(controls[i % 16u]) | (random() & 0xffff0000u);
    }
    uint32_t *a = nullptr, *b = nullptr; Products* c = nullptr;
    const size_t input_bytes = left.size() * sizeof(uint32_t), output_bytes = output.size() * sizeof(Products);
    check(hipMalloc(reinterpret_cast<void**>(&a), input_bytes));
    check(hipMalloc(reinterpret_cast<void**>(&b), input_bytes));
    check(hipMalloc(reinterpret_cast<void**>(&c), output_bytes));
    check(hipMemcpy(a, left.data(), input_bytes, hipMemcpyHostToDevice));
    check(hipMemcpy(b, right.data(), input_bytes, hipMemcpyHostToDevice));
    check(hipMemset(c, 0xa5, output_bytes));
    hipLaunchKernelGGL(paired_probe, dim3((kCount + 255u) / 256u), dim3(256u), 0u, nullptr, a + kGuard, b + kGuard, c + kGuard);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    check(hipMemcpy(output.data(), c, output_bytes, hipMemcpyDeviceToHost));
    std::vector<uint32_t> after(left.size());
    check(hipMemcpy(after.data(), a, input_bytes, hipMemcpyDeviceToHost));
    bool immutable = after == left;
    check(hipMemcpy(after.data(), b, input_bytes, hipMemcpyDeviceToHost)); immutable = immutable && after == right;
    check(hipFree(c)); check(hipFree(b)); check(hipFree(a));
    unsigned mismatches = 0, redzones = 0;
    for (unsigned i = 0; i < output.size(); ++i) {
        if (i < kGuard || i >= kGuard + kCount) {
            redzones += unsigned(output[i].low != 0xa5a5a5a5u || output[i].high != 0xa5a5a5a5u);
        } else {
            const uint32_t av = left[i], bv = right[i];
            auto reference = [](uint16_t x, uint16_t y) {
                return qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(x, y, -133));
            };
            mismatches += unsigned(output[i].low != reference(uint16_t(av), uint16_t(bv)));
            mismatches += unsigned(output[i].high != reference(uint16_t(av >> 16u), uint16_t(bv >> 16u)));
        }
    }
    std::printf("{\"kind\":\"paired_integer_products\",\"products\":%u,\"mismatches\":%u,\"redzone_mismatches\":%u,\"inputs_immutable\":%s,\"inference_acceptance\":false}\n",
        kCount * 2u, mismatches, redzones, immutable ? "true" : "false");
    return !mismatches && !redzones && immutable ? 0 : 2;
}
