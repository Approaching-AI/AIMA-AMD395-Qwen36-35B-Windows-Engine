// Numerical diagnostic using functions extracted verbatim from the live
// provider. The reference and prior native outputs are host-only comparisons.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>
#include "sm121_rsqrt_table.h"
#include "postnorm_live_extract.h"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t tokens = 7169, width = 2048, elements = tokens * width;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> std::vector<T> read(const char* path, size_t n) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(n * sizeof(T)))
        throw std::runtime_error("invalid capture length");
    std::vector<T> result(n); file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), n * sizeof(T)))
        throw std::runtime_error("short capture read");
    return result;
}
template<class T> void write(const std::string& path, const std::vector<T>& v) {
    HANDLE f = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("output already exists");
    DWORD written = 0, bytes = DWORD(v.size() * sizeof(T));
    bool ok = WriteFile(f, v.data(), bytes, &written, nullptr) && written == bytes;
    CloseHandle(f); if (!ok) throw std::runtime_error("short capture write");
}
float fp32(uint16_t h) { uint32_t bits = uint32_t(h) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
uint16_t bf16(float f) { uint32_t b; std::memcpy(&b, &f, 4); return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16); }
struct Device {
    void* data = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&data, bytes)); }
    ~Device() { if (data) (void)hipFree(data); }
    template<class T> T* as() { return static_cast<T*>(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

__global__ void normalization_scalars(const float* residual, const uint16_t* update,
                                    const uint8_t* correction, const unsigned char* table,
                                    float* output) {
    __shared__ float partial[kThreads];
    unsigned t = blockIdx.x, lane = threadIdx.x;
    float values[8];
    for (unsigned i = 0; i < 8; ++i) {
        size_t index = size_t(t) * width + lane * 8u + i;
        values[i] = __fadd_rn(device_bf16_round_to_float(residual[index]), device_bf16_to_float(update[index]));
    }
    float sum = vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(values), partial, lane);
    if (lane == 0) {
        float variance = __fadd_rn(sum / 2048.0f, 1.0e-6f);
        output[t * 4u] = sum; output[t * 4u + 1u] = variance;
        output[t * 4u + 2u] = device_sm121_rsqrt_from_gfx1151(variance, correction);
        output[t * 4u + 3u] = qrt_sm121_rsqrt::evaluate(table, variance);
    }
}
}

int main(int argc, char** argv) try {
    if (argc != 9) throw std::runtime_error("residual update weight reference native delta2 rsqrt_table output_prefix");
    auto start = std::chrono::steady_clock::now();
    hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, 0));
    if (std::string(prop.gcnArchName).find("gfx1151") != 0) throw std::runtime_error("requires gfx1151");
    auto residual = read<uint16_t>(argv[1], elements), update = read<uint16_t>(argv[2], elements);
    auto weight = read<uint16_t>(argv[3], width), reference = read<uint16_t>(argv[4], elements);
    auto native = read<uint16_t>(argv[5], elements);
    auto correction = read<uint8_t>(argv[6], 4194304);
    auto table = read<unsigned char>(argv[7], qrt_sm121_rsqrt::table_bytes);
    if (!qrt_sm121_rsqrt::valid_layout(table.data(), table.size())) throw std::runtime_error("invalid rsqrt table");
    std::vector<float> input(elements), output(elements), scalars(tokens * 4u);
    for (size_t i = 0; i < elements; ++i) input[i] = fp32(residual[i]);
    Device dr(elements * 4), du(elements * 2), dw(width * 2), dc(correction.size()), dt(table.size());
    Device dh(elements * 4), dout(elements * 4), ds(scalars.size() * 4);
    check(hipMemcpy(dr.data, input.data(), elements * 4, hipMemcpyHostToDevice));
    check(hipMemcpy(du.data, update.data(), elements * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(dw.data, weight.data(), width * 2, hipMemcpyHostToDevice));
    check(hipMemcpy(dc.data, correction.data(), correction.size(), hipMemcpyHostToDevice));
    check(hipMemcpy(dt.data, table.data(), table.size(), hipMemcpyHostToDevice));
    hipEvent_t begin{}, end{}; check(hipEventCreate(&begin)); check(hipEventCreate(&end));
    for (unsigned variant = 0; variant < 5; ++variant) {
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 30)
            throw std::runtime_error("aggregate deadline exceeded");
        check(hipEventRecord(begin));
#define LAUNCH(kernel, coefficients) hipLaunchKernelGGL(kernel, dim3(tokens), dim3(kThreads), 0, 0, \
    dr.as<float>(), du.as<uint16_t>(), dw.as<uint16_t>(), dh.as<float>(), dout.as<float>(), unsigned(tokens), coefficients)
        if (variant == 0) { LAUNCH(output_bf16_residual_postnorm_vllm_kernel, dc.as<uint8_t>()); }
        if (variant == 1) { LAUNCH(postnorm_table1_separate0_kernel, dt.as<uint8_t>()); }
        if (variant == 2) { LAUNCH(postnorm_table0_separate1_kernel, dc.as<uint8_t>()); }
        if (variant == 3) { LAUNCH(postnorm_table1_separate1_kernel, dt.as<uint8_t>()); }
        if (variant == 4) hipLaunchKernelGGL(normalization_scalars, dim3(tokens), dim3(kThreads), 0, 0,
            dr.as<float>(), du.as<uint16_t>(), dc.as<uint8_t>(), dt.as<unsigned char>(), ds.as<float>());
#undef LAUNCH
        check(hipGetLastError()); check(hipEventRecord(end)); check(hipEventSynchronize(end));
        float ms; check(hipEventElapsedTime(&ms, begin, end));
        if (!std::isfinite(ms) || ms > 3000) throw std::runtime_error("dispatch deadline exceeded");
        if (variant == 4) {
            check(hipMemcpy(scalars.data(), ds.data, scalars.size() * 4, hipMemcpyDeviceToHost));
            size_t differences = 0;
            for (size_t t = 0; t < tokens; ++t) differences += std::memcmp(&scalars[t * 4 + 2], &scalars[t * 4 + 3], 4) != 0;
            write(std::string(argv[8]) + "-scalars-f32.bin", scalars);
            std::cout << "{\"kind\":\"postnorm_scalars\",\"rsqrt_bit_mismatches\":" << differences
                      << ",\"tokens\":" << tokens << ",\"kernel_ms\":" << ms << "}\n";
        } else {
            check(hipMemcpy(output.data(), dout.data, elements * 4, hipMemcpyDeviceToHost));
            std::vector<uint16_t> rounded(elements);
            size_t gb10_bad = 0, native_bad = 0, nonfinite = 0;
            for (size_t i = 0; i < elements; ++i) {
                rounded[i] = bf16(output[i]); gb10_bad += rounded[i] != reference[i];
                native_bad += rounded[i] != native[i]; nonfinite += !std::isfinite(output[i]);
            }
            write(std::string(argv[8]) + "-variant" + std::to_string(variant) + "-bf16.bin", rounded);
            std::cout << "{\"kind\":\"postnorm_capture_replay\",\"variant\":" << variant
                      << ",\"elements\":" << elements << ",\"gb10_bf16_mismatches\":" << gb10_bad
                      << ",\"prior_native_bf16_mismatches\":" << native_bad << ",\"nonfinite\":" << nonfinite
                      << ",\"kernel_ms\":" << ms << ",\"inference_acceptance\":false}" << std::endl;
            if (variant == 0 && native_bad) throw std::runtime_error("verbatim replay differs from actual model capture");
            if (nonfinite) throw std::runtime_error("nonfinite replay output");
        }
    }
    check(hipEventDestroy(begin)); check(hipEventDestroy(end));
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
