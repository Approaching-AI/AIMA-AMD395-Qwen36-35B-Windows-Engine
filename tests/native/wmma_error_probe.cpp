// Isolated zero-C arithmetic probe. No model or runtime dispatch is changed.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace {
constexpr unsigned guard = 128u;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Inputs {
    std::array<uint32_t, 8> header{};
    std::vector<uint16_t> words;
    explicit Inputs(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "input open"); const auto bytes = file.tellg(); file.seekg(0);
        file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
        require(bool(file) && header[0] == 0x574d4931u && header[1] == 1u && header[3] == 12u &&
            header[4] >= 1u && header[4] <= 1024u && header[5] == 4u && header[6] == 16u && header[7] == 1u &&
            header[2] == header[3] * header[4] * header[5], "input header");
        require(bytes == std::streamoff(sizeof(header) + size_t(header[2]) * 1024u), "input span");
        words.resize(size_t(header[2]) * 512u + guard * 2u, 0xa5a5u);
        file.read(reinterpret_cast<char*>(words.data() + guard), size_t(header[2]) * 1024u);
        require(bool(file), "input read");
        for (size_t i = guard; i < words.size() - guard; ++i) {
            const unsigned exponent = (words[i] >> 7u) & 255u;
            require(!(words[i] & 0x7fffu) || (exponent >= 80u && exponent <= 174u), "input domain");
        }
    }
};
#if defined(__HIPCC__)
using U16x16 = unsigned short __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
__global__ void probe(const uint16_t* left, const uint16_t* right, uint32_t* output) {
    const unsigned lane = threadIdx.x, tile = blockIdx.x;
    U16x16 a{}, b{};
#pragma unroll
    for (unsigned k = 0u; k < 16u; ++k) {
        a[k] = left[tile * 256u + lane % 16u * 16u + k];
        b[k] = right[tile * 256u + lane % 16u * 16u + k];
    }
    const F32x8 zero{};
    const auto result = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, zero);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i)
        output[tile * 256u + (2u * i + lane / 16u) * 16u + lane % 16u] = __float_as_uint(result[i]);
}
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* data = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&data, bytes)); }
    ~Device() { if (data && hipFree(data) != hipSuccess) std::abort(); }
    template<class T> T* as() { return static_cast<T*>(data); }
};
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto result = hipEventQuery(event);
        if (result == hipSuccess) break;
        if (result != hipErrorNotReady) check(result);
        require(std::chrono::steady_clock::now() < deadline, "native completion deadline"); std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
void run(Inputs& input, const char* path) {
    const unsigned tiles = input.header[2], cells = tiles * 256u;
    Device operands(input.words.size() * 2u), destination((size_t(cells) + guard * 2u) * 4u);
    check(hipMemcpy(operands.data, input.words.data(), input.words.size() * 2u, hipMemcpyHostToDevice));
    std::vector<uint32_t> output(size_t(cells) + guard * 2u), baseline;
    for (unsigned repeat = 0u; repeat < 2u; ++repeat) {
        check(hipMemset(destination.data, 0xa5, output.size() * 4u));
        hipLaunchKernelGGL(probe, dim3(tiles), dim3(32u), 0u, nullptr,
            operands.as<uint16_t>() + guard, operands.as<uint16_t>() + guard + cells, destination.as<uint32_t>() + guard);
        check(hipGetLastError()); finish();
        check(hipMemcpy(output.data(), destination.data, output.size() * 4u, hipMemcpyDeviceToHost));
        for (unsigned i = 0u; i < guard; ++i)
            require(output[i] == 0xa5a5a5a5u && output[guard + cells + i] == 0xa5a5a5a5u, "output redzone");
        for (unsigned i = 0u; i < cells; ++i) require((output[guard + i] & 0x7f800000u) != 0x7f800000u, "nonfinite result");
        if (repeat) require(output == baseline, "repeated native result changed"); else baseline = output;
    }
    std::vector<uint16_t> after(input.words.size());
    check(hipMemcpy(after.data(), operands.data, after.size() * 2u, hipMemcpyDeviceToHost));
    require(after == input.words, "input or input redzone changed");
    require(!std::ifstream(path, std::ios::binary).good(), "output exists");
    std::ofstream file(path, std::ios::binary); require(bool(file), "output open");
    auto header = input.header; header[0] = 0x574d4f31u;
    file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    file.write(reinterpret_cast<const char*>(output.data() + guard), size_t(cells) * 4u);
    file.close(); require(bool(file), "output write");
    std::printf("{\"kind\":\"native_wmma_error_capture\",\"tiles\":%u,\"cells\":%u,\"repeats\":2,\"repeat_bit_parity\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"product_dispatch_changed\":false,\"inference_acceptance\":false}\n", tiles, cells);
}
#endif
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: wmma-error-probe input.bin output.bin (host: --check-input)");
        Inputs input(argv[1]);
        if (!std::strcmp(argv[2], "--check-input")) {
            std::printf("{\"kind\":\"wmma_error_input_check\",\"tiles\":%u,\"native_wmma_checked\":false}\n", input.header[2]); return 0;
        }
#if defined(__HIPCC__)
        run(input, argv[2]); return 0;
#else
        throw std::runtime_error("native HIP build required");
#endif
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
