#include <hip/hip_runtime.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_q1_gdn.h"

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template <typename T> std::vector<T> read(const char *path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T)))
        throw std::runtime_error(std::string("input size: ") + path);
    std::vector<T> values(count);
    file.seekg(0); file.read(reinterpret_cast<char *>(values.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("short input read");
    return values;
}
struct Scratch {
    std::vector<void *> pointers;
    ~Scratch() { for (void *p : pointers) (void)hipFree(p); }
    template <typename T> T *upload(const std::vector<T> &values) {
        T *p = nullptr; check(hipMalloc(reinterpret_cast<void **>(&p), values.size() * sizeof(T)));
        pointers.push_back(p);
        check(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
        return p;
    }
};
template <typename T> void save(const std::string &path, const std::vector<T> &values) {
    std::ifstream existing(path, std::ios::binary);
    if (existing.good()) throw std::runtime_error("existing output");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(T));
    if (!file) throw std::runtime_error("output write failed");
}

int main(int argc, char **argv) try {
    if (argc != 17) throw std::runtime_error(
        "ring qkv a b state weights g-table beta-table exp2 rsqrt silu expected-core expected-state expected-diagnostic output-prefix position");
    const size_t position = std::stoull(argv[16]);
    if (position < 3 || position > 263679) throw std::runtime_error("position bound");
    auto ring = read<float>(argv[1], 32768), qkv = read<float>(argv[2], 8192);
    auto a = read<float>(argv[3], 32), b = read<float>(argv[4], 32);
    auto state = read<float>(argv[5], 524288);
    auto weights = read<uint16_t>(argv[6], 32768);
    auto g = read<float>(argv[7], 2097152), beta = read<float>(argv[8], 65536);
    auto exp2 = read<unsigned char>(argv[9], qrt_sm121_exp2::table_bytes);
    auto rsqrt = read<unsigned char>(argv[10], qrt_sm121_rsqrt::table_bytes);
    auto silu = read<unsigned char>(argv[11], qrt_sm121_silu::table_bytes);
    auto expected_core = read<uint16_t>(argv[12], 4096);
    auto expected_state = read<float>(argv[13], 524288);
    auto expected_diagnostic = read<float>(argv[14], 16480);
    if (!qrt_sm121_exp2::valid_layout(exp2.data(), exp2.size()) ||
        !qrt_sm121_rsqrt::valid_layout(rsqrt.data(), rsqrt.size()) ||
        !qrt_sm121_silu::valid_layout(silu.data(), silu.size()))
        throw std::runtime_error("table layout");
    Scratch scratch;
    auto dr = scratch.upload(ring), dq = scratch.upload(qkv), da = scratch.upload(a), db = scratch.upload(b);
    auto ds = scratch.upload(state); auto dw = scratch.upload(weights);
    auto dg = scratch.upload(g), dbe = scratch.upload(beta);
    auto de = scratch.upload(exp2), drr = scratch.upload(rsqrt), dsi = scratch.upload(silu);
    std::vector<float> conv(8192), core(4096), diagnostic(16480);
    auto dc = scratch.upload(conv), dout = scratch.upload(core), ddiag = scratch.upload(diagnostic);
    hipLaunchKernelGGL(qrt_sm121_q1::convolution<float>, dim3(32), dim3(256), 0, nullptr,
                      dq, dr, dw, dc, position, dsi);
    check(hipGetLastError());
    hipLaunchKernelGGL(qrt_sm121_q1::recurrent, dim3(32), dim3(128), 0, nullptr,
                      dc, da, db, ds, true, dout, nullptr, nullptr, ddiag, dg, dbe, de, drr);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    check(hipMemcpy(conv.data(), dc, conv.size() * 4, hipMemcpyDeviceToHost));
    check(hipMemcpy(core.data(), dout, core.size() * 4, hipMemcpyDeviceToHost));
    check(hipMemcpy(state.data(), ds, state.size() * 4, hipMemcpyDeviceToHost));
    check(hipMemcpy(diagnostic.data(), ddiag, diagnostic.size() * 4, hipMemcpyDeviceToHost));
    size_t core_diff = 0, state_diff = 0, diagnostic_diff = 0;
    for (size_t i = 0; i < core.size(); ++i)
        core_diff += qrt_sm121_q1::bf16(core[i]) != expected_core[i];
    for (size_t h = 0; h < 32; ++h) for (size_t v = 0; v < 128; ++v) for (size_t k = 0; k < 128; ++k)
        state_diff += qrt_sm121_exp2::bits(state[(h * 128 + k) * 128 + v]) !=
                      qrt_sm121_exp2::bits(expected_state[(h * 128 + v) * 128 + k]);
    for (size_t i = 0; i < diagnostic.size(); ++i)
        diagnostic_diff += qrt_sm121_exp2::bits(diagnostic[i]) != qrt_sm121_exp2::bits(expected_diagnostic[i]);
    save(std::string(argv[15]) + "-conv-f32.bin", conv);
    save(std::string(argv[15]) + "-core-f32.bin", core);
    save(std::string(argv[15]) + "-state-key-major-f32.bin", state);
    save(std::string(argv[15]) + "-diagnostic-f32.bin", diagnostic);
    const bool passed = core_diff == 0 && state_diff == 0 && diagnostic_diff == 0;
    std::cout << "{\"kind\":\"q1_sm121_native_conv_recurrent_replay\",\"passed\":"
              << (passed ? "true" : "false") << ",\"core_mismatches\":" << core_diff
              << ",\"state_mismatches\":" << state_diff << ",\"diagnostic_mismatches\":"
              << diagnostic_diff << ",\"inference_acceptance\":false}\n";
    return passed ? 0 : 1;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
