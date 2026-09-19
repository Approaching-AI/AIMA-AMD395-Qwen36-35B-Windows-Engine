#include <hip/hip_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_q1_gdn.h"

void check(hipError_t code) { if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code)); }
template<class T> std::vector<T> read(const char* path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count * sizeof(T))) throw std::runtime_error(path);
    std::vector<T> values(count); file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("short read"); return values;
}
std::vector<float> floats(const std::vector<uint16_t>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) result[i] = qrt_sm121_q1::widen(input[i]);
    return result;
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() { for (auto p : pointers) (void)hipFree(p); }
    template<class T> T* upload(const std::vector<T>& values) {
        T* p = nullptr; check(hipMalloc(reinterpret_cast<void**>(&p), values.size() * sizeof(T)));
        pointers.push_back(p); check(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice)); return p;
    }
};
int main(int argc, char** argv) try {
    if (argc != 14) throw std::runtime_error("conv a b before after core g-table beta exp2 rsqrt sqrt reciprocal packed-mode");
    const std::string mode = argv[13]; if (mode != "0" && mode != "1") throw std::runtime_error("packed mode");
    const bool packed = mode == "1";
    const auto conv = floats(read<uint16_t>(argv[1], 8192));
    const auto a = floats(read<uint16_t>(argv[2], 32)), b = floats(read<uint16_t>(argv[3], 32));
    const auto before = read<float>(argv[4], 524288), expected = read<float>(argv[5], 524288);
    const auto expected_core = read<uint16_t>(argv[6], 4096);
    const auto gates = read<float>(argv[7], 2097152), beta = read<float>(argv[8], 65536);
    const auto exp2 = read<unsigned char>(argv[9], qrt_sm121_exp2::table_bytes);
    const auto rsqrt = read<unsigned char>(argv[10], qrt_sm121_rsqrt::table_bytes);
    const auto sqrt = read<unsigned char>(argv[11], qrt_sm121_sqrt::table_bytes);
    const auto reciprocal = read<unsigned char>(argv[12], qrt_sm121_attention_rcp::table_bytes);
    if (!qrt_sm121_exp2::valid_layout(exp2.data(), exp2.size()) || !qrt_sm121_rsqrt::valid_layout(rsqrt.data(), rsqrt.size()) ||
        !qrt_sm121_sqrt::valid_layout(sqrt.data(), sqrt.size()) || !qrt_sm121_attention_rcp::valid_layout(reciprocal.data(), reciprocal.size()))
        throw std::runtime_error("table layout");
    Scratch scratch;
    const auto dc = scratch.upload(conv), da = scratch.upload(a), db = scratch.upload(b);
    const auto dg = scratch.upload(gates), dbe = scratch.upload(beta);
    const auto de = scratch.upload(exp2), dr = scratch.upload(rsqrt), dsq = scratch.upload(sqrt), drec = scratch.upload(reciprocal);
    bool passed = true;
    for (unsigned layout = 0; layout < 2; ++layout) {
        constexpr size_t guard = 16;
        constexpr uint32_t sentinel = 0x4d2468e0u;
        const float marker = qrt_sm121_exp2::value(sentinel);
        std::vector<float> state(before.size() + guard * 2, marker), core(4096 + guard * 2, marker);
        for (size_t h = 0; h < 32; ++h) for (size_t v = 0; v < 128; ++v) for (size_t k = 0; k < 128; ++k) {
            const size_t destination = layout ? (h * 128 + k) * 128 + v : (h * 128 + v) * 128 + k;
            state[guard + destination] = before[(h * 128 + v) * 128 + k];
        }
        Scratch temporary;
        auto dstate = temporary.upload(state), dcore = temporary.upload(core);
        hipLaunchKernelGGL(qrt_sm121_q1::recurrent, dim3(32), dim3(128), 0, nullptr,
            dc, da, db, dstate + guard, layout != 0, dcore + guard, nullptr, nullptr, nullptr,
            dg, dbe, de, dr, packed, dsq, drec);
        check(hipGetLastError()); check(hipDeviceSynchronize());
        check(hipMemcpy(state.data(), dstate, state.size() * sizeof(float), hipMemcpyDeviceToHost));
        check(hipMemcpy(core.data(), dcore, core.size() * sizeof(float), hipMemcpyDeviceToHost));
        size_t state_bad = 0, core_bad = 0, guard_bad = 0;
        double maximum_error = 0;
        for (size_t h = 0; h < 32; ++h) for (size_t v = 0; v < 128; ++v) for (size_t k = 0; k < 128; ++k) {
            const float actual = state[guard + (layout ? (h * 128 + k) * 128 + v : (h * 128 + v) * 128 + k)];
            const float wanted = expected[(h * 128 + v) * 128 + k];
            state_bad += qrt_sm121_exp2::bits(actual) != qrt_sm121_exp2::bits(wanted);
            maximum_error = std::max(maximum_error, std::abs(double(actual) - wanted));
        }
        for (size_t i = 0; i < 4096; ++i)
            core_bad += qrt_sm121_exp2::bits(core[guard + i]) != (uint32_t(expected_core[i]) << 16u);
        for (size_t i = 0; i < guard; ++i) {
            guard_bad += qrt_sm121_exp2::bits(state[i]) != sentinel;
            guard_bad += qrt_sm121_exp2::bits(state[state.size() - 1 - i]) != sentinel;
            guard_bad += qrt_sm121_exp2::bits(core[i]) != sentinel;
            guard_bad += qrt_sm121_exp2::bits(core[core.size() - 1 - i]) != sentinel;
        }
        const bool current = state_bad == 0 && core_bad == 0 && guard_bad == 0; passed &= current;
        std::cout << "{\"kind\":\"original_q1_recurrence_native_replay\",\"packed_decode\":" << (packed ? "true" : "false")
            << ",\"key_major\":" << (layout ? "true" : "false") << ",\"state_f32_bit_mismatches\":" << state_bad
            << ",\"core_bf16_mismatches\":" << core_bad << ",\"guard_mismatches\":" << guard_bad
            << ",\"maximum_state_absolute_error\":" << maximum_error << ",\"passed\":" << (current ? "true" : "false")
            << ",\"inference_acceptance\":false}\n";
    }
    return passed ? 0 : 1;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
