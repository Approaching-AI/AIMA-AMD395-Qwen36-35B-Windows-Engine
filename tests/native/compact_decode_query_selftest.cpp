// Complete single-query original-score/PV comparisons; no model acceptance.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../native/providers/ck_fmha/blackwell_attention.h"

namespace {
namespace attention = qrt_blackwell_attention;
constexpr size_t redzone = 256u;
constexpr unsigned output_start = 3u, output_rows = 6u;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void finish() { check(hipDeviceSynchronize()); }
uint32_t bits(float value) { uint32_t word; std::memcpy(&word, &value, 4u); return word; }
uint16_t bf16(float value) {
    const uint32_t word = bits(value);
    if ((word & 0x7f800000u) == 0x7f800000u)
        return uint16_t((word >> 16u) | ((word & 0x7fffffu) ? 0x40u : 0u));
    return uint16_t((word + 0x7fffu + ((word >> 16u) & 1u)) >> 16u);
}
template<class T> std::vector<T> read_file(const char* path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != std::streamoff(count * sizeof(T)))
        throw std::runtime_error("fixture extent differs: " + std::string(path));
    input.seekg(0); std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(T)));
    if (!input) throw std::runtime_error("fixture read incomplete");
    return values;
}
struct Guarded {
    unsigned char* base = nullptr;
    size_t bytes;
    explicit Guarded(size_t size): bytes(size) {
        check(hipMalloc(reinterpret_cast<void**>(&base), bytes + 2u * redzone));
        try { reset(); }
        catch (...) { (void)hipFree(base); base = nullptr; throw; }
    }
    ~Guarded() { if (base) { (void)hipDeviceSynchronize(); (void)hipFree(base); } }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    template<class T> T* as() const { return reinterpret_cast<T*>(base + redzone); }
    void reset() { check(hipMemset(base, 0xa5, bytes + 2u * redzone)); }
    void guards() const {
        std::array<unsigned char, redzone> data{};
        for (size_t offset: {size_t(0), bytes + redzone}) {
            check(hipMemcpy(data.data(), base + offset, redzone, hipMemcpyDeviceToHost));
            for (auto value: data) if (value != 0xa5u) throw std::runtime_error("redzone changed");
        }
    }
    template<class T> void put(const std::vector<T>& data) {
        if (data.size() * sizeof(T) != bytes) throw std::runtime_error("upload extent");
        check(hipMemcpy(as<T>(), data.data(), bytes, hipMemcpyHostToDevice));
    }
    template<class T> std::vector<T> get() const {
        if (bytes % sizeof(T)) throw std::runtime_error("download alignment");
        std::vector<T> result(bytes / sizeof(T));
        check(hipMemcpy(result.data(), as<T>(), bytes, hipMemcpyDeviceToHost));
        guards(); return result;
    }
    template<class T> void immutable(const std::vector<T>& expected) const {
        if (expected.size() * sizeof(T) != bytes) throw std::runtime_error("immutable extent");
        constexpr size_t block = 4u * 1024u * 1024u;
        std::vector<unsigned char> observed(std::min(bytes, block));
        for (size_t first = 0; first < bytes; first += block) {
            const size_t count = std::min(block, bytes - first);
            check(hipMemcpy(observed.data(), as<unsigned char>() + first, count, hipMemcpyDeviceToHost));
            if (std::memcmp(observed.data(), reinterpret_cast<const unsigned char*>(expected.data()) + first, count))
                throw std::runtime_error("input changed");
        }
        guards();
    }
};
__global__ void verify_full_query(const uint16_t* full, const uint16_t* current,
    size_t words, unsigned first, unsigned* bad) {
    const size_t begin = size_t(first) * 4096u;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += size_t(gridDim.x) * blockDim.x) {
        const uint16_t expected = i >= begin && i < begin + 4096u ? current[i - begin] : uint16_t(0xffffu);
        if (full[i] != expected) atomicAdd(bad, 1u);
    }
}
struct Outputs {
    Guarded scores, value, accumulator, denominator;
    explicit Outputs(unsigned tokens): scores(size_t(tokens) * 16u * 4u),
        value(output_rows * 4096u * 4u), accumulator(output_rows * 4096u * 4u),
        denominator(output_rows * 16u * 4u) {}
    void reset() { scores.reset(); value.reset(); accumulator.reset(); denominator.reset(); }
    void guards() { scores.guards(); value.guards(); accumulator.guards(); denominator.guards(); }
};
struct Snapshot {
    std::vector<uint32_t> scores, value, accumulator, denominator;
};
Snapshot snapshot(Outputs& output) {
    return {output.scores.get<uint32_t>(), output.value.get<uint32_t>(),
        output.accumulator.get<uint32_t>(), output.denominator.get<uint32_t>()};
}
void validate_output(const Snapshot& data, const std::vector<uint16_t>& golden) {
    for (unsigned row = 0; row < output_rows; ++row) {
        for (unsigned f = 0; f < 4096u; ++f) {
            const size_t cell = size_t(row) * 4096u + f;
            if (row != output_start) {
                if (data.value[cell] != 0xa5a5a5a5u || data.accumulator[cell] != 0xa5a5a5a5u)
                    throw std::runtime_error("inactive output changed");
            } else {
                float value, accumulator;
                std::memcpy(&value, &data.value[cell], 4u);
                std::memcpy(&accumulator, &data.accumulator[cell], 4u);
                if (!std::isfinite(value) || !std::isfinite(accumulator))
                    throw std::runtime_error("nonfinite output");
                if (!golden.empty() && bf16(value) != golden[f])
                    throw std::runtime_error("original GB10 context differs");
            }
        }
        for (unsigned head = 0; head < 16u; ++head) {
            const auto word = data.denominator[size_t(row) * 16u + head];
            float value; std::memcpy(&value, &word, 4u);
            if (row == output_start ? !std::isfinite(value) || !(value > 0.0f) : word != 0xa5a5a5a5u)
                throw std::runtime_error("denominator or its unused rows changed");
        }
    }
}
void run_case(unsigned first, unsigned family, const std::vector<uint16_t>& q,
    const std::vector<uint16_t>& k, const std::vector<uint16_t>& v,
    const std::vector<uint16_t>& golden, const unsigned char* exp, const unsigned char* rcp) {
    const unsigned tokens = first + 1u;
    if (!first || tokens > attention::kSplitMaxTokens || q.size() != 4096u ||
        k.size() != size_t(tokens) * 512u || v.size() != k.size() || (!golden.empty() && golden.size() != 4096u))
        throw std::runtime_error("case extent");
    Guarded full(size_t(tokens) * 8192u), compact(8192u), dk(k.size() * 2u), dv(v.size() * 2u), bad(4u);
    compact.put(q); dk.put(k); dv.put(v);
    check(hipMemset(full.as<unsigned char>(), 0xff, full.bytes));
    check(hipMemcpy(full.as<uint16_t>() + size_t(first) * 4096u, q.data(), 8192u, hipMemcpyHostToDevice));
    Outputs output(tokens);
    Snapshot reference;
    std::array<double, 4> complete_ms{};
    // Alternate the call order after the first comparison, refreshing all
    // outputs at the same addresses. Neither arm consumes previous scores.
    for (unsigned attempt = 0; attempt < 4u; ++attempt) {
        const bool relative = attempt == 1u || attempt == 2u;
        output.reset(); finish();
        const auto begin = std::chrono::steady_clock::now();
        check(hipError_t(attention::launch_queries(relative ? compact.as<uint16_t>() : full.as<uint16_t>(),
            dk.as<uint16_t>(), dv.as<uint16_t>(), output.value.as<float>(), nullptr,
            first, 1u, output_start, exp, output.accumulator.as<float>(), output.denominator.as<float>(),
            true, rcp, 2u, output.scores.as<float>(), output.scores.bytes / 4u,
            nullptr, nullptr, nullptr, 0u, false, nullptr, nullptr, 0u, nullptr, nullptr,
            nullptr, 0u, 1u, 1u, false, false, false, 0u, false, nullptr, false, false,
            nullptr, nullptr, relative ? first : 0u)));
        finish();
        complete_ms[attempt] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        const auto data = snapshot(output);
        validate_output(data, golden);
        if (!attempt) reference = data;
        else if (data.scores != reference.scores || data.value != reference.value ||
            data.accumulator != reference.accumulator || data.denominator != reference.denominator)
            throw std::runtime_error("complete original score or raw PV surface differs");
        output.guards();
    }
    unsigned cpu_dots = 0u;
    for (unsigned head = 0; head < 16u; ++head) for (unsigned part = 0; part < 8u; ++part) {
        const unsigned key = unsigned(uint64_t(first) * part / 7u);
        const float expected = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
            q.data() + size_t(head) * 256u, k.data() + (size_t(key) * 2u + head / 8u) * 256u, 256u) * attention::kExactScale;
        if (reference.scores[size_t(head) * tokens + key] != bits(expected))
            throw std::runtime_error("independent original CPU score differs");
        ++cpu_dots;
    }
    compact.immutable(q); dk.immutable(k); dv.immutable(v);
    check(hipMemset(bad.as<unsigned>(), 0, 4u));
    hipLaunchKernelGGL(verify_full_query, dim3(4096u), dim3(256u), 0u, nullptr,
        full.as<uint16_t>(), compact.as<uint16_t>(), full.bytes / 2u, first, bad.as<unsigned>());
    check(hipGetLastError()); finish();
    if (bad.get<unsigned>()[0]) throw std::runtime_error("full Q or unused history changed");
    full.guards();
    std::printf("{\"kind\":\"compact_decode_query\",\"query_start\":%u,\"query_count\":1,\"key_tokens\":%u,\"family\":%u,\"attempts\":4,\"variants\":2,\"score_slots\":%zu,\"cpu_dots\":%u,\"gb10_context_cells\":%zu,\"full_query_bytes\":%zu,\"compact_query_bytes\":8192,\"complete_scores\":true,\"raw_pv_accumulator_denominator\":true,\"original_single_query_arithmetic\":true,\"guards_inputs_tails\":true,\"nonzero_output_offset\":true,\"refreshed_same_addresses\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false,\"complete_host_ms\":[%.6f,%.6f,%.6f,%.6f]}\n",
        first, tokens, family, reference.scores.size(), cpu_dots, golden.size(), full.bytes,
        complete_ms[0], complete_ms[1], complete_ms[2], complete_ms[3]);
}
void safety(const unsigned char* exp, const unsigned char* rcp) {
    for (unsigned first: {1u, 17u, 31u, 8191u, 8192u, 32768u, 131072u, 262144u, attention::kSplitMaxTokens - 1u}) {
        for (unsigned family = 0; family < 3u; ++family) {
            std::vector<uint16_t> q(4096u), k(size_t(first + 1u) * 512u), v(k.size());
            for (size_t i = 0; i < q.size(); ++i)
                q[i] = uint16_t(((i * 13u) & 0x807fu) | ((119u + i % 12u) << 7u));
            for (size_t i = 0; i < k.size(); ++i) {
                k[i] = uint16_t(((i * 31u) & 0x807fu) | ((117u + i % 14u) << 7u));
                v[i] = uint16_t(((i * 71u) & 0x807fu) | ((119u + i % 12u) << 7u));
                if (family == 1u && i % 511u == 0u) k[i] = uint16_t((i & 1u) ? 0x8001u : 1u);
                if (family == 2u) {
                    k[i] = uint16_t((i & 1u) ? 0x3f81u : 0xbf81u);
                    v[i] = uint16_t(((i / 512u) & 1u) ? 0x3f80u : 0xbf80u);
                }
            }
            if (family == 1u) { q[4095u] = 1u; q[0] = 0x8000u; }
            if (family == 2u) for (size_t i = 0; i < q.size(); ++i) q[i] = uint16_t((i & 1u) ? 0x3f80u : 0xbf80u);
            run_case(first, family, q, k, v, {}, exp, rcp);
        }
    }
}
void capture(char** argv, const unsigned char* exp, const unsigned char* rcp) {
    constexpr unsigned prefix = 16384u, suffix = 1024u, total = prefix + suffix;
    const auto oq = read_file<uint16_t>(argv[2], size_t(suffix) * 4096u);
    const auto ok = read_file<uint16_t>(argv[3], size_t(total) * 512u);
    const auto ov = read_file<uint16_t>(argv[4], size_t(total) * 512u);
    const auto expected = read_file<uint16_t>(argv[5], size_t(suffix) * 4096u);
    for (unsigned row: {0u, 31u, 511u, 1023u}) {
        const unsigned first = prefix + row;
        run_case(first, 3u,
            std::vector<uint16_t>(oq.begin() + size_t(row) * 4096u, oq.begin() + size_t(row + 1u) * 4096u),
            std::vector<uint16_t>(ok.begin(), ok.begin() + size_t(first + 1u) * 512u),
            std::vector<uint16_t>(ov.begin(), ov.begin() + size_t(first + 1u) * 512u),
            std::vector<uint16_t>(expected.begin() + size_t(row) * 4096u, expected.begin() + size_t(row + 1u) * 4096u), exp, rcp);
    }
}
} // namespace
int main(int argc, char** argv) try {
    const bool generated = argc == 4 && !std::strcmp(argv[1], "safety");
    const bool captured = argc == 8 && !std::strcmp(argv[1], "capture");
    if (!generated && !captured) throw std::runtime_error("usage: safety EXP RCP | capture Q K V GB10_CONTEXT EXP RCP");
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    const auto exp = read_file<unsigned char>(argv[argc - 2], attention::exp2_backend::table_bytes);
    const auto rcp = read_file<unsigned char>(argv[argc - 1], qrt_sm121_attention_rcp::table_bytes);
    if (!attention::exp2_backend::valid_layout(exp.data(), exp.size()) ||
        !qrt_sm121_attention_rcp::valid_layout(rcp.data(), rcp.size())) throw std::runtime_error("table layout");
    Guarded de(exp.size()), dr(rcp.size()); de.put(exp); dr.put(rcp);
    if (generated) safety(de.as<unsigned char>(), dr.as<unsigned char>());
    else capture(argv, de.as<unsigned char>(), dr.as<unsigned char>());
    de.immutable(exp); dr.immutable(rcp); return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "compact_decode_query_error=%s\n", error.what()); return 2;
}
