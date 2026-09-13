#include "../../native/providers/gdn/blackwell_kkt.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr unsigned guard = 64u;
void check(hipError_t value) {
    if (value != hipSuccess) throw std::runtime_error(hipGetErrorString(value));
}
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* at(size_t offset = 0u) { return static_cast<T*>(pointer) + offset; }
};
template<class T> void upload(Device& d, const std::vector<T>& data) {
    check(hipMemcpy(d.pointer, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> data(count);
    check(hipMemcpy(data.data(), d.pointer, count * sizeof(T), hipMemcpyDeviceToHost));
    return data;
}
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("KKT completion timeout");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
float from(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16u;
    float out; std::memcpy(&out, &bits, 4u); return out;
}
uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, 4u); return out; }
void dispatch(unsigned mode, Device& k, Device& beta, Device& out,
              unsigned tokens, unsigned first_chunk) {
    const unsigned total_chunks = tokens / 64u;
    for (unsigned base = 0u; base < total_chunks; base += 16u) {
        const unsigned first = base ? 0u : first_chunk;
        const unsigned chunks = std::min(16u, total_chunks - base) - first;
        check(qrt_fla_blackwell::launch_dot_chunks(k.at<uint16_t>(guard + size_t(base) * 64u * 2048u),
            beta.at<uint16_t>(guard + size_t(base) * 64u * 32u),
            out.at<float>(guard + size_t(base) * 64u * 2048u), first, chunks, mode, nullptr));
    }
}
void run(unsigned tokens, unsigned first_chunk, unsigned pattern) {
    const size_t cells = size_t(tokens) * 2048u;
    std::vector<uint16_t> k(cells + 2u * guard, 0x5a5au);
    std::vector<uint16_t> beta(size_t(tokens) * 32u + 2u * guard, 0x5a5au);
    for (size_t i = guard; i + guard < k.size(); ++i) {
        k[i] = uint16_t(((i * 37u + i / 19u) & 0x807fu) | ((123u + i % 7u) << 7u));
        if (pattern == 1u && i % 7u == 0u) k[i] = i % 3u ? 0u : 0x8000u;
        if (pattern == 2u && i % 127u == 0u) k[i] = uint16_t((63u << 7u) | (i & 0x807fu));
        if (pattern == 3u && i % 61u == 0u) k[i] = uint16_t((190u << 7u) | (i & 0x807fu));
        if (pattern == 4u) k[i] = uint16_t(0x3fffu | ((i / 16u) & 1u ? 0x8000u : 0u));
    }
    for (size_t i = guard; i + guard < beta.size(); ++i) {
        beta[i] = uint16_t((125u + i % 3u) << 7u | (i * 19u & 0x7fu));
        if (pattern == 1u && i % 5u == 0u) beta[i] = i % 3u ? 0u : 0x8000u;
        if (pattern >= 2u) beta[i] = uint16_t(0x3f80u | (i & 1u ? 0x8000u : 0u));
    }
    Device dk(k.size() * 2u), db(beta.size() * 2u);
    Device original((cells + 2u * guard) * 4u), candidate((cells + 2u * guard) * 4u);
    upload(dk, k); upload(db, beta);
    check(hipMemset(original.pointer, 0xa5, (cells + 2u * guard) * 4u));
    dispatch(0u, dk, db, original, tokens, first_chunk); finish();
    auto begin = std::chrono::steady_clock::now();
    dispatch(0u, dk, db, original, tokens, first_chunk); finish();
    const double original_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    const auto expected = download<uint32_t>(original, cells + 2u * guard);
    const size_t first_cell = size_t(first_chunk) * 64u * 2048u;
    for (unsigned mode : {1u, 2u}) {
        check(hipMemset(candidate.pointer, 0xa5, (cells + 2u * guard) * 4u));
        dispatch(mode, dk, db, candidate, tokens, first_chunk); finish();
        begin = std::chrono::steady_clock::now();
        dispatch(mode, dk, db, candidate, tokens, first_chunk); finish();
        const double query_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        const auto actual = download<uint32_t>(candidate, cells + 2u * guard);
        for (size_t i = 0u; i < actual.size(); ++i) {
            const bool untouched = i < guard + first_cell || i >= guard + cells;
            if (untouched && (expected[i] != 0xa5a5a5a5u || actual[i] != 0xa5a5a5a5u))
                throw std::runtime_error("KKT redzone or preceding chunk changed");
            if (!untouched && expected[i] != actual[i]) {
                std::fprintf(stderr, "mode=%u tokens=%u pattern=%u cell=%zu expected=%08x actual=%08x\n",
                    mode, tokens, pattern, i - guard, expected[i], actual[i]);
                throw std::runtime_error("KKT raw output differs");
            }
        }
        for (unsigned sample = 0u; sample < 256u; ++sample) {
            const unsigned chunk = first_chunk + sample % (tokens / 64u - first_chunk);
            const unsigned row = 1u + (sample * 17u) % 63u, column = (sample * 13u) % row;
            const unsigned head = (sample / 3u) % 32u;
            const size_t token = size_t(chunk) * 64u + row, other = size_t(chunk) * 64u + column;
            uint16_t left[128];
            const float scale = from(beta[guard + token * 32u + head]);
            for (unsigned feature = 0u; feature < 128u; ++feature)
                left[feature] = bf16(from(k[guard + (token * 16u + head / 2u) * 128u + feature]) * scale);
            const float reference = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                left, k.data() + guard + (other * 16u + head / 2u) * 128u, 128u);
            if (bits(reference) != actual[guard + (token * 32u + head) * 64u + column])
                throw std::runtime_error("KKT independent CPU dot differs");
        }
        if (download<uint16_t>(dk, k.size()) != k || download<uint16_t>(db, beta.size()) != beta)
            throw std::runtime_error("KKT inputs changed");
        std::printf("{\"kind\":\"tiled_kkt_control\",\"tokens\":%u,\"first_chunk\":%u,\"pattern\":%u,\"mode\":%u,\"cells\":%zu,\"cpu_dots\":256,\"raw_bit_mismatches\":0,\"original_ms\":%.6f,\"candidate_ms\":%.6f,\"warmups\":1,\"timed_iterations\":1,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens, first_chunk, pattern, mode, cells - first_cell, original_ms, query_ms);
        std::fflush(stdout);
    }
}
}
int main() {
    try {
        hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
        if (std::strncmp(device.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
        for (unsigned pattern = 0u; pattern < 5u; ++pattern) run(64u, 0u, pattern);
        run(192u, 1u, 2u); run(1024u, 0u, 1u); run(1024u, 15u, 3u); run(8192u, 0u, 0u);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
