// Compare the actual device lookup against the runtime's existing CPU BF16
// conversion and the complete-domain GB10 gate tables. No model is loaded.
#include "../../native/providers/gdn/gb10_gate_lookup.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../../native/src/qrt.c"
#include "../../native/src/qwen36_baseline.c"

namespace {
constexpr size_t guard = 128u;
constexpr uint32_t sentinel = 0x5a39c27du;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void hip_ok(hipError_t status, const char* stage) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(stage) + ": " + hipGetErrorString(status));
}
template<class T> std::vector<T> read(const std::string& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input) && input.tellg() == std::streamoff(count * sizeof(T)), "table size");
    std::vector<T> result(count + 2u * guard, T(sentinel));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data() + guard), count * sizeof(T));
    require(bool(input), "table read");
    return result;
}
template<class T> struct Device {
    T* base = nullptr;
    size_t count;
    explicit Device(const std::vector<T>& host) : count(host.size()) {
        hip_ok(hipMalloc(reinterpret_cast<void**>(&base), count * sizeof(T)), "allocate");
        write(host);
    }
    ~Device() { if (base) (void)hipFree(base); }
    Device(const Device&) = delete;
    T* data() { return base + guard; }
    void write(const std::vector<T>& host) {
        require(host.size() == count, "write size");
        hip_ok(hipMemcpy(base, host.data(), count * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    std::vector<T> read() {
        std::vector<T> result(count);
        hip_ok(hipMemcpy(result.data(), base, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return result;
    }
};

void run(unsigned layer, unsigned tokens, unsigned low,
         const std::vector<uint32_t>& g, const std::vector<uint16_t>& beta,
         Device<uint32_t>& dg, Device<uint16_t>& dbeta) {
    const size_t cells = size_t(tokens) * 32u;
    std::vector<uint32_t> a(cells + 2u * guard, sentinel), b = a;
    std::vector<uint32_t> blank(2u * cells + 2u * guard, sentinel);
    for (size_t i = 0; i < cells; ++i) {
        const unsigned token = unsigned(i / 32u), head = unsigned(i % 32u);
        a[guard + i] = (token << 16u) | low;
        b[guard + i] = (((token * 40503u + head * 17u) & 65535u) << 16u) | low;
    }
    Device<uint32_t> da(a), db(b), output(blank);
    hipStream_t stream = nullptr;
    hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    hipEvent_t done = nullptr;
    hip_ok(hipEventCreateWithFlags(&done, hipEventDisableTiming), "event");
    const auto start = std::chrono::steady_clock::now();
    hip_ok(qrt_gb10_gate_lookup::launch(reinterpret_cast<const float*>(da.data()),
        reinterpret_cast<const float*>(db.data()), dg.data(), dbeta.data(),
        reinterpret_cast<float*>(output.data()), tokens, stream), "lookup");
    hip_ok(hipEventRecord(done, stream), "record");
    for (;;) {
        const hipError_t status = hipEventQuery(done);
        if (status == hipSuccess) break;
        require(status == hipErrorNotReady &&
            std::chrono::steady_clock::now() - start < std::chrono::seconds(5), "lookup deadline");
        std::this_thread::yield();
    }
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    hip_ok(hipEventDestroy(done), "event destroy");
    hip_ok(hipStreamDestroy(stream), "stream destroy");
    const auto actual = output.read();
    size_t mismatches = 0;
    for (size_t i = 0; i < cells; ++i) {
        float av, bv;
        std::memcpy(&av, &a[guard + i], 4u);
        std::memcpy(&bv, &b[guard + i], 4u);
        const unsigned head = unsigned(i % 32u);
        const size_t destination = guard + (i / 32u) * 64u + head;
        mismatches += actual[destination] != g[guard + size_t(head) * 65536u + qrt_float_to_bf16(av)];
        mismatches += actual[destination + 32u] != (uint32_t(beta[guard + qrt_float_to_bf16(bv)]) << 16u);
    }
    for (size_t i = 0; i < guard; ++i)
        require(actual[i] == sentinel && actual[guard + 2u * cells + i] == sentinel, "output guard");
    require(da.read() == a && db.read() == b, "input changed");
    require(mismatches == 0, "CPU gate mismatch");
    std::cout << "{\"kind\":\"gb10_gate_device_lookup\",\"layer\":" << layer
              << ",\"tokens\":" << tokens << ",\"fp32_low_bits\":" << low
              << ",\"compared_raw_output_cells\":" << 2u * cells
              << ",\"raw_bit_mismatches\":0,\"immutable_inputs\":true,\"redzones_pass\":true"
              << ",\"completed_host_ms\":" << wall << "}" << std::endl;
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: gb10-gate-lookup.exe gate-table-directory");
        const std::string directory = argv[1];
        // Invalid arguments must fail before any device submission.
        float f = 0; uint32_t g = 0; uint16_t b = 0;
        for (unsigned choice = 0; choice < 7u; ++choice)
            require(qrt_gb10_gate_lookup::launch(choice == 0 ? nullptr : &f,
                choice == 1 ? nullptr : &f, choice == 2 ? nullptr : &g,
                choice == 3 ? nullptr : &b, choice == 4 ? nullptr : &f,
                choice == 5 ? 0u : choice == 6 ? 65537u : 1u) == hipErrorInvalidValue,
                "invalid contract submitted");
        const auto beta = read<uint16_t>(directory + "/sigmoid-beta-bf16.bin", 65536u);
        Device<uint16_t> dbeta(beta);
        for (unsigned layer = 0; layer < 40u; ++layer) {
            if (layer % 4u == 3u) continue;
            const auto gate = read<uint32_t>(directory + "/layer" + std::to_string(layer) + "-g-f32-head-major.bin", 32u * 65536u);
            Device<uint32_t> dg(gate);
            // Every BF16 code in every head, plus values on both sides of the
            // RNE midpoint, including signs, subnormals, infinities and NaNs.
            for (unsigned low : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu})
                run(layer, 65536u, low, gate, beta, dg, dbeta);
            if (layer == 0u)
                for (unsigned tokens : {1u, 7u, 8u, 9u, 31u, 33u, 8191u, 8192u, 8193u})
                    run(layer, tokens, 0x8000u, gate, beta, dg, dbeta);
            require(dg.read() == gate, "G table changed");
        }
        require(dbeta.read() == beta, "beta table changed");
        std::cout << "{\"kind\":\"gb10_gate_device_lookup_summary\",\"cases\":159,\"layers\":30,"
                     "\"invalid_arguments_rejected\":7,\"immutable_tables_and_guards\":true}" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
