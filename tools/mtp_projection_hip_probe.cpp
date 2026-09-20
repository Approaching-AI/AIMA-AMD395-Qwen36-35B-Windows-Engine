#include <hip/hip_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_mtp_projection.h"

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
std::vector<uint16_t> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(uint16_t))
        throw std::runtime_error("input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (bytes > 128u * 1024u * 1024u) throw std::runtime_error("file bound");
    std::vector<uint16_t> values(bytes / sizeof(uint16_t));
    file.seekg(0); file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file) throw std::runtime_error("short input");
    return values;
}
unsigned int width(const char* argument) {
    const std::string text(argument); size_t used = 0;
    const unsigned long value = std::stoul(text, &used);
    if (used != text.size() || value > 8192u) throw std::runtime_error("width");
    return static_cast<unsigned int>(value);
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() {
        if (hipDeviceSynchronize() != hipSuccess) return;
        for (void* pointer : pointers) (void)hipFree(pointer);
    }
    uint16_t* upload(const std::vector<uint16_t>& values) {
        uint16_t* pointer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer), values.size() * sizeof(uint16_t)));
        pointers.push_back(pointer);
        check(hipMemcpy(pointer, values.data(), values.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        return pointer;
    }
};
int main(int argc, char** argv) try {
    if (argc != 6) throw std::runtime_error("input weights expected input_width output_width");
    const auto input = read_file(argv[1]), weights = read_file(argv[2]), expected = read_file(argv[3]);
    const unsigned int inputs = width(argv[4]), outputs = width(argv[5]);
    const bool shape = (inputs == 4096u && outputs == 2048u) ||
                       (inputs == 2048u && (outputs == 8192u || outputs == 1024u));
    if (!shape || input.size() % inputs) throw std::runtime_error("shape");
    const size_t rows = input.size() / inputs;
    if (!rows || rows > 8192u || weights.size() != size_t(inputs) * outputs ||
        expected.size() != rows * outputs) throw std::runtime_error("tensor dimensions");
    constexpr size_t guard = 64u; constexpr uint16_t sentinel = 0xa5a5u;
    const auto guarded = [&](const std::vector<uint16_t>& values) {
        std::vector<uint16_t> result(values.size() + guard * 2u, sentinel);
        std::copy(values.begin(), values.end(), result.begin() + guard); return result;
    };
    const auto guarded_input = guarded(input), guarded_weights = guarded(weights);
    std::vector<uint16_t> output(expected.size() + guard * 2u, sentinel);
    Scratch scratch;
    auto* device_input = scratch.upload(guarded_input);
    auto* device_weights = scratch.upload(guarded_weights);
    auto* device_output = scratch.upload(output);
    using qrt_sm121_mtp::launch_projection;
    unsigned int invalid_cases = 0;
    const auto rejected = [&](hipError_t status) {
        if (status != hipErrorInvalidValue) throw std::runtime_error("invalid projection accepted");
        ++invalid_cases;
    };
    auto* w = device_weights + guard; auto* x = device_input + guard; auto* y = device_output + guard;
    rejected(launch_projection(nullptr, x, y, outputs, inputs, 1u));
    rejected(launch_projection(w, nullptr, y, outputs, inputs, 1u));
    rejected(launch_projection(w, x, nullptr, outputs, inputs, 1u));
    rejected(launch_projection(w, x, x, outputs, inputs, 1u));
    rejected(launch_projection(w, x, w, outputs, inputs, 1u));
    rejected(launch_projection(w, x, y, 1u, inputs, 1u));
    rejected(launch_projection(w, x, y, outputs, 1u, 1u));
    rejected(launch_projection(w, x, y, outputs, inputs, 0u));
    rejected(launch_projection(w, x, y, outputs, inputs, 8193u));
    rejected(launch_projection(w, x, y, outputs, inputs, 1u, 0u));
    rejected(launch_projection(w, x, y, outputs, inputs, 1u, 4097u));
    size_t mismatches = 0, guard_bad = 0, input_bad = 0, weights_bad = 0;
    for (unsigned int blocks : {7u, 1024u}) {
        std::fill(output.begin(), output.end(), sentinel);
        check(hipMemcpy(device_output, output.data(), output.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        check(launch_projection(w, x, y, outputs, inputs, static_cast<unsigned int>(rows), blocks));
        check(hipDeviceSynchronize());
        check(hipMemcpy(output.data(), device_output, output.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < expected.size(); ++i) mismatches += output[guard + i] != expected[i];
        for (size_t i = 0; i < guard; ++i) {
            guard_bad += output[i] != sentinel; guard_bad += output[output.size() - 1u - i] != sentinel;
        }
    }
    const auto changed = [&](const uint16_t* device, const std::vector<uint16_t>& before) {
        std::vector<uint16_t> after(before.size()); size_t bad = 0;
        check(hipMemcpy(after.data(), device, after.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < after.size(); ++i) bad += after[i] != before[i];
        return bad;
    };
    input_bad = changed(device_input, guarded_input); weights_bad = changed(device_weights, guarded_weights);
    const bool passed = !mismatches && !guard_bad && !input_bad && !weights_bad;
    std::cout << "{\"kind\":\"original_mtp_projection_native_replay\",\"rows\":" << rows
        << ",\"input_width\":" << inputs << ",\"output_width\":" << outputs
        << ",\"elements_per_launch\":" << expected.size() << ",\"launch_configurations\":2,\"maximum_blocks\":[7,1024]"
        << ",\"compared_elements\":" << expected.size() * 2u << ",\"bf16_mismatches\":" << mismatches
        << ",\"guard_mismatches\":" << guard_bad << ",\"input_mismatches\":" << input_bad
        << ",\"weights_mismatches\":" << weights_bad << ",\"invalid_launch_cases\":" << invalid_cases
        << ",\"projection_workspace_bytes\":0,\"passed\":" << (passed ? "true" : "false")
        << ",\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return passed ? 0 : 1;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
