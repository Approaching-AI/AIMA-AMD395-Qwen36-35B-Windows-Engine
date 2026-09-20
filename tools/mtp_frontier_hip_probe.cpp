#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "native/providers/gdn/sm121_mtp_prompt_cache.h"
#include "native/providers/gdn/sm121_mtp_residual.h"

void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(T))
        throw std::runtime_error("invalid input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (!bytes || bytes > 128u * 1024u * 1024u)
        throw std::runtime_error("empty file or size exceeds bound");
    std::vector<T> output(bytes / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char*>(output.data()), bytes);
    if (!file) throw std::runtime_error("short input file");
    return output;
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() {
        if (hipDeviceSynchronize() != hipSuccess) return;
        for (auto pointer : pointers) (void)hipFree(pointer);
    }
    template<class T> T* allocate(size_t count) {
        T* pointer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
        pointers.push_back(pointer);
        return pointer;
    }
    template<class T> T* upload(const std::vector<T>& values) {
        T* pointer = allocate<T>(values.size());
        check(hipMemcpy(pointer, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
        return pointer;
    }
};

int main(int argc, char** argv) try {
    if (argc != 5 && argc != 7 && argc != 8)
        throw std::runtime_error("input weights expected rsqrt_table [residual expected_residual | hidden hidden_weights actual_token_ids]");
    const bool fusion = argc == 8;
    const bool residual_norm = argc == 7;
    constexpr size_t guard = 64u;
    constexpr uint16_t sentinel = 0xa5a5u;
    const auto input = read_file<uint16_t>(argv[1]);
    const auto weights = read_file<uint16_t>(argv[2]);
    const auto expected = read_file<uint16_t>(argv[3]);
    const auto table = read_file<unsigned char>(argv[4]);
    const auto hidden = fusion ? read_file<uint16_t>(argv[5]) : std::vector<uint16_t>{};
    const auto hidden_weights = fusion ? read_file<uint16_t>(argv[6]) : std::vector<uint16_t>{};
    const auto residual = residual_norm ? read_file<uint16_t>(argv[5]) : std::vector<uint16_t>{};
    const auto expected_residual = residual_norm ? read_file<uint16_t>(argv[6]) : std::vector<uint16_t>{};
    auto ids = fusion ? read_file<uint32_t>(argv[7]) : std::vector<uint32_t>{};
    const size_t rows = input.size() / 2048u;
    const size_t output_width = fusion ? 4096u : 2048u;
    if (!rows || rows > 8192u || input.size() != rows * 2048u || weights.size() != 2048u ||
        expected.size() != rows * output_width ||
        (fusion && (hidden.size() != input.size() || hidden_weights.size() != 2048u || ids.size() != rows)) ||
        (residual_norm && (residual.size() != input.size() || expected_residual.size() != input.size())))
        throw std::runtime_error("inconsistent normalization shapes");
    if (!qrt_sm121_rsqrt::valid_layout(table.data(), table.size()))
        throw std::runtime_error("invalid reciprocal-root table layout");
    for (uint32_t id : ids) if (id >= 248320u)
        throw std::runtime_error("original token ID outside vocabulary");
    Scratch scratch;
    const auto device_weights = scratch.upload(weights);
    const auto device_table = scratch.upload(table);
    uint16_t* device_input = nullptr;
    uint16_t* device_hidden = nullptr;
    uint16_t* device_hidden_weights = nullptr;
    uint32_t* device_ids = nullptr;
    uint32_t* device_invalid = nullptr;
    uint16_t* device_residual = nullptr;
    if (fusion) {
        // Preserve actual vocabulary indices while uploading only the captured
        // rows. Duplicate IDs must carry the same original embedding bytes.
        device_input = scratch.allocate<uint16_t>(size_t(248320u) * 2048u);
        std::unordered_map<uint32_t, size_t> uploaded;
        for (size_t row = 0; row < rows; ++row) {
            const auto inserted = uploaded.emplace(ids[row], row);
            const auto begin = input.begin() + row * 2048u;
            if (!inserted.second) {
                if (!std::equal(begin, begin + 2048u, input.begin() + inserted.first->second * 2048u))
                    throw std::runtime_error("duplicate ID has different original embeddings");
            } else {
                check(hipMemcpy(device_input + size_t(ids[row]) * 2048u, input.data() + row * 2048u,
                                2048u * sizeof(uint16_t), hipMemcpyHostToDevice));
            }
        }
        device_hidden = scratch.upload(hidden);
        device_hidden_weights = scratch.upload(hidden_weights);
        device_ids = scratch.upload(ids);
        device_invalid = scratch.allocate<uint32_t>(1u);
    } else if (residual_norm) {
        std::vector<uint16_t> guarded(input.size() + 2u * guard, sentinel);
        std::copy(input.begin(), input.end(), guarded.begin() + guard);
        device_input = scratch.upload(guarded) + guard;
        std::copy(residual.begin(), residual.end(), guarded.begin() + guard);
        device_residual = scratch.upload(guarded) + guard;
    } else {
        device_input = scratch.upload(input);
    }
    std::vector<uint16_t> output(expected.size() + 2u * guard, sentinel);
    std::vector<uint16_t> residual_output = residual_norm ? output : std::vector<uint16_t>{};
    uint16_t* device_output = scratch.upload(output);
    uint16_t* device_residual_output = residual_norm ? scratch.upload(residual_output) : nullptr;
    const auto output_bytes = output.size() * sizeof(uint16_t);
    if (residual_norm) {
        for (unsigned int invalid_rows : {0u,8193u})
            if (qrt_sm121_mtp::launch_residual_normalize(device_input,device_residual,
                    device_weights,device_table,invalid_rows,device_output+guard,
                    device_residual_output+guard) != hipErrorInvalidValue)
                throw std::runtime_error("residual norm accepted invalid rows");
        if (qrt_sm121_mtp::launch_residual_normalize(device_input,device_residual,
                device_weights,device_table,static_cast<unsigned int>(rows),device_output+guard,
                device_output+guard) != hipErrorInvalidValue)
            throw std::runtime_error("residual norm accepted aliased outputs");
    }
    bool passed = true;
    for (unsigned int attempt = 0; attempt < ((fusion || residual_norm) ? 2u : 1u); ++attempt) {
        const bool invalid_case = fusion && attempt;
        const bool inplace = residual_norm && attempt;
        uint16_t* normalized_target = inplace ? device_input : device_output + guard;
        uint16_t* residual_target = residual_norm
            ? (inplace ? device_residual : device_residual_output + guard) : nullptr;
        check(hipMemset(device_output, 0xa5, output_bytes));
        if (fusion) {
            check(hipMemset(device_invalid, 0, sizeof(uint32_t)));
            if (invalid_case) {
                const uint32_t invalid_id = 248320u;
                check(hipMemcpy(device_ids, &invalid_id, sizeof(invalid_id), hipMemcpyHostToDevice));
            }
            check(qrt_sm121_mtp::launch_fusion_inputs(device_input, device_hidden, device_ids,
                device_weights, device_hidden_weights, device_table, static_cast<unsigned int>(rows),
                device_output + guard, device_invalid));
        } else if (residual_norm) {
            check(hipMemset(device_residual_output, 0xa5, output_bytes));
            check(qrt_sm121_mtp::launch_residual_normalize(device_input, device_residual,
                device_weights, device_table, static_cast<unsigned int>(rows), normalized_target, residual_target));
        } else {
            check(qrt_sm121_mtp::launch_normalize(device_input, device_weights, device_table,
                static_cast<unsigned int>(rows), device_output + guard));
        }
        check(hipDeviceSynchronize());
        check(hipMemcpy(output.data(), normalized_target - guard, output_bytes, hipMemcpyDeviceToHost));
        if (residual_norm)
            check(hipMemcpy(residual_output.data(), residual_target - guard, output_bytes, hipMemcpyDeviceToHost));
        uint32_t invalid_flag = 0;
        if (fusion) check(hipMemcpy(&invalid_flag, device_invalid, sizeof(invalid_flag), hipMemcpyDeviceToHost));
        size_t mismatches = 0, guard_mismatches = 0, skipped_row_mismatches = 0;
        size_t residual_mismatches = 0, input_mismatches = 0;
        size_t first = expected.size();
        double maximum_error = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (invalid_case && i < output_width) {
                skipped_row_mismatches += output[guard + i] != sentinel;
                continue;
            }
            const uint16_t actual = output[guard + i];
            const float actual_value = qrt_sm121_q1::widen(actual);
            const float wanted = qrt_sm121_q1::widen(expected[i]);
            if (!std::isfinite(actual_value) || !std::isfinite(wanted))
                throw std::runtime_error("nonfinite comparison value");
            maximum_error = std::max(maximum_error, std::abs(double(actual_value) - wanted));
            if (actual != expected[i]) {
                ++mismatches;
                if (first == expected.size()) first = i;
            }
            if (residual_norm) residual_mismatches += residual_output[guard + i] != expected_residual[i];
        }
        for (size_t i = 0; i < guard; ++i) {
            guard_mismatches += output[i] != sentinel;
            guard_mismatches += output[output.size() - 1u - i] != sentinel;
            if (residual_norm) {
                guard_mismatches += residual_output[i] != sentinel;
                guard_mismatches += residual_output[residual_output.size() - 1u - i] != sentinel;
            }
        }
        if (residual_norm && !inplace) {
            std::vector<uint16_t> readback(output.size());
            check(hipMemcpy(readback.data(), device_input - guard, output_bytes, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < input.size(); ++i) input_mismatches += readback[guard + i] != input[i];
            for (size_t i = 0; i < guard; ++i)
                guard_mismatches += (readback[i] != sentinel) + (readback[readback.size()-1u-i] != sentinel);
            check(hipMemcpy(readback.data(), device_residual - guard, output_bytes, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < residual.size(); ++i) input_mismatches += readback[guard + i] != residual[i];
            for (size_t i = 0; i < guard; ++i)
                guard_mismatches += (readback[i] != sentinel) + (readback[readback.size()-1u-i] != sentinel);
        }
        const bool current = !mismatches && !guard_mismatches && !skipped_row_mismatches &&
                             !residual_mismatches && !input_mismatches && invalid_flag == invalid_case;
        passed = passed && current;
        std::cout << "{\"kind\":\"original_mtp_frontier_native_replay\",\"fusion_inputs\":"
                  << (fusion ? "true" : "false") << ",\"rows\":" << rows
                  << ",\"residual_normalization\":" << (residual_norm ? "true" : "false")
                  << ",\"inplace\":" << (inplace ? "true" : "false")
                  << ",\"invalid_launch_cases\":" << (residual_norm ? 3u : 0u)
                  << ",\"compared_elements\":" << expected.size() - (invalid_case ? output_width : 0u)
                  << ",\"invalid_id_case\":" << (invalid_case ? "true" : "false")
                  << ",\"invalid_input_flag\":" << invalid_flag << ",\"bf16_mismatches\":" << mismatches
                  << ",\"guard_mismatches\":" << guard_mismatches
                  << ",\"skipped_row_mismatches\":" << skipped_row_mismatches
                  << ",\"residual_mismatches\":" << residual_mismatches
                  << ",\"input_mismatches\":" << input_mismatches
                  << ",\"maximum_absolute_error\":" << maximum_error << ",\"first_difference\":";
        if (mismatches) std::cout << "{\"index\":" << first << ",\"actual_bits\":" << output[guard + first]
                                  << ",\"expected_bits\":" << expected[first] << '}';
        else std::cout << "null";
        std::cout << ",\"passed\":" << (current ? "true" : "false")
                  << ",\"inference_acceptance\":false}\n";
    }
    return passed ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
