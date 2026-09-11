#include "../native/providers/gdn/sm121_exp2_interpolated.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

std::vector<unsigned char> read(const char* path, uint64_t size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(size)) throw std::runtime_error("table input size");
    std::vector<unsigned char> result(size); file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), size)) throw std::runtime_error("table input read");
    return result;
}
uint32_t original(const unsigned char* table, uint32_t page, unsigned cell) {
    if (page == qrt_sm121_exp2::pages) return 0u;
    const auto* entry = reinterpret_cast<const uint32_t*>(table + 48u) + page * 2u;
    const auto* payload = table + qrt_sm121_exp2::payload_start + (entry[1] & 0x3fffffffu);
    const unsigned kind = entry[1] >> 30u;
    return entry[0] + (kind == 0u ? 0u : kind == 1u ? payload[cell] : kind == 2u ?
        reinterpret_cast<const uint16_t*>(payload)[cell] : reinterpret_cast<const uint32_t*>(payload)[cell]);
}
int64_t floor_div256_reference(int64_t number) {
    return number < 0 ? -int64_t((uint64_t(-number) + 255u) / 256u) : number / 256u;
}
int main(int argc, char** argv) {
    try {
        const bool verify = argc == 4 && std::strcmp(argv[1], "--verify") == 0;
        if (!verify && argc != 3) throw std::runtime_error("usage: repack [--verify] ORIGINAL OUTPUT");
        const char* input_path = argv[verify ? 2 : 1];
        const char* output_path = argv[verify ? 3 : 2];
        const auto input = read(input_path, qrt_sm121_exp2::table_bytes);
        if (!qrt_sm121_exp2::valid_layout(input.data(), input.size())) throw std::runtime_error("original table layout");
        namespace packed = qrt_sm121_exp2_interpolated;
        std::vector<unsigned char> output;
        uint64_t kinds[8]{};
        if (verify) output = read(output_path, packed::table_bytes);
        else {
            if (std::filesystem::exists(output_path)) throw std::runtime_error("output already exists");
            output.resize(packed::payload_start, 0u);
            output.reserve(packed::table_bytes);
            std::memcpy(output.data(), "QEX2IPL1", 8u);
            const uint32_t header[] = {1u, 8u, packed::begin, packed::end,
                qrt_sm121_exp2::positive_one_end, packed::pages, 8u, 3u};
            std::memcpy(output.data() + 8u, header, sizeof(header));
            std::memcpy(output.data() + 48u, qrt_sm121_exp2::sha256, 32u);
            for (uint32_t page = 0u; page < packed::pages; ++page) {
                const uint32_t first = original(input.data(), page, 0u), next = original(input.data(), page + 1u, 0u);
                const int64_t slope = int64_t(next) - first;
                int32_t residuals[256]; int32_t minimum = 0, maximum = 0;
                for (unsigned cell = 0u; cell < 256u; ++cell) {
                    const int64_t predicted = first + floor_div256_reference(slope * cell);
                    const int32_t residual = int32_t(int64_t(original(input.data(), page, cell)) - predicted);
                    residuals[cell] = residual; minimum = std::min(minimum, residual); maximum = std::max(maximum, residual);
                }
                const unsigned kind = slope == 0 && minimum == 0 && maximum == 0 ? 0u :
                    minimum >= 0 && maximum <= 1 ? 1u : minimum >= -1 && maximum <= 0 ? 2u :
                    minimum >= -1 && maximum <= 2 ? 3u : minimum >= -7 && maximum <= 8 ? 4u :
                    minimum >= -127 && maximum <= 128 ? 5u : minimum >= -32767 && maximum <= 32768 ? 6u : 7u;
                ++kinds[kind];
                constexpr unsigned widths[] = {0u, 1u, 1u, 2u, 4u, 8u, 16u, 32u};
                constexpr int bias[] = {0, 0, -1, -1, -7, -127, -32767, 0};
                const size_t payload_offset = output.size() - packed::payload_start;
                if (payload_offset >= 0x20000000u) throw std::runtime_error("packed directory overflow");
                const uint32_t entry[] = {first, (kind << 29u) | (kind ? uint32_t(payload_offset) : 0u)};
                std::memcpy(output.data() + packed::header_bytes + uint64_t(page) * 8u, entry, 8u);
                const size_t start = output.size(); output.resize(start + 32u * widths[kind], 0u);
                if (!kind) continue;
                for (unsigned cell = 0u; cell < 256u; ++cell) {
                    const uint32_t code = kind == 7u ? original(input.data(), page, cell) : uint32_t(residuals[cell] - bias[kind]);
                    if (widths[kind] < 8u) {
                        const unsigned bit = cell * widths[kind];
                        output[start + bit / 8u] |= static_cast<unsigned char>(code << (bit % 8u));
                    } else std::memcpy(output.data() + start + cell * (widths[kind] / 8u), &code, widths[kind] / 8u);
                }
            }
            const uint64_t payload_size = output.size() - packed::payload_start;
            std::memcpy(output.data() + 40u, &payload_size, 8u);
        }
        if (!packed::valid_layout(output.data(), output.size())) throw std::runtime_error("interpolated table layout");
        uint64_t checked = 0u;
        for (uint32_t relative = 0u; relative < packed::end - packed::begin; ++relative) {
            const uint32_t expected = original(input.data(), relative >> 8u, relative & 255u);
            if (packed::decode(output.data(), relative) != expected) throw std::runtime_error("interpolated table bit mismatch");
            ++checked;
        }
        uint32_t seed = 0x3958192u;
        for (unsigned i = 0u; i < 100000u; ++i) {
            seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
            const float argument = qrt_sm121_exp2::value(seed);
            const float expected = qrt_sm121_exp2::evaluate(input.data(), argument);
            const float actual = packed::evaluate(output.data(), argument);
            if (qrt_sm121_exp2::bits(expected) != qrt_sm121_exp2::bits(actual)) throw std::runtime_error("exp2 domain boundary mismatch");
        }
        if (!verify) {
            std::ofstream file(output_path, std::ios::binary);
            if (!file.write(reinterpret_cast<const char*>(output.data()), output.size())) throw std::runtime_error("packed table output write");
        }
        std::printf("{\"kind\":\"lossless_exp2_repack\",\"verify_only\":%s,\"input_bytes\":%llu,\"output_bytes\":%llu,\"exact_cells\":%llu,\"random_domain_inputs\":100000,\"bit_mismatches\":0,\"page_kinds\":[",
            verify ? "true" : "false", (unsigned long long)input.size(), (unsigned long long)output.size(), (unsigned long long)checked);
        for (unsigned i = 0u; i < 8u; ++i) std::printf("%s%llu", i ? "," : "", (unsigned long long)kinds[i]);
        std::printf("],\"inference_acceptance\":false,\"performance_acceptance\":false}\n");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "exp2_repack_error=%s\n", error.what()); return 1; }
}
