#include "../native/providers/gdn/sm121_exp2_interpolated.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

// Check the actual SHA-bound table before using endpoint evaluations to
// enclose an interval. Native SFU outputs need not be assumed monotone.
// Maximum rise over the preceding running minimum bounds every upward
// excursion, including multi-cell rises, in positive FP32 bit-pattern units.
int main(int argc, char** argv) try {
    namespace table = qrt_sm121_exp2_interpolated;
    if (argc != 2) throw std::runtime_error("supply compact exp2 table");
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(table::table_bytes))
        throw std::runtime_error("table length");
    std::vector<unsigned char> data(table::table_bytes);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size()) ||
        !table::valid_layout(data.data(), data.size())) throw std::runtime_error("table layout/read");
    uint32_t previous = 0x3f800000u, minimum = previous, maximum_rise = 0u;
    uint32_t first_rise_argument = 0u, first_rise_previous = 0u, first_rise_current = 0u;
    uint64_t adjacent_rises = 0u;
    for (uint32_t relative = 0u; relative < table::end - table::begin; ++relative) {
        const uint32_t value = table::decode(data.data(), relative);
        if (value > 0x3f800000u) throw std::runtime_error("outside probability range");
        if (value > previous) {
            if (!adjacent_rises) {
                first_rise_argument = 0x80000000u | (table::begin + relative);
                first_rise_previous = previous; first_rise_current = value;
            }
            ++adjacent_rises;
        }
        minimum = std::min(minimum, value);
        maximum_rise = std::max(maximum_rise, value - minimum);
        previous = value;
    }
    std::printf("{\"kind\":\"sm121_exp2_monotonicity\",\"enumerated_interior_inputs\":%u,"
        "\"adjacent_rises\":%llu,\"maximum_rise_over_prior_minimum_bits\":%u,"
        "\"first_rise_argument_bits\":%u,\"first_rise_previous_output_bits\":%u,"
        "\"first_rise_current_output_bits\":%u,\"all_outputs_in_zero_one\":true,"
        "\"negative_exterior_outputs\":\"one before interior; zero after interior\","
        "\"model_loaded\":false,\"inference_acceptance\":false}\n",
        table::end - table::begin, static_cast<unsigned long long>(adjacent_rises), maximum_rise,
        first_rise_argument, first_rise_previous, first_rise_current);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "exp2_monotonicity_error=%s\n", error.what()); return 2;
}
