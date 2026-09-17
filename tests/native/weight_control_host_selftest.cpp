#include "../../native/providers/moe_accumulator/sm121_weight_control.h"
#include "../../native/providers/moe_accumulator/sm121_scaled_half_products.h"
#include <cstdio>
#include <stdexcept>
namespace compact = qrt_sm121_weight_control_projection;
namespace original = qrt_sm121_scaled_half_products;
int main() try {
    unsigned long long groups = 0u, pairs = 0u;
    for (unsigned x = 0u; x < 65536u; ++x) for (unsigned span = 0u; span <= 31u; ++span) {
        const unsigned e = (x >> 7u) & 255u;
        uint16_t values[16];
        for (unsigned i = 0u; i < 16u; ++i) values[i] = i & 1u ? uint16_t(x ^ 0x8000u) : uint16_t(x);
        values[2] = 0u; values[3] = 0x8000u;
        values[14] = uint16_t(((e + span < 255u ? e + span : 255u) << 7u) | (x & 127u));
        const auto expected = original::prepare(values);
        const unsigned control = compact::control(values);
        if (control != expected.control) throw std::runtime_error("control differs from original encoding");
        for (unsigned pair = 0u; pair < 8u; ++pair) {
            const unsigned words = unsigned(values[pair * 2u]) | (unsigned(values[pair * 2u + 1u]) << 16u);
            if (compact::encode_pair(words, control, pair) != expected.pairs[pair])
                throw std::runtime_error("pair differs from original encoding");
            ++pairs;
        }
        ++groups;
    }
    std::printf("{\"kind\":\"weight_control_exhaustive_encoding\",\"bf16_values\":65536,\"exponent_spans\":32,\"groups\":%llu,\"pairs\":%llu,\"mismatches\":0}\n",groups,pairs);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
