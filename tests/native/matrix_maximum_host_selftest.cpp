#include "../../native/providers/moe_accumulator/sm121_matrix_maximum.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace matrix = qrt_sm121_matrix_maximum;
void require(bool ok) { if (!ok) throw std::runtime_error("matrix maximum host comparison"); }
int main() try {
    unsigned valid_encodings = 0u, recovery_checks = 0u, group_checks = 0u;
    for (unsigned raw = 0u; raw < 65536u; ++raw) {
        const unsigned exponent = (raw >> 7u) & 255u;
        const uint16_t expected = !(raw & 0x7fffu) ? 0u : exponent >= 115u && exponent <= 139u
            ? uint16_t((127 + 5 * (int(exponent) - 127)) << 7u) : matrix::invalid_operand;
        require(matrix::encode(uint16_t(raw)) == expected);
        require((matrix::pack(uint16_t(raw)) >> 16u) == raw);
        valid_encodings += expected != matrix::invalid_operand;
    }
    // The extrema of the full allowed sum band enclose every distribution of
    // sixteen encoded products. Stress both ends and FP32 transitions inside.
    for (int maximum = -24; maximum <= 24; ++maximum) {
        const double leading = std::ldexp(1.0, 5 * maximum);
        for (unsigned sample = 0u; sample <= 4096u; ++sample) {
            const double exact = leading * (1.0 + 15.0 * double(sample) / 4096.0);
            for (double ratio : {0.75, 0.999998, 1.0, 1.000002, 1.25}) {
                require(matrix::recover(float(exact * ratio)) == maximum); ++recovery_checks;
            }
        }
    }
    require(matrix::recover(0.0f) == -133 && matrix::recover(-0.0f) == -133);
    for (float value : {-1.0f, INFINITY, -INFINITY, NAN}) require(matrix::recover(value) == matrix::invalid_maximum);
    uint32_t random = 0x510016u;
    auto next = [&] { random ^= random << 13u; random ^= random >> 17u; random ^= random << 5u; return random; };
    for (unsigned dot = 0u; dot < 16384u; ++dot) {
        float carry = 0.0f, reference = 0.0f;
        for (unsigned g = 0u; g < 16u; ++g) {
            uint16_t left[16], right[16];
            qrt_sm121_float_alignment::Group products;
            int maximum = -133; double positive = 0.0;
            for (unsigned i = 0u; i < 16u; ++i) {
                left[i] = uint16_t((next() & 0x807fu) | ((115u + next() % 25u) << 7u));
                right[i] = uint16_t((next() & 0x807fu) | ((115u + next() % 25u) << 7u));
                if (next() % 11u == 0u) left[i] &= 0x8000u;
                if (next() % 13u == 0u) right[i] &= 0x8000u;
                products.set(i, left[i], right[i]);
                if ((left[i] & 0x7fffu) && (right[i] & 0x7fffu)) {
                    const int exponent = int((left[i] >> 7u) & 255u) + int((right[i] >> 7u) & 255u) - 254;
                    maximum = std::max(maximum, exponent); positive += std::ldexp(1.0, 5 * exponent);
                }
            }
            require(matrix::recover(float(positive)) == maximum && products.maximum == maximum);
            products.maximum = matrix::recover(float(positive));
            float actual;
            require(qrt_sm121_f32_carry::accumulate<0u>(carry, products, &actual));
            reference = qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(reference, left, right, 16u);
            require(qrt_sm121_f32_carry::bits(actual) == qrt_sm121_f32_carry::bits(reference));
            carry = actual; ++group_checks;
        }
    }
    std::printf("{\"kind\":\"matrix_maximum_host\",\"bf16_encodings\":65536,\"valid_encodings\":%u,\"conditional_recovery_checks\":%u,\"canonical_groups\":%u,\"mismatches\":0,\"native_relative_error_condition\":0.25,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false}\n",
        valid_encodings, recovery_checks, group_checks);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
