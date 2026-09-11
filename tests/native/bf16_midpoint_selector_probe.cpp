#include "../../native/providers/moe_accumulator/bf16_midpoint_selector.h"
#include <cstdio>
#include <initializer_list>
#include <limits>

int main() {
    unsigned checked = 0;
    // Independent double-precision boundaries for every finite positive BF16
    // cell, their negative mirrors, and both sides of exponent transitions.
    for (uint32_t b = 1; b < 0x7f7fu; ++b) {
        const uint32_t c_bits = b << 16u;
        const uint32_t p_bits = (b - 1u) << 16u;
        const uint32_t n_bits = (b + 1u) << 16u;
        float c, p, n;
        memcpy(&c, &c_bits, 4); memcpy(&p, &p_bits, 4); memcpy(&n, &n_bits, 4);
        for (int side : {-1, 1}) {
            const double boundary = (double(c) + (side < 0 ? double(p) : double(n))) / 2.0;
            const float x = float(double(c) + (boundary - c) * 0.125);
            const float near = float(fmin(fabs(double(x) - (double(c) + p) / 2.0),
                                          fabs(double(x) - (double(c) + n) / 2.0)));
            for (float sign : {-1.0f, 1.0f}) {
                if (qrt_bf16_midpoint::nearest_distance(sign * x) != near ||
                    qrt_bf16_midpoint::within_error(sign * x, near * 0.5f) ||
                    !qrt_bf16_midpoint::within_error(sign * x, near * 1.01f)) return 1;
                ++checked;
            }
        }
    }
    if (!qrt_bf16_midpoint::within_error(1.0f, 0.002f) ||
        qrt_bf16_midpoint::within_error(1.0f, 0.001f) ||
        !qrt_bf16_midpoint::within_error(0.0f, 1.0e-30f) ||
        !qrt_bf16_midpoint::within_error(std::numeric_limits<float>::infinity(), 0.0f)) return 2;
    std::printf("{\"checked\":%u,\"mismatches\":0}\n", checked);
    return 0;
}
