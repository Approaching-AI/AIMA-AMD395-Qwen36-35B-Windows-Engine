#include "../../native/providers/ck_fmha/register_pv_policy.h"
#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <limits>

int main() {
    using qrt_register_pv_policy::select;
    bool active = true;
    for (const char* value : {static_cast<const char*>(nullptr), "", "0"})
        if (!select(value, 0u, 8192u, false, false, false, true, active) || active) return 1;
    for (const char* value : {"00", "01", "2", "-1", "+1", " 1", "1 ", "1\n", "true"})
        for (unsigned count : {1u, 8192u, 8193u})
            if (select(value, 0u, count, true, true, true, false, active) || active) return 2;
    unsigned cases = 0u;
    for (unsigned flags = 0; flags < 16u; ++flags) {
        const bool final = flags & 1u, direct = flags & 2u;
        const bool transpose = flags & 4u, selective = flags & 8u;
        for (unsigned start : {0u, 1u, 7168u, 8190u, 8191u, 8192u,
                                std::numeric_limits<unsigned>::max()})
            for (unsigned count : {0u, 1u, 2u, 128u, 1024u, 7169u, 8191u, 8192u,
                                    8193u, std::numeric_limits<unsigned>::max()}) {
                const bool eligible = count > 1u && uint64_t(start) + count <= 8192u;
                const bool compatible = flags == 7u;
                const bool ok = select("1", start, count, final, direct, transpose, selective, active);
                if (ok != (!eligible || compatible) || active != (eligible && compatible)) return 3;
                if (!select("0", start, count, final, direct, transpose, selective, active) || active) return 4;
                ++cases;
            }
    }
    std::printf("REGISTER_PV_POLICY cases=%u strict_parse=1 decode_and_long_fallback=1 incompatible_rejected=1\n", cases);
}
