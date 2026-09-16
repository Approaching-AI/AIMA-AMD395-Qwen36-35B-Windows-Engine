#include "../../native/providers/ck_fmha/selective_qk_tail_policy.h"
#include <cstdio>
#include <initializer_list>
#include <limits>

int main() {
    using namespace qrt_selective_qk_tail;
    unsigned tail = 99u;
    for (const char* text : {static_cast<const char*>(nullptr), "", "0"})
        if (!parse(text, tail) || tail) return 1;
    for (const char* text : {"00", "01", "-1", "+1", " 1", "1 ", "1\n",
                             "1x", "8193", "4294967296", "999999999999999999999"}) {
        tail = 123u;
        if (parse(text, tail) || tail != 123u) return 2;
    }
    unsigned long long cases = 0u;
    for (unsigned requested = 0u; requested <= 8192u; ++requested) {
        char text[16]; std::snprintf(text, sizeof(text), "%u", requested);
        if (!parse(text, tail) || tail != requested) return 3;
        for (unsigned batch : {32u, 64u, 128u})
            for (unsigned count : {0u, 1u, 31u, 32u, 33u, 127u, 128u, 129u,
                                   7168u, 7169u, 7170u, 8191u, 8192u, 8193u}) {
                const unsigned prefix = prefix_queries(requested, 0u, count, batch);
                if (!requested || count <= requested || count > 8192u) {
                    if (prefix) return 4;
                } else {
                    if (prefix % batch || prefix >= count || count-prefix < requested ||
                        count-prefix-requested >= batch) return 5;
                    unsigned matrix = 0u, original = 0u;
                    for (unsigned offset = 0u; offset < count; offset += batch) {
                        const unsigned n = count-offset < batch ? count-offset : batch;
                        if (offset < prefix) {
                            if (offset+n > prefix) return 6;
                            matrix += n;
                        } else original += n;
                    }
                    if (matrix != prefix || matrix+original != count || original < requested)
                        return 7;
                }
                for (unsigned start : {1u, 7168u, 8192u, std::numeric_limits<unsigned>::max()})
                    if (prefix_queries(requested, start, count, batch)) return 8;
                ++cases;
            }
    }
    for (unsigned invalid : {0u, 1u, 31u, 33u, 127u, 129u, 8192u})
        if (prefix_queries(1024u, 0u, 8192u, invalid)) return 9;
    if (prefix_queries(8193u, 0u, 8192u, 128u) ||
        prefix_queries(1024u, 0u, 8192u, 128u) != 7168u ||
        prefix_queries(1024u, 0u, 7169u, 128u) != 6144u) return 10;
    std::printf("SELECTIVE_QK_TAIL_POLICY cases=%llu complete_query_ownership=1\n", cases);
}
