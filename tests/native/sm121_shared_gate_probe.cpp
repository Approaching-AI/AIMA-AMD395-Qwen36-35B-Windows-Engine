#include "../../native/providers/moe_accumulator/sm121_shared_gate.h"
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::ifstream f(argv[1], std::ios::binary);
    unsigned count = 0;
    for (;;) {
        uint16_t expected = 0;
        if (!f.read(reinterpret_cast<char *>(&expected), sizeof(expected))) {
            if (f.eof() && f.gcount() == 0 && count == 6) return 0;
            return 3;
        }
        uint16_t input[2048], weight[2048];
        f.read(reinterpret_cast<char *>(input), sizeof(input));
        f.read(reinterpret_cast<char *>(weight), sizeof(weight));
        if (!f) return 4;
        float partial[qrt_sm121_shared_gate::lanes];
        for (unsigned lane = 0; lane < qrt_sm121_shared_gate::lanes; ++lane)
            partial[lane] = qrt_sm121_shared_gate::lane_dot(input, weight, lane);
        for (unsigned offset = qrt_sm121_shared_gate::lanes / 2; offset; offset /= 2)
            for (unsigned lane = 0; lane < offset; ++lane)
                partial[lane] = partial[lane] + partial[lane + offset];
        uint32_t bits;
        memcpy(&bits, partial, sizeof(bits));
        const uint16_t actual = static_cast<uint16_t>(
            (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
        if (actual != expected) {
            std::cerr << "case " << count << ": " << actual << " != " << expected << '\n';
            return 5;
        }
        ++count;
    }
}
