#include "../../native/providers/gdn/sm121_exp2_interpolated.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <vector>

namespace packed = qrt_sm121_exp2_interpolated;
namespace old = qrt_sm121_exp2;
void put32(std::vector<unsigned char>& table, size_t at, uint32_t value) {
    std::memcpy(table.data() + at, &value, 4u);
}
int64_t floor_reference(int64_t value) {
    return value < 0 ? -((-value + 255) / 256) : value / 256;
}
int main() {
    // All exterior inputs must return without dereferencing a table pointer.
    const uint32_t exterior[] = {0u, 0x80000000u, 1u, 0x80000001u,
        old::begin - 1u, 0x80000000u | (old::begin - 1u), old::begin,
        old::positive_one_end - 1u, old::positive_one_end, 0x3f800000u,
        0x80000000u | old::end, 0xff800000u, 0x7f800000u, 0x7fc00000u,
        0xffc00000u, 0x7fffffffu, 0xffffffffu};
    for (uint32_t bits : exterior)
        assert(old::bits(packed::evaluate(nullptr, old::value(bits))) ==
               old::bits(old::evaluate(nullptr, old::value(bits))));
    for (int32_t value : {INT_MIN, INT_MIN + 255, -5788245, -257, -256,
                         -255, -1, 0, 1, 255, 256, 257, 5788245, INT_MAX})
        assert(packed::floor_div256(value) == floor_reference(value));

    std::vector<unsigned char> table(packed::table_bytes, 0u);
    std::memcpy(table.data(), "QEX2IPL1", 8u);
    const uint32_t header[] = {1u, 8u, packed::begin, packed::end,
        old::positive_one_end, packed::pages, 8u, 3u};
    std::memcpy(table.data() + 8u, header, sizeof(header));
    std::memcpy(table.data() + 40u, &packed::payload_bytes, 8u);
    std::memcpy(table.data() + 48u, old::sha256, 32u);
    assert(packed::valid_layout(table.data(), table.size()));
    assert(!packed::valid_layout(nullptr, table.size()));
    assert(!packed::valid_layout(table.data(), table.size() - 1u));
    assert(!packed::valid_layout(table.data(), table.size() + 1u));
    for (size_t at : {0u, 8u, 12u, 16u, 20u, 24u, 28u, 32u, 36u, 40u, 48u, 80u, 95u}) {
        table[at] ^= 1u;
        assert(!packed::valid_layout(table.data(), table.size()));
        table[at] ^= 1u;
    }
    const auto invalid_entry = [&](size_t at, uint32_t value) {
        put32(table, at, value);
        assert(!packed::valid_layout(table.data(), table.size()));
        put32(table, at, 0u);
    };
    invalid_entry(packed::header_bytes, 0x3f800001u);
    invalid_entry(packed::header_bytes, 1u); // A constant page must remain flat.
    invalid_entry(packed::header_bytes + 4u, 32u);
    invalid_entry(packed::header_bytes + 4u, (1u << 29u) | 1u);
    invalid_entry(packed::header_bytes + 4u, (7u << 29u) | (uint32_t(packed::payload_bytes) - 32u));
    invalid_entry(packed::header_bytes + 4u, (1u << 29u) | 0x1fffffe0u);
    invalid_entry(packed::payload_start - 8u, 1u);
    invalid_entry(packed::payload_start - 4u, 1u << 29u);

    // Exercise every packing width, both slope signs, byte crossings, and the
    // actual first/last input. Expected values use independent int64 division.
    constexpr unsigned widths[] = {0u, 1u, 1u, 2u, 4u, 8u, 16u, 32u};
    constexpr int biases[] = {0, 0, -1, -1, -7, -127, -32767, 0};
    uint64_t cells = 0;
    for (unsigned kind = 0; kind < 8u; ++kind) {
        for (int slope : {-22699, -257, -1, 0, 1, 257, 22699}) {
            const uint32_t base = 0x3f000000u;
            for (uint32_t page : {0u, 17u, packed::pages - 1u}) {
                const size_t entry = packed::header_bytes + uint64_t(page) * 8u;
                put32(table, entry, base); put32(table, entry + 4u, kind << 29u);
                put32(table, entry + 8u, uint32_t(int64_t(base) + slope));
                std::memset(table.data() + packed::payload_start, 0, 1024u);
                uint32_t expected[256];
                for (unsigned cell = 0u; cell < 256u; ++cell) {
                    const unsigned width = widths[kind];
                    const uint32_t code = kind == 7u ? base - 256u * cell :
                        width == 0u ? 0u : (cell * 283u + cell / 7u) & ((1u << width) - 1u);
                    expected[cell] = kind == 0u ? base : kind == 7u ? code :
                        uint32_t(int64_t(base) + floor_reference(int64_t(slope) * cell) + code + biases[kind]);
                    for (unsigned bit = 0u; bit < width; ++bit) {
                        const unsigned position = cell * width + bit;
                        if ((code >> bit) & 1u)
                            table[packed::payload_start + position / 8u] |= 1u << (position % 8u);
                    }
                }
                for (unsigned cell = 0u; cell < 256u; ++cell) {
                    const uint32_t relative = page * 256u + cell;
                    assert(packed::decode(table.data(), relative) == expected[cell]);
                    assert(old::bits(packed::evaluate(table.data(), old::value(0x80000000u | (packed::begin + relative)))) == expected[cell]);
                    ++cells;
                }
            }
        }
    }
    std::printf("exp2_interpolated_cells=%llu domain_and_layout_guards=pass\n", (unsigned long long)cells);
}
