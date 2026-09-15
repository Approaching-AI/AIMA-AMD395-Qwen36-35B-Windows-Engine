#ifndef QRT_MODULAR_INTEGER_CORE_NATIVE_PROBE_H
#define QRT_MODULAR_INTEGER_CORE_NATIVE_PROBE_H
#include <cmath>
namespace modular_probe {
namespace core = qrt_sm121_modular_core;
constexpr unsigned tiles = 8192u;
struct Cell { int64_t recovered; float approximate; int32_t residue; uint32_t accepted, padding; };
__global__ void prepare(const uint16_t* input, core::Row* output, unsigned count) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    core::Row result{};
    for (unsigned i = 0u; i < 16u; ++i) result.original[i] = input[size_t(row) * 16u + i];
    core::prepare(result); output[row] = result;
}
__global__ void compare(const core::Row* rows, Cell* output) {
    const unsigned lane = threadIdx.x, base = blockIdx.x * 32u;
    const auto approximate = core::products(rows[base + lane % 16u], rows[base + 16u + lane % 16u]);
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const unsigned row = 2u * i + lane / 16u, column = lane % 16u;
        const auto low = core::remainders(rows[base + row], rows[base + 16u + column], 0u);
        Cell result{}; result.approximate = approximate[i]; result.residue = low.residue;
        result.accepted = unsigned(core::recover(result.approximate, uint32_t(result.residue), &result.recovered));
        output[blockIdx.x * 256u + row * 16u + column] = result;
    }
}
double half(uint16_t value) {
    const unsigned exponent = (value >> 10u) & 31u;
    return exponent ? ((value & 32768u) ? -1.0 : 1.0) * std::ldexp(double(1024u + (value & 1023u)), int(exponent) - 25) : 0.0;
}
void run() {
    uint32_t state = 0x8192395u;
    auto random = [&]() { state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u; return state; };
    std::vector<uint16_t> values(tiles * 512u + 2u * guard, 0x5a5au);
    std::vector<core::Row> expected_rows(tiles * 32u);
    std::vector<int> integers(tiles * 512u);
    size_t eligible_rows = 0u;
    for (unsigned row = 0u; row < expected_rows.size(); ++row) {
        const unsigned tile = row / 32u, mode = tile % 9u, exponent = 1u + random() % 246u;
        bool valid = true, nonzero = false; int maximum = 0;
        for (unsigned i = 0u; i < 16u; ++i) {
            uint16_t value = uint16_t((random() & 0x807fu) | ((exponent + random() % 8u) << 7u));
            if (tile < 128u) value = uint16_t(row * 16u + i); // All original BF16 bit patterns.
            else if (mode == 0u) value = uint16_t(0x3fffu | (random() & 0x8000u));
            else if (mode == 1u) value = uint16_t(0x3fffu | ((i & 1u) && row % 32u < 16u ? 0x8000u : 0u));
            else if (mode == 2u) value = i == row % 16u ? uint16_t(value & 0x7fffu) : 0u;
            else if (mode == 3u) value = uint16_t((random() & 0x807fu) | ((1u + random() % 254u) << 7u));
            else if (mode == 4u) value = uint16_t((random() & 0x807fu) | (127u << 7u));
            else if (mode == 5u) value = uint16_t((random() & 127u) | 0xbf80u);
            else if (mode == 6u && row % 32u < 16u) value = 0x8000u;
            values[guard + row * 16u + i] = value; expected_rows[row].original[i] = value;
            if (value & 0x7fffu) {
                const int e = int((value >> 7u) & 255u); nonzero = true;
                valid = valid && e != 0 && e != 255; maximum = std::max(maximum, e);
            }
        }
        const int unit = !valid ? -1 : !nonzero ? 127 : maximum > 8 ? maximum - 7 : 1;
        core::prepare(expected_rows[row]); eligible_rows += unsigned(unit >= 0);
        if (expected_rows[row].unit != unit) throw std::runtime_error("native probe unit differs from independent maximum");
        for (unsigned i = 0u; i < 16u; ++i) {
            const uint16_t value = values[guard + row * 16u + i];
            const int magnitude = unit < 0 || !(value & 0x7fffu) ? 0 :
                int(std::ldexp(double(128u + (value & 127u)), int((value >> 7u) & 255u) - unit));
            const int expected = value & 32768u ? -magnitude : magnitude;
            integers[row * 16u + i] = expected;
            if (magnitude > 32640 || half(expected_rows[row].half[i]) != expected ||
                ((expected_rows[row].magnitudes[i / 2u] >> ((i & 1u) * 16u)) & 65535u) != unsigned(magnitude))
                throw std::runtime_error("native probe exact FP16 core encoding failed");
        }
    }
    std::vector<unsigned char> encoded(expected_rows.size() * sizeof(core::Row) + 2u * guard, 0xa5u);
    std::memcpy(encoded.data() + guard, expected_rows.data(), expected_rows.size() * sizeof(core::Row));
    Device input(values.size() * 2u), prepared(encoded.size()), output((size_t(tiles) * 256u + 2u * guard) * sizeof(Cell));
    upload(input, values); check(hipMemset(prepared.pointer, 0xa5, encoded.size()));
    check(hipMemset(output.pointer, 0xa5, (size_t(tiles) * 256u + 2u * guard) * sizeof(Cell))); finish();
    const auto begin = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(prepare, dim3((expected_rows.size() + 255u) / 256u), dim3(256u), 0u, nullptr,
        input.as<uint16_t>() + guard, reinterpret_cast<core::Row*>(prepared.as<unsigned char>() + guard), unsigned(expected_rows.size()));
    check(hipGetLastError()); finish(); const double preparation_ms = elapsed(begin);
    unchanged(prepared, encoded);
    hipLaunchKernelGGL(compare, dim3(tiles), dim3(32u), 0u, nullptr,
        reinterpret_cast<core::Row*>(prepared.as<unsigned char>() + guard), output.as<Cell>() + guard);
    check(hipGetLastError()); finish();
    const auto actual = download<Cell>(output, size_t(tiles) * 256u + 2u * guard);
    size_t residue_bad = 0u, recovery_bad = 0u, bound_bad = 0u; double max_error = 0.0;
    for (size_t cell = 0u; cell < actual.size(); ++cell) {
        if (cell < guard || cell >= size_t(tiles) * 256u + guard) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&actual[cell]);
            if (std::any_of(bytes, bytes + sizeof(Cell), [](unsigned char b) { return b != 0xa5u; }))
                throw std::runtime_error("native modular core output guard changed");
            continue;
        }
        const unsigned index = unsigned(cell - guard), tile = index / 256u, row = index % 256u / 16u, column = index % 16u;
        int64_t expected = 0;
        for (unsigned k = 0u; k < 16u; ++k) expected += int64_t(integers[tile * 512u + row * 16u + k]) * integers[tile * 512u + 256u + column * 16u + k];
        const auto& value = actual[cell];
        const double error = std::abs(double(value.approximate) - double(expected)); max_error = std::max(max_error, error);
        bound_bad += unsigned(!(error < 32768.0));
        residue_bad += unsigned((uint32_t(value.residue) & 65535u) != (uint32_t(expected) & 65535u));
        if (!value.accepted || value.recovered != expected) {
            if (recovery_bad < 4u) std::printf("{\"kind\":\"modular_core_counterexample\",\"cell\":%u,\"expected\":%lld,\"actual\":%lld,\"approximate\":%.12g,\"residue\":%d,\"accepted\":%u}\n", index, static_cast<long long>(expected), static_cast<long long>(value.recovered), double(value.approximate), value.residue, value.accepted);
            ++recovery_bad;
        }
    }
    unchanged(input, values); unchanged(prepared, encoded);
    std::printf("{\"kind\":\"modular_integer_core_native\",\"tiles\":8192,\"dot_positions\":2097152,\"original_bf16_bit_patterns\":65536,\"eligible_rows\":%zu,\"exact_core_encodings_checked\":4194304,\"residue_mismatches\":%zu,\"recovery_mismatches\":%zu,\"conditional_bound_violations\":%zu,\"maximum_native_absolute_error\":%.9g,\"recovery_half_spacing\":32768,\"initial_preparation_ms\":%.6f,\"prepared_row_bytes\":116,\"redzones_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n", eligible_rows, residue_bad, recovery_bad, bound_bad, max_error, preparation_ms);
    std::fflush(stdout);
    if (residue_bad || recovery_bad || bound_bad) throw std::runtime_error("native modular core differs from independent integer oracle");
}
}
#endif
