#ifndef QRT_GATE_INPUT_CAPTURE_H
#define QRT_GATE_INPUT_CAPTURE_H
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace qrt_gate_capture {
struct Span { const float *data; size_t size; };
struct Bf16Span { const uint16_t *data; size_t size; };
// Host snapshots at the projection/gating boundary. Each projection is at most
// 1 MiB; parameters are exactly 32 values. No value is supplied to inference.
inline bool write(const char *directory, unsigned int layer, unsigned int tokens,
                  Span a, Span b, Bf16Span a_log, Bf16Span dt_bias, std::string *error) {
    static_assert(sizeof(float) == 4, "gate capture requires float32");
    if (!error) return false;
    if (!directory || !*directory || layer >= 40u || tokens == 0u || tokens > 8192u ||
        !a.data || !b.data || !a_log.data || !dt_bias.data ||
        a.size != size_t(tokens) * 32u || b.size != a.size ||
        a_log.size != 32u || dt_bias.size != 32u) {
        *error = "gate capture requires a complete bounded projection/parameter view";
        return false;
    }
    try {
        const std::filesystem::path root(directory);
        if (!std::filesystem::create_directory(root)) {
            *error = "gate capture refuses an existing directory"; return false;
        }
        const char *names[] = {"a-f32.bin", "b-f32.bin", "a-log-bf16.bin", "dt-bias-bf16.bin"};
        struct Payload { const void *data; size_t bytes; };
        const Payload spans[] = {{a.data, a.size * 4u}, {b.data, b.size * 4u},
                                 {a_log.data, a_log.size * 2u}, {dt_bias.data, dt_bias.size * 2u}};
        for (unsigned int i = 0u; i < 4u; ++i) {
            std::ofstream file(root / names[i], std::ios::binary);
            file.write(reinterpret_cast<const char *>(spans[i].data),
                       static_cast<std::streamsize>(spans[i].bytes));
            file.close();
            if (!file) { *error = "gate capture write or close failed"; return false; }
        }
        std::ofstream record(root / "capture.json");
        record << "{\"kind\":\"native_gate_inputs\",\"layer\":" << layer
               << ",\"tokens\":" << tokens
               << ",\"heads\":32,\"dtype\":\"float32\",\"layout\":\"token_rows\","
                  "\"parameter_dtype\":\"bfloat16\",\"complete\":true,\"inference_acceptance\":false}\n";
        record.close();
        if (!record) { *error = "gate capture completion write failed"; return false; }
        return true;
    } catch (...) {
        *error = "gate capture filesystem failure"; return false;
    }
}
}
#endif
