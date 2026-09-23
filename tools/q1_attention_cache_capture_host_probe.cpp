#include "native/providers/q1_attention_cache_capture.h"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace capture = qrt_q1_attention_cache_capture;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
std::vector<unsigned char> read(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file), "missing output");
    const auto count = file.tellg(); require(count > 0, "empty output");
    std::vector<unsigned char> data(static_cast<size_t>(count));
    file.seekg(0); file.read(reinterpret_cast<char *>(data.data()), count);
    require(bool(file), "output read"); return data;
}

int main(int argc, char **argv) try {
    require(argc == 2, "requires a new temporary directory");
    const std::filesystem::path root(argv[1]);
    require(std::filesystem::create_directory(root), "temporary directory exists");
    std::string error;
    capture::Plan plan;
    require(capture::parse(nullptr, nullptr, nullptr, plan, error) && !plan.directory, "disabled parser");
    for (const auto *layer : {"", "0", "18", "40", "-1", "19junk", "42949672960"})
        require(!capture::parse("unused", layer, "263291", plan, error), "invalid layer accepted");
    for (const auto *position : {"", "0", "-1", "263680", "42949672960"})
        require(!capture::parse("unused", "19", position, plan, error), "invalid position accepted");
    require(capture::parse("unused", "39", "263679", plan, error), "maximum parser");

    // A non-aligned logical prefix is larger than a staging chunk. Capacity
    // padding has a distinct sentinel and must not appear in the KV files.
    constexpr size_t prefix = 1537u, tail = 3u, tokens = prefix + tail, stride = tokens + 17u;
    std::vector<uint16_t> pk((prefix + 7u) * 512u, 0xfeed), pv(pk.size(), 0xbeef);
    std::vector<uint16_t> tk((tail + 5u) * 512u, 0xabcd), tv(tk.size(), 0xef01);
    for (size_t i = 0; i < prefix * 512u; ++i) { pk[i] = uint16_t(i * 17u); pv[i] = uint16_t(i * 19u); }
    for (size_t i = 0; i < tail * 512u; ++i) { tk[i] = uint16_t(i * 23u); tv[i] = uint16_t(i * 29u); }
    std::vector<float> rope(9216u, 1.25f), scores(16u * stride, -2.5f), context(4096u, 3.75f);
    std::vector<float> output(65536u, 4.5f), maxima(256u, -5.25f), sums(256u, 6.0f);
    capture::View v;
    v.prefix_k = pk.data(); v.prefix_v = pv.data(); v.tail_k = tk.data(); v.tail_v = tv.data();
    v.rope = rope.data(); v.scores = scores.data(); v.context = context.data();
    v.segment_output = output.data(); v.segment_max = maxima.data(); v.segment_sum = sums.data();
    v.prefix_tokens = prefix; v.tail_tokens = tail; v.score_stride = stride; v.segmented = true;
    v.prefix_k_capacity_bytes = pk.size() * 2u; v.prefix_v_capacity_bytes = pv.size() * 2u;
    v.tail_k_capacity_bytes = tk.size() * 2u; v.tail_v_capacity_bytes = tv.size() * 2u;
    require(capture::validate(v, tokens - 1u, error), "valid extents");
    auto invalid = v; invalid.tail_tokens = std::numeric_limits<size_t>::max();
    require(!capture::validate(invalid, tokens - 1u, error), "overflow tail accepted");
    invalid = v; invalid.prefix_tokens = std::numeric_limits<size_t>::max();
    require(!capture::validate(invalid, tokens - 1u, error), "overflow prefix accepted");
    invalid = v; invalid.prefix_v_capacity_bytes = prefix * capture::kv_row_bytes - 1u;
    require(!capture::validate(invalid, tokens - 1u, error), "truncated prefix accepted");
    invalid = v; invalid.tail_k_capacity_bytes = tail * capture::kv_row_bytes - 1u;
    require(!capture::validate(invalid, tokens - 1u, error), "truncated tail accepted");
    invalid = v; invalid.score_stride = tokens - 1u;
    require(!capture::validate(invalid, tokens - 1u, error), "truncated scores accepted");
    invalid = v; invalid.segmented = false;
    require(!capture::validate(invalid, tokens - 1u, error), "online path accepted");
    invalid = v; invalid.segment_sum = nullptr;
    require(!capture::validate(invalid, tokens - 1u, error), "missing surface accepted");
    require(!capture::validate(v, tokens, error), "wrong position accepted");
    auto maximum = v; maximum.prefix_tokens = 262144u;
    maximum.tail_tokens = capture::max_tokens - maximum.prefix_tokens;
    maximum.score_stride = capture::max_score_stride;
    maximum.prefix_k_capacity_bytes = maximum.prefix_v_capacity_bytes = maximum.prefix_tokens * capture::kv_row_bytes;
    maximum.tail_k_capacity_bytes = maximum.tail_v_capacity_bytes = maximum.tail_tokens * capture::kv_row_bytes;
    require(capture::validate(maximum, capture::max_tokens - 1u, error), "maximum logical extent rejected");
    require(capture::surface_bytes(maximum) <= capture::byte_limit, "maximum exceeds byte bound");

    size_t synchronizations = 0u, copies = 0u, bytes = 0u, max_chunk = 0u;
    auto sync = [&] { ++synchronizations; return true; };
    auto copy = [&](void *to, const void *from, size_t count) {
        ++copies; bytes += count; max_chunk = (std::max)(max_chunk, count);
        std::memcpy(to, from, count); return true;
    };
    bool captured = false;
    const std::string directory = (root / "complete").string();
    require(capture::parse(directory.c_str(), "19", std::to_string(tokens - 1u).c_str(), plan, error), "selected parser");
    require(capture::run(plan, 15u, tokens - 1u, v, captured, sync, copy, error), "unselected layer");
    require(capture::run(plan, 19u, tokens - 2u, v, captured, sync, copy, error), "unselected position");
    require(!captured && !synchronizations && !copies && !std::filesystem::exists(directory), "unselected capture had effects");
    require(capture::run(plan, 19u, tokens - 1u, v, captured, sync, copy, error), "capture failed");
    require(captured && synchronizations == 1u && bytes == capture::surface_bytes(v) &&
            max_chunk == qrt_prefix_linear_capture::copy_chunk_bytes, "bounded chunk/copy contract");
    const size_t initial_copies = copies;
    require(capture::run(plan, 19u, tokens - 1u, v, captured, sync, copy, error) && copies == initial_copies &&
            synchronizations == 1u, "repeat observation was not skipped");
    struct Surface { const char *name; const void *data; size_t bytes; };
    const Surface surfaces[] = {
        {"prefix-k-bf16.bin", pk.data(), prefix * capture::kv_row_bytes},
        {"prefix-v-bf16.bin", pv.data(), prefix * capture::kv_row_bytes},
        {"tail-k-bf16.bin", tk.data(), tail * capture::kv_row_bytes},
        {"tail-v-bf16.bin", tv.data(), tail * capture::kv_row_bytes},
        {"rope-f32.bin", rope.data(), capture::rope_bytes},
        {"scores-padded-f32.bin", scores.data(), scores.size() * 4u},
        {"segment-output-f32.bin", output.data(), capture::segment_output_bytes},
        {"segment-max-f32.bin", maxima.data(), capture::segment_scalar_bytes},
        {"segment-sum-f32.bin", sums.data(), capture::segment_scalar_bytes},
        {"context-f32.bin", context.data(), capture::context_bytes}
    };
    for (const auto &surface : surfaces) {
        const auto actual = read(std::filesystem::path(directory) / surface.name);
        require(actual.size() == surface.bytes && !std::memcmp(actual.data(), surface.data, surface.bytes), "surface bytes changed");
    }
    require(std::filesystem::is_regular_file(std::filesystem::path(directory) / "capture.json"), "no completion record");
    captured = false;
    require(!capture::run(plan, 19u, tokens - 1u, v, captured, sync, copy, error) && !captured &&
            synchronizations == 1u, "existing capture overwritten");

    const std::string failed = (root / "failed-copy").string();
    plan.directory = failed.c_str();
    require(!capture::run(plan, 19u, tokens - 1u, v, captured, sync,
        [](void *, const void *, size_t) { return false; }, error) && !captured, "failed copy accepted");
    require(!std::filesystem::exists(std::filesystem::path(failed) / "capture.json"), "false completion after failed copy");
    const std::string failed_sync = (root / "failed-sync").string();
    plan.directory = failed_sync.c_str();
    require(!capture::run(plan, 19u, tokens - 1u, v, captured, [] { return false; }, copy, error) &&
            !captured && copies == initial_copies, "failed synchronization copied inputs");
    const std::string expired = (root / "expired.bin").string();
    require(!qrt_prefix_linear_capture::save(expired, pk.data(), 2u, copy,
        qrt_prefix_linear_capture::Clock::now() - std::chrono::seconds(91), error) &&
        copies == initial_copies, "expired copy deadline ignored");
    std::cout << "{\"passed\":true,\"copied_bytes\":" << bytes << ",\"chunk_limit\":" << max_chunk
              << ",\"logical_kv_rows_only\":true,\"single_observation\":true,\"failure_controls\":true,"
                 "\"native_execution\":false,\"inference_acceptance\":false}\n";
    return 0;
} catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
