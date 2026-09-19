#include "../../native/providers/resident_text_shard_layout.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace layout = qrt_resident_text_shard;
static uint64_t copied = 0, tensors_checked = 0, failures_checked = 0;

static void generated() {
    std::mt19937 random(0x3958192u);
    assert(!layout::keep("model.visual.blocks.0.attn.qkv.weight"));
    assert(!layout::keep("mtp.layers.0.mlp.experts.down_proj"));
    for (const char* name : {"model.language_model.layers.0.weight", "lm_head.weight",
                            "model.visual_extra.weight", "mtp", "other.weight"})
        assert(layout::keep(name));
    for (unsigned trial = 0; trial < 1000; ++trial) {
        std::vector<layout::Tensor> tensors;
        uint64_t end = 8 + random() % 80;
        for (unsigned i = 0, n = 1 + random() % 48; i < n; ++i) {
            const uint64_t bytes = 1 + random() % 2048;
            tensors.push_back({end, bytes, trial % 11 == 0 ? false : random() % 3 != 0});
            end += bytes + (random() % 3 == 0 ? random() % 81 : 0);
        }
        const uint64_t file_bytes = end + random() % 127;
        std::vector<unsigned char> file(file_bytes);
        for (auto& v : file) v = static_cast<unsigned char>(random());
        std::shuffle(tensors.begin(), tensors.end(), random);
        for (bool text_only : {false, true}) {
            layout::Layout plan;
            assert(layout::build(file_bytes, tensors, text_only, &plan));
            constexpr size_t guard = 64;
            std::vector<unsigned char> device(plan.device_bytes + 2 * guard, 0xa5);
            std::vector<unsigned char> expected = device;
            if (!plan.packed) std::memcpy(expected.data() + guard, file.data(), file.size());
            for (const auto& t : tensors) {
                uint64_t destination = UINT64_MAX;
                const bool found = layout::offset(plan, t.begin, t.bytes, &destination);
                assert(found == (!text_only || t.keep));
                if (!found) { assert(destination == UINT64_MAX); continue; }
                assert(destination <= plan.device_bytes && t.bytes <= plan.device_bytes - destination);
                std::memcpy(expected.data() + guard + destination, file.data() + t.begin, t.bytes);
                uint64_t interior = UINT64_MAX;
                assert(layout::offset(plan, t.begin + t.bytes / 2, t.bytes - t.bytes / 2, &interior));
                assert(interior == destination + t.bytes / 2);
                ++tensors_checked;
            }
            for (uint64_t first = 0; first < file_bytes;) {
                const uint64_t bytes = (std::min)(uint64_t(1 + random() % 8192), file_bytes - first);
                assert(layout::copy(plan, first, bytes, [&](uint64_t to, uint64_t from, uint64_t n) {
                    assert(from <= bytes && n <= bytes - from);
                    assert(to <= plan.device_bytes && n <= plan.device_bytes - to);
                    std::memcpy(device.data() + guard + to, file.data() + first + from, n);
                    copied += n;
                    return true;
                }));
                first += bytes;
            }
            assert(device == expected);
            unsigned called = 0;
            assert(layout::copy(plan, 0, file_bytes, [&](uint64_t, uint64_t, uint64_t) {
                ++called; return false;
            }) == plan.spans.empty());
            assert(called == (plan.spans.empty() ? 0u : 1u));
            ++failures_checked;
        }
    }
}

static void invalid_and_wide() {
    layout::Layout untouched;
    untouched.file_bytes = 7;
    for (const auto& bad : std::vector<std::vector<layout::Tensor>>{
            {}, {{0, 0, true}}, {{0, 11, true}}, {{9, 2, false}},
            {{0, 6, true}, {5, 3, false}}, {{0, 1, false}, {UINT64_MAX, 1, true}}}) {
        assert(!layout::build(10, bad, true, &untouched));
        assert(untouched.file_bytes == 7 && untouched.spans.empty());
        ++failures_checked;
    }
    assert(!layout::build(0, {{0, 1, true}}, true, &untouched));
    assert(!layout::build(10, {{0, 1, true}}, true, nullptr));
    layout::Layout wide;
    assert(layout::build(UINT64_MAX, {{0, 1, false}, {UINT64_MAX - 1024, 1024, true}}, true, &wide));
    uint64_t offset = 123;
    assert(layout::offset(wide, UINT64_MAX - 2, 2, &offset) && offset == 1022);
    assert(!layout::offset(wide, UINT64_MAX - 2, 3, &offset) && offset == 1022);
    assert(!layout::offset(wide, UINT64_MAX, 0, &offset));
    assert(!layout::copy(wide, UINT64_MAX - 2, 3, [](auto, auto, auto) { return true; }));
    layout::Layout holes;
    assert(layout::build(100, {{10, 10, true}, {20, 10, false}, {30, 10, true}}, true, &holes));
    assert(!layout::offset(holes, 15, 20, &offset));
    assert(!layout::offset(holes, 20, 1, &offset));
    assert(layout::offset(holes, 30, 10, &offset) && offset == 256);
    unsigned calls = 0;
    assert(layout::copy(holes, 5, 40, [&](auto, auto, auto) { return ++calls < 2; }) == false);
    assert(calls == 2);
    // Destination padding can overflow even when all source intervals fit.
    assert(!layout::build(UINT64_MAX,
        {{0, 1, true}, {1, 1, false}, {2, UINT64_MAX - 2, true}}, true, &untouched));
    assert(untouched.file_bytes == 7);
}

static void original_metadata(const char* name) {
    std::ifstream input(name);
    unsigned shards = 0;
    if (!(input >> shards) || shards != 26) throw std::runtime_error("shard inventory count");
    for (unsigned s = 0; s < shards; ++s) {
        std::string shard;
        uint64_t bytes, count;
        if (!(input >> shard >> bytes >> count) || !count || count > 10000)
            throw std::runtime_error("shard inventory record");
        std::vector<layout::Tensor> tensors;
        for (uint64_t i = 0; i < count; ++i) {
            std::string tensor;
            uint64_t begin, size;
            if (!(input >> tensor >> begin >> size)) throw std::runtime_error("tensor inventory record");
            tensors.push_back({begin, size, layout::keep(tensor)});
        }
        layout::Layout packed;
        if (!layout::build(bytes, tensors, true, &packed)) throw std::runtime_error("original layout rejected");
        uint64_t mapped_bytes = 0, complete_bytes = 0, chunks = 0;
        for (const auto& t : tensors) {
            uint64_t mapped = UINT64_MAX;
            if (layout::offset(packed, t.begin, t.bytes, &mapped) != t.keep)
                throw std::runtime_error("original tensor mapping");
            if (t.keep) mapped_bytes += t.bytes;
            else if (mapped != UINT64_MAX) throw std::runtime_error("omitted tensor published");
        }
        constexpr uint64_t chunk = 32 * 1024 * 1024;
        for (uint64_t first = 0; first < bytes; first += (std::min)(chunk, bytes - first)) {
            if (!layout::copy(packed, first, (std::min)(chunk, bytes - first),
                [&](uint64_t, uint64_t, uint64_t size) { complete_bytes += size; return true; }))
                throw std::runtime_error("original chunk mapping");
            ++chunks;
        }
        uint64_t span_bytes = 0;
        for (const auto& span : packed.spans) span_bytes += span.bytes;
        if (complete_bytes != span_bytes || (packed.packed && mapped_bytes != span_bytes))
            throw std::runtime_error("original byte coverage");
        std::printf("{\"kind\":\"original_model_text_shard_layout\",\"shard\":\"%s\",\"file_bytes\":%llu,\"device_bytes\":%llu,\"kept_tensors\":%llu,\"omitted_tensors\":%llu,\"omitted_tensor_bytes\":%llu,\"spans\":%zu,\"logical_chunks\":%llu,\"copy_bytes\":%llu,\"weights_read\":false,\"memory_reduction_measured\":false}\n",
            shard.c_str(), (unsigned long long)bytes, (unsigned long long)packed.device_bytes,
            (unsigned long long)packed.kept_tensors, (unsigned long long)packed.omitted_tensors,
            (unsigned long long)packed.omitted_tensor_bytes, packed.spans.size(),
            (unsigned long long)chunks, (unsigned long long)complete_bytes);
    }
    std::string extra;
    if (input >> extra) throw std::runtime_error("extra inventory records");
}

int main(int argc, char** argv) try {
    if (argc > 2) throw std::runtime_error("optional original inventory argument");
    generated();
    invalid_and_wide();
    std::printf("{\"kind\":\"resident_text_shard_layout_host\",\"generated_cases\":2000,\"copied_bytes\":%llu,\"tensor_and_slice_checks\":%llu,\"failed_copy_or_layout_checks\":%llu,\"redzones_and_padding_pass\":true,\"mismatches\":0}\n",
        (unsigned long long)copied, (unsigned long long)tensors_checked, (unsigned long long)failures_checked);
    if (argc == 2) original_metadata(argv[1]);
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
