#include "native/providers/resident_ordered_shard_layout.h"
#include "native/providers/resident_fixed_weight_order.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>

namespace ordered = qrt_resident_ordered_shard;
namespace text_layout = qrt_resident_text_shard;
static uint64_t cases = 0, copied = 0, checked = 0, rejected = 0;

struct Buffer {
    std::vector<unsigned char> bytes, written;
    explicit Buffer(uint64_t size) : bytes(size + 64u, 0xa5), written(size, 0) {}
    unsigned char* data() { return bytes.data() + 32u; }
    void check() const {
        for (unsigned i = 0; i < 32u; ++i)
            assert(bytes[i] == 0xa5 && bytes[bytes.size() - 1u - i] == 0xa5);
        for (size_t i = 0; i < written.size(); ++i)
            if (!written[i]) assert(bytes[i + 32u] == 0xa5);
    }
};

static void exercise(const std::vector<ordered::Input>& inputs,
                     const std::vector<std::string>& order, bool text_only) {
    ordered::Plan plan;
    assert(ordered::build(inputs, order, text_only, &plan));
    Buffer fixed(plan.fixed_bytes);
    std::vector<Buffer> ordinary;
    for (const auto& shard : plan.shards) ordinary.emplace_back(shard.ordinary.device_bytes);
    std::unordered_set<std::string> selected(order.begin(), order.end());
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<unsigned char> source(inputs[i].file_bytes), consumed(source.size(), 0);
        for (size_t j = 0; j < source.size(); ++j)
            source[j] = static_cast<unsigned char>(j * 19u + j / 7u + i * 31u);
        for (unsigned part = 0; part < 2u; ++part) {
            const auto& layout = part ? plan.shards[i].fixed : plan.shards[i].ordinary;
            auto& destination = part ? fixed : ordinary[i];
            for (uint64_t first = 0; first < source.size(); first += 113u) {
                const uint64_t count = (std::min<uint64_t>)(113u, source.size() - first);
                unsigned calls = 0;
                auto copy = [&](uint64_t to, uint64_t from, uint64_t bytes) {
                    ++calls;
                    assert(to <= destination.written.size() && bytes <= destination.written.size() - to);
                    assert(from <= count && bytes <= count - from);
                    for (uint64_t k = 0; k < bytes; ++k) {
                        assert(!destination.written[to + k]++);
                        assert(!consumed[first + from + k]++);
                    }
                    std::memcpy(destination.data() + to, source.data() + first + from, bytes);
                    copied += bytes;
                    return true;
                };
                assert(text_layout::copy(layout, first, count, copy));
                if (calls) {
                    unsigned failed_calls = 0;
                    assert(!text_layout::copy(layout, first, count,
                        [&](uint64_t, uint64_t, uint64_t) { ++failed_calls; return false; }));
                    assert(failed_calls == 1u);
                    ++rejected;
                }
            }
        }
        for (const auto& tensor : inputs[i].tensors) {
            const bool omitted = text_only && !text_layout::keep(tensor.name);
            const bool is_fixed = selected.count(tensor.name) != 0;
            uint64_t a = UINT64_MAX, b = UINT64_MAX;
            const bool ordinary_found = text_layout::offset(plan.shards[i].ordinary, tensor.begin, tensor.bytes, &a);
            const bool fixed_found = text_layout::offset(plan.shards[i].fixed, tensor.begin, tensor.bytes, &b);
            assert(ordinary_found == (!omitted && !is_fixed));
            assert(fixed_found == is_fixed);
            if (!omitted) {
                const auto* actual = is_fixed ? fixed.data() + b : ordinary[i].data() + a;
                assert(std::memcmp(actual, source.data() + tensor.begin, tensor.bytes) == 0);
                const auto& layout = is_fixed ? plan.shards[i].fixed : plan.shards[i].ordinary;
                uint64_t slice = 0;
                assert(text_layout::offset(layout, tensor.begin + tensor.bytes / 2u,
                    tensor.bytes - tensor.bytes / 2u, &slice));
                assert(slice == (is_fixed ? b : a) + tensor.bytes / 2u);
            }
            for (uint64_t k = 0; k < tensor.bytes; ++k)
                assert(consumed[tensor.begin + k] == (omitted ? 0u : 1u));
            ++checked;
        }
    }
    fixed.check();
    for (const auto& buffer : ordinary) buffer.check();
    ++cases;
}

static void invalid(const std::vector<ordered::Input>& inputs,
                    const std::vector<std::string>& order) {
    ordered::Plan unchanged;
    unchanged.fixed_tensors = 777u;
    assert(!ordered::build(inputs, order, true, &unchanged));
    assert(unchanged.fixed_tensors == 777u && unchanged.shards.empty());
    ++rejected;
}

static void audit_inventory(const char* path) {
    std::ifstream input(path);
    size_t count = 0;
    input >> count;
    assert(count == 26u);
    std::vector<ordered::Input> shards(count);
    for (auto& shard : shards) {
        std::string name;
        size_t tensors = 0;
        input >> name >> shard.file_bytes >> tensors;
        shard.tensors.resize(tensors);
        for (auto& tensor : shard.tensors) input >> tensor.name >> tensor.begin >> tensor.bytes;
    }
    assert(input.good());
    std::vector<std::string> order;
    assert(qrt_resident_fixed_order::complete_names(&order) && order.size() == 611u);
    ordered::Plan plan;
    assert(ordered::build(shards, order, true, &plan));
    assert(plan.fixed_bytes == 3879600640u && plan.fixed_tensor_bytes == 3879589120u);
    assert(plan.omitted_tensors == 352u && plan.omitted_tensor_bytes == 2582424032u);
    assert(plan.file_bytes == 71903776776u);
    uint64_t tensors = 0, spans = 0, fixed_spans = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto& shard = plan.shards[i];
        spans += shard.ordinary.spans.size(); fixed_spans += shard.fixed.spans.size();
        for (const auto& tensor : shards[i].tensors) {
            uint64_t offset = 0;
            const bool a = text_layout::offset(shard.ordinary, tensor.begin, tensor.bytes, &offset);
            const bool b = text_layout::offset(shard.fixed, tensor.begin, tensor.bytes, &offset);
            assert(unsigned(a) + unsigned(b) == unsigned(text_layout::keep(tensor.name)));
            ++tensors;
        }
    }
    assert(tensors == 1045u);
    std::printf("{\"kind\":\"original_model_ordered_shard_layout\",\"tensors\":%llu,"
        "\"ordinary_bytes\":%llu,\"fixed_bytes\":%llu,\"total_device_bytes\":%llu,"
        "\"ordinary_spans\":%llu,\"fixed_spans\":%llu,\"fixed_tensors\":%llu,"
        "\"omitted_tensors\":%llu,\"verification_samples\":%llu,"
        "\"original_file_bytes\":%llu,\"weights_read\":false,\"mismatches\":0}\n",
        (unsigned long long)tensors, (unsigned long long)plan.ordinary_bytes,
        (unsigned long long)plan.fixed_bytes, (unsigned long long)(plan.ordinary_bytes + plan.fixed_bytes),
        (unsigned long long)spans, (unsigned long long)fixed_spans, (unsigned long long)plan.fixed_tensors,
        (unsigned long long)plan.omitted_tensors, (unsigned long long)(3u * (spans + fixed_spans)),
        (unsigned long long)plan.file_bytes);
}

int main(int argc, char** argv) {
    std::mt19937 random(491379u);
    for (unsigned iteration = 0; iteration < 500u; ++iteration) {
        std::vector<ordered::Input> inputs(1u + random() % 5u);
        std::vector<std::string> available;
        unsigned serial = 0;
        for (auto& input : inputs) {
            uint64_t next = 31u + random() % 41u;
            const unsigned count = 4u + random() % 19u;
            for (unsigned i = 0; i < count; ++i) {
                const bool omitted = i % 5u == 1u;
                const std::string name = (omitted ? "model.visual." : "unknown.retained.") + std::to_string(serial++);
                const uint64_t bytes = 1u + random() % 337u;
                input.tensors.push_back({name, next, bytes});
                next += bytes + random() % 29u;
                if (!omitted) available.push_back(name);
            }
            input.file_bytes = next + 19u;
        }
        std::shuffle(available.begin(), available.end(), random);
        available.resize(1u + random() % available.size());
        exercise(inputs, available, false);
        exercise(inputs, available, true);
    }
    invalid({}, {"a"});
    invalid({{1024u, {{"a", 0u, 512u}}}}, {});
    invalid({{1024u, {{"a", 0u, 512u}}}}, {"missing"});
    invalid({{1024u, {{"a", 0u, 512u}}}}, {"a", "a"});
    invalid({{1024u, {{"a", 0u, 512u}, {"a", 512u, 512u}}}}, {"a"});
    invalid({{1024u, {{"a", 0u, 512u}, {"b", 511u, 512u}}}}, {"a"});
    invalid({{1024u, {{"a", 1024u, 1u}}}}, {"a"});
    invalid({{1024u, {{"model.visual.a", 0u, 512u}}}}, {"model.visual.a"});
    invalid({{UINT64_MAX, {{"a", 0u, UINT64_MAX}}}, {1u, {{"b", 0u, 1u}}}}, {"a", "b"});
    invalid({{UINT64_MAX, {{"a", 0u, UINT64_MAX - 256u}, {"b", UINT64_MAX - 256u, 256u}}}}, {"a", "b"});
    // Global destination offsets may go backwards in original disk order.
    exercise({{1024u, {{"a", 17u, 257u}, {"b", 300u, 256u}, {"c", 700u, 255u}}}}, {"c", "a", "b"}, true);
    std::printf("{\"kind\":\"resident_ordered_shard_layout_host\",\"generated_cases\":%llu,"
        "\"copied_bytes\":%llu,\"tensor_checks\":%llu,\"failure_checks\":%llu,"
        "\"redzones_and_padding_pass\":true,\"mismatches\":0}\n",
        (unsigned long long)cases, (unsigned long long)copied,
        (unsigned long long)checked, (unsigned long long)rejected);
    if (argc == 2) audit_inventory(argv[1]);
    return 0;
}
