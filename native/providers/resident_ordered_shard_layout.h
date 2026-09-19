#pragma once
#include "resident_text_shard_layout.h"
#include <unordered_map>
#include <unordered_set>

namespace qrt_resident_ordered_shard {
using qrt_resident_text_shard::Layout;
using qrt_resident_text_shard::Span;
struct Tensor {
    std::string name;
    uint64_t begin = 0, bytes = 0;
};
struct Input {
    uint64_t file_bytes = 0;
    std::vector<Tensor> tensors;
};
struct Shard {
    Layout ordinary, fixed;
    uint64_t omitted_tensors = 0, omitted_tensor_bytes = 0;
};
struct Plan {
    std::vector<Shard> shards;
    uint64_t fixed_bytes = 0, fixed_tensor_bytes = 0, fixed_tensors = 0;
    uint64_t ordinary_bytes = 0, file_bytes = 0;
    uint64_t omitted_tensors = 0, omitted_tensor_bytes = 0;
};

inline bool add(uint64_t value, uint64_t* sum) {
    if (value > UINT64_MAX - *sum) return false;
    *sum += value;
    return true;
}

// One destination owner holds the requested fixed tensors in runtime order.
// Every other retained byte belongs to its ordinary shard owner. Original
// source offsets remain authoritative; no tensor is present in both owners.
// This is a metadata plan only and performs no allocation or data transfer.
inline bool build(const std::vector<Input>& inputs,
                  const std::vector<std::string>& fixed_order,
                  bool text_only, Plan* output) {
    if (!output || inputs.empty() || fixed_order.empty()) return false;
    struct Location { size_t shard; const Tensor* tensor; };
    std::unordered_map<std::string, Location> locations;
    std::unordered_set<std::string> selected;
    Plan next;
    next.shards.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].file_bytes || inputs[i].tensors.empty() ||
            !add(inputs[i].file_bytes, &next.file_bytes)) return false;
        for (const auto& tensor : inputs[i].tensors) {
            if (tensor.name.empty() || !tensor.bytes ||
                !locations.emplace(tensor.name, Location{i, &tensor}).second)
                return false;
        }
        next.shards[i].fixed.file_bytes = inputs[i].file_bytes;
    }
    for (const auto& name : fixed_order) {
        const auto found = locations.find(name);
        if (found == locations.end() || !selected.insert(name).second ||
            (text_only && !qrt_resident_text_shard::keep(name))) return false;
        const auto& tensor = *found->second.tensor;
        if (next.fixed_bytes > UINT64_MAX - 255u) return false;
        const uint64_t destination = (next.fixed_bytes + 255u) & ~uint64_t(255u);
        next.fixed_bytes = destination;
        if (!add(tensor.bytes, &next.fixed_bytes) ||
            !add(tensor.bytes, &next.fixed_tensor_bytes)) return false;
        auto& shard = next.shards[found->second.shard];
        shard.fixed.spans.push_back({tensor.begin, destination, tensor.bytes});
        ++shard.fixed.kept_tensors;
        ++next.fixed_tensors;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto& shard = next.shards[i];
        std::vector<qrt_resident_text_shard::Tensor> ordinary;
        ordinary.reserve(inputs[i].tensors.size());
        for (const auto& tensor : inputs[i].tensors) {
            const bool omitted = text_only && !qrt_resident_text_shard::keep(tensor.name);
            const bool fixed = selected.count(tensor.name) != 0;
            ordinary.push_back({tensor.begin, tensor.bytes, !omitted && !fixed});
            if (omitted) {
                ++shard.omitted_tensors;
                if (!add(tensor.bytes, &shard.omitted_tensor_bytes)) return false;
            }
        }
        // The existing planner validates every original range, including
        // optional and relocated tensors, before excluding any device bytes.
        if (!qrt_resident_text_shard::build(inputs[i].file_bytes,
                std::move(ordinary), true, &shard.ordinary) ||
            !add(shard.ordinary.device_bytes, &next.ordinary_bytes) ||
            !add(shard.omitted_tensors, &next.omitted_tensors) ||
            !add(shard.omitted_tensor_bytes, &next.omitted_tensor_bytes)) return false;
        auto& spans = shard.fixed.spans;
        std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) {
            return a.source < b.source;
        });
        std::vector<Span> merged;
        for (const auto& span : spans) {
            if (!merged.empty() &&
                merged.back().source + merged.back().bytes == span.source &&
                merged.back().destination + merged.back().bytes == span.destination) {
                merged.back().bytes += span.bytes;
            } else merged.push_back(span);
        }
        spans = std::move(merged);
        shard.fixed.device_bytes = next.fixed_bytes;
        shard.fixed.packed = true;
    }
    if (next.fixed_bytes > UINT64_MAX - next.ordinary_bytes) return false;
    *output = std::move(next);
    return true;
}
} // namespace qrt_resident_ordered_shard
