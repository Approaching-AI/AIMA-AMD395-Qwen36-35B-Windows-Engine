#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qrt_resident_text_shard {
struct Tensor { uint64_t begin, bytes; bool keep; };
struct Span { uint64_t source, destination, bytes; };
struct Layout {
    uint64_t file_bytes = 0, device_bytes = 0, omitted_tensor_bytes = 0;
    uint64_t kept_tensors = 0, omitted_tensors = 0;
    bool packed = false;
    std::vector<Span> spans;
};

// This explicit text-only option excludes only the two named optional
// namespaces. Unknown tensor families remain resident. Disk metadata keeps
// its original offsets; only device storage is packed.
inline bool keep(const std::string& name) {
    return name.compare(0, 13, "model.visual.") != 0 &&
           name.compare(0, 4, "mtp.") != 0;
}

inline bool build(uint64_t file_bytes, std::vector<Tensor> tensors,
                  bool text_only, Layout* output) {
    if (!output || !file_bytes || tensors.empty()) return false;
    std::sort(tensors.begin(), tensors.end(), [](const Tensor& a, const Tensor& b) {
        return a.begin < b.begin;
    });
    Layout next;
    next.file_bytes = file_bytes;
    uint64_t end = 0;
    for (const auto& t : tensors) {
        if (!t.bytes || t.begin < end || t.begin > file_bytes ||
            t.bytes > file_bytes - t.begin) return false;
        end = t.begin + t.bytes;
        if (text_only && !t.keep) {
            ++next.omitted_tensors;
            next.omitted_tensor_bytes += t.bytes;
        } else ++next.kept_tensors;
    }
    next.packed = next.omitted_tensors != 0;
    if (!next.packed) {
        next.device_bytes = file_bytes;
        next.spans.push_back({0, 0, file_bytes});
    } else {
        for (const auto& t : tensors) if (t.keep) {
            if (!next.spans.empty() &&
                next.spans.back().source + next.spans.back().bytes == t.begin) {
                if (t.bytes > UINT64_MAX - next.device_bytes) return false;
                next.spans.back().bytes += t.bytes;
                next.device_bytes += t.bytes;
                continue;
            }
            constexpr uint64_t mask = 255;
            if (next.device_bytes > UINT64_MAX - mask) return false;
            const uint64_t destination = (next.device_bytes + mask) & ~mask;
            if (t.bytes > UINT64_MAX - destination) return false;
            next.spans.push_back({t.begin, destination, t.bytes});
            next.device_bytes = destination + t.bytes;
        }
    }
    *output = std::move(next);
    return true;
}

// A tensor/slice must fit one retained interval. Cross-hole and omitted
// requests fail without exposing any pointer into unrelated packed bytes.
inline bool offset(const Layout& layout, uint64_t source, uint64_t bytes,
                   uint64_t* destination) {
    if (!destination || !bytes || source > layout.file_bytes ||
        bytes > layout.file_bytes - source) return false;
    for (const auto& span : layout.spans) {
        if (source < span.source) break;
        if (source - span.source <= span.bytes &&
            bytes <= span.bytes - (source - span.source)) {
            *destination = span.destination + source - span.source;
            return true;
        }
    }
    return false;
}

// Feed the existing aligned disk ring into retained device spans. Source
// offsets below are relative to this ring buffer; destinations are relative
// to the compact shard. An empty intersection performs no copy.
template<class Copy>
bool copy(const Layout& layout, uint64_t first, uint64_t bytes, Copy&& invoke) {
    if (!bytes || first > layout.file_bytes || bytes > layout.file_bytes - first)
        return false;
    const uint64_t end = first + bytes;
    for (const auto& span : layout.spans) {
        const uint64_t begin = (std::max)(first, span.source);
        const uint64_t last = (std::min)(end, span.source + span.bytes);
        if (begin < last && !invoke(span.destination + begin - span.source,
                                   begin - first, last - begin)) return false;
    }
    return true;
}
} // namespace qrt_resident_text_shard
