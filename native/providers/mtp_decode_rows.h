#pragma once
#include <array>
#include <limits>
#include "mtp_target_rows.h"

namespace qrt_mtp_target_rows {
// Completed target execution inside its still-private cache transaction.
// Acceptance uses actual target samples. Only accepted input/hidden rows and
// their shifted output IDs survive; rejected or capacity-clipped rows do not.
// This does not publish the enclosing target or MTP transaction.
class DecodeRows final {
public:
    bool capture(uint64_t first, const uint32_t* inputs, const uint32_t* samples,
        size_t scheduled_rows, size_t remaining_outputs,
        const std::vector<unsigned>& positions, const std::vector<float>& normalized) {
        if (completed_ || !inputs || !samples || !remaining_outputs ||
            !scheduled_rows || scheduled_rows > 2u ||
            first > (std::numeric_limits<uint64_t>::max)()-scheduled_rows ||
            positions.size() != scheduled_rows || normalized.size() != scheduled_rows*hidden_width)
            return false;
        for (size_t i = 0; i < scheduled_rows; ++i)
            if (positions[i] != i || inputs[i] >= vocabulary || samples[i] >= vocabulary) return false;
        const size_t accepted = scheduled_rows == 2u && inputs[1] == samples[0] ? 2u : 1u;
        const size_t rows = (std::min)(accepted, remaining_outputs);
        std::vector<uint16_t> next(rows*hidden_width);
        for (size_t i = 0; i < next.size(); ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &normalized[i], sizeof(bits));
            if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return false;
            const uint16_t rounded = static_cast<uint16_t>(
                (bits+UINT32_C(0x7fff)+((bits>>16u)&1u))>>16u);
            if ((rounded & 0x7f80u) == 0x7f80u) return false;
            next[i] = rounded;
        }
        std::vector<uint32_t> shifted(samples, samples+rows);
        std::copy_n(inputs, rows, inputs_.begin());
        hidden_.swap(next); shifted_.swap(shifted);
        first_ = first; accepted_ = accepted; scheduled_ = scheduled_rows; completed_ = true;
        return true;
    }
    bool completed() const { return completed_; }
    size_t rows() const { return shifted_.size(); }
    uint64_t first_position() const { return first_; }
    size_t scheduled_rows() const { return scheduled_; }
    size_t accepted_before_capacity_clip() const { return accepted_; }
    const std::array<uint32_t, 2>& inputs() const { return inputs_; }
    const std::vector<uint16_t>& hidden() const { return hidden_; }
    const std::vector<uint32_t>& shifted_tokens() const { return shifted_; }
private:
    uint64_t first_ = 0;
    size_t scheduled_ = 0, accepted_ = 0;
    bool completed_ = false;
    std::array<uint32_t, 2> inputs_{};
    std::vector<uint16_t> hidden_;
    std::vector<uint32_t> shifted_;
};
} // namespace qrt_mtp_target_rows
