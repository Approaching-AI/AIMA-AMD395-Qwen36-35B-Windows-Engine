#pragma once
#include <cstdint>
#include <limits>

namespace qrt_mtp_draft_schedule {
constexpr uint64_t reference_drafter_limit = 262144u;

struct Batch {
    uint64_t first_position = 0;
    unsigned int scheduled_rows = 0;
    bool speculative = false;
};

// The original batch-one MTP scheduler checks the entire scheduled target
// extent before rejected draft rows are discarded. Arithmetic switches only
// for the following batch. A caller must supply actual target/draft acceptance;
// this planner does not predict drafts or establish their numerical validity.
class Schedule {
public:
    bool reset(uint64_t prompt_tokens) {
        if (!prompt_tokens) return false;
        next_position_ = prompt_tokens;
        next_speculative_ = prompt_tokens < reference_drafter_limit;
        active_ = false;
        initialized_ = true;
        return true;
    }

    bool peek(Batch* output) const {
        if (!output || !initialized_ || active_) return false;
        const unsigned int rows = next_speculative_ ? 2u : 1u;
        if (next_position_ > (std::numeric_limits<uint64_t>::max)() - rows)
            return false;
        *output = {next_position_, rows, next_speculative_};
        return true;
    }

    bool begin(Batch* output) {
        if (!peek(output)) return false;
        current_ = *output;
        active_ = true;
        return true;
    }

    bool complete(unsigned int accepted_rows) {
        if (!active_ || !accepted_rows || accepted_rows > current_.scheduled_rows)
            return false;
        const uint64_t scheduled_end = current_.first_position + current_.scheduled_rows;
        next_position_ = current_.first_position + accepted_rows;
        // max_seq_len + one draft token must fit. The subtraction-free form
        // avoids overflow without substituting the shorter accepted extent.
        next_speculative_ = scheduled_end < reference_drafter_limit;
        active_ = false;
        return true;
    }

private:
    uint64_t next_position_ = 0;
    Batch current_{};
    bool next_speculative_ = false;
    bool active_ = false;
    bool initialized_ = false;
};
} // namespace qrt_mtp_draft_schedule
