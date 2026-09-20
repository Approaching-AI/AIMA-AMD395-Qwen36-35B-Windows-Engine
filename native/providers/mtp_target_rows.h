#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include "mtp_prompt_inputs.h"

namespace qrt_mtp_target_rows {
constexpr size_t hidden_width = 2048u;
constexpr size_t maximum_batch_rows = 8192u;
constexpr uint32_t vocabulary = 248320u;

// Retain one actual target prefill batch until the caller has consumed it.
// Positions passed to the prefill provider are local to that batch. The MTP
// cache instead appends at first_position(), within the complete real prompt.
// This object owns its token identity and never borrows a request's storage.
class PrefillRows final {
public:
    PrefillRows(const uint32_t *prompt, size_t prompt_tokens,
                size_t first_position, size_t rows) {
        if (!prompt || !prompt_tokens ||
            prompt_tokens > qrt_mtp_draft_schedule::reference_drafter_limit ||
            !rows || rows > maximum_batch_rows || first_position >= prompt_tokens ||
            rows > prompt_tokens - first_position) return;
        for (size_t i = 0; i < prompt_tokens; ++i)
            if (prompt[i] >= vocabulary) return;
        prompt_.assign(prompt, prompt + prompt_tokens);
        local_rows_.resize(rows);
        std::iota(local_rows_.begin(), local_rows_.end(), 0u);
        first_position_ = first_position;
    }
    PrefillRows(const PrefillRows &) = delete;
    PrefillRows &operator=(const PrefillRows &) = delete;

    bool valid() const { return !local_rows_.empty(); }
    bool staged() const { return staged_; }
    bool published() const { return published_; }
    size_t rows() const { return local_rows_.size(); }
    size_t first_position() const { return first_position_; }
    size_t prompt_tokens() const { return prompt_.size(); }
    const std::vector<uint32_t> &prompt() const { return prompt_; }
    bool discarded_prefill() const { return first_position_ + rows() < prompt_.size(); }
    uint32_t sampled_token() const { return published_ ? sampled_token_ : UINT32_MAX; }
    const std::vector<unsigned int> &local_rows() const { return local_rows_; }
    const std::vector<uint16_t> &hidden() const { return published_ ? hidden_ : empty_hidden_; }
    const std::vector<uint32_t> &shifted_tokens() const { return published_ ? shifted_tokens_ : empty_tokens_; }

    bool matches_input(const uint32_t *input, size_t count) const {
        return valid() && input && count == rows() &&
            std::equal(prompt_.begin() + first_position_,
                       prompt_.begin() + first_position_ + rows(), input);
    }

    // Call only after the real target normalization and sampler have completed.
    // Failure never exposes a partially converted batch. The sampled token
    // is retained even for discarded chunks, whose shifted tail intentionally
    // uses the complete request's last known prompt token instead.
    bool stage(const std::vector<unsigned int> &positions,
               const std::vector<float> &normalized, uint32_t sampled_token) {
        if (!valid() || staged_ || published_ || positions != local_rows_ ||
            normalized.size() != rows() * hidden_width || sampled_token >= vocabulary)
            return false;
        std::vector<uint16_t> next_hidden(normalized.size());
        for (size_t i = 0; i < normalized.size(); ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &normalized[i], sizeof(bits));
            if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return false;
            const uint16_t rounded = static_cast<uint16_t>(
                (bits + UINT32_C(0x7fff) + ((bits >> 16u) & 1u)) >> 16u);
            if ((rounded & 0x7f80u) == 0x7f80u) return false;
            next_hidden[i] = rounded;
        }
        std::vector<uint32_t> next_tokens(rows());
        if (!qrt_mtp_prompt_inputs::shift(prompt_.data(), prompt_.size(), first_position_,
                rows(), discarded_prefill(), sampled_token, next_tokens.data())) return false;
        hidden_.swap(next_hidden);
        shifted_tokens_.swap(next_tokens);
        sampled_token_ = sampled_token;
        staged_ = true;
        return true;
    }

    // Only the successful enclosing provider call may publish the staged
    // batch, and only if its actual returned sample still matches. Later
    // verification or request failures leave hidden()/shifted_tokens() empty.
    bool publish(uint32_t returned_token) {
        if (!staged_ || published_ || returned_token != sampled_token_) return false;
        published_ = true;
        return true;
    }
    void discard() noexcept {
        published_ = staged_ = false;
        hidden_.clear();
        shifted_tokens_.clear();
        local_rows_.clear();
        prompt_.clear();
        first_position_ = 0u;
        sampled_token_ = UINT32_MAX;
    }

private:
    std::vector<uint32_t> prompt_;
    std::vector<unsigned int> local_rows_;
    std::vector<uint16_t> hidden_;
    std::vector<uint32_t> shifted_tokens_;
    const std::vector<uint16_t> empty_hidden_;
    const std::vector<uint32_t> empty_tokens_;
    size_t first_position_ = 0;
    uint32_t sampled_token_ = UINT32_MAX;
    bool staged_ = false;
    bool published_ = false;
};

// A caller owns the batch and scopes exactly one synchronous provider call.
// A nested scope can suspend capture with nullptr; destruction restores its
// parent, including when the provider returns early or throws.
class Scope final {
public:
    inline static thread_local PrefillRows *active = nullptr;
    explicit Scope(PrefillRows *batch) : prior_(active) { active = batch; }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { active = prior_; }
    static bool requested(size_t tokens) {
        return active && active->valid() && !active->staged() &&
            !active->published() && active->rows() == tokens;
    }
private:
    PrefillRows *prior_;
};

// The provider constructs this after validating its capture scope. An early
// return or exception retracts even a staged/published batch, so a failed
// enclosing request cannot leave consumable target hidden behind.
class Publication final {
public:
    explicit Publication(PrefillRows *batch) : batch_(batch) {}
    Publication(const Publication &) = delete;
    Publication &operator=(const Publication &) = delete;
    ~Publication() { if (batch_ && !complete_) batch_->discard(); }
    void complete() { complete_ = true; }
private:
    PrefillRows *batch_;
    bool complete_ = false;
};

// The target needs all normalized rows for MTP KV construction, while its LM
// head needs only the original caller-selected rows. Preserve their order and
// duplicates, including prefix teacher rows. No additional projection is run.
inline bool select_head_rows(const std::vector<unsigned int> &source_positions,
                             const std::vector<float> &source,
                             const std::vector<unsigned int> &selected,
                             std::vector<float> *output) {
    if (!output || source_positions.empty() || source_positions.size() > maximum_batch_rows ||
        source.size() != source_positions.size() * hidden_width ||
        selected.empty() || selected.size() > maximum_batch_rows) return false;
    for (size_t i = 0; i < source_positions.size(); ++i)
        if (source_positions[i] != i) return false;
    for (unsigned int row : selected)
        if (row >= source_positions.size()) return false;
    std::vector<float> result(selected.size() * hidden_width);
    for (size_t i = 0; i < selected.size(); ++i)
        std::copy_n(source.data() + size_t(selected[i]) * hidden_width, hidden_width,
                    result.data() + i * hidden_width);
    output->swap(result);
    return true;
}
} // namespace qrt_mtp_target_rows
