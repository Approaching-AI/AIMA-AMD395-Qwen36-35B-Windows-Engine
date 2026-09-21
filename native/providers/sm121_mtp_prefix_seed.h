#pragma once
#include "gdn/sm121_mtp_request.h"

namespace qrt_sm121_mtp_runtime {
// One actual target suffix inside the caller's shadow transaction. Keep the
// original immutable checkpoint until the target rows, repaired MTP cache and
// final proposal all complete. The caller still owns target commit/rollback.
class PrefixPrefillSeed final {
public:
    PrefixPrefillSeed() = default;
    PrefixPrefillSeed(const PrefixPrefillSeed&) = delete;
    PrefixPrefillSeed& operator=(const PrefixPrefillSeed&) = delete;
    ~PrefixPrefillSeed() { end_capture(); }

    qrt_sm121_mtp::PromptStep begin(const qrt_sm121_mtp::RequestCheckpoint& checkpoint,
        std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source,
        const qrt_sm121_mtp::TargetFrontier& cached, const uint32_t* prompt,
        size_t tokens, unsigned capacity) {
        bool split = true;
        if (started_ || active_ || qrt_mtp_target_rows::Scope::active || !source ||
            source->epoch() != cached.model_epoch || !prompt ||
            !checkpoint.prefill_profile(cached,&split) || tokens <= cached.processed_count ||
            tokens-cached.processed_count != 1024u ||
            tokens >= qrt_mtp_draft_schedule::reference_drafter_limit ||
            capacity < tokens || capacity > 262144u ||
            !std::equal(prompt,prompt+cached.processed_count,cached.processed_inputs))
            return {hipErrorInvalidValue,"mtp_prefix_seed_contract"};
        try {
            auto rows = std::make_unique<qrt_mtp_target_rows::PrefillRows>(
                prompt,tokens,cached.processed_count,1024u);
            if (!rows->valid()) return {hipErrorInvalidValue,"mtp_prefix_seed_inputs"};
            std::vector<uint32_t> inputs(prompt,prompt+tokens);
            auto scope = std::make_unique<qrt_mtp_target_rows::Scope>(rows.get());
            checkpoint_ = checkpoint;
            source_ = std::move(source);
            inputs_.swap(inputs);
            cached_ = cached;
            cached_.processed_inputs = inputs_.data();
            rows_ = std::move(rows);
            scope_ = std::move(scope);
            capacity_ = capacity;
            split1024_ = split;
            started_ = true;
            active_ = this;
            return {hipSuccess,"mtp_prefix_seed_capture"};
        } catch (...) {
            return {hipErrorOutOfMemory,"mtp_prefix_seed_owner"};
        }
    }

    static bool captures(const void* engine, const uint32_t* input,
        size_t count, size_t first_position) {
        const auto* seed = active_;
        return seed && engine == seed->cached_.owner &&
            qrt_mtp_target_rows::Scope::active == seed->rows_.get() &&
            qrt_mtp_target_rows::Scope::requested(count) &&
            seed->rows_->first_position() == first_position &&
            seed->rows_->matches_input(input,count);
    }
    void end_capture() noexcept {
        if (active_ == this) active_ = nullptr;
        scope_.reset();
    }
    uint64_t epoch() const { return cached_.model_epoch; }

    qrt_sm121_mtp::PromptStep finish(const qrt_sm121_mtp::TargetFrontier& actual,
        qrt_sm121_mtp::RequestCheckpoint* output, std::vector<uint32_t>* processed_inputs,
        hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u) {
        end_capture();
        if (!started_ || finished_ || !output || !processed_inputs || !source_ ||
            source_->epoch() != actual.model_epoch || actual.model_epoch != cached_.model_epoch ||
            actual.owner != cached_.owner || actual.generation != cached_.generation ||
            !actual.processed_inputs || actual.processed_count != inputs_.size() ||
            !std::equal(inputs_.begin(),inputs_.end(),actual.processed_inputs) ||
            !rows_->published() || rows_->sampled_token() != actual.current_token)
            return {hipErrorInvalidValue,"mtp_prefix_seed_frontier"};
        finished_ = true;
        qrt_sm121_mtp::Request request;
        const auto extended = request.extend_prefill_prefix(checkpoint_,cached_,*rows_,actual,
            capacity_,split1024_,stream,maximum_blocks);
        if (extended.status != hipSuccess) return extended;
        qrt_sm121_mtp::RequestCheckpoint saved;
        const auto result = request.save(&saved,actual,stream);
        if (result.status != hipSuccess) return result;
        // Both swaps are nonthrowing. Neither caller-owned value changes on
        // any producer/validation failure, including unknown completion.
        *output = std::move(saved);
        processed_inputs->swap(inputs_);
        return result;
    }
private:
    inline static thread_local PrefixPrefillSeed* active_ = nullptr;
    qrt_sm121_mtp::RequestCheckpoint checkpoint_;
    std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source_;
    qrt_sm121_mtp::TargetFrontier cached_;
    std::vector<uint32_t> inputs_;
    std::unique_ptr<qrt_mtp_target_rows::PrefillRows> rows_;
    std::unique_ptr<qrt_mtp_target_rows::Scope> scope_;
    unsigned capacity_ = 0u;
    bool split1024_ = true, started_ = false, finished_ = false;
};
} // namespace qrt_sm121_mtp_runtime
