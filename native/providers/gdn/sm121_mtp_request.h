#pragma once
#include <cmath>
#include "sm121_mtp_target_inputs.h"

namespace qrt_sm121_mtp {
// The caller supplies this from the actual target session while holding its
// transaction lock. Processed inputs exclude current_token, which is the
// target sample to feed next. Exact IDs, owner, generation and model epoch are
// all required; an engine self-hash cannot establish cache compatibility.
struct TargetFrontier {
    const void* owner = nullptr;
    uint64_t generation = 0, model_epoch = 0;
    const uint32_t* processed_inputs = nullptr;
    size_t processed_count = 0;
    uint32_t current_token = UINT32_MAX;
};
struct TargetBatch {
    uint64_t first_position = 0;
    unsigned rows = 0;
    std::array<uint32_t, 2> inputs{};
};
struct AcceptedTarget {
    unsigned rows = 0;
    std::array<uint32_t, 2> outputs{};
    bool drafter_retired = false;
};

namespace mtp_request_detail {
struct State {
    const void* owner = nullptr;
    uint64_t generation = 0, epoch = 0;
    std::vector<uint32_t> inputs;
    uint32_t current = UINT32_MAX;
    qrt_mtp_draft_schedule::Schedule schedule;
    DraftStep proposal;
    ModelWeightBinding binding;

    bool matches(const TargetFrontier& actual) const {
        return binding.valid(actual.model_epoch) && actual.owner == owner &&
            actual.generation == generation && actual.model_epoch == epoch &&
            actual.current_token == current && actual.processed_inputs &&
            actual.processed_count == inputs.size() &&
            std::equal(inputs.begin(), inputs.end(), actual.processed_inputs);
    }
    bool speculative() const {
        qrt_mtp_draft_schedule::Batch next;
        return schedule.peek(&next) && next.speculative;
    }
    bool proposal_ready() const {
        return speculative() && proposal.status == hipSuccess && !proposal.completion_unknown &&
            proposal.rows == 1u && size_t(proposal.first_position)+1u == inputs.size() &&
            proposal.tokens[0] < qrt_mtp_target_rows::vocabulary &&
            std::isfinite(proposal.logits[0]);
    }
};
struct Saved {
    State state;
    DrafterCheckpoint cache;
};
} // namespace mtp_request_detail

// Immutable value for the copyable target session and its shadow transaction.
// It can resume only the exact paired target frontier, not a shorter prefix.
class RequestCheckpoint final {
public:
    bool matches(const TargetFrontier& actual) const {
        return saved_ && saved_->cache.valid(actual.model_epoch) &&
            saved_->cache.tokens() == saved_->state.inputs.size() &&
            saved_->state.proposal_ready() && saved_->state.matches(actual);
    }
    size_t tokens() const { return saved_ ? saved_->state.inputs.size() : 0u; }
    size_t allocated_bytes() const { return saved_ ? saved_->cache.allocated_bytes() : 0u; }
private:
    friend class Request;
    std::shared_ptr<const mtp_request_detail::Saved> saved_;
};

// One serialized live MTP branch, paired with a caller-owned target cache
// transaction. begin -> actual target execution -> prepare -> target commit
// -> commit. On any error the caller rolls back the target and calls abort.
// Prepared output is private until BOTH commits succeed.
//
// The live branch appends at most two accepted rows per step. Full KV copies
// happen only at save/restore boundaries, never at every target verification.
// No oracle, reference schedule or reference hidden is accepted by this API.
class Request final {
public:
    Request() = default;
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;

    PromptStep seed(const qrt_mtp_target_rows::PrefillRows& batch, const TargetFrontier& actual,
        const ModelWeightBinding& binding, const DrafterTables& tables, unsigned capacity,
        hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u) {
        if (terminal_ != hipSuccess) return unavailable();
        if (live_ || !actual.owner || !actual.generation || !binding.valid(actual.model_epoch) ||
            !batch.published() || batch.first_position() || batch.discarded_prefill() ||
            batch.rows() >= qrt_mtp_draft_schedule::reference_drafter_limit ||
            actual.processed_count != batch.rows() || actual.current_token != batch.sampled_token() ||
            !batch.matches_input(actual.processed_inputs, actual.processed_count) ||
            capacity < batch.rows() || capacity > 262144u)
            return invalid("request_seed_contract");
        try {
            auto next = std::make_unique<Live>();
            auto& state = next->state;
            state.owner = actual.owner; state.generation = actual.generation; state.epoch = actual.model_epoch;
            state.inputs.reserve(size_t(capacity)+2u);
            state.inputs.assign(actual.processed_inputs, actual.processed_inputs+actual.processed_count);
            state.current = actual.current_token; state.binding = binding;
            if (!state.schedule.reset(actual.processed_count)) return invalid("request_seed_schedule");
            const auto reserved = next->drafter.reserve(capacity, (std::max)(2u, static_cast<unsigned>(batch.rows())));
            if (reserved != hipSuccess) return {reserved, "request_seed_reserve"};
            if (!next->drafter.bind(binding, tables, actual.model_epoch)) return invalid("request_seed_binding");
            const auto appended = next->inputs.append(next->drafter, batch, true, actual.model_epoch, stream, maximum_blocks);
            if (appended.status != hipSuccess) return failed(appended);
            state.proposal = next->drafter.propose(static_cast<unsigned>(batch.rows()-1u), 1u,
                actual.model_epoch, stream, maximum_blocks);
            if (state.proposal.status != hipSuccess) return failed(state.proposal);
            if (!state.matches(actual) || !state.proposal_ready()) return invalid("request_seed_frontier");
            live_ = std::move(next);
            return complete();
        } catch (...) { return {hipErrorOutOfMemory, "request_seed_owner"}; }
    }

    PromptStep restore(const RequestCheckpoint& checkpoint, const TargetFrontier& actual,
        unsigned capacity, hipStream_t stream = nullptr) {
        if (terminal_ != hipSuccess) return unavailable();
        if (live_ || !checkpoint.matches(actual) || capacity < actual.processed_count || capacity > 262144u)
            return invalid("request_restore_contract");
        try {
            auto next = std::make_unique<Live>();
            next->state = checkpoint.saved_->state;
            next->state.inputs.reserve(size_t(capacity)+2u);
            const auto restored = next->drafter.restore(checkpoint.saved_->cache, capacity, 2u, actual.model_epoch, stream);
            if (restored.status != hipSuccess) return failed(restored);
            if (!checkpoint.matches(actual)) return invalid("request_restore_frontier");
            // The already-completed next proposal belongs to this exact cache
            // and current target token; restored mutable scratch is not used.
            live_ = std::move(next);
            return complete();
        } catch (...) { return {hipErrorOutOfMemory, "request_restore_owner"}; }
    }

    bool begin(const TargetFrontier& actual, size_t remaining_outputs, TargetBatch* output) {
        if (!output || !remaining_outputs || !ready() || pending_.active ||
            !live_->state.matches(actual) || !live_->state.proposal_ready()) return false;
        Pending next;
        next.schedule = live_->state.schedule;
        qrt_mtp_draft_schedule::Batch scheduled;
        if (!next.schedule.begin(&scheduled) || !scheduled.speculative || scheduled.scheduled_rows != 2u ||
            scheduled.first_position != actual.processed_count) return false;
        const size_t maximum_accept = (std::min)(remaining_outputs, size_t(2u));
        // A crossing batch retires MTP for the following target batch, even
        // when rejection shortens the committed extent.
        if (scheduled.first_position+scheduled.scheduled_rows < 262144u &&
            actual.processed_count+maximum_accept > live_->drafter.capacity()) return false;
        next.batch = {scheduled.first_position, scheduled.scheduled_rows,
            {live_->state.current, live_->state.proposal.tokens[0]}};
        next.remaining = remaining_outputs; next.active = true;
        pending_ = next;
        *output = next.batch;
        return true;
    }

    PromptStep prepare(const uint32_t* actual_inputs, const uint32_t* samples, size_t rows,
        const std::vector<unsigned>& positions, const std::vector<float>& normalized,
        uint64_t epoch, AcceptedTarget* output, hipStream_t stream = nullptr,
        unsigned maximum_blocks = 1024u) {
        if (terminal_ != hipSuccess) return unavailable();
        if (!output || !ready() || !pending_.active || pending_.prepared ||
            epoch != live_->state.epoch || !live_->state.binding.valid(epoch) ||
            !actual_inputs || rows != pending_.batch.rows ||
            !std::equal(pending_.batch.inputs.begin(), pending_.batch.inputs.begin()+rows, actual_inputs))
            return invalid("request_target_contract");
        try {
            qrt_mtp_target_rows::DecodeRows accepted;
            if (!accepted.capture(pending_.batch.first_position, actual_inputs, samples, rows,
                pending_.remaining, positions, normalized)) return invalid("request_target_rows");
            Pending next = pending_;
            if (!next.schedule.complete(static_cast<unsigned>(accepted.rows())))
                return invalid("request_target_schedule");
            qrt_mtp_draft_schedule::Batch following;
            if (!next.schedule.peek(&following)) return invalid("request_following_schedule");
            next.accepted.rows = static_cast<unsigned>(accepted.rows());
            std::copy(accepted.shifted_tokens().begin(), accepted.shifted_tokens().end(), next.accepted.outputs.begin());
            next.accepted.drafter_retired = !following.speculative;
            if (following.speculative) {
                const auto appended = live_->inputs.append(live_->drafter, accepted, epoch, stream, maximum_blocks);
                if (appended.status != hipSuccess) return undo_failure(appended, epoch);
                next.proposal = live_->drafter.propose(static_cast<unsigned>(
                    accepted.first_position()+accepted.rows()-1u), 1u, epoch, stream, maximum_blocks);
                if (next.proposal.status != hipSuccess) return undo_failure(
                    {next.proposal.status, next.proposal.stage, live_->drafter.retained_tokens(),
                        next.proposal.completion_unknown}, epoch);
            }
            if (!live_->state.binding.valid(epoch))
                return undo_failure(invalid("request_target_epoch"), epoch);
            next.prepared = true;
            pending_ = next;
            *output = next.accepted;
            return complete();
        } catch (...) { return undo_failure({hipErrorOutOfMemory, "request_target_owner"}, epoch); }
    }

    // Receipt is reconstructed from the target AFTER its accepted rows are
    // committed within the enclosing rollback-capable transaction.
    bool commit(const TargetFrontier& receipt) {
        if (!ready() || !pending_.active || !pending_.prepared) return false;
        auto& state = live_->state;
        const size_t added = pending_.accepted.rows;
        if (!state.binding.valid(receipt.model_epoch) || receipt.model_epoch != state.epoch ||
            receipt.owner != state.owner || receipt.generation != state.generation ||
            !receipt.processed_inputs || receipt.processed_count != state.inputs.size()+added ||
            receipt.current_token != pending_.accepted.outputs[added-1u] ||
            !std::equal(state.inputs.begin(), state.inputs.end(), receipt.processed_inputs) ||
            !std::equal(pending_.batch.inputs.begin(), pending_.batch.inputs.begin()+added,
                receipt.processed_inputs+state.inputs.size())) return false;
        // Capacity was reserved before any GPU work; these value updates
        // cannot allocate or throw after target publication.
        if (state.inputs.size()+added > state.inputs.capacity()) return false;
        state.inputs.insert(state.inputs.end(), pending_.batch.inputs.begin(), pending_.batch.inputs.begin()+added);
        state.current = receipt.current_token;
        state.schedule = pending_.schedule; state.proposal = pending_.proposal;
        pending_ = {};
        return true;
    }

    bool abort(uint64_t epoch) {
        if (!ready() || epoch != live_->state.epoch || !live_->state.binding.valid(epoch)) return false;
        if (live_->drafter.retained_tokens() != live_->state.inputs.size() &&
            !live_->drafter.truncate(static_cast<unsigned>(live_->state.inputs.size()), epoch)) return false;
        pending_ = {};
        return true;
    }

    PromptStep save(RequestCheckpoint* output, const TargetFrontier& actual, hipStream_t stream = nullptr) {
        if (terminal_ != hipSuccess) return unavailable();
        if (!output || !ready() || pending_.active || !live_->state.matches(actual) ||
            !live_->state.proposal_ready() || live_->drafter.retained_tokens() != actual.processed_count)
            return invalid("request_save_contract");
        try {
            auto saved = std::make_shared<mtp_request_detail::Saved>();
            saved->state = live_->state;
            const auto copied = live_->drafter.checkpoint(&saved->cache, actual.model_epoch, stream);
            if (copied.status != hipSuccess) return failed(copied);
            RequestCheckpoint next; next.saved_ = std::move(saved);
            if (!next.matches(actual)) return invalid("request_save_frontier");
            *output = std::move(next);
            return complete();
        } catch (...) { return {hipErrorOutOfMemory, "request_save_owner"}; }
    }

    bool matches(const TargetFrontier& actual) const { return ready() && live_->state.matches(actual); }
    bool retired() const { return ready() && !pending_.active && !live_->state.speculative(); }
    bool quarantined() const { return completion_unknown_; }
    size_t committed_tokens() const { return live_ ? live_->state.inputs.size() : 0u; }
    size_t retained_tokens() const { return live_ ? live_->drafter.retained_tokens() : 0u; }
    bool pending() const { return pending_.active; }

private:
    struct Live {
        mtp_request_detail::State state;
        TargetInputs inputs;
        Drafter drafter;
    };
    struct Pending {
        TargetBatch batch;
        AcceptedTarget accepted;
        DraftStep proposal;
        qrt_mtp_draft_schedule::Schedule schedule;
        size_t remaining = 0;
        bool active = false, prepared = false;
    };
    bool ready() const { return live_ && terminal_ == hipSuccess; }
    PromptStep invalid(const char* stage) const {
        return {hipErrorInvalidValue, stage, static_cast<unsigned>(committed_tokens())};
    }
    PromptStep complete() const { return {hipSuccess, "complete", static_cast<unsigned>(committed_tokens())}; }
    PromptStep unavailable() const {
        return {terminal_, "request_unavailable", static_cast<unsigned>(committed_tokens()), completion_unknown_};
    }
    PromptStep failed(const PromptStep& failure) {
        if (failure.completion_unknown) { terminal_ = failure.status; completion_unknown_ = true; }
        return failure;
    }
    PromptStep failed(const DraftStep& failure) {
        return failed(PromptStep{failure.status, failure.stage, static_cast<unsigned>(committed_tokens()), failure.completion_unknown});
    }
    PromptStep undo_failure(const PromptStep& failure, uint64_t epoch) {
        if (failure.completion_unknown) return failed(failure);
        if (!abort(epoch)) terminal_ = failure.status;
        return failure;
    }
    std::unique_ptr<Live> live_;
    Pending pending_;
    hipError_t terminal_ = hipSuccess;
    bool completion_unknown_ = false;
};
} // namespace qrt_sm121_mtp
