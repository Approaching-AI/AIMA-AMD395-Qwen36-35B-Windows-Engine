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
struct PrefillSeam {
    unsigned position = 0;
    uint32_t shifted_id = UINT32_MAX;
    std::array<uint16_t,qrt_mtp_target_rows::hidden_width> hidden{};
};
// Bounded actual target rows needed to reconstruct the original discarded
// chunk shifts when an aligned cold prefix gains another prompt suffix.
// Other chunk geometries and decode checkpoints are deliberately ineligible.
struct PrefillHistory {
    explicit PrefillHistory(bool split):split1024(split){seams.reserve(32u);}
    void append(const qrt_mtp_target_rows::PrefillRows& batch) noexcept {
        if(!eligible)return;
        if(batch.first_position()!=tokens || batch.rows()!=8192u || seams.size()>=32u){
            eligible=false;seams.clear();return;
        }
        PrefillSeam seam;
        seam.position=static_cast<unsigned>(batch.first_position()+batch.rows()-1u);
        seam.shifted_id=batch.shifted_tokens().back();
        std::copy_n(batch.hidden().end()-qrt_mtp_target_rows::hidden_width,
            qrt_mtp_target_rows::hidden_width,seam.hidden.begin());
        seams.push_back(seam);tokens+=batch.rows();
    }
    bool complete(size_t count)const{
        return eligible && count && tokens==count && seams.size()*8192u==count;
    }
    bool split1024 = true, eligible = true;
    size_t tokens = 0;
    std::vector<PrefillSeam> seams;
};
struct State {
    const void* owner = nullptr;
    uint64_t generation = 0, epoch = 0;
    std::vector<uint32_t> inputs;
    uint32_t current = UINT32_MAX;
    qrt_mtp_draft_schedule::Schedule schedule;
    DraftStep proposal;
    ModelWeightBinding binding;
    std::shared_ptr<const PrefillHistory> prefill;

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
        if (!saved_ || !saved_->state.matches(actual)) return false;
        if (!saved_->state.speculative()) return !saved_->cache.allocated_bytes();
        return saved_->cache.valid(actual.model_epoch) &&
            saved_->cache.tokens() == saved_->state.inputs.size() && saved_->state.proposal_ready();
    }
    bool retired(const TargetFrontier& actual) const {
        return matches(actual) && !saved_->state.speculative();
    }
    // After the crossing target transaction, the drafter has no further
    // cache or proposal. Preserve an immutable receipt of the actual one-row
    // target continuation, including the schedule that caused retirement.
    // The caller constructs both frontiers from its rollback-owned target.
    PromptStep advance_retired(const TargetFrontier& before,const TargetFrontier& after,
        RequestCheckpoint* output) const {
        if (!output || !retired(before) || after.owner!=before.owner ||
            after.generation!=before.generation || after.model_epoch!=before.model_epoch ||
            !after.processed_inputs || after.processed_count<=before.processed_count ||
            after.processed_count-before.processed_count>512u ||
            after.current_token>=qrt_mtp_target_rows::vocabulary ||
            !std::equal(saved_->state.inputs.begin(),saved_->state.inputs.end(),after.processed_inputs) ||
            after.processed_inputs[before.processed_count]!=before.current_token)
            return {hipErrorInvalidValue,"request_retired_frontier"};
        for(size_t i=before.processed_count;i<after.processed_count;++i)
            if(after.processed_inputs[i]>=qrt_mtp_target_rows::vocabulary)
                return {hipErrorInvalidValue,"request_retired_input"};
        try {
            auto saved=std::make_shared<mtp_request_detail::Saved>();
            saved->state=saved_->state;
            for(size_t i=before.processed_count;i<after.processed_count;++i){
                qrt_mtp_draft_schedule::Batch batch;
                if(!saved->state.schedule.begin(&batch) || batch.speculative ||
                    batch.scheduled_rows!=1u || batch.first_position!=i || !saved->state.schedule.complete(1u))
                    return {hipErrorInvalidValue,"request_retired_schedule"};
            }
            saved->state.inputs.assign(after.processed_inputs,after.processed_inputs+after.processed_count);
            saved->state.current=after.current_token;
            RequestCheckpoint next;next.saved_=std::move(saved);
            if(!next.retired(after))return {hipErrorInvalidValue,"request_retired_epoch"};
            *output=std::move(next);
            return {hipSuccess,"complete",static_cast<unsigned>(after.processed_count)};
        }catch(...){return {hipErrorOutOfMemory,"request_retired_owner"};}
    }
    size_t tokens() const { return saved_ ? saved_->state.inputs.size() : 0u; }
    size_t allocated_bytes() const { return saved_ ? saved_->cache.allocated_bytes() : 0u; }
    size_t model_pack_bytes() const { return saved_ ? saved_->state.binding.allocated_bytes() : 0u; }
    // Read the actual retained producer profile only after matching the
    // independent target frontier. Decode and partial-prefill checkpoints
    // have no complete cold history and cannot repair a new prompt suffix.
    bool prefill_profile(const TargetFrontier& actual, bool* split1024) const {
        if (!split1024 || !matches(actual) || !saved_->state.prefill ||
            !saved_->state.prefill->complete(actual.processed_count)) return false;
        *split1024 = saved_->state.prefill->split1024;
        return true;
    }
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

    // Cold multi-chunk prefill owns the complete real prompt identity. Each
    // chunk comes from the completed target, including its original discarded
    // chunk shift. No proposal, decode or checkpoint is visible until the last
    // chunk completes. The caller holds the actual target transaction lock.
    PromptStep seed_prefill_chunks(const qrt_mtp_target_rows::PrefillRows& first,
        const TargetFrontier& actual, const ModelWeightBinding& binding,
        const DrafterTables& tables, unsigned capacity,
        hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u,
        bool split1024_pre_fc_norm = true) {
        if (terminal_ != hipSuccess) return unavailable();
        if (live_ || !actual.owner || !actual.generation || !binding.valid(actual.model_epoch) ||
            !first.published() || first.first_position() || !first.discarded_prefill() ||
            first.prompt_tokens() >= qrt_mtp_draft_schedule::reference_drafter_limit ||
            actual.processed_count != first.rows() || actual.current_token != first.sampled_token() ||
            !first.matches_input(actual.processed_inputs,actual.processed_count) ||
            capacity < first.prompt_tokens() || capacity > 262144u)
            return invalid("request_chunk_seed_contract");
        try {
            auto next = std::make_unique<Live>();
            auto prompt = first.prompt();
            next->prefill=std::make_shared<mtp_request_detail::PrefillHistory>(split1024_pre_fc_norm);
            auto& state = next->state;
            state.owner = actual.owner; state.generation = actual.generation;
            state.epoch = actual.model_epoch; state.binding = binding;
            state.inputs.reserve(size_t(capacity)+2u);
            const auto reserved = next->drafter.reserve(capacity, static_cast<unsigned>(
                (std::min)(first.prompt_tokens(),qrt_mtp_target_rows::maximum_batch_rows)));
            if (reserved != hipSuccess) return {reserved,"request_chunk_seed_reserve"};
            if (!next->drafter.bind(binding,tables,actual.model_epoch))
                return invalid("request_chunk_seed_binding");
            live_ = std::move(next);
            prefill_prompt_.swap(prompt); prefill_pending_ = true;
            prefill_split1024_ = split1024_pre_fc_norm;
            return append_prefill_chunk(first,actual,stream,maximum_blocks);
        } catch (...) { return {hipErrorOutOfMemory,"request_chunk_seed_owner"}; }
    }

    PromptStep append_prefill_chunk(const qrt_mtp_target_rows::PrefillRows& batch,
        const TargetFrontier& actual, hipStream_t stream = nullptr,
        unsigned maximum_blocks = 1024u) {
        if (terminal_ != hipSuccess) return unavailable();
        if (!live_ || !prefill_pending_ || pending_.active || !batch.published() ||
            batch.prompt() != prefill_prompt_ || batch.first_position() != live_->state.inputs.size() ||
            !actual.processed_inputs || actual.processed_count != batch.first_position()+batch.rows() ||
            actual.processed_count > prefill_prompt_.size() || actual.current_token != batch.sampled_token() ||
            actual.owner != live_->state.owner || actual.generation != live_->state.generation ||
            actual.model_epoch != live_->state.epoch || !live_->state.binding.valid(actual.model_epoch) ||
            !std::equal(prefill_prompt_.begin(),prefill_prompt_.begin()+actual.processed_count,actual.processed_inputs))
            return invalid("request_chunk_frontier");
        auto& state = live_->state;
        const auto undo = [&](const PromptStep& failure) {
            if (failure.completion_unknown) return failed(failure);
            if (!live_->drafter.truncate(static_cast<unsigned>(state.inputs.size()),state.epoch))
                terminal_ = failure.status;
            return failure;
        };
        const auto appended = live_->inputs.append(live_->drafter,batch,prefill_split1024_,state.epoch,stream,maximum_blocks);
        if (appended.status != hipSuccess) return undo(appended);
        DraftStep proposal;
        if (!batch.discarded_prefill()) {
            proposal = live_->drafter.propose(static_cast<unsigned>(actual.processed_count-1u),1u,
                state.epoch,stream,maximum_blocks);
            if (proposal.status != hipSuccess)
                return undo({proposal.status,proposal.stage,static_cast<unsigned>(state.inputs.size()),proposal.completion_unknown});
        }
        if (!state.binding.valid(actual.model_epoch)) return undo(invalid("request_chunk_epoch"));
        // The full prompt capacity was reserved before the first submission.
        state.inputs.insert(state.inputs.end(),actual.processed_inputs+state.inputs.size(),
            actual.processed_inputs+actual.processed_count);
        state.current = actual.current_token;
        live_->prefill->append(batch);
        if (!batch.discarded_prefill()) {
            state.schedule.reset(actual.processed_count); state.proposal = proposal;
            if(live_->prefill->complete(actual.processed_count))state.prefill=live_->prefill;
            live_->prefill.reset();
            prefill_pending_ = false; prefill_prompt_.clear();
            if (!state.matches(actual) || !state.proposal_ready()) {
                terminal_ = hipErrorInvalidValue;
                return invalid("request_chunk_final_frontier");
            }
        }
        return complete();
    }

    PromptStep seed(const qrt_mtp_target_rows::PrefillRows& batch, const TargetFrontier& actual,
        const ModelWeightBinding& binding, const DrafterTables& tables, unsigned capacity,
        hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u,
        bool split1024_pre_fc_norm = true) {
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
            auto history=std::make_shared<mtp_request_detail::PrefillHistory>(split1024_pre_fc_norm);
            auto& state = next->state;
            state.owner = actual.owner; state.generation = actual.generation; state.epoch = actual.model_epoch;
            state.inputs.reserve(size_t(capacity)+2u);
            state.inputs.assign(actual.processed_inputs, actual.processed_inputs+actual.processed_count);
            state.current = actual.current_token; state.binding = binding;
            if (!state.schedule.reset(actual.processed_count)) return invalid("request_seed_schedule");
            const auto reserved = next->drafter.reserve(capacity, (std::max)(2u, static_cast<unsigned>(batch.rows())));
            if (reserved != hipSuccess) return {reserved, "request_seed_reserve"};
            if (!next->drafter.bind(binding, tables, actual.model_epoch)) return invalid("request_seed_binding");
            const auto appended = next->inputs.append(next->drafter, batch, split1024_pre_fc_norm, actual.model_epoch, stream, maximum_blocks);
            if (appended.status != hipSuccess) return failed(appended);
            state.proposal = next->drafter.propose(static_cast<unsigned>(batch.rows()-1u), 1u,
                actual.model_epoch, stream, maximum_blocks);
            if (state.proposal.status != hipSuccess) return failed(state.proposal);
            if (!state.matches(actual) || !state.proposal_ready()) return invalid("request_seed_frontier");
            history->append(batch);
            if(history->complete(actual.processed_count))state.prefill=std::move(history);
            live_ = std::move(next);
            return complete();
        } catch (...) { return {hipErrorOutOfMemory, "request_seed_owner"}; }
    }

    // Fork a completed, 8192-aligned cold prefix. The actual target owner must
    // have restored that exact prefix before producing first_suffix. Rebuild
    // each changed discarded-chunk tail on a private copy, then consume only
    // actual new target rows. There is no proposal/checkpoint between repairs.
    // A profile change needs a complete MTP reseed and cannot use this path.
    PromptStep extend_prefill_prefix(const RequestCheckpoint& checkpoint,
        const TargetFrontier& cached, const qrt_mtp_target_rows::PrefillRows& first_suffix,
        const TargetFrontier& actual, unsigned capacity, bool split1024_pre_fc_norm,
        hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u) {
        if(terminal_!=hipSuccess)return unavailable();
        if(live_ || !checkpoint.matches(cached) || !checkpoint.saved_->state.prefill ||
            !checkpoint.saved_->state.prefill->complete(cached.processed_count) ||
            checkpoint.saved_->state.prefill->split1024!=split1024_pre_fc_norm ||
            !first_suffix.published() || first_suffix.first_position()!=cached.processed_count ||
            first_suffix.prompt_tokens()>=qrt_mtp_draft_schedule::reference_drafter_limit ||
            !actual.processed_inputs || actual.owner!=cached.owner ||
            actual.generation!=cached.generation || actual.model_epoch!=cached.model_epoch ||
            actual.processed_count!=first_suffix.first_position()+first_suffix.rows() ||
            actual.current_token!=first_suffix.sampled_token() ||
            !std::equal(checkpoint.saved_->state.inputs.begin(),checkpoint.saved_->state.inputs.end(),
                first_suffix.prompt().begin()) ||
            !std::equal(first_suffix.prompt().begin(),first_suffix.prompt().begin()+actual.processed_count,
                actual.processed_inputs) || capacity<first_suffix.prompt_tokens() || capacity>262144u ||
            !maximum_blocks || maximum_blocks>4096u)
            return invalid("request_prefix_seed_contract");
        try {
            auto next=std::make_unique<Live>();
            next->state=checkpoint.saved_->state;
            next->state.inputs.reserve(size_t(capacity)+2u);
            next->state.prefill.reset();next->state.proposal={};
            auto prompt=first_suffix.prompt();
            next->prefill=std::make_shared<mtp_request_detail::PrefillHistory>(
                *checkpoint.saved_->state.prefill);
            next->prefill->seams.reserve(32u);
            const auto restored=next->drafter.restore(checkpoint.saved_->cache,capacity,
                static_cast<unsigned>((std::min)(size_t(8192u),prompt.size()-cached.processed_count)),
                actual.model_epoch,stream);
            if(restored.status!=hipSuccess)return failed(restored);
            for(auto& seam:next->prefill->seams){
                if(seam.shifted_id==prompt.back())continue;
                const auto repaired=next->inputs.replace_prefix_row(next->drafter,seam.hidden,
                    prompt.back(),seam.position,split1024_pre_fc_norm,actual.model_epoch,stream,maximum_blocks);
                if(repaired.status!=hipSuccess)return failed(repaired);
                seam.shifted_id=prompt.back();
            }
            if(!checkpoint.matches(cached))return invalid("request_prefix_seed_epoch");
            live_=std::move(next);prefill_prompt_.swap(prompt);
            prefill_pending_=true;prefill_split1024_=split1024_pre_fc_norm;
            return append_prefill_chunk(first_suffix,actual,stream,maximum_blocks);
        }catch(...){return {hipErrorOutOfMemory,"request_prefix_seed_owner"};}
    }

    PromptStep restore(const RequestCheckpoint& checkpoint, const TargetFrontier& actual,
        unsigned capacity, hipStream_t stream = nullptr) {
        if (terminal_ != hipSuccess) return unavailable();
        if (live_ || !checkpoint.matches(actual) || checkpoint.retired(actual) ||
            capacity < actual.processed_count || capacity > 262144u)
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
        state.prefill.reset();
        pending_ = {};
        return true;
    }

    bool abort(uint64_t epoch) {
        if (!ready() || epoch != live_->state.epoch || !live_->state.binding.valid(epoch)) return false;
        if (live_->state.speculative() && live_->drafter.retained_tokens() != live_->state.inputs.size() &&
            !live_->drafter.truncate(static_cast<unsigned>(live_->state.inputs.size()), epoch)) return false;
        pending_ = {};
        return true;
    }

    PromptStep save(RequestCheckpoint* output, const TargetFrontier& actual, hipStream_t stream = nullptr) {
        if (terminal_ != hipSuccess) return unavailable();
        if (!output || !ready() || pending_.active || !live_->state.matches(actual) ||
            (live_->state.speculative() &&
             (!live_->state.proposal_ready() || live_->drafter.retained_tokens() != actual.processed_count)))
            return invalid("request_save_contract");
        try {
            auto saved = std::make_shared<mtp_request_detail::Saved>();
            saved->state = live_->state;
            if(live_->state.speculative()){
                const auto copied = live_->drafter.checkpoint(&saved->cache, actual.model_epoch, stream);
                if (copied.status != hipSuccess) return failed(copied);
            }
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

    // Read-only diagnostics over the completed published branch. Observers
    // must quarantine this owner if their asynchronous borrow cannot finish.
    DraftStep proposal(uint64_t epoch) const {
        return ready() && !pending_.active && live_->state.binding.valid(epoch) &&
            live_->state.proposal_ready() ? live_->state.proposal : DraftStep{};
    }
    DrafterObservation observation(uint64_t epoch) const {
        return ready() && !pending_.active ? live_->drafter.observation(epoch) : DrafterObservation{};
    }
    const uint16_t* cache_data() const {
        return ready() && !pending_.active ? live_->drafter.cache_data() : nullptr;
    }
    size_t allocated_bytes() const { return live_ ? live_->drafter.allocated_bytes() : 0u; }
    size_t input_allocated_bytes() const { return live_ ? live_->inputs.allocated_bytes() : 0u; }
    bool quarantine_borrower(hipError_t status) {
        if (status == hipSuccess || !live_) return false;
        (void)live_->drafter.quarantine_borrower(status);
        terminal_ = status;
        completion_unknown_ = true;
        return true;
    }

private:
    struct Live {
        mtp_request_detail::State state;
        TargetInputs inputs;
        Drafter drafter;
        std::shared_ptr<mtp_request_detail::PrefillHistory> prefill;
    };
    struct Pending {
        TargetBatch batch;
        AcceptedTarget accepted;
        DraftStep proposal;
        qrt_mtp_draft_schedule::Schedule schedule;
        size_t remaining = 0;
        bool active = false, prepared = false;
    };
    bool ready() const { return live_ && !prefill_pending_ && terminal_ == hipSuccess; }
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
    std::vector<uint32_t> prefill_prompt_;
    bool prefill_pending_ = false;
    // A cold request keeps one declared pre-FC reduction order across chunks.
    // Both original GB10 launchers are supported; decode uses its own order.
    bool prefill_split1024_ = true;
    Pending pending_;
    hipError_t terminal_ = hipSuccess;
    bool completion_unknown_ = false;
};
} // namespace qrt_sm121_mtp
