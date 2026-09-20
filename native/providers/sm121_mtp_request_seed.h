#pragma once
#include "gdn/sm121_mtp_request.h"
#include "sm121_mtp_runtime_tables.h"

namespace qrt_sm121_mtp_runtime {
// The cold target owns the transaction. This owner consumes completed chunks
// and publishes the paired checkpoint only after the final actual frontier.
class ChunkedPrefillSeed final {
public:
    qrt_sm121_mtp::PromptStep begin(const qrt_mtp_target_rows::PrefillRows& first,
        std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source,
        const qrt_sm121_mtp::TargetFrontier& actual,unsigned capacity) {
        if (started_ || !source || source->epoch()!=actual.model_epoch ||
            !first.published() || first.first_position() || !first.discarded_prefill() ||
            !actual.owner || !actual.generation || actual.processed_count!=first.rows() ||
            actual.current_token!=first.sampled_token() ||
            !first.matches_input(actual.processed_inputs,actual.processed_count) ||
            first.prompt_tokens()>=qrt_mtp_draft_schedule::reference_drafter_limit ||
            capacity<first.prompt_tokens() || capacity>262144u)
            return {hipErrorInvalidValue,"mtp_chunked_seed_contract"};
        started_=true;epoch_=actual.model_epoch;
        const auto prepared=model_.prepare(std::move(source),epoch_);
        if(prepared.status!=hipSuccess)return {prepared.status,prepared.stage,0u,prepared.completion_unknown};
        qrt_sm121_mtp::DrafterTables tables;
        const auto status=prepare(&tables,capacity-1u);
        if(status!=hipSuccess)return {status,"mtp_chunked_seed_tables"};
        return request_.seed_prefill_chunks(first,actual,model_.binding(epoch_),tables,capacity);
    }
    qrt_sm121_mtp::PromptStep append(const qrt_mtp_target_rows::PrefillRows& batch,
        const qrt_sm121_mtp::TargetFrontier& actual) {
        return request_.append_prefill_chunk(batch,actual);
    }
    qrt_sm121_mtp::PromptStep save(qrt_sm121_mtp::RequestCheckpoint* output,
        const qrt_sm121_mtp::TargetFrontier& actual) {
        return request_.save(output,actual);
    }
    uint64_t epoch()const{return epoch_;}
private:
    qrt_sm121_mtp::ModelWeights model_;
    qrt_sm121_mtp::Request request_;
    uint64_t epoch_=0;
    bool started_=false;
};

// Build the exact paired request checkpoint from the actual completed target
// rows. This step takes no diagnostic filename and writes no output files.
// Request::save publishes output only after every private producer completes.
inline qrt_sm121_mtp::PromptStep seed_prefill_request(
    const qrt_mtp_target_rows::PrefillRows& batch,
    std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source,
    const qrt_sm121_mtp::TargetFrontier& actual, unsigned capacity,
    qrt_sm121_mtp::RequestCheckpoint* output) {
    using namespace qrt_sm121_mtp;
    if (!output || !source || actual.model_epoch != source->epoch() ||
        !actual.owner || !actual.generation || !batch.published() ||
        batch.first_position() || batch.discarded_prefill() ||
        actual.processed_count != batch.rows() || actual.current_token != batch.sampled_token() ||
        !batch.matches_input(actual.processed_inputs,actual.processed_count) ||
        capacity < batch.rows() || capacity > 262144u)
        return {hipErrorInvalidValue,"mtp_request_seed_contract"};
    try {
        ModelWeights model;
        const auto prepared = model.prepare(std::move(source),actual.model_epoch);
        if (prepared.status != hipSuccess)
            return {prepared.status,prepared.stage,0u,prepared.completion_unknown};
        DrafterTables tables;
        const auto ready = prepare(&tables,capacity-1u);
        if (ready != hipSuccess) return {ready,"mtp_request_seed_tables"};
        Request request;
        const auto seeded = request.seed(batch,actual,model.binding(actual.model_epoch),tables,capacity);
        if (seeded.status != hipSuccess) return seeded;
        return request.save(output,actual);
    } catch (...) {
        return {hipErrorOutOfMemory,"mtp_request_seed_owner"};
    }
}
} // namespace qrt_sm121_mtp_runtime
