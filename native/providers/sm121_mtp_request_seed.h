#pragma once
#include "gdn/sm121_mtp_request.h"
#include "sm121_mtp_runtime_tables.h"

namespace qrt_sm121_mtp_runtime {
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
