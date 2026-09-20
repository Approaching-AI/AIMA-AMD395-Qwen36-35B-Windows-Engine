#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include "sm121_mtp_runtime_tables.h"
#include "gdn/sm121_mtp_target_inputs.h"
#include "gdn/sm121_mtp_request.h"
#include "mtp_target_rows_trace.h"

namespace qrt_sm121_mtp_runtime {
namespace prefill_probe_detail {
struct Tensor { const char* name; const void* device; size_t bytes; };
inline std::array<Tensor, 25> tensors(const qrt_sm121_mtp::DrafterObservation& v,
    const uint16_t* cache, unsigned tokens) {
    using namespace qrt_sm121_mtp;
    MoeBuffers m{};
    if (v.rows != 1u || !bind_moe_buffers(const_cast<void*>(v.moe_workspace), v.moe_bytes, 1u, &m))
        return {};
    return {{{"query-projection", v.query_projection, 8192u*2u}, {"queries", v.queries, 4096u*2u},
        {"gates", v.gates, 4096u*2u}, {"context", v.context, 4096u*2u},
        {"gated", v.gated_context, 4096u*2u}, {"attention-output", v.output_projection, 2048u*2u},
        {"post-attention-norm", v.post_attention, 2048u*2u}, {"attention-residual", v.attention_residual, 2048u*2u},
        {"final-norm", v.final_hidden, 2048u*2u}, {"router", m.router, 256u*2u},
        {"shared-gate", m.shared_gate, 2u}, {"shared-gate-up", m.shared_gate_up, 1024u*2u},
        {"shared-activated", m.shared_activated, 512u*2u}, {"shared-down", m.shared_down, 2048u*2u},
        {"shared", m.shared, 2048u*2u}, {"routed-gate-up", m.routed_gate_up, 8192u*2u},
        {"routed-activated", m.routed_activated, 4096u*2u}, {"routed-weighted", m.routed_weighted, 16384u*2u},
        {"expert-part-1", m.routed, 2048u*2u}, {"moe-output", m.output, 2048u*2u},
        {"topk-ids", m.topk_ids, 8u*4u}, {"topk-weights", m.topk_weights, 8u*4u},
        {"final-residual", v.final_residual, 2048u*2u}, {"logits", v.vocabulary_logits, head_vocabulary*2u},
        {"history-kv", cache, size_t(tokens)*1024u*2u}}};
}
struct Download {
    unsigned char* host = nullptr;
    Download* quarantine_next = nullptr;
    ~Download() { if (host) (void)hipHostFree(host); }
};
inline std::atomic<Download*> quarantined_head{nullptr};
inline void quarantine(Download* storage) {
    Download* previous = quarantined_head.load(std::memory_order_relaxed);
    do { storage->quarantine_next = previous; }
    while (!quarantined_head.compare_exchange_weak(previous, storage,
        std::memory_order_release, std::memory_order_relaxed));
}
inline bool fresh(const std::string& path) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    return !error && !exists;
}
inline bool write(const std::string& path, const void* data, size_t bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return false;
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    file.close();
    return bool(file);
}

// Download only completed real device outputs. A failed observer fence must
// retain both its pinned destinations and every borrowed drafter allocation.
template<class Owner>
inline bool capture(Owner& drafter, const qrt_sm121_mtp::DraftStep& proposal,
    uint64_t epoch, const qrt_mtp_target_rows::PrefillRows& batch, const std::string& prefix,
    size_t model_pack_bytes, size_t input_bytes, uint64_t elapsed_ns,
    std::string& failure_stage, std::string& failure, hipStream_t stream = nullptr) {
    const auto fail = [&](const char* stage, const char* message) {
        failure_stage = stage; failure = message; return false;
    };
    const auto view = drafter.observation(epoch);
    if (proposal.status != hipSuccess || proposal.completion_unknown || proposal.rows != 1u ||
        !batch.published() || !batch.rows() || batch.first_position() || batch.discarded_prefill() ||
        drafter.retained_tokens() != batch.rows() || proposal.first_position != batch.rows()-1u ||
        view.rows != 1u || view.first_position != proposal.first_position || prefix.empty())
        return fail("mtp_prefill_capture_contract", "native MTP capture requires the completed full-prompt proposal");
    const auto records = tensors(view, drafter.cache_data(), drafter.retained_tokens());
    std::array<std::string, 25> paths;
    size_t total = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        if (!record.name || !record.device || !record.bytes || record.bytes > (16u<<20u))
            return fail("mtp_prefill_capture_tensor", "native MTP observation tensor is absent or outside its bounded extent");
        total += record.bytes;
        paths[i] = prefix + "." + record.name + ".bin";
        if (!fresh(paths[i])) return fail("mtp_prefill_capture_path", "native MTP capture path already exists or cannot be checked");
    }
    for (const char* suffix : {".json", ".target.json", ".target.hidden.bf16.bin", ".target.shifted.u32.bin"})
        if (!fresh(prefix + suffix)) return fail("mtp_prefill_capture_path", "native MTP metadata or target path already exists or cannot be checked");
    std::unique_ptr<Download> download(new (std::nothrow) Download);
    if (!download) return fail("mtp_prefill_capture_owner", "native MTP capture owner allocation failed");
    hipError_t status = hipHostMalloc(reinterpret_cast<void**>(&download->host), total);
    if (status != hipSuccess) return fail("mtp_prefill_capture_allocation", "native MTP capture pinned allocation failed");
    size_t offset = 0;
    for (const auto& record : records) {
        status = hipMemcpyAsync(download->host + offset, record.device, record.bytes, hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) break;
        offset += record.bytes;
    }
    const hipError_t completed = hipStreamSynchronize(stream);
    if (completed != hipSuccess) {
        (void)drafter.quarantine_borrower(completed);
        quarantine(download.release());
        return fail("mtp_prefill_capture_completion", "native MTP capture completion is unknown; observer and drafter storage retained");
    }
    if (status != hipSuccess) return fail("mtp_prefill_capture_copy", "native MTP capture copy failed after a completed drain");
    std::ostringstream metadata;
    metadata.imbue(std::locale::classic());
    metadata << std::setprecision(9)
        << "{\n  \"schema\": \"qrt-mtp-native-prefill-probe-v1\",\n"
        << "  \"source\": \"actual_target_hidden_original_resident_weights_native_drafter\",\n"
        << "  \"model_epoch\": " << epoch << ",\n  \"prompt_tokens\": " << batch.prompt_tokens()
        << ",\n  \"target_first_token\": " << batch.sampled_token()
        << ",\n  \"draft_position\": " << proposal.first_position
        << ",\n  \"draft_token\": " << proposal.tokens[0] << ",\n  \"draft_logit\": " << proposal.logits[0]
        << ",\n  \"retained_tokens\": " << drafter.retained_tokens()
        << ",\n  \"cache_layout\": \"token_major_K512_V512_bf16\",\n"
        << "  \"model_pack_bytes\": " << model_pack_bytes << ",\n  \"input_owned_bytes\": " << input_bytes
        << ",\n  \"drafter_owned_bytes\": " << drafter.allocated_bytes()
        << ",\n  \"compute_elapsed_ns\": " << elapsed_ns << ",\n  \"capture_bytes\": " << total
        << ",\n  \"native_drafter_executed\": true,\n  \"mtp_acceptance_enabled\": false,\n"
        << "  \"reference_data_used_by_compute\": false,\n  \"numerical_acceptance_claimed\": false,\n"
        << "  \"tensors\": [\n";
    offset = 0;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        if (!write(paths[i], download->host + offset, record.bytes))
            return fail("mtp_prefill_capture_write", "native MTP tensor write or close failed");
        offset += record.bytes;
        metadata << "    {\"name\": \"" << record.name << "\", \"bytes\": " << record.bytes << "}"
            << (i+1u == records.size() ? "\n" : ",\n");
    }
    metadata << "  ]\n}\n";
    if (!qrt_mtp_target_rows::write_trace(batch, prefix + ".target", &failure)) {
        failure_stage = "mtp_prefill_capture_target"; return false;
    }
    const std::string serialized = metadata.str();
    if (!write(prefix + ".json", serialized.data(), serialized.size()))
        return fail("mtp_prefill_capture_metadata", "native MTP metadata write or close failed");
    return true;
}
} // namespace prefill_probe_detail

// Diagnostic first proposal over the actual original product prompts. This
// does not perform speculative target verification or alter emitted tokens.
inline bool probe_prefill(const qrt_mtp_target_rows::PrefillRows& batch,
    std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source, const std::string& prefix,
    std::string& failure_stage, std::string& failure) {
    using namespace qrt_sm121_mtp;
    const auto start = std::chrono::steady_clock::now();
    const auto fail = [&](const char* stage, hipError_t status) {
        failure_stage = stage; failure = "native MTP prefill diagnostic status " + std::to_string(status); return false;
    };
    if (!source || !batch.published() || batch.first_position() || batch.discarded_prefill() ||
        (batch.rows() != 7169u && batch.rows() != 8192u) || prefix.empty())
        return fail("mtp_prefill_probe_contract", hipErrorInvalidValue);
    try {
        const uint64_t epoch = source->epoch();
        ModelWeights model;
        const auto prepared = model.prepare(std::move(source), epoch);
        if (prepared.status != hipSuccess) return fail(prepared.stage, prepared.status);
        DrafterTables tables;
        hipError_t status = prepare(&tables, static_cast<unsigned>(batch.rows()-1u));
        if (status != hipSuccess) return fail("mtp_prefill_probe_tables", status);
        const auto binding = model.binding(epoch);
        TargetInputs inputs;
        Drafter drafter;
        status = drafter.reserve(static_cast<unsigned>(batch.rows()), static_cast<unsigned>(batch.rows()));
        if (status != hipSuccess) return fail("mtp_prefill_probe_reserve", status);
        if (!drafter.bind(binding, tables, epoch)) return fail("mtp_prefill_probe_bind", hipErrorInvalidValue);
        const auto appended = inputs.append(drafter, batch, true, epoch);
        if (appended.status != hipSuccess) return fail(appended.stage, appended.status);
        const auto proposal = drafter.propose(static_cast<unsigned>(batch.rows()-1u), 1u, epoch);
        if (proposal.status != hipSuccess) return fail(proposal.stage, proposal.status);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        return prefill_probe_detail::capture(drafter, proposal, epoch, batch, prefix,
            binding.allocated_bytes(), inputs.allocated_bytes(), static_cast<uint64_t>(elapsed), failure_stage, failure);
    } catch (const std::exception& error) {
        failure_stage = "mtp_prefill_probe_exception"; failure = error.what(); return false;
    }
}

// Opt-in real-session seed. The caller supplies the actual committed target
// frontier under its session lock and publishes this checkpoint only if the
// enclosing prefill, native computation, copies and trace writes all succeed.
inline bool probe_prefill_request(const qrt_mtp_target_rows::PrefillRows& batch,
    std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source,
    const qrt_sm121_mtp::TargetFrontier& actual, unsigned capacity, const std::string& prefix,
    qrt_sm121_mtp::RequestCheckpoint* output, std::string& failure_stage, std::string& failure) {
    using namespace qrt_sm121_mtp;
    const auto start = std::chrono::steady_clock::now();
    const auto fail = [&](const char* stage, hipError_t status) {
        failure_stage = stage; failure = "native MTP request seed status " + std::to_string(status); return false;
    };
    if (!output || !source || actual.model_epoch != source->epoch() ||
        !actual.owner || !actual.generation || actual.processed_count != batch.rows() ||
        actual.current_token != batch.sampled_token() ||
        !batch.matches_input(actual.processed_inputs, actual.processed_count) ||
        capacity < batch.rows() || capacity > 262144u ||
        !batch.published() || batch.first_position() || batch.discarded_prefill() ||
        (batch.rows() != 7169u && batch.rows() != 8192u) || prefix.empty() ||
        !prefill_probe_detail::fresh(prefix+".request.json"))
        return fail("mtp_request_probe_contract", hipErrorInvalidValue);
    try {
        ModelWeights model;
        const auto prepared = model.prepare(std::move(source), actual.model_epoch);
        if (prepared.status != hipSuccess) return fail(prepared.stage, prepared.status);
        DrafterTables tables;
        const hipError_t status = prepare(&tables, capacity-1u);
        if (status != hipSuccess) return fail("mtp_request_probe_tables", status);
        const auto binding = model.binding(actual.model_epoch);
        Request request;
        const auto seeded = request.seed(batch, actual, binding, tables, capacity);
        if (seeded.status != hipSuccess) return fail(seeded.stage, seeded.status);
        const auto proposal = request.proposal(actual.model_epoch);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-start).count();
        if (!prefill_probe_detail::capture(request, proposal, actual.model_epoch, batch, prefix,
            binding.allocated_bytes(), request.input_allocated_bytes(), static_cast<uint64_t>(elapsed),
            failure_stage, failure)) return false;
        RequestCheckpoint checkpoint;
        const auto saved = request.save(&checkpoint, actual);
        if (saved.status != hipSuccess) return fail(saved.stage, saved.status);
        std::ostringstream metadata;
        metadata.imbue(std::locale::classic());
        metadata << std::setprecision(9)
            << "{\n  \"schema\": \"qrt-mtp-native-request-seed-v1\",\n"
            << "  \"model_epoch\": " << actual.model_epoch
            << ",\n  \"target_generation\": " << actual.generation
            << ",\n  \"processed_tokens\": " << checkpoint.tokens()
            << ",\n  \"target_current_token\": " << actual.current_token
            << ",\n  \"next_draft_token\": " << proposal.tokens[0]
            << ",\n  \"next_draft_logit\": " << proposal.logits[0]
            << ",\n  \"checkpoint_bytes\": " << checkpoint.allocated_bytes()
            << ",\n  \"actual_target_frontier_matched\": true,\n"
            << "  \"accepted_target_blocks\": 0,\n  \"mtp_acceptance_enabled\": false,\n"
            << "  \"reference_data_used_by_compute\": false,\n  \"numerical_acceptance_claimed\": false\n}\n";
        const std::string serialized = metadata.str();
        if (!prefill_probe_detail::write(prefix+".request.json", serialized.data(), serialized.size()))
            return fail("mtp_request_probe_metadata", hipErrorInvalidValue);
        *output = std::move(checkpoint);
        return true;
    } catch (const std::exception& error) {
        failure_stage = "mtp_request_probe_exception"; failure = error.what(); return false;
    }
}
} // namespace qrt_sm121_mtp_runtime
