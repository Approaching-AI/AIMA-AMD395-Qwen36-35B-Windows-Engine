#pragma once
#include <array>
#include <memory>
#include <new>
#include "sm121_mtp_prompt_cache.h"
#include "sm121_mtp_projection.h"
#include "sm121_mtp_query.h"
#include "sm121_mtp_attention.h"
#include "sm121_mtp_gate.h"
#include "sm121_mtp_residual.h"
#include "sm121_mtp_moe.h"
#include "sm121_mtp_head.h"
#include "sm121_mtp_model_weights.h"

namespace qrt_sm121_mtp {
struct DrafterWeights {
    PromptWeights prompt;
    const uint16_t* query = nullptr; // [8192,2048], interleaved Q256/gate256 per head.
    const uint16_t* query_norm = nullptr;
    const uint16_t* output = nullptr; // [2048,4096].
    const uint16_t* post_attention_norm = nullptr;
    MoeWeights moe;
    const uint16_t* final_norm = nullptr;
    const uint16_t* lm_head = nullptr;
};
struct DrafterTables {
    const unsigned char* rsqrt = nullptr;
    const uint16_t* rope = nullptr;
    unsigned rope_rows = 0;
    const unsigned char* exp2 = nullptr;
    const unsigned char* reciprocal = nullptr;
    MoeTables moe;
};
struct DraftStep {
    hipError_t status = hipSuccess;
    const char* stage = "complete";
    bool completion_unknown = false;
    unsigned first_position = 0;
    unsigned rows = 0; // Nonzero only after all producers and host copies complete.
    std::array<uint32_t, 2> tokens{};
    std::array<float, 2> logits{};
};
struct DrafterObservation {
    uint64_t generation = 0;
    unsigned first_position = 0, rows = 0;
    const uint16_t* query_projection = nullptr;
    const uint16_t* queries = nullptr;
    const uint16_t* gates = nullptr;
    const uint16_t* context = nullptr;
    const uint16_t* gated_context = nullptr;
    const uint16_t* output_projection = nullptr;
    const uint16_t* post_attention = nullptr;
    const uint16_t* attention_residual = nullptr;
    const void* moe_workspace = nullptr;
    size_t moe_bytes = 0;
    const uint16_t* final_hidden = nullptr;
    const uint16_t* final_residual = nullptr;
    const uint16_t* vocabulary_logits = nullptr;
};

// Synchronous request owner over asynchronous kernels. Input hidden rows must
// come from the actual target's final normalization. Target acceptance controls
// truncate; this class never consumes a reference acceptance schedule.
class Drafter {
public:
    Drafter() = default;
    Drafter(const Drafter&) = delete;
    Drafter& operator=(const Drafter&) = delete;
    ~Drafter() {
        // A proposal can still borrow the completed prompt cache when its
        // fence fails. Quarantine the cache storage as well as every GPU and
        // pinned-host allocation until process teardown establishes recovery.
        if (quarantined_) return;
        for (void* pointer : storage_.owned) if (pointer) (void)hipFree(pointer);
        if (storage_.host) (void)hipHostFree(storage_.host);
    }

    hipError_t reserve(unsigned tokens, unsigned append_rows = 8192u) {
        if (quarantined_) return completion_error_;
        if (!tokens || tokens > 262144u || !append_rows || append_rows > 8192u)
            return hipErrorInvalidValue;
        if (tokens <= storage_.capacity && append_rows <= storage_.append_rows) return hipSuccess;
        if (retained_tokens()) return hipErrorInvalidValue;
        Drafter next;
        next.cache_.reset(new (std::nothrow) PromptCache);
        if (!next.cache_) return hipErrorOutOfMemory;
        hipError_t status = next.cache_->reserve(tokens, append_rows);
        if (status != hipSuccess) return status;
        auto& s = next.storage_;
        s.capacity = tokens; s.append_rows = append_rows;
        s.score_stride = (tokens + 31u) & ~31u;
        s.moe_bytes = moe_workspace_bytes(2u);
        unsigned allocation = 0;
        const auto allocate = [&](auto** pointer, size_t count) {
            if (status != hipSuccess) return;
            const size_t bytes = count * sizeof(**pointer);
            status = hipMalloc(reinterpret_cast<void**>(pointer), bytes);
            if (status == hipSuccess) { s.owned[allocation++] = *pointer; s.bytes += bytes; }
        };
        allocate(&s.query_projection, 2u * 8192u);
        allocate(&s.queries, 2u * 4096u); allocate(&s.gates, 2u * 4096u);
        allocate(&s.context, 2u * 4096u); allocate(&s.gated_context, 2u * 4096u);
        allocate(&s.output_projection, 2u * 2048u); allocate(&s.post_attention, 2u * 2048u);
        allocate(&s.attention_residual, 2u * 2048u); allocate(&s.final_hidden, 2u * 2048u);
        allocate(&s.final_residual, 2u * 2048u);
        allocate(&s.scores, size_t(2u) * 16u * s.score_stride);
        allocate(&s.float_context, 2u * 4096u);
        allocate(&s.moe, s.moe_bytes);
        allocate(&s.logits, size_t(2u) * head_vocabulary);
        allocate(&s.tokens, 2u); allocate(&s.values, 2u); allocate(&s.head_invalid, 1u);
        if (status != hipSuccess) return status;
        status = hipHostMalloc(reinterpret_cast<void**>(&s.host), sizeof(Publication));
        if (status != hipSuccess) return status;
        s.bytes += sizeof(Publication);
        cache_.swap(next.cache_); std::swap(storage_, next.storage_);
        invalidate_observation();
        return hipSuccess;
    }

    // Storage epochs are supplied by the model owner and checked before every
    // append/proposal. Rebinding a live prefix would mix two sets of weights.
    bool bind(const DrafterWeights& weights, const DrafterTables& tables, uint64_t epoch) {
        const auto& p = weights.prompt;
        const auto& m = weights.moe;
        if (quarantined_ || retained_tokens() || !epoch ||
            !p.embeddings || !p.embedding_norm || !p.hidden_norm || !p.fusion ||
            !p.input_norm || !p.kv_projection || !p.key_norm || !weights.query ||
            !weights.query_norm || !weights.output || !weights.post_attention_norm ||
            !weights.final_norm || !weights.lm_head || !m.router || !m.shared_gate ||
            !m.shared_gate_up || !m.shared_down || !m.routed_gate_up || !m.routed_down ||
            !tables.rsqrt || !tables.rope || !tables.rope_rows || tables.rope_rows > 262144u ||
            !tables.exp2 || !tables.reciprocal || !tables.moe.silu || !tables.moe.sigmoid ||
            !tables.moe.router_exp_fraction) return false;
        weights_ = weights; tables_ = tables; epoch_ = epoch;
        model_binding_ = {};
        invalidate_observation();
        return true;
    }

    // This overload owns the model lease and the explicit K/V and shared
    // gate/up packs for every subsequent asynchronous stage.
    bool bind(const ModelWeightBinding& binding, const DrafterTables& tables, uint64_t epoch) {
        DrafterWeights weights;
        if (!binding.weights(epoch, &weights) || !bind(weights, tables, epoch)) return false;
        model_binding_ = binding;
        return true;
    }

    PromptStep append_target(const uint16_t* hidden, const uint32_t* shifted_ids,
        unsigned first_position, unsigned rows, bool split1024_pre_fc_norm,
        uint64_t current_epoch, hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u) {
        if (quarantined_) return {completion_error_, "quarantined", retained_tokens(), true};
        if (!cache_ || !model_current(current_epoch) || !maximum_blocks || maximum_blocks > 4096u)
            return {hipErrorInvalidValue, "model_contract", retained_tokens()};
        invalidate_observation();
        PromptWeights prompt = weights_.prompt;
        prompt.split1024_pre_fc_norm = split1024_pre_fc_norm;
        const auto result = cache_->append(prompt, hidden, shifted_ids, tables_.rsqrt,
            tables_.rope, tables_.rope_rows, first_position, rows, project,
            &maximum_blocks, stream);
        if (result.completion_unknown) quarantine(result.status);
        return result;
    }

    DraftStep propose(unsigned first_position, unsigned rows, uint64_t current_epoch,
        hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u) {
        if (quarantined_) return {completion_error_, "quarantined", true};
        if (!cache_ || !model_current(current_epoch) || !maximum_blocks || maximum_blocks > 4096u)
            return {hipErrorInvalidValue, "model_contract"};
        const PromptTail tail = cache_->tail(first_position, rows);
        if (!tail.fusion) return {hipErrorInvalidValue, "completed_tail_contract"};
        MoeBuffers moe;
        if (!bind_moe_buffers(storage_.moe, storage_.moe_bytes, rows, &moe))
            return {hipErrorInvalidValue, "workspace_contract"};
        invalidate_observation();
        auto& s = storage_;
        const auto fail = [&](hipError_t status, const char* stage) {
            const hipError_t drained = hipStreamSynchronize(stream);
            if (drained != hipSuccess) { quarantine(drained); return DraftStep{drained, stage, true}; }
            return DraftStep{status, stage};
        };
        hipError_t status = launch_projection(weights_.query, tail.normalized, s.query_projection,
            8192u, 2048u, rows, maximum_blocks, stream);
        if (status != hipSuccess) return fail(status, "query_projection");
        status = launch_queries(s.query_projection, weights_.query_norm, tables_.rsqrt, tables_.rope,
            tables_.rope_rows, first_position, rows, s.queries, s.gates, nullptr, stream);
        if (status != hipSuccess) return fail(status, "query_normalization_rope");
        status = launch_attention(s.queries, cache_->data(), cache_->retained_tokens(), first_position, rows,
            tables_.exp2, tables_.reciprocal, s.scores, s.score_stride, s.float_context, s.context, stream);
        if (status != hipSuccess) return fail(status, "full_history_attention");
        status = launch_gate(s.context, s.gates, tables_.moe.sigmoid, s.gated_context, rows, stream);
        if (status != hipSuccess) return fail(status, "attention_gate");
        status = launch_projection(weights_.output, s.gated_context, s.output_projection,
            2048u, 4096u, rows, maximum_blocks, stream);
        if (status != hipSuccess) return fail(status, "output_projection");
        status = launch_residual_normalize(s.output_projection, tail.fusion, weights_.post_attention_norm,
            tables_.rsqrt, rows, s.post_attention, s.attention_residual, stream);
        if (status != hipSuccess) return fail(status, "post_attention_normalization");
        status = launch_moe(s.post_attention, weights_.moe, tables_.moe, s.moe, s.moe_bytes,
            rows, maximum_blocks, stream);
        if (status != hipSuccess) return fail(status, "moe");
        status = launch_residual_normalize(moe.output, s.attention_residual, weights_.final_norm,
            tables_.rsqrt, rows, s.final_hidden, s.final_residual, stream);
        if (status != hipSuccess) return fail(status, "final_normalization");
        status = launch_head(weights_.lm_head, s.final_hidden, s.logits, s.tokens, s.values,
            s.head_invalid, rows, maximum_blocks, stream);
        if (status != hipSuccess) return fail(status, "output_head");
        *s.host = {};
        status = hipMemcpyAsync(&s.host->moe_invalid, moe.invalid, sizeof(uint32_t), hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) return fail(status, "moe_flag_copy");
        status = hipMemcpyAsync(&s.host->head_invalid, s.head_invalid, sizeof(uint32_t), hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) return fail(status, "head_flag_copy");
        status = hipMemcpyAsync(s.host->tokens, s.tokens, rows * sizeof(uint32_t), hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) return fail(status, "tokens_copy");
        status = hipMemcpyAsync(s.host->logits, s.values, rows * sizeof(float), hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) return fail(status, "logits_copy");
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) { quarantine(status); return {status, "proposal_completion", true}; }
        if (s.host->moe_invalid || s.host->head_invalid)
            return {hipErrorInvalidValue, "invalid_numerical_result"};
        DraftStep result;
        result.first_position = first_position; result.rows = rows;
        for (unsigned row = 0; row < rows; ++row) {
            if (s.host->tokens[row] >= head_vocabulary || !std::isfinite(s.host->logits[row]))
                return {hipErrorInvalidValue, "invalid_sampling_result"};
            result.tokens[row] = s.host->tokens[row]; result.logits[row] = s.host->logits[row];
        }
        observed_first_ = first_position; observed_rows_ = rows;
        return result;
    }

    bool truncate(unsigned tokens, uint64_t current_epoch) {
        if (quarantined_ || !cache_ || !model_current(current_epoch)) return false;
        const unsigned previous = cache_->retained_tokens();
        if (!cache_->truncate(tokens)) return false;
        if (tokens != previous) invalidate_observation();
        return true;
    }
    unsigned retained_tokens() const { return cache_ ? cache_->retained_tokens() : 0u; }
    unsigned capacity() const { return storage_.capacity; }
    bool quarantined() const { return quarantined_; }
    size_t allocated_bytes() const { return storage_.bytes + (cache_ ? cache_->allocated_bytes() : 0u); }
    const uint16_t* cache_data() const { return quarantined_ || !cache_ ? nullptr : cache_->data(); }
    DrafterObservation observation(uint64_t current_epoch) const {
        if (quarantined_ || !model_current(current_epoch) || !observed_rows_) return {};
        const auto& s = storage_;
        return {generation_, observed_first_, observed_rows_, s.query_projection, s.queries, s.gates,
            s.context, s.gated_context, s.output_projection, s.post_attention, s.attention_residual,
            s.moe, s.moe_bytes, s.final_hidden, s.final_residual, s.logits};
    }

private:
    struct Publication { uint32_t moe_invalid, head_invalid, tokens[2]; float logits[2]; };
    struct Storage {
        std::array<void*, 17> owned{};
        uint16_t *query_projection = nullptr, *queries = nullptr, *gates = nullptr;
        uint16_t *context = nullptr, *gated_context = nullptr, *output_projection = nullptr;
        uint16_t *post_attention = nullptr, *attention_residual = nullptr;
        uint16_t *final_hidden = nullptr, *final_residual = nullptr, *logits = nullptr;
        float *scores = nullptr, *float_context = nullptr, *values = nullptr;
        unsigned char* moe = nullptr;
        uint32_t *tokens = nullptr, *head_invalid = nullptr;
        Publication* host = nullptr;
        size_t moe_bytes = 0, bytes = 0;
        unsigned capacity = 0, append_rows = 0, score_stride = 0;
    } storage_;
    static hipError_t project(void* context, const uint16_t* weights, const uint16_t* input,
        uint16_t* output, unsigned output_features, unsigned input_features, unsigned rows, hipStream_t stream) {
        return launch_projection(weights, input, output, output_features, input_features, rows,
            *static_cast<const unsigned*>(context), stream);
    }
    void invalidate_observation() { observed_rows_ = 0; ++generation_; }
    bool model_current(uint64_t epoch) const {
        return epoch && epoch_ == epoch && (!model_binding_.epoch() || model_binding_.valid(epoch));
    }
    void quarantine(hipError_t status) {
        invalidate_observation(); quarantined_ = true; completion_error_ = status;
        if (cache_) (void)cache_->quarantine_borrower(status);
        model_binding_.quarantine();
    }
    std::unique_ptr<PromptCache> cache_;
    ModelWeightBinding model_binding_;
    DrafterWeights weights_;
    DrafterTables tables_;
    uint64_t epoch_ = 0, generation_ = 0;
    unsigned observed_first_ = 0, observed_rows_ = 0;
    bool quarantined_ = false;
    hipError_t completion_error_ = hipSuccess;
};
} // namespace qrt_sm121_mtp
