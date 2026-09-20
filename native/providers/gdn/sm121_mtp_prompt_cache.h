#pragma once
#include <hip/hip_runtime.h>
#include <array>
#include <utility>
#include "sm121_mtp_frontier.h"
#include "sm121_mtp_kv.h"

namespace qrt_sm121_mtp {
struct PromptWeights {
    const uint16_t* embeddings = nullptr;
    const uint16_t* embedding_norm = nullptr;
    const uint16_t* hidden_norm = nullptr;
    const uint16_t* fusion = nullptr;
    const uint16_t* input_norm = nullptr;
    const uint16_t* kv_projection = nullptr; // Contiguous [K512,V512] x 2048.
    const uint16_t* key_norm = nullptr;
};

// The caller supplies a qualified BF16 projection implementation. Outputs
// have [tokens, output_features] layout; weights are output-major. The callback
// must enqueue its work on the supplied stream and return a HIP status.
using Project = hipError_t (*)(void* context, const uint16_t* weights,
    const uint16_t* input, uint16_t* output, unsigned int output_features,
    unsigned int input_features, unsigned int tokens, hipStream_t stream);

struct PromptStep {
    hipError_t status = hipSuccess;
    const char* stage = "complete";
    unsigned int retained_tokens = 0;
    // The caller must also quarantine borrowed inputs and model storage when
    // completion cannot be established. The request cannot resume afterward.
    bool completion_unknown = false;
};

// One request owns this storage. A completed append publishes only a
// contiguous prefix; failed work cannot advance its visible cache extent.
// Each input row must be the actual target's post-final-norm hidden row.
class PromptCache {
public:
    PromptCache() = default;
    PromptCache(const PromptCache&) = delete;
    PromptCache& operator=(const PromptCache&) = delete;
    ~PromptCache() {
        // A failed fence does not authorize freeing buffers still in use.
        // Process teardown owns recovery from this terminal request failure.
        if (quarantined_) return;
        for (void* pointer : owned_) if (pointer) (void)hipFree(pointer);
        if (host_invalid_) (void)hipHostFree(host_invalid_);
    }

    hipError_t reserve(unsigned int tokens, unsigned int rows = 8192u) {
        if (quarantined_) return completion_error_;
        if (!tokens || tokens > 262144u || !rows || rows > 8192u)
            return hipErrorInvalidValue;
        if (tokens <= capacity_ && rows <= row_capacity_) return hipSuccess;
        if (retained_) return hipErrorInvalidValue;
        PromptCache next;
        next.capacity_ = tokens;
        next.row_capacity_ = rows;
        hipError_t status = hipSuccess;
        size_t allocation = 0;
        const auto allocate = [&](auto** pointer, size_t count) {
            if (status != hipSuccess) return;
            status = hipMalloc(reinterpret_cast<void**>(pointer), count * sizeof(**pointer));
            if (status == hipSuccess) next.owned_[allocation++] = *pointer;
        };
        allocate(&next.cache_, size_t(tokens) * 1024u);
        allocate(&next.fusion_input_, size_t(rows) * 4096u);
        allocate(&next.fusion_output_, size_t(rows) * 2048u);
        allocate(&next.normalized_input_, size_t(rows) * 2048u);
        allocate(&next.projected_kv_, size_t(rows) * 1024u);
        allocate(&next.invalid_input_, 1u);
        if (status != hipSuccess) return status;
        status = hipHostMalloc(reinterpret_cast<void**>(&next.host_invalid_), sizeof(uint32_t));
        if (status != hipSuccess) return status;
        std::swap(owned_, next.owned_);
        std::swap(cache_, next.cache_);
        std::swap(fusion_input_, next.fusion_input_);
        std::swap(fusion_output_, next.fusion_output_);
        std::swap(normalized_input_, next.normalized_input_);
        std::swap(projected_kv_, next.projected_kv_);
        std::swap(invalid_input_, next.invalid_input_);
        std::swap(host_invalid_, next.host_invalid_);
        std::swap(capacity_, next.capacity_);
        std::swap(row_capacity_, next.row_capacity_);
        return hipSuccess;
    }

    PromptStep append(const PromptWeights& weights, const uint16_t* target_hidden,
        const uint32_t* shifted_ids, const unsigned char* rsqrt_table,
        const uint16_t* rope_table, unsigned int rope_rows,
        unsigned int first_position, unsigned int rows, Project project,
        void* project_context, hipStream_t stream = nullptr) {
        if (quarantined_)
            return {completion_error_, "quarantined", retained_, true};
        if (!project || !target_hidden || !shifted_ids || !rsqrt_table || !rope_table ||
            !weights.embeddings || !weights.embedding_norm || !weights.hidden_norm ||
            !weights.fusion || !weights.input_norm || !weights.kv_projection || !weights.key_norm ||
            !cache_ || first_position != retained_ || !rows || rows > row_capacity_ ||
            first_position >= capacity_ || rows > capacity_ - first_position ||
            first_position >= rope_rows || rows > rope_rows - first_position)
            return {hipErrorInvalidValue, "input_contract", retained_};
        const auto fail = [&](hipError_t status, const char* stage) {
            // Drain any preceding launches before scratch storage can be
            // reused, including a projection that failed after partial work.
            const hipError_t drained = hipStreamSynchronize(stream);
            if (drained != hipSuccess) return quarantine(drained, stage);
            return PromptStep{status == hipSuccess ? drained : status, stage, retained_};
        };
        hipError_t status = hipMemsetAsync(invalid_input_, 0, sizeof(uint32_t), stream);
        if (status != hipSuccess) return fail(status, "clear_input_flag");
        status = launch_fusion_inputs(weights.embeddings, target_hidden, shifted_ids,
            weights.embedding_norm, weights.hidden_norm, rsqrt_table, rows,
            fusion_input_, invalid_input_, stream);
        if (status != hipSuccess) return fail(status, "fusion_inputs");
        *host_invalid_ = 0;
        status = hipMemcpyAsync(host_invalid_, invalid_input_, sizeof(uint32_t), hipMemcpyDeviceToHost, stream);
        if (status != hipSuccess) return fail(status, "input_flag_copy");
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) return quarantine(status, "input_flag_completion");
        if (*host_invalid_) return {hipErrorInvalidValue, "invalid_token", retained_};
        status = project(project_context, weights.fusion, fusion_input_, fusion_output_,
                         2048u, 4096u, rows, stream);
        if (status != hipSuccess) return fail(status, "fusion_projection");
        status = launch_normalize(fusion_output_, weights.input_norm, rsqrt_table,
                                  rows, normalized_input_, stream);
        if (status != hipSuccess) return fail(status, "input_normalization");
        status = project(project_context, weights.kv_projection, normalized_input_, projected_kv_,
                         1024u, 2048u, rows, stream);
        if (status != hipSuccess) return fail(status, "kv_projection");
        status = launch_key_values(projected_kv_, weights.key_norm, rsqrt_table,
            rope_table, rope_rows, first_position, rows, capacity_, cache_, nullptr, stream);
        if (status != hipSuccess) return fail(status, "key_normalization_rope");
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) return quarantine(status, "cache_completion");
        retained_ += rows;
        return {hipSuccess, "complete", retained_};
    }

    // The caller may discard a provisional suffix after actual native target
    // acceptance. It must never infer this extent from reference decisions.
    bool truncate(unsigned int tokens) {
        if (quarantined_ || tokens > retained_) return false;
        retained_ = tokens;
        return true;
    }
    const uint16_t* data() const { return quarantined_ ? nullptr : cache_; }
    bool quarantined() const { return quarantined_; }
    unsigned int retained_tokens() const { return retained_; }
    unsigned int capacity() const { return capacity_; }
    size_t allocated_bytes() const {
        return cache_ ? size_t(capacity_) * 2048u + size_t(row_capacity_) * 18432u + 2u * sizeof(uint32_t) : 0u;
    }

private:
    PromptStep quarantine(hipError_t status, const char* stage) {
        quarantined_ = true;
        completion_error_ = status;
        return {status, stage, retained_, true};
    }
    std::array<void*, 6> owned_{};
    uint16_t* cache_ = nullptr;
    uint16_t* fusion_input_ = nullptr;
    uint16_t* fusion_output_ = nullptr;
    uint16_t* normalized_input_ = nullptr;
    uint16_t* projected_kv_ = nullptr;
    uint32_t* invalid_input_ = nullptr;
    uint32_t* host_invalid_ = nullptr;
    unsigned int capacity_ = 0;
    unsigned int row_capacity_ = 0;
    unsigned int retained_ = 0;
    bool quarantined_ = false;
    hipError_t completion_error_ = hipSuccess;
};
} // namespace qrt_sm121_mtp
