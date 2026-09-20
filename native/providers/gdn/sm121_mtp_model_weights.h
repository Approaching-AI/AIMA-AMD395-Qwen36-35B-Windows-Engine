#pragma once
#include <hip/hip_runtime.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

namespace qrt_sm121_mtp {
struct ModelTensorSpec {
    const char* name;
    unsigned rank;
    std::array<size_t, 3> shape;
    size_t bytes() const {
        size_t result = sizeof(uint16_t);
        for (unsigned i = 0; i < rank; ++i) result *= shape[i];
        return result;
    }
};
enum class ModelWeight : unsigned {
    Embedding, EmbeddingNorm, HiddenNorm, Fusion, InputNorm, Key, Value, KeyNorm,
    Query, QueryNorm, Output, PostAttentionNorm, Router, SharedGate,
    SharedGateProjection, SharedUpProjection, SharedDown, RoutedGateUp, RoutedDown,
    FinalNorm, LmHead, Count
};
inline constexpr std::array<ModelTensorSpec, static_cast<unsigned>(ModelWeight::Count)> model_weight_specs{{
    {"model.language_model.embed_tokens.weight", 2, {248320, 2048, 0}},
    {"mtp.pre_fc_norm_embedding.weight", 1, {2048, 0, 0}},
    {"mtp.pre_fc_norm_hidden.weight", 1, {2048, 0, 0}},
    {"mtp.fc.weight", 2, {2048, 4096, 0}},
    {"mtp.layers.0.input_layernorm.weight", 1, {2048, 0, 0}},
    {"mtp.layers.0.self_attn.k_proj.weight", 2, {512, 2048, 0}},
    {"mtp.layers.0.self_attn.v_proj.weight", 2, {512, 2048, 0}},
    {"mtp.layers.0.self_attn.k_norm.weight", 1, {256, 0, 0}},
    {"mtp.layers.0.self_attn.q_proj.weight", 2, {8192, 2048, 0}},
    {"mtp.layers.0.self_attn.q_norm.weight", 1, {256, 0, 0}},
    {"mtp.layers.0.self_attn.o_proj.weight", 2, {2048, 4096, 0}},
    {"mtp.layers.0.post_attention_layernorm.weight", 1, {2048, 0, 0}},
    {"mtp.layers.0.mlp.gate.weight", 2, {256, 2048, 0}},
    {"mtp.layers.0.mlp.shared_expert_gate.weight", 2, {1, 2048, 0}},
    {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", 2, {512, 2048, 0}},
    {"mtp.layers.0.mlp.shared_expert.up_proj.weight", 2, {512, 2048, 0}},
    {"mtp.layers.0.mlp.shared_expert.down_proj.weight", 2, {2048, 512, 0}},
    {"mtp.layers.0.mlp.experts.gate_up_proj", 3, {256, 1024, 2048}},
    {"mtp.layers.0.mlp.experts.down_proj", 3, {256, 2048, 512}},
    {"mtp.norm.weight", 1, {2048, 0, 0}},
    {"lm_head.weight", 2, {248320, 2048, 0}},
}};

struct ModelTensorView {
    const char* name = nullptr;
    const uint16_t* device = nullptr;
    unsigned rank = 0;
    std::array<size_t, 3> shape{};
    size_t bytes = 0;
    uint64_t epoch = 0;
    bool bf16 = false;
    bool contiguous = false;
};

// Implementations must OWN or PIN every returned device allocation until this
// source's last shared reference is released. An epoch counter alone is not a
// memory lease. Lookup reports original tensor names, shapes and row-major BF16
// layout; no original/reference hidden data participates in weight resolution.
class ModelWeightSource {
public:
    virtual ~ModelWeightSource() = default;
    virtual uint64_t epoch() const noexcept = 0;
    virtual bool tensor(const char* name, ModelTensorView* out) const = 0;
};

struct ModelWeightStep {
    hipError_t status = hipSuccess;
    const char* stage = "complete";
    bool completion_unknown = false;
};

namespace mtp_model_weight_detail {
constexpr size_t part_elements = size_t(512) * 2048;
constexpr size_t packed_bytes = 4 * part_elements * sizeof(uint16_t);
struct Storage {
    std::shared_ptr<const ModelWeightSource> source;
    std::array<const uint16_t*, model_weight_specs.size()> views{};
    uint16_t* packed = nullptr;
    uint64_t epoch = 0;
    bool quarantined = false;
    Storage* quarantine_next = nullptr;
    std::shared_ptr<Storage> quarantine_hold;
    ~Storage() { if (packed) (void)hipFree(packed); }
};
// A failed fence cannot allocate a recovery record. The already allocated
// storage roots its own lease and pack until process teardown. There is no
// request-level recovery API that could prematurely free unknown GPU inputs.
inline std::atomic<Storage*> quarantined_head{nullptr};
inline void quarantine(const std::shared_ptr<Storage>& storage) {
    if (!storage || storage->quarantined) return;
    storage->quarantined = true;
    storage->quarantine_hold = storage;
    Storage* previous = quarantined_head.load(std::memory_order_relaxed);
    do { storage->quarantine_next = previous; }
    while (!quarantined_head.compare_exchange_weak(previous, storage.get(),
        std::memory_order_release, std::memory_order_relaxed));
}
inline bool valid(const std::shared_ptr<Storage>& storage, uint64_t epoch) {
    return storage && !storage->quarantined && epoch && storage->epoch == epoch &&
        storage->source && storage->source->epoch() == epoch;
}
} // namespace mtp_model_weight_detail

// Keep this binding alive for the complete lifetime of any Drafter bound to
// its pointer values. Copies pin the same model and packed weights. Request
// use, rebind, and quarantine are serialized by the caller, like Drafter use.
class ModelWeightBinding {
public:
    uint64_t epoch() const { return storage_ ? storage_->epoch : 0; }
    bool valid(uint64_t epoch) const { return mtp_model_weight_detail::valid(storage_, epoch); }
    size_t allocated_bytes() const { return storage_ ? mtp_model_weight_detail::packed_bytes : 0; }
    void quarantine() const { mtp_model_weight_detail::quarantine(storage_); }

    // Instantiate with DrafterWeights. Keeping the adapter independent of the
    // device kernel headers allows its real ownership code to run under ASan.
    template<class Weights>
    bool weights(uint64_t current_epoch, Weights* out) const {
        if (!out) return false;
        *out = {};
        if (!valid(current_epoch)) return false;
        const auto get = [&](ModelWeight weight) { return storage_->views[static_cast<unsigned>(weight)]; };
        auto& prompt = out->prompt;
        prompt.embeddings = get(ModelWeight::Embedding);
        prompt.embedding_norm = get(ModelWeight::EmbeddingNorm);
        prompt.hidden_norm = get(ModelWeight::HiddenNorm);
        prompt.fusion = get(ModelWeight::Fusion);
        prompt.input_norm = get(ModelWeight::InputNorm);
        prompt.kv_projection = storage_->packed;
        prompt.key_norm = get(ModelWeight::KeyNorm);
        out->query = get(ModelWeight::Query); out->query_norm = get(ModelWeight::QueryNorm);
        out->output = get(ModelWeight::Output); out->post_attention_norm = get(ModelWeight::PostAttentionNorm);
        out->moe.router = get(ModelWeight::Router); out->moe.shared_gate = get(ModelWeight::SharedGate);
        out->moe.shared_gate_up = storage_->packed + 2 * mtp_model_weight_detail::part_elements;
        out->moe.shared_down = get(ModelWeight::SharedDown);
        out->moe.routed_gate_up = get(ModelWeight::RoutedGateUp); out->moe.routed_down = get(ModelWeight::RoutedDown);
        out->final_norm = get(ModelWeight::FinalNorm); out->lm_head = get(ModelWeight::LmHead);
        return true;
    }

private:
    friend class ModelWeights;
    std::shared_ptr<mtp_model_weight_detail::Storage> storage_;
};

class ModelWeights {
public:
    ModelWeights() = default;
    ModelWeights(const ModelWeights&) = delete;
    ModelWeights& operator=(const ModelWeights&) = delete;

    ModelWeightStep prepare(std::shared_ptr<const ModelWeightSource> source,
        uint64_t expected_epoch, hipStream_t stream = nullptr) {
        using namespace mtp_model_weight_detail;
        if (quarantined()) return {hipErrorInvalidValue, "quarantined", true};
        if (!source || !expected_epoch || source->epoch() != expected_epoch)
            return {hipErrorInvalidValue, "model_lease"};
        std::shared_ptr<Storage> next;
        try {
            next = std::make_shared<Storage>();
            next->source = std::move(source); next->epoch = expected_epoch;
            for (size_t i = 0; i < model_weight_specs.size(); ++i) {
                const auto& expected = model_weight_specs[i];
                ModelTensorView view;
                if (!next->source->tensor(expected.name, &view) || !view.name ||
                    std::strcmp(view.name, expected.name) || !view.device ||
                    reinterpret_cast<uintptr_t>(view.device) % alignof(uint16_t) ||
                    !view.bf16 || !view.contiguous || view.rank != expected.rank ||
                    view.shape != expected.shape || view.bytes != expected.bytes() ||
                    view.epoch != expected_epoch)
                    return {hipErrorInvalidValue, "tensor_contract"};
                next->views[i] = view.device;
            }
        } catch (const std::bad_alloc&) { return {hipErrorOutOfMemory, "tensor_lookup"}; }
          catch (...) { return {hipErrorInvalidValue, "tensor_lookup"}; }
        if (next->source->epoch() != expected_epoch) return {hipErrorInvalidValue, "model_epoch"};
        hipError_t status = hipMalloc(reinterpret_cast<void**>(&next->packed), packed_bytes);
        if (status != hipSuccess) return {status, "packed_allocation"};
        constexpr std::array<ModelWeight, 4> copies{{ModelWeight::Key, ModelWeight::Value,
            ModelWeight::SharedGateProjection, ModelWeight::SharedUpProjection}};
        for (size_t i = 0; i < copies.size(); ++i) {
            status = hipMemcpyAsync(next->packed + i * part_elements,
                next->views[static_cast<unsigned>(copies[i])], part_elements * sizeof(uint16_t),
                hipMemcpyDeviceToDevice, stream);
            if (status != hipSuccess) break;
        }
        // A failing enqueue can still have submitted work; drain it before
        // discarding the temporary pack or releasing its model lease.
        const hipError_t completed = hipStreamSynchronize(stream);
        if (completed != hipSuccess) {
            quarantine(next); terminal_ = true;
            return {completed, "packed_completion", true};
        }
        if (status != hipSuccess) return {status, "packed_copy"};
        if (next->source->epoch() != expected_epoch) return {hipErrorInvalidValue, "model_epoch"};
        storage_.swap(next);
        return {};
    }

    bool quarantined() const { return terminal_ || (storage_ && storage_->quarantined); }
    ModelWeightBinding binding(uint64_t current_epoch) const {
        ModelWeightBinding result;
        if (!quarantined() && mtp_model_weight_detail::valid(storage_, current_epoch)) result.storage_ = storage_;
        return result;
    }
private:
    std::shared_ptr<mtp_model_weight_detail::Storage> storage_;
    bool terminal_ = false;
};
} // namespace qrt_sm121_mtp
