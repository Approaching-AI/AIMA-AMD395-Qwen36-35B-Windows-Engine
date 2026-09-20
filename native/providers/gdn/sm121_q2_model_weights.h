#pragma once
#include "sm121_mtp_model_weights.h"
#include <stdexcept>
#include <string>

namespace qrt_sm121_q2 {
enum class WeightRole : unsigned {
    Embedding, FinalNorm, Head, InputNorm, PostNorm, Router, SharedGate,
    SharedGateProjection, SharedUpProjection, SharedDown, RoutedGateUp, RoutedDown,
    Qkv, Z, A, B, Output, LinearNorm, Convolution, Query, Key, Value, QueryNorm, KeyNorm, Count
};
constexpr unsigned target_layers = 40u;
constexpr size_t target_weight_count = 633u;
struct TensorSpec {
    std::string name;
    unsigned layer = target_layers;
    WeightRole role = WeightRole::Count;
    unsigned rank = 0;
    std::array<size_t, 3> shape{};
    size_t bytes() const {
        size_t result = sizeof(uint16_t);
        for (unsigned i = 0; i < rank; ++i) result *= shape[i];
        return result;
    }
};
// Original safetensors identities. Linear gating tables are separately bound
// numerical inputs; they are not synthesized from a guessed BF16 weight view.
inline const std::array<TensorSpec, target_weight_count>& target_weight_specs() {
    static const auto specs = [] {
        std::array<TensorSpec, target_weight_count> result{};
        size_t index = 0;
        const auto add = [&](std::string name, unsigned layer, WeightRole role,
                             unsigned rank, std::array<size_t, 3> shape) {
            result.at(index++) = {std::move(name), layer, role, rank, shape};
        };
        add("model.language_model.embed_tokens.weight", target_layers, WeightRole::Embedding, 2, {248320,2048,0});
        add("model.language_model.norm.weight", target_layers, WeightRole::FinalNorm, 1, {2048,0,0});
        add("lm_head.weight", target_layers, WeightRole::Head, 2, {248320,2048,0});
        for (unsigned layer = 0; layer < target_layers; ++layer) {
            const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
            const auto item = [&](const char* suffix, WeightRole role, unsigned rank, std::array<size_t,3> shape) {
                add(prefix + suffix, layer, role, rank, shape);
            };
            item("input_layernorm.weight", WeightRole::InputNorm, 1, {2048,0,0});
            item("post_attention_layernorm.weight", WeightRole::PostNorm, 1, {2048,0,0});
            item("mlp.gate.weight", WeightRole::Router, 2, {256,2048,0});
            item("mlp.shared_expert_gate.weight", WeightRole::SharedGate, 2, {1,2048,0});
            item("mlp.shared_expert.gate_proj.weight", WeightRole::SharedGateProjection, 2, {512,2048,0});
            item("mlp.shared_expert.up_proj.weight", WeightRole::SharedUpProjection, 2, {512,2048,0});
            item("mlp.shared_expert.down_proj.weight", WeightRole::SharedDown, 2, {2048,512,0});
            item("mlp.experts.gate_up_proj", WeightRole::RoutedGateUp, 3, {256,1024,2048});
            item("mlp.experts.down_proj", WeightRole::RoutedDown, 3, {256,2048,512});
            if (layer % 4u != 3u) {
                item("linear_attn.in_proj_qkv.weight", WeightRole::Qkv, 2, {8192,2048,0});
                item("linear_attn.in_proj_z.weight", WeightRole::Z, 2, {4096,2048,0});
                item("linear_attn.in_proj_a.weight", WeightRole::A, 2, {32,2048,0});
                item("linear_attn.in_proj_b.weight", WeightRole::B, 2, {32,2048,0});
                item("linear_attn.out_proj.weight", WeightRole::Output, 2, {2048,4096,0});
                item("linear_attn.norm.weight", WeightRole::LinearNorm, 1, {128,0,0});
                item("linear_attn.conv1d.weight", WeightRole::Convolution, 3, {8192,1,4});
            } else {
                item("self_attn.q_proj.weight", WeightRole::Query, 2, {8192,2048,0});
                item("self_attn.k_proj.weight", WeightRole::Key, 2, {512,2048,0});
                item("self_attn.v_proj.weight", WeightRole::Value, 2, {512,2048,0});
                item("self_attn.o_proj.weight", WeightRole::Output, 2, {2048,4096,0});
                item("self_attn.q_norm.weight", WeightRole::QueryNorm, 1, {256,0,0});
                item("self_attn.k_norm.weight", WeightRole::KeyNorm, 1, {256,0,0});
            }
        }
        if (index != result.size()) throw std::logic_error("target tensor specification count");
        return result;
    }();
    return specs;
}

namespace target_weight_detail {
constexpr size_t shared_part_elements = size_t(512u) * 2048u;
constexpr size_t packed_bytes = target_layers * 2u * shared_part_elements * sizeof(uint16_t);
struct Storage {
    std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source;
    std::array<std::array<const uint16_t*, static_cast<unsigned>(WeightRole::Count)>, target_layers+1u> views{};
    uint16_t* packed = nullptr;
    uint64_t epoch = 0;
    bool quarantined = false;
    Storage* quarantine_next = nullptr;
    std::shared_ptr<Storage> quarantine_hold;
    ~Storage() { if (packed) (void)hipFree(packed); }
};
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
} // namespace target_weight_detail

// Copies pin the same original model allocations and 160 MiB shared-expert
// pack. Keep a binding for every pending target launch, including failed fences.
// Callers serialize binding use, model replacement and quarantine.
class ModelWeightBinding {
public:
    uint64_t epoch() const { return storage_ ? storage_->epoch : 0; }
    bool valid(uint64_t epoch) const { return target_weight_detail::valid(storage_, epoch); }
    size_t allocated_bytes() const { return storage_ ? target_weight_detail::packed_bytes : 0; }
    void quarantine() const { target_weight_detail::quarantine(storage_); }
    const uint16_t* tensor(uint64_t epoch, unsigned layer, WeightRole role) const {
        return valid(epoch) && layer <= target_layers && role < WeightRole::Count
            ? storage_->views[layer][static_cast<unsigned>(role)] : nullptr;
    }
    const uint16_t* shared_gate_up(uint64_t epoch, unsigned layer) const {
        return valid(epoch) && layer < target_layers
            ? storage_->packed + size_t(layer) * 2u * target_weight_detail::shared_part_elements : nullptr;
    }
    // Instantiate with the real private LinearLayerViews/AttentionLayerViews.
    // Cache and scratch pointers are supplied independently by the target owner.
    template<class Layer> bool linear_weights(uint64_t epoch, unsigned layer, Layer* output) const {
        if (!output || !valid(epoch) || layer >= target_layers || layer % 4u == 3u) return false;
        common_weights(epoch, layer, output);
        auto& v = output->linear;
        v.qkv_weights = tensor(epoch,layer,WeightRole::Qkv); v.z_weights = tensor(epoch,layer,WeightRole::Z);
        v.a_weights = tensor(epoch,layer,WeightRole::A); v.b_weights = tensor(epoch,layer,WeightRole::B);
        v.output_weights = tensor(epoch,layer,WeightRole::Output); v.norm_weights = tensor(epoch,layer,WeightRole::LinearNorm);
        v.convolution.weights = tensor(epoch,layer,WeightRole::Convolution);
        return true;
    }
    template<class Layer> bool attention_weights(uint64_t epoch, unsigned layer, Layer* output) const {
        if (!output || !valid(epoch) || layer >= target_layers || layer % 4u != 3u) return false;
        common_weights(epoch, layer, output);
        auto& v = output->attention;
        v.q_weights = tensor(epoch,layer,WeightRole::Query); v.k_weights = tensor(epoch,layer,WeightRole::Key);
        v.v_weights = tensor(epoch,layer,WeightRole::Value); v.output_weights = tensor(epoch,layer,WeightRole::Output);
        v.q_norm_weights = tensor(epoch,layer,WeightRole::QueryNorm); v.k_norm_weights = tensor(epoch,layer,WeightRole::KeyNorm);
        return true;
    }
private:
    template<class Layer> void common_weights(uint64_t epoch, unsigned layer, Layer* output) const {
        output->input_norm_weights = tensor(epoch,layer,WeightRole::InputNorm);
        output->post_norm_weights = tensor(epoch,layer,WeightRole::PostNorm);
        auto& v = output->moe_weights;
        v.router = tensor(epoch,layer,WeightRole::Router); v.shared_gate = tensor(epoch,layer,WeightRole::SharedGate);
        v.shared_gate_up = shared_gate_up(epoch,layer); v.shared_down = tensor(epoch,layer,WeightRole::SharedDown);
        v.routed_gate_up = tensor(epoch,layer,WeightRole::RoutedGateUp); v.routed_down = tensor(epoch,layer,WeightRole::RoutedDown);
    }
    friend class ModelWeights;
    std::shared_ptr<target_weight_detail::Storage> storage_;
};

class ModelWeights {
public:
    ModelWeights() = default;
    ModelWeights(const ModelWeights&) = delete;
    ModelWeights& operator=(const ModelWeights&) = delete;
    qrt_sm121_mtp::ModelWeightStep prepare(std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> source,
        uint64_t expected_epoch, hipStream_t stream = nullptr) {
        using namespace target_weight_detail;
        if (quarantined()) return {hipErrorInvalidValue,"quarantined",true};
        if (!source || !expected_epoch || source->epoch() != expected_epoch)
            return {hipErrorInvalidValue,"target_model_lease"};
        std::shared_ptr<Storage> next;
        try {
            next = std::make_shared<Storage>(); next->source = std::move(source); next->epoch = expected_epoch;
            for (const auto& spec : target_weight_specs()) {
                qrt_sm121_mtp::ModelTensorView view;
                if (!next->source->tensor(spec.name.c_str(), &view) || !view.name ||
                    spec.name != view.name || !view.device ||
                    reinterpret_cast<uintptr_t>(view.device) % alignof(uint16_t) ||
                    view.bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(view.device) ||
                    !view.bf16 || !view.contiguous || view.rank != spec.rank ||
                    view.shape != spec.shape || view.bytes != spec.bytes() || view.epoch != expected_epoch)
                    return {hipErrorInvalidValue,"target_tensor_contract"};
                next->views[spec.layer][static_cast<unsigned>(spec.role)] = view.device;
            }
        } catch (const std::bad_alloc&) { return {hipErrorOutOfMemory,"target_tensor_lookup"}; }
          catch (...) { return {hipErrorInvalidValue,"target_tensor_lookup"}; }
        if (next->source->epoch() != expected_epoch) return {hipErrorInvalidValue,"target_model_epoch"};
        hipError_t status = hipMalloc(reinterpret_cast<void**>(&next->packed), packed_bytes);
        if (status != hipSuccess) return {status,"target_shared_pack_allocation"};
        for (unsigned layer = 0; layer < target_layers && status == hipSuccess; ++layer) {
            for (unsigned part = 0; part < 2u; ++part) {
                const auto role = part ? WeightRole::SharedUpProjection : WeightRole::SharedGateProjection;
                status = hipMemcpyAsync(next->packed + (size_t(layer)*2u+part)*shared_part_elements,
                    next->views[layer][static_cast<unsigned>(role)], shared_part_elements*sizeof(uint16_t),
                    hipMemcpyDeviceToDevice, stream);
                if (status != hipSuccess) break;
            }
        }
        // Even a failed enqueue may have submitted work. Drain before release;
        // an unknown completion keeps both the pack and original source pinned.
        const hipError_t completed = hipStreamSynchronize(stream);
        if (completed != hipSuccess) {
            quarantine(next); terminal_ = true;
            return {completed,"target_shared_pack_completion",true};
        }
        if (status != hipSuccess) return {status,"target_shared_pack_copy"};
        if (next->source->epoch() != expected_epoch) return {hipErrorInvalidValue,"target_model_epoch"};
        storage_.swap(next);
        return {};
    }
    bool quarantined() const { return terminal_ || (storage_ && storage_->quarantined); }
    ModelWeightBinding binding(uint64_t epoch) const {
        ModelWeightBinding result;
        if (!quarantined() && target_weight_detail::valid(storage_,epoch)) result.storage_ = storage_;
        return result;
    }
private:
    std::shared_ptr<target_weight_detail::Storage> storage_;
    bool terminal_ = false;
};
} // namespace qrt_sm121_q2
