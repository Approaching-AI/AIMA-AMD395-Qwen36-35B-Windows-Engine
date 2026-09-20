#pragma once
#include "sm121_q2_linear_layer.h"
#include "sm121_q2_attention_layer.h"
#include "sm121_q2_head.h"
#include "sm121_q2_model_weights.h"
#include <atomic>
#include <cmath>
#include <memory>

namespace qrt_sm121_q2 {
struct TargetLinearCache {
    const float* state = nullptr;
    const void* ring = nullptr;
    unsigned ring_element_bytes = 2u;
    bool key_major = false;
};
struct TargetTables {
    std::array<const float*, target_layers> g{};
    const float* beta = nullptr;
    const float* gated_silu = nullptr;
    const unsigned char* convolution_silu = nullptr;
    AttentionBlockTables attention;
    qrt_sm121_mtp::MoeTables moe;
};
struct TargetSnapshot {
    const void* owner = nullptr;
    uint64_t generation = 0, model_epoch = 0;
    unsigned processed_tokens = 0, current_token = 0;
    // Entries use actual layer numbers; only the corresponding layer kind is read.
    std::array<TargetLinearCache, target_layers> linear{};
    std::array<CacheView, target_layers> attention{};
    TargetTables tables;
};

// The source pins every cache and numerical-table allocation in its snapshot.
// The enclosing request holds its transaction lock through evaluate/selection/
// publication. matches checks the actual owner, generation, model epoch, input
// extent and current token; self-hashes cannot establish this boundary.
class TargetStateSource {
public:
    virtual ~TargetStateSource() = default;
    virtual bool snapshot(TargetSnapshot*) const = 0;
    virtual bool matches(const TargetSnapshot&) const noexcept = 0;
    // Mark the actual owner unusable if GPU completion cannot be established.
    virtual void quarantine() const noexcept = 0;
};
struct TargetHostResult {
    std::array<uint32_t, target_layers+1u> invalid{};
    std::array<uint32_t, 2> tokens{};
    std::array<float, 2> logits{};
    std::array<uint16_t, 4096> normalized{};
};
struct TargetStep {
    hipError_t status = hipSuccess;
    const char* stage = "complete";
    bool completion_unknown = false;
};
struct TargetLinearSelection {
    const float* state = nullptr;
    const void* ring = nullptr;
    unsigned ring_element_bytes = 0, first_position = 0, rows = 0;
    bool key_major = false;
};

namespace target_detail {
struct Scratch {
    uint16_t *embedding = nullptr, *zero_residual = nullptr, *normalized_input = nullptr;
    uint16_t *input_residual = nullptr, *moe_input = nullptr, *final_norm = nullptr, *final_residual = nullptr;
    std::array<uint16_t*, 2> residual{};
    uint16_t *qkv = nullptr, *z = nullptr, *a = nullptr, *b = nullptr;
    uint16_t *convolution = nullptr, *core = nullptr, *linear_gated = nullptr, *linear_output = nullptr;
    uint16_t *q_projected = nullptr, *kv_projected = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    uint16_t *queries = nullptr, *gates = nullptr, *context = nullptr, *attention_gated = nullptr, *attention_output = nullptr;
    float *scores = nullptr, *float_context = nullptr;
    std::array<unsigned char*, 2> moe{};
    std::array<qrt_sm121_mtp::MoeBuffers, 2> moe_buffers{};
    std::array<float*, target_layers> states{};
    std::array<unsigned char*, target_layers> rings{};
    std::array<uint16_t*, target_layers> kv{};
    uint16_t* vocabulary_logits = nullptr;
    uint32_t *tokens = nullptr, *invalid = nullptr;
    float* values = nullptr;
    unsigned score_stride = 0;
};

// Every region has its own aligned extent. Layer-local producers reuse scratch
// in stream order; the two MoE outputs/residuals alternate between layers.
inline size_t layout(unsigned capacity, unsigned char* base, Scratch* output) {
    if (!capacity || capacity > target_context_limit) return 0;
    Scratch s; s.score_stride = (capacity + 31u) & ~31u;
    size_t bytes = 0;
    const auto take = [&](auto** pointer, size_t count) {
        using Pointer = std::remove_reference_t<decltype(*pointer)>;
        if (base) *pointer = reinterpret_cast<Pointer>(base + bytes);
        bytes += qrt_sm121_mtp::moe_aligned_bytes(count * sizeof(**pointer));
    };
    take(&s.embedding,4096u); take(&s.zero_residual,4096u); take(&s.normalized_input,4096u);
    take(&s.input_residual,4096u); take(&s.moe_input,4096u);
    for (auto& p : s.residual) take(&p,4096u);
    take(&s.final_norm,4096u); take(&s.final_residual,4096u);
    take(&s.qkv,16384u); take(&s.z,8192u); take(&s.a,64u); take(&s.b,64u);
    take(&s.convolution,16384u); take(&s.core,8192u); take(&s.linear_gated,8192u); take(&s.linear_output,4096u);
    take(&s.q_projected,16384u); take(&s.kv_projected,2048u); take(&s.q_norm,8192u); take(&s.k_norm,1024u);
    take(&s.queries,8192u); take(&s.gates,8192u); take(&s.context,8192u);
    take(&s.attention_gated,8192u); take(&s.attention_output,4096u);
    take(&s.scores,size_t(32u)*s.score_stride); take(&s.float_context,8192u);
    for (unsigned i = 0; i < 2u; ++i) {
        take(&s.moe[i],qrt_sm121_mtp::moe_workspace_bytes(2u));
        if (base && !qrt_sm121_mtp::bind_moe_buffers(s.moe[i],qrt_sm121_mtp::moe_workspace_bytes(2u),2u,&s.moe_buffers[i])) return 0;
    }
    for (unsigned layer = 0; layer < target_layers; ++layer) {
        if (layer % 4u == 3u) take(&s.kv[layer],2048u);
        else {
            take(&s.states[layer],size_t(2u)*state_elements);
            // A fixed FP32-sized extent supports either actual resident ring kind.
            take(&s.rings[layer],size_t(2u)*ring_elements*sizeof(float));
        }
    }
    take(&s.vocabulary_logits,size_t(2u)*qrt_sm121_mtp::head_vocabulary);
    take(&s.tokens,2u); take(&s.values,2u); take(&s.invalid,target_layers+1u);
    if (output) *output = s;
    return bytes;
}
struct Storage {
    unsigned char* device = nullptr;
    TargetHostResult* host = nullptr;
    size_t bytes = 0;
    unsigned capacity = 0;
    Scratch scratch;
    ModelWeightBinding weights;
    std::shared_ptr<const TargetStateSource> source;
    TargetSnapshot snapshot;
    std::array<uint32_t, 2> inputs{};
    bool complete = false, quarantined = false;
    Storage* quarantine_next = nullptr;
    std::shared_ptr<Storage> quarantine_hold;
    ~Storage() {
        if (device) (void)hipFree(device);
        if (host) (void)hipHostFree(host);
    }
};
inline std::atomic<Storage*> quarantined_head{nullptr};
inline void quarantine(const std::shared_ptr<Storage>& storage) {
    if (!storage || storage->quarantined) return;
    storage->complete = false; storage->quarantined = true;
    storage->weights.quarantine();
    if (storage->source) storage->source->quarantine();
    // No allocation is needed on the error path. Preserve device, pinned host,
    // original weights, cache owners and tables until process teardown.
    storage->quarantine_hold = storage;
    Storage* previous = quarantined_head.load(std::memory_order_relaxed);
    do { storage->quarantine_next = previous; }
    while (!quarantined_head.compare_exchange_weak(previous,storage.get(),std::memory_order_release,std::memory_order_relaxed));
}
inline bool current(const Storage& s) {
    return !s.quarantined && s.source && s.weights.valid(s.snapshot.model_epoch) && s.source->matches(s.snapshot);
}
template<class Layer> inline void common_layer(const Storage& s, unsigned layer, Layer* v) {
    const auto& p = s.scratch;
    v->hidden = layer ? p.moe_buffers[(layer-1u)%2u].output : p.embedding;
    v->residual = layer ? p.residual[(layer-1u)%2u] : p.zero_residual;
    v->normalized_input = p.normalized_input; v->input_residual = p.input_residual;
    v->moe_input = p.moe_input; v->output_residual = p.residual[layer%2u];
    v->moe_workspace = p.moe[layer%2u]; v->moe_workspace_bytes = qrt_sm121_mtp::moe_workspace_bytes(2u);
}
template<class Element> inline LinearLayerViews<Element> linear_layer(const Storage& s, unsigned layer) {
    LinearLayerViews<Element> v; common_layer(s,layer,&v);
    (void)s.weights.linear_weights(s.snapshot.model_epoch,layer,&v);
    const auto& p = s.scratch; const auto& cache = s.snapshot.linear[layer]; auto& l = v.linear;
    l.normalized_input = p.normalized_input; l.qkv = p.qkv; l.z = p.z; l.a = p.a; l.b = p.b;
    l.gated = p.linear_gated; l.output = p.linear_output;
    l.convolution.qkv = p.qkv; l.convolution.initial_ring = static_cast<const Element*>(cache.ring);
    l.convolution.silu = s.snapshot.tables.convolution_silu;
    l.convolution.staged_rings = reinterpret_cast<Element*>(p.rings[layer]);
    l.convolution.staged_convolution = p.convolution; l.convolution.first_position = s.snapshot.processed_tokens;
    l.recurrent = {p.convolution,p.a,p.b,cache.state,p.states[layer],p.core,cache.key_major};
    return v;
}
inline LinearLayerTables linear_tables(const TargetSnapshot& s, unsigned layer) {
    return {{ {s.tables.g[layer],s.tables.beta,s.tables.attention.exp2,s.tables.attention.rsqrt},
        s.tables.gated_silu },s.tables.moe};
}
inline AttentionLayerViews attention_layer(const Storage& s, unsigned layer) {
    AttentionLayerViews v; common_layer(s,layer,&v);
    (void)s.weights.attention_weights(s.snapshot.model_epoch,layer,&v);
    const auto& p = s.scratch; auto& a = v.attention;
    a.normalized_input = p.normalized_input; a.cache = s.snapshot.attention[layer];
    a.cache.staged = p.kv[layer]; a.first_position = s.snapshot.processed_tokens;
    a.q_projected = p.q_projected; a.kv_projected = p.kv_projected;
    a.q_norm = p.q_norm; a.k_norm = p.k_norm; a.queries = p.queries; a.gates = p.gates;
    a.staged_kv = p.kv[layer]; a.scores = p.scores; a.score_stride = p.score_stride;
    a.float_context = p.float_context; a.context = p.context; a.gated = p.attention_gated; a.output = p.attention_output;
    return v;
}

// A bad later layer must fail before even the embedding copies. The entire
// private allocation is also checked against EVERY borrowed read, preventing
// one layer's scratch or retained result from overwriting another layer's input.
inline bool valid(const Storage& s) {
    using recurrent_detail::Span;
    const auto& snap = s.snapshot; const auto& t = snap.tables;
    if (!current(s) || !snap.owner ||
        snap.processed_tokens > target_context_limit-2u || snap.processed_tokens+2u > s.capacity ||
        snap.current_token != s.inputs[0] ||
        s.inputs[0] >= qrt_sm121_mtp::head_vocabulary || s.inputs[1] >= qrt_sm121_mtp::head_vocabulary) return false;
    const Span workspace{s.device,s.bytes,256u};
    const auto read = [&](const Span& span) { return recurrent_detail::disjoint(workspace,span); };
    for (const auto& spec : target_weight_specs())
        if (!read({s.weights.tensor(snap.model_epoch,spec.layer,spec.role),spec.bytes(),2u})) return false;
    for (unsigned layer = 0; layer < target_layers; ++layer)
        if (!read({s.weights.shared_gate_up(snap.model_epoch,layer),size_t(1024u)*2048u*2u,2u})) return false;
    const Span tables[] = {
        {t.beta,65536u*4u,4u}, {t.gated_silu,65536u*4u,4u},
        {t.convolution_silu,qrt_sm121_silu::table_bytes,1u},
        {t.attention.rsqrt,qrt_sm121_rsqrt::table_bytes,1u}, {t.attention.exp2,qrt_sm121_exp2::table_bytes,1u},
        {t.attention.reciprocal,qrt_sm121_attention_rcp::table_bytes,1u},
        {t.attention.rope,size_t(t.attention.rope_rows)*64u*2u,2u}, {t.attention.sigmoid,65536u*2u,2u},
        {t.moe.silu,65536u*2u,2u}, {t.moe.sigmoid,65536u*2u,2u}, {t.moe.router_exp_fraction,8388608u*4u,4u}
    };
    for (const auto& span : tables) if (!read(span)) return false;
    for (unsigned layer = 0; layer < target_layers; ++layer) {
        if (layer % 4u == 3u) {
            if (snap.attention[layer].staged || snap.attention[layer].committed_tokens() != snap.processed_tokens ||
                !valid_attention_layer(attention_layer(s,layer),{t.attention,t.moe})) return false;
            for (const auto& span : cache_view_detail::history_spans(snap.attention[layer]))
                if (span.bytes && !read(span)) return false;
        } else {
            const auto& cache = snap.linear[layer];
            if ((cache.ring_element_bytes != 2u && cache.ring_element_bytes != 4u) ||
                !read({cache.state,state_elements*sizeof(float),alignof(float)}) ||
                !read({cache.ring,ring_elements*cache.ring_element_bytes,cache.ring_element_bytes}) ||
                !read({t.g[layer],32u*65536u*4u,4u})) return false;
            const auto tables_for_layer = linear_tables(snap,layer);
            if (cache.ring_element_bytes == 2u ? !valid_linear_layer(linear_layer<uint16_t>(s,layer),tables_for_layer)
                                             : !valid_linear_layer(linear_layer<float>(s,layer),tables_for_layer)) return false;
        }
    }
    return current(s);
}
} // namespace target_detail

// Copyable, immutable completed private result. Keep this owner alive through
// all copies/publication of accepted state. Selection does not modify live KV,
// recurrence, token history, streaming callbacks or MTP state.
class TargetResult {
public:
    bool ready() const { return storage_ && storage_->complete && target_detail::current(*storage_); }
    const TargetHostResult* host() const { return ready() ? storage_->host : nullptr; }
    const TargetSnapshot* frontier() const { return ready() ? &storage_->snapshot : nullptr; }
    const std::array<uint32_t,2>* inputs() const { return ready() ? &storage_->inputs : nullptr; }
    const uint16_t* vocabulary_logits() const { return ready() ? storage_->scratch.vocabulary_logits : nullptr; }
    size_t allocated_bytes() const { return storage_ ? storage_->bytes : 0; }
    TargetLinearSelection linear(unsigned layer, unsigned rows) const {
        if (!ready() || layer >= target_layers || layer%4u == 3u || rows < 1u || rows > 2u) return {};
        const auto& s = *storage_; const auto& cache = s.snapshot.linear[layer];
        return {s.scratch.states[layer]+size_t(rows-1u)*state_elements,
            s.scratch.rings[layer]+size_t(rows-1u)*ring_elements*cache.ring_element_bytes,
            cache.ring_element_bytes,s.snapshot.processed_tokens,rows,cache.key_major};
    }
    AttentionSelection attention(unsigned layer, unsigned rows) const {
        if (!ready() || layer >= target_layers || layer%4u != 3u || rows < 1u || rows > 2u) return {};
        return {storage_->scratch.kv[layer],storage_->snapshot.processed_tokens,rows};
    }
private:
    friend class Target;
    std::shared_ptr<target_detail::Storage> storage_;
};

// The complete two-row target producer. Kernel arithmetic belongs to the
// individually qualified layer/head implementations. This owner only composes
// them and establishes completion/lifetime; actual acceptance is request-owned.
class Target {
public:
    Target() = default;
    Target(const Target&) = delete;
    Target& operator=(const Target&) = delete;
    TargetStep evaluate(const ModelWeightBinding& weights, std::shared_ptr<const TargetStateSource> source,
        const std::array<uint32_t,2>& inputs, TargetResult* output,
        unsigned maximum_blocks = 1024u, hipStream_t stream = nullptr) {
        if (output) *output = {};
        if (terminal_ != hipSuccess) return {terminal_,"target_quarantined",true};
        if (!output || !source || !maximum_blocks || maximum_blocks > 4096u) return invalid("target_contract");
        TargetSnapshot snapshot;
        try {
            if (!source->snapshot(&snapshot)) return invalid("target_snapshot");
        } catch (const std::bad_alloc&) { return {hipErrorOutOfMemory,"target_snapshot"}; }
          catch (...) { return invalid("target_snapshot"); }
        if (snapshot.processed_tokens > target_context_limit-2u || !weights.valid(snapshot.model_epoch) ||
            !source->matches(snapshot) || snapshot.current_token != inputs[0] ||
            inputs[0] >= qrt_sm121_mtp::head_vocabulary || inputs[1] >= qrt_sm121_mtp::head_vocabulary)
            return invalid("target_frontier");
        const auto reserved = reserve(snapshot.processed_tokens+2u);
        if (reserved != hipSuccess) return {reserved,"target_workspace"};
        auto& s = *storage_;
        s.complete = false; s.weights = weights; s.source = std::move(source); s.snapshot = snapshot; s.inputs = inputs;
        if (!target_detail::valid(s)) return invalid("target_graph");
        *s.host = {};
        auto& p = s.scratch;
        const auto fail = [&](hipError_t error, const char* stage) {
            const auto completed = hipStreamSynchronize(stream);
            if (completed != hipSuccess) {
                target_detail::quarantine(storage_); terminal_ = completed;
                return TargetStep{completed,stage,true};
            }
            return TargetStep{error,stage};
        };
        const uint16_t* embedding = weights.tensor(snapshot.model_epoch,target_layers,WeightRole::Embedding);
        hipError_t status = hipMemsetAsync(p.zero_residual,0,8192u,stream);
        if (status != hipSuccess) return fail(status,"target_zero_residual");
        for (unsigned row = 0; row < 2u; ++row) {
            status = hipMemcpyAsync(p.embedding+size_t(row)*2048u,embedding+size_t(inputs[row])*2048u,
                4096u,hipMemcpyDeviceToDevice,stream);
            if (status != hipSuccess) return fail(status,"target_embedding");
        }
        for (unsigned layer = 0; layer < target_layers; ++layer) {
            if (layer%4u == 3u)
                status = launch_attention_layer(target_detail::attention_layer(s,layer),
                    {snapshot.tables.attention,snapshot.tables.moe},maximum_blocks,stream);
            else if (snapshot.linear[layer].ring_element_bytes == 2u)
                status = launch_linear_layer(target_detail::linear_layer<uint16_t>(s,layer),
                    target_detail::linear_tables(snapshot,layer),maximum_blocks,stream);
            else
                status = launch_linear_layer(target_detail::linear_layer<float>(s,layer),
                    target_detail::linear_tables(snapshot,layer),maximum_blocks,stream);
            if (status != hipSuccess) return fail(status,"target_layer");
            // Save each flag before that MoE workspace is reused two layers later.
            status = hipMemcpyAsync(p.invalid+layer,p.moe_buffers[layer%2u].invalid,
                sizeof(uint32_t),hipMemcpyDeviceToDevice,stream);
            if (status != hipSuccess) return fail(status,"target_layer_validity");
        }
        status = qrt_sm121_mtp::launch_residual_normalize(p.moe_buffers[1].output,p.residual[1],
            weights.tensor(snapshot.model_epoch,target_layers,WeightRole::FinalNorm),snapshot.tables.attention.rsqrt,
            2u,p.final_norm,p.final_residual,stream);
        if (status != hipSuccess) return fail(status,"target_final_norm");
        status = launch_target_head(weights.tensor(snapshot.model_epoch,target_layers,WeightRole::Head),p.final_norm,
            p.vocabulary_logits,p.tokens,p.values,p.invalid+target_layers,maximum_blocks,stream);
        if (status != hipSuccess) return fail(status,"target_head");
        status = hipMemcpyAsync(s.host->invalid.data(),p.invalid,sizeof(s.host->invalid),hipMemcpyDeviceToHost,stream);
        if (status != hipSuccess) return fail(status,"target_invalid_download");
        status = hipMemcpyAsync(s.host->tokens.data(),p.tokens,sizeof(s.host->tokens),hipMemcpyDeviceToHost,stream);
        if (status != hipSuccess) return fail(status,"target_token_download");
        status = hipMemcpyAsync(s.host->logits.data(),p.values,sizeof(s.host->logits),hipMemcpyDeviceToHost,stream);
        if (status != hipSuccess) return fail(status,"target_logit_download");
        status = hipMemcpyAsync(s.host->normalized.data(),p.final_norm,sizeof(s.host->normalized),hipMemcpyDeviceToHost,stream);
        if (status != hipSuccess) return fail(status,"target_hidden_download");
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) {
            target_detail::quarantine(storage_); terminal_ = status;
            return {status,"target_completion",true};
        }
        if (!target_detail::current(s)) return invalid("target_completed_frontier");
        for (const auto flag : s.host->invalid) if (flag) return invalid("target_numerical_validity");
        for (unsigned row = 0; row < 2u; ++row)
            if (s.host->tokens[row] >= qrt_sm121_mtp::head_vocabulary || !std::isfinite(s.host->logits[row]))
                return invalid("target_sample_validity");
        for (const auto value : s.host->normalized)
            if (!std::isfinite(qrt_sm121_q1::widen(value))) return invalid("target_hidden_validity");
        s.complete = true; output->storage_ = storage_;
        return {};
    }
    bool quarantined() const { return terminal_ != hipSuccess; }
private:
    static TargetStep invalid(const char* stage) { return {hipErrorInvalidValue,stage}; }
    hipError_t reserve(unsigned capacity) {
        if (storage_ && storage_.use_count() == 1 && storage_->capacity >= capacity) return hipSuccess;
        std::shared_ptr<target_detail::Storage> next;
        try { next = std::make_shared<target_detail::Storage>(); }
        catch (...) { return hipErrorOutOfMemory; }
        next->capacity = (capacity+31u)&~31u;
        next->bytes = target_detail::layout(next->capacity,nullptr,nullptr);
        if (!next->bytes) return hipErrorInvalidValue;
        auto status = hipMalloc(reinterpret_cast<void**>(&next->device),next->bytes);
        if (status != hipSuccess) return status;
        if (target_detail::layout(next->capacity,next->device,&next->scratch) != next->bytes) return hipErrorInvalidValue;
        status = hipHostMalloc(reinterpret_cast<void**>(&next->host),sizeof(TargetHostResult));
        if (status != hipSuccess) return status;
        storage_ = std::move(next);
        return hipSuccess;
    }
    std::shared_ptr<target_detail::Storage> storage_;
    hipError_t terminal_ = hipSuccess;
};
} // namespace qrt_sm121_q2
