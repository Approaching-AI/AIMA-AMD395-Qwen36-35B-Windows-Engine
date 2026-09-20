#pragma once
#include "sm121_q2_publication.h"
#include "sm121_q2_cache_lifetime.h"
#include <algorithm>
#include <limits>

namespace qrt_sm121_q2 {
namespace resident_cache_detail {
inline bool same_plane(const CachePlane& a, const CachePlane& b) {
    return a.keys == b.keys && a.values == b.values && a.tokens == b.tokens &&
        a.capacity == b.capacity && a.stride == b.stride && a.element_bytes == b.element_bytes;
}
inline bool same_snapshot(const TargetSnapshot& a, const TargetSnapshot& b) {
    if (a.owner != b.owner || a.generation != b.generation || a.model_epoch != b.model_epoch ||
        a.processed_tokens != b.processed_tokens || a.current_token != b.current_token) return false;
    for (unsigned i = 0; i < target_layers; ++i) {
        const auto& x = a.linear[i]; const auto& y = b.linear[i];
        if (x.state != y.state || x.ring != y.ring || x.ring_element_bytes != y.ring_element_bytes ||
            x.key_major != y.key_major || !same_plane(a.attention[i].prefix,b.attention[i].prefix) ||
            !same_plane(a.attention[i].decoded,b.attention[i].decoded) ||
            a.attention[i].staged != b.attention[i].staged) return false;
    }
    const auto& x = a.tables; const auto& y = b.tables;
    return x.g == y.g && x.beta == y.beta && x.gated_silu == y.gated_silu &&
        x.convolution_silu == y.convolution_silu && x.attention.rsqrt == y.attention.rsqrt &&
        x.attention.exp2 == y.attention.exp2 && x.attention.reciprocal == y.attention.reciprocal &&
        x.attention.rope == y.attention.rope && x.attention.rope_rows == y.attention.rope_rows &&
        x.attention.sigmoid == y.attention.sigmoid && x.moe.silu == y.moe.silu &&
        x.moe.sigmoid == y.moe.sigmoid && x.moe.router_exp_fraction == y.moe.router_exp_fraction;
}
inline bool at(const void* pointer, const void* allocation, size_t offset) {
    const auto base = reinterpret_cast<uintptr_t>(allocation);
    return allocation && offset <= UINTPTR_MAX-base && reinterpret_cast<uintptr_t>(pointer) == base+offset;
}
} // namespace resident_cache_detail

// Instantiated with the provider's actual session and element-kind declarations.
// The saved metadata pins its immutable checkpoint owners; the lifetime pins
// the raw cache allocations. Every candidate uses the exact processed inputs.
template<class Session, class ElementKind>
class ResidentCacheOwner final : public TargetCacheOwner {
public:
    static std::shared_ptr<ResidentCacheOwner> create(Session& session,
        const std::atomic<uint64_t>& epoch, const TargetTables& tables,
        std::shared_ptr<const ResidentCacheLifetime> lifetime) {
        if (!lifetime || !lifetime->ready(&session) || !lifetime->rollback_ready(&session)) return {};
        auto result = std::shared_ptr<ResidentCacheOwner>(new ResidentCacheOwner(session,epoch,tables,std::move(lifetime)));
        if (!result->describe(session,&result->frontier_) || !result->matches(result->frontier_)) return {};
        return result;
    }
    bool snapshot(TargetSnapshot* output) const override {
        if (!output || !matches(frontier_)) return false;
        *output = frontier_; return true;
    }
    bool matches(const TargetSnapshot& other) const noexcept override {
        if (!lifetime_->ready(owner_) || epoch_->load(std::memory_order_acquire) != frontier_.model_epoch ||
            owner_->owner_engine != saved_.owner_engine || owner_->model_dir != saved_.model_dir ||
            owner_->generation != saved_.generation || owner_->prefix_tokens != saved_.prefix_tokens ||
            owner_->committed_decode_token_count != saved_.committed_decode_token_count ||
            owner_->current_token_id != saved_.current_token_id || !owner_->current_token_valid ||
            !owner_->valid || !owner_->provider_completed || !owner_->route_active ||
            owner_->native_mtp_processed_inputs != saved_.native_mtp_processed_inputs ||
            !resident_cache_detail::same_snapshot(other,frontier_)) return false;
        for (unsigned i = 0; i < target_layers; ++i) {
            if (i%4u == 3u) {
                const auto& a = owner_->full_attention_layers[i]; const auto& b = saved_.full_attention_layers[i];
                if (!a.valid || a.device_allocation != b.device_allocation || a.device_k != b.device_k || a.device_v != b.device_v ||
                    a.device_decode_tail_allocation != b.device_decode_tail_allocation ||
                    a.device_decode_tail_k != b.device_decode_tail_k || a.device_decode_tail_v != b.device_decode_tail_v ||
                    a.k_bytes != b.k_bytes || a.v_bytes != b.v_bytes || a.decode_tail_k_bytes != b.decode_tail_k_bytes ||
                    a.decode_tail_v_bytes != b.decode_tail_v_bytes || a.decode_tail_capacity_tokens != b.decode_tail_capacity_tokens ||
                    a.decode_tail_token_count != b.decode_tail_token_count || a.history_tokens != b.history_tokens ||
                    a.prefill_reserved_tokens != b.prefill_reserved_tokens || a.decode_tail_contiguous != b.decode_tail_contiguous ||
                    a.element_kind != b.element_kind) return false;
            } else {
                const auto& a = owner_->linear_layers[i]; const auto& b = saved_.linear_layers[i];
                if (!a.valid || a.device_allocation != b.device_allocation || a.device_recurrent_state != b.device_recurrent_state ||
                    a.device_qkv_ring != b.device_qkv_ring || a.recurrent_state_bytes != b.recurrent_state_bytes ||
                    a.qkv_ring_bytes != b.qkv_ring_bytes || a.prefix_tokens != b.prefix_tokens ||
                    a.decode_qkv_token_count != b.decode_qkv_token_count || a.decode_recurrent_token_count != b.decode_recurrent_token_count ||
                    a.qkv_element_kind != b.qkv_element_kind || a.recurrent_state_key_major != b.recurrent_state_key_major) return false;
            }
        }
        return true;
    }
    bool prepare_publication(const TargetSnapshot& snapshot, unsigned rows) const override {
        if (rows < 1u || rows > 2u || !matches(snapshot) || !lifetime_->rollback_ready(owner_) ||
            rows > target_context_limit-snapshot.processed_tokens) return false;
        for (unsigned i = 3u; i < target_layers; i += 4u) {
            const auto& tail = snapshot.attention[i].decoded;
            if (tail.tokens > tail.capacity || rows > tail.capacity-tail.tokens) return false;
        }
        // Both reserves happen before CachePublisher submits its first write.
        const size_t count = snapshot.processed_tokens+rows;
        owner_->native_mtp_processed_inputs.reserve(count);
        saved_.native_mtp_processed_inputs.reserve(count);
        prepared_rows_ = rows;
        return true;
    }
    void invalidate() const noexcept override { owner_->valid = false; owner_->route_active = false; }
    void quarantine() const noexcept override { lifetime_->quarantine(); invalidate(); }

    // Request::prepare has already selected accepted rows. CachePublisher must
    // have completed before host state advances. The enclosing Shadow remains
    // responsible for rollback if the paired drafter receipt cannot commit.
    bool commit_metadata(const TargetResult& result, unsigned rows,
        const std::array<uint32_t,2>& outputs) noexcept {
        if (rows < 1u || rows > 2u || rows != prepared_rows_ || !result.owned_by(*this) ||
            !result.cache_published() || !result.frontier() || !matches(*result.frontier()) ||
            !lifetime_->rollback_ready(owner_)) return false;
        const auto* host = result.host(); const auto* inputs = result.inputs();
        const size_t new_count = frontier_.processed_tokens+rows;
        if (!host || !inputs || owner_->native_mtp_processed_inputs.capacity() < new_count ||
            saved_.native_mtp_processed_inputs.capacity() < new_count) return false;
        for (unsigned i = 0; i < rows; ++i) if (outputs[i] != host->tokens[i]) return false;
        const auto advance = [&](Session& session) {
            for (unsigned i = 0; i < rows; ++i) session.native_mtp_processed_inputs.push_back((*inputs)[i]);
            for (unsigned i = 0; i < target_layers; ++i) {
                if (i%4u == 3u) session.full_attention_layers[i].decode_tail_token_count += rows;
                else {
                    session.linear_layers[i].decode_qkv_token_count += rows;
                    session.linear_layers[i].decode_recurrent_token_count += rows;
                }
            }
            session.committed_decode_token_count += rows;
            session.current_token_id = outputs[rows-1u]; session.current_token_valid = true;
            session.last_decode_top2_valid = false;
            std::copy_n(host->normalized.data()+size_t(rows-1u)*2048u,2048u,session.mtp_target_hidden_bf16.data());
            session.mtp_target_hidden_position = new_count-1u;
            session.mtp_target_hidden_output_token_id = session.current_token_id;
            session.mtp_target_hidden_valid = true; session.mtp_target_hidden_device_only = false;
        };
        advance(*owner_); advance(saved_);
        // Keep the saved checkpoint pinned until this owner is retired. It no
        // longer describes the live target; Request::save will replace it.
        owner_->native_mtp_checkpoint = {};
        frontier_.processed_tokens += rows; frontier_.current_token = outputs[rows-1u];
        for (unsigned i = 3u; i < target_layers; i += 4u) frontier_.attention[i].decoded.tokens += rows;
        prepared_rows_ = 0;
        return true;
    }
private:
    ResidentCacheOwner(Session& session, const std::atomic<uint64_t>& epoch, const TargetTables& tables,
        std::shared_ptr<const ResidentCacheLifetime> lifetime)
        : owner_(&session), saved_(session), epoch_(&epoch), tables_(tables), lifetime_(std::move(lifetime)) {}

    static unsigned element_bytes(ElementKind kind) {
        return kind == ElementKind::kBf16 ? 2u : kind == ElementKind::kF32 ? 4u : 0u;
    }
    bool describe(const Session& s, TargetSnapshot* out) const {
        using recurrent_detail::Span;
        using resident_cache_detail::at;
        if (!out || !s.owner_engine || !s.generation || !s.valid || !s.route_active || !s.provider_completed ||
            !s.current_token_valid || s.current_token_id >= qrt_sm121_mtp::head_vocabulary || !s.prefix_tokens ||
            s.prefix_tokens > target_context_limit-2u || s.committed_decode_token_count > target_context_limit-2u-s.prefix_tokens ||
            s.native_mtp_processed_inputs.size() != s.prefix_tokens+s.committed_decode_token_count ||
            s.q2_prefetched_valid || s.dflash_prefetched_valid || s.dflash_decode_transaction_pending) return false;
        for (const auto token : s.native_mtp_processed_inputs) if (token >= qrt_sm121_mtp::head_vocabulary) return false;
        TargetSnapshot result;
        result.owner = &s; result.generation = s.generation; result.model_epoch = epoch_->load(std::memory_order_acquire);
        result.processed_tokens = static_cast<unsigned>(s.native_mtp_processed_inputs.size());
        result.current_token = s.current_token_id; result.tables = tables_;
        if (!result.model_epoch) return false;
        std::array<Span,60> allocations{}; size_t allocation_count = 0;
        const auto allocation = [&](const void* base, size_t bytes, unsigned alignment) {
            const Span next{base,bytes,alignment};
            if (!recurrent_detail::valid_span(next) || allocation_count == allocations.size()) return false;
            for (size_t i = 0; i < allocation_count; ++i)
                if (!recurrent_detail::disjoint(next,allocations[i])) return false;
            allocations[allocation_count++] = next; return true;
        };
        for (unsigned i = 0; i < target_layers; ++i) {
            if (i%4u == 3u) {
                const auto& a = s.full_attention_layers[i]; const unsigned bytes = element_bytes(a.element_kind);
                if (!a.valid || !bytes || a.prefill_reserved_tokens || a.history_tokens != s.prefix_tokens ||
                    a.decode_tail_token_count != s.committed_decode_token_count || a.decode_tail_capacity_tokens < a.decode_tail_token_count ||
                    a.decode_tail_capacity_tokens > target_context_limit-s.prefix_tokens ||
                    a.decode_tail_capacity_tokens-a.decode_tail_token_count < 2u ||
                    a.k_bytes != s.prefix_tokens*512u*bytes || a.v_bytes != a.k_bytes ||
                    a.decode_tail_k_bytes != a.decode_tail_capacity_tokens*512u*bytes || a.decode_tail_v_bytes != a.decode_tail_k_bytes ||
                    !at(a.device_k,a.device_allocation,0u)) return false;
                if (a.decode_tail_contiguous) {
                    if (a.device_decode_tail_allocation || !at(a.device_v,a.device_allocation,a.k_bytes+a.decode_tail_k_bytes) ||
                        !at(a.device_decode_tail_k,a.device_k,a.k_bytes) || !at(a.device_decode_tail_v,a.device_v,a.v_bytes) ||
                        !allocation(a.device_allocation,a.k_bytes+a.v_bytes+a.decode_tail_k_bytes+a.decode_tail_v_bytes,bytes)) return false;
                } else if (!at(a.device_v,a.device_allocation,a.k_bytes) ||
                    !at(a.device_decode_tail_k,a.device_decode_tail_allocation,0u) ||
                    !at(a.device_decode_tail_v,a.device_decode_tail_allocation,a.decode_tail_k_bytes) ||
                    !allocation(a.device_allocation,a.k_bytes+a.v_bytes,bytes) ||
                    !allocation(a.device_decode_tail_allocation,a.decode_tail_k_bytes+a.decode_tail_v_bytes,bytes)) return false;
                result.attention[i] = {{a.device_k,a.device_v,static_cast<unsigned>(s.prefix_tokens),static_cast<unsigned>(s.prefix_tokens),512u,bytes},
                    {a.device_decode_tail_k,a.device_decode_tail_v,static_cast<unsigned>(a.decode_tail_token_count),
                        static_cast<unsigned>(a.decode_tail_capacity_tokens),512u,bytes},nullptr};
            } else {
                const auto& a = s.linear_layers[i]; const unsigned bytes = element_bytes(a.qkv_element_kind);
                if (!a.valid || !bytes || a.prefix_tokens != s.prefix_tokens ||
                    a.decode_qkv_token_count != s.committed_decode_token_count || a.decode_recurrent_token_count != s.committed_decode_token_count ||
                    a.recurrent_state_bytes != state_elements*sizeof(float) || a.qkv_ring_bytes != ring_elements*bytes ||
                    !at(a.device_recurrent_state,a.device_allocation,0u) || !at(a.device_qkv_ring,a.device_allocation,a.recurrent_state_bytes) ||
                    !allocation(a.device_allocation,a.recurrent_state_bytes+a.qkv_ring_bytes,4u)) return false;
                result.linear[i] = {a.device_recurrent_state,a.device_qkv_ring,bytes,a.recurrent_state_key_major};
            }
        }
        *out = result; return true;
    }
    Session* owner_;
    mutable Session saved_;
    const std::atomic<uint64_t>* epoch_;
    TargetTables tables_;
    std::shared_ptr<const ResidentCacheLifetime> lifetime_;
    TargetSnapshot frontier_;
    mutable unsigned prepared_rows_ = 0;
};
} // namespace qrt_sm121_q2
