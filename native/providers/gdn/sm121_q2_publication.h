#pragma once
#include "sm121_q2_target.h"

namespace qrt_sm121_q2 {
// Implemented by the actual mutable session under its transaction lock. The
// snapshot pins the writable state/ring and decode-tail allocations as well as
// immutable prefix/history storage. The caller must reserve host metadata and
// provide rollback before granting permission. No token or callback is published
// by this cache-copy boundary.
class TargetCacheOwner : public TargetStateSource {
public:
    virtual bool prepare_publication(const TargetSnapshot&, unsigned accepted_rows) const = 0;
    // A completed failed copy may already have changed part of the cache.
    // Only the enclosing transaction may restore this invalidated owner.
    virtual void invalidate() const noexcept = 0;
};

namespace publication_detail {
#if defined(__HIPCC__) || defined(__CUDACC__)
__global__ void widen_plane(const uint16_t* input, float* output, unsigned rows, unsigned stride) {
    const unsigned index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < rows*512u) {
        union { uint32_t bits; float value; } converted;
        converted.bits = uint32_t(input[size_t(index/512u)*1024u+index%512u]) << 16u;
        output[size_t(index/512u)*stride+index%512u] = converted.value;
    }
}
inline hipError_t copy_plane(const uint16_t* input, void* output, unsigned rows,
    unsigned stride, unsigned element_bytes, hipStream_t stream) {
    if (element_bytes == 2u)
        return hipMemcpy2DAsync(output,size_t(stride)*2u,input,2048u,1024u,rows,
            hipMemcpyDeviceToDevice,stream);
    hipLaunchKernelGGL(widen_plane,dim3(rows*2u),dim3(256u),0,stream,
        input,static_cast<float*>(output),rows,stride);
    return hipGetLastError();
}
#else
hipError_t copy_plane(const uint16_t*, void*, unsigned, unsigned, unsigned, hipStream_t);
#endif

using recurrent_detail::Span;
struct Plan {
    std::array<Span,80> writes{};
    std::array<const void*,80> sources{};
    std::array<unsigned,80> layers{};
    size_t count = 0;
};
inline bool make_plan(const target_detail::Storage& s, unsigned rows, Plan* output) {
    if (!output || rows < 1u || rows > 2u) return false;
    Plan p;
    const auto add = [&](void* destination, size_t bytes, unsigned alignment,
        const void* source, unsigned layer) {
        p.writes[p.count] = {destination,bytes,alignment};
        p.sources[p.count] = source; p.layers[p.count++] = layer;
    };
    const auto& snap = s.snapshot;
    for (unsigned layer = 0; layer < target_layers; ++layer) {
        if (layer%4u == 3u) {
            const auto& tail = snap.attention[layer].decoded;
            // Mutable native tails use compact separate planes. Interleaved
            // captured histories remain supported as read-only Target inputs.
            if (!cache_view_detail::valid_plane(tail) || tail.stride != 512u || tail.tokens > tail.capacity ||
                rows > tail.capacity-tail.tokens) return false;
            const size_t offset = size_t(tail.tokens)*tail.stride*tail.element_bytes;
            const size_t bytes = (size_t(rows-1u)*tail.stride+512u)*tail.element_bytes;
            add(const_cast<unsigned char*>(static_cast<const unsigned char*>(tail.keys))+offset,
                bytes,tail.element_bytes,s.scratch.kv[layer],layer);
            add(const_cast<unsigned char*>(static_cast<const unsigned char*>(tail.values))+offset,
                bytes,tail.element_bytes,s.scratch.kv[layer]+512u,layer);
        } else {
            const auto& cache = snap.linear[layer];
            add(const_cast<float*>(cache.state),state_elements*sizeof(float),4u,
                s.scratch.states[layer]+size_t(rows-1u)*state_elements,layer);
            add(const_cast<void*>(cache.ring),ring_elements*cache.ring_element_bytes,cache.ring_element_bytes,
                s.scratch.rings[layer]+size_t(rows-1u)*ring_elements*cache.ring_element_bytes,layer);
        }
    }
    if (p.count != 80u) return false;
    // All destination ranges are checked before the first submission, including
    // cross-layer aliases and the entire immutable model/table/prefix extents.
    for (size_t i = 0; i < p.count; ++i) {
        const auto& write = p.writes[i];
        if (!recurrent_detail::valid_span(write) ||
            !recurrent_detail::disjoint(write,{s.device,s.bytes,256u})) return false;
        for (size_t j = 0; j < i; ++j)
            if (!recurrent_detail::disjoint(write,p.writes[j])) return false;
        for (const auto& spec : target_weight_specs())
            if (!recurrent_detail::disjoint(write,{s.weights.tensor(snap.model_epoch,spec.layer,spec.role),spec.bytes(),2u}))
                return false;
        for (unsigned layer = 0; layer < target_layers; ++layer) {
            if (!recurrent_detail::disjoint(write,{s.weights.shared_gate_up(snap.model_epoch,layer),
                    size_t(1024u)*2048u*2u,2u})) return false;
            if (layer%4u != 3u) {
                if (!recurrent_detail::disjoint(write,{snap.tables.g[layer],size_t(32u)*65536u*4u,4u})) return false;
            } else {
                const auto& prefix = snap.attention[layer].prefix;
                for (const auto& span : cache_view_detail::history_spans({prefix,{},{}}))
                    if (span.bytes && !recurrent_detail::disjoint(write,span)) return false;
                const auto& tail = snap.attention[layer].decoded;
                if (tail.tokens) {
                    const size_t bytes = (size_t(tail.tokens-1u)*tail.stride+512u)*tail.element_bytes;
                    if (!recurrent_detail::disjoint(write,{tail.keys,bytes,tail.element_bytes}) ||
                        !recurrent_detail::disjoint(write,{tail.values,bytes,tail.element_bytes})) return false;
                }
            }
        }
        const auto& t = snap.tables;
        const Span tables[] = {
            {t.beta,65536u*4u,4u}, {t.gated_silu,65536u*4u,4u},
            {t.convolution_silu,qrt_sm121_silu::table_bytes,1u},
            {t.attention.rsqrt,qrt_sm121_rsqrt::table_bytes,1u}, {t.attention.exp2,qrt_sm121_exp2::table_bytes,1u},
            {t.attention.reciprocal,qrt_sm121_attention_rcp::table_bytes,1u},
            {t.attention.rope,size_t(t.attention.rope_rows)*64u*2u,2u}, {t.attention.sigmoid,65536u*2u,2u},
            {t.moe.silu,65536u*2u,2u}, {t.moe.sigmoid,65536u*2u,2u}, {t.moe.router_exp_fraction,8388608u*4u,4u}
        };
        for (const auto& table : tables)
            if (!recurrent_detail::disjoint(write,table)) return false;
    }
    *output = p;
    return true;
}
} // namespace publication_detail

class CachePublisher final {
public:
    static TargetStep publish(const TargetResult& result, const TargetCacheOwner& owner,
        unsigned rows, hipStream_t stream = nullptr) {
        if (!result.ready() || result.storage_->source.get() != &owner ||
            result.storage_->publication_started || rows < 1u || rows > 2u)
            return {hipErrorInvalidValue,"target_publication_contract"};
        auto& s = *result.storage_;
        publication_detail::Plan plan;
        try {
            if (!owner.prepare_publication(s.snapshot,rows) || !result.ready() ||
                !publication_detail::make_plan(s,rows,&plan))
                return {hipErrorInvalidValue,"target_publication_plan"};
        } catch (const std::bad_alloc&) { return {hipErrorOutOfMemory,"target_publication_plan"}; }
          catch (...) { return {hipErrorInvalidValue,"target_publication_plan"}; }
        s.publication_started = true;
        hipError_t status = hipSuccess;
        for (size_t i = 0; i < plan.count && status == hipSuccess; ++i) {
            const unsigned layer = plan.layers[i];
            void* destination = const_cast<void*>(plan.writes[i].pointer);
            if (layer%4u == 3u) {
                const auto& tail = s.snapshot.attention[layer].decoded;
                status = publication_detail::copy_plane(static_cast<const uint16_t*>(plan.sources[i]),
                    destination,rows,tail.stride,tail.element_bytes,stream);
            } else {
                status = hipMemcpyAsync(destination,plan.sources[i],plan.writes[i].bytes,hipMemcpyDeviceToDevice,stream);
            }
        }
        const hipError_t completed = hipStreamSynchronize(stream);
        if (completed != hipSuccess) {
            (void)result.quarantine_borrower(completed);
            return {completed,"target_publication_completion",true};
        }
        if (status != hipSuccess || !result.ready()) {
            owner.invalidate();
            return {status != hipSuccess ? status : hipErrorInvalidValue,"target_publication_copy"};
        }
        s.cache_published = true;
        return {};
    }
};
} // namespace qrt_sm121_q2
