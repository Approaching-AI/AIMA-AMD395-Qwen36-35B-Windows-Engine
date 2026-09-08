#ifndef QRT_PROJECTION_OUTPUT_POLICY_H
#define QRT_PROJECTION_OUTPUT_POLICY_H

namespace qrt_projection_output {

// A BF16 consumer can omit its ordinary F32 staging buffer, but an override
// that writes F32 still needs that buffer and a conversion to the consumer.
constexpr bool needs_f32_buffer(bool consumer_needs_f32, bool producer_writes_f32) {
    return consumer_needs_f32 || producer_writes_f32;
}

constexpr bool valid_buffers(const void *f32, const void *bf16, bool writes_bf16) {
    return f32 != nullptr && (!writes_bf16 || bf16 != nullptr);
}

}  // namespace qrt_projection_output

#endif
