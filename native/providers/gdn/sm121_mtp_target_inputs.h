#pragma once
#include <atomic>
#include <cstring>
#include <memory>
#include <new>
#include "sm121_mtp_drafter.h"
#include "../mtp_target_rows.h"
#include "../mtp_decode_rows.h"

namespace qrt_sm121_mtp {
namespace mtp_target_input_detail {
struct Storage {
    unsigned char* host = nullptr;
    unsigned char* device = nullptr;
    size_t bytes = 0;
    Storage* quarantine_next = nullptr;
    ~Storage() {
        if (device) (void)hipFree(device);
        if (host) (void)hipHostFree(host);
    }
};
// An unknown completion cannot allocate recovery bookkeeping. The existing
// node remains rooted until process teardown, including its pinned host copy.
inline std::atomic<Storage*> quarantined_head{nullptr};
inline void quarantine(Storage* storage) {
    if (!storage) return;
    Storage* previous = quarantined_head.load(std::memory_order_relaxed);
    do { storage->quarantine_next = previous; }
    while (!quarantined_head.compare_exchange_weak(previous, storage,
        std::memory_order_release, std::memory_order_relaxed));
}
} // namespace mtp_target_input_detail

// Own the transfer independently of the target's host vectors. The upload is
// completed before invoking the drafter, so even its early contract rejection
// cannot abandon an in-flight H2D copy. Request operations are serialized.
class TargetInputs final {
public:
    TargetInputs() = default;
    TargetInputs(const TargetInputs&) = delete;
    TargetInputs& operator=(const TargetInputs&) = delete;

    PromptStep append(Drafter& drafter, const qrt_mtp_target_rows::PrefillRows& batch,
        bool split1024_pre_fc_norm, uint64_t epoch, hipStream_t stream = nullptr,
        unsigned maximum_blocks = 1024u) {
        return append_completed(drafter, batch, batch.published(), split1024_pre_fc_norm,
            epoch, stream, maximum_blocks);
    }
    PromptStep append(Drafter& drafter, const qrt_mtp_target_rows::DecodeRows& batch,
        uint64_t epoch, hipStream_t stream = nullptr, unsigned maximum_blocks = 1024u) {
        return append_completed(drafter, batch, batch.completed(), false, epoch, stream, maximum_blocks);
    }
    bool quarantined() const { return terminal_ != hipSuccess; }
    size_t allocated_bytes() const { return storage_ ? 2u * storage_->bytes : 0u; }

private:
    template<class Rows>
    PromptStep append_completed(Drafter& drafter, const Rows& batch, bool completed_rows,
        bool split1024_pre_fc_norm, uint64_t epoch, hipStream_t stream, unsigned maximum_blocks) {
        if (terminal_ != hipSuccess)
            return {terminal_, "target_input_quarantined", drafter.retained_tokens(), true};
        if (!completed_rows || !batch.rows() ||
            batch.rows() > qrt_mtp_target_rows::maximum_batch_rows ||
            batch.first_position() >= 262144u || batch.rows() > 262144u - batch.first_position() ||
            batch.hidden().size() != batch.rows() * qrt_mtp_target_rows::hidden_width ||
            batch.shifted_tokens().size() != batch.rows())
            return {hipErrorInvalidValue, "published_target_input_contract", drafter.retained_tokens()};
        const size_t hidden_bytes = batch.hidden().size() * sizeof(uint16_t);
        const size_t token_bytes = batch.shifted_tokens().size() * sizeof(uint32_t);
        const size_t bytes = hidden_bytes + token_bytes;
        if (!storage_ || storage_->bytes < bytes) {
            std::unique_ptr<mtp_target_input_detail::Storage> next(
                new (std::nothrow) mtp_target_input_detail::Storage);
            if (!next) return {hipErrorOutOfMemory, "target_input_owner", drafter.retained_tokens()};
            hipError_t status = hipHostMalloc(reinterpret_cast<void**>(&next->host), bytes);
            if (status == hipSuccess)
                status = hipMalloc(reinterpret_cast<void**>(&next->device), bytes);
            if (status != hipSuccess) return {status, "target_input_allocation", drafter.retained_tokens()};
            next->bytes = bytes;
            storage_.swap(next);
        }
        std::memcpy(storage_->host, batch.hidden().data(), hidden_bytes);
        std::memcpy(storage_->host + hidden_bytes, batch.shifted_tokens().data(), token_bytes);
        const hipError_t copied = hipMemcpyAsync(storage_->device, storage_->host, bytes,
            hipMemcpyHostToDevice, stream);
        const hipError_t completed = hipStreamSynchronize(stream);
        if (completed != hipSuccess) {
            quarantine(completed);
            return {completed, "target_input_completion", drafter.retained_tokens(), true};
        }
        if (copied != hipSuccess) return {copied, "target_input_copy", drafter.retained_tokens()};
        const auto result = drafter.append_target(reinterpret_cast<const uint16_t*>(storage_->device),
            reinterpret_cast<const uint32_t*>(storage_->device + hidden_bytes),
            static_cast<unsigned>(batch.first_position()), static_cast<unsigned>(batch.rows()),
            split1024_pre_fc_norm, epoch, stream, maximum_blocks);
        if (result.completion_unknown) quarantine(result.status);
        return result;
    }
    void quarantine(hipError_t status) {
        terminal_ = status;
        mtp_target_input_detail::quarantine(storage_.release());
    }
    std::unique_ptr<mtp_target_input_detail::Storage> storage_;
    hipError_t terminal_ = hipSuccess;
};
} // namespace qrt_sm121_mtp
