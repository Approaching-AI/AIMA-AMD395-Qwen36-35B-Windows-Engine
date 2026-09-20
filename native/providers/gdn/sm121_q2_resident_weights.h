#pragma once
#include "sm121_q2_model_weights.h"
#include "sm121_mtp_resident_weights.h"

namespace qrt_sm121_q2 {
// Target and MTP sources share the existing resident shard owner. Invalidating
// an epoch closes lookup while outstanding bindings still pin the allocations.
class ResidentTargetWeightSource final : public qrt_sm121_mtp::ModelWeightSource {
public:
    using Views = std::array<qrt_sm121_mtp::ModelTensorView, target_weight_count>;
    ResidentTargetWeightSource(std::shared_ptr<qrt_sm121_mtp::ResidentWeightStorage> storage,
        const std::atomic<uint64_t>* current_epoch, uint64_t captured_epoch, Views views)
        : storage_(std::move(storage)), current_epoch_(current_epoch), captured_epoch_(captured_epoch),
          views_(std::move(views)) {
        const auto& specs = target_weight_specs();
        for (size_t i = 0; i < views_.size(); ++i)
            views_[i].name = views_[i].name && specs[i].name == views_[i].name ? specs[i].name.c_str() : nullptr;
    }
    uint64_t epoch() const noexcept override {
        return current_epoch_ ? current_epoch_->load(std::memory_order_acquire) : 0;
    }
    bool tensor(const char* name, qrt_sm121_mtp::ModelTensorView* output) const override {
        if (!output) return false;
        *output = {};
        if (!name || !storage_ || !storage_->adopted() || !captured_epoch_ || epoch() != captured_epoch_) return false;
        const auto& specs = target_weight_specs();
        for (size_t i = 0; i < specs.size(); ++i) if (specs[i].name == name) {
            const auto& view = views_[i];
            if (!view.name || view.epoch != captured_epoch_ || !storage_->contains(view.device, view.bytes)) return false;
            *output = view;
            output->name = specs[i].name.c_str();
            return true;
        }
        return false;
    }
private:
    std::shared_ptr<qrt_sm121_mtp::ResidentWeightStorage> storage_;
    const std::atomic<uint64_t>* current_epoch_ = nullptr;
    uint64_t captured_epoch_ = 0;
    Views views_{};
};
} // namespace qrt_sm121_q2
