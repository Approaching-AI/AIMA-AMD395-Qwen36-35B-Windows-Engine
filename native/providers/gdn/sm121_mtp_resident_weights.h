#pragma once
#include "sm121_mtp_model_weights.h"

namespace qrt_sm121_mtp {
// Optional ownership transfer from the ordinary Windows resident shard store.
// Describing a candidate never frees the caller's allocations. adopt() occurs
// only after the store has retained this shared owner. All normal model views
// continue to borrow the same addresses, without copies or a second arena.
class ResidentWeightStorage {
public:
    ResidentWeightStorage() = default;
    ResidentWeightStorage(const ResidentWeightStorage&) = delete;
    ResidentWeightStorage& operator=(const ResidentWeightStorage&) = delete;
    ~ResidentWeightStorage() {
        if (adopted_) for (size_t i = 0; i < count_; ++i) (void)hipFree(allocations_[i].base);
    }
    bool describe(void* base, size_t bytes) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
        if (adopted_ || !base || !bytes || bytes > UINTPTR_MAX - begin || count_ == allocations_.size()) return false;
        for (size_t i = 0; i < count_; ++i) {
            const uintptr_t other = reinterpret_cast<uintptr_t>(allocations_[i].base);
            if (begin < other + allocations_[i].bytes && other < begin + bytes) return false;
        }
        allocations_[count_++] = {base, bytes}; bytes_ += bytes;
        return true;
    }
    bool contains(const void* pointer, size_t bytes) const {
        if (!pointer || !bytes) return false;
        const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
        for (size_t i = 0; i < count_; ++i) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(allocations_[i].base);
            if (address >= base && address - base <= allocations_[i].bytes &&
                bytes <= allocations_[i].bytes - (address - base)) return true;
        }
        return false;
    }
    bool adopt() {
        if (adopted_ || !count_) return false;
        adopted_ = true;
        return true;
    }
    bool adopted() const { return adopted_; }
    size_t bytes() const { return bytes_; }
    size_t allocations() const { return count_; }
private:
    struct Allocation { void* base = nullptr; size_t bytes = 0; };
    std::array<Allocation, 32> allocations_{};
    size_t count_ = 0, bytes_ = 0;
    bool adopted_ = false;
};

class ResidentModelWeightSource final : public ModelWeightSource {
public:
    using Views = std::array<ModelTensorView, model_weight_specs.size()>;
    ResidentModelWeightSource(std::shared_ptr<ResidentWeightStorage> storage,
        const std::atomic<uint64_t>* current_epoch, uint64_t captured_epoch, Views views)
        : storage_(std::move(storage)), current_epoch_(current_epoch),
          captured_epoch_(captured_epoch), views_(std::move(views)) {
        for (size_t i = 0; i < views_.size(); ++i) {
            views_[i].name = views_[i].name && !std::strcmp(views_[i].name, model_weight_specs[i].name)
                ? model_weight_specs[i].name : nullptr;
        }
    }
    uint64_t epoch() const noexcept override {
        return current_epoch_ ? current_epoch_->load(std::memory_order_acquire) : 0;
    }
    bool tensor(const char* name, ModelTensorView* out) const override {
        if (!out) return false;
        *out = {};
        if (!name || !storage_ || !storage_->adopted() || !captured_epoch_ || epoch() != captured_epoch_) return false;
        for (size_t i = 0; i < model_weight_specs.size(); ++i) if (!std::strcmp(name, model_weight_specs[i].name)) {
            const auto& value = views_[i];
            if (!value.name || std::strcmp(name, value.name) || value.epoch != captured_epoch_ ||
                !storage_->contains(value.device, value.bytes)) return false;
            *out = value;
            // The source owns an immutable name, even after the original
            // store's unordered_map and header strings have been cleared.
            out->name = model_weight_specs[i].name;
            return true;
        }
        return false;
    }
private:
    std::shared_ptr<ResidentWeightStorage> storage_;
    const std::atomic<uint64_t>* current_epoch_ = nullptr;
    uint64_t captured_epoch_ = 0;
    Views views_{};
};
} // namespace qrt_sm121_mtp
