#pragma once
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <memory>

namespace qrt_sm121_mtp::mtp_cache_snapshot_detail {
struct Storage {
    uint16_t* device = nullptr;
    unsigned tokens = 0;
    bool quarantined = false;
    Storage* quarantine_next = nullptr;
    std::shared_ptr<Storage> quarantine_hold;
    ~Storage() { if (device) (void)hipFree(device); }
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
} // namespace qrt_sm121_mtp::mtp_cache_snapshot_detail
