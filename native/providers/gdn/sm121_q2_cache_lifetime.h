#pragma once

namespace qrt_sm121_q2 {
// The actual session transaction supplies this pin. It must block retirement
// of cache/table allocations and retain rollback until the borrower is gone.
// No mutex is stored here: the request owns its lock on the calling thread.
class ResidentCacheLifetime {
public:
    virtual ~ResidentCacheLifetime() = default;
    virtual bool ready(const void* owner) const noexcept = 0;
    virtual bool rollback_ready(const void* owner) const noexcept = 0;
    virtual void quarantine() const noexcept = 0;
};
} // namespace qrt_sm121_q2
