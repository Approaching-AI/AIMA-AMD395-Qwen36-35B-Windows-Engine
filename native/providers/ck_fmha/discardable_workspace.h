#pragma once
#include <cstddef>

namespace qrt_discardable_workspace {
// The owner lock is held and prior consumers have drained. Contents are
// scratch, regenerated completely by the next producer; no model/session
// state may use this operation. Releasing first avoids overlapping old and
// new multi-gigabyte allocations. A failed allocation leaves an empty owner
// that the next call can rebuild. Failed release preserves the old owner.
template<class T, class Count>
hipError_t grow(T*& pointer, Count& capacity, Count required, size_t bytes) {
    if (!required || !bytes || bool(pointer) != bool(capacity))
        return hipErrorInvalidValue;
    if (capacity >= required) return hipSuccess;
    if (pointer) {
        const auto status = hipFree(pointer);
        if (status != hipSuccess) return status;
        pointer = nullptr;
        capacity = 0;
    }
    T* next = nullptr;
    const auto status = hipMalloc(reinterpret_cast<void**>(&next), bytes);
    if (status != hipSuccess) return status;
    pointer = next;
    capacity = required;
    return hipSuccess;
}
} // namespace qrt_discardable_workspace
