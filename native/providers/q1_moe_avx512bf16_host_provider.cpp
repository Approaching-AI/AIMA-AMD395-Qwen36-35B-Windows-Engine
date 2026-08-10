#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include "q1_moe_avx512bf16_host_provider.h"
#include "moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"

#ifndef _WIN32
#error "The q1 AVX-512 BF16 host provider requires Windows."
#endif

#include <Windows.h>
#include <immintrin.h>
#include <intrin.h>
#include <malloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr size_t kHidden = QRT_Q1_MOE_AVX512BF16_HOST_HIDDEN;
constexpr size_t kIntermediate = QRT_Q1_MOE_AVX512BF16_HOST_INTERMEDIATE;
constexpr size_t kRoutes = QRT_Q1_MOE_AVX512BF16_HOST_ROUTES;
constexpr size_t kExpertCount = QRT_Q1_MOE_AVX512BF16_HOST_EXPERT_COUNT;
constexpr size_t kWorkers = QRT_Q1_MOE_AVX512BF16_HOST_WORKERS;
constexpr size_t kWaveLanes = 32u;
constexpr size_t kGateUpBytesPerExpert =
    QRT_Q1_MOE_AVX512BF16_HOST_GATE_UP_BYTES_PER_EXPERT;
constexpr size_t kDownBytesPerExpert =
    QRT_Q1_MOE_AVX512BF16_HOST_DOWN_BYTES_PER_EXPERT;
constexpr size_t kBytesPerExpertPair =
    QRT_Q1_MOE_AVX512BF16_HOST_BYTES_PER_EXPERT_PAIR;
constexpr size_t kGateUpElementsPerExpert =
    kGateUpBytesPerExpert / sizeof(uint16_t);
constexpr size_t kDownElementsPerExpert =
    kDownBytesPerExpert / sizeof(uint16_t);
constexpr size_t kGateElementsPerExpert = kIntermediate * kHidden;

static_assert(kRoutes == 8u);
static_assert(kWorkers == 16u);
static_assert(kGateUpElementsPerExpert == 2u * kIntermediate * kHidden);
static_assert(kDownElementsPerExpert == kHidden * kIntermediate);
static_assert((kHidden / 2u) % kWaveLanes == 0u);
static_assert((kIntermediate / 2u) % kWaveLanes == 0u);

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point start, Clock::time_point stop) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
            .count()
    );
}

float bf16_to_float(uint16_t bits) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + bias) >> 16u);
}

float add_separate(float left, float right) {
    volatile float sum = left + right;
    return sum;
}

float mul_separate(float left, float right) {
    volatile float product = left * right;
    return product;
}

float wave32_tree_sum(std::array<float, kWaveLanes> lanes) {
    for (size_t offset = kWaveLanes / 2u; offset > 0u; offset >>= 1u) {
        for (size_t lane = 0u; lane + offset < kWaveLanes; ++lane) {
            lanes[lane] = add_separate(lanes[lane], lanes[lane + offset]);
        }
    }
    return lanes[0];
}

float avx512bf16_wave32_dot(const uint16_t *input,
                            const uint16_t *weight,
                            size_t elements) {
    __m512 lane_0_15 = _mm512_setzero_ps();
    __m512 lane_16_31 = _mm512_setzero_ps();
    const size_t pairs = elements / 2u;
    for (size_t pair = 0u; pair < pairs; pair += kWaveLanes) {
        const __m512bh input_0_15 = (__m512bh)_mm512_loadu_si512(
            static_cast<const void *>(input + pair * 2u)
        );
        const __m512bh weight_0_15 = (__m512bh)_mm512_loadu_si512(
            static_cast<const void *>(weight + pair * 2u)
        );
        const __m512bh input_16_31 = (__m512bh)_mm512_loadu_si512(
            static_cast<const void *>(
                input + (pair + kWaveLanes / 2u) * 2u
            )
        );
        const __m512bh weight_16_31 = (__m512bh)_mm512_loadu_si512(
            static_cast<const void *>(
                weight + (pair + kWaveLanes / 2u) * 2u
            )
        );
        lane_0_15 = _mm512_dpbf16_ps(
            lane_0_15,
            input_0_15,
            weight_0_15
        );
        lane_16_31 = _mm512_dpbf16_ps(
            lane_16_31,
            input_16_31,
            weight_16_31
        );
    }
    alignas(64) std::array<float, kWaveLanes> lanes{};
    _mm512_store_ps(lanes.data(), lane_0_15);
    _mm512_store_ps(lanes.data() + kWaveLanes / 2u, lane_16_31);
    return wave32_tree_sum(lanes);
}

uint64_t fnv1a64(const void *data, size_t bytes) {
    const auto *cursor = static_cast<const unsigned char *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

class PersistentWorkerPool {
public:
    PersistentWorkerPool() {
        workers_.reserve(kWorkers);
        try {
            for (size_t worker = 0u; worker < kWorkers; ++worker) {
                workers_.emplace_back([this, worker]() { worker_loop(worker); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
                ++epoch_;
            }
            start_.notify_all();
            for (std::thread &worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            throw;
        }
    }

    PersistentWorkerPool(const PersistentWorkerPool &) = delete;
    PersistentWorkerPool &operator=(const PersistentWorkerPool &) = delete;

    ~PersistentWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++epoch_;
        }
        start_.notify_all();
        for (std::thread &worker : workers_) {
            worker.join();
        }
    }

    void parallel_for(size_t task_count, std::function<void(size_t)> task) {
        if (task_count == 0u || !task) {
            throw std::runtime_error("invalid worker-pool dispatch");
        }
        std::unique_lock<std::mutex> lock(mutex_);
        task_ = std::move(task);
        task_count_ = task_count;
        completed_workers_ = 0u;
        next_task_.store(0u, std::memory_order_relaxed);
        ++epoch_;
        start_.notify_all();
        done_.wait(lock, [this]() { return completed_workers_ == kWorkers; });
    }

private:
    void worker_loop(size_t worker_index) {
        (void)worker_index;
        uint64_t observed_epoch = 0u;
        for (;;) {
            std::function<void(size_t)> task;
            size_t task_count = 0u;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_.wait(lock, [this, &observed_epoch]() {
                    return stopping_ || epoch_ != observed_epoch;
                });
                if (stopping_) {
                    return;
                }
                observed_epoch = epoch_;
                task = task_;
                task_count = task_count_;
            }
            for (;;) {
                const size_t index =
                    next_task_.fetch_add(1u, std::memory_order_relaxed);
                if (index >= task_count) {
                    break;
                }
                task(index);
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completed_workers_;
                if (completed_workers_ == kWorkers) {
                    done_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    std::function<void(size_t)> task_;
    std::atomic<size_t> next_task_{0u};
    size_t task_count_ = 0u;
    size_t completed_workers_ = 0u;
    uint64_t epoch_ = 0u;
    bool stopping_ = false;
};

struct CpuFeatures {
    bool osxsave = false;
    bool avx = false;
    bool avx512f = false;
    bool avx512bf16 = false;
    bool zmm_state = false;

    bool supported() const {
        return osxsave && avx && avx512f && avx512bf16 && zmm_state;
    }
};

CpuFeatures query_cpu_features() {
    CpuFeatures features;
    int registers[4] = {};
    __cpuidex(registers, 0, 0);
    const unsigned int max_leaf = static_cast<unsigned int>(registers[0]);
    if (max_leaf < 1u) {
        return features;
    }
    __cpuidex(registers, 1, 0);
    features.osxsave = (registers[2] & (1 << 27)) != 0;
    features.avx = (registers[2] & (1 << 28)) != 0;
    if (features.osxsave) {
        const uint64_t xcr0 = _xgetbv(0);
        features.zmm_state = (xcr0 & UINT64_C(0xe6)) == UINT64_C(0xe6);
    }
    if (max_leaf < 7u) {
        return features;
    }
    __cpuidex(registers, 7, 0);
    const unsigned int max_subleaf = static_cast<unsigned int>(registers[0]);
    features.avx512f = (registers[1] & (1 << 16)) != 0;
    if (max_subleaf >= 1u) {
        __cpuidex(registers, 7, 1);
        features.avx512bf16 = (registers[0] & (1 << 5)) != 0;
    }
    return features;
}

bool utf8_to_wide(const char *text, std::wstring *out) {
    if (text == nullptr || text[0] == '\0' || out == nullptr) {
        return false;
    }
    const int length = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        text,
        -1,
        nullptr,
        0
    );
    if (length <= 1) {
        return false;
    }
    std::wstring wide(static_cast<size_t>(length), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            text,
            -1,
            wide.data(),
            length
        ) != length) {
        return false;
    }
    wide.resize(static_cast<size_t>(length - 1));
    *out = std::move(wide);
    return true;
}

enum class PathForm {
    relative,
    absolute,
    ambiguous
};

bool is_path_separator(wchar_t character) {
    return character == L'\\' || character == L'/';
}

PathForm classify_path(const std::wstring &path) {
    if (path.empty()) {
        return PathForm::ambiguous;
    }
    if (path.size() >= 2u && path[1] == L':') {
        if (std::iswalpha(static_cast<wint_t>(path[0])) == 0 ||
            path.size() < 3u || !is_path_separator(path[2])) {
            // "C:foo" and "C:" are drive-relative, not absolute.
            return PathForm::ambiguous;
        }
        return PathForm::absolute;
    }
    if (is_path_separator(path[0])) {
        // A single leading separator is relative to the current drive.  Only
        // UNC/device paths with two leading separators are unambiguous.
        return path.size() >= 2u && is_path_separator(path[1])
            ? PathForm::absolute
            : PathForm::ambiguous;
    }
    return PathForm::relative;
}

bool full_path(const std::wstring &path, std::wstring *out) {
    if (path.empty() || out == nullptr) {
        return false;
    }
    const DWORD required = GetFullPathNameW(path.c_str(), 0u, nullptr, nullptr);
    if (required == 0u) {
        return false;
    }
    std::wstring result(static_cast<size_t>(required), L'\0');
    const DWORD written = GetFullPathNameW(
        path.c_str(),
        required,
        result.data(),
        nullptr
    );
    if (written == 0u || written >= required) {
        return false;
    }
    result.resize(static_cast<size_t>(written));
    *out = std::move(result);
    return true;
}

std::wstring lowercase_path(std::wstring path) {
    for (wchar_t &character : path) {
        character = static_cast<wchar_t>(
            std::towlower(static_cast<wint_t>(character))
        );
    }
    return path;
}

bool resolve_model_and_shard(const char *model_dir_utf8,
                             const char *shard_utf8,
                             std::wstring *model_key,
                             std::wstring *shard_path) {
    std::wstring model;
    std::wstring shard;
    std::wstring model_full;
    std::wstring shard_full;
    if (!utf8_to_wide(model_dir_utf8, &model) ||
        !utf8_to_wide(shard_utf8, &shard) ||
        classify_path(model) == PathForm::ambiguous ||
        classify_path(shard) == PathForm::ambiguous ||
        !full_path(model, &model_full)) {
        return false;
    }
    std::wstring joined;
    if (classify_path(shard) == PathForm::absolute) {
        joined = shard;
    } else {
        joined = model_full;
        if (!joined.empty() && joined.back() != L'\\' && joined.back() != L'/') {
            joined.push_back(L'\\');
        }
        joined += shard;
    }
    if (!full_path(joined, &shard_full)) {
        return false;
    }
    *model_key = lowercase_path(model_full);
    *shard_path = std::move(shard_full);
    return true;
}

class ReadOnlyMappedView {
public:
    ReadOnlyMappedView() = default;
    ReadOnlyMappedView(const ReadOnlyMappedView &) = delete;
    ReadOnlyMappedView &operator=(const ReadOnlyMappedView &) = delete;

    ReadOnlyMappedView(ReadOnlyMappedView &&other) noexcept {
        move_from(std::move(other));
    }

    ReadOnlyMappedView &operator=(ReadOnlyMappedView &&other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    ~ReadOnlyMappedView() { reset(); }

    bool map(const std::wstring &path,
             uint64_t absolute_begin,
             uint64_t bytes,
             uint32_t *win32_error) {
        reset();
        if (bytes == 0u || bytes > static_cast<uint64_t>(SIZE_MAX)) {
            return false;
        }
        HANDLE file = CreateFileW(
            path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr
        );
        if (file == INVALID_HANDLE_VALUE) {
            if (win32_error != nullptr) {
                *win32_error = GetLastError();
            }
            return false;
        }
        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0 ||
            absolute_begin > static_cast<uint64_t>(file_size.QuadPart) ||
            bytes > static_cast<uint64_t>(file_size.QuadPart) - absolute_begin) {
            if (win32_error != nullptr) {
                *win32_error = GetLastError();
            }
            CloseHandle(file);
            return false;
        }
        HANDLE mapping = CreateFileMappingW(
            file,
            nullptr,
            PAGE_READONLY,
            0u,
            0u,
            nullptr
        );
        if (mapping == nullptr) {
            if (win32_error != nullptr) {
                *win32_error = GetLastError();
            }
            CloseHandle(file);
            return false;
        }
        SYSTEM_INFO system_info{};
        GetSystemInfo(&system_info);
        const uint64_t granularity =
            static_cast<uint64_t>(system_info.dwAllocationGranularity);
        if (granularity == 0u || (granularity & (granularity - 1u)) != 0u) {
            CloseHandle(mapping);
            CloseHandle(file);
            return false;
        }
        const uint64_t aligned_begin = absolute_begin & ~(granularity - 1u);
        const uint64_t delta = absolute_begin - aligned_begin;
        if (delta > static_cast<uint64_t>(SIZE_MAX) ||
            bytes > static_cast<uint64_t>(SIZE_MAX) - delta) {
            CloseHandle(mapping);
            CloseHandle(file);
            return false;
        }
        const SIZE_T view_bytes = static_cast<SIZE_T>(delta + bytes);
        void *base = MapViewOfFile(
            mapping,
            FILE_MAP_READ,
            static_cast<DWORD>(aligned_begin >> 32u),
            static_cast<DWORD>(aligned_begin & UINT64_C(0xffffffff)),
            view_bytes
        );
        if (base == nullptr) {
            if (win32_error != nullptr) {
                *win32_error = GetLastError();
            }
            CloseHandle(mapping);
            CloseHandle(file);
            return false;
        }
        file_ = file;
        mapping_ = mapping;
        base_ = base;
        data_ = static_cast<const unsigned char *>(base) + delta;
        view_bytes_ = view_bytes;
        data_bytes_ = bytes;
        absolute_begin_ = absolute_begin;
        path_key_ = lowercase_path(path);
        return true;
    }

    void prefetch_and_touch(bool prefetch,
                            bool touch,
                            uint64_t *prefetch_requested_bytes,
                            uint64_t *touched_bytes) const {
        if (data_ == nullptr || data_bytes_ == 0u) {
            return;
        }
        if (prefetch) {
            using PrefetchVirtualMemoryFn = BOOL(WINAPI *)(
                HANDLE,
                ULONG_PTR,
                PWIN32_MEMORY_RANGE_ENTRY,
                ULONG
            );
            static const PrefetchVirtualMemoryFn prefetch_fn = []() {
                HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
                return kernel == nullptr
                    ? nullptr
                    : reinterpret_cast<PrefetchVirtualMemoryFn>(
                          GetProcAddress(kernel, "PrefetchVirtualMemory")
                      );
            }();
            if (prefetch_fn != nullptr) {
                WIN32_MEMORY_RANGE_ENTRY range{};
                range.VirtualAddress = const_cast<unsigned char *>(data_);
                range.NumberOfBytes = static_cast<SIZE_T>(data_bytes_);
                (void)prefetch_fn(GetCurrentProcess(), 1u, &range, 0u);
                if (prefetch_requested_bytes != nullptr) {
                    *prefetch_requested_bytes += data_bytes_;
                }
            }
        }
        if (touch) {
            SYSTEM_INFO system_info{};
            GetSystemInfo(&system_info);
            const size_t page_bytes = system_info.dwPageSize != 0u
                ? static_cast<size_t>(system_info.dwPageSize)
                : 4096u;
            volatile unsigned char sink = 0u;
            for (uint64_t offset = 0u; offset < data_bytes_; offset += page_bytes) {
                sink = static_cast<unsigned char>(sink ^ data_[offset]);
            }
            sink = static_cast<unsigned char>(sink ^ data_[data_bytes_ - 1u]);
            if (sink == UINT8_C(0xff)) {
                touch_sink_.fetch_xor(sink, std::memory_order_relaxed);
            }
            if (touched_bytes != nullptr) {
                *touched_bytes += data_bytes_;
            }
        }
    }

    const uint16_t *bf16() const {
        return reinterpret_cast<const uint16_t *>(data_);
    }

    const std::wstring &path_key() const { return path_key_; }
    uint64_t absolute_begin() const { return absolute_begin_; }
    uint64_t data_bytes() const { return data_bytes_; }

private:
    void reset() {
        if (base_ != nullptr) {
            (void)UnmapViewOfFile(base_);
        }
        if (mapping_ != nullptr) {
            (void)CloseHandle(mapping_);
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            (void)CloseHandle(file_);
        }
        file_ = INVALID_HANDLE_VALUE;
        mapping_ = nullptr;
        base_ = nullptr;
        data_ = nullptr;
        view_bytes_ = 0u;
        data_bytes_ = 0u;
        absolute_begin_ = 0u;
        path_key_.clear();
    }

    void move_from(ReadOnlyMappedView &&other) {
        file_ = other.file_;
        mapping_ = other.mapping_;
        base_ = other.base_;
        data_ = other.data_;
        view_bytes_ = other.view_bytes_;
        data_bytes_ = other.data_bytes_;
        absolute_begin_ = other.absolute_begin_;
        path_key_ = std::move(other.path_key_);
        other.file_ = INVALID_HANDLE_VALUE;
        other.mapping_ = nullptr;
        other.base_ = nullptr;
        other.data_ = nullptr;
        other.view_bytes_ = 0u;
        other.data_bytes_ = 0u;
        other.absolute_begin_ = 0u;
    }

    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    void *base_ = nullptr;
    const unsigned char *data_ = nullptr;
    SIZE_T view_bytes_ = 0u;
    uint64_t data_bytes_ = 0u;
    uint64_t absolute_begin_ = 0u;
    std::wstring path_key_;
    static std::atomic<unsigned char> touch_sink_;
};

std::atomic<unsigned char> ReadOnlyMappedView::touch_sink_{0u};

enum class CacheSourceMode : uint32_t {
    none = 0u,
    file_mapping = 1u,
    resident_device = 2u
};

class AlignedOwnedBuffer {
public:
    AlignedOwnedBuffer() = default;
    AlignedOwnedBuffer(const AlignedOwnedBuffer &) = delete;
    AlignedOwnedBuffer &operator=(const AlignedOwnedBuffer &) = delete;

    AlignedOwnedBuffer(AlignedOwnedBuffer &&other) noexcept {
        move_from(std::move(other));
    }

    AlignedOwnedBuffer &operator=(AlignedOwnedBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    ~AlignedOwnedBuffer() { reset(); }

    bool allocate(
        uint64_t bytes,
        qrt_q1_moe_avx512bf16_host_allocate_fn host_allocate,
        qrt_q1_moe_avx512bf16_host_free_fn host_free,
        void *allocator_context,
        int32_t *allocator_error,
        uint64_t *allocation_call_count,
        uint64_t *allocated_bytes,
        uint64_t *allocation_elapsed_ns
    ) {
        reset();
        if (allocator_error != nullptr) {
            *allocator_error = 0;
        }
        if (bytes == 0u || bytes > static_cast<uint64_t>(SIZE_MAX) ||
            (host_allocate == nullptr) != (host_free == nullptr)) {
            return false;
        }
        const Clock::time_point start = Clock::now();
        if (allocation_call_count != nullptr) {
            ++*allocation_call_count;
        }
        void *candidate = nullptr;
        int32_t callback_error = 0;
        if (host_allocate == nullptr) {
            candidate = _aligned_malloc(static_cast<size_t>(bytes), 64u);
        } else {
            try {
                callback_error = host_allocate(
                    allocator_context,
                    bytes,
                    64u,
                    &candidate
                );
            } catch (...) {
                callback_error = (std::numeric_limits<int32_t>::min)();
            }
        }
        const bool aligned = candidate != nullptr &&
            reinterpret_cast<uintptr_t>(candidate) % 64u == 0u;
        if (callback_error != 0 || !aligned) {
            if (allocator_error != nullptr) {
                *allocator_error = callback_error != 0
                    ? callback_error
                    : candidate == nullptr
                        ? (std::numeric_limits<int32_t>::min)() + 1
                        : (std::numeric_limits<int32_t>::min)() + 2;
            }
            if (candidate != nullptr) {
                if (host_free != nullptr) {
                    try {
                        (void)host_free(
                            allocator_context,
                            candidate,
                            bytes,
                            64u
                        );
                    } catch (...) {
                    }
                } else {
                    _aligned_free(candidate);
                }
            }
            if (allocation_elapsed_ns != nullptr) {
                *allocation_elapsed_ns += elapsed_ns(start, Clock::now());
            }
            return false;
        }
        data_ = candidate;
        bytes_ = bytes;
        host_free_ = host_free;
        allocator_context_ = allocator_context;
        if (allocated_bytes != nullptr) {
            *allocated_bytes += bytes;
        }
        if (allocation_elapsed_ns != nullptr) {
            *allocation_elapsed_ns += elapsed_ns(start, Clock::now());
        }
        return true;
    }

    void *data() { return data_; }
    const uint16_t *bf16(uint64_t byte_offset = 0u) const {
        return reinterpret_cast<const uint16_t *>(
            static_cast<const unsigned char *>(data_) + byte_offset
        );
    }
    uint64_t bytes() const { return bytes_; }

private:
    void reset() {
        if (data_ != nullptr) {
            if (host_free_ != nullptr) {
                try {
                    (void)host_free_(
                        allocator_context_,
                        data_,
                        bytes_,
                        64u
                    );
                } catch (...) {
                }
            } else {
                _aligned_free(data_);
            }
        }
        data_ = nullptr;
        bytes_ = 0u;
        host_free_ = nullptr;
        allocator_context_ = nullptr;
    }

    void move_from(AlignedOwnedBuffer &&other) {
        data_ = other.data_;
        bytes_ = other.bytes_;
        host_free_ = other.host_free_;
        allocator_context_ = other.allocator_context_;
        other.data_ = nullptr;
        other.bytes_ = 0u;
        other.host_free_ = nullptr;
        other.allocator_context_ = nullptr;
    }

    void *data_ = nullptr;
    uint64_t bytes_ = 0u;
    qrt_q1_moe_avx512bf16_host_free_fn host_free_ = nullptr;
    void *allocator_context_ = nullptr;
};

struct CacheEntry {
    uint64_t model_generation = 0u;
    uint32_t layer_index = 0u;
    uint32_t expert_id = 0u;
    uint64_t last_use = 0u;
    uint32_t pin_count = 0u;
    CacheSourceMode source_mode = CacheSourceMode::none;
    ReadOnlyMappedView gate_up;
    ReadOnlyMappedView down;
    AlignedOwnedBuffer resident_pair;
    uint64_t gate_up_device_begin = 0u;
    uint64_t down_device_begin = 0u;

    const uint16_t *gate_up_bf16() const {
        return source_mode == CacheSourceMode::resident_device
            ? resident_pair.bf16()
            : gate_up.bf16();
    }

    const uint16_t *down_bf16() const {
        return source_mode == CacheSourceMode::resident_device
            ? resident_pair.bf16(kGateUpBytesPerExpert)
            : down.bf16();
    }
};

struct TensorSourceIdentity {
    CacheSourceMode mode = CacheSourceMode::none;
    std::wstring model_key;
    std::wstring gate_path_key;
    std::wstring down_path_key;
    uint64_t gate_tensor_absolute_begin = 0u;
    uint64_t down_tensor_absolute_begin = 0u;
    uint64_t gate_bytes_per_expert = 0u;
    uint64_t down_bytes_per_expert = 0u;
    uint32_t layer_index = 0u;
    uint64_t source_generation = 0u;
    uint64_t gate_up_device_base = 0u;
    uint64_t down_device_base = 0u;
    qrt_q1_moe_avx512bf16_host_device_copy_to_host_fn copy_to_host = nullptr;
    void *copy_context = nullptr;
    qrt_q1_moe_avx512bf16_host_allocate_fn host_allocate = nullptr;
    qrt_q1_moe_avx512bf16_host_free_fn host_free = nullptr;
    void *allocator_context = nullptr;

    bool matches(const TensorSourceIdentity &other) const {
        return mode == other.mode &&
            model_key == other.model_key &&
            gate_path_key == other.gate_path_key &&
            down_path_key == other.down_path_key &&
            gate_tensor_absolute_begin == other.gate_tensor_absolute_begin &&
            down_tensor_absolute_begin == other.down_tensor_absolute_begin &&
            gate_bytes_per_expert == other.gate_bytes_per_expert &&
            down_bytes_per_expert == other.down_bytes_per_expert &&
            layer_index == other.layer_index &&
            source_generation == other.source_generation &&
            gate_up_device_base == other.gate_up_device_base &&
            down_device_base == other.down_device_base &&
            copy_to_host == other.copy_to_host &&
            copy_context == other.copy_context &&
            host_allocate == other.host_allocate &&
            host_free == other.host_free &&
            allocator_context == other.allocator_context;
    }

    void swap(TensorSourceIdentity &other) noexcept {
        std::swap(mode, other.mode);
        model_key.swap(other.model_key);
        gate_path_key.swap(other.gate_path_key);
        down_path_key.swap(other.down_path_key);
        std::swap(gate_tensor_absolute_begin, other.gate_tensor_absolute_begin);
        std::swap(down_tensor_absolute_begin, other.down_tensor_absolute_begin);
        std::swap(gate_bytes_per_expert, other.gate_bytes_per_expert);
        std::swap(down_bytes_per_expert, other.down_bytes_per_expert);
        std::swap(layer_index, other.layer_index);
        std::swap(source_generation, other.source_generation);
        std::swap(gate_up_device_base, other.gate_up_device_base);
        std::swap(down_device_base, other.down_device_base);
        std::swap(copy_to_host, other.copy_to_host);
        std::swap(copy_context, other.copy_context);
        std::swap(host_allocate, other.host_allocate);
        std::swap(host_free, other.host_free);
        std::swap(allocator_context, other.allocator_context);
    }
};

struct ProviderImpl {
    explicit ProviderImpl(const qrt_q1_moe_avx512bf16_host_config_t &input)
        : config(input) {}

    qrt_q1_moe_avx512bf16_host_config_t config{};
    PersistentWorkerPool pool;
    std::mutex run_mutex;
    std::mutex state_mutex;
    std::vector<std::unique_ptr<CacheEntry>> entries;
    TensorSourceIdentity source_identity;
    bool source_identity_set = false;
    uint64_t model_generation = 0u;
    uint64_t clock = 0u;
    uint64_t cache_bytes = 0u;
    uint64_t cache_peak_bytes = 0u;
    uint32_t cache_peak_entries = 0u;
    qrt_q1_moe_avx512bf16_host_stats_t cumulative{};
    std::string last_error;
    uint32_t last_error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE;
    uint32_t last_win32_error = 0u;
    int32_t last_resident_callback_error = 0;
    std::array<uint16_t, kRoutes * kIntermediate> gate{};
    std::array<uint16_t, kRoutes * kIntermediate> up{};
    std::array<uint16_t, kRoutes * kIntermediate> activated{};

    uint64_t next_clock() {
        ++clock;
        if (clock == 0u) {
            ++clock;
        }
        return clock;
    }

    void set_error(uint32_t error_code,
                   uint32_t win32_error,
                   const std::string &message) {
        last_error_code = error_code;
        last_win32_error = win32_error;
        last_error = message;
        cumulative.last_error_code = error_code;
        cumulative.last_win32_error = win32_error;
        last_resident_callback_error = 0;
        cumulative.last_resident_callback_error = 0;
    }

    void set_resident_copy_error(int32_t callback_error,
                                 const std::string &message) {
        set_error(
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_RESIDENT_COPY,
            0u,
            message
        );
        last_resident_callback_error = callback_error;
        cumulative.last_resident_callback_error = callback_error;
    }

    void clear_error() {
        set_error(QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE, 0u, "");
    }

    CacheEntry *find_entry(uint32_t layer_index, uint32_t expert_id) {
        for (const std::unique_ptr<CacheEntry> &entry : entries) {
            if (entry->model_generation == model_generation &&
                entry->layer_index == layer_index &&
                entry->expert_id == expert_id) {
                return entry.get();
            }
        }
        return nullptr;
    }

    bool source_matches(const CacheEntry &entry,
                        const std::wstring &gate_path_key,
                        uint64_t gate_begin,
                        const std::wstring &down_path_key,
                        uint64_t down_begin) const {
        return entry.source_mode == CacheSourceMode::file_mapping &&
            entry.gate_up.path_key() == gate_path_key &&
            entry.gate_up.absolute_begin() == gate_begin &&
            entry.gate_up.data_bytes() == kGateUpBytesPerExpert &&
            entry.down.path_key() == down_path_key &&
            entry.down.absolute_begin() == down_begin &&
            entry.down.data_bytes() == kDownBytesPerExpert;
    }

    bool resident_source_matches(const CacheEntry &entry,
                                 uint64_t gate_begin,
                                 uint64_t down_begin) const {
        return entry.source_mode == CacheSourceMode::resident_device &&
            entry.gate_up_device_begin == gate_begin &&
            entry.down_device_begin == down_begin &&
            entry.resident_pair.bytes() == kBytesPerExpertPair;
    }

    void bump_model_generation() {
        ++model_generation;
        if (model_generation == 0u) {
            ++model_generation;
        }
    }

    bool is_protected(
        const CacheEntry &entry,
        const std::vector<std::pair<uint32_t, uint32_t>> &protected_keys
    ) const {
        for (const auto &key : protected_keys) {
            if (entry.layer_index == key.first && entry.expert_id == key.second) {
                return true;
            }
        }
        return false;
    }

    bool evict_one(
        const std::vector<std::pair<uint32_t, uint32_t>> &protected_keys
    ) {
        size_t victim = entries.size();
        uint64_t oldest = (std::numeric_limits<uint64_t>::max)();
        for (size_t index = 0u; index < entries.size(); ++index) {
            const CacheEntry &entry = *entries[index];
            if (entry.pin_count == 0u &&
                !is_protected(entry, protected_keys) &&
                entry.last_use < oldest) {
                oldest = entry.last_use;
                victim = index;
            }
        }
        if (victim == entries.size()) {
            return false;
        }
        cache_bytes -= kBytesPerExpertPair;
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(victim));
        return true;
    }

    void clear_entries() {
        entries.clear();
        cache_bytes = 0u;
    }

    void snapshot_stats(qrt_q1_moe_avx512bf16_host_stats_t *out) const {
        if (out == nullptr) {
            return;
        }
        *out = cumulative;
        out->struct_size = sizeof(*out);
        out->abi_version = QRT_Q1_MOE_AVX512BF16_HOST_ABI_VERSION;
        out->worker_count = config.worker_count;
        out->cache_entry_capacity = config.cache_entry_capacity;
        out->cache_byte_capacity = config.cache_byte_capacity;
        out->cache_entry_count = static_cast<uint32_t>(entries.size());
        out->cache_peak_entry_count = cache_peak_entries;
        out->cache_bytes = cache_bytes;
        out->cache_peak_bytes = cache_peak_bytes;
        out->cache_clock = clock;
        out->last_error_code = last_error_code;
        out->last_win32_error = last_win32_error;
        out->last_resident_callback_error = last_resident_callback_error;
    }
};

void account_resident_prewarm_exception(
    ProviderImpl *impl,
    const qrt_q1_moe_avx512bf16_host_resident_prewarm_request_t *request,
    qrt_q1_moe_avx512bf16_host_resident_prewarm_result_t *result,
    uint32_t error_code,
    const char *message,
    bool call_started,
    bool result_accounted
) noexcept {
    if (impl == nullptr || request == nullptr || result == nullptr ||
        !call_started || result_accounted) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(impl->state_mutex);
        const uint32_t processed = result->copied_count +
            result->already_present_count + result->failed_count;
        if (processed < request->expert_id_count) {
            result->failed_count += request->expert_id_count - processed;
        }
        result->cache_entry_count = static_cast<uint32_t>(impl->entries.size());
        result->cache_bytes = impl->cache_bytes;
        result->cache_peak_bytes = impl->cache_peak_bytes;
        impl->last_error_code = error_code;
        impl->last_win32_error = 0u;
        impl->last_resident_callback_error = 0;
        impl->cumulative.last_error_code = error_code;
        impl->cumulative.last_win32_error = 0u;
        impl->cumulative.last_resident_callback_error = 0;
        try {
            impl->last_error.assign(message != nullptr ? message : "");
        } catch (...) {
            impl->last_error.clear();
        }
        ++impl->cumulative.resident_prewarm_error_count;
        impl->cumulative.resident_prewarm_copied_entry_count +=
            result->copied_count;
        impl->cumulative.resident_prewarm_already_present_count +=
            result->already_present_count;
        impl->cumulative.resident_prewarm_failed_entry_count +=
            result->failed_count;
        impl->cumulative.resident_prewarm_eviction_count +=
            result->eviction_count;
        impl->cumulative.resident_copy_call_count += result->copy_call_count;
        impl->cumulative.resident_copied_bytes += result->copied_bytes;
        impl->cumulative.resident_copy_elapsed_ns += result->copy_elapsed_ns;
        impl->cumulative.resident_prewarm_elapsed_ns += result->elapsed_ns;
    } catch (...) {
        // The public C ABI has already been reduced to an error result.  A
        // secondary mutex/runtime failure must not escape while accounting it.
    }
}

bool bf16_values_finite(const uint16_t *values, size_t count) {
    for (size_t index = 0u; index < count; ++index) {
        if ((values[index] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) {
            return false;
        }
    }
    return true;
}

uint16_t activation_endpoint(uint16_t gate_bits, uint16_t up_bits) {
    const float gate = bf16_to_float(gate_bits);
    const float up = bf16_to_float(up_bits);
    const float silu = gate / (1.0f + std::exp(-gate));
    return float_to_bf16(silu * up);
}

uint16_t vllm_activation_endpoint(uint16_t gate_bits, uint16_t up_bits) {
    const float gate = bf16_to_float(gate_bits);
    const float up = bf16_to_float(up_bits);
    const float silu = gate / (1.0f + std::exp(-gate));
    const uint16_t silu_bits = float_to_bf16(silu);
    return float_to_bf16(
        mul_separate(bf16_to_float(silu_bits), up)
    );
}

float hawkeye_dot(const uint16_t *left,
                   const uint16_t *right,
                   size_t elements,
                   uint32_t arithmetic_mode) {
    switch (arithmetic_mode) {
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_HOPPER:
        return qrt_q1_moe_hawkeye::dot_bf16_hopper(
            left,
            right,
            elements
        );
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP8_WIDTH26:
        return qrt_q1_moe_hawkeye::dot_bf16_group8_width26(
            left,
            right,
            elements
        );
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP16_WIDTH25:
        return qrt_q1_moe_hawkeye::dot_bf16_group16_width25(
            left,
            right,
            elements
        );
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP4_WIDTH25:
        return qrt_q1_moe_hawkeye::dot_bf16_group4_width25(
            left,
            right,
            elements
        );
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_FACTOR_GROUP8_WIDTH24:
        return qrt_q1_moe_hawkeye::dot_bf16_group8_width24(
            left,
            right,
            elements
        );
    default:
        return qrt_q1_moe_hawkeye::dot_bf16_ampere_lovelace(
            left,
            right,
            elements
        );
    }
}

bool uses_hawkeye_accumulator(uint32_t arithmetic_mode) {
    switch (arithmetic_mode) {
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_VLLM_ENDPOINTS:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_FINAL_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_FINAL_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_SEQUENTIAL:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_WEIGHTED_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_FINAL_BF16:
        return false;
    default:
        return true;
    }
}

constexpr uint32_t kEndpointActivationBf16 = UINT32_C(1);
constexpr uint32_t kEndpointWeightedDownBf16 = UINT32_C(2);
constexpr uint32_t kEndpointRouteSumVt4 = UINT32_C(4);
constexpr uint32_t kEndpointFinalBf16 = UINT32_C(8);
constexpr uint32_t kEndpointAllVllm = UINT32_C(15);

uint32_t endpoint_mask_for_mode(uint32_t arithmetic_mode) {
    switch (arithmetic_mode) {
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_LEGACY_ENDPOINTS:
        return 0u;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_FINAL_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_FINAL_BF16:
        return kEndpointFinalBf16;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_ACTIVATION_FINAL_BF16:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_FINAL_BF16:
        return kEndpointActivationBf16 | kEndpointFinalBf16;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_HAWKEYE_AMPERE_LOVELACE_WEIGHTED_SEQUENTIAL:
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_SEQUENTIAL:
        return kEndpointActivationBf16 |
            kEndpointWeightedDownBf16 |
            kEndpointFinalBf16;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_BF16:
        return kEndpointActivationBf16;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_BF16:
        return kEndpointWeightedDownBf16;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_ACTIVATION_WEIGHTED_BF16:
        return kEndpointActivationBf16 |
            kEndpointWeightedDownBf16;
    case QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_FINAL_BF16:
        return kEndpointWeightedDownBf16 |
            kEndpointFinalBf16;
    default:
        return kEndpointAllVllm;
    }
}

float routed_dot(const uint16_t *left,
                 const uint16_t *right,
                 size_t elements,
                 uint32_t arithmetic_mode) {
    return uses_hawkeye_accumulator(arithmetic_mode)
        ? hawkeye_dot(left, right, elements, arithmetic_mode)
        : avx512bf16_wave32_dot(left, right, elements);
}

float route_sum_vt4(const std::array<float, kRoutes> &route_values) {
    const float pair_0 = add_separate(
        route_values[0u],
        route_values[4u]
    );
    const float pair_1 = add_separate(
        route_values[1u],
        route_values[5u]
    );
    const float pair_2 = add_separate(
        route_values[2u],
        route_values[6u]
    );
    const float pair_3 = add_separate(
        route_values[3u],
        route_values[7u]
    );
    const float first = add_separate(pair_0, pair_1);
    const float second = add_separate(first, pair_2);
    return add_separate(second, pair_3);
}

float route_sum_sequential(const std::array<float, kRoutes> &route_values) {
    float combined = 0.0f;
    for (float value : route_values) {
        combined = add_separate(combined, value);
    }
    return combined;
}

bool struct_size_valid(uint32_t provided, size_t expected) {
    return provided >= expected && expected <= UINT32_MAX;
}

}  // namespace

struct qrt_q1_moe_avx512bf16_host_provider_t {
    std::unique_ptr<ProviderImpl> impl;
};

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_query_support(
    qrt_q1_moe_avx512bf16_host_support_t *out_support
) {
    if (out_support == nullptr ||
        !struct_size_valid(out_support->struct_size, sizeof(*out_support))) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    const uint32_t caller_size = out_support->struct_size;
    const CpuFeatures features = query_cpu_features();
    *out_support = qrt_q1_moe_avx512bf16_host_support_t{};
    out_support->struct_size = caller_size;
    out_support->abi_version = QRT_Q1_MOE_AVX512BF16_HOST_ABI_VERSION;
    out_support->cpuid_osxsave = features.osxsave ? 1u : 0u;
    out_support->cpuid_avx = features.avx ? 1u : 0u;
    out_support->cpuid_avx512f = features.avx512f ? 1u : 0u;
    out_support->cpuid_avx512bf16 = features.avx512bf16 ? 1u : 0u;
    out_support->xcr0_zmm_state = features.zmm_state ? 1u : 0u;
    out_support->supported = features.supported() ? 1u : 0u;
    return features.supported()
        ? QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS
        : QRT_Q1_MOE_AVX512BF16_HOST_STATUS_FALLBACK_CACHE_MISS;
}

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_create(
    const qrt_q1_moe_avx512bf16_host_config_t *config,
    qrt_q1_moe_avx512bf16_host_provider_t **out_provider
) {
    if (out_provider != nullptr) {
        *out_provider = nullptr;
    }
    if (config == nullptr || out_provider == nullptr ||
        !struct_size_valid(config->struct_size, sizeof(*config)) ||
        config->abi_version != QRT_Q1_MOE_AVX512BF16_HOST_ABI_VERSION ||
        config->worker_count != kWorkers ||
        config->cache_entry_capacity == 0u ||
        config->cache_byte_capacity < kBytesPerExpertPair) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    if (!query_cpu_features().supported()) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    try {
        auto provider = std::make_unique<qrt_q1_moe_avx512bf16_host_provider_t>();
        provider->impl = std::make_unique<ProviderImpl>(*config);
        provider->impl->cumulative.struct_size =
            sizeof(provider->impl->cumulative);
        provider->impl->cumulative.abi_version =
            QRT_Q1_MOE_AVX512BF16_HOST_ABI_VERSION;
        *out_provider = provider.release();
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS;
    } catch (...) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
}

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_prewarm(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    const qrt_q1_moe_avx512bf16_host_prewarm_request_t *request,
    qrt_q1_moe_avx512bf16_host_prewarm_result_t *out_result
) {
    if (out_result == nullptr ||
        !struct_size_valid(out_result->struct_size, sizeof(*out_result))) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    const uint32_t caller_size = out_result->struct_size;
    *out_result = qrt_q1_moe_avx512bf16_host_prewarm_result_t{};
    out_result->struct_size = caller_size;
    const Clock::time_point start = Clock::now();
    try {
        if (provider == nullptr || provider->impl == nullptr || request == nullptr ||
            !struct_size_valid(request->struct_size, sizeof(*request)) ||
            request->layer_index >= 40u ||
            request->model_dir_utf8 == nullptr ||
            request->gate_up_shard_utf8 == nullptr ||
            request->down_shard_utf8 == nullptr ||
            request->gate_up_bytes_per_expert != kGateUpBytesPerExpert ||
            request->down_bytes_per_expert != kDownBytesPerExpert ||
            request->expert_ids_by_priority == nullptr ||
            request->expert_id_count == 0u ||
            request->expert_id_count > kExpertCount) {
            out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
            out_result->error_code =
                QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INVALID_ARGUMENT;
            out_result->elapsed_ns = elapsed_ns(start, Clock::now());
            return out_result->status;
        }

    ProviderImpl &impl = *provider->impl;
    std::lock_guard<std::mutex> operation_lock(impl.run_mutex);
    std::lock_guard<std::mutex> lock(impl.state_mutex);
    ++impl.cumulative.prewarm_call_count;
    out_result->requested_count = request->expert_id_count;
    impl.cumulative.prewarm_requested_entry_count += request->expert_id_count;
    std::wstring gate_model_key;
    std::wstring down_model_key;
    std::wstring gate_path;
    std::wstring down_path;
    if (!resolve_model_and_shard(
            request->model_dir_utf8,
            request->gate_up_shard_utf8,
            &gate_model_key,
            &gate_path
        ) ||
        !resolve_model_and_shard(
            request->model_dir_utf8,
            request->down_shard_utf8,
            &down_model_key,
            &down_path
        ) ||
        gate_model_key != down_model_key) {
        impl.set_error(
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_WIN32_PATH,
            GetLastError(),
            "prewarm could not resolve the UTF-8 model or shard path"
        );
        ++impl.cumulative.prewarm_error_count;
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = impl.last_error_code;
        out_result->win32_error = impl.last_win32_error;
        out_result->failed_count = request->expert_id_count;
        impl.cumulative.prewarm_failed_entry_count += request->expert_id_count;
        out_result->elapsed_ns = elapsed_ns(start, Clock::now());
        impl.cumulative.prewarm_elapsed_ns += out_result->elapsed_ns;
        return out_result->status;
    }
    TensorSourceIdentity requested_source;
    requested_source.mode = CacheSourceMode::file_mapping;
    requested_source.model_key = std::move(gate_model_key);
    requested_source.gate_path_key = lowercase_path(gate_path);
    requested_source.down_path_key = lowercase_path(down_path);
    requested_source.gate_tensor_absolute_begin =
        request->gate_up_tensor_absolute_begin;
    requested_source.down_tensor_absolute_begin =
        request->down_tensor_absolute_begin;
    requested_source.gate_bytes_per_expert =
        request->gate_up_bytes_per_expert;
    requested_source.down_bytes_per_expert = request->down_bytes_per_expert;
    requested_source.layer_index = request->layer_index;
    const bool source_changed = !impl.source_identity_set ||
        !impl.source_identity.matches(requested_source);
    if (source_changed) {
        for (const std::unique_ptr<CacheEntry> &entry : impl.entries) {
            if (entry->pin_count != 0u) {
                impl.set_error(
                    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_CACHE_BUSY,
                    0u,
                    "prewarm cannot replace a tensor source while cache entries are pinned"
                );
                ++impl.cumulative.prewarm_error_count;
                out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
                out_result->error_code = impl.last_error_code;
                out_result->failed_count = request->expert_id_count;
                impl.cumulative.prewarm_failed_entry_count +=
                    request->expert_id_count;
                out_result->elapsed_ns = elapsed_ns(start, Clock::now());
                impl.cumulative.prewarm_elapsed_ns += out_result->elapsed_ns;
                return out_result->status;
            }
        }
        impl.clear_entries();
        impl.source_identity.swap(requested_source);
        impl.source_identity_set = true;
        impl.bump_model_generation();
    }

    impl.clear_error();
    std::vector<std::pair<uint32_t, uint32_t>> protected_keys;
    protected_keys.reserve(request->expert_id_count);
    for (uint32_t priority = 0u;
         priority < request->expert_id_count;
         ++priority) {
        const uint32_t expert_id = request->expert_ids_by_priority[priority];
        if (expert_id >= kExpertCount) {
            ++out_result->failed_count;
            continue;
        }
        bool duplicate_priority = false;
        for (const auto &key : protected_keys) {
            if (key.first == request->layer_index && key.second == expert_id) {
                duplicate_priority = true;
                break;
            }
        }
        if (duplicate_priority) {
            ++out_result->already_present_count;
            continue;
        }
        const uint64_t gate_expert_begin =
            request->gate_up_tensor_absolute_begin +
            static_cast<uint64_t>(expert_id) * kGateUpBytesPerExpert;
        const uint64_t down_expert_begin =
            request->down_tensor_absolute_begin +
            static_cast<uint64_t>(expert_id) * kDownBytesPerExpert;
        if (gate_expert_begin < request->gate_up_tensor_absolute_begin ||
            down_expert_begin < request->down_tensor_absolute_begin) {
            ++out_result->failed_count;
            continue;
        }
        CacheEntry *existing = impl.find_entry(request->layer_index, expert_id);
        if (existing != nullptr && impl.source_matches(
                *existing,
                impl.source_identity.gate_path_key,
                gate_expert_begin,
                impl.source_identity.down_path_key,
                down_expert_begin
            )) {
            // A prewarm request is also a residency refresh.  The mapping can
            // survive while Windows trims its file-backed pages, so an LRU hit
            // alone does not make the next batch-1 expert invocation warm.
            // Repeat the configured prefetch/touch on both source-matched
            // views and account those bytes exactly like a newly mapped view.
            existing->gate_up.prefetch_and_touch(
                impl.config.prefetch_virtual_memory != 0u,
                impl.config.touch_mapped_pages != 0u,
                &out_result->prefetch_requested_bytes,
                &out_result->touched_bytes
            );
            existing->down.prefetch_and_touch(
                impl.config.prefetch_virtual_memory != 0u,
                impl.config.touch_mapped_pages != 0u,
                &out_result->prefetch_requested_bytes,
                &out_result->touched_bytes
            );
            existing->last_use = impl.next_clock();
            protected_keys.emplace_back(request->layer_index, expert_id);
            ++out_result->already_present_count;
            continue;
        }
        if (existing != nullptr) {
            if (existing->pin_count != 0u) {
                ++out_result->failed_count;
                continue;
            }
            for (size_t index = 0u; index < impl.entries.size(); ++index) {
                if (impl.entries[index].get() == existing) {
                    impl.entries.erase(
                        impl.entries.begin() + static_cast<std::ptrdiff_t>(index)
                    );
                    impl.cache_bytes -= kBytesPerExpertPair;
                    break;
                }
            }
        }
        while (impl.entries.size() >= impl.config.cache_entry_capacity ||
               impl.cache_bytes + kBytesPerExpertPair >
                   impl.config.cache_byte_capacity) {
            if (!impl.evict_one(protected_keys)) {
                ++out_result->failed_count;
                goto next_expert;
            }
            ++out_result->eviction_count;
        }
        try {
            auto entry = std::make_unique<CacheEntry>();
            entry->model_generation = impl.model_generation;
            entry->layer_index = request->layer_index;
            entry->expert_id = expert_id;
            entry->source_mode = CacheSourceMode::file_mapping;
            uint32_t win32_error = 0u;
            if (!entry->gate_up.map(
                    gate_path,
                    gate_expert_begin,
                    kGateUpBytesPerExpert,
                    &win32_error
                )) {
                impl.set_error(
                    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_WIN32_MAPPING,
                    win32_error,
                    "prewarm could not map the gate/up expert view"
                );
                ++out_result->failed_count;
                goto next_expert;
            }
            if (!entry->down.map(
                    down_path,
                    down_expert_begin,
                    kDownBytesPerExpert,
                    &win32_error
                )) {
                impl.set_error(
                    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_WIN32_MAPPING,
                    win32_error,
                    "prewarm could not map the down expert view"
                );
                ++out_result->failed_count;
                goto next_expert;
            }
            entry->gate_up.prefetch_and_touch(
                impl.config.prefetch_virtual_memory != 0u,
                impl.config.touch_mapped_pages != 0u,
                &out_result->prefetch_requested_bytes,
                &out_result->touched_bytes
            );
            entry->down.prefetch_and_touch(
                impl.config.prefetch_virtual_memory != 0u,
                impl.config.touch_mapped_pages != 0u,
                &out_result->prefetch_requested_bytes,
                &out_result->touched_bytes
            );
            entry->last_use = impl.next_clock();
            impl.entries.push_back(std::move(entry));
            impl.cache_bytes += kBytesPerExpertPair;
            impl.cache_peak_bytes = (std::max)(
                impl.cache_peak_bytes,
                impl.cache_bytes
            );
            impl.cache_peak_entries = (std::max)(
                impl.cache_peak_entries,
                static_cast<uint32_t>(impl.entries.size())
            );
            protected_keys.emplace_back(request->layer_index, expert_id);
            ++out_result->mapped_count;
            out_result->mapped_bytes += kBytesPerExpertPair;
        } catch (...) {
            impl.set_error(
                QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY,
                0u,
                "prewarm allocation failed"
            );
            ++out_result->failed_count;
        }
next_expert:
        continue;
    }

    out_result->cache_entry_count = static_cast<uint32_t>(impl.entries.size());
    out_result->cache_bytes = impl.cache_bytes;
    out_result->cache_peak_bytes = impl.cache_peak_bytes;
    out_result->elapsed_ns = elapsed_ns(start, Clock::now());
    impl.cumulative.prewarm_mapped_entry_count += out_result->mapped_count;
    impl.cumulative.prewarm_already_present_count +=
        out_result->already_present_count;
    impl.cumulative.prewarm_failed_entry_count += out_result->failed_count;
    impl.cumulative.prewarm_eviction_count += out_result->eviction_count;
    impl.cumulative.prewarm_mapped_bytes += out_result->mapped_bytes;
    impl.cumulative.prewarm_prefetch_requested_bytes +=
        out_result->prefetch_requested_bytes;
    impl.cumulative.prewarm_touched_bytes += out_result->touched_bytes;
    impl.cumulative.prewarm_elapsed_ns += out_result->elapsed_ns;
    if (out_result->failed_count == 0u) {
        ++impl.cumulative.prewarm_success_count;
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS;
        out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE;
        out_result->win32_error = 0u;
    } else {
        ++impl.cumulative.prewarm_error_count;
        if (impl.last_error_code == QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE) {
            impl.set_error(
                QRT_Q1_MOE_AVX512BF16_HOST_ERROR_CACHE_BUSY,
                0u,
                "prewarm capacity could not admit every priority entry"
            );
        }
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = impl.last_error_code;
        out_result->win32_error = impl.last_win32_error;
    }
    return out_result->status;
    } catch (const std::bad_alloc &) {
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY;
        out_result->win32_error = 0u;
        out_result->elapsed_ns = elapsed_ns(start, Clock::now());
        return out_result->status;
    } catch (...) {
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL;
        out_result->win32_error = 0u;
        out_result->elapsed_ns = elapsed_ns(start, Clock::now());
        return out_result->status;
    }
}

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_prewarm_resident(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    const qrt_q1_moe_avx512bf16_host_resident_prewarm_request_t *request,
    qrt_q1_moe_avx512bf16_host_resident_prewarm_result_t *out_result
) {
    if (out_result == nullptr ||
        !struct_size_valid(out_result->struct_size, sizeof(*out_result))) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    const uint32_t caller_size = out_result->struct_size;
    *out_result = qrt_q1_moe_avx512bf16_host_resident_prewarm_result_t{};
    out_result->struct_size = caller_size;
    const Clock::time_point start = Clock::now();
    ProviderImpl *registered_impl = nullptr;
    bool resident_call_started = false;
    bool resident_result_accounted = false;
    try {
        if (provider == nullptr || provider->impl == nullptr || request == nullptr ||
            !struct_size_valid(request->struct_size, sizeof(*request)) ||
            request->layer_index >= 40u || request->source_generation == 0u ||
            request->gate_up_device_base == 0u ||
            request->down_device_base == 0u ||
            request->gate_up_bytes_per_expert != kGateUpBytesPerExpert ||
            request->down_bytes_per_expert != kDownBytesPerExpert ||
            request->copy_to_host == nullptr ||
            (request->host_allocate == nullptr) !=
                (request->host_free == nullptr) ||
            request->expert_ids_by_priority == nullptr ||
            request->expert_id_count == 0u ||
            request->expert_id_count > kExpertCount) {
            out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
            out_result->error_code =
                QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INVALID_ARGUMENT;
            out_result->elapsed_ns = elapsed_ns(start, Clock::now());
            return out_result->status;
        }

        ProviderImpl &impl = *provider->impl;
        std::lock_guard<std::mutex> operation_lock(impl.run_mutex);
        std::lock_guard<std::mutex> lock(impl.state_mutex);
        registered_impl = &impl;
        ++impl.cumulative.resident_prewarm_call_count;
        resident_call_started = true;
        out_result->requested_count = request->expert_id_count;
        impl.cumulative.resident_prewarm_requested_entry_count +=
            request->expert_id_count;

        TensorSourceIdentity requested_source;
        requested_source.mode = CacheSourceMode::resident_device;
        requested_source.gate_bytes_per_expert =
            request->gate_up_bytes_per_expert;
        requested_source.down_bytes_per_expert =
            request->down_bytes_per_expert;
        requested_source.layer_index = request->layer_index;
        requested_source.source_generation = request->source_generation;
        requested_source.gate_up_device_base = request->gate_up_device_base;
        requested_source.down_device_base = request->down_device_base;
        requested_source.copy_to_host = request->copy_to_host;
        requested_source.copy_context = request->copy_context;
        requested_source.host_allocate = request->host_allocate;
        requested_source.host_free = request->host_free;
        requested_source.allocator_context = request->allocator_context;
        const bool source_changed = !impl.source_identity_set ||
            !impl.source_identity.matches(requested_source);
        if (source_changed) {
            for (const std::unique_ptr<CacheEntry> &entry : impl.entries) {
                if (entry->pin_count != 0u) {
                    impl.set_error(
                        QRT_Q1_MOE_AVX512BF16_HOST_ERROR_CACHE_BUSY,
                        0u,
                        "resident prewarm cannot replace a tensor source while cache entries are pinned"
                    );
                    ++impl.cumulative.resident_prewarm_error_count;
                    out_result->status =
                        QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
                    out_result->error_code = impl.last_error_code;
                    out_result->failed_count = request->expert_id_count;
                    impl.cumulative.resident_prewarm_failed_entry_count +=
                        request->expert_id_count;
                    out_result->elapsed_ns = elapsed_ns(start, Clock::now());
                    impl.cumulative.resident_prewarm_elapsed_ns +=
                        out_result->elapsed_ns;
                    resident_result_accounted = true;
                    return out_result->status;
                }
            }
            impl.clear_entries();
            impl.source_identity.swap(requested_source);
            impl.source_identity_set = true;
            impl.bump_model_generation();
        }

        impl.clear_error();
        std::vector<std::pair<uint32_t, uint32_t>> protected_keys;
        protected_keys.reserve(request->expert_id_count);
        for (uint32_t priority = 0u;
             priority < request->expert_id_count;
             ++priority) {
            const uint32_t expert_id = request->expert_ids_by_priority[priority];
            if (expert_id >= kExpertCount) {
                ++out_result->failed_count;
                continue;
            }
            bool duplicate_priority = false;
            for (const auto &key : protected_keys) {
                if (key.first == request->layer_index && key.second == expert_id) {
                    duplicate_priority = true;
                    break;
                }
            }
            if (duplicate_priority) {
                ++out_result->already_present_count;
                continue;
            }
            const uint64_t gate_offset =
                static_cast<uint64_t>(expert_id) * kGateUpBytesPerExpert;
            const uint64_t down_offset =
                static_cast<uint64_t>(expert_id) * kDownBytesPerExpert;
            if (gate_offset > UINT64_MAX - request->gate_up_device_base ||
                down_offset > UINT64_MAX - request->down_device_base) {
                ++out_result->failed_count;
                continue;
            }
            const uint64_t gate_expert_begin =
                request->gate_up_device_base + gate_offset;
            const uint64_t down_expert_begin =
                request->down_device_base + down_offset;
            CacheEntry *existing = impl.find_entry(request->layer_index, expert_id);
            if (existing != nullptr && impl.resident_source_matches(
                    *existing,
                    gate_expert_begin,
                    down_expert_begin
                )) {
                // Owned resident copies cannot be trimmed out from underneath
                // the provider.  An LRU hit therefore performs no callback,
                // PrefetchVirtualMemory request, or page-touch pass.
                existing->last_use = impl.next_clock();
                protected_keys.emplace_back(request->layer_index, expert_id);
                ++out_result->already_present_count;
                continue;
            }
            if (existing != nullptr) {
                if (existing->pin_count != 0u) {
                    ++out_result->failed_count;
                    continue;
                }
                for (size_t index = 0u; index < impl.entries.size(); ++index) {
                    if (impl.entries[index].get() == existing) {
                        impl.entries.erase(
                            impl.entries.begin() +
                                static_cast<std::ptrdiff_t>(index)
                        );
                        impl.cache_bytes -= kBytesPerExpertPair;
                        break;
                    }
                }
            }
            while (impl.entries.size() >= impl.config.cache_entry_capacity ||
                   impl.cache_bytes + kBytesPerExpertPair >
                       impl.config.cache_byte_capacity) {
                if (!impl.evict_one(protected_keys)) {
                    ++out_result->failed_count;
                    goto next_resident_expert;
                }
                ++out_result->eviction_count;
            }
            try {
                auto entry = std::make_unique<CacheEntry>();
                entry->model_generation = impl.model_generation;
                entry->layer_index = request->layer_index;
                entry->expert_id = expert_id;
                entry->source_mode = CacheSourceMode::resident_device;
                entry->gate_up_device_begin = gate_expert_begin;
                entry->down_device_begin = down_expert_begin;
                if (!entry->resident_pair.allocate(
                        kBytesPerExpertPair,
                        request->host_allocate,
                        request->host_free,
                        request->allocator_context,
                        &out_result->allocator_error,
                        &out_result->allocation_call_count,
                        &out_result->allocated_bytes,
                        &out_result->allocation_elapsed_ns
                    )) {
                    impl.set_error(
                        request->host_allocate != nullptr
                            ? QRT_Q1_MOE_AVX512BF16_HOST_ERROR_RESIDENT_ALLOCATION
                            : QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY,
                        0u,
                        request->host_allocate != nullptr
                            ? "resident prewarm host allocator failed"
                            : "resident prewarm aligned allocation failed"
                    );
                    ++out_result->failed_count;
                    goto next_resident_expert;
                }

                int32_t callback_error = 0;
                Clock::time_point copy_start = Clock::now();
                ++out_result->copy_call_count;
                try {
                    callback_error = request->copy_to_host(
                        request->copy_context,
                        entry->resident_pair.data(),
                        gate_expert_begin,
                        kGateUpBytesPerExpert
                    );
                } catch (...) {
                    callback_error = (std::numeric_limits<int32_t>::min)();
                }
                out_result->copy_elapsed_ns +=
                    elapsed_ns(copy_start, Clock::now());
                if (callback_error != 0) {
                    out_result->callback_error = callback_error;
                    impl.set_resident_copy_error(
                        callback_error,
                        "resident prewarm gate/up device copy failed"
                    );
                    ++out_result->failed_count;
                    goto next_resident_expert;
                }
                out_result->copied_bytes += kGateUpBytesPerExpert;

                copy_start = Clock::now();
                ++out_result->copy_call_count;
                try {
                    callback_error = request->copy_to_host(
                        request->copy_context,
                        static_cast<unsigned char *>(
                            entry->resident_pair.data()
                        ) + kGateUpBytesPerExpert,
                        down_expert_begin,
                        kDownBytesPerExpert
                    );
                } catch (...) {
                    callback_error = (std::numeric_limits<int32_t>::min)();
                }
                out_result->copy_elapsed_ns +=
                    elapsed_ns(copy_start, Clock::now());
                if (callback_error != 0) {
                    out_result->callback_error = callback_error;
                    impl.set_resident_copy_error(
                        callback_error,
                        "resident prewarm down device copy failed"
                    );
                    ++out_result->failed_count;
                    goto next_resident_expert;
                }
                out_result->copied_bytes += kDownBytesPerExpert;

                entry->last_use = impl.next_clock();
                impl.entries.push_back(std::move(entry));
                impl.cache_bytes += kBytesPerExpertPair;
                impl.cache_peak_bytes = (std::max)(
                    impl.cache_peak_bytes,
                    impl.cache_bytes
                );
                impl.cache_peak_entries = (std::max)(
                    impl.cache_peak_entries,
                    static_cast<uint32_t>(impl.entries.size())
                );
                protected_keys.emplace_back(request->layer_index, expert_id);
                ++out_result->copied_count;
            } catch (const std::bad_alloc &) {
                impl.set_error(
                    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY,
                    0u,
                    "resident prewarm allocation failed"
                );
                ++out_result->failed_count;
            } catch (...) {
                impl.set_error(
                    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL,
                    0u,
                    "resident prewarm cache fill failed"
                );
                ++out_result->failed_count;
            }
next_resident_expert:
            continue;
        }

        out_result->cache_entry_count =
            static_cast<uint32_t>(impl.entries.size());
        out_result->cache_bytes = impl.cache_bytes;
        out_result->cache_peak_bytes = impl.cache_peak_bytes;
        out_result->elapsed_ns = elapsed_ns(start, Clock::now());
        if (out_result->failed_count == 0u) {
            out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS;
            out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE;
            out_result->callback_error = 0;
        } else {
            if (impl.last_error_code == QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE) {
                impl.set_error(
                    QRT_Q1_MOE_AVX512BF16_HOST_ERROR_CACHE_BUSY,
                    0u,
                    "resident prewarm could not admit every priority entry"
                );
            }
            out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
            out_result->error_code = impl.last_error_code;
            out_result->callback_error = impl.last_resident_callback_error;
        }
        impl.cumulative.resident_prewarm_copied_entry_count +=
            out_result->copied_count;
        impl.cumulative.resident_prewarm_already_present_count +=
            out_result->already_present_count;
        impl.cumulative.resident_prewarm_failed_entry_count +=
            out_result->failed_count;
        impl.cumulative.resident_prewarm_eviction_count +=
            out_result->eviction_count;
        impl.cumulative.resident_copy_call_count += out_result->copy_call_count;
        impl.cumulative.resident_copied_bytes += out_result->copied_bytes;
        impl.cumulative.resident_copy_elapsed_ns += out_result->copy_elapsed_ns;
        impl.cumulative.resident_prewarm_elapsed_ns += out_result->elapsed_ns;
        if (out_result->failed_count == 0u) {
            ++impl.cumulative.resident_prewarm_success_count;
        } else {
            ++impl.cumulative.resident_prewarm_error_count;
        }
        resident_result_accounted = true;
        return out_result->status;
    } catch (const std::bad_alloc &) {
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY;
        out_result->elapsed_ns = elapsed_ns(start, Clock::now());
        account_resident_prewarm_exception(
            registered_impl,
            request,
            out_result,
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_OUT_OF_MEMORY,
            "resident prewarm allocation escaped the cache-fill loop",
            resident_call_started,
            resident_result_accounted
        );
        return out_result->status;
    } catch (...) {
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL;
        out_result->elapsed_ns = elapsed_ns(start, Clock::now());
        account_resident_prewarm_exception(
            registered_impl,
            request,
            out_result,
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL,
            "resident prewarm failed outside the cache-fill loop",
            resident_call_started,
            resident_result_accounted
        );
        return out_result->status;
    }
}

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_run(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    const qrt_q1_moe_avx512bf16_host_run_request_t *request,
    qrt_q1_moe_avx512bf16_host_run_result_t *out_result
) {
    if (out_result == nullptr ||
        !struct_size_valid(out_result->struct_size, sizeof(*out_result))) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    const uint32_t caller_size = out_result->struct_size;
    *out_result = qrt_q1_moe_avx512bf16_host_run_result_t{};
    out_result->struct_size = caller_size;
    const Clock::time_point total_start = Clock::now();
    if (provider == nullptr || provider->impl == nullptr || request == nullptr ||
        !struct_size_valid(request->struct_size, sizeof(*request)) ||
        request->layer_index >= 40u || request->input_bf16 == nullptr ||
        request->expert_ids == nullptr || request->route_weights_f32 == nullptr ||
        request->output_f32 == nullptr ||
        request->arithmetic_mode >
            QRT_Q1_MOE_AVX512BF16_HOST_ARITHMETIC_AVX512_WAVE32_WEIGHTED_FINAL_BF16) {
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code =
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INVALID_ARGUMENT;
        out_result->total_elapsed_ns = elapsed_ns(total_start, Clock::now());
        return out_result->status;
    }
    out_result->arithmetic_mode = request->arithmetic_mode;

    ProviderImpl &impl = *provider->impl;
    std::lock_guard<std::mutex> run_lock(impl.run_mutex);
    std::array<CacheEntry *, kRoutes> route_entries{};
    const Clock::time_point lookup_start = Clock::now();
    {
        std::lock_guard<std::mutex> state_lock(impl.state_mutex);
        ++impl.cumulative.run_call_count;
        impl.clear_error();
        bool valid = bf16_values_finite(request->input_bf16, kHidden);
        for (size_t route = 0u; route < kRoutes; ++route) {
            valid = valid && request->expert_ids[route] < kExpertCount &&
                std::isfinite(request->route_weights_f32[route]) &&
                request->route_weights_f32[route] >= 0.0f;
            for (size_t prior = 0u; prior < route; ++prior) {
                valid = valid &&
                    request->expert_ids[prior] != request->expert_ids[route];
            }
        }
        if (!valid) {
            impl.set_error(
                QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONFINITE,
                0u,
                "run input, routes, or route weights are invalid"
            );
            ++impl.cumulative.run_error_count;
            out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
            out_result->error_code = impl.last_error_code;
            out_result->lookup_elapsed_ns = elapsed_ns(lookup_start, Clock::now());
            out_result->total_elapsed_ns = elapsed_ns(total_start, Clock::now());
            impl.cumulative.run_lookup_elapsed_ns += out_result->lookup_elapsed_ns;
            impl.cumulative.run_total_elapsed_ns += out_result->total_elapsed_ns;
            return out_result->status;
        }
        for (size_t route = 0u; route < kRoutes; ++route) {
            CacheEntry *entry = impl.find_entry(
                request->layer_index,
                request->expert_ids[route]
            );
            ++impl.cumulative.run_route_lookup_count;
            if (entry == nullptr) {
                ++out_result->route_cache_misses;
                ++impl.cumulative.run_route_miss_count;
            } else {
                route_entries[route] = entry;
                ++entry->pin_count;
                entry->last_use = impl.next_clock();
                ++out_result->route_cache_hits;
                ++impl.cumulative.run_route_hit_count;
            }
        }
        out_result->cache_entry_count =
            static_cast<uint32_t>(impl.entries.size());
        out_result->cache_bytes = impl.cache_bytes;
        out_result->cache_peak_bytes = impl.cache_peak_bytes;
    }
    out_result->lookup_elapsed_ns = elapsed_ns(lookup_start, Clock::now());
    if (out_result->route_cache_misses != 0u) {
        std::lock_guard<std::mutex> state_lock(impl.state_mutex);
        for (CacheEntry *entry : route_entries) {
            if (entry != nullptr) {
                --entry->pin_count;
            }
        }
        ++impl.cumulative.run_fallback_count;
        impl.set_error(
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE,
            0u,
            "cache miss: retain the GPU route for this layer"
        );
        out_result->status =
            QRT_Q1_MOE_AVX512BF16_HOST_STATUS_FALLBACK_CACHE_MISS;
        out_result->all_routes_hit = 0u;
        out_result->total_elapsed_ns = elapsed_ns(total_start, Clock::now());
        impl.cumulative.run_lookup_elapsed_ns += out_result->lookup_elapsed_ns;
        impl.cumulative.run_total_elapsed_ns += out_result->total_elapsed_ns;
        return out_result->status;
    }

    out_result->all_routes_hit = 1u;
    const Clock::time_point gate_start = Clock::now();
    try {
        impl.pool.parallel_for(
            kRoutes * kIntermediate,
            [&](size_t route_row) {
                const size_t route = route_row / kIntermediate;
                const size_t row = route_row % kIntermediate;
                const uint16_t *gate_up = route_entries[route]->gate_up_bf16();
                const uint16_t *gate_row = gate_up + row * kHidden;
                const uint16_t *up_row = gate_up +
                    kGateElementsPerExpert + row * kHidden;
                const uint16_t gate_bits = float_to_bf16(
                    routed_dot(
                        request->input_bf16,
                        gate_row,
                        kHidden,
                        request->arithmetic_mode
                    )
                );
                const uint16_t up_bits = float_to_bf16(
                    routed_dot(
                        request->input_bf16,
                        up_row,
                        kHidden,
                        request->arithmetic_mode
                    )
                );
                impl.gate[route_row] = gate_bits;
                impl.up[route_row] = up_bits;
                impl.activated[route_row] =
                    (endpoint_mask_for_mode(request->arithmetic_mode) &
                        kEndpointActivationBf16) != 0u
                        ? vllm_activation_endpoint(gate_bits, up_bits)
                        : activation_endpoint(gate_bits, up_bits);
            }
        );
    } catch (...) {
        std::lock_guard<std::mutex> state_lock(impl.state_mutex);
        for (CacheEntry *entry : route_entries) {
            --entry->pin_count;
        }
        impl.set_error(
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL,
            0u,
            "gate/up worker dispatch failed"
        );
        ++impl.cumulative.run_error_count;
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = impl.last_error_code;
        out_result->gate_up_elapsed_ns = elapsed_ns(gate_start, Clock::now());
        out_result->total_elapsed_ns = elapsed_ns(total_start, Clock::now());
        return out_result->status;
    }
    out_result->gate_up_elapsed_ns = elapsed_ns(gate_start, Clock::now());

    const Clock::time_point down_start = Clock::now();
    try {
        impl.pool.parallel_for(
            kHidden,
            [&](size_t row) {
                const uint32_t endpoint_mask =
                    endpoint_mask_for_mode(request->arithmetic_mode);
                std::array<float, kRoutes> route_values{};
                for (size_t route = 0u; route < kRoutes; ++route) {
                    const uint16_t *down_row =
                        route_entries[route]->down_bf16() +
                        row * kIntermediate;
                    const float down = routed_dot(
                        impl.activated.data() + route * kIntermediate,
                        down_row,
                        kIntermediate,
                        request->arithmetic_mode
                    );
                    if ((endpoint_mask &
                            kEndpointWeightedDownBf16) != 0u) {
                        route_values[route] = bf16_to_float(float_to_bf16(
                            mul_separate(
                                down,
                                request->route_weights_f32[route]
                            )
                        ));
                    } else {
                        route_values[route] = mul_separate(
                            bf16_to_float(float_to_bf16(down)),
                            request->route_weights_f32[route]
                        );
                    }
                }
                float combined =
                    (endpoint_mask & kEndpointRouteSumVt4) != 0u
                        ? route_sum_vt4(route_values)
                        : route_sum_sequential(route_values);
                if ((endpoint_mask & kEndpointFinalBf16) != 0u) {
                    combined = bf16_to_float(float_to_bf16(combined));
                }
                request->output_f32[row] = combined;
            }
        );
    } catch (...) {
        std::lock_guard<std::mutex> state_lock(impl.state_mutex);
        for (CacheEntry *entry : route_entries) {
            --entry->pin_count;
        }
        impl.set_error(
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_INTERNAL,
            0u,
            "down/combine worker dispatch failed"
        );
        ++impl.cumulative.run_error_count;
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code = impl.last_error_code;
        out_result->down_combine_elapsed_ns = elapsed_ns(down_start, Clock::now());
        out_result->total_elapsed_ns = elapsed_ns(total_start, Clock::now());
        return out_result->status;
    }
    out_result->down_combine_elapsed_ns = elapsed_ns(down_start, Clock::now());
    bool finite = true;
    for (size_t row = 0u; row < kHidden; ++row) {
        finite = finite && std::isfinite(request->output_f32[row]);
    }

    out_result->weight_bytes_consumed = kRoutes * kBytesPerExpertPair;
    out_result->input_bytes = kHidden * sizeof(uint16_t);
    out_result->output_bytes = kHidden * sizeof(float);
    out_result->output_fnv1a64 = fnv1a64(
        request->output_f32,
        kHidden * sizeof(float)
    );
    out_result->total_elapsed_ns = elapsed_ns(total_start, Clock::now());
    {
        std::lock_guard<std::mutex> state_lock(impl.state_mutex);
        for (CacheEntry *entry : route_entries) {
            --entry->pin_count;
        }
        impl.cumulative.run_lookup_elapsed_ns += out_result->lookup_elapsed_ns;
        impl.cumulative.run_gate_up_elapsed_ns += out_result->gate_up_elapsed_ns;
        impl.cumulative.run_down_combine_elapsed_ns +=
            out_result->down_combine_elapsed_ns;
        impl.cumulative.run_total_elapsed_ns += out_result->total_elapsed_ns;
        if (finite) {
            ++impl.cumulative.run_success_count;
            impl.cumulative.run_weight_bytes_consumed +=
                out_result->weight_bytes_consumed;
            impl.cumulative.run_input_bytes += out_result->input_bytes;
            impl.cumulative.run_output_bytes += out_result->output_bytes;
            impl.clear_error();
        } else {
            ++impl.cumulative.run_error_count;
            impl.set_error(
                QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONFINITE,
                0u,
                "run produced a non-finite output"
            );
        }
    }
    if (!finite) {
        out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
        out_result->error_code =
            QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONFINITE;
        return out_result->status;
    }
    out_result->status = QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS;
    out_result->error_code = QRT_Q1_MOE_AVX512BF16_HOST_ERROR_NONE;
    return out_result->status;
}

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_get_stats(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    qrt_q1_moe_avx512bf16_host_stats_t *out_stats
) {
    if (provider == nullptr || provider->impl == nullptr || out_stats == nullptr ||
        !struct_size_valid(out_stats->struct_size, sizeof(*out_stats))) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    std::lock_guard<std::mutex> lock(provider->impl->state_mutex);
    provider->impl->snapshot_stats(out_stats);
    return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS;
}

extern "C" const char *QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_last_error(
    qrt_q1_moe_avx512bf16_host_provider_t *provider
) {
    static thread_local std::array<char, 512u> snapshot{};
    const auto copy_snapshot = [](std::array<char, 512u> *destination,
                                  const char *source,
                                  size_t source_bytes) {
        const size_t copied = (std::min)(source_bytes, destination->size() - 1u);
        if (copied != 0u) {
            std::memcpy(destination->data(), source, copied);
        }
        (*destination)[copied] = '\0';
    };
    if (provider == nullptr || provider->impl == nullptr) {
        constexpr char message[] = "provider handle is null";
        copy_snapshot(&snapshot, message, sizeof(message) - 1u);
        return snapshot.data();
    }
    try {
        std::lock_guard<std::mutex> lock(provider->impl->state_mutex);
        const std::string &message = provider->impl->last_error;
        copy_snapshot(&snapshot, message.data(), message.size());
        return snapshot.data();
    } catch (...) {
        return "provider last-error snapshot failed";
    }
}

extern "C" int32_t QRT_Q1_MOE_AVX512BF16_HOST_CALL
qrt_q1_moe_avx512bf16_host_release(
    qrt_q1_moe_avx512bf16_host_provider_t *provider,
    qrt_q1_moe_avx512bf16_host_stats_t *out_final_stats
) {
    if (provider == nullptr || provider->impl == nullptr) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    if (out_final_stats != nullptr &&
        !struct_size_valid(out_final_stats->struct_size, sizeof(*out_final_stats))) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
    try {
        // prewarm and run both take run_mutex before state_mutex.  Taking the
        // same pair here waits for either active operation to finish before
        // any mappings, scratch storage, or worker threads are destroyed.
        std::unique_lock<std::mutex> operation_lock(provider->impl->run_mutex);
        std::unique_lock<std::mutex> state_lock(provider->impl->state_mutex);
        if (out_final_stats != nullptr) {
            provider->impl->snapshot_stats(out_final_stats);
        }
        state_lock.unlock();
        operation_lock.unlock();
        delete provider;
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_SUCCESS;
    } catch (...) {
        return QRT_Q1_MOE_AVX512BF16_HOST_STATUS_ERROR;
    }
}
