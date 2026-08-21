#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#ifndef QRT_TRITON_MOE_BLOCK_M
#define QRT_TRITON_MOE_BLOCK_M 64
#endif
#ifndef QRT_TRITON_MOE_BLOCK_N
#define QRT_TRITON_MOE_BLOCK_N 64
#endif
#ifndef QRT_TRITON_MOE_THREADS
#define QRT_TRITON_MOE_THREADS 128
#endif
#ifndef QRT_TRITON_MOE_GATE_BLOCK_N
#define QRT_TRITON_MOE_GATE_BLOCK_N QRT_TRITON_MOE_BLOCK_N
#endif
#ifndef QRT_TRITON_MOE_DOWN_BLOCK_N
#define QRT_TRITON_MOE_DOWN_BLOCK_N QRT_TRITON_MOE_BLOCK_N
#endif
#ifndef QRT_TRITON_MOE_ROUTE_THREADS
#define QRT_TRITON_MOE_ROUTE_THREADS QRT_TRITON_MOE_THREADS
#endif
#ifndef QRT_TRITON_MOE_GATE_THREADS
#define QRT_TRITON_MOE_GATE_THREADS QRT_TRITON_MOE_THREADS
#endif
#ifndef QRT_TRITON_MOE_DOWN_THREADS
#define QRT_TRITON_MOE_DOWN_THREADS QRT_TRITON_MOE_THREADS
#endif
#ifndef QRT_TRITON_MOE_GATE_SHARED_BYTES
#define QRT_TRITON_MOE_GATE_SHARED_BYTES 8192
#endif
#ifndef QRT_TRITON_MOE_DOWN_SHARED_BYTES
#define QRT_TRITON_MOE_DOWN_SHARED_BYTES 8192
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_ROUTED
#define QRT_TRITON_MOE_NATIVE_WMMA_ROUTED 0
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_GATE
#define QRT_TRITON_MOE_NATIVE_WMMA_GATE QRT_TRITON_MOE_NATIVE_WMMA_ROUTED
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_DOWN
#define QRT_TRITON_MOE_NATIVE_WMMA_DOWN QRT_TRITON_MOE_NATIVE_WMMA_ROUTED
#endif
#ifndef QRT_TRITON_MOE_TRANSPOSED_ROUTER
#define QRT_TRITON_MOE_TRANSPOSED_ROUTER 0
#endif
#ifndef QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE
#define QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE 0
#endif
#ifndef QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS
#define QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS 16
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
#define QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8 0
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_GROUP_VALUES
#define QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_GROUP_VALUES 128
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_FP16_SCALES
#define QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_FP16_SCALES 0
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_PALETTE
#define QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_PALETTE 0
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
#define QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE 0
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE && \
    !QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_PALETTE
#error "row-palette smoke requires lossless-palette support"
#endif
namespace {

#if defined(_WIN32)
using ProviderModule = HMODULE;

ProviderModule load_provider_module(const char *path) {
    return LoadLibraryA(path);
}

void *load_provider_symbol(ProviderModule module, const char *name) {
    return reinterpret_cast<void *>(GetProcAddress(module, name));
}

std::string provider_load_error() {
    return "win32_error=" + std::to_string(GetLastError());
}

void close_provider_module(ProviderModule module) {
    (void)FreeLibrary(module);
}
#else
using ProviderModule = void *;

ProviderModule load_provider_module(const char *path) {
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

void *load_provider_symbol(ProviderModule module, const char *name) {
    return dlsym(module, name);
}

std::string provider_load_error() {
    const char *message = dlerror();
    return message != nullptr ? message : "unknown dlopen error";
}

void close_provider_module(ProviderModule module) {
    (void)dlclose(module);
}
#endif

std::string local_host_name() {
    std::array<char, 256> name{};
#if defined(_WIN32)
    DWORD size = static_cast<DWORD>(name.size());
    if (GetComputerNameA(name.data(), &size) != 0 && size != 0u) {
        return std::string(name.data(), size);
    }
#else
    if (gethostname(name.data(), name.size() - 1u) == 0 && name[0] != '\0') {
        name.back() = '\0';
        return std::string(name.data());
    }
#endif
    return "unknown";
}

constexpr uint32_t kTokens = 8192;
constexpr uint32_t kTopK = 8;
constexpr uint32_t kRoutes = kTokens * kTopK;
constexpr uint32_t kExperts = 256;
constexpr uint32_t kActiveExperts = 194;
constexpr uint32_t kHidden = 2048;
constexpr uint32_t kIntermediate = 512;
constexpr uint32_t kGateUpRows = 2 * kIntermediate;
constexpr uint32_t kBlockM = QRT_TRITON_MOE_BLOCK_M;
constexpr uint32_t kGateBlockN = QRT_TRITON_MOE_GATE_BLOCK_N;
constexpr uint32_t kDownBlockN = QRT_TRITON_MOE_DOWN_BLOCK_N;
constexpr uint32_t kMaxSortedRoutes = kRoutes + kExperts * kBlockM - kTopK;
constexpr uint32_t kMaxRouteBlocks =
    (kMaxSortedRoutes + kBlockM - 1u) / kBlockM;
constexpr uint32_t kRouteThreads = QRT_TRITON_MOE_ROUTE_THREADS;
constexpr uint32_t kGateThreads = QRT_TRITON_MOE_GATE_THREADS;
constexpr uint32_t kDownThreads = QRT_TRITON_MOE_DOWN_THREADS;
constexpr uint32_t kGateUpGridN = kGateUpRows / kGateBlockN;
constexpr uint32_t kDownGridN = kHidden / kDownBlockN;
constexpr uint32_t kGateSharedBytes = QRT_TRITON_MOE_GATE_SHARED_BYTES;
constexpr uint32_t kDownSharedBytes = QRT_TRITON_MOE_DOWN_SHARED_BYTES;
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
constexpr uint32_t kLosslessRowPaletteBytes = 16u;
constexpr uint32_t kLosslessOverflowSentinel = UINT32_MAX;
__host__ __device__ constexpr size_t lossless_row_packed_bytes(
    uint32_t row_values
) {
    return static_cast<size_t>(row_values) + row_values / 2u +
        kLosslessRowPaletteBytes;
}
#endif
static_assert(kGateUpRows % kGateBlockN == 0u);
static_assert(kHidden % kDownBlockN == 0u);
// Submit one call beyond the fixed event ring. This is the smallest chain
// that proves a full ring applies bounded backpressure instead of rejecting
// an otherwise valid long-context request.
constexpr uint32_t kFullV3AsyncChain =
    QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS + 1u;
static_assert(kFullV3AsyncChain > QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS);
constexpr uint32_t kExpectedProviderBackendMask =
    (QRT_TRITON_MOE_NATIVE_WMMA_GATE ? UINT32_C(1) : UINT32_C(0)) |
    (QRT_TRITON_MOE_NATIVE_WMMA_DOWN ? UINT32_C(2) : UINT32_C(0)) |
    (QRT_TRITON_MOE_TRANSPOSED_ROUTER ? UINT32_C(4) : UINT32_C(0)) |
    (QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE ?
        UINT32_C(8) : UINT32_C(0));
constexpr uint64_t kExpectedFullProviderHash =
    UINT64_C(0xc8b9f2290b8bbd3);
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8 && \
    QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_FP16_SCALES
constexpr float kWeightInt8ProviderOutputTolerance = 0.0625f;
#else
constexpr float kWeightInt8ProviderOutputTolerance = 0.03125f;
#endif

struct ModuleKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    uint32_t grid_x = 0;
    uint32_t threads = 0;
    uint32_t shared_bytes = 0;
};

int fail(const char *stage, hipError_t status) {
    std::cerr << "q8192_triton_selected_moe_smoke stage=" << stage
              << " hip_status=" << static_cast<int>(status)
              << " hip_error=" << hipGetErrorString(status) << std::endl;
    return 1;
}

uint16_t float_to_bf16(float value) {
    union {
        float f;
        uint32_t u;
    } bits{value};
    bits.u += UINT32_C(0x7fff) + ((bits.u >> 16u) & 1u);
    return static_cast<uint16_t>(bits.u >> 16u);
}

float bf16_to_float(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits{static_cast<uint32_t>(value) << 16u};
    return bits.f;
}

uint64_t fnv1a64_f32(const std::vector<float> &values) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
    const size_t byte_count = values.size() * sizeof(float);
    for (size_t index = 0; index < byte_count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

__device__ uint16_t device_float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}

__device__ float device_bf16_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16u);
}

__global__ void fill_input_kernel(float *input, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        input[index] = 1.0f;
    }
}

__global__ void fill_nonuniform_input_kernel(float *input, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const size_t token = index / kHidden;
        const size_t column = index - token * kHidden;
        input[index] = 0.75f +
            static_cast<float>(token % 13u) * 0.01f +
            static_cast<float>(column % 17u) * 0.005f;
    }
}

__global__ void fill_bf16_matrix_pattern_kernel(
    uint16_t *weights,
    size_t elements,
    uint32_t columns,
    uint32_t salt,
    float denominator
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        const uint32_t row = static_cast<uint32_t>(index / columns);
        const uint32_t column = static_cast<uint32_t>(index % columns);
        const uint32_t numerator =
            1u + ((row * 13u + column * 7u + salt) % 11u);
        weights[index] = device_float_to_bf16(
            static_cast<float>(numerator) / denominator
        );
    }
}

__global__ void fill_router_debug_oracle_kernel(
    uint16_t *weights,
    size_t elements
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const uint32_t expert = static_cast<uint32_t>(index / kHidden);
    const bool selected = expert >= 10u && expert <= 87u &&
        (expert - 10u) % 11u == 0u;
    weights[index] = device_float_to_bf16(
        (selected ? 2.0f : 1.0f) / static_cast<float>(kHidden)
    );
}

__global__ void convert_input_kernel(
    const float *input,
    uint16_t *output,
    size_t elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = device_float_to_bf16(input[index]);
    }
}

__global__ void fill_gate_up_kernel(uint16_t *weights, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    constexpr size_t kElementsPerExpert =
        static_cast<size_t>(kGateUpRows) * kHidden;
    const uint32_t expert = static_cast<uint32_t>(index / kElementsPerExpert);
    const size_t within_expert = index % kElementsPerExpert;
    const uint32_t row = static_cast<uint32_t>(within_expert / kHidden);
    const float value = row < kIntermediate
        ? static_cast<float>(expert % 7u + 1u) / static_cast<float>(kHidden)
        : static_cast<float>(expert % 5u + 1u) /
              static_cast<float>(2u * kHidden);
    weights[index] = device_float_to_bf16(value);
}

__global__ void fill_down_kernel(uint16_t *weights, size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    constexpr size_t kElementsPerExpert =
        static_cast<size_t>(kHidden) * kIntermediate;
    const uint32_t expert = static_cast<uint32_t>(index / kElementsPerExpert);
    const float value = static_cast<float>(expert % 3u + 1u) /
        static_cast<float>(kIntermediate);
    weights[index] = device_float_to_bf16(value);
}

#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
// The smoke's synthetic routed matrices are constant within each row. Pack
// those rows on-device so the provider test exercises its compact-weight ABI
// without adding a second multi-gigabyte host copy to the validation path.
__global__ void pack_constant_lossless_rows_kernel(
    const uint16_t *source,
    uint8_t *packed_rows,
    uint32_t *overflow_indices,
    uint32_t row_values,
    uint32_t row_count
) {
    const size_t packed_stride = lossless_row_packed_bytes(row_values);
    for (uint32_t row = blockIdx.x; row < row_count; row += gridDim.x) {
        const uint16_t *const source_row =
            source + static_cast<size_t>(row) * row_values;
        uint8_t *const packed_row =
            packed_rows + static_cast<size_t>(row) * packed_stride;
        for (uint32_t column = threadIdx.x; column < row_values;
             column += blockDim.x) {
            packed_row[column] = static_cast<uint8_t>(source_row[column]);
        }
        for (uint32_t code = threadIdx.x; code < row_values / 2u;
             code += blockDim.x) {
            packed_row[row_values + code] = 0u;
        }
        if (threadIdx.x < kLosslessRowPaletteBytes) {
            packed_row[row_values + row_values / 2u + threadIdx.x] =
                threadIdx.x == 0u
                    ? static_cast<uint8_t>(source_row[0] >> 8u)
                    : 0u;
        }
        if (threadIdx.x == 0u) {
            overflow_indices[row] = kLosslessOverflowSentinel;
        }
    }
}
#endif

#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
constexpr uint32_t kWeightInt8GroupValues =
    QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_GROUP_VALUES;
constexpr uint32_t kWeightInt8WaveThreads = 32u;
constexpr uint32_t kWeightInt8BlockThreads = 256u;
constexpr uint32_t kWeightInt8WavesPerBlock =
    kWeightInt8BlockThreads / kWeightInt8WaveThreads;
constexpr uint32_t kWeightInt8ValuesPerLane =
    kWeightInt8GroupValues / kWeightInt8WaveThreads;
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_FP16_SCALES
using WeightInt8Scale = __half;
#else
using WeightInt8Scale = float;
#endif
constexpr uint32_t kWeightInt8ScaleBytes = sizeof(WeightInt8Scale);
static_assert(kWeightInt8ScaleBytes == 2u || kWeightInt8ScaleBytes == 4u);
static_assert(
    kWeightInt8GroupValues == 32u ||
        kWeightInt8GroupValues == 64u ||
        kWeightInt8GroupValues == 128u,
    "weight-int8 smoke supports group32, group64, or group128"
);
static_assert(kWeightInt8GroupValues % kWeightInt8WaveThreads == 0u);

__device__ __forceinline__ float weight_int8_wave_max(float value) {
#pragma unroll
    for (uint32_t offset = kWeightInt8WaveThreads / 2u;
         offset > 0u;
         offset >>= 1u) {
        value = fmaxf(
            value,
            __shfl_down(value, offset, kWeightInt8WaveThreads)
        );
    }
    return value;
}

__device__ __forceinline__ int weight_int8_quantize(
    float value,
    float inverse_scale
) {
    const float scaled = value * inverse_scale;
    const int rounded = static_cast<int>(
        scaled + (scaled >= 0.0f ? 0.5f : -0.5f)
    );
    return rounded < -127 ? -127 : (rounded > 127 ? 127 : rounded);
}

__global__ void quantize_weight_int8_rows_kernel(
    const uint16_t *source,
    int8_t *destination,
    WeightInt8Scale *scales,
    uint32_t row_size,
    uint32_t row_count
) {
    const uint32_t lane = threadIdx.x & (kWeightInt8WaveThreads - 1u);
    const uint32_t wave = threadIdx.x / kWeightInt8WaveThreads;
    const uint32_t group_count = row_size / kWeightInt8GroupValues;
    for (uint32_t row = blockIdx.x; row < row_count; row += gridDim.x) {
        const size_t row_base = static_cast<size_t>(row) * row_size;
        const size_t scale_base = static_cast<size_t>(row) * group_count;
        for (uint32_t group = wave;
             group < group_count;
             group += kWeightInt8WavesPerBlock) {
            float values[kWeightInt8ValuesPerLane];
            float local_maximum = 0.0f;
#pragma unroll
            for (uint32_t segment = 0u;
                 segment < kWeightInt8ValuesPerLane;
                 ++segment) {
                const uint32_t column = group * kWeightInt8GroupValues +
                    segment * kWeightInt8WaveThreads + lane;
                values[segment] = device_bf16_to_float(
                    source[row_base + column]
                );
                local_maximum = fmaxf(local_maximum, fabsf(values[segment]));
            }
            const float maximum = __shfl(
                weight_int8_wave_max(local_maximum),
                0u,
                kWeightInt8WaveThreads
            );
            const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
            const float inverse_scale = 1.0f / scale;
#pragma unroll
            for (uint32_t segment = 0u;
                 segment < kWeightInt8ValuesPerLane;
                 ++segment) {
                const uint32_t column = group * kWeightInt8GroupValues +
                    segment * kWeightInt8WaveThreads + lane;
                destination[row_base + column] = static_cast<int8_t>(
                    weight_int8_quantize(values[segment], inverse_scale)
                );
            }
            if (lane == 0u) {
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_FP16_SCALES
                scales[scale_base + group] = __float2half(scale);
#else
                scales[scale_base + group] = scale;
#endif
            }
        }
    }
}
#endif

__global__ void combine_route_order_kernel(
    const float *route_outputs,
    const float *topk_weights,
    float *outputs,
    size_t output_elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements) {
        return;
    }
    const size_t token = index / kHidden;
    const size_t column = index - token * kHidden;
    const size_t route_base = token * kTopK;
    float value = 0.0f;
    for (uint32_t route_order = 0; route_order < kTopK; ++route_order) {
        const size_t route = route_base + route_order;
        value += topk_weights[route] *
            route_outputs[route * kHidden + column];
    }
    outputs[index] = value;
}

bool load_kernel(
    const std::filesystem::path &directory,
    const char *stem,
    const char *symbol,
    uint32_t grid_x,
    uint32_t threads,
    uint32_t shared_bytes,
    ModuleKernel *kernel
) {
    const std::filesystem::path path =
        directory / (std::string("q8192_selected_moe_") + stem + ".hsaco");
    hipError_t status = hipModuleLoad(&kernel->module, path.string().c_str());
    if (status != hipSuccess) {
        (void)fail((std::string("hipModuleLoad(") + stem + ")").c_str(), status);
        return false;
    }
    status = hipModuleGetFunction(&kernel->function, kernel->module, symbol);
    if (status != hipSuccess) {
        (void)fail((std::string("hipModuleGetFunction(") + stem + ")").c_str(), status);
        return false;
    }
    kernel->grid_x = grid_x;
    kernel->threads = threads;
    kernel->shared_bytes = shared_bytes;
    return true;
}

hipError_t launch(ModuleKernel &kernel, std::vector<void *> arguments) {
    void *global_scratch = nullptr;
    void *profile_scratch = nullptr;
    arguments.push_back(&global_scratch);
    arguments.push_back(&profile_scratch);
    return hipModuleLaunchKernel(
        kernel.function,
        kernel.grid_x,
        1u,
        1u,
        kernel.threads,
        1u,
        1u,
        kernel.shared_bytes,
        nullptr,
        arguments.data(),
        nullptr
    );
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        std::cerr
            << "usage: q8192_triton_selected_moe_smoke KERNEL_DIR "
               "[REPETITIONS] [PROVIDER_DLL] [LIGHT_LOGICAL_TOKENS]\n";
        return 2;
    }
    const std::filesystem::path kernel_dir(argv[1]);
    const uint32_t repetitions = argc >= 3
        ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10))
        : 5u;
    if (repetitions == 0u) {
        std::cerr << "repetitions must be positive\n";
        return 2;
    }
    uint32_t light_logical_tokens = 0u;
    if (argc == 5) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(argv[4], &end, 10);
        if (end == argv[4] || end == nullptr || end[0] != '\0' ||
            parsed == 0u || parsed > kTokens) {
            std::cerr << "light logical tokens must be in 1..8192\n";
            return 2;
        }
        light_logical_tokens = static_cast<uint32_t>(parsed);
    }
    std::vector<uint32_t> light_token_sequence;
    if (light_logical_tokens != 0u) {
        const char *sequence_text =
            std::getenv("QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE");
        if (sequence_text == nullptr || sequence_text[0] == '\0') {
            light_token_sequence.push_back(light_logical_tokens);
        } else {
            const char *cursor = sequence_text;
            while (cursor[0] != '\0') {
                char *end = nullptr;
                const unsigned long parsed = std::strtoul(cursor, &end, 10);
                if (end == cursor || parsed == 0u || parsed > kTokens ||
                    (end[0] != '\0' && end[0] != ',')) {
                    std::cerr
                        << "QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE must be "
                           "a comma-separated list of values in 1..8192\n";
                    return 2;
                }
                light_token_sequence.push_back(
                    static_cast<uint32_t>(parsed)
                );
                if (end[0] == '\0') {
                    break;
                }
                cursor = end + 1;
                if (cursor[0] == '\0') {
                    std::cerr
                        << "QRT_PRODUCT_RADIUS_LIGHT_TOKEN_SEQUENCE must not "
                           "end with a comma\n";
                    return 2;
                }
            }
        }
    }

    hipError_t status = hipInit(0);
    if (status != hipSuccess) {
        return fail("hipInit", status);
    }
    status = hipSetDevice(0);
    if (status != hipSuccess) {
        return fail("hipSetDevice", status);
    }

    ModuleKernel count, prefix, padded_prefix, scatter, gate_up, down;
    if (!load_kernel(kernel_dir, "route_count", "_route_count_kernel", 256u, kRouteThreads, 0u, &count) ||
        !load_kernel(kernel_dir, "route_prefix_by_program", "_route_prefix_by_program_kernel", 256u, kRouteThreads, 0u, &prefix) ||
        !load_kernel(kernel_dir, "route_padded_prefix", "_route_padded_prefix_kernel", 1u, kRouteThreads, 0u, &padded_prefix) ||
        !load_kernel(kernel_dir, "route_scatter", "_route_scatter_kernel", 256u, kRouteThreads, 0u, &scatter) ||
        !load_kernel(kernel_dir, "gate_up_silu", "_gate_up_silu_kernel", kMaxRouteBlocks * kGateUpGridN, kGateThreads, kGateSharedBytes, &gate_up) ||
        !load_kernel(kernel_dir, "down", "_down_kernel", kMaxRouteBlocks * kDownGridN, kDownThreads, kDownSharedBytes, &down)) {
        return 1;
    }

    const size_t input_elements = static_cast<size_t>(kTokens) * kHidden;
    const size_t gate_up_elements =
        static_cast<size_t>(kExperts) * kGateUpRows * kHidden;
    const size_t down_elements =
        static_cast<size_t>(kExperts) * kHidden * kIntermediate;
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
    const size_t gate_up_int8_scale_elements =
        gate_up_elements / kWeightInt8GroupValues;
    const size_t down_int8_scale_elements =
        down_elements / kWeightInt8GroupValues;
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
    constexpr uint32_t kGateUpRowCount = kExperts * kGateUpRows;
    constexpr uint32_t kDownRowCount = kExperts * kHidden;
    const size_t gate_up_lossless_packed_bytes =
        static_cast<size_t>(kGateUpRowCount) *
        lossless_row_packed_bytes(kHidden);
    const size_t down_lossless_packed_bytes =
        static_cast<size_t>(kDownRowCount) *
        lossless_row_packed_bytes(kIntermediate);
#endif
    const size_t activated_elements = static_cast<size_t>(kRoutes) * kIntermediate;
    const size_t route_output_elements = static_cast<size_t>(kRoutes) * kHidden;
    const size_t output_elements = static_cast<size_t>(kTokens) * kHidden;
    // Dynamic q8192 writes the complete output, so its no-overwrite sentinel
    // must live one element beyond the fixed-shape allocation.
    const size_t guarded_output_elements = output_elements + 1u;
    const size_t count_elements = static_cast<size_t>(kExperts + 1u) * kExperts;
    const size_t router_elements = static_cast<size_t>(kExperts) * kHidden;
    const size_t shared_projection_elements =
        static_cast<size_t>(kIntermediate) * kHidden;
    const size_t shared_down_elements =
        static_cast<size_t>(kHidden) * kIntermediate;

    float *device_input = nullptr;
    uint16_t *device_input_bf16 = nullptr;
    uint16_t *device_gate_up = nullptr;
    uint16_t *device_down = nullptr;
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
    int8_t *device_gate_up_int8 = nullptr;
    WeightInt8Scale *device_gate_up_int8_scales = nullptr;
    int8_t *device_down_int8 = nullptr;
    WeightInt8Scale *device_down_int8_scales = nullptr;
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
    uint8_t *device_gate_up_lossless_packed = nullptr;
    uint32_t *device_gate_up_lossless_overflow_indices = nullptr;
    uint8_t *device_down_lossless_packed = nullptr;
    uint32_t *device_down_lossless_overflow_indices = nullptr;
#endif
    int32_t *device_topk_ids = nullptr;
    float *device_topk_weights = nullptr;
    int32_t *device_counts = nullptr;
    int32_t *device_cumsum = nullptr;
    int32_t *device_total_post_pad = nullptr;
    int32_t *device_sorted_routes = nullptr;
    int32_t *device_block_experts = nullptr;
    uint16_t *device_activated = nullptr;
    float *device_route_outputs = nullptr;
    float *device_outputs = nullptr;
    uint16_t *device_router = nullptr;
    uint16_t *device_shared_gate = nullptr;
    uint16_t *device_shared_gate_projection = nullptr;
    uint16_t *device_shared_up_projection = nullptr;
    uint16_t *device_shared_down = nullptr;
    float *device_residual_output = nullptr;
    float *device_async_residual_output = nullptr;
    float *device_v3_residual_output = nullptr;
    std::vector<float *> device_v3_async_residual_outputs(
        kFullV3AsyncChain,
        nullptr
    );

#define ALLOCATE(pointer, bytes, name)                                              \
    do {                                                                            \
        status = hipMalloc(reinterpret_cast<void **>(&(pointer)), (bytes));          \
        if (status != hipSuccess) {                                                  \
            return fail("hipMalloc(" name ")", status);                            \
        }                                                                           \
    } while (false)

    ALLOCATE(device_input, input_elements * sizeof(float), "input");
    ALLOCATE(device_input_bf16, input_elements * sizeof(uint16_t), "input_bf16");
    ALLOCATE(device_gate_up, gate_up_elements * sizeof(uint16_t), "gate_up");
    ALLOCATE(device_down, down_elements * sizeof(uint16_t), "down");
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
    ALLOCATE(device_gate_up_int8, gate_up_elements, "gate_up_int8");
    ALLOCATE(
        device_gate_up_int8_scales,
        gate_up_int8_scale_elements * kWeightInt8ScaleBytes,
        "gate_up_int8_scales"
    );
    ALLOCATE(device_down_int8, down_elements, "down_int8");
    ALLOCATE(
        device_down_int8_scales,
        down_int8_scale_elements * kWeightInt8ScaleBytes,
        "down_int8_scales"
    );
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
    ALLOCATE(
        device_gate_up_lossless_packed,
        gate_up_lossless_packed_bytes,
        "gate_up_lossless_packed"
    );
    ALLOCATE(
        device_gate_up_lossless_overflow_indices,
        static_cast<size_t>(kGateUpRowCount) * sizeof(uint32_t),
        "gate_up_lossless_overflow_indices"
    );
    ALLOCATE(
        device_down_lossless_packed,
        down_lossless_packed_bytes,
        "down_lossless_packed"
    );
    ALLOCATE(
        device_down_lossless_overflow_indices,
        static_cast<size_t>(kDownRowCount) * sizeof(uint32_t),
        "down_lossless_overflow_indices"
    );
#endif
    ALLOCATE(device_topk_ids, static_cast<size_t>(kRoutes) * sizeof(int32_t), "topk_ids");
    ALLOCATE(device_topk_weights, static_cast<size_t>(kRoutes) * sizeof(float), "topk_weights");
    ALLOCATE(device_counts, count_elements * sizeof(int32_t), "counts");
    ALLOCATE(device_cumsum, static_cast<size_t>(kExperts + 1u) * sizeof(int32_t), "cumsum");
    ALLOCATE(device_total_post_pad, sizeof(int32_t), "total_post_pad");
    ALLOCATE(device_sorted_routes, static_cast<size_t>(kMaxSortedRoutes) * sizeof(int32_t), "sorted_routes");
    ALLOCATE(device_block_experts, static_cast<size_t>(kMaxRouteBlocks) * sizeof(int32_t), "block_experts");
    ALLOCATE(device_activated, activated_elements * sizeof(uint16_t), "activated");
    ALLOCATE(device_route_outputs, route_output_elements * sizeof(float), "route_outputs_f32");
    ALLOCATE(device_outputs, output_elements * sizeof(float), "outputs");
    ALLOCATE(device_router, router_elements * sizeof(uint16_t), "router");
    ALLOCATE(device_shared_gate, static_cast<size_t>(kHidden) * sizeof(uint16_t), "shared_gate");
    ALLOCATE(device_shared_gate_projection, shared_projection_elements * sizeof(uint16_t), "shared_gate_projection");
    ALLOCATE(device_shared_up_projection, shared_projection_elements * sizeof(uint16_t), "shared_up_projection");
    ALLOCATE(device_shared_down, shared_down_elements * sizeof(uint16_t), "shared_down");
    ALLOCATE(device_residual_output, output_elements * sizeof(float), "residual_output");
    ALLOCATE(device_async_residual_output, output_elements * sizeof(float), "async_residual_output");
    ALLOCATE(device_v3_residual_output, output_elements * sizeof(float), "v3_residual_output");
    for (float *&output : device_v3_async_residual_outputs) {
        ALLOCATE(
            output,
            guarded_output_elements * sizeof(float),
            "v3_async_residual_output"
        );
    }
#undef ALLOCATE

    std::vector<int32_t> topk_ids(kRoutes);
    std::vector<float> topk_weights(kRoutes, 1.0f / static_cast<float>(kTopK));
    for (uint32_t token = 0; token < kTokens; ++token) {
        for (uint32_t route = 0; route < kTopK; ++route) {
            topk_ids[static_cast<size_t>(token) * kTopK + route] =
                static_cast<int32_t>((token * 37u + route * 23u) % kActiveExperts);
        }
    }
    status = hipMemcpy(device_topk_ids, topk_ids.data(), topk_ids.size() * sizeof(topk_ids[0]), hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        return fail("hipMemcpy(topk_ids)", status);
    }
    status = hipMemcpy(device_topk_weights, topk_weights.data(), topk_weights.size() * sizeof(topk_weights[0]), hipMemcpyHostToDevice);
    if (status != hipSuccess) {
        return fail("hipMemcpy(topk_weights)", status);
    }

    const auto grid_for = [](size_t elements) {
        return dim3(static_cast<uint32_t>((elements + 255u) / 256u));
    };
    hipLaunchKernelGGL(fill_input_kernel, grid_for(input_elements), dim3(256), 0, 0, device_input, input_elements);
    hipLaunchKernelGGL(fill_gate_up_kernel, grid_for(gate_up_elements), dim3(256), 0, 0, device_gate_up, gate_up_elements);
    hipLaunchKernelGGL(fill_down_kernel, grid_for(down_elements), dim3(256), 0, 0, device_down, down_elements);
    hipLaunchKernelGGL(
        fill_bf16_matrix_pattern_kernel,
        grid_for(router_elements),
        dim3(256),
        0,
        0,
        device_router,
        router_elements,
        kHidden,
        1u,
        static_cast<float>(kHidden * 64u)
    );
    status = hipGetLastError();
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            fill_bf16_matrix_pattern_kernel,
            grid_for(kHidden),
            dim3(256),
            0,
            0,
            device_shared_gate,
            static_cast<size_t>(kHidden),
            kHidden,
            3u,
            static_cast<float>(kHidden * 16u)
        );
        status = hipGetLastError();
    }
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            fill_bf16_matrix_pattern_kernel,
            grid_for(shared_projection_elements),
            dim3(256),
            0,
            0,
            device_shared_gate_projection,
            shared_projection_elements,
            kHidden,
            5u,
            static_cast<float>(kHidden * 64u)
        );
        status = hipGetLastError();
    }
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            fill_bf16_matrix_pattern_kernel,
            grid_for(shared_projection_elements),
            dim3(256),
            0,
            0,
            device_shared_up_projection,
            shared_projection_elements,
            kHidden,
            7u,
            static_cast<float>(kHidden * 64u)
        );
        status = hipGetLastError();
    }
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            fill_bf16_matrix_pattern_kernel,
            grid_for(shared_down_elements),
            dim3(256),
            0,
            0,
            device_shared_down,
            shared_down_elements,
            kIntermediate,
            9u,
            static_cast<float>(kIntermediate * 64u)
        );
        status = hipGetLastError();
    }
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            quantize_weight_int8_rows_kernel,
            dim3(4096u),
            dim3(kWeightInt8BlockThreads),
            0,
            0,
            device_gate_up,
            device_gate_up_int8,
            device_gate_up_int8_scales,
            kHidden,
            kExperts * kGateUpRows
        );
        status = hipGetLastError();
    }
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            quantize_weight_int8_rows_kernel,
            dim3(4096u),
            dim3(kWeightInt8BlockThreads),
            0,
            0,
            device_down,
            device_down_int8,
            device_down_int8_scales,
            kIntermediate,
            kExperts * kHidden
        );
        status = hipGetLastError();
    }
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            pack_constant_lossless_rows_kernel,
            dim3(4096u),
            dim3(256u),
            0,
            0,
            device_gate_up,
            device_gate_up_lossless_packed,
            device_gate_up_lossless_overflow_indices,
            kHidden,
            kGateUpRowCount
        );
        status = hipGetLastError();
    }
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            pack_constant_lossless_rows_kernel,
            dim3(4096u),
            dim3(256u),
            0,
            0,
            device_down,
            device_down_lossless_packed,
            device_down_lossless_overflow_indices,
            kIntermediate,
            kDownRowCount
        );
        status = hipGetLastError();
    }
#endif
    if (status != hipSuccess) return fail("initialize_full_provider_weights", status);
    hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, device_residual_output, output_elements);
    hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, device_async_residual_output, output_elements);
    hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, device_v3_residual_output, output_elements);
    for (float *output : device_v3_async_residual_outputs) {
        hipLaunchKernelGGL(
            fill_input_kernel,
            grid_for(guarded_output_elements),
            dim3(256),
            0,
            0,
            output,
            guarded_output_elements
        );
    }
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        return fail("initialize_weights", status);
    }

    int32_t logical_routes = static_cast<int32_t>(kRoutes);
    auto run_pipeline = [&]() -> hipError_t {
        hipLaunchKernelGGL(
            convert_input_kernel,
            grid_for(input_elements),
            dim3(256),
            0,
            0,
            device_input,
            device_input_bf16,
            input_elements
        );
        hipError_t local = hipGetLastError();
        if (local != hipSuccess) return local;
        local = hipMemset(device_counts, 0, count_elements * sizeof(int32_t));
        if (local != hipSuccess) return local;
        local = hipMemset(device_cumsum, 0, static_cast<size_t>(kExperts + 1u) * sizeof(int32_t));
        if (local != hipSuccess) return local;
        local = hipMemsetD32(
            reinterpret_cast<hipDeviceptr_t>(device_sorted_routes),
            static_cast<int>(kRoutes),
            kMaxSortedRoutes
        );
        if (local != hipSuccess) return local;
        local = launch(
            count,
            {&device_topk_ids, &device_counts, &logical_routes}
        );
        if (local != hipSuccess) return local;
        local = launch(prefix, {&device_counts});
        if (local != hipSuccess) return local;
        local = launch(padded_prefix, {&device_total_post_pad, &device_counts, &device_cumsum});
        if (local != hipSuccess) return local;
        local = launch(
            scatter,
            {
                &device_topk_ids,
                &device_sorted_routes,
                &device_block_experts,
                &device_counts,
                &device_cumsum,
                &logical_routes,
            }
        );
        if (local != hipSuccess) return local;
        local = launch(gate_up, {&device_input_bf16, &device_gate_up, &device_sorted_routes, &device_block_experts, &device_total_post_pad, &device_activated});
        if (local != hipSuccess) return local;
        local = launch(down, {&device_activated, &device_down, &device_sorted_routes, &device_block_experts, &device_total_post_pad, &device_route_outputs});
        if (local != hipSuccess) return local;
        hipLaunchKernelGGL(
            combine_route_order_kernel,
            grid_for(output_elements),
            dim3(256),
            0,
            0,
            device_route_outputs,
            device_topk_weights,
            device_outputs,
            output_elements
        );
        return hipGetLastError();
    };

    status = run_pipeline();
    if (status != hipSuccess) {
        return fail("run_pipeline(warmup)", status);
    }
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        return fail("sync(warmup)", status);
    }

    int32_t total_post_pad = 0;
    status = hipMemcpy(&total_post_pad, device_total_post_pad, sizeof(total_post_pad), hipMemcpyDeviceToHost);
    if (status != hipSuccess) {
        return fail("hipMemcpy(total_post_pad)", status);
    }
    if (total_post_pad <= 0 || total_post_pad > static_cast<int32_t>(kMaxSortedRoutes) || total_post_pad % static_cast<int32_t>(kBlockM) != 0) {
        std::cerr << "invalid total_post_pad=" << total_post_pad << std::endl;
        return 3;
    }
    std::vector<int32_t> sorted_routes(static_cast<size_t>(total_post_pad));
    std::vector<int32_t> block_experts(static_cast<size_t>(total_post_pad) / kBlockM);
    status = hipMemcpy(sorted_routes.data(), device_sorted_routes, sorted_routes.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (status != hipSuccess) return fail("hipMemcpy(sorted_routes)", status);
    status = hipMemcpy(block_experts.data(), device_block_experts, block_experts.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (status != hipSuccess) return fail("hipMemcpy(block_experts)", status);
    size_t valid_routes = 0;
    int32_t previous_expert = -1;
    int32_t previous_route = -1;
    for (size_t block = 0; block < block_experts.size(); ++block) {
        const int32_t expert = block_experts[block];
        if (expert < previous_expert || expert < 0 || expert >= static_cast<int32_t>(kExperts)) {
            std::cerr << "invalid block expert at block=" << block << " expert=" << expert << std::endl;
            return 3;
        }
        if (expert != previous_expert) {
            previous_route = -1;
            previous_expert = expert;
        }
        for (uint32_t lane = 0; lane < kBlockM; ++lane) {
            const int32_t route = sorted_routes[block * kBlockM + lane];
            if (route < 0 || route >= static_cast<int32_t>(kRoutes)) {
                continue;
            }
            const int32_t route_expert = topk_ids[static_cast<size_t>(route)];
            if (route_expert != expert || route <= previous_route) {
                std::cerr << "invalid stable route block=" << block << " lane=" << lane
                          << " route=" << route << " expert=" << expert
                          << " route_expert=" << route_expert
                          << " previous_route=" << previous_route << std::endl;
                return 3;
            }
            previous_route = route;
            ++valid_routes;
        }
    }
    if (valid_routes != kRoutes) {
        std::cerr << "valid route count mismatch actual=" << valid_routes
                  << " expected=" << kRoutes << std::endl;
        return 3;
    }

    std::vector<float> output_samples(kTokens);
    std::vector<float> expected_samples(kTokens);
    for (uint32_t token = 0; token < kTokens; ++token) {
        status = hipMemcpy(&output_samples[token], device_outputs + static_cast<size_t>(token) * kHidden, sizeof(float), hipMemcpyDeviceToHost);
        if (status != hipSuccess) return fail("hipMemcpy(output_sample)", status);
        float expected = 0.0f;
        for (uint32_t route = 0; route < kTopK; ++route) {
            const uint32_t expert = static_cast<uint32_t>(topk_ids[static_cast<size_t>(token) * kTopK + route]);
            const float gate = static_cast<float>(expert % 7u + 1u);
            const float up = static_cast<float>(expert % 5u + 1u) * 0.5f;
            const float activated = bf16_to_float(float_to_bf16((gate / (1.0f + std::exp(-gate))) * up));
            const float down_value =
                activated * static_cast<float>(expert % 3u + 1u);
            expected += down_value / static_cast<float>(kTopK);
        }
        expected_samples[token] = expected;
        if (!std::isfinite(output_samples[token]) || std::abs(output_samples[token] - expected) > 0.03125f) {
            std::cerr << "output mismatch token=" << token
                      << " actual=" << output_samples[token]
                      << " expected=" << expected << std::endl;
            return 3;
        }
    }
    std::vector<float> provider_expected_samples = expected_samples;
    const char *sorted_bf16_route_sum = std::getenv(
        "QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM"
    );
    if (sorted_bf16_route_sum != nullptr &&
        sorted_bf16_route_sum[0] != '\0' &&
        std::strcmp(sorted_bf16_route_sum, "0") != 0) {
        for (uint32_t token = 0; token < kTokens; ++token) {
            std::array<uint32_t, kTopK> route_order{};
            for (uint32_t route = 0; route < kTopK; ++route) {
                route_order[route] = route;
            }
            const size_t route_base = static_cast<size_t>(token) * kTopK;
            std::sort(
                route_order.begin(),
                route_order.end(),
                [&](uint32_t left, uint32_t right) {
                    return topk_ids[route_base + left] <
                        topk_ids[route_base + right];
                }
            );
            float expected = 0.0f;
            for (const uint32_t route : route_order) {
                const uint32_t expert = static_cast<uint32_t>(
                    topk_ids[route_base + route]
                );
                const float gate = static_cast<float>(expert % 7u + 1u);
                const float up =
                    static_cast<float>(expert % 5u + 1u) * 0.5f;
                const float activated = bf16_to_float(float_to_bf16(
                    (gate / (1.0f + std::exp(-gate))) * up
                ));
                const float down_bf16 = bf16_to_float(float_to_bf16(
                    activated * static_cast<float>(expert % 3u + 1u)
                ));
                const float contribution_bf16 = bf16_to_float(float_to_bf16(
                    topk_weights[route_base + route] * down_bf16
                ));
                expected = bf16_to_float(float_to_bf16(
                    expected + contribution_bf16
                ));
            }
            provider_expected_samples[token] = expected;
        }
    }

    auto measure = [&](auto &&operation) -> float {
        hipEvent_t local_start = nullptr;
        hipEvent_t local_stop = nullptr;
        if (hipEventCreate(&local_start) != hipSuccess ||
            hipEventCreate(&local_stop) != hipSuccess ||
            hipEventRecord(local_start, nullptr) != hipSuccess) {
            return -1.0f;
        }
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            if (operation() != hipSuccess) {
                return -1.0f;
            }
        }
        if (hipEventRecord(local_stop, nullptr) != hipSuccess ||
            hipEventSynchronize(local_stop) != hipSuccess) {
            return -1.0f;
        }
        float elapsed = 0.0f;
        if (hipEventElapsedTime(&elapsed, local_start, local_stop) != hipSuccess) {
            return -1.0f;
        }
        (void)hipEventDestroy(local_stop);
        (void)hipEventDestroy(local_start);
        return elapsed / static_cast<float>(repetitions);
    };
    const float sort_ms = measure([&]() -> hipError_t {
        hipError_t local = hipMemset(device_counts, 0, count_elements * sizeof(int32_t));
        if (local != hipSuccess) return local;
        local = hipMemset(device_cumsum, 0, static_cast<size_t>(kExperts + 1u) * sizeof(int32_t));
        if (local != hipSuccess) return local;
        local = launch(
            count,
            {&device_topk_ids, &device_counts, &logical_routes}
        );
        if (local != hipSuccess) return local;
        local = launch(prefix, {&device_counts});
        if (local != hipSuccess) return local;
        local = launch(padded_prefix, {&device_total_post_pad, &device_counts, &device_cumsum});
        if (local != hipSuccess) return local;
        return launch(
            scatter,
            {
                &device_topk_ids,
                &device_sorted_routes,
                &device_block_experts,
                &device_counts,
                &device_cumsum,
                &logical_routes,
            }
        );
    });
    const float input_convert_ms = measure([&]() -> hipError_t {
        hipLaunchKernelGGL(
            convert_input_kernel,
            grid_for(input_elements),
            dim3(256),
            0,
            0,
            device_input,
            device_input_bf16,
            input_elements
        );
        return hipGetLastError();
    });
    const float gate_up_ms = measure([&]() {
        return launch(gate_up, {&device_input_bf16, &device_gate_up, &device_sorted_routes, &device_block_experts, &device_total_post_pad, &device_activated});
    });
    const float down_ms = measure([&]() {
        return launch(down, {&device_activated, &device_down, &device_sorted_routes, &device_block_experts, &device_total_post_pad, &device_route_outputs});
    });
    const float combine_ms = measure([&]() -> hipError_t {
        hipLaunchKernelGGL(
            combine_route_order_kernel,
            grid_for(output_elements),
            dim3(256),
            0,
            0,
            device_route_outputs,
            device_topk_weights,
            device_outputs,
            output_elements
        );
        return hipGetLastError();
    });

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    status = hipEventCreate(&start);
    if (status != hipSuccess) return fail("hipEventCreate(start)", status);
    status = hipEventCreate(&stop);
    if (status != hipSuccess) return fail("hipEventCreate(stop)", status);
    status = hipEventRecord(start, nullptr);
    if (status != hipSuccess) return fail("hipEventRecord(start)", status);
    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        status = run_pipeline();
        if (status != hipSuccess) return fail("run_pipeline(timed)", status);
    }
    status = hipEventRecord(stop, nullptr);
    if (status != hipSuccess) return fail("hipEventRecord(stop)", status);
    status = hipEventSynchronize(stop);
    if (status != hipSuccess) return fail("hipEventSynchronize(stop)", status);
    float elapsed_ms = 0.0f;
    status = hipEventElapsedTime(&elapsed_ms, start, stop);
    if (status != hipSuccess) return fail("hipEventElapsedTime", status);

    float provider_ms = -1.0f;
    float full_provider_ms = -1.0f;
    float full_provider_warmup_ms = -1.0f;
    float full_provider_async_ms = -1.0f;
    float full_provider_async_warmup_ms = -1.0f;
    float full_provider_v3_ms = -1.0f;
    float full_provider_v3_warmup_ms = -1.0f;
    float full_provider_v3_async_total_ms = -1.0f;
    float full_provider_v3_async_ms = -1.0f;
    size_t full_provider_async_mismatches = 0u;
    size_t full_provider_v3_mismatches = 0u;
    size_t full_provider_v3_async_mismatches = 0u;
    uint64_t full_provider_sync_hash = 0u;
    uint64_t full_provider_async_hash = 0u;
    uint64_t full_provider_v3_hash = 0u;
    uint64_t full_provider_v3_async_hash = 0u;
    bool full_provider_v3_async_hashes_match = true;
    uint64_t provider_scratch_bytes = 0;
    uint32_t provider_backend_mask = UINT32_MAX;
    float router_debug_ms = -1.0f;
    size_t router_debug_id_mismatches = 0u;
    size_t router_debug_weight_mismatches = 0u;
    uint32_t router_debug_first_id = UINT32_MAX;
    // Keep the historical probes first because the build record reports their
    // timings by name.  The remaining cases exercise every kind of cliff that
    // matters to the product sweep: interval edges, exact 64-token alignment,
    // both neighbors of an aligned length, the q8192 predecessor, and q8192
    // itself.  The final case proves that the export used by the random-length
    // service is bitwise identical to the product-qualified fixed export.
    constexpr std::array<uint32_t, 32u> kDynamicLogicalTokens{
        2073u,
        2156u,
        2560u,
        3073u,
        4609u,
        6145u,
        2049u,
        2175u,
        2176u,
        2177u,
        2559u,
        2561u,
        3071u,
        3072u,
        3583u,
        3584u,
        3585u,
        4095u,
        4096u,
        4097u,
        4607u,
        4608u,
        6143u,
        6144u,
        7167u,
        7168u,
        7169u,
        7679u,
        7680u,
        7681u,
        8191u,
        8192u,
    };
    constexpr size_t kDynamicLogicalTimingSamples = 3u;
    struct DynamicLogicalTimingSample {
        float total_ms;
        float submit_ms;
    };
    std::array<float, kDynamicLogicalTokens.size()> dynamic_logical_ms{};
    std::array<float, kDynamicLogicalTokens.size()>
        dynamic_logical_submit_ms{};
    std::array<float, kDynamicLogicalTokens.size()>
        dynamic_logical_device_completion_ms{};
    std::array<float, kDynamicLogicalTokens.size()>
        dynamic_logical_first_sample_ms{};
    std::array<float, kDynamicLogicalTokens.size()>
        dynamic_logical_min_ms{};
    std::array<float, kDynamicLogicalTokens.size()>
        dynamic_logical_max_ms{};
    size_t dynamic_logical_mismatches = 0u;
    size_t dynamic_logical_nonfinite = 0u;
    float dynamic_logical_max_abs_diff = 0.0f;
    size_t dynamic_logical_q8192_mismatches = 0u;
    float dynamic_logical_q8192_max_abs_diff = 0.0f;
    bool dynamic_logical_tail_guard_pass = true;
    if (argc >= 4) {
        using PrepareFunction = int (*)(const char *);
        using LaunchFunction = int (*)(
            const float *,
            const uint16_t *,
            const uint16_t *,
            const uint32_t *,
            const float *,
            float *,
            void *
        );
        using FullLaunchFunction = int (*)(
            const float *,
            const float *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            float *,
            void *
        );
        using DynamicFullLaunchFunction = int (*)(
            const float *,
            const float *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            const uint16_t *,
            float *,
            uint32_t,
            void *
        );
        using SetWeightInt8Function = int (*)(
            const int8_t *,
            const void *,
            const int8_t *,
            const void *
        );
        using SetLosslessPaletteFunction = int (*)(
            const uint8_t *,
            const uint32_t *,
            const uint16_t *,
            const uint8_t *,
            const uint32_t *,
            const uint16_t *
        );
        using LastErrorFunction = const char *(*)();
        using BackendMaskFunction = uint32_t (*)();
        using WeightInt8GroupValuesFunction = uint32_t (*)();
        using WeightInt8ScaleBytesFunction = uint32_t (*)();
        using RouterLaunchFunction = int (*)(
            const float *,
            const uint16_t *,
            void *
        );
        using CopyTopkFunction = int (*)(uint32_t *, float *);
        using ScratchBytesFunction = uint64_t (*)();
        using ReleaseFunction = void (*)();
        const ProviderModule provider = load_provider_module(argv[3]);
        if (provider == nullptr) {
            std::cerr << "provider module load failed error="
                      << provider_load_error() << std::endl;
            return 4;
        }
        const auto prepare = reinterpret_cast<PrepareFunction>(
            load_provider_symbol(provider, "qrt_triton_moe_q8192_prepare"));
        const auto provider_launch = reinterpret_cast<LaunchFunction>(
            load_provider_symbol(provider, "qrt_triton_moe_q8192_launch"));
        const auto provider_full_launch = reinterpret_cast<FullLaunchFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_launch_full_v2"
            ));
        const auto provider_full_launch_async = reinterpret_cast<FullLaunchFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_launch_full_v2_async"
            ));
        const auto provider_full_launch_v3 = reinterpret_cast<FullLaunchFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_launch_full_v3"
            ));
        const auto provider_full_launch_v3_async =
            reinterpret_cast<FullLaunchFunction>(load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_launch_full_v3_async"
            ));
        const auto provider_dynamic_full_launch =
            reinterpret_cast<DynamicFullLaunchFunction>(load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_launch_full_v4_dynamic_async"
            ));
        const auto set_weight_int8 =
            reinterpret_cast<SetWeightInt8Function>(load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_set_weight_int8_weights"
            ));
        const auto set_lossless_row_palette =
            reinterpret_cast<SetLosslessPaletteFunction>(load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_set_lossless_row_palette_weights"
            ));
        const auto last_error = reinterpret_cast<LastErrorFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_last_error"
            ));
        const auto backend_mask = reinterpret_cast<BackendMaskFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_backend_mask"
            ));
        const auto weight_int8_group_values =
            reinterpret_cast<WeightInt8GroupValuesFunction>(load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_weight_int8_group_values"
            ));
        const auto weight_int8_scale_bytes =
            reinterpret_cast<WeightInt8ScaleBytesFunction>(load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_weight_int8_scale_bytes"
            ));
        const auto router_launch = reinterpret_cast<RouterLaunchFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_launch_router_debug"
            ));
        const auto copy_topk = reinterpret_cast<CopyTopkFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_copy_topk_debug"
            ));
        const auto scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
            load_provider_symbol(
                provider,
                "qrt_triton_moe_q8192_scratch_bytes"
            ));
        const auto release = reinterpret_cast<ReleaseFunction>(
            load_provider_symbol(provider, "qrt_triton_moe_q8192_release"));
        if (prepare == nullptr || provider_launch == nullptr ||
            provider_full_launch == nullptr ||
            provider_full_launch_async == nullptr ||
            provider_full_launch_v3 == nullptr ||
            provider_full_launch_v3_async == nullptr ||
            provider_dynamic_full_launch == nullptr ||
            last_error == nullptr ||
            backend_mask == nullptr ||
            router_launch == nullptr || copy_topk == nullptr ||
            scratch_bytes == nullptr || release == nullptr
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
            || set_weight_int8 == nullptr ||
            weight_int8_group_values == nullptr ||
            weight_int8_scale_bytes == nullptr
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
            || set_lossless_row_palette == nullptr
#endif
        ) {
            std::cerr << "provider ABI is incomplete" << std::endl;
            return 4;
        }
        provider_backend_mask = backend_mask();
        if (provider_backend_mask != kExpectedProviderBackendMask) {
            std::cerr << "provider backend mask mismatch actual="
                      << provider_backend_mask
                      << " expected=" << kExpectedProviderBackendMask
                      << std::endl;
            return 4;
        }
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
        if (weight_int8_group_values() != kWeightInt8GroupValues) {
            std::cerr << "provider weight-int8 group mismatch actual="
                      << weight_int8_group_values()
                      << " expected=" << kWeightInt8GroupValues
                      << std::endl;
            return 4;
        }
        if (weight_int8_scale_bytes() != kWeightInt8ScaleBytes) {
            std::cerr << "provider weight-int8 scale bytes mismatch actual="
                      << weight_int8_scale_bytes()
                      << " expected=" << kWeightInt8ScaleBytes
                      << std::endl;
            return 4;
        }
#else
        (void)weight_int8_group_values;
        (void)weight_int8_scale_bytes;
#endif
        const std::string kernel_dir_string = kernel_dir.string();
        if (prepare(kernel_dir_string.c_str()) == 0) {
            std::cerr << "provider prepare failed error=" << last_error() << std::endl;
            return 4;
        }
#if QRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8
        if (set_weight_int8(
                device_gate_up_int8,
                device_gate_up_int8_scales,
                device_down_int8,
                device_down_int8_scales
            ) == 0) {
            std::cerr << "weight-int8 provider setup failed error="
                      << last_error() << std::endl;
            return 4;
        }
#else
        (void)set_weight_int8;
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE
        if (set_lossless_row_palette(
                device_gate_up_lossless_packed,
                device_gate_up_lossless_overflow_indices,
                nullptr,
                device_down_lossless_packed,
                device_down_lossless_overflow_indices,
                nullptr
            ) == 0) {
            std::cerr << "lossless-row-palette provider setup failed error="
                      << last_error() << std::endl;
            return 4;
        }
#else
        (void)set_lossless_row_palette;
#endif
        hipStream_t full_provider_stream = nullptr;
        status = hipStreamCreate(&full_provider_stream);
        if (status != hipSuccess) {
            return fail("hipStreamCreate(full_provider)", status);
        }
        provider_scratch_bytes = scratch_bytes();
        if (light_logical_tokens != 0u) {
            constexpr size_t kLightTimingSamples = 3u;
            hipLaunchKernelGGL(
                fill_nonuniform_input_kernel,
                grid_for(input_elements),
                dim3(256),
                0,
                full_provider_stream,
                device_input,
                input_elements
            );
            status = hipGetLastError();
            if (status != hipSuccess) {
                return fail("light_full_provider_input", status);
            }
            bool all_light_pass = true;
            for (size_t light_sequence_index = 0u;
                 light_sequence_index < light_token_sequence.size();
                 ++light_sequence_index) {
                const uint32_t current_light_tokens =
                    light_token_sequence[light_sequence_index];
                const size_t light_elements =
                    static_cast<size_t>(current_light_tokens) * kHidden;
                std::array<DynamicLogicalTimingSample, kLightTimingSamples>
                    light_samples{};
                for (size_t sample_index = 0u;
                     sample_index <= kLightTimingSamples;
                     ++sample_index) {
                    hipLaunchKernelGGL(
                        fill_input_kernel,
                        grid_for(guarded_output_elements),
                        dim3(256),
                        0,
                        full_provider_stream,
                        device_v3_async_residual_outputs[0],
                        guarded_output_elements
                    );
                    status = hipGetLastError();
                    if (status == hipSuccess) {
                        status = hipStreamSynchronize(full_provider_stream);
                    }
                    if (status != hipSuccess) {
                        return fail("light_full_provider_reset", status);
                    }
                    const auto light_start = std::chrono::steady_clock::now();
                    double light_submit_ms = 0.0;
                    for (uint32_t repetition = 0u;
                         repetition < repetitions;
                         ++repetition) {
                        const auto light_submit_start =
                            std::chrono::steady_clock::now();
                        if (provider_dynamic_full_launch(
                                device_input,
                                device_v3_async_residual_outputs[0],
                                device_router,
                                device_gate_up,
                                device_down,
                                device_shared_gate,
                                device_shared_gate_projection,
                                device_shared_up_projection,
                                device_shared_down,
                                device_v3_async_residual_outputs[0],
                                current_light_tokens,
                                full_provider_stream
                            ) == 0) {
                            std::cerr
                                << "light dynamic full provider launch failed "
                                   "tokens="
                                << current_light_tokens
                                << " repetition=" << repetition
                                << " error=" << last_error() << std::endl;
                            return 4;
                        }
                        const auto light_submit_stop =
                            std::chrono::steady_clock::now();
                        light_submit_ms +=
                            std::chrono::duration<double, std::milli>(
                                light_submit_stop - light_submit_start
                            ).count();
                        // Preserve the layer-serial dependency of product prefill.
                        // Every requested token length shares one loaded
                        // process and one device state.
                        status = hipStreamSynchronize(full_provider_stream);
                        if (status != hipSuccess) {
                            return fail("light_full_provider_sync", status);
                        }
                    }
                    const auto light_stop = std::chrono::steady_clock::now();
                    if (sample_index != 0u) {
                        DynamicLogicalTimingSample &sample =
                            light_samples[sample_index - 1u];
                        sample.total_ms = static_cast<float>(
                            std::chrono::duration<double, std::milli>(
                                light_stop - light_start
                            ).count() / static_cast<double>(repetitions)
                        );
                        sample.submit_ms = static_cast<float>(
                            light_submit_ms / static_cast<double>(repetitions)
                        );
                    }
                }
                std::sort(
                    light_samples.begin(),
                    light_samples.end(),
                    [](const DynamicLogicalTimingSample &left,
                       const DynamicLogicalTimingSample &right) {
                        return left.total_ms < right.total_ms;
                    }
                );
                const DynamicLogicalTimingSample &median =
                    light_samples[light_samples.size() / 2u];
                std::vector<float> light_output(light_elements);
                status = hipMemcpy(
                    light_output.data(),
                    device_v3_async_residual_outputs[0],
                    light_elements * sizeof(float),
                    hipMemcpyDeviceToHost
                );
                float tail_guard = 0.0f;
                if (status == hipSuccess) {
                    status = hipMemcpy(
                        &tail_guard,
                        device_v3_async_residual_outputs[0] + light_elements,
                        sizeof(float),
                        hipMemcpyDeviceToHost
                    );
                }
                if (status != hipSuccess) {
                    return fail("light_full_provider_output", status);
                }
                size_t nonfinite = 0u;
                for (const float value : light_output) {
                    nonfinite += std::isfinite(value) ? 0u : 1u;
                }
                const uint64_t light_hash = fnv1a64_f32(light_output);
                const bool light_pass =
                    nonfinite == 0u && tail_guard == 1.0f;
                all_light_pass = all_light_pass && light_pass;
                std::cout
                    << "moe_product_radius_light status="
                    << (light_pass ? "pass" : "fail")
                    << " host=" << local_host_name()
                    << " tokens=" << current_light_tokens
                    << " sequence_index=" << light_sequence_index
                    << " sequence_count=" << light_token_sequence.size()
                    << " total_ms=" << median.total_ms
                    << " submit_ms=" << median.submit_ms
                    << " device_completion_ms="
                    << (std::max)(0.0f, median.total_ms - median.submit_ms)
                    << " timing_samples=" << kLightTimingSamples
                    << " launches_per_sample=" << repetitions
                    << " timing_stat=median"
                    << " min_ms=" << light_samples.front().total_ms
                    << " max_ms=" << light_samples.back().total_ms
                    << " output_f32_fnv1a64=" << std::hex << light_hash
                    << std::dec
                    << " nonfinite=" << nonfinite
                    << " tail_guard_pass=" << (tail_guard == 1.0f ? 1 : 0)
                    << " provider_backend_mask=" << provider_backend_mask
                    << " provider_scratch_bytes=" << provider_scratch_bytes
                    << " reset_included=0"
                    << " component_only=1"
                    << " inference_success_claimed=0"
                    << std::endl;
            }
            release();
            status = hipStreamDestroy(full_provider_stream);
            if (status != hipSuccess) {
                close_provider_module(provider);
                return fail("hipStreamDestroy(light_full_provider)", status);
            }
            close_provider_module(provider);
            return all_light_pass ? 0 : 4;
        }
        auto run_provider = [&]() -> hipError_t {
            if (provider_launch(
                    device_input,
                    device_gate_up,
                    device_down,
                    reinterpret_cast<const uint32_t *>(topk_ids.data()),
                    topk_weights.data(),
                    device_outputs,
                    nullptr) == 0) {
                std::cerr << "provider launch failed error=" << last_error() << std::endl;
                return hipErrorUnknown;
            }
            return hipSuccess;
        };
        status = run_provider();
        if (status != hipSuccess) return fail("provider(warmup)", status);
        status = hipDeviceSynchronize();
        if (status != hipSuccess) return fail("provider_sync(warmup)", status);
        for (uint32_t token = 0; token < kTokens; ++token) {
            float actual = 0.0f;
            status = hipMemcpy(
                &actual,
                device_outputs + static_cast<size_t>(token) * kHidden,
                sizeof(actual),
                hipMemcpyDeviceToHost
            );
            if (status != hipSuccess) return fail("hipMemcpy(provider_output_sample)", status);
            if (!std::isfinite(actual) ||
                std::abs(actual - provider_expected_samples[token]) >
                    kWeightInt8ProviderOutputTolerance) {
                std::cerr << "provider output mismatch token=" << token
                          << " actual=" << actual
                          << " expected=" << provider_expected_samples[token]
                          << std::endl;
                return 4;
            }
        }

        hipLaunchKernelGGL(
            fill_input_kernel,
            grid_for(output_elements),
            dim3(256),
            0,
            full_provider_stream,
            device_async_residual_output,
            output_elements
        );
        const auto full_async_start = std::chrono::steady_clock::now();
        if (provider_full_launch_async(
                device_input,
                device_async_residual_output,
                device_router,
                device_gate_up,
                device_down,
                device_shared_gate,
                device_shared_gate_projection,
                device_shared_up_projection,
                device_shared_down,
                device_async_residual_output,
                full_provider_stream
            ) == 0) {
            std::cerr << "async full provider launch failed error="
                      << last_error() << std::endl;
            return 4;
        }
        status = hipStreamSynchronize(full_provider_stream);
        if (status != hipSuccess) {
            return fail("full_provider_async_sync(warmup)", status);
        }
        const auto full_async_stop = std::chrono::steady_clock::now();
        full_provider_async_warmup_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_async_stop - full_async_start
            ).count()
        );
        provider_ms = measure(run_provider);
        if (provider_ms < 0.0f) {
            std::cerr << "provider measurement failed error=" << last_error() << std::endl;
            return 4;
        }

        hipLaunchKernelGGL(
            fill_nonuniform_input_kernel,
            grid_for(input_elements),
            dim3(256),
            0,
            full_provider_stream,
            device_input,
            input_elements
        );
        status = hipStreamSynchronize(full_provider_stream);
        if (status != hipSuccess) {
            return fail("full_provider_nonuniform_input", status);
        }

        hipLaunchKernelGGL(
            fill_input_kernel,
            grid_for(output_elements),
            dim3(256),
            0,
            full_provider_stream,
            device_residual_output,
            output_elements
        );
        status = hipDeviceSynchronize();
        if (status != hipSuccess) return fail("full_provider_residual_reset", status);
        const auto full_start = std::chrono::steady_clock::now();
        if (provider_full_launch(
                device_input,
                device_residual_output,
                device_router,
                device_gate_up,
                device_down,
                device_shared_gate,
                device_shared_gate_projection,
                device_shared_up_projection,
                device_shared_down,
                device_residual_output,
                full_provider_stream
            ) == 0) {
            std::cerr << "full provider launch failed error="
                      << last_error() << std::endl;
            return 4;
        }
        const auto full_stop = std::chrono::steady_clock::now();
        full_provider_warmup_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_stop - full_start
            ).count()
        );

        hipLaunchKernelGGL(
            fill_input_kernel,
            grid_for(output_elements),
            dim3(256),
            0,
            full_provider_stream,
            device_v3_residual_output,
            output_elements
        );
        const auto full_v3_start = std::chrono::steady_clock::now();
        if (provider_full_launch_v3(
                device_input,
                device_v3_residual_output,
                device_router,
                device_gate_up,
                device_down,
                device_shared_gate,
                device_shared_gate_projection,
                device_shared_up_projection,
                device_shared_down,
                device_v3_residual_output,
                full_provider_stream
            ) == 0) {
            std::cerr << "full-v3 provider launch failed error="
                      << last_error() << std::endl;
            return 4;
        }
        const auto full_v3_stop = std::chrono::steady_clock::now();
        full_provider_v3_warmup_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_v3_stop - full_v3_start
            ).count()
        );

        const auto full_timed_start = std::chrono::steady_clock::now();
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            hipLaunchKernelGGL(
                fill_input_kernel,
                grid_for(output_elements),
                dim3(256),
                0,
                full_provider_stream,
                device_residual_output,
                output_elements
            );
            if (provider_full_launch(
                    device_input,
                    device_residual_output,
                    device_router,
                    device_gate_up,
                    device_down,
                    device_shared_gate,
                    device_shared_gate_projection,
                    device_shared_up_projection,
                    device_shared_down,
                    device_residual_output,
                    full_provider_stream
                ) == 0) {
                std::cerr << "timed full provider launch failed error="
                          << last_error() << std::endl;
                return 4;
            }
        }
        const auto full_timed_stop = std::chrono::steady_clock::now();
        full_provider_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_timed_stop - full_timed_start
            ).count() /
            static_cast<double>(repetitions)
        );

        const auto full_async_timed_start = std::chrono::steady_clock::now();
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            hipLaunchKernelGGL(
                fill_input_kernel,
                grid_for(output_elements),
                dim3(256),
                0,
                full_provider_stream,
                device_async_residual_output,
                output_elements
            );
            if (provider_full_launch_async(
                    device_input,
                    device_async_residual_output,
                    device_router,
                    device_gate_up,
                    device_down,
                    device_shared_gate,
                    device_shared_gate_projection,
                    device_shared_up_projection,
                    device_shared_down,
                    device_async_residual_output,
                    full_provider_stream
                ) == 0) {
                std::cerr << "timed async full provider launch failed error="
                          << last_error() << std::endl;
                return 4;
            }
        }
        status = hipStreamSynchronize(full_provider_stream);
        if (status != hipSuccess) {
            return fail("full_provider_async_sync(timed)", status);
        }
        const auto full_async_timed_stop = std::chrono::steady_clock::now();
        full_provider_async_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_async_timed_stop - full_async_timed_start
            ).count() /
            static_cast<double>(repetitions)
        );

        const auto full_v3_timed_start = std::chrono::steady_clock::now();
        for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
            hipLaunchKernelGGL(
                fill_input_kernel,
                grid_for(output_elements),
                dim3(256),
                0,
                full_provider_stream,
                device_v3_residual_output,
                output_elements
            );
            if (provider_full_launch_v3(
                    device_input,
                    device_v3_residual_output,
                    device_router,
                    device_gate_up,
                    device_down,
                    device_shared_gate,
                    device_shared_gate_projection,
                    device_shared_up_projection,
                    device_shared_down,
                    device_v3_residual_output,
                    full_provider_stream
                ) == 0) {
                std::cerr << "timed full-v3 provider launch failed error="
                          << last_error() << std::endl;
                return 4;
            }
        }
        const auto full_v3_timed_stop = std::chrono::steady_clock::now();
        full_provider_v3_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_v3_timed_stop - full_v3_timed_start
            ).count() /
            static_cast<double>(repetitions)
        );

        const auto full_v3_async_start = std::chrono::steady_clock::now();
        for (uint32_t launch = 0; launch < kFullV3AsyncChain; ++launch) {
            hipLaunchKernelGGL(
                fill_input_kernel,
                grid_for(output_elements),
                dim3(256),
                0,
                full_provider_stream,
                device_v3_async_residual_outputs[launch],
                output_elements
            );
            if (provider_full_launch_v3_async(
                    device_input,
                    device_v3_async_residual_outputs[launch],
                    device_router,
                    device_gate_up,
                    device_down,
                    device_shared_gate,
                    device_shared_gate_projection,
                    device_shared_up_projection,
                    device_shared_down,
                    device_v3_async_residual_outputs[launch],
                    full_provider_stream
                ) == 0) {
                std::cerr << "full-v3 async chain launch failed launch="
                          << launch << " error=" << last_error() << std::endl;
                return 4;
            }
        }
        status = hipStreamSynchronize(full_provider_stream);
        if (status != hipSuccess) {
            return fail("full_provider_v3_async_sync(chain)", status);
        }
        const auto full_v3_async_stop = std::chrono::steady_clock::now();
        full_provider_v3_async_total_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(
                full_v3_async_stop - full_v3_async_start
            ).count()
        );
        full_provider_v3_async_ms = full_provider_v3_async_total_ms /
            static_cast<float>(kFullV3AsyncChain);

        std::vector<float> full_sync_output(output_elements);
        std::vector<float> full_async_output(output_elements);
        std::vector<float> full_v3_output(output_elements);
        std::vector<float> full_v3_async_output(output_elements);
        status = hipMemcpy(
            full_sync_output.data(),
            device_residual_output,
            output_elements * sizeof(float),
            hipMemcpyDeviceToHost
        );
        if (status == hipSuccess) {
            status = hipMemcpy(
                full_async_output.data(),
                device_async_residual_output,
                output_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                full_v3_output.data(),
                device_v3_residual_output,
                output_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status != hipSuccess) {
            return fail("hipMemcpy(full_provider_parity)", status);
        }
        for (size_t index = 0u; index < output_elements; ++index) {
            if (!std::isfinite(full_sync_output[index]) ||
                !std::isfinite(full_async_output[index]) ||
                !std::isfinite(full_v3_output[index])) {
                std::cerr << "full provider produced non-finite output index="
                          << index << std::endl;
                return 4;
            }
            if (std::memcmp(
                    &full_sync_output[index],
                    &full_async_output[index],
                    sizeof(float)
                ) != 0) {
                ++full_provider_async_mismatches;
            }
            if (std::memcmp(
                    &full_sync_output[index],
                    &full_v3_output[index],
                    sizeof(float)
                ) != 0) {
                ++full_provider_v3_mismatches;
            }
        }
        full_provider_sync_hash = fnv1a64_f32(full_sync_output);
        full_provider_async_hash = fnv1a64_f32(full_async_output);
        full_provider_v3_hash = fnv1a64_f32(full_v3_output);
        for (uint32_t launch = 0; launch < kFullV3AsyncChain; ++launch) {
            status = hipMemcpy(
                full_v3_async_output.data(),
                device_v3_async_residual_outputs[launch],
                output_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
            if (status != hipSuccess) {
                return fail("hipMemcpy(full_provider_v3_async_parity)", status);
            }
            for (size_t index = 0u; index < output_elements; ++index) {
                if (!std::isfinite(full_v3_async_output[index])) {
                    std::cerr << "full-v3 async produced non-finite output launch="
                              << launch << " index=" << index << std::endl;
                    return 4;
                }
                if (std::memcmp(
                        &full_sync_output[index],
                        &full_v3_async_output[index],
                        sizeof(float)
                    ) != 0) {
                    ++full_provider_v3_async_mismatches;
                }
            }
            full_provider_v3_async_hash = fnv1a64_f32(
                full_v3_async_output
            );
            if (full_provider_v3_async_hash != full_provider_sync_hash) {
                full_provider_v3_async_hashes_match = false;
            }
        }
        if (full_provider_async_mismatches != 0u ||
            full_provider_sync_hash != full_provider_async_hash) {
            std::cerr << "full provider async parity failed mismatches="
                      << full_provider_async_mismatches
                      << " sync_hash=" << std::hex << full_provider_sync_hash
                      << " async_hash=" << full_provider_async_hash
                      << std::dec << std::endl;
            return 4;
        }
        if (full_provider_v3_mismatches != 0u ||
            full_provider_sync_hash != full_provider_v3_hash) {
            std::cerr << "full provider v3 parity failed mismatches="
                      << full_provider_v3_mismatches
                      << " v2_hash=" << std::hex << full_provider_sync_hash
                      << " v3_hash=" << full_provider_v3_hash
                      << std::dec << std::endl;
            return 4;
        }
        if (full_provider_v3_async_mismatches != 0u ||
            !full_provider_v3_async_hashes_match ||
            full_provider_sync_hash != full_provider_v3_async_hash) {
            std::cerr << "full provider v3 async parity failed mismatches="
                      << full_provider_v3_async_mismatches
                      << " v2_hash=" << std::hex << full_provider_sync_hash
                      << " v3_async_hash=" << full_provider_v3_async_hash
                      << std::dec << std::endl;
            return 4;
        }
        for (size_t case_index = 0u;
             case_index < kDynamicLogicalTokens.size();
             ++case_index) {
            const uint32_t logical_tokens =
                kDynamicLogicalTokens[case_index];
            const size_t logical_elements =
                static_cast<size_t>(logical_tokens) * kHidden;
            std::array<
                DynamicLogicalTimingSample,
                kDynamicLogicalTimingSamples
            > timing_samples{};
            for (size_t sample_index = 0u;
                 sample_index < timing_samples.size();
                 ++sample_index) {
                hipLaunchKernelGGL(
                    fill_input_kernel,
                    grid_for(guarded_output_elements),
                    dim3(256),
                    0,
                    full_provider_stream,
                    device_v3_async_residual_outputs[0],
                    guarded_output_elements
                );
                status = hipGetLastError();
                if (status != hipSuccess) {
                    return fail("dynamic_full_provider_reset", status);
                }
                // The reset is correctness setup, not provider work. Complete
                // it before each sample so every logical length is measured
                // against the same warm-engine component boundary.
                status = hipStreamSynchronize(full_provider_stream);
                if (status != hipSuccess) {
                    return fail("dynamic_full_provider_reset_sync", status);
                }
                const auto dynamic_start =
                    std::chrono::steady_clock::now();
                if (provider_dynamic_full_launch(
                        device_input,
                        device_v3_async_residual_outputs[0],
                        device_router,
                        device_gate_up,
                        device_down,
                        device_shared_gate,
                        device_shared_gate_projection,
                        device_shared_up_projection,
                        device_shared_down,
                        device_v3_async_residual_outputs[0],
                        logical_tokens,
                        full_provider_stream
                    ) == 0) {
                    std::cerr
                        << "dynamic full provider launch failed tokens="
                        << logical_tokens << " sample=" << sample_index
                        << " error=" << last_error() << std::endl;
                    return 4;
                }
                const auto dynamic_submit_stop =
                    std::chrono::steady_clock::now();
                status = hipStreamSynchronize(full_provider_stream);
                if (status != hipSuccess) {
                    return fail("dynamic_full_provider_sync", status);
                }
                const auto dynamic_stop =
                    std::chrono::steady_clock::now();
                timing_samples[sample_index].total_ms = static_cast<float>(
                    std::chrono::duration<double, std::milli>(
                        dynamic_stop - dynamic_start
                    ).count()
                );
                timing_samples[sample_index].submit_ms = static_cast<float>(
                    std::chrono::duration<double, std::milli>(
                        dynamic_submit_stop - dynamic_start
                    ).count()
                );
            }
            dynamic_logical_first_sample_ms[case_index] =
                timing_samples.front().total_ms;
            std::sort(
                timing_samples.begin(),
                timing_samples.end(),
                [](const DynamicLogicalTimingSample &left,
                   const DynamicLogicalTimingSample &right) {
                    return left.total_ms < right.total_ms;
                }
            );
            const DynamicLogicalTimingSample &median_sample =
                timing_samples[timing_samples.size() / 2u];
            dynamic_logical_ms[case_index] = median_sample.total_ms;
            dynamic_logical_submit_ms[case_index] = median_sample.submit_ms;
            dynamic_logical_min_ms[case_index] =
                timing_samples.front().total_ms;
            dynamic_logical_max_ms[case_index] =
                timing_samples.back().total_ms;
            dynamic_logical_device_completion_ms[case_index] = (std::max)(
                0.0f,
                dynamic_logical_ms[case_index] -
                    dynamic_logical_submit_ms[case_index]
            );
            std::cout
                << "dynamic_logical_case index=" << case_index
                << " tokens=" << logical_tokens
                << " total_ms=" << dynamic_logical_ms[case_index]
                << " submit_ms=" << dynamic_logical_submit_ms[case_index]
                << " device_completion_ms="
                << dynamic_logical_device_completion_ms[case_index]
                << " timing_samples=" << kDynamicLogicalTimingSamples
                << " timing_stat=median"
                << " first_sample_ms="
                << dynamic_logical_first_sample_ms[case_index]
                << " min_ms=" << dynamic_logical_min_ms[case_index]
                << " max_ms=" << dynamic_logical_max_ms[case_index]
                << " reset_included=0"
                << " component_only=1"
                << " inference_success_claimed=0"
                << std::endl;
            std::vector<float> dynamic_output(logical_elements);
            status = hipMemcpy(
                dynamic_output.data(),
                device_v3_async_residual_outputs[0],
                logical_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
            if (status != hipSuccess) {
                return fail("hipMemcpy(dynamic_full_provider)", status);
            }
            for (size_t index = 0u; index < logical_elements; ++index) {
                const float candidate = dynamic_output[index];
                const float reference = full_sync_output[index];
                if (!std::isfinite(candidate) || !std::isfinite(reference)) {
                    ++dynamic_logical_nonfinite;
                } else {
                    const float difference = std::abs(candidate - reference);
                    dynamic_logical_max_abs_diff = (std::max)(
                        dynamic_logical_max_abs_diff,
                        difference
                    );
                    if (logical_tokens == kTokens) {
                        dynamic_logical_q8192_max_abs_diff = (std::max)(
                            dynamic_logical_q8192_max_abs_diff,
                            difference
                        );
                    }
                }
                if (std::memcmp(
                        &candidate,
                        &reference,
                        sizeof(float)
                    ) != 0) {
                    ++dynamic_logical_mismatches;
                    if (logical_tokens == kTokens) {
                        ++dynamic_logical_q8192_mismatches;
                    }
                }
            }
            float tail_guard = 0.0f;
            status = hipMemcpy(
                &tail_guard,
                device_v3_async_residual_outputs[0] + logical_elements,
                sizeof(float),
                hipMemcpyDeviceToHost
            );
            if (status != hipSuccess) {
                return fail("hipMemcpy(dynamic_tail_guard)", status);
            }
            dynamic_logical_tail_guard_pass =
                dynamic_logical_tail_guard_pass && tail_guard == 1.0f;
        }
        if (!dynamic_logical_tail_guard_pass ||
            dynamic_logical_nonfinite != 0u ||
            !std::isfinite(dynamic_logical_max_abs_diff) ||
            dynamic_logical_max_abs_diff > 0.03125f ||
            dynamic_logical_q8192_mismatches != 0u ||
            dynamic_logical_q8192_max_abs_diff != 0.0f) {
            std::cerr
                << "dynamic full provider parity failed mismatches="
                << dynamic_logical_mismatches
                << " nonfinite=" << dynamic_logical_nonfinite
                << " max_abs_diff=" << dynamic_logical_max_abs_diff
                << " q8192_mismatches="
                << dynamic_logical_q8192_mismatches
                << " q8192_max_abs_diff="
                << dynamic_logical_q8192_max_abs_diff
                << " tail_guard="
                << (dynamic_logical_tail_guard_pass ? 1 : 0)
                << std::endl;
            return 4;
        }
        hipLaunchKernelGGL(
            fill_input_kernel,
            grid_for(output_elements),
            dim3(256),
            0,
            full_provider_stream,
            device_input,
            output_elements
        );
        hipLaunchKernelGGL(
            fill_router_debug_oracle_kernel,
            grid_for(router_elements),
            dim3(256),
            0,
            full_provider_stream,
            device_router,
            router_elements
        );
        status = hipGetLastError();
        if (status != hipSuccess) {
            return fail("fill_router_debug_oracle", status);
        }
        if (router_launch(
                device_input,
                device_router,
                full_provider_stream
            ) == 0) {
            std::cerr << "router debug warmup failed error="
                      << last_error() << std::endl;
            return 4;
        }
        status = hipStreamSynchronize(full_provider_stream);
        if (status != hipSuccess) {
            return fail("router_debug_sync(warmup)", status);
        }
        status = hipEventRecord(start, full_provider_stream);
        if (status != hipSuccess) {
            return fail("hipEventRecord(router_debug_start)", status);
        }
        for (uint32_t launch = 0u; launch < kFullV3AsyncChain; ++launch) {
            if (router_launch(
                    device_input,
                    device_router,
                    full_provider_stream
                ) == 0) {
                std::cerr << "router debug launch failed error="
                          << last_error() << std::endl;
                return 4;
            }
        }
        status = hipEventRecord(stop, full_provider_stream);
        if (status != hipSuccess) {
            return fail("hipEventRecord(router_debug_stop)", status);
        }
        status = hipEventSynchronize(stop);
        if (status != hipSuccess) {
            return fail("hipEventSynchronize(router_debug_stop)", status);
        }
        float router_debug_total_ms = 0.0f;
        status = hipEventElapsedTime(
            &router_debug_total_ms,
            start,
            stop
        );
        if (status != hipSuccess) {
            return fail("hipEventElapsedTime(router_debug)", status);
        }
        router_debug_ms =
            router_debug_total_ms / static_cast<float>(kFullV3AsyncChain);
        std::vector<uint32_t> router_debug_ids(kRoutes);
        std::vector<float> router_debug_weights(kRoutes);
        if (copy_topk(
                router_debug_ids.data(),
                router_debug_weights.data()
            ) == 0) {
            std::cerr << "copy top-k debug failed error="
                      << last_error() << std::endl;
            return 4;
        }
        router_debug_first_id = router_debug_ids.front();
        for (uint32_t token = 0u; token < kTokens; ++token) {
            for (uint32_t route = 0u; route < kTopK; ++route) {
                const size_t index =
                    static_cast<size_t>(token) * kTopK + route;
                const uint32_t expected_id = 10u + route * 11u;
                if (router_debug_ids[index] != expected_id) {
                    ++router_debug_id_mismatches;
                }
                if (router_debug_weights[index] != 0.125f) {
                    ++router_debug_weight_mismatches;
                }
            }
        }
        provider_scratch_bytes = scratch_bytes();
        status = hipStreamDestroy(full_provider_stream);
        if (status != hipSuccess) {
            return fail("hipStreamDestroy(full_provider)", status);
        }
        release();
        close_provider_module(provider);
    }

    std::cout << "q8192_triton_selected_moe_smoke status=pass"
              << " host=" << local_host_name()
              << " tokens=" << kTokens
              << " routes=" << kRoutes
              << " total_post_pad=" << total_post_pad
              << " valid_routes=" << valid_routes
              << " repetitions=" << repetitions
              << " mean_pipeline_ms=" << elapsed_ms / static_cast<float>(repetitions)
              << " sort_ms=" << sort_ms
              << " input_convert_ms=" << input_convert_ms
              << " gate_up_ms=" << gate_up_ms
              << " down_ms=" << down_ms
              << " combine_ms=" << combine_ms
              << " provider_ms=" << provider_ms
              << " full_provider_ms=" << full_provider_ms
              << " full_provider_v2_reset_included_ms=" << full_provider_ms
              << " full_provider_warmup_ms=" << full_provider_warmup_ms
              << " full_provider_async_one_sync_ms=" << full_provider_async_ms
              << " full_provider_async_warmup_ms="
              << full_provider_async_warmup_ms
              << " full_provider_v3_reset_included_ms="
              << full_provider_v3_ms
              << " full_provider_v3_warmup_ms="
              << full_provider_v3_warmup_ms
              << " full_provider_v3_async_one_sync_ms="
              << full_provider_v3_async_ms
              << " full_provider_v3_async_chain_total_ms="
              << full_provider_v3_async_total_ms
              << " full_provider_v3_async_per_call_ms="
              << full_provider_v3_async_ms
              << " full_provider_v3_async_chain_calls="
              << kFullV3AsyncChain
              << " full_provider_v3_async_checked_outputs="
              << kFullV3AsyncChain
              << " full_provider_async_mismatches="
              << full_provider_async_mismatches
              << " full_provider_v3_mismatches="
              << full_provider_v3_mismatches
              << " full_provider_v3_async_mismatches="
              << full_provider_v3_async_mismatches
              << " dynamic_logical_q2073_ms="
              << dynamic_logical_ms[0]
              << " dynamic_logical_q2156_ms="
              << dynamic_logical_ms[1]
              << " dynamic_logical_q2560_ms="
              << dynamic_logical_ms[2]
              << " dynamic_logical_q3073_ms="
              << dynamic_logical_ms[3]
              << " dynamic_logical_q4609_ms="
              << dynamic_logical_ms[4]
              << " dynamic_logical_q6145_ms="
              << dynamic_logical_ms[5]
              << " dynamic_logical_case_count="
              << kDynamicLogicalTokens.size()
              << " dynamic_logical_timing_samples="
              << kDynamicLogicalTimingSamples
              << " dynamic_logical_timing_stat=median"
              << " dynamic_logical_min_tokens=2049"
              << " dynamic_logical_max_tokens=8192"
              << " dynamic_logical_mismatches="
              << dynamic_logical_mismatches
              << " dynamic_logical_nonfinite="
              << dynamic_logical_nonfinite
              << " dynamic_logical_max_abs_diff="
              << dynamic_logical_max_abs_diff
              << " dynamic_logical_q8192_mismatches="
              << dynamic_logical_q8192_mismatches
              << " dynamic_logical_q8192_max_abs_diff="
              << dynamic_logical_q8192_max_abs_diff
              << " dynamic_logical_tail_guard_pass="
              << (dynamic_logical_tail_guard_pass ? 1 : 0)
              << " dynamic_logical_reset_included=0"
              << " dynamic_logical_component_only=1"
              << " inference_success_claimed=0"
              << " full_provider_sync_hash=" << std::hex
              << full_provider_sync_hash
              << " full_provider_async_hash=" << full_provider_async_hash
              << " full_provider_v3_hash=" << full_provider_v3_hash
              << " full_provider_v3_async_hash="
              << full_provider_v3_async_hash
              << " expected_full_provider_hash="
              << kExpectedFullProviderHash
              << std::dec
              << " expected_full_provider_hash_pass="
              << (full_provider_sync_hash == kExpectedFullProviderHash ? 1 : 0)
              << " provider_backend_mask=" << provider_backend_mask
              << " expected_provider_backend_mask="
              << kExpectedProviderBackendMask
              << " provider_backend_mask_pass="
              << (provider_backend_mask == kExpectedProviderBackendMask ? 1 : 0)
              << " full_provider_v3_async_under_29ms="
              << (full_provider_v3_async_ms > 0.0f &&
                          full_provider_v3_async_ms <= 29.0f
                      ? 1
                      : 0)
              << " router_debug_ms=" << router_debug_ms
              << " router_debug_id_mismatches="
              << router_debug_id_mismatches
              << " router_debug_weight_mismatches="
              << router_debug_weight_mismatches
              << " router_debug_first_id=" << router_debug_first_id
              << " router_debug_oracle=ranked_equal_top8_bf16"
              << " provider_scratch_bytes=" << provider_scratch_bytes
              << " full_input_pattern=nonuniform"
              << " full_weight_pattern=nonzero"
              << " output0=" << output_samples.front()
              << " output_last=" << output_samples.back()
              << std::endl;
    return 0;
}
