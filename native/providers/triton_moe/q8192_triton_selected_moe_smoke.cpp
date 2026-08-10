#include <hip/hip_runtime.h>

#include <algorithm>
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
namespace {

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
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: q8192_triton_selected_moe_smoke KERNEL_DIR [REPETITIONS] [PROVIDER_DLL]\n";
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
    const size_t activated_elements = static_cast<size_t>(kRoutes) * kIntermediate;
    const size_t route_output_elements = static_cast<size_t>(kRoutes) * kHidden;
    const size_t output_elements = static_cast<size_t>(kTokens) * kHidden;
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
        ALLOCATE(output, output_elements * sizeof(float), "v3_async_residual_output");
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
    if (status != hipSuccess) return fail("initialize_full_provider_weights", status);
    hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, device_residual_output, output_elements);
    hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, device_async_residual_output, output_elements);
    hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, device_v3_residual_output, output_elements);
    for (float *output : device_v3_async_residual_outputs) {
        hipLaunchKernelGGL(fill_input_kernel, grid_for(output_elements), dim3(256), 0, 0, output, output_elements);
    }
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        return fail("initialize_weights", status);
    }

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
        local = launch(count, {&device_topk_ids, &device_counts});
        if (local != hipSuccess) return local;
        local = launch(prefix, {&device_counts});
        if (local != hipSuccess) return local;
        local = launch(padded_prefix, {&device_total_post_pad, &device_counts, &device_cumsum});
        if (local != hipSuccess) return local;
        local = launch(scatter, {&device_topk_ids, &device_sorted_routes, &device_block_experts, &device_counts, &device_cumsum});
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
            if (topk_ids[static_cast<size_t>(route)] != expert || route <= previous_route) {
                std::cerr << "invalid stable route block=" << block << " lane=" << lane
                          << " route=" << route << " expert=" << expert << std::endl;
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
        local = launch(count, {&device_topk_ids, &device_counts});
        if (local != hipSuccess) return local;
        local = launch(prefix, {&device_counts});
        if (local != hipSuccess) return local;
        local = launch(padded_prefix, {&device_total_post_pad, &device_counts, &device_cumsum});
        if (local != hipSuccess) return local;
        return launch(scatter, {&device_topk_ids, &device_sorted_routes, &device_block_experts, &device_counts, &device_cumsum});
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
    if (argc == 4) {
#if defined(_WIN32)
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
        using LastErrorFunction = const char *(*)();
        using BackendMaskFunction = uint32_t (*)();
        using RouterLaunchFunction = int (*)(
            const float *,
            const uint16_t *,
            void *
        );
        using CopyTopkFunction = int (*)(uint32_t *, float *);
        using ScratchBytesFunction = uint64_t (*)();
        using ReleaseFunction = void (*)();
        const HMODULE provider = LoadLibraryA(argv[3]);
        if (provider == nullptr) {
            std::cerr << "LoadLibrary(provider) failed win32_error=" << GetLastError() << std::endl;
            return 4;
        }
        const auto prepare = reinterpret_cast<PrepareFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_prepare"));
        const auto provider_launch = reinterpret_cast<LaunchFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_launch"));
        const auto provider_full_launch = reinterpret_cast<FullLaunchFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_launch_full_v2"));
        const auto provider_full_launch_async = reinterpret_cast<FullLaunchFunction>(
            GetProcAddress(
                provider,
                "qrt_triton_moe_q8192_launch_full_v2_async"
            ));
        const auto provider_full_launch_v3 = reinterpret_cast<FullLaunchFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_launch_full_v3"));
        const auto provider_full_launch_v3_async =
            reinterpret_cast<FullLaunchFunction>(GetProcAddress(
                provider,
                "qrt_triton_moe_q8192_launch_full_v3_async"
            ));
        const auto last_error = reinterpret_cast<LastErrorFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_last_error"));
        const auto backend_mask = reinterpret_cast<BackendMaskFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_backend_mask"));
        const auto router_launch = reinterpret_cast<RouterLaunchFunction>(
            GetProcAddress(
                provider,
                "qrt_triton_moe_q8192_launch_router_debug"
            ));
        const auto copy_topk = reinterpret_cast<CopyTopkFunction>(
            GetProcAddress(
                provider,
                "qrt_triton_moe_q8192_copy_topk_debug"
            ));
        const auto scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_scratch_bytes"));
        const auto release = reinterpret_cast<ReleaseFunction>(
            GetProcAddress(provider, "qrt_triton_moe_q8192_release"));
        if (prepare == nullptr || provider_launch == nullptr ||
            provider_full_launch == nullptr ||
            provider_full_launch_async == nullptr ||
            provider_full_launch_v3 == nullptr ||
            provider_full_launch_v3_async == nullptr ||
            last_error == nullptr ||
            backend_mask == nullptr ||
            router_launch == nullptr || copy_topk == nullptr ||
            scratch_bytes == nullptr || release == nullptr) {
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
        const std::string kernel_dir_string = kernel_dir.string();
        if (prepare(kernel_dir_string.c_str()) == 0) {
            std::cerr << "provider prepare failed error=" << last_error() << std::endl;
            return 4;
        }
        hipStream_t full_provider_stream = nullptr;
        status = hipStreamCreate(&full_provider_stream);
        if (status != hipSuccess) {
            return fail("hipStreamCreate(full_provider)", status);
        }
        provider_scratch_bytes = scratch_bytes();
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
            if (!std::isfinite(actual) || std::abs(actual - expected_samples[token]) > 0.03125f) {
                std::cerr << "provider output mismatch token=" << token
                          << " actual=" << actual
                          << " expected=" << expected_samples[token] << std::endl;
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
        FreeLibrary(provider);
#else
        std::cerr << "provider DLL smoke is only supported on Windows" << std::endl;
        return 4;
#endif
    }

    std::cout << "q8192_triton_selected_moe_smoke status=pass"
              << " host=baiying"
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
              << " provider_scratch_bytes=" << provider_scratch_bytes
              << " full_input_pattern=nonuniform"
              << " full_weight_pattern=nonzero"
              << " output0=" << output_samples.front()
              << " output_last=" << output_samples.back()
              << std::endl;
    return 0;
}
