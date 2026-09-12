// Use the original provider's producer, norms, candidate predicate and replay.
// The 8192-row slab repeats three observed input rows to preserve the GEMM
// shape. Only the three original positions are compared to GB10 boundaries.
#define main qrt_original_diagnostic_main
#include "../../native/providers/whole_provider.cpp"
#undef main
#include <stdexcept>

namespace selector_replay {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
std::vector<uint16_t> read(const char *path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    std::vector<uint16_t> v(count); f.read(reinterpret_cast<char *>(v.data()), count*2);
    if (f.gcount() != std::streamsize(count*2)) throw std::runtime_error("input span");
    char extra; if (f.read(&extra, 1)) throw std::runtime_error("input tail");
    return v;
}
__global__ void admission(const float *raw, const float *norms, const float *weight_norms,
                          unsigned *selected, unsigned ppb) {
    unsigned cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= 3*2048) return;
    size_t index = size_t(2169)*2048 + cell;
    selected[cell] = selected_bf16_projection_hawkeye_candidate(
        raw[index], index, 2048, 512, 0, ppb, nullptr, norms, weight_norms);
}
}
int main(int argc, char **argv) try {
    using namespace selector_replay;
    if (argc != 4) throw std::runtime_error("inputs weights expected required");
    hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, "gfx1151", 7)) throw std::runtime_error("requires gfx1151");
    auto original = read(argv[1], 3*4096), weights = read(argv[2], 2048*4096);
    auto expected = read(argv[3], 3*2048);
    std::vector<uint16_t> inputs(8192*4096);
    for (unsigned row = 0; row < 8192; ++row)
        std::copy_n(original.data()+(row%3)*4096, 4096, inputs.data()+row*4096);
    uint16_t *dx = nullptr, *dw = nullptr;
    float *raw = nullptr, *xn = nullptr, *wn = nullptr; unsigned *flags = nullptr;
    check(hipMalloc(reinterpret_cast<void **>(&dx), inputs.size()*2));
    check(hipMalloc(reinterpret_cast<void **>(&dw), weights.size()*2));
    check(hipMalloc(reinterpret_cast<void **>(&raw), size_t(8192)*2048*4));
    check(hipMalloc(reinterpret_cast<void **>(&xn), 8192*4));
    check(hipMalloc(reinterpret_cast<void **>(&wn), 2048*4));
    check(hipMalloc(reinterpret_cast<void **>(&flags), 3*2048*4));
    check(hipMemcpy(dx, inputs.data(), inputs.size()*2, hipMemcpyHostToDevice));
    check(hipMemcpy(dw, weights.data(), weights.size()*2, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel, dim3(8192), dim3(256), 0, nullptr,
                      dx, xn, 8192, 4096);
    hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel, dim3(2048), dim3(256), 0, nullptr,
                      dw, wn, 2048, 4096);
    check(hipGetLastError()); check(hipDeviceSynchronize());
    float input_norm, weight_norm;
    check(hipMemcpy(&input_norm, xn+2170, 4, hipMemcpyDeviceToHost));
    check(hipMemcpy(&weight_norm, wn+1875, 4, hipMemcpyDeviceToHost));
    std::vector<float> before(3*2048), after(before.size()); std::vector<unsigned> selected(before.size());
    for (unsigned ppb : {1000u, 10000u}) {
        std::string stage, failure;
        if (!resident_bf16_matrix_matmul_f32_output(dw, dx, raw, 2048, 4096, 8192,
                nullptr, "original_projection_shape_replay", &stage, &failure))
            throw std::runtime_error(stage+": "+failure);
        check(hipDeviceSynchronize());
        check(hipMemcpy(before.data(), raw+size_t(2169)*2048, before.size()*4, hipMemcpyDeviceToHost));
        hipLaunchKernelGGL(selector_replay::admission, dim3(24), dim3(256), 0, nullptr,
                          raw, xn, wn, flags, ppb);
        check(hipGetLastError()); check(hipDeviceSynchronize());
        check(hipMemcpy(selected.data(), flags, selected.size()*4, hipMemcpyDeviceToHost));
        unsigned initial_bad=0, missed=0, admitted=0;
        for (unsigned i=0; i<before.size(); ++i) {
            bool bad=host_full_attention_bf16_rne(before[i])!=expected[i];
            initial_bad+=bad;missed+=bad&&!selected[i];admitted+=selected[i]!=0;
        }
        check(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw, dx, nullptr,
            xn, wn, raw, 2048, 8192, 4096, 512, 0, ppb, 4096, nullptr));
        check(hipMemcpy(after.data(), raw+size_t(2169)*2048, after.size()*4, hipMemcpyDeviceToHost));
        unsigned final_bad=0;
        for (unsigned i=0;i<after.size();++i)final_bad+=host_full_attention_bf16_rne(after[i])!=expected[i];
        std::printf("{\"kind\":\"original_projection_admission\",\"repeated_input_rows\":3,\"execution_tokens\":8192,\"compared_original_positions\":[18553,18554,18555],\"elements\":6144,\"ppb\":%u,\"initial_bf16_mismatches\":%u,\"admitted\":%u,\"missed_mismatches\":%u,\"final_bf16_mismatches\":%u,\"cell_before\":%.17g,\"cell_after\":%.17g,\"cell_selected\":%u,\"cell_input_norm\":%.17g,\"cell_weight_norm\":%.17g,\"cell_nearest_midpoint_distance\":%.17g,\"inference_acceptance\":false}\n",
            ppb,initial_bad,admitted,missed,final_bad,before[2048+1875],after[2048+1875],
            selected[2048+1875],input_norm,weight_norm,qrt_bf16_midpoint::nearest_distance(before[2048+1875]));
        std::fflush(stdout);
        if (ppb == 1000 && (missed == 0 || final_bad == 0))
            throw std::runtime_error("original admission miss was not reproduced");
        if (ppb == 10000 && (missed != 0 || final_bad != 0))
            throw std::runtime_error("expanded admission differs from original GB10 rows");
    }
    std::vector<uint16_t> after_inputs(inputs.size()), after_weights(weights.size());
    check(hipMemcpy(after_inputs.data(), dx, inputs.size()*2, hipMemcpyDeviceToHost));
    check(hipMemcpy(after_weights.data(), dw, weights.size()*2, hipMemcpyDeviceToHost));
    if (after_inputs != inputs || after_weights != weights)
        throw std::runtime_error("original inputs or model weights changed");
    free_device(flags);free_device(wn);free_device(xn);free_device(raw);free_device(dw);free_device(dx);
    return 0;
} catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 2;}
