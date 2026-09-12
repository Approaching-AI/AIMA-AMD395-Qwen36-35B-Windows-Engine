// Read-only captured-input numerical replay. No model, oracle output, or
// reference tensor is consumed by device computation.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "blackwell_attention.h"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
void check(hipError_t result) {
    if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}
template<class T> std::vector<T> read(const std::string& path, size_t elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(elements * sizeof(T)))
        throw std::runtime_error("capture size mismatch: " + path);
    std::vector<T> result(elements);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(result.data()), elements * sizeof(T)))
        throw std::runtime_error("short capture read: " + path);
    return result;
}
template<class T> void write(const std::string& path, const std::vector<T>& data) {
    // CREATE_NEW is atomic: a prior capture can never be overwritten.
    HANDLE file = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr,
                             CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("output exists: " + path);
    DWORD written = 0;
    const DWORD bytes = DWORD(data.size() * sizeof(T));
    const bool ok = WriteFile(file, data.data(), bytes, &written, nullptr) && written == bytes;
    CloseHandle(file);
    if (!ok) throw std::runtime_error("short capture write: " + path);
}
uint16_t bf16(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return uint16_t((bits >> 16) | ((bits & 0x7fffffu) ? 0x40u : 0u));
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float fp32(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) (void)hipEventDestroy(value); }
};
float finish(Event& begin, Event& end, float limit_ms = 3000.0f) {
    check(hipEventRecord(end.value)); check(hipEventSynchronize(end.value));
    float result; check(hipEventElapsedTime(&result, begin.value, end.value));
    if (!std::isfinite(result) || result > limit_ms)
        throw std::runtime_error("component interval exceeds its time bound");
    return result;
}
bool report(const char* route, const std::vector<float>& output,
            const std::vector<uint16_t>& reference, unsigned start,
            const std::string& prefix, float total_ms, float max_ms,
            unsigned memory_layout = 0u, float scores_ms = 0.0f,
            float probabilities_ms = 0.0f, float value_ms = 0.0f,
            float preparation_ms = 0.0f, bool native_products = false,
            double completed_host_ms = 0.0, uint64_t compacted_pv_cells = 0u) {
    size_t mismatches = 0, nonfinite = 0, first = size_t(-1), affected = 0;
    double error2 = 0, norm2 = 0; float maximum_error = 0;
    std::vector<uint16_t> rounded(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        const auto expected = reference[size_t(start) * 4096u + i];
        rounded[i] = bf16(output[i]);
        if (!std::isfinite(output[i])) ++nonfinite;
        if (rounded[i] != expected) { ++mismatches; if (first == size_t(-1)) first = i; }
        const double difference = double(fp32(rounded[i])) - fp32(expected);
        error2 += difference * difference; norm2 += double(fp32(expected)) * fp32(expected);
        maximum_error = std::max(maximum_error, float(std::abs(difference)));
    }
    for (size_t token = 0; token < output.size() / 4096u; ++token) {
        const auto* r = reference.data() + (size_t(start) + token) * 4096u;
        if (std::memcmp(rounded.data() + token * 4096u, r, 8192u)) ++affected;
    }
    size_t affected_heads = 0u;
    constexpr double margin_ppm[] = {1.0, 4.0, 16.0, 64.0, 256.0};
    size_t candidate_cells[5]{}, candidate_heads[5]{}, missed[5]{};
    for (size_t head = 0u; head < output.size() / 256u; ++head) {
        const auto* expected = reference.data() + size_t(start) * 4096u + head * 256u;
        affected_heads += std::memcmp(rounded.data() + head * 256u, expected, 512u) != 0;
        if (memory_layout != 6u && memory_layout != 7u) continue;
        double peak = 0.0;
        for (unsigned d = 0u; d < 256u; ++d) peak = std::max(peak, std::abs(double(output[head * 256u + d])));
        bool selected[5]{};
        for (unsigned d = 0u; d < 256u; ++d) {
            const size_t index = head * 256u + d;
            const uint32_t magnitude = qrt_sm121_native_product::float_bits(output[index]) & 0x7fffffffu;
            const double midpoint = qrt_sm121_native_product::from_bits((magnitude & 0xffff0000u) | 0x8000u);
            const double margin = std::abs(std::abs(double(output[index])) - midpoint);
            for (unsigned j = 0u; j < 5u; ++j) {
                const bool candidate = margin <= peak * margin_ppm[j] * 1e-6;
                candidate_cells[j] += candidate; selected[j] |= candidate;
                missed[j] += rounded[index] != expected[d] && !candidate;
            }
        }
        for (unsigned j = 0u; j < 5u; ++j) candidate_heads[j] += selected[j];
    }
    write(prefix + "-" + route + "-f32.bin", output);
    write(prefix + "-" + route + "-bf16.bin", rounded);
    const char* interval_kind = "kernel_dispatch";
    if (std::strcmp(route, "ck") == 0) interval_kind = "provider_call";
    else if (memory_layout == 15u) interval_kind = "shared_operand_exact_qk_and_exact_pv";
    else if (memory_layout == 16u) interval_kind = "shared_exact_qk_warp_softmax_exact_pv";
    else if (memory_layout == 17u) interval_kind = "shared_exact_qk_prepared_exact_pv";
    else if (memory_layout == 18u) interval_kind = "cell_parallel_integer_qk_prepared_exact_pv";
    else if (memory_layout == 19u) interval_kind = "sparse_integer_core_qk_prepared_exact_pv";
    else if (memory_layout == 20u) interval_kind = "cell_parallel_sparse_integer_core_qk_prepared_exact_pv";
    else if (memory_layout == 21u) interval_kind = "prepacked_sparse_integer_core_qk_prepared_exact_pv";
    else if (memory_layout == 22u) interval_kind = "tiled_exact_qk_native_pv_global_exact_replay";
    else if (memory_layout == 24u) interval_kind = "parallel_probability_tiled_exact_qk_global_pv_replay";
    else if (memory_layout == 23u) interval_kind = "tiled_exact_qk_native_pv_local_exact_replay";
    else if (memory_layout == 14u) interval_kind = "native_qk_and_exact_probability_pv";
    else if (memory_layout == 13u) interval_kind = "exact_qk_native_pv_and_selective_exact_pv_replay";
    else if (memory_layout >= 10u) interval_kind = "key_transpose_and_strided_pair_qk_pv";
    else if (memory_layout == 9u) interval_kind = "prepacked_integer_qk_probability_pv_triplets";
    else if (memory_layout == 8u) interval_kind = "cooperative_qk_probability_pv_triplets";
    else if (memory_layout >= 6u) interval_kind = "key_transpose_and_native_mma_triplets";
    else if (memory_layout == 5u) interval_kind = "key_transpose_and_mantissa_wmma_triplets";
    else if (memory_layout == 4u) interval_kind = "key_transpose_and_qk_pv_pairs";
    else if (memory_layout == 3u) interval_kind = "qk_probability_pv_dispatch_triplet";
    else if (memory_layout == 2u) interval_kind = "qk_pv_dispatch_pair";
    std::cout << "{\"kind\":\"full_attention_capture_replay\",\"route\":\"" << route
              << "\",\"query_start\":" << start << ",\"elements\":" << output.size()
              << ",\"bf16_mismatches\":" << mismatches << ",\"affected_tokens\":" << affected
              << ",\"nonfinite\":" << nonfinite << ",\"first_mismatch\":" << (first == size_t(-1) ? -1LL : (long long)first)
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1e-300))
              << ",\"interval_kind\":\"" << interval_kind
              << "\",\"interval_total_ms\":" << total_ms << ",\"maximum_interval_ms\":" << max_ms
              << ",\"memory_layout\":" << memory_layout
              << ",\"native_products\":" << (native_products ? "true" : "false")
              << ",\"paired_products\":" << (std::strcmp(route, "ck") && qrt_blackwell_attention::kPairedProducts &&
                  !native_products && (memory_layout == 4u || memory_layout == 8u) ? "true" : "false")
              << ",\"mantissa_wmma\":" << (memory_layout == 5u ? "true" : "false")
              << ",\"integer_wmma\":" << ((memory_layout == 5u || memory_layout == 9u || (memory_layout >= 18u && memory_layout <= 21u)) ? "true" : "false")
              << ",\"cell_parallel_integer_qk\":" << (memory_layout == 18u || memory_layout == 20u ? "true" : "false")
              << ",\"sparse_integer_core_qk\":" << (memory_layout >= 19u && memory_layout <= 21u ? "true" : "false")
              << ",\"prepacked_integer_core\":" << (memory_layout == 21u ? "true" : "false")
              << ",\"prepacked_integer\":" << (memory_layout == 9u ? "true" : "false")
              << ",\"native_mma_pv\":" << ((memory_layout == 6u || memory_layout == 7u || memory_layout == 13u || memory_layout == 22u || memory_layout == 23u || memory_layout == 24u) ? "true" : "false")
              << ",\"selective_exact_pv_replay\":" << (memory_layout == 13u || memory_layout == 22u || memory_layout == 23u || memory_layout == 24u ? "true" : "false")
              << ",\"compacted_pv_replay\":" << ((memory_layout == 22u || memory_layout == 24u) ? "true" : "false")
              << ",\"parallel_probability\":" << (memory_layout == 24u ? "true" : "false")
              << ",\"compacted_pv_cells\":" << compacted_pv_cells
              << ",\"completed_host_ms\":" << completed_host_ms
              << ",\"replay_count_host_reads_component_only\":" << ((memory_layout == 22u || memory_layout == 24u) ? "true" : "false")
              << ",\"tiled_exact_qk\":" << ((memory_layout >= 15u && memory_layout <= 17u) || memory_layout == 22u || memory_layout == 23u || memory_layout == 24u ? "true" : "false")
              << ",\"prepared_value_encoding\":" << (memory_layout >= 17u && memory_layout <= 21u ? "true" : "false")
              << ",\"native_mma_qk\":" << ((memory_layout == 7u || memory_layout == 14u) ? "true" : "false")
              << ",\"strided_pair_qk\":" << ((memory_layout == 10u || memory_layout == 11u) ? "true" : "false")
              << ",\"strided_pair_pv\":" << ((memory_layout == 10u || memory_layout == 12u) ? "true" : "false")
              << ",\"score_probability_redzones_checked\":" << (std::strcmp(route, "ck") ? "true" : "false")
              << ",\"stage_timing_enabled\":" << (memory_layout >= 2u ? "true" : "false")
              << ",\"scores_ms\":" << scores_ms
              << ",\"probabilities_ms\":" << probabilities_ms
              << ",\"value_ms\":" << value_ms
              << ",\"preparation_ms\":" << preparation_ms
              << ",\"key_transpose_and_redzones_checked\":" << ((qrt_blackwell_attention::split_transposed_keys(memory_layout)) ? "true" : "false")
              << ",\"affected_query_heads\":" << affected_heads
              << ",\"candidate_bounds_are_diagnostics\":true,\"head_peak_margin_sweep\":[";
    if (memory_layout == 6u || memory_layout == 7u) for (unsigned j = 0u; j < 5u; ++j)
        std::cout << (j ? "," : "") << "{\"ppm\":" << margin_ppm[j]
                  << ",\"candidate_cells\":" << candidate_cells[j]
                  << ",\"candidate_heads\":" << candidate_heads[j]
                  << ",\"missed_bf16_differences\":" << missed[j] << "}";
    std::cout << "],\"reference_is_compute_input\":false,\"inference_acceptance\":false}" << std::endl;
    return mismatches == 0u && nonfinite == 0u;
}
unsigned parse(const char* text, unsigned maximum) {
    char* end = nullptr; const auto n = std::strtoul(text, &end, 10);
    if (text == end || *end || n > maximum) throw std::runtime_error("invalid bounded integer");
    return unsigned(n);
}

int tiled_qk_safety() {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::string(properties.gcnArchName).rfind("gfx1151", 0) != 0)
        throw std::runtime_error("tiled QK safety requires gfx1151");
    constexpr unsigned tokens=67u, heads=16u, dim=256u;
    constexpr unsigned starts[]={0u,3u,33u,64u}, counts[]={1u,17u,32u,3u};
    size_t compared=0, attention_compared=0;
    for(unsigned test=0u;test<4u;++test) {
        std::vector<uint16_t> q(size_t(tokens)*heads*dim), k(size_t(tokens)*2u*dim);
        for(size_t i=0;i<q.size();++i) q[i]=uint16_t(((i*37u)&0x807fu)|((124u+i%6u)<<7u));
        for(size_t i=0;i<k.size();++i) k[i]=uint16_t(((i*53u)&0x807fu)|((122u+i%8u)<<7u));
        if(test>=2u) {
            // Force raw reloads for exponent-range and subnormal operands.
            q[size_t(starts[test])*heads*dim]=uint16_t((63u<<7u)|19u);
            q[size_t(starts[test])*heads*dim+1u]=1u;
            k[0]=uint16_t((192u<<7u)|11u); k[1]=0x8000u;
        }
        const unsigned start=starts[test], count=counts[test], stride=start+count;
        const size_t cells=size_t(count)*heads*stride;
        Device dq(q.size()*2u), dk(k.size()*2u), transposed(k.size()*2u);
        Device exact((cells+128u)*4u), tiled((cells+128u)*4u);
        check(hipMemcpy(dq.pointer,q.data(),q.size()*2u,hipMemcpyHostToDevice));
        check(hipMemcpy(dk.pointer,k.data(),k.size()*2u,hipMemcpyHostToDevice));
        check(hipMemset(exact.pointer,0xa5,(cells+128u)*4u));
        check(hipMemset(tiled.pointer,0xa5,(cells+128u)*4u));
        check(hipError_t(qrt_blackwell_attention::transpose_keys(
            dk.as<uint16_t>(),transposed.as<uint16_t>(),k.size(),tokens,nullptr)));
        Event begin,end; check(hipEventRecord(begin.value));
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_blackwell_attention::blackwell_transposed_scores_kernel<false>),
            dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>(),transposed.as<uint16_t>(),exact.as<float>()+64u,start,count,stride,tokens);
        check(hipGetLastError());
        hipLaunchKernelGGL(qrt_blackwell_attention::blackwell_tiled_exact_scores_kernel,
            dim3((stride+31u)/32u,heads,(count+7u)/8u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>(),transposed.as<uint16_t>(),tiled.as<float>()+64u,start,count,stride,tokens);
        check(hipGetLastError()); (void)finish(begin,end,1000.0f);
        std::vector<uint32_t> a(cells+128u),b(cells+128u);
        check(hipMemcpy(a.data(),exact.pointer,a.size()*4u,hipMemcpyDeviceToHost));
        check(hipMemcpy(b.data(),tiled.pointer,b.size()*4u,hipMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();++i) {
            if(i<64u || i>=cells+64u) {
                if(a[i]!=0xa5a5a5a5u || b[i]!=0xa5a5a5a5u) throw std::runtime_error("tiled QK redzone changed");
            } else if(a[i]!=b[i]) throw std::runtime_error("tiled QK score differs from exact scalar control");
        }
        compared+=cells;
        std::vector<uint16_t> v(k.size());
        for(size_t i=0;i<v.size();++i) v[i]=uint16_t(((i*97u)&0x807fu)|((124u+i%6u)<<7u));
        Device dv(v.size()*2u);
        check(hipMemcpy(dv.pointer,v.data(),v.size()*2u,hipMemcpyHostToDevice));
        constexpr unsigned output_start=2u;
        const size_t output_cells=size_t(output_start+count)*heads*dim;
        Device old_output((output_cells+128u)*4u), warp_output((output_cells+128u)*4u);
        for(bool vllm_sum : {false,true}) {
            check(hipMemset(old_output.pointer,0xa5,(output_cells+128u)*4u));
            check(hipMemset(warp_output.pointer,0xa5,(output_cells+128u)*4u));
            Event pv_begin,pv_end; check(hipEventRecord(pv_begin.value));
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_blackwell_attention::blackwell_exact_attention_kernel<true,true>),
                dim3(heads,count),dim3(dim),0u,nullptr,
                dq.as<uint16_t>(),dk.as<uint16_t>(),dv.as<uint16_t>(),old_output.as<float>()+64u,
                start,output_start,nullptr,nullptr,nullptr,vllm_sum,nullptr,exact.as<float>()+64u,stride,nullptr,0u);
            check(hipGetLastError());
            hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_blackwell_attention::blackwell_exact_attention_kernel<true,true,false,false,false,true>),
                dim3(heads,count),dim3(dim),0u,nullptr,
                dq.as<uint16_t>(),dk.as<uint16_t>(),dv.as<uint16_t>(),warp_output.as<float>()+64u,
                start,output_start,nullptr,nullptr,nullptr,vllm_sum,nullptr,exact.as<float>()+64u,stride,nullptr,0u);
            check(hipGetLastError()); (void)finish(pv_begin,pv_end,1000.0f);
            std::vector<uint32_t> old_values(output_cells+128u),warp_values(output_cells+128u);
            check(hipMemcpy(old_values.data(),old_output.pointer,old_values.size()*4u,hipMemcpyDeviceToHost));
            check(hipMemcpy(warp_values.data(),warp_output.pointer,warp_values.size()*4u,hipMemcpyDeviceToHost));
            for(size_t i=0;i<old_values.size();++i) {
                if(i<64u+size_t(output_start)*heads*dim || i>=64u+output_cells) {
                    if(old_values[i]!=0xa5a5a5a5u || warp_values[i]!=0xa5a5a5a5u)
                        throw std::runtime_error("warp softmax output redzone changed");
                } else if(old_values[i]!=warp_values[i]) throw std::runtime_error("warp softmax differs from original FP32 attention");
            }
            attention_compared+=size_t(count)*heads*dim;
        }
    }
    std::cout << "{\"kind\":\"tiled_exact_qk_safety\",\"cases\":4,\"fp32_scores_compared\":" << compared
              << ",\"warp_softmax_fp32_outputs_compared\":" << attention_compared
              << ",\"mismatches\":0,\"offset_and_partial_tiles\":true,\"range_and_subnormal_fallback\":true,"
                 "\"redzones_pass\":true,\"model_loaded\":false,\"inference_acceptance\":false}" << std::endl;
    return 0;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--tiled-qk-safety") == 0) return tiled_qk_safety();
        if (argc != 13 && argc != 14) throw std::runtime_error(
            "usage: replay Q K V reference CK_DLL output_prefix tokens query_start count batch exp2_table_or_dash baseline_0_or_1 [memory_layout_0_to_24]");
        const unsigned tokens = parse(argv[7], qrt_blackwell_attention::kSplitMaxTokens);
        const unsigned start = parse(argv[8], qrt_blackwell_attention::kSplitMaxTokens - 1u);
        const unsigned count = parse(argv[9], 8192), batch = parse(argv[10], 128);
        const bool baseline = parse(argv[12], 1) != 0;
        const unsigned memory_layout = argc == 14 ? parse(argv[13], 24) : 0u;
        const char* native_product_option = std::getenv("QRT_CK_SM121_NATIVE_PRODUCTS");
        const bool native_products = native_product_option && native_product_option[0] != '\0' &&
            std::strcmp(native_product_option, "0") != 0;
        if (native_products && memory_layout != 4u) throw std::runtime_error("native products require transposed split attention");
        bool matched = true;
        if (!tokens || !count || !batch || batch > qrt_blackwell_attention::split_query_limit(memory_layout, start + count) ||
            start >= tokens || count > tokens - start)
            throw std::runtime_error("invalid query span");
        if (baseline && tokens > 8192u)
            throw std::runtime_error("CK baseline is bounded to 8192 tokens");
        hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
        if (std::string(properties.gcnArchName).find("gfx1151") != 0)
            throw std::runtime_error("requires gfx1151");
        const auto begun = Clock::now();
        const auto q = read<uint16_t>(argv[1], size_t(tokens) * 4096u);
        const auto k = read<uint16_t>(argv[2], size_t(tokens) * 512u);
        const auto v = read<uint16_t>(argv[3], size_t(tokens) * 512u);
        const auto reference = read<uint16_t>(argv[4], size_t(tokens) * 4096u);
        Device dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2);
        check(hipMemcpy(dq.pointer, q.data(), q.size() * 2, hipMemcpyHostToDevice));
        check(hipMemcpy(dk.pointer, k.data(), k.size() * 2, hipMemcpyHostToDevice));
        check(hipMemcpy(dv.pointer, v.data(), v.size() * 2, hipMemcpyHostToDevice));
        Event begin, end;
        if (baseline) {
            HMODULE dll = LoadLibraryA(argv[5]);
            if (!dll) throw std::runtime_error("cannot load CK provider");
            using Launch = int (*)(const uint16_t*, const uint16_t*, const uint16_t*, float*, void*, unsigned);
            auto launch = reinterpret_cast<Launch>(GetProcAddress(dll, "qrt_ck_fmha_dynamic_bf16_launch"));
            if (!launch) { FreeLibrary(dll); throw std::runtime_error("missing dynamic CK export"); }
            Device output(size_t(tokens) * 4096u * 4u);
            check(hipEventRecord(begin.value));
            check(hipError_t(launch(dq.as<uint16_t>(), dk.as<uint16_t>(), dv.as<uint16_t>(), output.as<float>(), nullptr, tokens)));
            // A provider call may contain many internally bounded dispatches.
            const float ms = finish(begin, end, 20000.0f);
            std::vector<float> host(size_t(tokens) * 4096u);
            check(hipMemcpy(host.data(), output.pointer, host.size() * 4, hipMemcpyDeviceToHost));
            matched &= report("ck", host, reference, 0, argv[6], ms, ms);
            FreeLibrary(dll);
        }
        const bool use_table = std::string(argv[11]) != "-";
        std::vector<unsigned char> table;
        if (use_table) {
            table = read<unsigned char>(argv[11], qrt_blackwell_attention::exp2_backend::table_bytes);
            if (!qrt_blackwell_attention::exp2_backend::valid_layout(table.data(), table.size()))
                throw std::runtime_error("invalid exponent table layout");
            BCRYPT_ALG_HANDLE algorithm = nullptr; unsigned char digest[32]{};
            if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
                throw std::runtime_error("SHA256 provider unavailable");
            const auto status = BCryptHash(algorithm, nullptr, 0, table.data(), ULONG(table.size()), digest, sizeof(digest));
            BCryptCloseAlgorithmProvider(algorithm, 0);
            if (status < 0 || std::memcmp(digest, qrt_blackwell_attention::exp2_backend::sha256, 32))
                throw std::runtime_error("exponent table fingerprint mismatch");
        }
        Device dt(use_table ? table.size() : 4u), output(size_t(count) * 4096u * 4u);
        Device accumulator(size_t(count) * 4096u * 4u), denominator(size_t(count) * 16u * 4u);
        const char* rcp_path = std::getenv("QRT_CK_FMHA_SM121_RCP_TABLE");
        const bool use_rcp = rcp_path && *rcp_path;
        std::vector<unsigned char> rcp_table;
        if (use_rcp) {
            rcp_table = read<unsigned char>(rcp_path, qrt_sm121_attention_rcp::table_bytes);
            if (!qrt_sm121_attention_rcp::valid_layout(rcp_table.data(), rcp_table.size()))
                throw std::runtime_error("invalid reciprocal table layout");
            BCRYPT_ALG_HANDLE algorithm = nullptr; unsigned char digest[32]{};
            if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
                throw std::runtime_error("SHA256 provider unavailable");
            const auto status = BCryptHash(algorithm, nullptr, 0, rcp_table.data(), ULONG(rcp_table.size()), digest, sizeof(digest));
            BCryptCloseAlgorithmProvider(algorithm, 0);
            if (status < 0 || std::memcmp(digest, qrt_sm121_attention_rcp::sha256, 32))
                throw std::runtime_error("reciprocal table fingerprint mismatch");
        }
        Device dr(use_rcp ? rcp_table.size() : 4u);
        if (use_rcp) check(hipMemcpy(dr.pointer, rcp_table.data(), rcp_table.size(), hipMemcpyHostToDevice));
        if (use_table) check(hipMemcpy(dt.pointer, table.data(), table.size(), hipMemcpyHostToDevice));
        float total = 0, maximum = 0, scores_total = 0, probabilities_total = 0, value_total = 0;
        Event scores_done, probabilities_done;
        const bool needs_transpose = qrt_blackwell_attention::split_transposed_keys(memory_layout);
        Device transposed(needs_transpose ? (k.size() + 256u) * 2u : 4u);
        auto* transposed_data = needs_transpose ? transposed.as<uint16_t>() + 128u : nullptr;
        float preparation_ms = 0;
        if (transposed_data) {
            check(hipMemset(transposed.pointer, 0xa5, (k.size() + 256u) * 2u));
            check(hipEventRecord(begin.value));
            check(hipError_t(qrt_blackwell_attention::transpose_keys(
                dk.as<uint16_t>(), transposed_data, k.size(), tokens, nullptr)));
            preparation_ms = finish(begin, end, 100.0f);
            total = maximum = preparation_ms;
        }
        using PackedRow = qrt_blackwell_attention::IntegerOperandRow;
        using PackedKind = qrt_blackwell_attention::IntegerRowKind;
        const bool prepacked = memory_layout == 9u;
        const size_t packed_key_rows = prepacked ? qrt_blackwell_attention::integer_row_count(PackedKind::Key, tokens, batch) : 0u;
        const size_t packed_value_rows = prepacked ? qrt_blackwell_attention::integer_row_count(PackedKind::Value, tokens, batch) : 0u;
        const size_t packed_query_rows = prepacked ? qrt_blackwell_attention::integer_row_count(PackedKind::Query, tokens, batch) : 0u;
        const size_t packed_probability_rows = prepacked ? qrt_blackwell_attention::integer_row_count(PackedKind::Probability, tokens, batch) : 0u;
        Device packed_key((packed_key_rows + 2u) * sizeof(PackedRow));
        Device packed_value((packed_value_rows + 2u) * sizeof(PackedRow));
        Device packed_query((packed_query_rows + 2u) * sizeof(PackedRow));
        Device packed_probability((packed_probability_rows + 2u) * sizeof(PackedRow));
        qrt_blackwell_attention::PrepackedIntegerWorkspace prepared;
        if (prepacked) {
            prepared = {packed_key.as<PackedRow>() + 1u, packed_value.as<PackedRow>() + 1u,
                packed_query.as<PackedRow>() + 1u, packed_probability.as<PackedRow>() + 1u, tokens, batch};
            check(hipMemset(packed_key.pointer, 0xa5, (packed_key_rows + 2u) * sizeof(PackedRow)));
            check(hipMemset(packed_value.pointer, 0xa5, (packed_value_rows + 2u) * sizeof(PackedRow)));
            check(hipMemset(packed_query.pointer, 0xa5, (packed_query_rows + 2u) * sizeof(PackedRow)));
            check(hipMemset(packed_probability.pointer, 0xa5, (packed_probability_rows + 2u) * sizeof(PackedRow)));
            check(hipEventRecord(begin.value));
            check(hipError_t(qrt_blackwell_attention::prepare_integer_rows<PackedKind::Key>(
                dk.as<uint16_t>(), prepared.key, tokens, 0u, 0u, nullptr)));
            check(hipError_t(qrt_blackwell_attention::prepare_integer_rows<PackedKind::Value>(
                dv.as<uint16_t>(), prepared.value, tokens, 0u, 0u, nullptr)));
            preparation_ms = finish(begin, end, 100.0f);
            total = maximum = preparation_ms;
        }
        using CoreRow = qrt_sm121_integer_core::Row;
        const bool prepacked_core = memory_layout == 21u;
        const size_t core_key_rows = prepacked_core ? qrt_blackwell_attention::integer_row_count(PackedKind::Key, tokens, batch) : 0u;
        const size_t core_query_rows = prepacked_core ? qrt_blackwell_attention::integer_row_count(PackedKind::Query, tokens, batch) : 0u;
        Device core_key((core_key_rows + 2u) * sizeof(CoreRow)), core_query((core_query_rows + 2u) * sizeof(CoreRow));
        qrt_blackwell_attention::CoreIntegerWorkspace core_prepared;
        if (prepacked_core) {
            core_prepared = {core_key.as<CoreRow>() + 1u, core_query.as<CoreRow>() + 1u, tokens, batch};
            check(hipMemset(core_key.pointer, 0xa5, (core_key_rows + 2u) * sizeof(CoreRow)));
            check(hipMemset(core_query.pointer, 0xa5, (core_query_rows + 2u) * sizeof(CoreRow)));
            check(hipEventRecord(begin.value));
            check(hipError_t(qrt_blackwell_attention::prepare_integer_core_rows<PackedKind::Key>(
                dk.as<uint16_t>(), core_prepared.key, tokens, 0u, 0u, nullptr)));
            const float ms = finish(begin, end, 100.0f);
            preparation_ms += ms; total += ms; maximum = std::max(maximum, ms);
        }
        const bool prepare_values = memory_layout >= 17u && memory_layout <= 21u;
        Device prepared_values(prepare_values ? (v.size() + 128u) * sizeof(uint32_t) : 4u);
        auto* prepared_value_data = prepare_values ? prepared_values.as<uint32_t>() + 64u : nullptr;
        if (prepare_values) {
            check(hipMemset(prepared_values.pointer, 0xa5, (v.size() + 128u) * sizeof(uint32_t)));
            check(hipEventRecord(begin.value));
            check(hipError_t(qrt_blackwell_attention::prepare_value_encoding(
                dv.as<uint16_t>(), prepared_value_data, v.size(), tokens, nullptr)));
            const float ms = finish(begin, end, 100.0f);
            preparation_ms += ms; total += ms; maximum = std::max(maximum, ms);
        }
        const size_t score_elements = memory_layout >= 2u
            ? qrt_blackwell_attention::split_scratch_elements(batch, tokens, memory_layout) : 1u;
        Device scores((score_elements + 128u) * sizeof(float));
        auto* score_data = scores.as<float>() + 64u;
        check(hipMemset(scores.pointer, 0xa5, (score_elements + 128u) * sizeof(float)));
        double completed_host_ms = 0.0;
        uint64_t compacted_pv_cells = 0u;
        for (unsigned offset = 0; offset < count; offset += batch) {
            if (std::chrono::duration<double>(Clock::now() - begun).count() > 150.0)
                throw std::runtime_error("replay aggregate deadline exceeded");
            const auto host_begin = Clock::now();
            check(hipEventRecord(begin.value));
            check(hipError_t(qrt_blackwell_attention::launch_queries(dq.as<uint16_t>(), dk.as<uint16_t>(),
                dv.as<uint16_t>(), output.as<float>(), nullptr, start + offset,
                std::min(batch, count - offset), offset, use_table ? dt.as<unsigned char>() : nullptr,
                accumulator.as<float>(), denominator.as<float>(), true,
                use_rcp ? dr.as<unsigned char>() : nullptr, memory_layout,
                score_data, score_elements,
                memory_layout >= 2u ? scores_done.value : nullptr,
                (qrt_blackwell_attention::split_separate_probability(memory_layout)) ? probabilities_done.value : nullptr,
                transposed_data, tokens, native_products, prepacked ? &prepared : nullptr,
                prepared_value_data, prepare_values ? tokens : 0u,
                prepacked_core ? &core_prepared : nullptr)));
            const float ms = finish(begin, end); total += ms; maximum = std::max(maximum, ms);
            completed_host_ms += std::chrono::duration<double, std::milli>(Clock::now() - host_begin).count();
            if (memory_layout == 22u || memory_layout == 24u) {
                // Component diagnostics only, after completion and outside the
                // timed interval. The product provider never reads this count.
                const unsigned queries = std::min(batch, count - offset);
                const size_t used = qrt_blackwell_attention::split_scratch_elements(
                    queries, start + offset + queries, memory_layout);
                unsigned selected = 0u;
                check(hipMemcpy(&selected, score_data + used - 1u, sizeof(selected), hipMemcpyDeviceToHost));
                if (selected > queries * 16u * 256u) throw std::runtime_error("PV candidate count overflow");
                compacted_pv_cells += selected;
            }
            if (memory_layout >= 2u) {
                float stage_ms = 0;
                check(hipEventElapsedTime(&stage_ms, begin.value, scores_done.value));
                scores_total += stage_ms;
                if (qrt_blackwell_attention::split_separate_probability(memory_layout)) {
                    check(hipEventElapsedTime(&stage_ms, scores_done.value, probabilities_done.value));
                    probabilities_total += stage_ms;
                }
                check(hipEventElapsedTime(&stage_ms,
                    (qrt_blackwell_attention::split_separate_probability(memory_layout)) ? probabilities_done.value : scores_done.value, end.value));
                value_total += stage_ms;
            }
        }
        uint32_t score_guards[128];
        check(hipMemcpy(score_guards, scores.pointer, 64u * sizeof(uint32_t), hipMemcpyDeviceToHost));
        check(hipMemcpy(score_guards + 64u, score_data + score_elements, 64u * sizeof(uint32_t), hipMemcpyDeviceToHost));
        for (uint32_t guard : score_guards)
            if (guard != 0xa5a5a5a5u) throw std::runtime_error("score/probability workspace redzone changed");
        if (prepare_values) {
            std::vector<uint32_t> checked(v.size() + 128u);
            check(hipMemcpy(checked.data(), prepared_values.pointer, checked.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
            for (size_t i = 0u; i < 64u; ++i)
                if (checked[i] != 0xa5a5a5a5u || checked[64u + v.size() + i] != 0xa5a5a5a5u)
                    throw std::runtime_error("prepared value redzone changed");
            for (size_t i = 0u; i < v.size(); ++i)
                if (checked[64u+i] != qrt_sm121_prepared_bf16::encode_wide(v[i]))
                    throw std::runtime_error("prepared value cell changed");
            std::vector<uint16_t> original(v.size());
            check(hipMemcpy(original.data(), dv.pointer, original.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
            if (original != v) throw std::runtime_error("original value input changed");
            std::cerr << "PREPARED_VALUE cells_checked=" << v.size() << " redzones=pass input_immutable=pass"
                << " workspace_bytes=" << checked.size() * sizeof(uint32_t)
                << " preparation_included_in_total=1\n";
        }
        if (prepacked_core) {
            auto validate_rows = [&](Device& storage, size_t capacity, PackedKind kind,
                const std::vector<uint16_t>& input, unsigned stride, unsigned first, unsigned queries) {
                std::vector<CoreRow> rows(capacity + 2u);
                check(hipMemcpy(rows.data(), storage.pointer, rows.size() * sizeof(CoreRow), hipMemcpyDeviceToHost));
                const auto* raw = reinterpret_cast<const unsigned char*>(rows.data());
                for (size_t i = 0u; i < sizeof(CoreRow); ++i)
                    if (raw[i] != 0xa5u || raw[(capacity + 1u) * sizeof(CoreRow) + i] != 0xa5u)
                        throw std::runtime_error("integer core workspace redzone changed");
                const size_t active = qrt_blackwell_attention::integer_row_count(kind, stride, queries);
                if (active > capacity) throw std::runtime_error("integer core capacity exceeded");
                for (size_t row = 0u; row < active; ++row) {
                    CoreRow expected{};
                    for (unsigned column = 0u; column < 16u; ++column) {
                        const size_t index = qrt_blackwell_attention::integer_row_input_index(kind, row, column, stride, first, queries);
                        if (index != static_cast<size_t>(-1) && index >= input.size())
                            throw std::runtime_error("integer core input outside captured extent");
                        expected.original[column] = index == static_cast<size_t>(-1) ? 0u : input[index];
                    }
                    qrt_sm121_integer_core::prepare(expected);
                    if (std::memcmp(&expected, &rows[row + 1u], sizeof(expected)))
                        throw std::runtime_error("integer core operand or encoding mismatch");
                }
                return active;
            };
            const unsigned last_count = (count - 1u) % batch + 1u;
            const unsigned last_start = start + count - last_count, stride = start + count;
            const size_t checked = validate_rows(core_key, core_key_rows, PackedKind::Key, k, tokens, 0u, 0u) +
                validate_rows(core_query, core_query_rows, PackedKind::Query, q, stride, last_start, last_count);
            for (const auto& input : {std::make_pair(&dq, &q), std::make_pair(&dk, &k)}) {
                std::vector<uint16_t> after(input.second->size());
                check(hipMemcpy(after.data(), input.first->pointer, after.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
                if (after != *input.second) throw std::runtime_error("immutable Q/K core input changed");
            }
            std::cerr << "PREPACKED_INTEGER_CORE rows_checked=" << checked << " redzones=pass inputs_immutable=pass"
                << " workspace_bytes=" << (core_key_rows + core_query_rows + 4u) * sizeof(CoreRow)
                << " preparation_included_in_total=1\n";
        }
        if (prepacked) {
            auto validate_rows = [&](Device& storage, size_t capacity, PackedKind kind,
                const std::vector<uint16_t>& input, unsigned stride, unsigned first, unsigned queries) {
                std::vector<PackedRow> rows(capacity + 2u);
                check(hipMemcpy(rows.data(), storage.pointer, rows.size() * sizeof(PackedRow), hipMemcpyDeviceToHost));
                const auto* raw = reinterpret_cast<const unsigned char*>(rows.data());
                for (size_t i = 0u; i < sizeof(PackedRow); ++i)
                    if (raw[i] != 0xa5u || raw[(capacity + 1u) * sizeof(PackedRow) + i] != 0xa5u)
                        throw std::runtime_error("prepacked integer redzone changed");
                const size_t active = qrt_blackwell_attention::integer_row_count(kind, stride, queries);
                if (active > capacity) throw std::runtime_error("prepacked integer capacity exceeded");
                for (size_t row = 0u; row < active; ++row) {
                    PackedRow expected{};
                    for (unsigned column = 0u; column < 16u; ++column) {
                        const size_t index = qrt_blackwell_attention::integer_row_input_index(kind, row, column, stride, first, queries);
                        if (index != static_cast<size_t>(-1) && index >= input.size())
                            throw std::runtime_error("prepacked integer input outside captured extent");
                        expected.original[column] = index == static_cast<size_t>(-1) ? 0u : input[index];
                    }
                    expected.minimum = qrt_sm121_integer_parts::row_unit_range(expected.original, &expected.maximum);
                    for (unsigned word = 0u; word < 4u; ++word) {
                        uint32_t high = 0u, low = 0u, trailing = 0u;
                        for (unsigned byte = 0u; byte < 4u; ++byte) {
                            const auto encoded = qrt_sm121_integer_parts::encode(expected.original[word * 4u + byte], expected.minimum);
                            high |= uint32_t(encoded >> 8u) << (byte * 8u);
                            low |= uint32_t(encoded & 255u) << (byte * 8u);
                            trailing |= qrt_sm121_integer_parts::trailing_bits(encoded) << (byte * 8u);
                        }
                        expected.high[word] = int(high); expected.low[word] = int(low); expected.trailing[word] = trailing;
                    }
                    if (std::memcmp(&expected, &rows[row + 1u], sizeof(expected)))
                        throw std::runtime_error("prepacked integer operand or encoding mismatch");
                }
                return active;
            };
            const unsigned last_count = (count - 1u) % batch + 1u;
            const unsigned last_start = start + count - last_count, stride = start + count;
            const size_t probability_cells = size_t(last_count) * 16u * stride;
            std::vector<uint16_t> last_probability(probability_cells);
            check(hipMemcpy(last_probability.data(), reinterpret_cast<const uint16_t*>(score_data + probability_cells),
                probability_cells * sizeof(uint16_t), hipMemcpyDeviceToHost));
            const size_t checked = validate_rows(packed_key, packed_key_rows, PackedKind::Key, k, tokens, 0u, 0u) +
                validate_rows(packed_value, packed_value_rows, PackedKind::Value, v, tokens, 0u, 0u) +
                validate_rows(packed_query, packed_query_rows, PackedKind::Query, q, stride, last_start, last_count) +
                validate_rows(packed_probability, packed_probability_rows, PackedKind::Probability, last_probability, stride, last_start, last_count);
            for (const auto& input : {std::make_pair(&dq, &q), std::make_pair(&dk, &k), std::make_pair(&dv, &v)}) {
                std::vector<uint16_t> after(input.second->size());
                check(hipMemcpy(after.data(), input.first->pointer, after.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
                if (after != *input.second) throw std::runtime_error("immutable attention operand changed");
            }
            std::cerr << "PREPACKED_INTEGER rows_checked=" << checked << " redzones=pass inputs_immutable=pass"
                << " workspace_bytes=" << (packed_key_rows + packed_value_rows + packed_query_rows + packed_probability_rows + 8u) * sizeof(PackedRow)
                << " kv_preparation_ms=" << preparation_ms << " included_in_total=1\n";
        }
        std::vector<float> host(size_t(count) * 4096u);
        check(hipMemcpy(host.data(), output.pointer, host.size() * 4, hipMemcpyDeviceToHost));
        if (transposed_data) {
            // Compare every transformed input cell and both 128-element redzones
            // after all score/PV work. This observer is outside GPU timing.
            std::vector<uint16_t> checked(k.size() + 256u);
            check(hipMemcpy(checked.data(), transposed.pointer, checked.size() * 2u, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < 128u; ++i) {
                if (checked[i] != 0xa5a5u || checked[128u + k.size() + i] != 0xa5a5u)
                    throw std::runtime_error("key transpose redzone changed");
            }
            for (size_t row = 0; row < tokens; ++row) {
                for (size_t column = 0; column < 512u; ++column) {
                    if (checked[128u + column * tokens + row] != k[row * 512u + column])
                        throw std::runtime_error("key transpose input mismatch");
                }
            }
        }
        const char* route = use_table
            ? (use_rcp ? "blackwell-sm121-exp-rcp" : "blackwell-sm121-exp")
            : (use_rcp ? "blackwell-amd-exp-rcp" : "blackwell-amd-exp");
        matched &= report(route, host, reference, start, argv[6], total, maximum, memory_layout,
                          scores_total, probabilities_total, value_total, preparation_ms, native_products,
                          completed_host_ms, compacted_pv_cells);
        check(hipMemcpy(host.data(), accumulator.pointer, host.size() * 4, hipMemcpyDeviceToHost));
        write(std::string(argv[6]) + "-accumulator-f32.bin", host);
        host.resize(size_t(count) * 16u);
        check(hipMemcpy(host.data(), denominator.pointer, host.size() * 4, hipMemcpyDeviceToHost));
        write(std::string(argv[6]) + "-denominator-f32.bin", host);
        return matched ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "attention_capture_replay_error=" << error.what() << std::endl;
        return 1;
    }
}
