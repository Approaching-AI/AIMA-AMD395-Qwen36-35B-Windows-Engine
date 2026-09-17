#pragma once
#include "../../native/providers/gdn/conv_consumer_interval.h"
#include <thread>

namespace projection_safety_test {
namespace conv_audit {
namespace c = qrt_conv_consumer;
constexpr unsigned rows = 8192u, tokens = 8192u, source_tokens = 7169u, k = 2048u;
constexpr unsigned constant_bit = 1u << 16u, selected_bit = 1u << 17u;
constexpr size_t cells = size_t(rows) * tokens;

__global__ void certificates(const float* native, const float* input_l2,
    const float* weight_l2, const uint16_t* weights, const unsigned char* table,
    uint32_t* result) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= cells) return;
    const unsigned token = unsigned(i / rows), feature = unsigned(i % rows);
    c::c::Range ranges[4]; uint16_t taps[4]; unsigned present = 0u;
    bool own_selected = false;
    for (unsigned tap = 0u; tap < 4u; ++tap) {
        taps[tap] = weights[feature * 4u + tap];
        if (token + tap < 3u) continue;
        const unsigned source = token + tap - 3u;
        const size_t index = size_t(source) * rows + feature;
        const bool selected = selected_bf16_projection_hawkeye_candidate(
            native[index],index,rows,512u,0u,1000u,nullptr,input_l2,weight_l2);
        const float error = (input_l2[source] * weight_l2[feature]) * (1000.0f * 1.0e-9f);
        ranges[tap] = c::endpoint(native[index],error,selected);
        present |= 1u << tap;
        if (tap == 3u) own_selected = selected;
    }
    const auto cert = c::certify(ranges,taps,present,table);
    result[i] = (cert.constant ? constant_bit | cert.output : 0u) |
        (own_selected ? selected_bit : 0u);
}

struct Stream {
    hipStream_t value = nullptr;
    Stream() { hip_ok(hipStreamCreateWithFlags(&value,hipStreamNonBlocking),"conv_audit_stream"); }
    ~Stream() { if (value) (void)hipStreamDestroy(value); }
    void finish() {
        hipEvent_t event = nullptr;
        hip_ok(hipEventCreateWithFlags(&event,hipEventDisableTiming),"conv_audit_event");
        hip_ok(hipEventRecord(event,value),"conv_audit_record");
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const auto status = hipEventQuery(event);
            if (status == hipSuccess) break;
            require(status == hipErrorNotReady && std::chrono::steady_clock::now() < end,"conv audit completion deadline");
            std::this_thread::yield();
        }
        hip_ok(hipEventDestroy(event),"conv_audit_event_destroy");
    }
};
template<class T> void guards(const std::vector<T>& data, T guard) {
    for (size_t i = 0u; i < kGuard; ++i)
        require(data[i] == guard && data[data.size()-kGuard+i] == guard,"conv audit redzone");
}
} // namespace conv_audit

void run_conv_consumer_audit(const char* input_path, const char* projection_weight_path,
    const char* projection_reference_path, const char* conv_weight_path,
    const char* conv_reference_dir, const char* table_path) {
    using namespace conv_audit;
    auto input = read_replay_tensor<uint16_t>(input_path,size_t(source_tokens)*k,kBf16Guard);
    const auto weights = read_replay_tensor<uint16_t>(projection_weight_path,size_t(rows)*k,kBf16Guard);
    const auto taps = read_replay_tensor<uint16_t>(conv_weight_path,size_t(rows)*4u,kBf16Guard);
    input.resize(size_t(tokens)*k + 2u*kGuard,kBf16Guard);
    std::copy_n(input.data()+kGuard,size_t(tokens-source_tokens)*k,input.data()+kGuard+size_t(source_tokens)*k);
    std::vector<float> projection(cells+2u*kGuard,kF32Guard), convolution(projection);
    std::vector<float> input_l2(tokens+2u*kGuard,kF32Guard), weight_l2(rows+2u*kGuard,kF32Guard);
    constexpr uint32_t certificate_guard = 0xa5a5a5a5u;
    std::vector<uint32_t> certificate(cells+2u*kGuard,certificate_guard);
    DeviceBuffer<uint16_t> di(input), dw(weights), dt(taps);
    DeviceBuffer<float> dp(projection), dc(convolution), dix(input_l2), dwx(weight_l2);
    DeviceBuffer<uint32_t> dcert(certificate);
    const unsigned char* table = nullptr;
    hip_ok(qrt_sm121_silu_runtime::prepare(table_path,&table),"conv_audit_table");
    Stream stream;
    hip_ok(launch_selected_bf16_projection_wmma_checked(dw.data(),di.data(),dp.data(),rows,tokens,0u,0u,stream.value),"conv_audit_producer");
    hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(tokens),dim3(256u),0u,stream.value,di.data(),dix.data(),tokens,k);
    hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(rows),dim3(256u),0u,stream.value,dw.data(),dwx.data(),rows,k);
    hip_ok(hipGetLastError(),"conv_audit_bounds"); stream.finish();
    dp.read(projection); dix.read(input_l2); dwx.read(weight_l2);
    const auto native = projection;
    const auto start = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(certificates,dim3((cells+255u)/256u),dim3(256u),0u,stream.value,
        dp.data(),dix.data(),dwx.data(),dt.data(),table,dcert.data());
    hip_ok(hipGetLastError(),"conv_audit_certificates"); stream.finish();
    const double certificate_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    // Every original candidate is still replayed. The certificate cannot
    // influence either original projection correction or convolution.
    hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw.data(),di.data(),nullptr,
        dix.data(),dwx.data(),dp.data(),rows,tokens,k,512u,0u,1000u,4096u,stream.value),"conv_audit_original_replay");
    hipLaunchKernelGGL(selected_conv_qkv_window_kernel,dim3(rows/256u,tokens),dim3(256u),0u,stream.value,
        dp.data(),dt.data(),nullptr,dc.data(),tokens,3u,nullptr,nullptr,0u,table);
    hip_ok(hipGetLastError(),"conv_audit_original_convolution"); stream.finish();
    dp.read(projection); dc.read(convolution); dcert.read(certificate);
    guards(projection,kF32Guard); guards(convolution,kF32Guard); guards(certificate,certificate_guard);
    guards(input_l2,kF32Guard); guards(weight_l2,kF32Guard);
    // References are loaded only after all GPU calculations have completed.
    auto reference = read_replay_tensor<uint16_t>(projection_reference_path,size_t(source_tokens)*rows,kBf16Guard);
    reference.resize(cells+2u*kGuard,kBf16Guard);
    std::copy_n(reference.data()+kGuard,size_t(tokens-source_tokens)*rows,reference.data()+kGuard+size_t(source_tokens)*rows);
    const std::string base = std::string(conv_reference_dir)+"/full-";
    const auto q = read_replay_tensor<uint16_t>((base+"q-bf16.bin").c_str(),size_t(source_tokens)*2048u,kBf16Guard);
    const auto key = read_replay_tensor<uint16_t>((base+"k-bf16.bin").c_str(),size_t(source_tokens)*2048u,kBf16Guard);
    const auto v = read_replay_tensor<uint16_t>((base+"v-bf16.bin").c_str(),size_t(source_tokens)*4096u,kBf16Guard);
    uint64_t selected = 0u, valid_ranges = 0u, range_failures = 0u, constants = 0u;
    uint64_t omitted = 0u, halo_selected = 0u, changed = 0u, omitted_changed = 0u;
    uint64_t selector_failures = 0u, certificate_failures = 0u, unselected_changes = 0u;
    uint64_t projection_mismatches = 0u, conv_mismatches[3]{}, wide_ranges = 0u, wide_omittable = 0u, tiny_selected = 0u;
    for (size_t i = 0u; i < cells; ++i) {
        const unsigned token = unsigned(i/rows), feature = unsigned(i%rows);
        const float raw = native[kGuard+i], corrected = projection[kGuard+i];
        require(c::c::finite(raw) && c::c::finite(corrected) && c::c::finite(convolution[kGuard+i]),"nonfinite captured result");
        const uint32_t bits = c::c::bits(raw), low = bits & 65535u;
        const unsigned distance = low >= 32768u ? low-32768u : 32768u-low;
        const float error = (input_l2[kGuard+token]*weight_l2[kGuard+feature])*(1000.0f*1.0e-9f);
        const bool candidate = distance <= 512u || ((bits>>23u)&255u) < 32u || qrt_bf16_midpoint::within_error(raw,error);
        const uint32_t cert = certificate[kGuard+i];
        selector_failures += candidate != bool(cert & selected_bit);
        require(!(cert & ~uint32_t(0x3ffffu)),"unwritten or invalid certificate word");
        const bool differs = bf16(raw) != bf16(corrected);
        changed += differs;
        unselected_changes += !candidate && differs;
        projection_mismatches += bf16(corrected) != reference[kGuard+i];
        if (cert & constant_bit) {
            ++constants;
            certificate_failures += uint16_t(cert) != bf16(convolution[kGuard+i]);
        }
        if (candidate) {
            ++selected;
            const auto range = c::endpoint(raw,error,true);
            const bool wide = range.valid && unsigned(range.high-range.low)>8u;
            wide_ranges += wide; tiny_selected += ((bits>>23u)&255u)<32u;
            valid_ranges += range.valid;
            range_failures += range.valid && !c::c::contains(range,corrected);
            bool following[4]{};
            for (unsigned j = 0u; j < 4u && token+j < tokens; ++j)
                following[j] = (certificate[kGuard+i+size_t(j)*rows] & constant_bit) != 0u;
            const bool skip = c::can_omit(token,tokens,following);
            require(!skip || range.valid,"omitted candidate lacks valid range");
            omitted += skip; omitted_changed += skip && differs;
            wide_omittable += skip && wide;
            halo_selected += tokens-token <= 3u;
        }
        if (token < source_tokens) {
            const unsigned surface = feature < 2048u ? 0u : feature < 4096u ? 1u : 2u;
            const auto expected = surface == 0u ? q[kGuard+size_t(token)*2048u+feature]
                : surface == 1u ? key[kGuard+size_t(token)*2048u+feature-2048u]
                : v[kGuard+size_t(token)*4096u+feature-4096u];
            conv_mismatches[surface] += bf16(convolution[kGuard+i]) != expected;
        }
    }
    auto after_input = input, after_weight = weights, after_taps = taps;
    di.read(after_input); dw.read(after_weight); dt.read(after_taps);
    require(after_input == input && after_weight == weights && after_taps == taps,"conv audit immutable input changed");
    std::cout << "{\"type\":\"conv_consumer_audit\",\"tokens\":" << tokens << ",\"source_tokens\":" << source_tokens
        << ",\"elements\":" << cells << ",\"selected\":" << selected << ",\"valid_ranges\":" << valid_ranges
        << ",\"wide_intervals\":true,\"wide_ranges\":" << wide_ranges << ",\"wide_omittable\":" << wide_omittable << ",\"tiny_selected\":" << tiny_selected
        << ",\"omittable\":" << omitted << ",\"constant_outputs\":" << constants << ",\"halo_selected_protected\":" << halo_selected
        << ",\"projection_bf16_changes\":" << changed << ",\"omittable_bf16_changes\":" << omitted_changed
        << ",\"selector_failures\":" << selector_failures << ",\"range_failures\":" << range_failures
        << ",\"certificate_failures\":" << certificate_failures << ",\"unselected_endpoint_changes\":" << unselected_changes
        << ",\"gb10_projection_mismatches\":" << projection_mismatches
        << ",\"gb10_convolution_mismatches\":[" << conv_mismatches[0] << ',' << conv_mismatches[1] << ',' << conv_mismatches[2]
        << "],\"certificate_completed_wall_ms\":" << certificate_ms
        << ",\"all_original_candidates_replayed\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}" << std::endl;
    require(selected && constants && !selector_failures && !range_failures && !certificate_failures && !unselected_changes &&
        !projection_mismatches && !conv_mismatches[0] && !conv_mismatches[1] && !conv_mismatches[2],"conv consumer audit mismatch");
}
} // namespace projection_safety_test
