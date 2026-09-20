// Original q2 recurrence in both resident layouts, including rejected row one.
#ifdef QRT_Q2_CPU_PROBE
#include "native/providers/gdn/sm121_q2_linear_layout.h"
#else
#include "native/providers/gdn/sm121_q2_linear.h"
#endif
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qrt_sm121_q2;
const char* phase = "read_inputs";
unsigned active_configuration = ~0u;
template<class T> std::vector<T> read(const char* path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(count * sizeof(T))) throw std::runtime_error(path);
    std::vector<T> result(count);
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (!file) throw std::runtime_error(path);
    return result;
}

template<class T> struct GuardedHost {
    std::vector<T> storage;
    size_t bytes;
    explicit GuardedHost(size_t count):storage((count + 512u) / sizeof(T)),bytes(count) {
        if (count % 4u) throw std::runtime_error("guard alignment");
        std::memset(storage.data(),0xa5,count+512u);
    }
    void* data() { return reinterpret_cast<unsigned char*>(storage.data()) + 256u; }
    size_t guard_errors() const {
        size_t errors = 0;
        const auto* raw = reinterpret_cast<const unsigned char*>(storage.data());
        for (size_t i = 0; i < 256u; ++i) {
            errors += raw[i] != 0xa5u;
            errors += raw[bytes+256u+i] != 0xa5u;
        }
        return errors;
    }
};

#ifndef QRT_Q2_CPU_PROBE
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void kernel_resources(const char* name, const void* kernel) {
    hipFuncAttributes attributes{};
    check(hipFuncGetAttributes(&attributes, kernel));
    std::cerr << "{\"kind\":\"q2_kernel_resources\",\"kernel\":\"" << name
        << "\",\"maximum_threads\":" << attributes.maxThreadsPerBlock
        << ",\"registers\":" << attributes.numRegs
        << ",\"local_bytes\":" << attributes.localSizeBytes
        << ",\"shared_bytes\":" << attributes.sharedSizeBytes << "}\n";
}
struct Device {
    std::vector<void*> pointers;
    ~Device() {
        if (hipDeviceSynchronize() != hipSuccess) return;
        for (void* p : pointers) (void)hipFree(p);
    }
    template<class T> T* upload(const std::vector<T>& values) {
        void* p = nullptr; check(hipMalloc(&p, values.size() * sizeof(T)));
        pointers.push_back(p); check(hipMemcpy(p, values.data(), values.size()*sizeof(T), hipMemcpyHostToDevice));
        return static_cast<T*>(p);
    }
    template<class T> size_t differences(const T* device, const std::vector<T>& original) const {
        std::vector<T> current(original.size());
        check(hipMemcpy(current.data(), device, current.size()*sizeof(T), hipMemcpyDeviceToHost));
        const auto* a = reinterpret_cast<const unsigned char*>(current.data());
        const auto* b = reinterpret_cast<const unsigned char*>(original.data());
        size_t count = 0;
        for (size_t i = 0; i < current.size()*sizeof(T); ++i) count += a[i] != b[i];
        return count;
    }
};
#endif

int main(int argc, char** argv) try {
    if (argc != 11 && argc != 18) throw std::runtime_error(
        "conv a b before after core g beta exp2 rsqrt [qkv conv-before conv-after conv-weights silu position history-offset]");
    const bool linear = argc == 18;
    const auto conv = read<uint16_t>(argv[1], 2u * 8192u);
    const auto a = read<uint16_t>(argv[2], 2u * 32u), b = read<uint16_t>(argv[3], 2u * 32u);
    const auto initial = read<float>(argv[4], state_elements);
    const auto expected = read<float>(argv[5], 2u * state_elements);
    const auto expected_core = read<uint16_t>(argv[6], 2u * core_elements);
    const auto g = read<float>(argv[7], 32u * 65536u), beta = read<float>(argv[8], 65536u);
    const auto exp2 = read<unsigned char>(argv[9], qrt_sm121_exp2::table_bytes);
    const auto rsqrt = read<unsigned char>(argv[10], qrt_sm121_rsqrt::table_bytes);
    if (!qrt_sm121_exp2::valid_layout(exp2.data(), exp2.size()) ||
        !qrt_sm121_rsqrt::valid_layout(rsqrt.data(), rsqrt.size())) throw std::runtime_error("table layout");
    const auto qkv = linear ? read<uint16_t>(argv[11],2u*8192u) : std::vector<uint16_t>{};
    const auto conv_before = linear ? read<uint16_t>(argv[12],ring_elements) : std::vector<uint16_t>{};
    const auto conv_after = linear ? read<uint16_t>(argv[13],ring_elements) : std::vector<uint16_t>{};
    const auto conv_weights = linear ? read<uint16_t>(argv[14],ring_elements) : std::vector<uint16_t>{};
    const auto silu = linear ? read<unsigned char>(argv[15],qrt_sm121_silu::table_bytes) : std::vector<unsigned char>{};
    const size_t position = linear ? std::stoull(argv[16]) : 0u;
    const size_t history_offset = linear ? std::stoull(argv[17]) : 0u;
    if(linear && (position<3u || position>263678u || history_offset>1u ||
        !qrt_sm121_silu::valid_layout(silu.data(),silu.size())))throw std::runtime_error("original convolution layout");
    size_t state_bad[2] = {}, core_bad[2] = {}, selection_bad = 0, input_bad = 0, guard_bad = 0;
    size_t conv_bad = 0, ring_bad = 0;
    unsigned rejected = 0;
    std::cout << "{\"kind\":\"original_q2_recurrence\",\"layouts\":[\"value_key\",\"key_value\"],\"native_execution\":";
#ifdef QRT_Q2_CPU_PROBE
    std::cout << "false";
    const RecurrentTables tables{g.data(), beta.data(), exp2.data(), rsqrt.data()};
    const auto* convolution = linear ? nullptr : conv.data(); const auto* projection_a = a.data(); const auto* projection_b = b.data();
    const auto* actual_qkv = qkv.data(); const auto* actual_conv_weights = conv_weights.data(); const auto* actual_silu = silu.data();
#else
    std::cout << "true";
    phase = "kernel_resources";
    kernel_resources("recurrence", reinterpret_cast<const void*>(recurrent_detail::kernel));
    if (linear) {
        kernel_resources("convolution_f32", reinterpret_cast<const void*>(linear_detail::convolution<float>));
        kernel_resources("convolution_bf16", reinterpret_cast<const void*>(linear_detail::convolution<uint16_t>));
    }
    phase = "upload_inputs";
    Device device;
    const RecurrentTables tables{device.upload(g), device.upload(beta), device.upload(exp2), device.upload(rsqrt)};
    const auto* convolution = linear ? nullptr : device.upload(conv); const auto* projection_a = device.upload(a); const auto* projection_b = device.upload(b);
    const auto* actual_qkv = linear ? device.upload(qkv) : nullptr;
    const auto* actual_conv_weights = linear ? device.upload(conv_weights) : nullptr;
    const auto* actual_silu = linear ? device.upload(silu) : nullptr;
#endif
    const unsigned configurations = linear ? 4u : 2u;
    for (unsigned configuration=0;configuration<configurations;++configuration) {
        active_configuration = configuration;
        phase = "prepare_configuration";
        const bool key_major=(configuration&1u)!=0u, bf16_ring=configuration>=2u;
        std::vector<float> before(state_elements);
        for (unsigned head = 0; head < 32u; ++head)
            for (unsigned v = 0; v < 128u; ++v)
                for (unsigned k = 0; k < 128u; ++k)
                    before[state_offset(head,v,key_major)+k*(key_major?128u:1u)] = initial[(head*128u+v)*128u+k];
        const auto untouched = before;
        GuardedHost<float> staged(staged_state_bytes);
        GuardedHost<uint16_t> core(staged_core_bytes);
        GuardedHost<uint16_t> connected_conv(2u*8192u*sizeof(uint16_t));
        GuardedHost<float> float_rings(2u*ring_elements*sizeof(float));
        GuardedHost<uint16_t> bf16_rings(2u*ring_elements*sizeof(uint16_t));
        std::vector<uint16_t> initial_bf16_ring(ring_elements,0x7fc0u);
        std::vector<float> initial_float_ring(ring_elements,qrt_sm121_q1::widen(0x7fc0u));
        if(linear)for(unsigned feature=0;feature<8192u;++feature)for(unsigned row=0;row<3u;++row) {
            const size_t at=((position-3u+row)%4u)*8192u+feature;
            const uint16_t value=conv_before[feature*4u+history_offset+row];
            initial_bf16_ring[at]=value;initial_float_ring[at]=qrt_sm121_q1::widen(value);
        }
        RecurrentViews view;
        view.convolution = convolution; view.a = projection_a; view.b = projection_b; view.key_major = key_major;
#ifdef QRT_Q2_CPU_PROBE
        view.initial_state = before.data();
        view.staged_states = static_cast<float*>(staged.data());
        view.staged_core = static_cast<uint16_t*>(core.data());
        const auto* float_ring_input=initial_float_ring.data();const auto* bf16_ring_input=initial_bf16_ring.data();
        auto* float_ring_output=static_cast<float*>(float_rings.data());auto* bf16_ring_output=static_cast<uint16_t*>(bf16_rings.data());
        auto* conv_output=static_cast<uint16_t*>(connected_conv.data());
#else
        view.initial_state = device.upload(before);
        auto* state_allocation = device.upload(staged.storage);
        auto* core_allocation = device.upload(core.storage);
        view.staged_states = state_allocation+64u;
        view.staged_core = core_allocation+128u;
        const auto* float_ring_input=linear && !bf16_ring ? device.upload(initial_float_ring) : nullptr;
        const auto* bf16_ring_input=linear && bf16_ring ? device.upload(initial_bf16_ring) : nullptr;
        auto* float_ring_allocation=linear && !bf16_ring ? device.upload(float_rings.storage) : nullptr;
        auto* bf16_ring_allocation=linear && bf16_ring ? device.upload(bf16_rings.storage) : nullptr;
        auto* conv_allocation=linear ? device.upload(connected_conv.storage) : nullptr;
        auto* float_ring_output=float_ring_allocation ? float_ring_allocation+64u : nullptr;
        auto* bf16_ring_output=bf16_ring_allocation ? bf16_ring_allocation+128u : nullptr;
        auto* conv_output=conv_allocation ? conv_allocation+128u : nullptr;
#endif
        if(linear)view.convolution=conv_output;
        if (!valid_recurrent_views(view,tables)) throw std::runtime_error("disjoint staging contract");
        // A late input element or another output may not alias either staged
        // result; these requests must not write anything or enqueue a kernel.
        auto bad = view; bad.staged_states = const_cast<float*>(view.initial_state)+1u;
        auto output_overlap = view; output_overlap.staged_core = reinterpret_cast<uint16_t*>(view.staged_states)+128u;
        auto late_input = view; late_input.convolution = reinterpret_cast<uint16_t*>(view.staged_states)-16u;
        for (const auto& invalid : {bad, output_overlap, late_input}) {
#ifdef QRT_Q2_CPU_PROBE
            if (valid_recurrent_views(invalid,tables)) throw std::runtime_error("alias accepted");
#else
            if (launch_recurrent(invalid,tables) != hipErrorInvalidValue) throw std::runtime_error("alias accepted");
#endif
            ++rejected;
        }
        if(linear) {
            phase = "launch_connected_linear";
            const auto submit=[&](const auto* ring,auto* staged_ring) {
                using Element=std::remove_const_t<std::remove_pointer_t<decltype(ring)>>;
                ConvolutionViews<Element> cv{actual_qkv,ring,actual_conv_weights,actual_silu,staged_ring,conv_output,position};
                if(!valid_linear_views(cv,view,tables))throw std::runtime_error("connected linear staging contract");
#ifdef QRT_Q2_CPU_PROBE
                for(unsigned feature=0;feature<8192u;++feature)convolution_feature(cv.qkv,cv.initial_ring,cv.weights,
                    cv.silu,cv.first_position,feature,cv.staged_rings,cv.staged_convolution);
#else
                check(launch_linear(cv,view,tables));
#endif
                for(unsigned count:{1u,2u}) {
                    const auto selected=accepted_linear(cv,view,count);
                    selection_bad+=selected.rows!=count || selected.state!=accepted_state(view,count) ||
                        selected.ring!=staged_ring+(count-1u)*ring_elements;
                }
            };
            if(bf16_ring)submit(bf16_ring_input,bf16_ring_output);
            else submit(float_ring_input,float_ring_output);
        }
#ifdef QRT_Q2_CPU_PROBE
        for (unsigned row = 0; row < 2u; ++row) {
            for (unsigned head = 0; head < 32u; ++head) {
                RecurrentHead prepared;
                prepare_head(prepared,view.convolution+row*8192u,a[row*32u+head],b[row*32u+head],head,tables);
                for (unsigned v = 0; v < 128u; ++v) {
                    const size_t offset = state_offset(head,v,key_major);
                    const float* from = row ? view.staged_states : view.initial_state;
                    view.staged_core[row*core_elements+head*128u+v] = recurrent_value(
                        from+offset,view.staged_states+row*state_elements+offset,key_major?128u:1u,
                        prepared,view.convolution[row*8192u+4096u+head*128u+v],v);
                }
            }
        }
        input_bad += std::memcmp(before.data(),untouched.data(),before.size()*sizeof(float)) != 0;
#else
        phase = linear ? "connected_completion" : "launch_recurrence";
        if(!linear)check(launch_recurrent(view,tables));
        check(hipDeviceSynchronize());
        phase = "copy_results";
        check(hipMemcpy(staged.storage.data(),state_allocation,staged.storage.size()*4u,hipMemcpyDeviceToHost));
        check(hipMemcpy(core.storage.data(),core_allocation,core.storage.size()*2u,hipMemcpyDeviceToHost));
        input_bad += device.differences(view.initial_state,untouched);
        if(linear) {
            check(hipMemcpy(connected_conv.storage.data(),conv_allocation,connected_conv.storage.size()*2u,hipMemcpyDeviceToHost));
            if(bf16_ring) {
                check(hipMemcpy(bf16_rings.storage.data(),bf16_ring_allocation,bf16_rings.storage.size()*2u,hipMemcpyDeviceToHost));
                input_bad+=device.differences(bf16_ring_input,initial_bf16_ring);
            } else {
                check(hipMemcpy(float_rings.storage.data(),float_ring_allocation,float_rings.storage.size()*4u,hipMemcpyDeviceToHost));
                input_bad+=device.differences(float_ring_input,initial_float_ring);
            }
        }
#endif
        if(linear) {
            guard_bad+=connected_conv.guard_errors()+float_rings.guard_errors()+bf16_rings.guard_errors();
            const auto* computed=static_cast<const uint16_t*>(connected_conv.data());
            for(size_t i=0;i<conv.size();++i)conv_bad+=computed[i]!=conv[i];
            const auto compare_rings=[&](const auto* rings) {
                for(unsigned feature=0;feature<8192u;++feature)for(unsigned row=0;row<2u;++row)
                    for(unsigned retained=0;retained<4u;++retained) {
                        const size_t absolute=position-3u+row+retained;
                        const size_t at=row*ring_elements+(absolute%4u)*8192u+feature;
                        const uint16_t wanted=!row && !retained ? conv_before[feature*4u+history_offset]
                            : conv_after[feature*4u+row+retained-1u];
                        ring_bad+=qrt_sm121_exp2::bits(ring_float(rings[at]))!=
                            qrt_sm121_exp2::bits(qrt_sm121_q1::widen(wanted));
                    }
            };
            if(bf16_ring)compare_rings(static_cast<const uint16_t*>(bf16_rings.data()));
            else compare_rings(static_cast<const float*>(float_rings.data()));
        }
        guard_bad += staged.guard_errors()+core.guard_errors();
        const auto* actual = static_cast<const float*>(staged.data());
        const auto* actual_core = static_cast<const uint16_t*>(core.data());
        for (unsigned row = 0; row < 2u; ++row) {
            for (unsigned head = 0; head < 32u; ++head) for (unsigned v = 0; v < 128u; ++v) {
                for (unsigned k = 0; k < 128u; ++k) {
                    const size_t wanted = row*state_elements+(head*128u+v)*128u+k;
                    const size_t at = row*state_elements+state_offset(head,v,key_major)+k*(key_major?128u:1u);
                    state_bad[row] += qrt_sm121_exp2::bits(actual[at]) != qrt_sm121_exp2::bits(expected[wanted]);
                }
                const size_t at = row*core_elements+head*128u+v;
                core_bad[row] += actual_core[at] != expected_core[at];
            }
        }
        selection_bad += accepted_state(view,1u) != view.staged_states;
        selection_bad += accepted_state(view,2u) != view.staged_states+state_elements;
        selection_bad += accepted_state(view,0u) != nullptr || accepted_state(view,3u) != nullptr;
    }
#ifndef QRT_Q2_CPU_PROBE
    phase = "verify_immutable_inputs";
    // Connected execution computes convolution in private staging. Its
    // captured comparison buffer is never uploaded as a compute operand.
    if (!linear) input_bad += device.differences(convolution,conv);
    input_bad += device.differences(projection_a,a)+device.differences(projection_b,b);
    input_bad += device.differences(tables.g,g)+device.differences(tables.beta,beta);
    input_bad += device.differences(tables.exp2,exp2)+device.differences(tables.rsqrt,rsqrt);
    if(linear)input_bad+=device.differences(actual_qkv,qkv)+device.differences(actual_conv_weights,conv_weights)+device.differences(actual_silu,silu);
#endif
    const bool passed = !(state_bad[0] || state_bad[1] || core_bad[0] || core_bad[1] || selection_bad || input_bad || guard_bad || conv_bad || ring_bad);
    std::cout << ",\"connected_convolution\":" << (linear?"true":"false") << ",\"configurations\":" << configurations
        << ",\"convolution_bf16_mismatches\":" << conv_bad << ",\"staged_ring_mismatches\":" << ring_bad
        << ",\"ring_element_configurations\":[" << (linear?"\"f32\",\"bf16\"":"") << ']'
        << ",\"state_elements\":" << configurations*2u*state_elements << ",\"core_elements\":" << configurations*2u*core_elements
        << ",\"state_f32_bit_mismatches_by_row\":[" << state_bad[0] << ',' << state_bad[1]
        << "],\"core_bf16_mismatches_by_row\":[" << core_bad[0] << ',' << core_bad[1]
        << "],\"accepted_state_selection_errors\":" << selection_bad << ",\"immutable_input_errors\":" << input_bad
        << ",\"guard_errors\":" << guard_bad << ",\"rejected_aliases\":" << rejected
        << ",\"passed\":" << (passed?"true":"false") << ",\"resident_cache_published\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
} catch (const std::exception& error) {
    std::cerr << "phase=" << phase << " configuration=" << active_configuration
        << " error=" << error.what() << '\n';
    return 2;
}
