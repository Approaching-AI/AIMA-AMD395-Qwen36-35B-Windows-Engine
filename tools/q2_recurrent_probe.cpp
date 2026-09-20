// Original q2 recurrence in both resident layouts, including rejected row one.
#ifdef QRT_Q2_CPU_PROBE
#include "native/providers/gdn/sm121_q2_recurrent_layout.h"
#else
#include "native/providers/gdn/sm121_q2_recurrent.h"
#endif
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qrt_sm121_q2;
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
    if (argc != 11) throw std::runtime_error("conv a b before after core g beta exp2 rsqrt");
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
    size_t state_bad[2] = {}, core_bad[2] = {}, selection_bad = 0, input_bad = 0, guard_bad = 0;
    unsigned rejected = 0;
    std::cout << "{\"kind\":\"original_q2_recurrence\",\"layouts\":[\"value_key\",\"key_value\"],\"native_execution\":";
#ifdef QRT_Q2_CPU_PROBE
    std::cout << "false";
    const RecurrentTables tables{g.data(), beta.data(), exp2.data(), rsqrt.data()};
    const auto* convolution = conv.data(); const auto* projection_a = a.data(); const auto* projection_b = b.data();
#else
    std::cout << "true";
    Device device;
    const RecurrentTables tables{device.upload(g), device.upload(beta), device.upload(exp2), device.upload(rsqrt)};
    const auto* convolution = device.upload(conv); const auto* projection_a = device.upload(a); const auto* projection_b = device.upload(b);
#endif
    for (bool key_major : {false, true}) {
        std::vector<float> before(state_elements);
        for (unsigned head = 0; head < 32u; ++head)
            for (unsigned v = 0; v < 128u; ++v)
                for (unsigned k = 0; k < 128u; ++k)
                    before[state_offset(head,v,key_major)+k*(key_major?128u:1u)] = initial[(head*128u+v)*128u+k];
        const auto untouched = before;
        GuardedHost<float> staged(staged_state_bytes);
        GuardedHost<uint16_t> core(staged_core_bytes);
        RecurrentViews view;
        view.convolution = convolution; view.a = projection_a; view.b = projection_b; view.key_major = key_major;
#ifdef QRT_Q2_CPU_PROBE
        view.initial_state = before.data();
        view.staged_states = static_cast<float*>(staged.data());
        view.staged_core = static_cast<uint16_t*>(core.data());
#else
        view.initial_state = device.upload(before);
        auto* state_allocation = device.upload(staged.storage);
        auto* core_allocation = device.upload(core.storage);
        view.staged_states = state_allocation+64u;
        view.staged_core = core_allocation+128u;
#endif
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
#ifdef QRT_Q2_CPU_PROBE
        for (unsigned row = 0; row < 2u; ++row) {
            for (unsigned head = 0; head < 32u; ++head) {
                RecurrentHead prepared;
                prepare_head(prepared,conv.data()+row*8192u,a[row*32u+head],b[row*32u+head],head,tables);
                for (unsigned v = 0; v < 128u; ++v) {
                    const size_t offset = state_offset(head,v,key_major);
                    const float* from = row ? view.staged_states : view.initial_state;
                    view.staged_core[row*core_elements+head*128u+v] = recurrent_value(
                        from+offset,view.staged_states+row*state_elements+offset,key_major?128u:1u,
                        prepared,conv[row*8192u+4096u+head*128u+v],v);
                }
            }
        }
        input_bad += std::memcmp(before.data(),untouched.data(),before.size()*sizeof(float)) != 0;
#else
        check(launch_recurrent(view,tables)); check(hipDeviceSynchronize());
        check(hipMemcpy(staged.storage.data(),state_allocation,staged.storage.size()*4u,hipMemcpyDeviceToHost));
        check(hipMemcpy(core.storage.data(),core_allocation,core.storage.size()*2u,hipMemcpyDeviceToHost));
        input_bad += device.differences(view.initial_state,untouched);
#endif
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
    input_bad += device.differences(convolution,conv)+device.differences(projection_a,a)+device.differences(projection_b,b);
    input_bad += device.differences(tables.g,g)+device.differences(tables.beta,beta);
    input_bad += device.differences(tables.exp2,exp2)+device.differences(tables.rsqrt,rsqrt);
#endif
    const bool passed = !(state_bad[0] || state_bad[1] || core_bad[0] || core_bad[1] || selection_bad || input_bad || guard_bad);
    std::cout << ",\"state_elements\":" << 4u*state_elements << ",\"core_elements\":" << 4u*core_elements
        << ",\"state_f32_bit_mismatches_by_row\":[" << state_bad[0] << ',' << state_bad[1]
        << "],\"core_bf16_mismatches_by_row\":[" << core_bad[0] << ',' << core_bad[1]
        << "],\"accepted_state_selection_errors\":" << selection_bad << ",\"immutable_input_errors\":" << input_bad
        << ",\"guard_errors\":" << guard_bad << ",\"rejected_aliases\":" << rejected
        << ",\"passed\":" << (passed?"true":"false") << ",\"resident_cache_published\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
