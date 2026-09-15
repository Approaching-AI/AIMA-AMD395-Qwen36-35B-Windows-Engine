#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/shared_parallel_pv.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
constexpr unsigned guard = 64u;
constexpr unsigned variants[] = {0u, 1u, 2u};
constexpr unsigned variant_count = sizeof(variants) / sizeof(variants[0]);
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer && hipFree(pointer) != hipSuccess) std::abort(); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
template<class T> void upload(Device& d, const std::vector<T>& values) {
    check(hipMemcpy(d.pointer, values.data(), values.size()*sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> values(count);
    check(hipMemcpy(values.data(), d.pointer, count*sizeof(T), hipMemcpyDeviceToHost));
    return values;
}
template<class T> void unchanged(Device& d, const std::vector<T>& expected) {
    const auto actual = download<T>(d, expected.size());
    if (std::memcmp(actual.data(), expected.data(), expected.size()*sizeof(T)))
        throw std::runtime_error("QK input, encoding or redzone changed");
}
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("QK completion deadline");
        // Keep the explicit deadline without quantizing each short GPU slab
        // through a fixed host sleep. The benchmark reports completed host time.
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, sizeof(u)); return u; }
double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
}
struct Prepared {
    unsigned tokens;
    std::vector<uint32_t> qpacked, kpacked, qflags, kflags;
    Device qp, kp, qf, kf;
    double ms = 0.0;
    Prepared(const uint16_t* q, const uint16_t* k, uint16_t* transposed,
        const uint16_t* hq, const uint16_t* hk, unsigned n)
        : tokens(n), qpacked(size_t(n)*kQueryHeads*kHeadDim+2u*guard,0xa5a5a5a5u),
          kpacked(size_t(n)*kKvHeads*kHeadDim+2u*guard,0xa5a5a5a5u),
          qflags(n*kQueryHeads+2u*guard,0xa5a5a5a5u), kflags(n*kKvHeads+2u*guard,0xa5a5a5a5u),
          qp(qpacked.size()*4u), kp(kpacked.size()*4u), qf(qflags.size()*4u), kf(kflags.size()*4u) {
        for(Device* d:{&qp,&kp,&qf,&kf}) {
            const size_t words=d==&qp?qpacked.size():d==&kp?kpacked.size():d==&qf?qflags.size():kflags.size();
            check(hipMemset(d->pointer,0xa5,words*4u));
        }
        const auto begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::prepare<false>),dim3(n*kQueryHeads),dim3(kHeadDim),0u,nullptr,
            q,qp.as<uint32_t>()+guard,qf.as<unsigned>()+guard,nullptr,n);
        check(hipGetLastError());
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::prepare<true>),dim3(n*kKvHeads),dim3(kHeadDim),0u,nullptr,
            k,kp.as<uint32_t>()+guard,kf.as<unsigned>()+guard,transposed,n);
        check(hipGetLastError());finish();ms=elapsed(begin);
        for(unsigned key=0u;key<2u;++key) {
            const unsigned heads=key?kKvHeads:kQueryHeads;
            const auto* source=key?hk:hq;auto& packed=key?kpacked:qpacked;auto& flags=key?kflags:qflags;
            for(unsigned row=0u;row<n*heads;++row) {
                bool eligible=true;
                for(unsigned c=0u;c<kHeadDim;++c) {
                    const uint16_t x=source[size_t(row)*kHeadDim+c];
                    const unsigned e=(x>>7u)&255u;
                    eligible &= !(x&0x7fffu) || (e>=64u && e<=190u);
                    const int exponent=(x&0x7fffu)?int(e)-127:-512;
                    const size_t index=key?(size_t(row%heads)*kHeadDim+c)*n+row/heads:size_t(row)*kHeadDim+c;
                    packed[guard+index]=(uint32_t(x)<<16u)|uint16_t(exponent);
                }
                flags[guard+row]=unsigned(eligible);
            }
        }
        verify();
    }
    void verify() { unchanged(qp,qpacked);unchanged(kp,kpacked);unchanged(qf,qflags);unchanged(kf,kflags); }
};

template<class T>std::vector<T> read_values(const char* path,size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file||file.tellg()!=std::streamoff(count*sizeof(T)))throw std::runtime_error("PV capture size");
    std::vector<T> result(count);file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(result.data()),std::streamsize(count*sizeof(T))))throw std::runtime_error("PV capture read");
    return result;
}
__global__ void compare(const uint32_t* expected,const uint32_t* actual,size_t count,unsigned* bad) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count&&expected[i]!=actual[i])atomicAdd(bad,1u);
}
void compare_device(Device& a,Device& b,size_t words,Device& bad) {
    hipLaunchKernelGGL(compare,dim3((words+255u)/256u),dim3(256u),0u,nullptr,
        a.as<uint32_t>(),b.as<uint32_t>(),words,bad.as<unsigned>());check(hipGetLastError());
}
float cpu_pv(const uint16_t* p,const uint16_t* v,const float* scales,unsigned tokens,unsigned column) {
    float accumulator=0.0f;
    for(unsigned tile=0u;tile<(tokens+31u)/32u;++tile) {
        volatile float scaled=accumulator*scales[tile];
        auto partial=qrt_q1_moe_hawkeye::value_from_float(scaled,-133);
        for(unsigned group=0u;group<2u;++group) {
            qrt_q1_moe_hawkeye::Value values[17];values[0]=partial;
            for(unsigned i=0u;i<16u;++i) {
                const unsigned key=tile*32u+group*16u+i;
                values[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(key<tokens?p[key]:0u,key<tokens?v[size_t(key)*512u+column]:0u,-133);
            }
            partial=qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
            partial=qrt_q1_moe_hawkeye::value_from_float(qrt_q1_moe_hawkeye::value_to_float(
                qrt_sm121_group16::finish_accumulator(partial)),-133);
        }
        accumulator=qrt_q1_moe_hawkeye::value_to_float(partial);
    }
    return accumulator;
}


// The remaining checks exercise the new producer and ordered reducer.
namespace candidate = qrt_parallel_pv;
namespace bound = qrt_sm121_pv_bound;
struct PartialWorkspace {
    size_t capacity;
    Device device;
    PartialWorkspace(unsigned start, unsigned count)
        : capacity(candidate::partial_count(start, count)), device((capacity + 2u * guard) * sizeof(candidate::Partial)) {
        reset();
    }
    candidate::Partial* data() { return device.as<candidate::Partial>() + guard; }
    void reset() { check(hipMemset(device.pointer, 0xa5, (capacity + 2u * guard) * sizeof(candidate::Partial))); }
    void verify_outer() {
        uint64_t first[guard], last[guard];
        check(hipMemcpy(first, device.pointer, sizeof(first), hipMemcpyDeviceToHost));
        check(hipMemcpy(last, data() + capacity, sizeof(last), hipMemcpyDeviceToHost));
        for (unsigned i = 0u; i < guard; ++i)
            if (first[i] != 0xa5a5a5a5a5a5a5a5ull || last[i] != 0xa5a5a5a5a5a5a5a5ull) throw std::runtime_error("partial workspace outer guard");
    }
    void verify(unsigned start, unsigned count) {
        // Reused slabs can leave earlier valid values in their unused query
        // rows. Check outer guards and every location never written by any
        // submitted slab, using the union of their exact write domains.
        const auto words = download<uint64_t>(device, capacity + 2u * guard);
        unsigned largest_groups[16]{};
        for (unsigned offset = 0u; offset < count; offset += 16u) {
            const unsigned active = std::min(16u, count - offset);
            for (unsigned row = 0u; row < active; ++row)
                largest_groups[row] = std::max(largest_groups[row], candidate::groups(start + offset + active));
        }
        for (size_t i = 0u; i < words.size(); ++i) {
            const size_t relative = i - guard;
            const unsigned row = unsigned(relative / 4096u % 16u), group = unsigned(relative / (16u * 4096u));
            const bool live = i >= guard && i < guard + capacity && group < largest_groups[row];
            if (!live && words[i] != 0xa5a5a5a5a5a5a5a5ull) throw std::runtime_error("partial workspace guard or untouched slot");
        }
    }
};
struct Outputs {
    unsigned queries, offset, cells;
    size_t words, den_words;
    Device output, accumulator, denominator, error, indices;
    explicit Outputs(unsigned q, unsigned start = 0u)
        : queries(q), offset(start), cells(q * 4096u), words(size_t(q + start) * 4096u + 2u * guard),
          den_words(size_t(q + start) * 16u + 2u * guard), output(words * 4u), accumulator(words * 4u),
          denominator(den_words * 4u), error((cells + 2u * guard) * 4u), indices((cells + 1u + 2u * guard) * 4u) { reset(); }
    void reset() {
        check(hipMemset(output.pointer, 0xa5, words * 4u)); check(hipMemset(accumulator.pointer, 0xa5, words * 4u));
        check(hipMemset(denominator.pointer, 0xa5, den_words * 4u));
        check(hipMemset(error.pointer, 0xa5, (cells + 2u * guard) * 4u));
        check(hipMemset(indices.pointer, 0xa5, (cells + 1u + 2u * guard) * 4u));
    }
    void verify_guards() {
        for (Device* d : {&output, &accumulator, &denominator, &error}) {
            const size_t n = d == &denominator ? den_words : d == &error ? cells + 2u * guard : words;
            const size_t begin = guard + (d == &error ? 0u : offset * (d == &denominator ? 16u : 4096u));
            const size_t live = d == &denominator ? queries * 16u : cells;
            const auto v = download<uint32_t>(*d, n);
            for (size_t i = 0u; i < n; ++i)
                if ((i < begin || i >= begin + live) && v[i] != 0xa5a5a5a5u) throw std::runtime_error("PV output, error or denominator guard");
        }
    }
    std::vector<unsigned> candidate_ids() {
        const auto v = download<unsigned>(indices, cells + 1u + 2u * guard);
        const unsigned count = v[guard + cells];
        if (count > cells) throw std::runtime_error("PV candidate count");
        std::vector<bool> seen(cells, false);
        for (unsigned i = 0u; i < count; ++i) {
            const unsigned cell = v[guard + i];
            if (cell >= cells || seen[cell]) throw std::runtime_error("PV candidate ownership");
            seen[cell] = true;
        }
        for (size_t i = 0u; i < v.size(); ++i)
            if ((i < guard || (i >= guard + count && i < guard + cells) || i > guard + cells) && v[i] != 0xa5a5a5a5u)
                throw std::runtime_error("PV candidate guard or unused tail");
        return {v.begin() + guard, v.begin() + guard + count};
    }
};
void approximate(unsigned variant, const uint16_t* value, const uint16_t* probability,
    const float* scales, unsigned start, unsigned queries, unsigned stride,
    const unsigned char* rcp, Outputs& result, PartialWorkspace& workspace) {
    if (!variant) {
        hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true>),
            dim3(2u, 16u, (queries + 15u) / 16u), dim3(256u), 0u, nullptr,
            value, probability, scales, result.output.as<float>() + guard, start, queries, result.offset, stride,
            rcp, result.accumulator.as<float>() + guard, result.denominator.as<float>() + guard,
            nullptr, nullptr, result.error.as<float>() + guard);
        check(hipGetLastError());
    } else if (variant == 2u) {
        check(hipError_t(candidate::launch_shared(value, probability, scales, result.output.as<float>() + guard,
            result.error.as<float>() + guard, start, queries, result.offset, stride, rcp, nullptr,
            result.accumulator.as<float>() + guard, result.denominator.as<float>() + guard)));
    } else {
        if (variant != 1u) throw std::runtime_error("PV variant");
        check(hipError_t(candidate::launch(value, probability, scales, result.output.as<float>() + guard,
            result.error.as<float>() + guard, start, queries, result.offset, stride, rcp,
            workspace.data(), workspace.capacity, nullptr, result.accumulator.as<float>() + guard,
            result.denominator.as<float>() + guard)));
    }
}
void exact_replay(const uint16_t* value, const uint16_t* probability, const float* scales,
    unsigned start, unsigned queries, unsigned stride, const unsigned char* rcp, Outputs& result,
    const uint16_t* transposed = nullptr, unsigned value_stride = 0u) {
    check(hipError_t(launch_compacted_pv_replay(value, probability, scales,
        result.output.as<float>() + guard, start, queries, result.offset, stride, rcp,
        result.accumulator.as<float>() + guard, result.denominator.as<float>() + guard,
        result.error.as<float>() + guard, result.indices.as<unsigned>() + guard,
        result.indices.as<unsigned>() + guard + result.cells, nullptr, nullptr, transposed, value_stride)));
}
void generated(unsigned start, unsigned queries, unsigned mode, const unsigned char* rcp) {
    const unsigned stride = start + queries, rows = queries * 16u, tiles = (stride + 31u) / 32u;
    std::vector<uint16_t> p(size_t(rows) * stride + 2u * guard, 0x5a5au), v(size_t(stride) * 512u + 2u * guard, 0x5a5au);
    std::vector<float> s(size_t(rows) * (tiles + 1u) + 2u * guard, 12345.0f);
    unsigned seed = 0x8192395u ^ (start + queries * 977u + mode * 12345u);
    const auto random = [&seed]() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; };
    for (size_t i = guard; i < p.size() - guard; ++i) {
        const unsigned exponent = mode == 2u ? 64u + random() % 64u : 118u + random() % 10u;
        p[i] = uint16_t((exponent << 7u) | (random() & 127u));
        if (mode == 3u && i % 3u) p[i] = i & 1u ? 0x8000u : 0u;
    }
    for (size_t i = guard; i < v.size() - guard; ++i) {
        const unsigned exponent = mode == 2u ? 1u + random() % 190u : 119u + random() % 16u;
        v[i] = uint16_t((exponent << 7u) | (random() & 0x807fu));
        if (mode == 1u && (i - guard) / 512u % 2u) v[i] = v[i - 512u] ^ 0x8000u;
    }
    for (unsigned row = 0u; row < rows; ++row) {
        if (mode == 1u) for (unsigned key = 1u; key < start + row / 16u + 1u; key += 2u)
            p[guard + size_t(row) * stride + key] = p[guard + size_t(row) * stride + key - 1u];
        for (unsigned tile = 0u; tile < tiles; ++tile)
            s[guard + size_t(row) * (tiles + 1u) + tile] = !tile ? 0.0f : tile % 8u ? 1.0f : mode == 3u ? 0.0f : 0.625f;
        s[guard + size_t(row) * (tiles + 1u) + tiles] = 1.0f + float(row % 31u) / 16.0f;
    }
    Device dp(p.size() * 2u), dv(v.size() * 2u), ds(s.size() * 4u);
    upload(dp, p); upload(dv, v); upload(ds, s);
    const unsigned offset = queries + 3u <= 8192u ? 3u : 0u;
    Outputs exact(queries, offset), test(queries, offset);
    PartialWorkspace workspace(start, queries);
    check(hipError_t(launch_all_pv_replay(dv.as<uint16_t>() + guard, dp.as<uint16_t>() + guard,
        ds.as<float>() + guard, exact.output.as<float>() + guard, start, queries, offset, stride, rcp,
        exact.accumulator.as<float>() + guard, exact.denominator.as<float>() + guard, nullptr)));
    finish();
    const auto expected = download<float>(exact.output, exact.words), expected_acc = download<float>(exact.accumulator, exact.words);
    const auto expected_den = download<float>(exact.denominator, exact.den_words);
    unsigned cpu_checks = 0u;
    for (unsigned cell : {0u, 257u, test.cells / 2u, test.cells - 1u}) {
        const unsigned row = cell / 256u, column = cell % 256u;
        const float cpu = cpu_pv(p.data() + guard + size_t(row) * stride, v.data() + guard,
            s.data() + guard + size_t(row) * (tiles + 1u), start + row / 16u + 1u, (row % 16u / 8u) * 256u + column);
        if (bits(cpu) != bits(expected_acc[guard + size_t(offset) * 4096u + cell])) throw std::runtime_error("wide CPU PV mismatch");
        ++cpu_checks;
    }
    std::vector<uint32_t> global_proxy, global_accumulator, global_error, global_denominator;
    for (unsigned variant : variants) {
        test.reset(); workspace.reset(); finish();
        approximate(variant, dv.as<uint16_t>() + guard, dp.as<uint16_t>() + guard, ds.as<float>() + guard,
            start, queries, stride, rcp, test, workspace); finish();
        const auto proxy = download<float>(test.output, test.words), error = download<float>(test.error, test.cells + 2u * guard);
        const auto den = download<float>(test.denominator, test.den_words);
        if (std::memcmp(den.data(), expected_den.data(), den.size() * 4u)) throw std::runtime_error("PV denominator changed");
        unsigned admitted = 0u, underestimates = 0u;
        for (unsigned cell = 0u; cell < test.cells; ++cell) {
            const size_t index = guard + size_t(offset) * 4096u + cell;
            underestimates += unsigned(std::fabs(double(proxy[index]) - expected[index]) > double(error[guard + cell]));
            if (bound::same_bf16(proxy[index], error[guard + cell])) {
                if (bound::bf16(proxy[index]) != bound::bf16(expected[index])) throw std::runtime_error("false PV admission");
                ++admitted;
            }
        }
        if (underestimates) throw std::runtime_error("PV envelope underestimates exact output");
        if (variant == 1u) {
            global_proxy = download<uint32_t>(test.output, test.words);
            global_accumulator = download<uint32_t>(test.accumulator, test.words);
            global_error = download<uint32_t>(test.error, test.cells + 2u * guard);
            global_denominator = download<uint32_t>(test.denominator, test.den_words);
        } else if (variant == 2u) {
            if (download<uint32_t>(test.output, test.words) != global_proxy ||
                download<uint32_t>(test.accumulator, test.words) != global_accumulator ||
                download<uint32_t>(test.error, test.cells + 2u * guard) != global_error ||
                download<uint32_t>(test.denominator, test.den_words) != global_denominator)
                throw std::runtime_error("shared partials changed proxy, accumulator, bound or denominator bits");
        }
        test.verify_guards(); if (variant == 1u) workspace.verify(start, queries); else workspace.verify_outer();
        exact_replay(dv.as<uint16_t>() + guard, dp.as<uint16_t>() + guard, ds.as<float>() + guard,
            start, queries, stride, rcp, test); finish();
        const auto ids = test.candidate_ids();
        const auto final_output = download<float>(test.output, test.words);
        if (ids.size() + admitted != test.cells) throw std::runtime_error("PV admission partition");
        for (unsigned cell = 0u; cell < test.cells; ++cell) {
            const size_t index = guard + size_t(offset) * 4096u + cell;
            if (bound::bf16(final_output[index]) != bound::bf16(expected[index])) throw std::runtime_error("corrected PV differs from canonical BF16");
        }
        test.verify_guards();
        std::printf("{\"kind\":\"parallel_pv_safety\",\"start\":%u,\"queries\":%u,\"mode\":%u,\"variant\":%u,\"cells\":%u,\"candidates\":%zu,\"admitted\":%u,\"cpu_dots\":%u,\"underestimates\":0,\"false_admissions\":0,\"bf16_mismatches\":0,\"denominator_raw_mismatches\":0,\"workspace_bytes\":%zu,\"shared_bytes\":%zu,\"shared_matches_global_proxy_raw\":%s,\"redzones_pass\":true,\"immutable_inputs\":true,\"unique_candidates\":true,\"inference_acceptance\":false}\n",
            start, queries, mode, variant, test.cells, ids.size(), admitted, cpu_checks, variant == 1u ? workspace.capacity * sizeof(candidate::Partial) : 0u, variant == 2u ? candidate::shared_bytes : 0u, variant == 2u ? "true" : "false");
        std::fflush(stdout);
    }
    unchanged(dp, p); unchanged(dv, v); unchanged(ds, s); exact.verify_guards();
}

void captured(const char* qfile, const char* kfile, const char* vfile, const char* reference_file,
    const char* exp_file, const char* rcp_file, bool extend) {
    constexpr unsigned original_tokens = 7169u, batch = 128u, attempts = 4u;
    const unsigned tokens = extend ? 8192u : original_tokens;
    auto q = read_values<uint16_t>(qfile, size_t(original_tokens) * 4096u);
    auto k = read_values<uint16_t>(kfile, size_t(original_tokens) * 512u);
    auto v = read_values<uint16_t>(vfile, size_t(original_tokens) * 512u);
    const auto golden = read_values<uint16_t>(reference_file, size_t(original_tokens) * 4096u);
    const auto exp = read_values<unsigned char>(exp_file, exp2_backend::table_bytes);
    const auto rcp = read_values<unsigned char>(rcp_file, qrt_sm121_attention_rcp::table_bytes);
    if (!exp2_backend::valid_layout(exp.data(), exp.size()) || !qrt_sm121_attention_rcp::valid_layout(rcp.data(), rcp.size()))
        throw std::runtime_error("PV SFU table layout");
    if (extend) {
        // A declared component extension, never a new real-model prompt.
        // Causality preserves the original7169 GB10 context rows; appended
        // rows are compared against complete canonical GPU replay below.
        for (auto* data : {&q, &k, &v}) {
            const unsigned width = data == &q ? 4096u : 512u;
            data->resize(size_t(tokens) * width);
            std::copy_n(data->begin(), size_t(tokens - original_tokens) * width, data->begin() + size_t(original_tokens) * width);
        }
    }
    for (auto* data : {&q, &k, &v}) {
        data->insert(data->begin(), guard, 0x5a5au); data->insert(data->end(), guard, 0x5a5au);
    }
    std::vector<uint16_t> tv(v.size(), 0x5a5au);
    for (unsigned key = 0u; key < tokens; ++key) for (unsigned c = 0u; c < 512u; ++c)
        tv[guard + size_t(c) * tokens + key] = v[guard + size_t(key) * 512u + c];
    Device dq(q.size() * 2u), dk(k.size() * 2u), dv(v.size() * 2u), dkt(k.size() * 2u), dvt(v.size() * 2u), dex(exp.size()), drcp(rcp.size());
    upload(dq, q); upload(dk, k); upload(dv, v); upload(dex, exp); upload(drcp, rcp);
    check(hipMemset(dkt.pointer, 0x5a, k.size() * 2u)); check(hipMemset(dvt.pointer, 0x5a, v.size() * 2u)); finish();
    auto begin = std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>() + guard, dvt.as<uint16_t>() + guard, size_t(tokens) * 512u, tokens, nullptr)));
    finish(); const double transpose_ms = elapsed(begin); unchanged(dvt, tv);
    Prepared prepared(dq.as<uint16_t>() + guard, dk.as<uint16_t>() + guard, dkt.as<uint16_t>() + guard, q.data() + guard, k.data() + guard, tokens);
    const auto kt_before = download<uint16_t>(dkt, k.size());
    const size_t score_capacity = size_t(batch) * 16u * tokens, scale_capacity = size_t(batch) * 16u * ((tokens + 31u) / 32u + 1u);
    Device scores((score_capacity + 2u * guard) * 4u), probability((score_capacity + 2u * guard) * 2u), scales((scale_capacity + 2u * guard) * 4u);
    PartialWorkspace workspace(tokens - 1u, 1u);
    double samples[3][3]{}, maximum[3]{}, common_ms = 0.0;
    size_t compared[3]{}, external[3]{}, appended[3]{}, selected[3]{};
    unsigned cpu_dots[3]{};
    for (unsigned start = 0u; start < tokens; start += batch) {
        const unsigned queries = std::min(batch, tokens - start), stride = start + queries, cells = queries * 4096u, rows = queries * 16u, tiles = (stride + 31u) / 32u;
        check(hipMemset(scores.pointer, 0xa5, (score_capacity + 2u * guard) * 4u));
        check(hipMemset(probability.pointer, 0x5a, (score_capacity + 2u * guard) * 2u));
        check(hipMemset(scales.pointer, 0xa5, (scale_capacity + 2u * guard) * 4u)); finish();
        begin = std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),
            dim3((stride + 15u) / 16u, 16u, (queries + 15u) / 16u), dim3(256u), 0u, nullptr,
            dq.as<uint16_t>() + guard, dkt.as<uint16_t>() + guard, prepared.qp.as<uint32_t>() + guard,
            prepared.kp.as<uint32_t>() + guard, prepared.qf.as<unsigned>() + guard, prepared.kf.as<unsigned>() + guard,
            scores.as<float>() + guard, start, queries, stride, tokens);
        check(hipGetLastError());
        hipLaunchKernelGGL(blackwell_online_probability_kernel, dim3(16u, queries), dim3(32u), 0u, nullptr,
            scores.as<float>() + guard, probability.as<uint16_t>() + guard, scales.as<float>() + guard, start, stride, dex.as<unsigned char>(), true);
        check(hipGetLastError()); finish(); common_ms += elapsed(begin);
        const auto p = download<uint16_t>(probability, score_capacity + 2u * guard);
        const auto s = download<float>(scales, scale_capacity + 2u * guard);
        const auto scores_before = download<uint32_t>(scores, score_capacity + 2u * guard);
        // Check untouched input guards/tails, including causal padding.
        for (size_t i = 0u; i < p.size(); ++i) {
            bool live = i >= guard && i < guard + size_t(rows) * stride;
            if (live) {
                const unsigned row = unsigned((i - guard) / stride), key = unsigned((i - guard) % stride);
                live = key < (start + row / 16u + 32u) / 32u * 32u;
                if (live && key >= start + row / 16u + 1u && p[i] != 0u) throw std::runtime_error("PV causal probability padding");
            }
            if (!live && p[i] != 0x5a5au) throw std::runtime_error("PV probability untouched tail");
        }
        for (size_t i = 0u; i < s.size(); ++i)
            if ((i < guard || i >= guard + size_t(rows) * (tiles + 1u)) && bits(s[i]) != 0xa5a5a5a5u) throw std::runtime_error("PV scale outer guard");
        Outputs result0(queries), result1(queries), result2(queries), exact(queries);
        Outputs* result[] = {&result0, &result1, &result2};
        std::vector<float> canonical;
        if (extend && start + queries > original_tokens) {
            check(hipError_t(launch_all_pv_replay(dv.as<uint16_t>() + guard, probability.as<uint16_t>() + guard,
                scales.as<float>() + guard, exact.output.as<float>() + guard, start, queries, 0u, stride,
                drcp.as<unsigned char>(), exact.accumulator.as<float>() + guard, exact.denominator.as<float>() + guard,
                nullptr, nullptr, dvt.as<uint16_t>() + guard, tokens)));
            finish(); canonical = download<float>(exact.output, exact.words); exact.verify_guards();
        }
        std::vector<uint32_t> first_output[3], first_accumulator[3], first_denominator[3], first_errors[3];
        std::vector<unsigned> first_candidates[3];
        for (unsigned attempt = 0u; attempt < attempts; ++attempt) for (unsigned position = 0u; position < 3u; ++position) {
            const unsigned variant = (position + attempt + start / batch) % 3u;
            auto& out = *result[variant]; out.reset(); if (variant == 1u) workspace.reset(); finish();
            begin = std::chrono::steady_clock::now();
            approximate(variant, dv.as<uint16_t>() + guard, probability.as<uint16_t>() + guard, scales.as<float>() + guard,
                start, queries, stride, drcp.as<unsigned char>(), out, workspace);
            exact_replay(dv.as<uint16_t>() + guard, probability.as<uint16_t>() + guard, scales.as<float>() + guard,
                start, queries, stride, drcp.as<unsigned char>(), out, dvt.as<uint16_t>() + guard, tokens);
            finish(); const double wall = elapsed(begin);
            if (attempt) { samples[variant][attempt - 1u] += wall; maximum[variant] = std::max(maximum[variant], wall); }
            out.verify_guards();
            if (variant == 1u) { workspace.verify_outer(); if (attempt == attempts - 1u) workspace.verify(start, queries); }
            const auto output = download<float>(out.output, out.words), accumulator = download<float>(out.accumulator, out.words);
            const auto den = download<float>(out.denominator, out.den_words), error = download<float>(out.error, cells + 2u * guard);
            auto ids = out.candidate_ids(); std::sort(ids.begin(), ids.end());
            for (unsigned cell = 0u; cell < cells; ++cell) {
                const size_t global = size_t(start) * 4096u + cell;
                if (global < golden.size()) {
                    if (bound::bf16(output[guard + cell]) != golden[global]) throw std::runtime_error("PV original GB10 BF16 context mismatch");
                    if (!attempt) ++external[variant];
                } else {
                    if (canonical.empty() || bound::bf16(output[guard + cell]) != bound::bf16(canonical[guard + cell]))
                        throw std::runtime_error("PV appended full-shape canonical BF16 mismatch");
                    if (!attempt) ++appended[variant];
                }
            }
            if (!attempt) {
                first_output[variant] = download<uint32_t>(out.output, out.words);
                first_accumulator[variant] = download<uint32_t>(out.accumulator, out.words);
                first_denominator[variant] = download<uint32_t>(out.denominator, out.den_words);
                first_errors[variant] = download<uint32_t>(out.error, cells + 2u * guard);
                first_candidates[variant] = ids; selected[variant] += ids.size();
                for (unsigned sample = 0u; sample < 8u; ++sample) {
                    const unsigned cell = unsigned(size_t(sample) * (cells - 1u) / 7u), row = cell / 256u, column = cell % 256u;
                    const float cpu = cpu_pv(p.data() + guard + size_t(row) * stride, v.data() + guard,
                        s.data() + guard + size_t(row) * (tiles + 1u), start + row / 16u + 1u, (row % 16u / 8u) * 256u + column);
                    const float reciprocal = qrt_sm121_attention_rcp::evaluate(rcp.data(), s[guard + size_t(row) * (tiles + 1u) + tiles]);
                    const float cpu_output = qrt_sm121_pv_final_bound::multiply(cpu, reciprocal);
                    if (bound::bf16(cpu_output) != bound::bf16(output[guard + cell])) throw std::runtime_error("captured PV independent CPU endpoint");
                    if (std::binary_search(ids.begin(), ids.end(), cell) && bits(cpu) != bits(accumulator[guard + cell])) throw std::runtime_error("captured exact replay CPU numerator");
                    if (bits(den[guard + row]) != bits(s[guard + size_t(row) * (tiles + 1u) + tiles])) throw std::runtime_error("captured denominator");
                    ++cpu_dots[variant];
                }
            } else {
                if (download<uint32_t>(out.output, out.words) != first_output[variant] ||
                    download<uint32_t>(out.accumulator, out.words) != first_accumulator[variant] ||
                    download<uint32_t>(out.denominator, out.den_words) != first_denominator[variant] ||
                    download<uint32_t>(out.error, cells + 2u * guard) != first_errors[variant] || ids != first_candidates[variant])
                    throw std::runtime_error("PV repeated raw result or candidate set differs");
                compared[variant] += cells;
            }
        }
        if (first_denominator[0] != first_denominator[1] || first_denominator[1] != first_denominator[2])
            throw std::runtime_error("PV comparison denominator changed");
        if (first_output[1] != first_output[2] || first_accumulator[1] != first_accumulator[2] ||
            first_errors[1] != first_errors[2] || first_candidates[1] != first_candidates[2])
            throw std::runtime_error("shared partials changed corrected raw output, bound or candidate set");
        unchanged(probability, p); unchanged(scales, s); unchanged(scores, scores_before);
        std::printf("{\"kind\":\"parallel_pv_capture_progress\",\"tokens\":%u,\"completed_queries\":%u,\"control_candidates\":%zu,\"parallel_candidates\":%zu}\n", tokens, start + queries, selected[0], selected[1]); std::fflush(stdout);
    }
    unchanged(dq, q); unchanged(dk, k); unchanged(dv, v); unchanged(dkt, kt_before); unchanged(dvt, tv); unchanged(dex, exp); unchanged(drcp, rcp); prepared.verify();
    for (unsigned variant : variants) {
        if (external[variant] != golden.size() || external[variant] + appended[variant] != size_t(tokens) * 4096u)
            throw std::runtime_error("PV incomplete external or appended context");
        double sorted[] = {samples[variant][0], samples[variant][1], samples[variant][2]}; std::sort(sorted, sorted + 3u);
        std::printf("{\"kind\":\"parallel_pv_captured_component\",\"variant\":%u,\"tokens\":%u,\"query_batch\":128,\"partial_query_tile\":16,\"external_gb10_bf16_cells\":%zu,\"appended_canonical_bf16_cells\":%zu,\"bf16_mismatches\":0,\"selected_cells\":%zu,\"cpu_dots\":%u,\"repeat_raw_output_cells\":%zu,\"same_variant_repeat_raw_mismatches\":0,\"common_qk_probability_ms\":%.6f,\"query_key_preparation_ms\":%.6f,\"value_transpose_ms\":%.6f,\"pv_and_exact_replay_ms\":%.6f,\"complete_component_ms\":%.6f,\"pv_samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"workspace_bytes\":%zu,\"shared_bytes\":%zu,\"shared_matches_global_proxy_raw\":%s,\"allocation_and_sentinel_reset_timed\":false,\"all_attempts_verified\":true,\"original_reference_used_as_compute_input\":false,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant, tokens, external[variant], appended[variant], selected[variant], cpu_dots[variant], compared[variant], common_ms, prepared.ms,
            transpose_ms, sorted[1], common_ms + prepared.ms + transpose_ms + sorted[1], samples[variant][0], samples[variant][1], samples[variant][2], maximum[variant], variant == 1u ? workspace.capacity * sizeof(candidate::Partial) : 0u, variant == 2u ? candidate::shared_bytes : 0u, variant == 2u ? "true" : "false");
        std::fflush(stdout);
    }
}
} // namespace
int main(int argc, char** argv) try {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    if (argc == 8 && (!std::strcmp(argv[1], "--q7169") || !std::strcmp(argv[1], "--q8192"))) {
        captured(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], !std::strcmp(argv[1], "--q8192")); return 0;
    }
    if (argc != 3 || std::strcmp(argv[1], "--selftest")) throw std::runtime_error("use --selftest RCP or --q7169/--q8192 Q K V GB10_CONTEXT EXP RCP");
    const auto rcp = read_values<unsigned char>(argv[2], qrt_sm121_attention_rcp::table_bytes);
    if (!qrt_sm121_attention_rcp::valid_layout(rcp.data(), rcp.size())) throw std::runtime_error("PV reciprocal table layout");
    Device drcp(rcp.size()); upload(drcp, rcp);
    for (unsigned mode : {0u, 1u, 2u, 3u}) {
        // start16/count17 makes the final singleton slab cross into a new
        // K32 tile while the preceding fifteen query slots remain untouched.
        for (auto shape : {std::pair<unsigned,unsigned>{0u,1u}, {15u,2u}, {16u,17u}, {31u,17u}, {63u,128u}, {255u,33u}, {1023u,16u}, {7168u,17u}, {8175u,17u}})
            generated(shape.first, shape.second, mode, drcp.as<unsigned char>());
    }
    unchanged(drcp, rcp); return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "parallel_pv_error=%s\n", error.what()); return 1;
}
