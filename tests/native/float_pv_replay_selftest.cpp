#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace qrt_blackwell_attention;
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* pointer = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
};
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("PV completion deadline");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(event));
}
template<class T> void upload(Device& d, const std::vector<T>& v) {
    check(hipMemcpy(d.pointer, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& d, size_t count) {
    std::vector<T> v(count); check(hipMemcpy(v.data(), d.pointer, count * sizeof(T), hipMemcpyDeviceToHost)); return v;
}
template<class T> void immutable(Device& d, const std::vector<T>& expected) {
    auto actual = download<T>(d, expected.size());
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T))) throw std::runtime_error("input changed");
}
void run(unsigned start, unsigned queries, unsigned mode, bool transposed_value, unsigned input_mode) {
    constexpr unsigned guard = 64u, output_start = 3u;
    const unsigned tokens = start + queries, rows = queries * kQueryHeads, cells = rows * kHeadDim;
    const unsigned tiles = (tokens + 31u) / 32u;
    std::vector<uint16_t> value(size_t(tokens) * kKvHeads * kHeadDim + 2u * guard, 0x5a5au);
    std::vector<uint16_t> probability(size_t(rows) * tokens + 2u * guard, 0x5a5au);
    std::vector<float> scales(size_t(rows) * (tiles + 1u) + 2u * guard, 12345.0f);
    std::vector<float> errors(cells + 2u * guard, 12345.0f);
    for (size_t i = guard; i + guard < value.size(); ++i)
        value[i] = qrt_sm121_pv_bound::bf16(float(int((i * 173u + i / 17u) % 63u) - 31) / 64.0f);
    for (size_t i = guard; i + guard < probability.size(); ++i)
        probability[i] = qrt_sm121_pv_bound::bf16(float((i * 47u + i / 13u) % 31u) / 32.0f);
    if (input_mode) {
        for(unsigned feature=0u;feature<512u;++feature) {
            if(feature%7u==0u) value[guard+feature]=0x1fffu;
            else if(feature%7u==1u) value[guard+feature]=0x2001u;
            else if(feature%7u==2u) value[guard+feature]=0x5f7fu;
            else if(feature%7u==3u) value[guard+feature]=0x8000u;
            else if(feature%7u==4u) value[guard+feature]=0x0001u;
            else if(feature%7u==6u) value[guard+feature]=0x5f80u;
        }
        for(unsigned row=0u;row<rows;++row)
            if(row%11u==0u) probability[guard+size_t(row)*tokens]=0x1f81u;
    }
    for (unsigned row = 0u; row < rows; ++row) {
        for (unsigned tile = 0u; tile < tiles; ++tile)
            scales[guard + size_t(row) * (tiles + 1u) + tile] = tile % 7u ? 1.0f : (tile % 3u ? 0.5f : 0.875f);
        scales[guard + size_t(row) * (tiles + 1u) + tiles] = 1.0f + float(row % 31u) / 16.0f;
    }
    unsigned selected = 0u;
    for (unsigned cell = 0u; cell < cells; ++cell) {
        const bool active = mode == 1u || (mode == 2u && (cell % 257u == 0u || cell % 251u == 3u)) ||
            (mode == 3u && ((cell * 7179u) & 15u) < 7u);
        errors[guard + cell] = active ? qrt_sm121_pv_bound::infinity() : 0.0f;
        selected += active;
    }
    std::vector<uint16_t> transposed(value.size(), 0x5a5au);
    for (unsigned token=0u;token<tokens;++token)
        for (unsigned feature=0u;feature<kKvHeads*kHeadDim;++feature)
            transposed[guard+size_t(feature)*tokens+token]=value[guard+size_t(token)*kKvHeads*kHeadDim+feature];
    const size_t output_cells = size_t(output_start + queries) * kQueryHeads * kHeadDim + 2u * guard;
    const size_t denominator_cells = size_t(output_start + queries) * kQueryHeads + 2u * guard;
    std::vector<float> initial(output_cells, 12345.0f), acc(initial), den(denominator_cells, 12345.0f);
    std::fill(initial.begin() + guard + output_start * kQueryHeads * kHeadDim, initial.begin() + guard + output_start * kQueryHeads * kHeadDim + cells, 0.75f);
    std::vector<unsigned> scratch(cells + 1u + 2u * guard, 0xa5a5a5a5u);
    Device dv(value.size()*2u), dp(probability.size()*2u), ds(scales.size()*4u), de(errors.size()*4u);
    Device dtv(transposed.size()*2u);
    Device dout(initial.size()*4u), da(acc.size()*4u), dd(den.size()*4u), di(scratch.size()*4u);
    upload(dv,value); upload(dp,probability); upload(ds,scales); upload(de,errors);
    std::vector<uint16_t> transposed_initial(transposed.size(), 0x5a5au); upload(dtv,transposed_initial);
    if (transposed_value) {
        check(hipError_t(transpose_keys(dv.as<uint16_t>()+guard,dtv.as<uint16_t>()+guard,
            size_t(tokens)*kKvHeads*kHeadDim,tokens,nullptr)));
        finish(); immutable(dtv,transposed);
    }
    upload(dout,initial); upload(da,acc); upload(dd,den);
    hipLaunchKernelGGL(blackwell_probability_value_kernel, dim3(kQueryHeads,queries), dim3(kHeadDim), 0u, nullptr,
        dv.as<uint16_t>()+guard, dp.as<uint16_t>()+guard, ds.as<float>()+guard, dout.as<float>()+guard,
        start,output_start,tokens,nullptr,da.as<float>()+guard,dd.as<float>()+guard,de.as<float>()+guard);
    check(hipGetLastError()); finish();
    const auto control=download<float>(dout,initial.size()), control_acc=download<float>(da,acc.size()), control_den=download<float>(dd,den.size());
    const size_t flag_words=size_t(rows)+512u;
    std::vector<unsigned> flag_initial(flag_words+2u*guard,0xa5a5a5a5u);
    Device df(flag_initial.size()*4u);upload(df,flag_initial);
    auto* flag_data=df.as<unsigned>()+guard;
    check(hipError_t(qrt_sm121_float_pv::prepare(dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,
        start,queries,tokens,transposed_value ? dtv.as<uint16_t>()+guard : nullptr,
        transposed_value ? tokens : 0u,flag_data,flag_words,nullptr)));
    finish();auto metadata=download<unsigned>(df,flag_initial.size());
    for(size_t i=0u;i<metadata.size();++i) {
        if(i<guard || i>=guard+flag_words) {
            if(metadata[i]!=0xa5a5a5a5u) throw std::runtime_error("metadata guard changed");
            continue;
        }
        const unsigned row=unsigned(i-guard);unsigned valid=1u;
        const unsigned extent=row<rows ? start+row/16u+1u : tokens;
        for(unsigned key=0u;key<extent;++key) {
            const uint16_t word=row<rows ? probability[guard+size_t(row)*tokens+key] :
                value[guard+size_t(key)*512u+row-rows];
            valid &= unsigned(qrt_sm121_float_alignment::eligible(word));
        }
        if(metadata[i]!=valid) throw std::runtime_error("eligibility differs from CPU row scan");
    }
    for(unsigned variant:{0u,1u,4u}) {
    upload(dout,initial); upload(da,acc); upload(dd,den); upload(di,scratch);
    check(hipError_t(launch_compacted_pv_replay(dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,
        dout.as<float>()+guard,start,queries,output_start,tokens,nullptr,da.as<float>()+guard,dd.as<float>()+guard,
        de.as<float>()+guard,di.as<unsigned>()+guard,di.as<unsigned>()+guard+cells,nullptr,nullptr,
        transposed_value ? dtv.as<uint16_t>()+guard : nullptr,transposed_value ? tokens : 0u,variant,variant ? flag_data : nullptr)));
    finish();
    const auto actual=download<float>(dout,initial.size()), actual_acc=download<float>(da,acc.size()), actual_den=download<float>(dd,den.size());
    auto indices=download<unsigned>(di,scratch.size());
    if(indices[guard+cells]!=selected) throw std::runtime_error("candidate count mismatch");
    std::vector<bool> seen(cells,false);
    for(unsigned i=0;i<selected;++i) {
        unsigned cell=indices[guard+i];
        if(cell>=cells || seen[cell] || errors[guard+cell]==0.0f) throw std::runtime_error("candidate ownership");
        seen[cell]=true;
    }
    for(unsigned i=0;i<indices.size();++i)
        if((i<guard || (i>=guard+selected && i<guard+cells) || i>guard+cells) && indices[i]!=0xa5a5a5a5u)
            throw std::runtime_error("candidate redzone or unused slot changed");
    unsigned bad=0u;
    for(size_t i=0;i<actual.size();++i) {
        bad += std::memcmp(&actual[i],&control[i],4u)!=0;
        bad += std::memcmp(&actual_acc[i],&control_acc[i],4u)!=0;
        const bool live=i>=guard+output_start*kQueryHeads*kHeadDim && i<guard+output_start*kQueryHeads*kHeadDim+cells;
        if(!live && (actual[i]!=12345.0f || actual_acc[i]!=12345.0f)) throw std::runtime_error("output redzone");
    }
    for(size_t i=0;i<actual_den.size();++i) bad += std::memcmp(&actual_den[i],&control_den[i],4u)!=0;
    if(bad) throw std::runtime_error("float PV output/accumulator/denominator differs from original per-query replay");
    std::printf("{\"kind\":\"prevalidated_float_pv\",\"query_start\":%u,\"queries\":%u,\"mode\":%u,\"input_mode\":%u,\"lanes\":%u,\"transposed_value\":%u,\"cells\":%u,\"selected_cells\":%u,\"raw_bit_mismatches\":%u,\"metadata_words\":%zu,\"metadata_cpu_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"unique_candidates\":true,\"inference_acceptance\":false}\n",start,queries,mode,input_mode,variant,unsigned(transposed_value),cells,selected,bad,flag_words);
    }
    immutable(dv,value); immutable(dp,probability); immutable(ds,scales); immutable(de,errors);
    immutable(dtv,transposed_value ? transposed : transposed_initial);
    immutable(df,metadata);

}
}
int main() try {
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    if(std::strncmp(properties.gcnArchName,"gfx1151",7u)) throw std::runtime_error("requires gfx1151");
    for(unsigned input_mode:{0u,1u}) for(bool transposed_value : {false,true}) {
    for (auto shape : {std::pair<unsigned,unsigned>{0,1},{31,2},{17,32},{64,3},{8191,1}})
        for(unsigned mode : {0u,1u,2u,3u}) run(shape.first,shape.second,mode,transposed_value,input_mode);
    for (auto shape : {std::pair<unsigned,unsigned>{17,65},{31,128}})
        for(unsigned mode : {0u,1u,2u,3u}) run(shape.first,shape.second,mode,transposed_value,input_mode);
    run(8064u,128u,2u,transposed_value,input_mode);  // Long K, every query, sparse candidates across all slabs.
    }
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"float_pv_selftest_error=%s\n",e.what());return 2; }
