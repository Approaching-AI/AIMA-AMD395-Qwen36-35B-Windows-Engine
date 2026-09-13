#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#include <fstream>

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

void run(unsigned start, unsigned queries, unsigned mode, Device& rcp) {
    constexpr unsigned guard=64u, output_start=3u;
    const unsigned tokens=start+queries, rows=queries*kQueryHeads, cells=rows*kHeadDim, tiles=(tokens+31u)/32u;
    std::vector<uint16_t> value(size_t(tokens)*kKvHeads*kHeadDim+2u*guard,0x5a5au);
    std::vector<uint16_t> probability(size_t(rows)*tokens+2u*guard,0x5a5au);
    std::vector<float> scales(size_t(rows)*(tiles+1u)+2u*guard,12345.0f);
    for(size_t i=guard;i+guard<value.size();++i) {
        const unsigned exponent=mode==2u ? 1u+unsigned(i%32u) : 119u+unsigned(i%12u);
        value[i]=uint16_t(((i*977u)&0x807fu)|(exponent<<7u));
        if(mode==1u && i%2u) value[i]=value[i-1u]^0x8000u;
    }
    for(size_t i=guard;i+guard<probability.size();++i)
        probability[i]=uint16_t((i*173u&0x007fu)|((118u+unsigned(i%9u))<<7u));
    for(unsigned row=0;row<rows;++row) {
        for(unsigned t=0;t<tiles;++t)
            scales[guard+size_t(row)*(tiles+1u)+t]=t%7u ? 1.0f : (t%3u ? 0.625f : 0.99609375f);
        scales[guard+size_t(row)*(tiles+1u)+tiles]=1.0f+float(row%31u)/16.0f;
    }
    const size_t out_cells=size_t(output_start+queries)*kQueryHeads*kHeadDim+2u*guard;
    const size_t den_cells=size_t(output_start+queries)*kQueryHeads+2u*guard;
    std::vector<float> initial(out_cells,12345.0f), den_initial(den_cells,12345.0f), error_initial(cells+2u*guard,12345.0f);
    Device dv(value.size()*2u),dp(probability.size()*2u),ds(scales.size()*4u);
    Device dout(initial.size()*4u),da(initial.size()*4u),dd(den_initial.size()*4u),de(error_initial.size()*4u);
    upload(dv,value);upload(dp,probability);upload(ds,scales);
    std::vector<float> outputs[2],accumulators[2],denominators[2],errors[2];
    for(unsigned final=0;final<2u;++final) {
        upload(dout,initial);upload(da,initial);upload(dd,den_initial);upload(de,error_initial);
        if(final) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_value_kernel<true,false,true,true>),
                dim3(kHeadDim/kIntegerMatrixColumns,kQueryHeads,(queries+15u)/16u),dim3(kThreads),0,nullptr,
                dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,dout.as<float>()+guard,
                start,queries,output_start,tokens,rcp.as<unsigned char>(),da.as<float>()+guard,dd.as<float>()+guard,
                nullptr,nullptr,de.as<float>()+guard);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_mantissa_value_kernel<true,false,true>),
                dim3(kHeadDim/kIntegerMatrixColumns,kQueryHeads,(queries+15u)/16u),dim3(kThreads),0,nullptr,
                dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,dout.as<float>()+guard,
                start,queries,output_start,tokens,rcp.as<unsigned char>(),da.as<float>()+guard,dd.as<float>()+guard,
                nullptr,nullptr,de.as<float>()+guard);
        }
        check(hipGetLastError());finish();
        outputs[final]=download<float>(dout,initial.size());accumulators[final]=download<float>(da,initial.size());
        denominators[final]=download<float>(dd,den_initial.size());errors[final]=download<float>(de,error_initial.size());
    }
    if(std::memcmp(outputs[0].data(),outputs[1].data(),initial.size()*4u) ||
       std::memcmp(accumulators[0].data(),accumulators[1].data(),initial.size()*4u) ||
       std::memcmp(denominators[0].data(),denominators[1].data(),den_initial.size()*4u))
        throw std::runtime_error("native PV arithmetic changed");
    for(size_t i=0;i<initial.size();++i) {
        const bool live=i>=guard+output_start*kQueryHeads*kHeadDim && i<guard+output_start*kQueryHeads*kHeadDim+cells;
        if(!live && (outputs[0][i]!=12345.0f || accumulators[0][i]!=12345.0f)) throw std::runtime_error("output guard changed");
    }
    for(size_t i=0;i<den_initial.size();++i) {
        const bool live=i>=guard+output_start*kQueryHeads && i<guard+output_start*kQueryHeads+rows;
        if(!live && denominators[0][i]!=12345.0f) throw std::runtime_error("denominator guard changed");
    }
    for(size_t i=0;i<error_initial.size();++i) {
        if(i<guard || i>=guard+cells) {
            if(errors[0][i]!=12345.0f || errors[1][i]!=12345.0f) throw std::runtime_error("bound guard changed");
        } else if(!(errors[1][i]>=errors[0][i])) throw std::runtime_error("final bound smaller than old bound");
    }
    upload(dout,initial);
    hipLaunchKernelGGL(blackwell_probability_value_kernel,dim3(kQueryHeads,queries),dim3(kHeadDim),0,nullptr,
        dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,dout.as<float>()+guard,
        start,output_start,tokens,rcp.as<unsigned char>(),nullptr,nullptr,nullptr);
    check(hipGetLastError());finish();const auto exact=download<float>(dout,initial.size());
    std::vector<bool> selected[2];unsigned counts[2]{};unsigned bad=0u;
    for(unsigned final=0;final<2u;++final) {
        std::vector<unsigned> scratch(cells+1u+2u*guard,0xa5a5a5a5u);Device indices(scratch.size()*4u);
        upload(indices,scratch);upload(dout,outputs[final]);upload(de,errors[final]);
        check(hipError_t(launch_compacted_pv_replay(dv.as<uint16_t>()+guard,dp.as<uint16_t>()+guard,ds.as<float>()+guard,
            dout.as<float>()+guard,start,queries,output_start,tokens,rcp.as<unsigned char>(),nullptr,nullptr,
            de.as<float>()+guard,indices.as<unsigned>()+guard,indices.as<unsigned>()+guard+cells,nullptr)));
        finish();auto index=download<unsigned>(indices,scratch.size());auto output=download<float>(dout,initial.size());
        counts[final]=index[guard+cells];if(counts[final]>cells) throw std::runtime_error("invalid candidate count");
        selected[final].resize(cells,false);
        for(unsigned i=0;i<counts[final];++i) {
            const unsigned cell=index[guard+i];if(cell>=cells || selected[final][cell]) throw std::runtime_error("candidate ownership");
            selected[final][cell]=true;
        }
        for(unsigned i=0;i<index.size();++i)
            if((i<guard || (i>=guard+counts[final] && i<guard+cells) || i>guard+cells) && index[i]!=0xa5a5a5a5u)
                throw std::runtime_error("candidate guard changed");
        for(unsigned i=0;i<cells;++i) {
            const size_t offset=guard+output_start*kQueryHeads*kHeadDim+i;
            bad+=qrt_sm121_pv_bound::bf16(output[offset])!=qrt_sm121_pv_bound::bf16(exact[offset]);
        }
    }
    for(unsigned i=0;i<cells;++i) if(selected[0][i] && !selected[1][i]) throw std::runtime_error("candidate set shrank");
    immutable(dv,value);immutable(dp,probability);immutable(ds,scales);
    std::printf("{\"kind\":\"final_pv_kernel\",\"query_start\":%u,\"queries\":%u,\"mode\":%u,\"cells\":%u,\"old_candidates\":%u,\"new_candidates\":%u,\"raw_bit_mismatches\":0,\"underestimates\":0,\"bf16_mismatches\":%u,\"candidate_superset\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        start,queries,mode,cells,counts[0],counts[1],bad);
    if(bad) throw std::runtime_error("selective PV differs from canonical BF16");
}
}
int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("supply SHA-verified reciprocal table");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    if(std::strncmp(properties.gcnArchName,"gfx1151",7u)) throw std::runtime_error("requires gfx1151");
    std::vector<unsigned char> table(qrt_sm121_attention_rcp::table_bytes);
    std::ifstream file(argv[1],std::ios::binary);file.read(reinterpret_cast<char*>(table.data()),table.size());
    if(!file || file.peek()!=EOF || !qrt_sm121_attention_rcp::valid_layout(table.data(),table.size())) throw std::runtime_error("invalid table");
    Device rcp(table.size());upload(rcp,table);
    for(auto shape : {std::pair<unsigned,unsigned>{0,1},{31,2},{17,32},{64,3},{17,65},{31,128},{8191,1},{8064,128}})
        for(unsigned mode=0;mode<3u;++mode) run(shape.first,shape.second,mode,rcp);
    immutable(rcp,table);
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"final_pv_kernel_error=%s\n",e.what());return 2;}
