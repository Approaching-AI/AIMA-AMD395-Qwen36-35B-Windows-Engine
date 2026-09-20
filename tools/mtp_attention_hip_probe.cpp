#include <hip/hip_runtime.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native/providers/gdn/sm121_mtp_attention.h"

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file || file.tellg()<=0 || file.tellg()%sizeof(T) || file.tellg()>512u*1024u*1024u)
        throw std::runtime_error("input file bound");
    const size_t bytes=static_cast<size_t>(file.tellg());std::vector<T> values(bytes/sizeof(T));
    file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),bytes);
    if (!file) throw std::runtime_error("short input");return values;
}
struct Scratch {
    std::vector<void*> pointers;
    ~Scratch() {
        if (hipDeviceSynchronize()!=hipSuccess) return;
        for(void* pointer:pointers)(void)hipFree(pointer);
    }
    template<class T> T* upload(const std::vector<T>& values) {
        T* pointer=nullptr;check(hipMalloc(reinterpret_cast<void**>(&pointer),values.size()*sizeof(T)));
        pointers.push_back(pointer);
        check(hipMemcpy(pointer,values.data(),values.size()*sizeof(T),hipMemcpyHostToDevice));return pointer;
    }
};
int main(int argc,char** argv) try {
    if(argc!=8)throw std::runtime_error("q k v expected positions exp2 rcp");
    const auto query=read_file<uint16_t>(argv[1]),key=read_file<uint16_t>(argv[2]),value=read_file<uint16_t>(argv[3]);
    const auto expected=read_file<uint16_t>(argv[4]);const auto positions=read_file<uint32_t>(argv[5]);
    const auto exp2=read_file<unsigned char>(argv[6]),rcp=read_file<unsigned char>(argv[7]);
    const size_t rows=positions.size();const unsigned tokens=static_cast<unsigned>(key.size()/512u);
    if(!rows || rows>8192u || query.size()!=rows*4096u || expected.size()!=query.size() ||
       !tokens || tokens>262144u || key.size()!=size_t(tokens)*512u || value.size()!=key.size() ||
       !qrt_sm121_exp2::valid_layout(exp2.data(),exp2.size()) || !qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))
        throw std::runtime_error("shape or table");
    for(size_t i=0;i<rows;++i)
        if(positions[i]>=tokens || (i && positions[i]<=positions[i-1]))throw std::runtime_error("position order");
    constexpr size_t guard=64u;constexpr uint16_t sentinel=0xa5a5u;constexpr uint32_t float_sentinel=0x7fc12345u;
    std::vector<uint16_t> guarded_query(query.size()+2u*guard,sentinel),cache(size_t(tokens)*1024u+2u*guard,sentinel);
    std::copy(query.begin(),query.end(),guarded_query.begin()+guard);
    for(unsigned token=0;token<tokens;++token) {
        std::copy_n(key.begin()+size_t(token)*512u,512u,cache.begin()+guard+size_t(token)*1024u);
        std::copy_n(value.begin()+size_t(token)*512u,512u,cache.begin()+guard+size_t(token)*1024u+512u);
    }
    const unsigned stride=(tokens+31u)&~31u;const size_t score_elements=size_t(2u)*16u*stride;
    std::vector<uint32_t> scores(score_elements+2u*guard,float_sentinel),context(8192u+2u*guard,float_sentinel);
    std::vector<uint16_t> output(expected.size()+2u*guard,sentinel);
    Scratch scratch;auto* dq=scratch.upload(guarded_query);auto* dc=scratch.upload(cache);
    auto* ds=scratch.upload(scores);auto* df=scratch.upload(context);auto* dy=scratch.upload(output);
    auto* de=scratch.upload(exp2);auto* dr=scratch.upload(rcp);
    const auto launch=[&](size_t row,unsigned position,unsigned count,unsigned score_stride,unsigned retained) {
        return qrt_sm121_mtp::launch_attention(dq+guard+row*4096u,dc+guard,retained,position,count,de,dr,
            reinterpret_cast<float*>(ds+guard),score_stride,reinterpret_cast<float*>(df+guard),dy+guard+row*4096u);
    };
    unsigned invalid_cases=0;
    const auto reject=[&](hipError_t status) {
        if(status!=hipErrorInvalidValue)throw std::runtime_error("invalid attention launch accepted");++invalid_cases;
    };
    reject(launch(0,0,0,stride,tokens));reject(launch(0,0,3,stride,tokens));
    reject(launch(0,tokens,1,stride,tokens));reject(launch(0,tokens-1u,2,stride,tokens));
    reject(launch(0,0,1,stride,0));reject(launch(0,0,1,stride,262145u));
    reject(launch(0,0,1,0,tokens));reject(launch(0,0,1,33,tokens));
    reject(launch(0,0,1,262176u,tokens));
    size_t mismatches=0,guard_bad=0,unused_bad=0,mask_bad=0;unsigned launches=0;
    for(unsigned maximum_rows:{1u,2u}) {
        std::fill(output.begin(),output.end(),sentinel);
        check(hipMemcpy(dy,output.data(),output.size()*2u,hipMemcpyHostToDevice));
        for(size_t first=0;first<rows;) {
            unsigned count=1u;
            if(maximum_rows==2u && first+1u<rows && positions[first+1u]==positions[first]+1u)count=2u;
            std::fill(scores.begin(),scores.end(),float_sentinel);std::fill(context.begin(),context.end(),float_sentinel);
            check(hipMemcpy(ds,scores.data(),scores.size()*4u,hipMemcpyHostToDevice));
            check(hipMemcpy(df,context.data(),context.size()*4u,hipMemcpyHostToDevice));
            check(launch(first,positions[first],count,stride,tokens));++launches;
            check(hipDeviceSynchronize());
            check(hipMemcpy(scores.data(),ds,scores.size()*4u,hipMemcpyDeviceToHost));
            check(hipMemcpy(context.data(),df,context.size()*4u,hipMemcpyDeviceToHost));
            for(const auto* values:{&scores,&context})for(size_t i=0;i<guard;++i) {
                guard_bad+=(*values)[i]!=float_sentinel;guard_bad+=(*values)[values->size()-1u-i]!=float_sentinel;
            }
            for(size_t i=size_t(count)*16u*stride;i<score_elements;++i)unused_bad+=scores[guard+i]!=float_sentinel;
            for(size_t i=size_t(count)*4096u;i<8192u;++i)unused_bad+=context[guard+i]!=float_sentinel;
            for(unsigned row=0;row<count;++row)for(unsigned head=0;head<16u;++head)
                for(unsigned token=positions[first]+row+1u;token<stride;++token)
                    mask_bad+=scores[guard+(size_t(row)*16u+head)*stride+token]!=0xff800000u;
            first+=count;
        }
        check(hipMemcpy(output.data(),dy,output.size()*2u,hipMemcpyDeviceToHost));
        for(size_t i=0;i<expected.size();++i)mismatches+=output[guard+i]!=expected[i];
        for(size_t i=0;i<guard;++i){guard_bad+=output[i]!=sentinel;guard_bad+=output[output.size()-1u-i]!=sentinel;}
    }
    const auto changed=[&](const auto* device,const auto& before) {
        auto after=before;size_t bad=0;
        check(hipMemcpy(after.data(),device,after.size()*sizeof(after[0]),hipMemcpyDeviceToHost));
        for(size_t i=0;i<after.size();++i)bad+=after[i]!=before[i];return bad;
    };
    const size_t query_bad=changed(dq,guarded_query),cache_bad=changed(dc,cache);
    const size_t table_bad=changed(de,exp2)+changed(dr,rcp);
    const bool passed=!mismatches&&!guard_bad&&!unused_bad&&!mask_bad&&!query_bad&&!cache_bad&&!table_bad;
    std::cout<<"{\"kind\":\"original_mtp_attention_native_replay\",\"rows\":"<<rows
        <<",\"cache_tokens\":"<<tokens<<",\"cache_layout\":\"token_K512_V512\",\"launches\":"<<launches
        <<",\"row_batch_configurations\":2,\"elements_per_configuration\":"<<expected.size()
        <<",\"compared_elements\":"<<expected.size()*2u<<",\"bf16_mismatches\":"<<mismatches
        <<",\"guard_mismatches\":"<<guard_bad<<",\"unused_scratch_mismatches\":"<<unused_bad
        <<",\"causal_mask_mismatches\":"<<mask_bad<<",\"query_mismatches\":"<<query_bad
        <<",\"cache_mismatches\":"<<cache_bad<<",\"table_mismatches\":"<<table_bad
        <<",\"invalid_launch_cases\":"<<invalid_cases<<",\"workspace_bytes\":"
        <<qrt_sm121_mtp::attention_workspace_bytes(2u,stride)<<",\"passed\":"<<(passed?"true":"false")
        <<",\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return passed?0:1;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 2;}
