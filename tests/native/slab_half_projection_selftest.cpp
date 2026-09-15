#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_slab_half_projection.h"
#include "strong_float_replay_cases.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace slab=qrt_sm121_slab_half_projection;
namespace layout=qrt_sm121_slab_half_layout;
namespace staged=qrt_sm121_staged_half_projection;
namespace cases=qrt_strong_replay_cases;
constexpr unsigned guard=65u;
struct Result { float value; unsigned transformed,original; };
void check(hipError_t s) { if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s)); }
void require(bool okay,const char* message) { if(!okay)throw std::runtime_error(message); }
struct Device {
    void* pointer=nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&pointer,bytes));check(hipMemset(pointer,0xa5,bytes)); }
    ~Device() { if(pointer)(void)hipFree(pointer); }
    template<class T> T* data() { return static_cast<T*>(pointer)+guard; }
};
template<class T> std::vector<T> read(Device& d,size_t count) {
    std::vector<T> out(count+2u*guard);check(hipMemcpy(out.data(),d.pointer,out.size()*sizeof(T),hipMemcpyDeviceToHost));return out;
}
template<class T> void guards(const std::vector<T>& v) {
    const auto* b=reinterpret_cast<const unsigned char*>(v.data());
    for(size_t i=0u;i<guard*sizeof(T);++i)require(b[i]==0xa5u && b[(v.size()-guard)*sizeof(T)+i]==0xa5u,"redzone changed");
}
void finish() {
    hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) { const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);
        require(std::chrono::steady_clock::now()<deadline,"completion deadline");std::this_thread::yield(); }
    check(hipEventDestroy(event));
}
bool eligible(const uint16_t* row) {
    unsigned lo=255u,hi=0u;bool any=false;
    for(unsigned i=0u;i<16u;++i)if(row[i]&0x7fffu) {
        const unsigned e=(row[i]>>7u)&255u;if(!e || e==255u)return false;
        lo=std::min(lo,e);hi=std::max(hi,e);any=true;
    }
    return !any || hi-lo<=29u;
}
__host__ __device__ unsigned left_row(unsigned row,unsigned rows) { return (row*17u+3u)%rows; }
__host__ __device__ unsigned right_row(unsigned row,unsigned rows) { return (row*37u+11u)%rows; }
template<unsigned Variant,bool Audit>
__global__ void execute(const uint32_t* a,const uint32_t* b,Result* out,uint32_t* trace,
    unsigned left_rows,unsigned right_rows,unsigned outputs,unsigned width) {
    const unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(row>=outputs)return;
    const unsigned l=left_row(row,left_rows),r=right_row(row,right_rows),groups=width/16u;
    staged::Stats stats;float value;
    uint32_t* raw=Audit?trace+size_t(row)*groups*3u:nullptr;
    if constexpr(Variant==0u)value=staged::dot<2u,Audit>(reinterpret_cast<const staged::Row*>(a)+size_t(l)*groups,
        reinterpret_cast<const staged::Row*>(b)+size_t(r)*groups,width,raw,Audit?&stats:nullptr);
    else if constexpr(Variant==1u)value=slab::dot<256u,16u,Audit>(a,b,left_rows,right_rows,l,r,width,raw,Audit?&stats:nullptr);
    else value=slab::dot<64u,64u,Audit>(a,b,left_rows,right_rows,l,r,width,raw,Audit?&stats:nullptr);
    if(!(threadIdx.x&3u))out[row]={value,stats.transformed,stats.original};
}
template<unsigned Rows> void verify_encoding(Device& packed,const std::vector<uint16_t>& input,
    unsigned rows,unsigned width,bool original) {
    const size_t records=original?size_t(rows)*(width/16u):layout::records<Rows>(rows,width);
    const auto words=read<uint32_t>(packed,records*9u);guards(words);
    for(size_t record=0u;record<records;++record) {
        const unsigned row=original?unsigned(record/(width/16u)):layout::row<Rows>(width,record);
        const unsigned group=original?unsigned(record%(width/16u)):layout::group<Rows>(width,record);
        staged::Row expected{},actual{};expected.control=uint16_t(-15);
        if(row<rows && group<width/16u)expected=staged::half::prepare(input.data()+size_t(row)*width+group*16u);
        if(original)std::memcpy(&actual,words.data()+guard+record*9u,sizeof(actual));
        else {std::memcpy(actual.pairs,words.data()+guard+record*8u,32u);actual.control=words[guard+records*8u+record];}
        require(!std::memcmp(&actual,&expected,sizeof(actual)),"encoded operand or padding differs");
        if(row<rows && group<width/16u)for(unsigned i=0u;i<16u;++i)
            require(staged::half::original(actual,i)==input[size_t(row)*width+group*16u+i],"operand roundtrip differs");
    }
}
template<unsigned Variant> void variant(unsigned left_rows,unsigned right_rows,unsigned outputs,unsigned width,
    Device& a,Device& b,const std::vector<uint16_t>& left,const std::vector<uint16_t>& right,
    const std::vector<uint32_t>& expected,const std::vector<unsigned>& transformed) {
    constexpr unsigned LeftRows=Variant==1u?256u:64u,RightRows=Variant==1u?16u:64u;
    const size_t lw=Variant?layout::words<LeftRows>(left_rows,width):left.size()/16u*9u;
    const size_t rw=Variant?layout::words<RightRows>(right_rows,width):right.size()/16u*9u;
    Device pa((lw+2u*guard)*4u),pb((rw+2u*guard)*4u),out((outputs+2u*guard)*sizeof(Result)),trace((expected.size()+2u*guard)*4u);
    if constexpr(Variant) {
        require(slab::prepare<LeftRows>(a.data<uint16_t>(),left.size()-1u,pa.data<uint32_t>(),lw,left_rows,width,nullptr)==hipErrorInvalidValue,"short input accepted");
        require(slab::prepare<LeftRows>(a.data<uint16_t>(),left.size(),pa.data<uint32_t>(),lw-1u,left_rows,width,nullptr)==hipErrorInvalidValue,"short output accepted");
        require(slab::prepare<LeftRows>(nullptr,left.size(),pa.data<uint32_t>(),lw,left_rows,width,nullptr)==hipErrorInvalidValue,"null input accepted");
        check(slab::prepare<LeftRows>(a.data<uint16_t>(),left.size(),pa.data<uint32_t>(),lw,left_rows,width,nullptr));
        check(slab::prepare<RightRows>(b.data<uint16_t>(),right.size(),pb.data<uint32_t>(),rw,right_rows,width,nullptr));
    } else {
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3(unsigned((left.size()/16u+255u)/256u)),dim3(256u),0u,nullptr,a.data<uint16_t>(),reinterpret_cast<staged::Row*>(pa.data<uint32_t>()),left_rows,width);check(hipGetLastError());
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3(unsigned((right.size()/16u+255u)/256u)),dim3(256u),0u,nullptr,b.data<uint16_t>(),reinterpret_cast<staged::Row*>(pb.data<uint32_t>()),right_rows,width);check(hipGetLastError());
    }
    hipLaunchKernelGGL((execute<Variant,true>),dim3((outputs+63u)/64u),dim3(256u),0u,nullptr,pa.data<uint32_t>(),pb.data<uint32_t>(),out.data<Result>(),trace.data<uint32_t>(),left_rows,right_rows,outputs,width);check(hipGetLastError());finish();
    const auto values=read<Result>(out,outputs);const auto actual=read<uint32_t>(trace,expected.size());guards(values);guards(actual);
    require(!std::memcmp(actual.data()+guard,expected.data(),expected.size()*4u),"ordered K16 carry differs from CPU original arithmetic");
    unsigned floating=0u,original=0u;
    for(unsigned row=0u;row<outputs;++row) {
        const size_t base=((size_t(row)+1u)*(width/16u)-1u)*3u;
        const qrt_q1_moe_hawkeye::Value carry{expected[base],int16_t(int32_t(expected[base+1u])),expected[base+2u]!=0u};
        const float ref=qrt_q1_moe_hawkeye::value_to_float(qrt_q1_moe_hawkeye::group_sum<26,-133>(&carry,1u));
        require(!std::memcmp(&ref,&values[guard+row].value,4u),"final endpoint differs");
        require(values[guard+row].transformed==transformed[row] && values[guard+row].original==width/16u-transformed[row],"arithmetic path count differs");
        floating+=values[guard+row].transformed;original+=values[guard+row].original;
    }
    check(hipMemset(out.pointer,0xa5,(outputs+2u*guard)*sizeof(Result)));
    hipLaunchKernelGGL((execute<Variant,false>),dim3((outputs+63u)/64u),dim3(256u),0u,nullptr,pa.data<uint32_t>(),pb.data<uint32_t>(),out.data<Result>(),trace.data<uint32_t>(),left_rows,right_rows,outputs,width);check(hipGetLastError());finish();
    const auto production=read<Result>(out,outputs);guards(production);
    for(unsigned row=0u;row<outputs;++row)require(!std::memcmp(&production[guard+row].value,&values[guard+row].value,4u),"production and audit differ");
    require(read<uint32_t>(trace,expected.size())==actual,"production modified audit trace");
    verify_encoding<LeftRows>(pa,left,left_rows,width,Variant==0u);verify_encoding<RightRows>(pb,right,right_rows,width,Variant==0u);
    const auto after_left=read<uint16_t>(a,left.size()),after_right=read<uint16_t>(b,right.size());guards(after_left);guards(after_right);
    require(!std::memcmp(after_left.data()+guard,left.data(),left.size()*2u) && !std::memcmp(after_right.data()+guard,right.data(),right.size()*2u),"original operands changed");
    require(floating && original,"missing numerical path coverage");
    std::printf("{\"kind\":\"slab_half_projection_safety\",\"variant\":%u,\"left_rows\":%u,\"right_rows\":%u,\"outputs\":%u,\"width\":%u,\"ordered_raw_carry_states\":%zu,\"transformed_groups\":%u,\"original_groups\":%u,\"raw_bit_mismatches\":0,\"production_diagnostic_parity\":true,\"all_encoded_words_and_padding_checked\":true,\"unaligned_operands\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",Variant,left_rows,right_rows,outputs,width,expected.size()/3u,floating,original);std::fflush(stdout);
}
void run(unsigned rows,unsigned width) {
    const unsigned right_rows=rows+13u,outputs=rows+5u;const size_t groups=width/16u;
    std::vector<uint16_t> left(size_t(rows)*width),right(size_t(right_rows)*width);
    for(unsigned row=0u;row<rows;++row)for(unsigned k=0u;k<width;++k)left[size_t(row)*width+k]=cases::input(row,k/16u,k%16u).x;
    for(unsigned row=0u;row<right_rows;++row)for(unsigned k=0u;k<width;++k)right[size_t(row)*width+k]=cases::input(row+7u,k/16u,k%16u).y;
    std::vector<uint32_t> expected(size_t(outputs)*groups*3u);std::vector<unsigned> transformed(outputs,0u);
    for(unsigned row=0u;row<outputs;++row) {
        const auto* l=left.data()+size_t(left_row(row,rows))*width;const auto* r=right.data()+size_t(right_row(row,right_rows))*width;
        qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
        for(unsigned group=0u;group<groups;++group) {
            qrt_q1_moe_hawkeye::Value terms[17];terms[0]=carry;
            for(unsigned i=0u;i<16u;++i)terms[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(l[group*16u+i],r[group*16u+i],-133);
            transformed[row]+=eligible(l+group*16u)&&eligible(r+group*16u);carry=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);
            const size_t index=(size_t(row)*groups+group)*3u;
            expected[index]=carry.significand;expected[index+1u]=uint32_t(int32_t(carry.exponent));expected[index+2u]=unsigned(carry.negative);
        }
    }
    Device a((left.size()+2u*guard)*2u),b((right.size()+2u*guard)*2u);
    check(hipMemcpy(a.data<uint16_t>(),left.data(),left.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(b.data<uint16_t>(),right.data(),right.size()*2u,hipMemcpyHostToDevice));
    variant<0u>(rows,right_rows,outputs,width,a,b,left,right,expected,transformed);
    variant<1u>(rows,right_rows,outputs,width,a,b,left,right,expected,transformed);
    variant<2u>(rows,right_rows,outputs,width,a,b,left,right,expected,transformed);
}
int main() try {
    hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
    run(257u,16u);run(4096u,272u);run(2048u,2048u);run(1024u,4096u);run(129u,4112u);run(129u,8192u);return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
