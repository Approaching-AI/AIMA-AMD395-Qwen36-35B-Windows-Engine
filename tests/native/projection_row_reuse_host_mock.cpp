#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <thread>
#include <vector>
#include "projection_row_identity.h"
using hipStream_t=void*;using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorNotReady=2,hipErrorLaunchTimeOut=3,
    hipErrorOutOfMemory=4,hipMemcpyDeviceToHost=5,hipMemcpyHostToDevice=6,error=99;
struct dim3 {explicit dim3(unsigned value):x(value){} unsigned x;};
std::map<uintptr_t,size_t> allocations;
std::vector<std::function<void()>> pending;
unsigned operation=0,failed=0,allocated=0,freed=0,fingerprints=0,verifications=0;
bool collide=false;
bool fail(){return ++operation==failed;}
void range(const void* p,size_t size) {
    const uintptr_t address=reinterpret_cast<uintptr_t>(p);auto it=allocations.upper_bound(address);
    assert(it!=allocations.begin());--it;
    assert(address>=it->first&&size<=it->second&&address-it->first<=it->second-size);
}
void drain(){auto work=std::move(pending);pending.clear();for(auto& job:work)job();}
int hipMalloc(void** p,size_t size){if(fail())return error;*p=std::malloc(size);assert(*p);allocations.emplace(reinterpret_cast<uintptr_t>(*p),size);++allocated;return 0;}
int hipMemsetAsync(void* p,int value,size_t size,hipStream_t){range(p,size);if(fail())return error;pending.push_back([=]{std::memset(p,value,size);});return 0;}
int hipMemcpy(void* dst,const void* src,size_t size,int kind){assert(kind==hipMemcpyDeviceToHost&&pending.empty());range(src,size);if(fail())return error;std::memcpy(dst,src,size);return 0;}
int hipMemcpyAsync(void* dst,const void* src,size_t size,int kind,hipStream_t){assert(kind==hipMemcpyHostToDevice);range(dst,size);if(fail())return error;pending.push_back([=]{std::memcpy(dst,src,size);});return 0;}
int hipGetLastError(){return fail()?error:0;}
int hipStreamQuery(hipStream_t){if(fail())return error;drain();return 0;}
int hipStreamSynchronize(hipStream_t){drain();return fail()?error:0;}
int hipFree(void* p){assert(pending.empty());assert(allocations.erase(reinterpret_cast<uintptr_t>(p))==1);std::free(p);++freed;return fail()?error:0;}
#define hipLaunchKernelGGL(kernel,blocks,threads,shared,stream,...) \
    do {assert((blocks).x==8192u&&(threads).x==256u);pending.push_back([=]{kernel(__VA_ARGS__);});} while(0)
namespace qrt_projection_row_reuse_audit {
void fingerprint(const uint16_t* input,unsigned width,qrt_projection_row_identity::Key* output) {
    ++fingerprints;range(output,8192u*sizeof(*output));
    for(unsigned token=0;token<8192u;++token) {
        qrt_projection_row_identity::Key key;
        for(unsigned k=0;k<width;++k){const auto word=qrt_projection_row_identity::word(input[size_t(token)*width+k],k);key.first^=word.first;key.second+=word.second;}
        output[token]=key;
    }
    if(collide)output[8191u]=output[8190u];
}
void verify(const uint16_t* input,unsigned width,const unsigned* canonical,unsigned* differences) {
    ++verifications;range(canonical,8192u*4u);range(differences,8192u*4u);
    for(unsigned token=0;token<8192u;++token) {
        assert(canonical[token]<=token);unsigned count=0;
        for(unsigned k=0;k<width;++k)count+=input[size_t(token)*width+k]!=input[size_t(canonical[token])*width+k];
        differences[token]=count;
    }
}
}
#include "row_reuse_actual_owner.h"
void setting(const char* value) {
#ifdef _WIN32
    _putenv_s("QRT_QWEN36_Q8192_ROW_REUSE_AUDIT",value?value:"");
#else
    if(value)setenv("QRT_QWEN36_Q8192_ROW_REUSE_AUDIT",value,1);
    else unsetenv("QRT_QWEN36_Q8192_ROW_REUSE_AUDIT");
#endif
}
void reset(unsigned failure=0){assert(allocations.empty()&&pending.empty());operation=allocated=freed=fingerprints=verifications=0;failed=failure;setting("1");}
int main() {
    std::vector<uint16_t> input(8192u*16u);
    for(unsigned token=0;token<8192u;++token)for(unsigned k=0;k<16u;++k)
        input[token*16u+k]=uint16_t((token/4u)*17u+k*7919u);
    const auto original=input;
    auto run=[&](unsigned tokens=8192u,unsigned width=16u,const uint16_t* source=nullptr){return qrt_projection_row_reuse_audit::run(source?source:input.data(),2048u,tokens,width,nullptr);};
    reset();assert(run()==0&&allocated==3&&freed==3&&fingerprints==1&&verifications==1);
    const unsigned operations=operation;
    for(unsigned failure=1;failure<=operations;++failure){reset(failure);assert(run()!=0);assert(input==original&&allocated==freed&&allocations.empty()&&pending.empty());}
    // Deliberately collide unequal rows. The full original-word comparison
    // must remove that hash match from the reusable population.
    collide=true;input.back()^=1u;reset();assert(run()==0&&allocated==freed);assert(input.back()==uint16_t(original.back()^1u));
    collide=false;input=original;
    for(const char* off:{static_cast<const char*>(nullptr),"","0"}){reset();setting(off);assert(run(8192u,0u)==0&&!operation);}
    for(const char* bad:{"2","1x"," 1"}){reset();setting(bad);assert(run()==hipErrorInvalidValue&&!operation);}
    reset();assert(run(7169u,0u)==0&&!operation);assert(run(8192u,0u)==hipErrorInvalidValue&&!operation);assert(run(8192u,4097u)==hipErrorInvalidValue&&!operation);
    assert(qrt_projection_row_reuse_audit::run(nullptr,2048u,8192u,16u,nullptr)==hipErrorInvalidValue&&!operation);
    assert(input==original);
    std::printf("row_reuse_owner_pass tokens=8192 input_words=131072 verified_duplicate_rows=6144 deliberate_collision_rows=1 injected_failures=%u\n",operations);
}
