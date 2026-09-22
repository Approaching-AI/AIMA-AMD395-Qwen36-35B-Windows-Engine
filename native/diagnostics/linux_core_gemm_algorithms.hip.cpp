// SPDX-License-Identifier: Apache-2.0
// Full q8192 geometry, matching the port's column-major BF16/F32 descriptors.
// Integer-valued reference products make every mathematical output exact.
// Preserve all FP32 differences and test the existing producer-selector bound;
// this is an algorithm/ABI diagnostic, never a model correctness oracle.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include "gemm_probe_identity.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>

namespace {
constexpr unsigned m = 8192, guard = 256, rounds = 3;
constexpr std::size_t workspace_limit = 128ULL << 20;
using Clock = std::chrono::steady_clock;
void hip(hipError_t e, const char* s) {
  if (e != hipSuccess) throw std::runtime_error(std::string(s) + ": " + hipGetErrorString(e));
}
void blas(hipblasStatus_t e, const char* s) {
  if (e != HIPBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(s) + ": " + std::to_string(int(e)));
}
struct Buffer {
  void* pointer = nullptr;
  explicit Buffer(std::size_t bytes) { hip(hipMalloc(&pointer,bytes),"allocation"); }
  ~Buffer() { if(pointer)hipFree(pointer); }
  Buffer(const Buffer&)=delete;
  template<class T>T* data()const{return static_cast<T*>(pointer);}
};
__host__ __device__ int input_word(unsigned row,unsigned column,bool weight) {
  return weight ? int((row*5u+column*11u)%29u)-14 : int((row*13u+column*7u)%31u)-15;
}
__global__ void fill_input(uint16_t* all,unsigned rows,unsigned width,bool weight) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x, count=rows*width;
  if(i>=count+2u*guard)return;
  if(i<guard || i>=guard+count){all[i]=0x5a7bu;return;}
  const unsigned cell=i-guard;
  all[i]=uint16_t(__float_as_uint(float(input_word(cell/width,cell%width,weight))*0.0625f)>>16u);
}
__global__ void verify_input(const uint16_t* all,unsigned rows,unsigned width,bool weight,unsigned* failures) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x, count=rows*width;
  if(i>=count+2u*guard)return;
  uint16_t expected=0x5a7bu;
  if(i>=guard && i<guard+count) {
    const unsigned cell=i-guard;
    expected=uint16_t(__float_as_uint(float(input_word(cell/width,cell%width,weight))*0.0625f)>>16u);
  }
  if(all[i]!=expected)atomicAdd(failures+1,1u);
}
__global__ void fill_output(float* all,unsigned count) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count+2u*guard)all[i]=__uint_as_float(0x4a5a7b00u);
}
__global__ void verify_output(const float* all,const int* reference,const float* allowances,unsigned n,unsigned* failures) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x, count=m*n;
  if(i>=count+2u*guard)return;
  if(i<guard || i>=guard+count) {
    if(__float_as_uint(all[i])!=0x4a5a7b00u)atomicAdd(failures+2,1u);
  } else {
    const unsigned cell=i-guard;
    const unsigned which=((cell/n)%31u)*29u+(cell%n)%29u;
    const float expected=float(reference[which])*0.00390625f;
    const float error=fabsf(all[i]-expected);
    if(!(error<=allowances[which]))atomicAdd(failures,1u);
    if(all[i]!=expected)atomicAdd(failures+3,1u);
    atomicMax(failures+4,__float_as_uint(error));
  }
}
struct Plan {
  hipblasLtHandle_t handle=nullptr;
  hipblasLtMatmulDesc_t operation=nullptr;
  hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;
  hipblasLtMatmulPreference_t preference=nullptr;
  Plan(unsigned n,unsigned k) {
    blas(hipblasLtCreate(&handle),"create");
    blas(hipblasLtMatmulDescCreate(&operation,HIPBLAS_COMPUTE_32F,HIP_R_32F),"operation");
    const hipblasOperation_t transpose=HIPBLAS_OP_T;
    blas(hipblasLtMatmulDescSetAttribute(operation,HIPBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)),"transpose");
    blas(hipblasLtMatrixLayoutCreate(&a,HIP_R_16BF,k,n,k),"weight layout");
    blas(hipblasLtMatrixLayoutCreate(&b,HIP_R_16BF,k,m,k),"input layout");
    blas(hipblasLtMatrixLayoutCreate(&c,HIP_R_32F,n,m,n),"C layout");
    blas(hipblasLtMatrixLayoutCreate(&d,HIP_R_32F,n,m,n),"D layout");
    blas(hipblasLtMatmulPreferenceCreate(&preference),"preference");
    blas(hipblasLtMatmulPreferenceSetAttribute(preference,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit,sizeof(workspace_limit)),"workspace limit");
  }
  ~Plan() {
    if(preference)hipblasLtMatmulPreferenceDestroy(preference);
    if(d)hipblasLtMatrixLayoutDestroy(d);if(c)hipblasLtMatrixLayoutDestroy(c);
    if(b)hipblasLtMatrixLayoutDestroy(b);if(a)hipblasLtMatrixLayoutDestroy(a);
    if(operation)hipblasLtMatmulDescDestroy(operation);if(handle)hipblasLtDestroy(handle);
  }
};
std::string identity(const hipblasLtMatmulAlgo_t& algorithm) {
  const auto* bytes=reinterpret_cast<const unsigned char*>(&algorithm);
  const char* hex="0123456789abcdef";std::string result;
  for(std::size_t i=0;i<sizeof(algorithm);++i){result+=hex[bytes[i]>>4];result+=hex[bytes[i]&15];}
  return result;
}
void sweep(unsigned n,unsigned k) {
  Plan plan(n,k);
  std::array<hipblasLtMatmulHeuristicResult_t,32> choices{};int count=0,version=0;
  blas(hipblasLtGetVersion(plan.handle,&version),"version");
  blas(hipblasLtMatmulAlgoGetHeuristic(plan.handle,plan.operation,plan.a,plan.b,plan.c,plan.d,
      plan.preference,int(choices.size()),choices.data(),&count),"heuristics");
  if(count<0 || count>int(choices.size()))throw std::runtime_error("Invalid heuristic count");
  std::printf("{\"event\":\"shape\",\"m\":%u,\"n\":%u,\"k\":%u,\"heuristics\":%d,\"library_version\":%d,\"workspace_limit\":%zu}\n",
      m,n,k,count,version,workspace_limit);std::fflush(stdout);
  if(!count)return;
  Buffer x((std::size_t(m)*k+2u*guard)*2u),w((std::size_t(n)*k+2u*guard)*2u);
  Buffer y((std::size_t(m)*n+2u*guard)*4u),work(workspace_limit),reference(31u*29u*4u),bounds(31u*29u*4u),bad(5u*4u);
  std::array<int,31u*29u> expected{};
  std::array<float,31u*29u> allowances{};
  for(unsigned t=0;t<31u;++t)for(unsigned row=0;row<29u;++row) {
    int left_norm=0,right_norm=0;
    for(unsigned column=0;column<k;++column) {
      const int left=input_word(t,column,false),right=input_word(row,column,true);
      expected[t*29u+row]+=left*right;left_norm+=left*left;right_norm+=right*right;
    }
    // Same 1000-ppb L2 admission radius as the port's input projections.
    // This diagnoses a producer; it does not establish a universal error
    // proof or replace the complete model's unchanged GB10 boundary.
    allowances[t*29u+row]=std::nextafter(float(std::sqrt(double(left_norm)*right_norm)*0.00390625*1.e-6),
                                        std::numeric_limits<float>::infinity());
  }
  hip(hipMemcpy(reference.pointer,expected.data(),sizeof(expected),hipMemcpyHostToDevice),"reference upload");
  hip(hipMemcpy(bounds.pointer,allowances.data(),sizeof(allowances),hipMemcpyHostToDevice),"bound upload");
  hipLaunchKernelGGL(fill_input,dim3((m*k+2u*guard+255u)/256u),dim3(256),0,nullptr,x.data<uint16_t>(),m,k,false);
  hip(hipGetLastError(),"fill input");
  hipLaunchKernelGGL(fill_input,dim3((n*k+2u*guard+255u)/256u),dim3(256),0,nullptr,w.data<uint16_t>(),n,k,true);
  hip(hipGetLastError(),"fill weight");hip(hipDeviceSynchronize(),"input drain");
  auto cpu_input_check=[&](const Buffer& buffer,unsigned rows,bool weight) {
    std::vector<uint16_t> actual(std::size_t(rows)*k+2u*guard);
    hip(hipMemcpy(actual.data(),buffer.pointer,actual.size()*2u,hipMemcpyDeviceToHost),"CPU input control");
    for(std::size_t i=0;i<actual.size();++i) {
      uint16_t wanted=0x5a7bu;
      if(i>=guard && i<actual.size()-guard) {
        const auto cell=i-guard;const unsigned period=weight?29u:31u;
        const int integer=int((((cell/k)%period)*(weight?5u:13u)+((cell%k)%period)*(weight?11u:7u))%period)-(weight?14:15);
        const float value=float(integer)/16.0f;uint32_t bits=0;std::memcpy(&bits,&value,4u);wanted=uint16_t(bits>>16u);
      }
      if(actual[i]!=wanted)throw std::runtime_error("Independent CPU input control differs");
    }
  };
  cpu_input_check(x,m,false);cpu_input_check(w,n,true);
  std::printf("{\"event\":\"cpu_input_control\",\"n\":%u,\"k\":%u,\"verified_elements\":%zu}\n",n,k,(std::size_t(m)+n)*k+4u*guard);
  unsigned completed=0;const auto deadline=Clock::now()+std::chrono::seconds(120);
  for(int index=0;index<count;++index) {
    if(Clock::now()>deadline)throw std::runtime_error("Shape deadline exceeded");
    auto& choice=choices[index];
    if(choice.state!=HIPBLAS_STATUS_SUCCESS || choice.workspaceSize>workspace_limit) {
      std::printf("{\"event\":\"skipped\",\"n\":%u,\"k\":%u,\"index\":%d,\"state\":%d,\"workspace\":%zu}\n",n,k,index,int(choice.state),choice.workspaceSize);
      continue;
    }
    const float alpha=1,beta=0;
    auto launch=[&]{blas(hipblasLtMatmul(plan.handle,plan.operation,&alpha,w.data<uint16_t>()+guard,plan.a,
        x.data<uint16_t>()+guard,plan.b,&beta,y.data<float>()+guard,plan.c,y.data<float>()+guard,plan.d,
        &choice.algo,work.pointer,choice.workspaceSize,nullptr),"matmul");hip(hipDeviceSynchronize(),"matmul drain");};
    hipLaunchKernelGGL(fill_output,dim3((m*n+2u*guard+255u)/256u),dim3(256),0,nullptr,y.data<float>(),m*n);
    hip(hipGetLastError(),"fill output");
    std::printf("{\"event\":\"algorithm_start\",\"n\":%u,\"k\":%u,\"index\":%d}\n",n,k,index);std::fflush(stdout);
    launch();
    auto validate=[&]{
      hip(hipMemset(bad.pointer,0,20u),"clear failures");
      hipLaunchKernelGGL(verify_output,dim3((m*n+2u*guard+255u)/256u),dim3(256),0,nullptr,y.data<float>(),reference.data<int>(),bounds.data<float>(),n,bad.data<unsigned>());
      hip(hipGetLastError(),"verify output");
      hipLaunchKernelGGL(verify_input,dim3((m*k+2u*guard+255u)/256u),dim3(256),0,nullptr,x.data<uint16_t>(),m,k,false,bad.data<unsigned>());
      hip(hipGetLastError(),"verify input");
      hipLaunchKernelGGL(verify_input,dim3((n*k+2u*guard+255u)/256u),dim3(256),0,nullptr,w.data<uint16_t>(),n,k,true,bad.data<unsigned>());
      hip(hipGetLastError(),"verify weight");
      std::array<unsigned,5> failures{};
      hip(hipMemcpy(failures.data(),bad.pointer,20u,hipMemcpyDeviceToHost),"failure read");return failures;
    };
    const auto first=validate();
    if(first[0] || first[1] || first[2]) {
      std::printf("{\"event\":\"validation_failure\",\"n\":%u,\"k\":%u,\"index\":%d,\"out_of_bound_cells\":%u,\"input_mismatches\":%u,\"guard_mismatches\":%u}\n",
          n,k,index,first[0],first[1],first[2]);
      for(unsigned t:{0u,1u,30u,31u,m-1u})for(unsigned row:{0u,1u,28u,29u,n-1u}) {
        float actual=0;hip(hipMemcpy(&actual,y.data<float>()+guard+std::size_t(t)*n+row,4u,hipMemcpyDeviceToHost),"failure sample");
        std::printf("{\"event\":\"failure_sample\",\"token\":%u,\"row\":%u,\"actual\":%.9g,\"expected\":%.9g}\n",
            t,row,double(actual),double(expected[(t%31u)*29u+row%29u])*0.00390625);
      }
      std::fflush(stdout);
      throw std::runtime_error("Algorithm exceeds producer bound or changed inputs/guards");
    }
    // Positive controls demonstrate that the numerical and guard checkers
    // detect independent corruptions; restore the real output before timing.
    if(!completed) {
      const float poison=123456.0f;
      hip(hipMemcpy(y.data<float>()+guard,&poison,4u,hipMemcpyHostToDevice),"inject output control");
      hip(hipMemcpy(y.data<float>(),&poison,4u,hipMemcpyHostToDevice),"inject guard control");
      const auto control=validate();
      if(control[0]!=1 || control[1] || control[2]!=1)throw std::runtime_error("Numerical checker control failed");
      hipLaunchKernelGGL(fill_output,dim3((m*n+2u*guard+255u)/256u),dim3(256),0,nullptr,y.data<float>(),m*n);
      hip(hipGetLastError(),"restore output");
    }
    launch();std::array<double,rounds> elapsed{};
    for(auto& ms:elapsed) {const auto begin=Clock::now();launch();ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
    const auto last=validate();
    if(last[0] || last[1] || last[2])throw std::runtime_error("Repeated algorithm exceeds producer bound or changed inputs/guards");
    float max_error=0;std::memcpy(&max_error,&last[4],4u);
    auto sorted=elapsed;std::sort(sorted.begin(),sorted.end());
    std::printf("{\"event\":\"algorithm\",\"m\":%u,\"n\":%u,\"k\":%u,\"index\":%d,\"workspace\":%zu,\"algorithm_hex\":\"%s\",\"completed_host_ms\":[%.9f,%.9f,%.9f],\"median_ms\":%.9f,\"verified_output_cells\":%u,\"nonexact_output_cells\":%u,\"max_absolute_error\":%.9g,\"selector_bound_ppb\":1000,\"out_of_bound_cells\":0,\"input_and_guards_unchanged\":true,\"stream_drained\":true}\n",
        m,n,k,index,choice.workspaceSize,identity(choice.algo).c_str(),elapsed[0],elapsed[1],elapsed[2],sorted[1],m*n,last[3],double(max_error));
    std::fflush(stdout);++completed;
  }
  if(!completed)throw std::runtime_error("No completed supported algorithm");
}
}
int main() {
  try {
    const char* hostname=std::getenv("COMPUTERNAME");std::string host=hostname?hostname:"";
    std::transform(host.begin(),host.end(),host.begin(),[](unsigned char c){return char(std::tolower(c));});
    if(host!="baiying")throw std::runtime_error("Requires native baiying");
    hipDeviceProp_t device{};hip(hipGetDeviceProperties(&device,0),"device properties");
    if(std::string(device.gcnArchName).find("gfx1151")!=0)throw std::runtime_error("Requires gfx1151");
    std::printf("{\"event\":\"start\",\"host\":\"baiying\",\"source_commit\":\"%s\",\"model_loaded\":false,\"inference_acceptance\":false}\n",GEMM_PROBE_SOURCE_COMMIT);
    sweep(8192,2048);sweep(4096,2048);sweep(2048,4096);
    std::puts("{\"type\":\"summary\",\"status\":\"pass\",\"shapes\":3,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}");
    return 0;
  } catch(const std::exception& error) {std::fprintf(stderr,"GEMM algorithm probe failed: %s\n",error.what());return 1;}
}
