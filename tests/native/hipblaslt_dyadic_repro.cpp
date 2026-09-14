// Standalone library diagnostic: deliberately includes no engine provider code.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
void require(bool ok,const char* stage){if(!ok)throw std::runtime_error(stage);}
void hip_check(hipError_t s,const char* stage){require(s==hipSuccess,stage);}
void blas_check(hipblasStatus_t s,const char* stage){require(s==HIPBLAS_STATUS_SUCCESS,stage);}
uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
uint16_t bf16(float x){const uint32_t b=bits(x);return uint16_t((b+0x7fffu+((b>>16)&1))>>16);}
constexpr size_t guard=128;
template<class T>struct Device{
 T* base=nullptr;
 explicit Device(const std::vector<T>& v){hip_check(hipMalloc(reinterpret_cast<void**>(&base),v.size()*sizeof(T)),"allocate");hip_check(hipMemcpy(base,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice),"upload");}
 ~Device(){if(base && hipFree(base)!=hipSuccess)std::abort();}
 T* data(){return base+guard;}
 void read(std::vector<T>& v){hip_check(hipMemcpy(v.data(),base,v.size()*sizeof(T),hipMemcpyDeviceToHost),"readback");}
};
void complete(hipStream_t stream,hipEvent_t event){
 hip_check(hipEventRecord(event,stream),"completion record");
 const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
 for(;;){const auto status=hipEventQuery(event);if(status==hipSuccess)break;require(status==hipErrorNotReady && std::chrono::steady_clock::now()<deadline,"completion deadline");std::this_thread::yield();}
}
void module_identity(){
 HMODULE module=GetModuleHandleA("hipblaslt.dll");if(!module)module=GetModuleHandleA("libhipblaslt.dll");
 require(module!=nullptr,"loaded hipBLASLt module");char path[32768];const DWORD n=GetModuleFileNameA(module,path,sizeof(path));require(n && n<sizeof(path),"module path");
 std::cout<<"{\"type\":\"loaded_hipblaslt_module\",\"path\":\"";
 for(unsigned i=0;i<n;i++){if(path[i]=='\\'||path[i]=='\"')std::cout<<'\\';std::cout<<path[i];}
 std::cout<<"\",\"engine_provider_included\":false}"<<std::endl;
}
int main() try{
 hipDeviceProp_t properties{};hip_check(hipGetDeviceProperties(&properties,0),"device");require(std::strncmp(properties.gcnArchName,"gfx1151",7)==0,"requires gfx1151");
 hipblasLtHandle_t handle=nullptr;blas_check(hipblasLtCreate(&handle),"handle");module_identity();
 hipStream_t stream=nullptr;hipEvent_t event=nullptr;hip_check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"stream");hip_check(hipEventCreateWithFlags(&event,hipEventDisableTiming),"event");
 constexpr unsigned tokens=8192;
 for(auto shape:{std::pair{8192u,2048u},std::pair{2048u,4096u}}){
  const unsigned rows=shape.first,k=shape.second;const size_t cells=size_t(rows)*tokens;
  std::vector<uint16_t> weights(size_t(rows)*k+2*guard,0x55aa),inputs(size_t(tokens)*k+2*guard,0x55aa);
  std::vector<float> output(cells+2*guard,16.5f);
  for(unsigned r=0;r<rows;r++)for(unsigned j=0;j<k;j++)weights[guard+size_t(r)*k+j]=bf16(float(int((r*7+j*3)%11)-5)/8.0f);
  for(unsigned t=0;t<tokens;t++)for(unsigned j=0;j<k;j++)inputs[guard+size_t(t)*k+j]=bf16(float(int((t*3+j)%7)-3)/16.0f);
  float reference[7][11]{};
  for(unsigned t=0;t<7;t++)for(unsigned r=0;r<11;r++){int64_t numerator=0;for(unsigned j=0;j<k;j++)numerator+=int64_t(int((r*7+j*3)%11)-5)*(int((t*3+j)%7)-3);reference[t][r]=float(numerator)/128.0f;}
  Device<uint16_t> dw(weights),di(inputs);Device<float> out(output);
  hipblasLtMatmulDesc_t operation=nullptr;hipblasLtMatrixLayout_t wl=nullptr,il=nullptr,ol=nullptr;hipblasLtMatmulPreference_t preference=nullptr;
  blas_check(hipblasLtMatmulDescCreate(&operation,HIPBLAS_COMPUTE_32F,HIP_R_32F),"operation");
  const hipblasOperation_t trans_a=HIPBLAS_OP_T,trans_b=HIPBLAS_OP_N;
  blas_check(hipblasLtMatmulDescSetAttribute(operation,HIPBLASLT_MATMUL_DESC_TRANSA,&trans_a,sizeof(trans_a)),"transposeA");
  blas_check(hipblasLtMatmulDescSetAttribute(operation,HIPBLASLT_MATMUL_DESC_TRANSB,&trans_b,sizeof(trans_b)),"transposeB");
  blas_check(hipblasLtMatrixLayoutCreate(&wl,HIP_R_16BF,k,rows,k),"weight layout");
  blas_check(hipblasLtMatrixLayoutCreate(&il,HIP_R_16BF,k,tokens,k),"input layout");
  blas_check(hipblasLtMatrixLayoutCreate(&ol,HIP_R_32F,rows,tokens,rows),"output layout");
  blas_check(hipblasLtMatmulPreferenceCreate(&preference),"preference");const uint64_t cap=256u*1024u*1024u;
  blas_check(hipblasLtMatmulPreferenceSetAttribute(preference,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cap,sizeof(cap)),"workspace cap");
  std::array<hipblasLtMatmulHeuristicResult_t,16> algorithms{};int returned=0;
  blas_check(hipblasLtMatmulAlgoGetHeuristic(handle,operation,wl,il,ol,ol,preference,16,algorithms.data(),&returned),"heuristics");require(returned>4,"two requested algorithms");
  for(unsigned choice:{0u,4u}){
   const auto& algorithm=algorithms[choice];require(algorithm.state==HIPBLAS_STATUS_SUCCESS && algorithm.workspaceSize<=cap,"algorithm state");void* workspace=nullptr;
   if(algorithm.workspaceSize)hip_check(hipMalloc(&workspace,algorithm.workspaceSize),"workspace");
   std::fill(output.begin(),output.end(),16.5f);hip_check(hipMemcpy(out.base,output.data(),output.size()*4,hipMemcpyHostToDevice),"output reset");
   // Host alpha/beta remain live through completed GPU execution.
   const float alpha=1.0f,beta=0.0f;
   blas_check(hipblasLtMatmul(handle,operation,&alpha,dw.data(),wl,di.data(),il,&beta,out.data(),ol,out.data(),ol,&algorithm.algo,workspace,algorithm.workspaceSize,stream),"matmul");complete(stream,event);
   out.read(output);auto actual_weights=weights,actual_inputs=inputs;dw.read(actual_weights);di.read(actual_inputs);require(actual_weights==weights && actual_inputs==inputs,"immutable operands");
   for(size_t i=0;i<guard;i++)require(output[i]==16.5f && output[guard+cells+i]==16.5f,"output guards");
   size_t raw_bad=0,bf16_bad=0;double max_error=0;
   for(unsigned t=0;t<tokens;t++)for(unsigned r=0;r<rows;r++){
    const float actual=output[guard+size_t(t)*rows+r],expected=reference[t%7][r%11];require(std::isfinite(actual),"finite output");
    raw_bad+=bits(actual)!=bits(expected);bf16_bad+=bf16(actual)!=bf16(expected);max_error=(std::max)(max_error,std::abs(double(actual)-expected));
   }
   std::cout.precision(12);std::cout<<"{\"type\":\"standalone_dyadic_case\",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<k<<",\"choice\":"<<choice<<",\"elements\":"<<cells<<",\"raw_f32_bit_mismatches\":"<<raw_bad<<",\"bf16_mismatches\":"<<bf16_bad<<",\"raw_max_abs_diff\":"<<max_error<<",\"workspace_bytes\":"<<algorithm.workspaceSize<<",\"first_77_classes\":[";
   for(unsigned t=0;t<7;t++)for(unsigned r=0;r<11;r++){if(t||r)std::cout<<',';std::cout<<"["<<bits(output[guard+size_t(t)*rows+r])<<','<<bits(reference[t][r])<<"]";}
   std::cout<<"],\"alpha_live_until_completion\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"engine_provider_included\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
   if(workspace)hip_check(hipFree(workspace),"free workspace");require(!bf16_bad,"generated BF16 endpoint");
  }
  blas_check(hipblasLtMatmulPreferenceDestroy(preference),"destroy preference");blas_check(hipblasLtMatrixLayoutDestroy(wl),"destroy weights");blas_check(hipblasLtMatrixLayoutDestroy(il),"destroy inputs");blas_check(hipblasLtMatrixLayoutDestroy(ol),"destroy outputs");blas_check(hipblasLtMatmulDescDestroy(operation),"destroy operation");
 }
 hip_check(hipEventDestroy(event),"destroy event");hip_check(hipStreamDestroy(stream),"destroy stream");blas_check(hipblasLtDestroy(handle),"destroy handle");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"standalone_dyadic_error=%s\n",e.what());return 1;}
