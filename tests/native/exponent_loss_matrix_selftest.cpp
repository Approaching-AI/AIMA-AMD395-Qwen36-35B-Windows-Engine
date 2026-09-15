#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_exponent_loss_matrix.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace matrix=qrt_sm121_coarse_projection_matrix;
namespace bound=qrt_sm121_coarse_projection_bound;
namespace improved=qrt_sm121_exponent_loss_matrix;
namespace loss=qrt_sm121_exponent_loss_bound;
constexpr unsigned guard=65u;
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
struct Device{
 void* pointer=nullptr;explicit Device(size_t bytes){check(hipMalloc(&pointer,bytes));check(hipMemset(pointer,0xa5,bytes));}
 ~Device(){if(pointer&&hipFree(pointer)!=hipSuccess)std::abort();}
 template<class T>T* data(){return static_cast<T*>(pointer)+guard;}
};
template<class T>std::vector<T> read(Device& d,size_t n){std::vector<T> result(n+2u*guard);check(hipMemcpy(result.data(),d.pointer,result.size()*sizeof(T),hipMemcpyDeviceToHost));return result;}
template<class T>void guards(const std::vector<T>& values){const auto* bytes=reinterpret_cast<const unsigned char*>(values.data());for(size_t i=0u;i<guard*sizeof(T);++i)require(bytes[i]==0xa5u&&bytes[(values.size()-guard)*sizeof(T)+i]==0xa5u,"coarse matrix guard");}
void finish(){hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);require(std::chrono::steady_clock::now()<deadline,"coarse matrix completion deadline");std::this_thread::yield();}check(hipEventDestroy(event));}
template<bool Improved>
void variant(unsigned rows,unsigned tokens,unsigned width,unsigned mode,Device& w,Device& x,Device& wf,Device& xf,
 Device& centers,Device& errors,Device& output,Device& indices,Device& counter,const std::vector<float>& canonical,
 const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,const std::vector<unsigned>& weight_ok,const std::vector<unsigned>& input_ok,Device& wm,Device& xm,std::vector<float>& old_center,std::vector<float>& old_error){
 const unsigned cells=rows*tokens;
 check(hipMemset(indices.pointer,0xa5,(size_t(cells)+2u*guard)*4u));check(hipMemset(counter.data<unsigned>(),0,4u));
 if constexpr(Improved){hipLaunchKernelGGL(improved::produce,dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,w.data<uint16_t>(),x.data<uint16_t>(),wf.data<unsigned>(),xf.data<unsigned>(),wm.data<loss::Summary>(),xm.data<loss::Summary>(),centers.data<float>(),errors.data<float>(),rows,tokens,width);}
 else{hipLaunchKernelGGL((matrix::produce<64u,1u>),dim3((rows+127u)/128u,(tokens+15u)/16u),dim3(256u),0u,nullptr,w.data<uint16_t>(),x.data<uint16_t>(),wf.data<unsigned>(),xf.data<unsigned>(),centers.data<float>(),errors.data<float>(),rows,tokens,width);}
 check(hipGetLastError());
 hipLaunchKernelGGL(matrix::compact,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,centers.data<float>(),errors.data<float>(),output.data<float>(),indices.data<unsigned>(),counter.data<unsigned>(),cells);check(hipGetLastError());finish();
 const auto c=read<float>(centers,cells),e=read<float>(errors,cells),out=read<float>(output,cells);
 const auto ids=read<unsigned>(indices,cells),count=read<unsigned>(counter,1u);guards(c);guards(e);guards(out);guards(ids);guards(count);
 const unsigned selected=count[guard];require(selected<=cells,"coarse matrix selected capacity");std::vector<unsigned char> seen(cells,0u);
 for(unsigned i=0u;i<selected;++i){const unsigned cell=ids[guard+i];require(cell<cells&&!seen[cell],"coarse matrix duplicate candidate");seen[cell]=1u;}
 for(unsigned i=selected;i<cells;++i)require(ids[guard+i]==0xa5a5a5a5u,"coarse matrix unused index tail");
 if constexpr(!Improved){old_center=c;old_error=e;}
 unsigned rejected=0u;
 for(unsigned cell=0u;cell<cells;++cell){
  require(!std::memcmp(&c[guard+cell],&old_center[guard+cell],4u),"metadata changed raw matrix center");
  require(e[guard+cell]<=old_error[guard+cell],"metadata envelope widened");
  const bool eligible=weight_ok[cell%rows]&&input_ok[cell/rows];const bound::State value{c[guard+cell],e[guard+cell]};
  require(std::isfinite(value.center)&&std::isfinite(canonical[cell]),"coarse matrix finite center");
  require(bool(seen[cell])==!bound::certified(value),"coarse matrix complete compaction mask");
  require(!std::memcmp(&out[guard+cell],&c[guard+cell],4u),"coarse matrix output copy");
  if(eligible)require(std::isfinite(value.error)&&std::abs(double(canonical[cell])-double(value.center))<=double(value.error),"coarse matrix canonical interval");
  else{require(std::isinf(value.error)&&seen[cell],"coarse matrix unsupported row must replay");++rejected;}
  if(!seen[cell])require(bound::scalar::bf16(value.center)==bound::scalar::bf16(canonical[cell]),"coarse matrix false BF16 certificate");
 }
 for(unsigned side=0u;side<2u;++side){const auto& expected=side?inputs:weights;const auto source=read<uint16_t>(side?x:w,expected.size());guards(source);require(!std::memcmp(source.data()+guard,expected.data(),expected.size()*2u),"coarse matrix input changed");
  const auto& flags=side?input_ok:weight_ok;const auto actual=read<unsigned>(side?xf:wf,flags.size());guards(actual);require(!std::memcmp(actual.data()+guard,flags.data(),flags.size()*4u),"coarse matrix eligibility changed");}
 for(unsigned side=0;side<2u;++side){const auto& raw=side?inputs:weights;const unsigned n=side?tokens:rows,chunks=(width+63u)/64u;
  const auto metadata=read<loss::Summary>(side?xm:wm,size_t(n)*chunks);guards(metadata);
  for(unsigned row=0;row<n;++row)for(unsigned chunk=0;chunk<chunks;++chunk){const auto expected=loss::summarize(raw.data()+size_t(row)*width+chunk*64u,std::min(64u,width-chunk*64u));
   require(!std::memcmp(&metadata[guard+size_t(row)*chunks+chunk],&expected,sizeof(expected)),"complete exponent metadata mismatch or mutation");}}
 std::printf("{\"kind\":\"exponent_loss_matrix_safety\",\"rows\":%u,\"tokens\":%u,\"width\":%u,\"mode\":%u,\"variant\":%u,\"metadata_entries\":%zu,\"cells\":%u,\"candidates\":%u,\"unsupported_cells\":%u,\"independent_cpu_dots\":%u,\"all_metadata_checked\":true,\"raw_centers_identical\":true,\"envelope_never_wider\":true,\"canonical_interval_undercoverage\":0,\"false_certificates\":0,\"complete_candidate_permutation_checked\":true,\"unused_index_tail_pass\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"hardware_error_bound_proven\":false,\"inference_acceptance\":false}\n",rows,tokens,width,mode,unsigned(Improved),size_t(rows+tokens)*((width+63u)/64u),cells,selected,rejected,cells);std::fflush(stdout);
}
void run(unsigned rows,unsigned tokens,unsigned width,unsigned mode){
 const unsigned cells=rows*tokens;std::vector<uint16_t> weights(size_t(rows)*width),inputs(size_t(tokens)*width);std::vector<unsigned> weight_ok(rows,1u),input_ok(tokens,1u);
 for(unsigned side=0u;side<2u;++side){auto& values=side?inputs:weights;auto& flags=side?input_ok:weight_ok;const unsigned n=side?tokens:rows;
  for(unsigned row=0u;row<n;++row)for(unsigned k=0u;k<width;++k){
   uint16_t x=uint16_t((((row+1u)*37u+k*53u+k/17u)&0x807fu)|((120u+(row*7u+k*11u)%17u)<<7u));
   if(mode==1u&&(row%5u==0u || k%7u==0u))x=uint16_t((k&1u)<<15u);
   if(mode==2u&&k==width/2u&&row%7u==0u)x=side?0x8001u:1u;
   if(mode==2u&&k==width-1u&&row%11u==0u)x=uint16_t((side?175u:79u)<<7u|23u);
   values[size_t(row)*width+k]=x;flags[row]&=unsigned(bound::eligible(x));
  }
 }
 std::vector<float> canonical(cells);for(unsigned cell=0u;cell<cells;++cell)canonical[cell]=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,inputs.data()+size_t(cell/rows)*width,weights.data()+size_t(cell%rows)*width,width);
 Device w((weights.size()+2u*guard)*2u),x((inputs.size()+2u*guard)*2u),wf((rows+2u*guard)*4u),xf((tokens+2u*guard)*4u),centers((cells+2u*guard)*4u),errors((cells+2u*guard)*4u),output((cells+2u*guard)*4u),indices((cells+2u*guard)*4u),counter((1u+2u*guard)*4u);
 check(hipMemcpy(w.data<uint16_t>(),weights.data(),weights.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(x.data<uint16_t>(),inputs.data(),inputs.size()*2u,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(matrix::eligibility,dim3(rows),dim3(256u),0u,nullptr,w.data<uint16_t>(),wf.data<unsigned>(),rows,width);check(hipGetLastError());
 hipLaunchKernelGGL(matrix::eligibility,dim3(tokens),dim3(256u),0u,nullptr,x.data<uint16_t>(),xf.data<unsigned>(),tokens,width);check(hipGetLastError());finish();
 const size_t wg=size_t(rows)*((width+63u)/64u),ig=size_t(tokens)*((width+63u)/64u);
 Device wm((wg+2u*guard)*sizeof(loss::Summary)),xm((ig+2u*guard)*sizeof(loss::Summary));
 hipLaunchKernelGGL(improved::prepare,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,w.data<uint16_t>(),wm.data<loss::Summary>(),rows,width);check(hipGetLastError());
 hipLaunchKernelGGL(improved::prepare,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,x.data<uint16_t>(),xm.data<loss::Summary>(),tokens,width);check(hipGetLastError());finish();
 std::vector<float> old_center,old_error;
 variant<false>(rows,tokens,width,mode,w,x,wf,xf,centers,errors,output,indices,counter,canonical,weights,inputs,weight_ok,input_ok,wm,xm,old_center,old_error);
 variant<true>(rows,tokens,width,mode,w,x,wf,xf,centers,errors,output,indices,counter,canonical,weights,inputs,weight_ok,input_ok,wm,xm,old_center,old_error);
}
int main()try{hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
 for(unsigned mode=0u;mode<3u;++mode){run(19u,17u,16u,mode);run(65u,67u,64u,mode);run(31u,33u,80u,mode);run(129u,35u,272u,mode);run(65u,67u,512u,mode);run(33u,129u,4096u,mode);}return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
