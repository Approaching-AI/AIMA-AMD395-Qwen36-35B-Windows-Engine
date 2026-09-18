#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_sparse_byte_projection.h"
namespace projection_safety_test {
namespace sparse_byte=qrt_sm121_sparse_byte_projection;
using SparseByteRow=sparse_byte::staged::Row;
__global__ void sparse_byte_original_replay(const SparseByteRow* w,const SparseByteRow* x,
 const unsigned* indices,float* output,unsigned rows,unsigned width,unsigned count){
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
 const unsigned cell=indices[slot];
 const float value=qrt_sm121_staged_half_projection::dot<2u>(x+size_t(cell/rows)*(width/16u),w+size_t(cell%rows)*(width/16u),width);
 if(!(threadIdx.x&3u))output[cell]=value;
}
void run_sparse_byte_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& selected,
 unsigned rows,unsigned tokens,unsigned width,bool external_reference=true,
 const std::vector<float>* cpu_raw=nullptr,uint64_t* safety_totals=nullptr){
 const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
 constexpr unsigned marker=0xa5a5a5a5u;SparseByteRow guard;std::memset(&guard,0xa5,sizeof(guard));
 std::vector<SparseByteRow> wp(wg+2u*kGuard,guard),ip(ig+2u*kGuard,guard);DeviceBuffer<SparseByteRow> pw(wp),pi(ip);
 std::vector<unsigned> indices(selected.size()+2u*kGuard,marker);
 std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);DeviceBuffer<unsigned> ids(indices);
 using Metadata=sparse_byte::Metadata;
 Metadata metadata_guard;std::memset(&metadata_guard,0xa5,sizeof(metadata_guard));
 std::vector<Metadata> wm(wg+2u*kGuard,metadata_guard),im(ig+2u*kGuard,metadata_guard);
 DeviceBuffer<Metadata> mw(wm),mi(im);
 std::vector<unsigned> wf(rows+2u*kGuard,marker),inf(tokens+2u*kGuard,marker);
 const size_t mask_words=(cells+31u)/32u;
 std::vector<unsigned> mask(mask_words+2u*kGuard,marker),stats(4u+2u*kGuard,marker);
 DeviceBuffer<unsigned> fw(wf),fi(inf),dm(mask),ds(stats);
 std::vector<unsigned> expected_mask(mask_words,0u);
 for(unsigned cell:selected)expected_mask[cell/32u]|=1u<<(cell%32u);
 std::vector<float> control;double samples[3][3]{};unsigned reports[3][4]{};

 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<3u;++position){
  const unsigned variant=(position+attempt)%3u;
  hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"sparse_byte_reset_output");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"sparse_byte_prepare_weights");
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"sparse_byte_prepare_inputs");
  if(!variant){
   if(!selected.empty()){
    hipLaunchKernelGGL(sparse_byte_original_replay,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
     pw.data(),pi.data(),ids.data(),dout.data(),rows,width,unsigned(selected.size()));hip_ok(hipGetLastError(),"sparse_byte_control");
   }
  }else{
   hipLaunchKernelGGL(sparse_byte::prepare,dim3(rows),dim3(32u),0u,nullptr,dw.data(),mw.data(),fw.data(),rows,width);hip_ok(hipGetLastError(),"sparse_byte_weight_metadata");
   hipLaunchKernelGGL(sparse_byte::prepare,dim3(tokens),dim3(32u),0u,nullptr,di.data(),mi.data(),fi.data(),tokens,width);hip_ok(hipGetLastError(),"sparse_byte_input_metadata");
   hip_ok(qrt_sm121_tiled_projection::mark(ids.data(),unsigned(selected.size()),unsigned(cells),dm.data(),mask_words,nullptr),"sparse_byte_mask");
   hip_ok(hipMemsetAsync(ds.data(),0,16u,nullptr),"sparse_byte_counts");
   hip_ok(sparse_byte::launch(dw.data(),di.data(),pw.data(),wg,pi.data(),ig,mw.data(),mi.data(),fw.data(),fi.data(),dm.data(),mask_words,
    dout.data(),cells,ds.data(),rows,tokens,width,variant==1u?32u:64u,nullptr),"sparse_byte_replay");
  }
  complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;
  require(!control.empty(),"sparse byte original control missing");
  for(size_t cell=0u;cell<cells;++cell){
   require(std::isfinite(output[kGuard+cell]),"sparse byte nonfinite");
   require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"sparse byte raw candidate or inactive output differs");
   require(bf16(output[kGuard+cell])==reference[kGuard+cell],"sparse byte GB10 endpoint differs");
   if(cpu_raw)require(!std::memcmp(&output[kGuard+cell],&(*cpu_raw)[kGuard+cell],4u),"sparse byte independent CPU raw dot differs");
  }
  auto raw_w=weights,raw_i=inputs;dw.read(raw_w);di.read(raw_i);
  require(raw_w==weights&&raw_i==inputs,"sparse byte original operands modified");
  auto after_indices=indices;ids.read(after_indices);require(after_indices==indices,"sparse byte candidate identity changed");
  pw.read(wp);pi.read(ip);
  for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
   for(size_t group=0u;group<count;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"sparse byte prepared row changed");}
   for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&guard,sizeof(guard))&&!std::memcmp(&packed[kGuard+count+i],&guard,sizeof(guard)),"sparse byte operand guard");
  }
  if(variant){
   mw.read(wm);mi.read(im);fw.read(wf);fi.read(inf);dm.read(mask);ds.read(stats);
   require(size_t(stats[kGuard])+stats[kGuard+1u]==selected.size(),"sparse byte ownership count");
   if(!attempt)for(unsigned i=0u;i<4u;++i)reports[variant][i]=stats[kGuard+i];
   else for(unsigned i=0u;i<4u;++i)require(stats[kGuard+i]==reports[variant][i],"sparse byte ownership changed");
   for(size_t i=0u;i<mask_words;++i)require(mask[kGuard+i]==expected_mask[i],"sparse byte candidate mask");
   for(unsigned side=0u;side<2u;++side){
    const auto& raw=side?inputs:weights;const auto& metadata=side?im:wm;const auto& flags=side?inf:wf;const unsigned n=side?tokens:rows;
    for(unsigned row=0u;row<n;++row){unsigned valid=1u;
     for(unsigned group=0u;group<width/16u;++group){const size_t i=size_t(row)*(width/16u)+group;
      const auto expected=sparse_byte::byte::prepare(raw.data()+kGuard+i*16u);
      require(!std::memcmp(&expected,&metadata[kGuard+i],sizeof(expected)),"sparse byte exact metadata");valid&=expected.maximum>=0;
     }
     require(flags[kGuard+row]==valid,"sparse byte whole row admission");
    }
    const size_t count=size_t(n)*(width/16u);
    for(size_t i=0u;i<kGuard;++i){
     require(!std::memcmp(&metadata[i],&metadata_guard,sizeof(metadata_guard))&&!std::memcmp(&metadata[kGuard+count+i],&metadata_guard,sizeof(metadata_guard)),"sparse byte metadata guard");
     require(flags[i]==marker&&flags[kGuard+n+i]==marker,"sparse byte flags guard");
    }
   }
   for(size_t i=0u;i<kGuard;++i)require(mask[i]==marker&&mask[kGuard+mask_words+i]==marker&&stats[i]==marker&&stats[kGuard+4u+i]==marker,"sparse byte mask/stat guard");
  }
  for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"sparse byte output guard");
 }
 for(unsigned variant=0u;variant<3u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  if(safety_totals&&variant)for(unsigned i=0u;i<4u;++i)safety_totals[i]+=reports[variant][i];
  std::cout<<"{\"type\":\"sparse_byte_projection_comparison\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"tile\":"<<(variant==1u?32u:variant==2u?64u:0u)<<",\"fast_candidates\":"<<reports[variant][0]<<",\"original_candidates\":"<<(variant?reports[variant][1]:selected.size())<<",\"sparse_tiles\":"<<reports[variant][2]<<",\"dense_tiles\":"<<reports[variant][3]
   <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_and_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"byte_metadata_checked\":true,\"candidate_mask_checked\":true,\"gb10_reference\":"<<(external_reference?"true":"false")<<",\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
unsigned run_sparse_byte_safety(){
 struct Shape{unsigned rows,tokens,width;};
 const Shape shapes[]={{1u,1u,16u},{33u,17u,256u},{65u,33u,512u},{17u,9u,8192u}};
 unsigned cases=0u;uint64_t cpu_dots=0u,totals[4]{};
 for(const auto shape:shapes)for(unsigned mode=0u;mode<6u;++mode)for(unsigned selection=0u;selection<3u;++selection){
  const unsigned rows=shape.rows,tokens=shape.tokens,width=shape.width;const size_t cells=size_t(rows)*tokens;
  std::vector<uint16_t> w(size_t(rows)*width+2u*kGuard,kBf16Guard),x(size_t(tokens)*width+2u*kGuard,kBf16Guard);
  for(unsigned side=0u;side<2u;++side){auto& a=side?x:w;const unsigned n=side?tokens:rows;
   for(unsigned row=0u;row<n;++row)for(unsigned k=0u;k<width;++k){
    uint16_t value=uint16_t(((row*37u+k*53u+side*19u)&0x807fu)|((121u+(row+k)%10u)<<7u));
    if(mode==1u&&k%3u)value&=0x8000u;
    if(mode==2u)value=uint16_t((127u<<7u)|127u|((side&&(k/16u)&1u)?0x8000u:0u));
    if(mode==3u)value=uint16_t(((k/16u)%2u?159u:95u)<<7u|127u);
    if(mode==4u&&row%3u==0u&&k%16u==0u)value=side?uint16_t(1u):uint16_t(94u<<7u|1u);
    if(mode==5u&&row%2u==0u)value=uint16_t(((k%16u==15u)?127u:95u)<<7u|17u);
    a[kGuard+size_t(row)*width+k]=value;
   }
  }
  std::vector<unsigned> selected;
  std::vector<float> initial(cells+2u*kGuard,kF32Guard);
  std::vector<float> cpu_raw(initial);
  std::vector<uint16_t> reference(cells+2u*kGuard,kBf16Guard);
  for(size_t cell=0u;cell<cells;++cell){
   const float seed=float(int(cell%9u)-4)/8.0f;initial[kGuard+cell]=seed;cpu_raw[kGuard+cell]=seed;reference[kGuard+cell]=bf16(seed);
   if(selection==2u||(selection==1u&&(cell%11u==0u||cell+1u==cells))){
    selected.push_back(unsigned(cell));
    const float expected=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
     x.data()+kGuard+(cell/rows)*width,w.data()+kGuard+(cell%rows)*width,width);
    reference[kGuard+cell]=bf16(expected);cpu_raw[kGuard+cell]=expected;++cpu_dots;
   }
  }
  // Reverse the original queue: membership, rather than sorting or order,
  // determines tile ownership. The original control consumes this same queue.
  if(mode&1u)std::reverse(selected.begin(),selected.end());
  DeviceBuffer<uint16_t> dw(w),dx(x);DeviceBuffer<float> out(initial);
  run_sparse_byte_replays(dw,dx,out,w,x,reference,initial,selected,rows,tokens,width,false,&cpu_raw,totals);++cases;
 }
 sparse_byte::staged::Row row{};sparse_byte::Metadata meta{};uint16_t value=0u;unsigned word=0u;float output=0.0f;
 unsigned invalids=0u;
 for(unsigned which=0u;which<8u;++which){
  const auto status=sparse_byte::launch(which==0u?nullptr:&value,&value,&row,which==1u?0u:1u,&row,1u,
   &meta,&meta,&word,&word,&word,which==2u?0u:1u,&output,which==3u?0u:1u,&word,
   which==4u?0u:1u,1u,which==5u?8193u:which==6u?0u:16u,which==7u?16u:32u,nullptr);
  require(status==hipErrorInvalidValue,"sparse byte invalid launch reached GPU");++invalids;
 }
 require(totals[0]&&totals[1]&&totals[2]&&totals[3],"sparse byte fast/fallback and sparse/dense branches not exercised");
 std::printf("{\"type\":\"sparse_byte_projection_safety\",\"cases\":%u,\"variants_per_case\":3,\"cpu_original_dots\":%llu,\"invalid_launches\":%u,\"raw_bit_mismatches\":0,\"fast_candidates\":%llu,\"original_candidates\":%llu,\"sparse_tiles\":%llu,\"dense_tiles\":%llu,\"reversed_candidates\":true,\"wide_k8192\":true,\"domain_and_span_rejection\":true,\"guards_pass\":true}\n",cases,(unsigned long long)cpu_dots,invalids,(unsigned long long)totals[0],(unsigned long long)totals[1],(unsigned long long)totals[2],(unsigned long long)totals[3]);
 return cases;
}
} // namespace projection_safety_test
