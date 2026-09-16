#pragma once
namespace projection_safety_test {
void variance_consumer(const float* residual,const uint16_t* update,const uint16_t* norm,const uint8_t* table,
 float* result,float* normalized,uint16_t* normalized_bf16,unsigned tokens) {
 hipLaunchKernelGGL(output_bf16_residual_postnorm_vllm_kernel,dim3(tokens),dim3(256u),0u,nullptr,
     residual,update,norm,result,normalized,tokens,table,normalized_bf16);
 hip_ok(hipGetLastError(),"variance_consumer");
}

void run_out_variance_safety(const char* correction_path) {
 const uint8_t* table=nullptr;std::string failure;
 require(load_gfx1151_sm121_rsqrt_correction(&table,&failure,correction_path),failure.c_str());
 constexpr unsigned rows=2048u;
 uint64_t total_cells=0,total_first=0,total_additional=0,total_skipped=0,total_fallback=0,total_boundaries=0;
 unsigned cases=0;
 for(unsigned width:{16u,32u,272u})for(unsigned count:{1u,3u,17u})for(unsigned mode=0;mode<7u;++mode) {
  const size_t cells=size_t(rows)*count;
  std::vector<uint16_t> weights(size_t(rows)*width+2*kGuard,kBf16Guard),inputs(size_t(count)*width+2*kGuard,kBf16Guard);
  std::vector<uint16_t> norm(rows+2*kGuard,kBf16Guard),reference(cells+2*kGuard,kBf16Guard);
  std::vector<float> centers(cells+2*kGuard,kF32Guard),residual(cells+2*kGuard,kF32Guard);
  for(size_t i=0;i<size_t(rows)*width;++i)weights[kGuard+i]=bf16(float(int((i*37u+i/19u)%43u)-21)/64.0f);
  for(size_t i=0;i<size_t(count)*width;++i)inputs[kGuard+i]=bf16(float(int((i*23u+i/11u)%31u)-15)/32.0f);
  if(mode==4u){for(size_t i=0;i<size_t(rows)*width;++i)weights[kGuard+i]=uint16_t((i&1u)?1u:0x8001u);for(size_t i=0;i<size_t(count)*width;++i)inputs[kGuard+i]=uint16_t((i%3u)?0x3f80u:0u);}
  if(mode==5u){for(size_t i=0;i<size_t(rows)*width;++i)weights[kGuard+i]=uint16_t((190u<<7u)|127u);for(size_t i=0;i<size_t(count)*width;++i)inputs[kGuard+i]=uint16_t((190u<<7u)|127u);}
  for(unsigned row=0;row<rows;++row)norm[kGuard+row]=bf16(float(int(row%19u)-9)/64.0f);
  if(mode==3u||mode==6u)norm[kGuard]=0x7fc0u;
  for(unsigned token=0;token<count;++token)for(unsigned row=0;row<rows;++row) {
   const size_t cell=size_t(token)*rows+row;
   const float value=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
       inputs.data()+kGuard+size_t(token)*width,weights.data()+kGuard+size_t(row)*width,width);
   reference[kGuard+cell]=bf16(value);
   const float rounded=qrt_out_consumer_interval::value(uint32_t(reference[kGuard+cell])<<16u);
   uint32_t raw_bits=uint32_t(reference[kGuard+cell])<<16u;
   // Force boundary candidates in finite nonzero cells. Both neighboring
   // BF16 endpoints remain inside the transported projection interval.
   if((raw_bits&0x7fffffffu)&&((raw_bits>>23u)&255u)<254u&&(mode==0u||cell%4u==0u))raw_bits|=0x8000u;
   // Deliberately invalid envelope: every synthetic center is selected and
   // initially has a stable residual BF16 endpoint. The NaN norm above
   // forces replay, which must observe the missed bound and finish all
   // original selected cells, rather than certify any remaining skip.
   if(mode==6u)raw_bits=0x3e008000u;
   centers[kGuard+cell]=qrt_out_consumer_interval::value(raw_bits);
   residual[kGuard+cell]=mode==0u?0.0f:mode==2u?-rounded:128.0f+float(row%8u);
  }
  DeviceBuffer<uint16_t> dw(weights),dx(inputs),dn(norm),dref(reference);
  DeviceBuffer<float> dc(centers),dr(residual);
  std::vector<float> wn(rows+2*kGuard,kF32Guard),xn(count+2*kGuard,kF32Guard);
  DeviceBuffer<float> dwn(wn),dxn(xn);
  using Row=qrt_sm121_staged_half_projection::Row;
  Row guarded;std::memset(&guarded,0xa5,sizeof(guarded));
  std::vector<Row> pw(size_t(rows)*(width/16u)+2*kGuard,guarded),px(size_t(count)*(width/16u)+2*kGuard,guarded);
  DeviceBuffer<Row> dpw(pw),dpx(px);
  std::vector<unsigned> reports(size_t(count)*qrt_out_variance_replay::fields+2*kGuard,0xa5a5a5a5u);DeviceBuffer<unsigned> ds(reports);
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((size_t(rows)*(width/16u)+255u)/256u),dim3(256u),0u,nullptr,dw.data(),dpw.data(),rows,width);
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((size_t(count)*(width/16u)+255u)/256u),dim3(256u),0u,nullptr,dx.data(),dpx.data(),count,width);
  hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dwn.data(),rows,width);
  hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(count),dim3(256u),0u,nullptr,dx.data(),dxn.data(),count,width);
  hip_ok(hipGetLastError(),"variance_prepare");hip_ok(hipDeviceSynchronize(),"variance_prepare_complete");
  dpw.read(pw);dpx.read(px);dwn.read(wn);dxn.read(xn);
  for(size_t i=0;i<kGuard;++i) {
   require(!std::memcmp(&pw[i],&guarded,sizeof(Row))&&!std::memcmp(&pw[pw.size()-kGuard+i],&guarded,sizeof(Row))&&
           !std::memcmp(&px[i],&guarded,sizeof(Row))&&!std::memcmp(&px[px.size()-kGuard+i],&guarded,sizeof(Row)),"variance prepared guards");
   require(wn[i]==kF32Guard&&wn[wn.size()-kGuard+i]==kF32Guard&&xn[i]==kF32Guard&&xn[xn.size()-kGuard+i]==kF32Guard,"variance norm guards");
  }
  hipLaunchKernelGGL(qrt_out_variance_replay::execute,dim3(count),dim3(256u),0u,nullptr,dpw.data(),dpx.data(),dc.data(),dr.data(),dn.data(),table,dxn.data(),dwn.data(),count,width,512u,1000u,ds.data());
  hip_ok(hipGetLastError(),"variance_execute");hip_ok(hipDeviceSynchronize(),"variance_execute_complete");
  ds.read(reports);
  std::vector<uint16_t> candidate(cells+2*kGuard,kBf16Guard),norm_ref(candidate),norm_got(candidate);
  DeviceBuffer<uint16_t> dout(candidate),dbref(norm_ref),dbgot(norm_got);
  hipLaunchKernelGGL(f32_to_bf16_kernel,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,dc.data(),dout.data(),cells);
  std::vector<float> rref(cells+2*kGuard,kF32Guard),rg(rref),nref(rref),ng(rref);
  DeviceBuffer<float> drr(rref),drg(rg),dnr(nref),dng(ng);
  variance_consumer(dr.data(),dref.data(),dn.data(),table,drr.data(),dnr.data(),dbref.data(),count);
  variance_consumer(dr.data(),dout.data(),dn.data(),table,drg.data(),dng.data(),dbgot.data(),count);
  hip_ok(hipDeviceSynchronize(),"variance_consumer_complete");
  drr.read(rref);drg.read(rg);dnr.read(nref);dng.read(ng);dbref.read(norm_ref);dbgot.read(norm_got);dout.read(candidate);
  require(!std::memcmp(rref.data(),rg.data(),rref.size()*4u),"variance residual differs from original");
  require(!std::memcmp(nref.data(),ng.data(),nref.size()*4u),"variance normalized F32 carrier differs from original");
  require(norm_ref==norm_got,"variance normalized BF16 differs from original");
  for(size_t i=0;i<kGuard;++i)for(size_t at:{i,kGuard+cells+i}) {
   require(rref[at]==kF32Guard&&rg[at]==kF32Guard&&nref[at]==kF32Guard&&ng[at]==kF32Guard,"variance consumer F32 guards");
   require(norm_ref[at]==kBf16Guard&&norm_got[at]==kBf16Guard,"variance consumer BF16 guards");
  }
  auto wcheck=weights,xcheck=inputs,ncheck=norm,refcheck=reference;dw.read(wcheck);dx.read(xcheck);dn.read(ncheck);dref.read(refcheck);
  require(wcheck==weights&&xcheck==inputs&&ncheck==norm&&refcheck==reference,"variance immutable BF16 inputs");
  auto rcheck=residual;dr.read(rcheck);require(!std::memcmp(rcheck.data(),residual.data(),residual.size()*4u),"variance immutable residual input");
  auto pwcheck=pw,pxcheck=px;dpw.read(pwcheck);dpx.read(pxcheck);require(!std::memcmp(pwcheck.data(),pw.data(),pw.size()*sizeof(Row))&&!std::memcmp(pxcheck.data(),px.data(),px.size()*sizeof(Row)),"variance immutable prepared inputs");
  auto wncheck=wn,xncheck=xn;dwn.read(wncheck);dxn.read(xncheck);require(wncheck==wn&&xncheck==xn,"variance immutable norms");
  dc.read(centers);
  for(size_t i=0;i<kGuard;++i){require(centers[i]==kF32Guard&&centers[kGuard+cells+i]==kF32Guard,"variance raw guards");require(candidate[i]==kBf16Guard&&candidate[kGuard+cells+i]==kBf16Guard,"variance BF16 guards");require(reports[i]==0xa5a5a5a5u&&reports[kGuard+size_t(count)*8u+i]==0xa5a5a5a5u,"variance report guards");}
  uint64_t first=0,additional=0,skipped=0,fallback=0,boundaries=0;
  for(unsigned token=0;token<count;++token){
   const auto* r=reports.data()+kGuard+token*8u;
   require(r[0]<=rows&&r[1]<=r[0]&&r[2]<=r[0]-r[1]&&r[3]==r[0]-r[1]-r[2]&&r[4]>=1u&&r[4]<=18u,"variance report counts");
   require(r[5]<=1u&&r[6]<=1u&&r[7]<=r[2]&&(!r[7]||!r[6]),"variance report state");
   require(r[6]||(!r[3]&&r[5]),"variance uncertified skip");
   if(mode==6u)require(r[0]==rows&&!r[1]&&r[2]==rows&&!r[3]&&r[4]==18u&&r[5]&&!r[6]&&r[7],"variance invalid envelope must finish original replay");
   first+=r[1];additional+=r[2];skipped+=r[3];fallback+=r[5];boundaries+=r[7];
  }
  total_cells+=cells;total_first+=first;total_additional+=additional;total_skipped+=skipped;total_fallback+=fallback;total_boundaries+=boundaries;++cases;
  std::printf("{\"kind\":\"out_variance_replay_safety\",\"width\":%u,\"tokens\":%u,\"mode\":%u,\"cells\":%zu,\"first_replay\":%llu,\"additional_replay\":%llu,\"skipped\":%llu,\"fallback_rows\":%llu,\"observed_boundary_failures\":%llu,\"all_residual_and_norm_bits_match\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",width,count,mode,cells,(unsigned long long)first,(unsigned long long)additional,(unsigned long long)skipped,(unsigned long long)fallback,(unsigned long long)boundaries);std::fflush(stdout);
 }
 require(total_first&&total_additional&&total_skipped&&total_fallback&&total_boundaries,"variance branch coverage");
 std::printf("{\"kind\":\"out_variance_replay_safety_complete\",\"cases\":%u,\"cpu_dots\":%llu,\"all_residual_and_norm_bits_match\":true,\"inference_acceptance\":false}\n",cases,(unsigned long long)total_cells);std::fflush(stdout);
}
} // namespace projection_safety_test
