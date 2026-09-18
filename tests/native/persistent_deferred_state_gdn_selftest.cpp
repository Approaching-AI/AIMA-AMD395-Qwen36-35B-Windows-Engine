// Reuse the independent integer CPU oracle and original guarded GDN fixture.
#define main qrt_persistent_deferred_prior_gdn_main
#include "tiled_scalar_gdn_selftest.cpp"
#undef main
#include "../../native/providers/gdn/persistent_deferred_state.h"
#include <array>
namespace deferred=qrt_fla_deferred_state;

struct DeferredWorkspace {
    unsigned steps,audit_states;
    Device lower,upper,cache,flags,scaled,coefficients,statistics,audit_lower,audit_upper;
    explicit DeferredWorkspace(unsigned count):steps((std::min(1024u,count)+63u)/64u),
        audit_states((count+63u)/64u+(count+1023u)/1024u),
        lower((size_t(steps+1u)*state_cells+2u*guard)*4u),upper(lower.bytes),
        cache((size_t(steps)*state_cells+2u*guard)*4u),flags(size_t(steps)*state_cells+2u*guard),
        scaled((size_t(std::min(1024u,count))*4096u+2u*guard)*2u),coefficients((steps*32u+2u*guard)*4u),
        statistics((size_t(audit_states)*(state_cells/256u)*7u+2u*guard)*4u),audit_lower((size_t(audit_states)*state_cells+2u*guard)*4u),audit_upper(audit_lower.bytes){}
    void reset(){for(auto* d:{&lower,&upper,&cache,&flags,&scaled,&coefficients,&statistics,&audit_lower,&audit_upper})d->reset();}
};
void deferred_launch(unsigned variant,unsigned count,Device& q,Device& k,Device& v,Device& beta,
    Device& inverse,Device& g,Device& scores,Device& u,Device& w,Device& output,Device& h,Device& vn,
    Device& state,Device& table,bool alias,DeferredWorkspace& workspace,bool audit){
    require(variant<=2u,"invalid persistent deferred variant");
    const auto* exp=reinterpret_cast<unsigned char*>(table.data<uint32_t>());
    auto* actual_u=alias?v.data<uint16_t>():u.data<uint16_t>();
    for(unsigned offset=0u;offset<count;offset+=1024u){
        const unsigned n=std::min(1024u,count-offset),steps=(n+63u)/64u;
        const auto* keys=k.data<uint16_t>()+size_t(offset)*2048u;
        const auto* gates=g.data<float>()+size_t(offset)*32u;
        auto* hw=h.data<uint16_t>()+size_t(offset/64u)*state_cells;
        auto* updated=vn.data<uint16_t>()+size_t(offset)*4096u;
        const size_t audit_offset=size_t(offset/64u+offset/1024u)*state_cells;
        auto* statistics=workspace.statistics.data<unsigned>()+(audit_offset/256u)*7u;
        float* audit_lo=audit?workspace.audit_lower.data<float>()+audit_offset:nullptr;
        float* audit_hi=audit?workspace.audit_upper.data<float>()+audit_offset:nullptr;
        hipLaunchKernelGGL(scalar::wu_kernel,dim3(16u,32u,steps),dim3(256u),0u,nullptr,
            keys,v.data<uint16_t>()+size_t(offset)*4096u,beta.data<uint16_t>()+size_t(offset)*32u,
            inverse.data<uint16_t>()+size_t(offset)*2048u,gates,w.data<uint16_t>()+size_t(offset)*4096u,
            actual_u+size_t(offset)*4096u,n,exp);check(hipGetLastError());
        if(!variant){
            hipLaunchKernelGGL((qrt_fla_lifetime::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,
                keys,actual_u+size_t(offset)*4096u,w.data<uint16_t>()+size_t(offset)*4096u,gates,
                hw,updated,state.data<float>(),n,exp);check(hipGetLastError());
        }else{
            hipLaunchKernelGGL(deferred::initialize_history,dim3(state_cells/256u),dim3(256u),0u,nullptr,
                state.data<float>(),gates,workspace.lower.data<float>(),workspace.upper.data<float>(),
                workspace.coefficients.data<float>(),n,exp,audit_lo,audit_hi);check(hipGetLastError());
#define QRT_PERSISTENT_STATE_ARGS keys,actual_u+size_t(offset)*4096u,w.data<uint16_t>()+size_t(offset)*4096u,gates,hw,updated,state.data<float>(),n,exp,workspace.lower.data<float>(),workspace.upper.data<float>(),workspace.cache.data<float>(),workspace.flags.data<unsigned char>(),workspace.scaled.data<uint16_t>(),workspace.coefficients.data<float>(),statistics,audit_lo,audit_hi
            if(variant==1u){hipLaunchKernelGGL((qrt_fla_persistent_deferred::state<false>),dim3(16u,32u),dim3(256u),0u,nullptr,QRT_PERSISTENT_STATE_ARGS);}
            else{hipLaunchKernelGGL((qrt_fla_persistent_deferred::state<true>),dim3(16u,32u),dim3(256u),0u,nullptr,QRT_PERSISTENT_STATE_ARGS);}
#undef QRT_PERSISTENT_STATE_ARGS
            check(hipGetLastError());
        }
        hipLaunchKernelGGL(qrt_fla_lifetime::output_kernel,dim3(16u,32u,steps),dim3(256u),0u,nullptr,
            q.data<uint16_t>()+size_t(offset)*2048u,updated,hw,gates,scores.data<uint16_t>()+size_t(offset)*2048u,
            output.data<float>()+size_t(offset)*4096u,n,exp);check(hipGetLastError());
    }
}

// Original state64 calls generate an observer-only FP32 trace at every
// checkpoint. Their complete H/Vnew/final state are compared to the original
// state8 segment fixture before any candidate is run.
void reference_history(unsigned count,Device& k,Device& g,const std::vector<uint16_t>& host_w,
    const std::vector<uint16_t>& host_u,const std::vector<float>& seed,Device& table,Device& trace,
    const std::vector<uint16_t>& expected_h,const std::vector<uint16_t>& expected_vn,const std::vector<float>& expected_final){
    const size_t large=size_t(count)*4096u,checkpoints=size_t((count+63u)/64u)*state_cells;
    Device w((large+2u*guard)*2u),u(w.bytes),state((state_cells+2u*guard)*4u),h((checkpoints+2u*guard)*2u),vn(w.bytes);
    w.upload(std::vector<uint16_t>(host_w.begin()+guard,host_w.end()-guard));
    u.upload(std::vector<uint16_t>(host_u.begin()+guard,host_u.end()-guard));state.upload(seed);
    check(hipMemcpyAsync(trace.data<float>(),state.data<float>(),state_cells*4u,hipMemcpyDeviceToDevice,nullptr));
    for(unsigned offset=0u;offset<count;offset+=64u){
        const unsigned valid=std::min(64u,count-offset);
        hipLaunchKernelGGL((scalar::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,
            k.data<uint16_t>()+size_t(offset)*2048u,u.data<uint16_t>()+size_t(offset)*4096u,
            w.data<uint16_t>()+size_t(offset)*4096u,g.data<float>()+size_t(offset)*32u,
            h.data<uint16_t>()+size_t(offset/64u)*state_cells,vn.data<uint16_t>()+size_t(offset)*4096u,
            state.data<float>(),valid,reinterpret_cast<unsigned char*>(table.data<uint32_t>()));check(hipGetLastError());
        check(hipMemcpyAsync(trace.data<float>()+size_t(offset/64u+1u)*state_cells,state.data<float>(),
            state_cells*4u,hipMemcpyDeviceToDevice,nullptr));
    }
    finish();const auto actual_state=read<float>(state,state_cells);
    require(!std::memcmp(actual_state.data(),expected_final.data(),actual_state.size()*4u),"original trace final state differs");
    require(read<uint16_t>(h,checkpoints)==expected_h,"original trace H differs");
    require(read<uint16_t>(vn,large)==expected_vn,"original trace Vnew differs");
}
__global__ void inspect_history(const float* original,const float* lower,const float* upper,
    unsigned count,bool audit,unsigned states,unsigned* bad){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(states)*state_cells)return;
    const unsigned step=unsigned(i/state_cells),cell=unsigned(i%state_cells);
    const unsigned original_step=audit?(step/17u)*16u+step%17u:((count-1u)/1024u)*16u+step;
    const float expected=original[size_t(original_step)*state_cells+cell];
    if(!(expected>=lower[i] && expected<=upper[i]))atomicAdd(bad,1u);
}
void device_guards(Device& d,size_t width){
    std::vector<unsigned char> bytes(guard*width*2u);
    check(hipMemcpy(bytes.data(),d.p,guard*width,hipMemcpyDeviceToHost));
    check(hipMemcpy(bytes.data()+guard*width,static_cast<unsigned char*>(d.p)+d.bytes-guard*width,guard*width,hipMemcpyDeviceToHost));
    for(auto b:bytes)require(b==0xa5u,"deferred scratch redzone changed");
}
std::array<unsigned,7> verify_deferred(DeferredWorkspace& w,Device& reference,unsigned count,bool audit){
    Device bad((1u+2u*guard)*4u);check(hipMemset(bad.data<unsigned>(),0,4u));
    if(audit){
        hipLaunchKernelGGL(inspect_history,dim3((size_t(w.audit_states)*state_cells+255u)/256u),dim3(256u),0u,nullptr,
            reference.data<float>(),w.audit_lower.data<float>(),w.audit_upper.data<float>(),count,true,w.audit_states,bad.data<unsigned>());check(hipGetLastError());
    }
    const unsigned last_count=(count-1u)%1024u+1u,last_steps=(last_count+63u)/64u;
    hipLaunchKernelGGL(inspect_history,dim3((size_t(last_steps+1u)*state_cells+255u)/256u),dim3(256u),0u,nullptr,
        reference.data<float>(),w.lower.data<float>(),w.upper.data<float>(),count,false,last_steps+1u,bad.data<unsigned>());check(hipGetLastError());finish();
    const auto errors=read<unsigned>(bad,1u);guards(errors);require(errors[guard]==0u,"native state interval escaped original arithmetic");
    const size_t records=size_t(w.audit_states)*(state_cells/256u);
    const auto statistics=read<unsigned>(w.statistics,records*7u);guards(statistics);std::array<unsigned,7> result{};
    for(size_t record=0u;record<records;++record)for(unsigned i=0u;i<7u;++i){
        const unsigned value=statistics[guard+record*7u+i];
        result[i]=i==4u?std::max(result[i],value):result[i]+value;
    }
    require(result[0]==state_cells*((count+63u)/64u+(count+1023u)/1024u),"incomplete deferred boundaries");
    require(result[2]<=state_cells*((count+63u)/64u) && result[4]<=16u && result[6]==0u,"deferred replay ownership or completion failure");
    for(auto* d:{&w.lower,&w.upper,&w.cache,&w.coefficients,&w.audit_lower,&w.audit_upper})device_guards(*d,4u);
    device_guards(w.flags,1u);device_guards(w.scaled,2u);device_guards(reference,4u);return result;
}

void deferred_run(unsigned count,unsigned mode,unsigned measured,Device& table,const std::vector<unsigned char>& host_table){
 const size_t small=size_t(count)*2048u,large=size_t(count)*4096u,gate=size_t(count)*32u,checkpoints=size_t((count+63u)/64u)*state_cells;
 std::vector<uint16_t> q(small),k(small),v(large),beta(gate),inverse(small),scores(small);std::vector<float> g(gate),seed(state_cells);
 auto fill=[&](std::vector<uint16_t>& x,unsigned salt,unsigned exponent){for(size_t i=0u;i<x.size();++i){const unsigned r=random_word(unsigned(i)^salt);x[i]=mode?uint16_t((r&0x807fu)|((exponent+r%4u)<<7u)):uint16_t(r&0x8000u);if(mode==2u && i%29u==0u)x[i]=uint16_t(r&0x807fu);}};
 fill(q,395u,115u);fill(k,8192u,115u);fill(v,35u,120u);fill(inverse,3u,116u);fill(scores,121u,112u);
 for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<32u;++head){
  beta[size_t(row)*32u+head]=uint16_t(0x3e80u|((row+head)&127u));g[size_t(row)*32u+head]=-float((row%64u+1u)*(head%7u+1u))*.0078125f;
  for(unsigned future=row%64u+1u;future<64u;++future){scores[(size_t(row)*32u+head)*64u+future]=0u;inverse[(size_t(row)*32u+head)*64u+future]=0u;}
  inverse[(size_t(row)*32u+head)*64u+row%64u]=0x3f80u;
 }
 for(size_t i=0u;i<state_cells;++i){const unsigned r=random_word(unsigned(i)^0x395u);seed[i]=mode==0u?from(uint16_t(r&0x8000u)):from(uint16_t((r&0x807fu)|((118u+r%4u)<<7u)));}
 Device dq((small+2u*guard)*2u),dk((small+2u*guard)*2u),dv((large+2u*guard)*2u),db((gate+2u*guard)*2u),dinv((small+2u*guard)*2u),ds((small+2u*guard)*2u),dg((gate+2u*guard)*4u);
 dq.upload(q);dk.upload(k);db.upload(beta);dinv.upload(inverse);ds.upload(scores);dg.upload(g);
 Device du((large+2u*guard)*2u),dw((large+2u*guard)*2u),output((large+2u*guard)*4u),state((state_cells+2u*guard)*4u),h((checkpoints+2u*guard)*2u),vn((large+2u*guard)*2u);
 DeferredWorkspace workspace(count);Device reference_trace(((size_t((count+63u)/64u)+1u)*state_cells+2u*guard)*4u);
 std::array<std::array<std::array<unsigned,7>,3>,2> work{};
 std::vector<float> expected,final;std::vector<uint16_t> old_w,old_u,old_h,old_vn;size_t cpu_dots=0u;double samples[2][3][3]{};
 const unsigned attempts=measured+1u;
 for(unsigned attempt=0u;attempt<attempts;++attempt)for(unsigned position=0u;position<6u;++position){
  const unsigned choice=(attempt+position)%6u,variant=choice%3u;const bool alias=choice>=3u;
  dv.reset();dv.upload(v);du.reset();dw.reset();output.reset();state.reset();state.upload(seed);h.reset();vn.reset();workspace.reset();finish();
  const auto begin=std::chrono::steady_clock::now();deferred_launch(variant,count,dq,dk,dv,db,dinv,dg,ds,du,dw,output,h,vn,state,table,alias,workspace,attempt==0u);finish();
  if(attempt){
   samples[unsigned(alias)][variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();

  }
  const auto actual=read<float>(output,large),next=read<float>(state,state_cells);
  const auto actual_w=read<uint16_t>(dw,large),actual_u=read<uint16_t>(alias?dv:du,large),actual_h=read<uint16_t>(h,checkpoints),actual_vn=read<uint16_t>(vn,large);
  guards(actual);guards(next);guards(actual_w);guards(actual_u);guards(actual_h);guards(actual_vn);
  if(!attempt && !choice){
   expected=actual;final=next;old_w=actual_w;old_u=actual_u;old_h=actual_h;old_vn=actual_vn;
   reference_history(count,dk,dg,old_w,old_u,seed,table,reference_trace,old_h,old_vn,final);
   if(mode<2u){
    cpu_dots+=cpu_wu(count,k,v,beta,inverse,g,host_table,old_w,old_u);
    const std::vector<uint16_t> hw(old_w.begin()+guard,old_w.end()-guard),hu(old_u.begin()+guard,old_u.end()-guard);
    for(const auto& column:std::vector<std::pair<unsigned,unsigned>>{{0u,0u},{3u,7u},{31u,127u}})cpu_dots+=cpu_column(count,column.first,column.second,q,k,hu,hw,g,scores,seed,host_table,expected,final,old_h,old_vn);
   }
  }
  require(!expected.empty() && !std::memcmp(actual.data(),expected.data(),actual.size()*4u),"persistent-deferred GDN output differs");require(!std::memcmp(next.data(),final.data(),next.size()*4u),"persistent-deferred GDN state differs");
  require(actual_w==old_w && actual_u==old_u && actual_h==old_h && actual_vn==old_vn,"persistent-deferred GDN intermediate or alias differs");
  if(variant){
   const auto counts=verify_deferred(workspace,reference_trace,count,attempt==0u);
   if(!attempt)work[unsigned(alias)][variant]=counts;else require(work[unsigned(alias)][variant]==counts,"deferred work counts changed");
  }

  unchanged(dq,q);unchanged(dk,k);unchanged(db,beta);unchanged(dinv,inverse);unchanged(dg,g);unchanged(ds,scores);
  if(!alias)unchanged(dv,v);else{const auto unused=read<uint16_t>(du,large);const auto* bytes=reinterpret_cast<const unsigned char*>(unused.data());require(std::all_of(bytes,bytes+unused.size()*2u,[](unsigned char x){return x==0xa5u;}),"inactive U buffer changed");}
 }
 for(unsigned alias=0u;alias<2u;++alias)for(unsigned variant=0u;variant<3u;++variant){
  double sorted[3]={samples[alias][variant][0],samples[alias][variant][1],samples[alias][variant][2]};std::sort(sorted,sorted+3u);
  const auto& n=work[alias][variant];
  std::printf("{\"kind\":\"persistent_deferred_state_gdn_component\",\"tokens\":%u,\"mode\":%u,\"variant\":%u,\"deferred_state\":%s,\"coarse_residual\":%s,\"segment_tokens\":1024,\"segments\":%u,\"u_aliases_v\":%s,\"output_cells\":%zu,\"state_cells\":%zu,\"checkpoint_cells\":%zu,\"wu_and_residual_cells\":%zu,\"independent_cpu_dots\":%zu,\"warmups\":1,\"measured_attempts\":%u,\"complete_wu_state_output_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"boundary_cells\":%u,\"refined_boundaries\":%u,\"original_state_dots\":%u,\"total_refinement_depth\":%u,\"maximum_refinement_depth\":%u,\"full_tail_resolutions\":%u,\"unresolved\":%u,\"producer_enclosure_escapes\":0,\"producer_intervals_checked_on_warmup\":true,\"intermediate_host_synchronization\":false,\"all_attempts_verified\":true,\"raw_bit_mismatches\":0,\"intermediate_and_alias_ownership_checked\":true,\"redzones_pass\":true,\"immutable_nonaliased_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",count,mode,variant,variant?"true":"false",variant==2u?"true":"false",(count+1023u)/1024u,alias?"true":"false",large,state_cells,checkpoints,large*3u,cpu_dots,measured,sorted[1],samples[alias][variant][0],samples[alias][variant][1],samples[alias][variant][2],n[0],n[1],n[2],n[3],n[4],n[5],n[6]);
 }
 std::fflush(stdout);
}
int main(int argc,char** argv)try{
 require(argc==3,"requires verified SM121 exponential table and safety or throughput");hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
 const bool timing=!std::strcmp(argv[2],"throughput");require(timing || !std::strcmp(argv[2],"safety"),"unknown action");
 std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(file&&file.tellg()==std::streamoff(qrt_sm121_exp2::table_bytes),"table span");std::vector<unsigned char> table(qrt_sm121_exp2::table_bytes);file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());require(bool(file)&&qrt_sm121_exp2::valid_layout(table.data(),table.size()),"table layout");
 std::vector<uint32_t> table_words(table.size()/4u);require(table.size()%4u==0u,"table alignment");std::memcpy(table_words.data(),table.data(),table.size());Device dt((table_words.size()+2u*guard)*4u);dt.upload(table_words);
 if(timing)deferred_run(8192u,1u,3u,dt,table);else for(unsigned count:{1u,63u,64u,65u,129u,1023u,1024u,1025u})for(unsigned mode=0u;mode<3u;++mode)deferred_run(count,mode,0u,dt,table);
 unchanged(dt,table_words);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
