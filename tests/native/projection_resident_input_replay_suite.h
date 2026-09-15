#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/replay_weight_buckets.h"
#include "../../native/providers/moe_accumulator/sm121_resident_input_projection.h"
namespace projection_safety_test {
namespace input_buckets=qrt_replay_weight_buckets;
namespace resident_input=qrt_sm121_resident_input_projection;
using ResidentInputRow=qrt_sm121_staged_half_projection::Row;
__global__ void resident_input_control(const ResidentInputRow* weights,const ResidentInputRow* inputs,
    const unsigned* indices,const unsigned* status,float* output,unsigned rows,unsigned width,unsigned count){
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if(slot>=count || (status&&*status))return;
    const unsigned cell=indices[slot];const float value=qrt_sm121_staged_half_projection::dot<2u>(
        inputs+size_t(cell/rows)*(width/16u),weights+size_t(cell%rows)*(width/16u),width);
    if(!(threadIdx.x&3u))output[cell]=value;
}
void run_resident_input_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,DeviceBuffer<float>& dout,
    const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,const std::vector<uint16_t>& reference,
    const std::vector<float>& initial,const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width){
    require(std::is_sorted(selected.begin(),selected.end())&&std::adjacent_find(selected.begin(),selected.end())==selected.end(),"resident input candidates must be unique and sorted");
    const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
    constexpr unsigned marker=0xa5a5a5a5u;ResidentInputRow row_guard;std::memset(&row_guard,0xa5,sizeof(row_guard));
    std::vector<ResidentInputRow> wp(wg+2u*kGuard,row_guard),ip(ig+2u*kGuard,row_guard);DeviceBuffer<ResidentInputRow> pw(wp),pi(ip);
    std::vector<unsigned> indices(selected.size()+2u*kGuard,marker),ordered=indices;std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);
    DeviceBuffer<unsigned> ids(indices),reordered(ordered);const input_buckets::Plan plan{rows,tokens,rows,1u};
    const size_t capacity=input_buckets::workspace_words(plan);require(capacity==size_t(tokens)*3u+2u,"resident input bucket extent");
    std::vector<unsigned> workspace(capacity+2u*kGuard,marker);DeviceBuffer<unsigned> scratch(workspace);
    std::vector<unsigned> histogram(tokens,0u);for(unsigned cell:selected)++histogram[cell/rows];
    std::vector<float> control;double samples[4][3]{};
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<4u;++position){
        const unsigned variant=(attempt+position)%4u;
        hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"resident input output reset");
        hip_ok(hipMemset(reordered.base,0xa5,ordered.size()*4u),"resident input index reset");
        hip_ok(hipMemset(scratch.base,0xa5,workspace.size()*4u),"resident input workspace reset");
        complete_strong_projection();const auto start=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"resident input prepare weights");
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"resident input prepare inputs");
        const unsigned* active_indices=ids.data();unsigned* status=nullptr;
        if(variant){hip_ok(input_buckets::launch(ids.data(),selected.size(),unsigned(selected.size()),plan,reordered.data(),selected.size(),scratch.data(),capacity,nullptr),"resident input bucket histogram prefix scatter");active_indices=reordered.data();status=scratch.data()+capacity-1u;}
        if(variant>=2u){hip_ok(resident_input::launch(pw.data(),wg,pi.data(),ig,active_indices,selected.size(),unsigned(selected.size()),scratch.data()+tokens,tokens+1u,status,dout.data(),cells,rows,tokens,width,variant-1u,nullptr),"resident input CTA replay");}
        else for(unsigned offset=0u;offset<selected.size();offset+=262144u){const unsigned count=std::min(262144u,unsigned(selected.size())-offset);
            hipLaunchKernelGGL(resident_input_control,dim3((count+63u)/64u),dim3(256u),0u,nullptr,pw.data(),pi.data(),active_indices+offset,status,dout.data(),rows,width,count);hip_ok(hipGetLastError(),"resident input control replay");}
        complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;require(!control.empty(),"resident input missing control");
        for(size_t cell=0u;cell<cells;++cell){require(std::isfinite(output[kGuard+cell]),"resident input nonfinite output");require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"resident input raw result differs");require(bf16(output[kGuard+cell])==reference[kGuard+cell],"resident input GB10 BF16 differs");}
        reordered.read(ordered);scratch.read(workspace);
        if(variant){const unsigned* counts=workspace.data()+kGuard;const unsigned* starts=counts+tokens;const unsigned* cursors=starts+tokens+1u;
            require(cursors[tokens]==0u&&starts[0]==0u&&starts[tokens]==selected.size(),"resident input status or extent");
            for(unsigned t=0u;t<tokens;++t){require(counts[t]==histogram[t]&&starts[t+1u]-starts[t]==counts[t]&&cursors[t]==starts[t+1u],"resident input histogram or cursor");for(unsigned at=starts[t];at<starts[t+1u];++at)require(ordered[kGuard+at]/rows==t,"resident input candidate ownership");}
            auto permutation=std::vector<unsigned>(ordered.begin()+kGuard,ordered.end()-kGuard);std::sort(permutation.begin(),permutation.end());require(permutation==selected,"resident input complete permutation");
        }else require(ordered==std::vector<unsigned>(ordered.size(),marker)&&workspace==std::vector<unsigned>(workspace.size(),marker),"resident input control changed reorder storage");
        auto after_ids=indices;ids.read(after_ids);auto after_w=weights,after_i=inputs;dw.read(after_w);di.read(after_i);require(after_ids==indices&&after_w==weights&&after_i==inputs,"resident input original source changed");
        pw.read(wp);pi.read(ip);
        for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
            for(size_t group=0u;group<count;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"resident input encoded row differs");for(unsigned i=0u;i<16u;++i)require(qrt_sm121_scaled_half_products::original(packed[kGuard+group],i)==raw[kGuard+group*16u+i],"resident input original roundtrip");}
            for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&row_guard,sizeof(row_guard))&&!std::memcmp(&packed[kGuard+count+i],&row_guard,sizeof(row_guard)),"resident input operand guard");}
        for(size_t i=0u;i<kGuard;++i){require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"resident input output guard");require(ordered[i]==marker&&ordered[kGuard+selected.size()+i]==marker&&workspace[i]==marker&&workspace[kGuard+capacity+i]==marker,"resident input index/workspace guard");}
    }
    for(unsigned variant=0u;variant<4u;++variant){std::array<double,3> time{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(time.begin(),time.end());
        std::cout<<"{\"type\":\"resident_input_projection_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"ctas_per_token\":"<<(variant>=2u?variant-1u:0u)<<",\"input_groups_per_cta\":"<<(variant>=2u?width/16u:0u)<<",\"buckets\":"<<(variant?tokens:0u)<<",\"extra_workspace_bytes\":"<<(variant?(selected.size()+capacity)*4u:0u)<<",\"preparation_partition_replay_ms\":"<<time[1]<<",\"samples_ms\":["<<samples[variant][0]<<','<<samples[variant][1]<<','<<samples[variant][2]<<"],\"warmups\":1,\"measured_attempts\":3,\"outputs_per_cta_batch\":64,\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"complete_permutation_every_attempt\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
    }
}
}
