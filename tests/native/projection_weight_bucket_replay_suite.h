#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/replay_weight_buckets.h"
#include "../../native/providers/moe_accumulator/sm121_staged_half_projection.h"
namespace projection_safety_test {
namespace weight_buckets = qrt_replay_weight_buckets;
using WeightBucketRow = qrt_sm121_staged_half_projection::Row;
__global__ void weight_bucket_replay_kernel(const WeightBucketRow* weights,const WeightBucketRow* inputs,
    const unsigned* indices,const unsigned* status,float* output,unsigned rows,unsigned width,unsigned count) {
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if(slot>=count || (status && *status))return;
    const unsigned cell=indices[slot];
    const float value=qrt_sm121_staged_half_projection::dot<2u>(inputs+size_t(cell/rows)*(width/16u),weights+size_t(cell%rows)*(width/16u),width);
    if(!(threadIdx.x&3u))output[cell]=value;
}
void run_weight_bucket_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,DeviceBuffer<float>& dout,
    const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,const std::vector<uint16_t>& reference,
    const std::vector<float>& initial,const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    require(std::is_sorted(selected.begin(),selected.end()) && std::adjacent_find(selected.begin(),selected.end())==selected.end(),"bucket candidates must be unique and sorted");
    const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
    constexpr unsigned marker=0xa5a5a5a5u;
    WeightBucketRow row_guard;std::memset(&row_guard,0xa5,sizeof(row_guard));
    std::vector<WeightBucketRow> wp(wg+2u*kGuard,row_guard),ip(ig+2u*kGuard,row_guard);
    DeviceBuffer<WeightBucketRow> pw(wp),pi(ip);
    std::vector<unsigned> indices(selected.size()+2u*kGuard,marker),ordered=indices;
    std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);
    DeviceBuffer<unsigned> ids(indices),reordered(ordered);
    const weight_buckets::Plan plans[]={{rows,tokens,1u,8192u},{rows,tokens,16u,256u},{rows,tokens,64u,64u}};
    size_t capacity=0u;for(auto plan:plans)capacity=std::max(capacity,weight_buckets::workspace_words(plan));
    std::vector<unsigned> workspace(capacity+2u*kGuard,marker);DeviceBuffer<unsigned> scratch(workspace);
    std::vector<float> control;double samples[4][3]{};
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<4u;++position) {
        const unsigned variant=(attempt+position)%4u;
        hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"weight bucket output reset");
        hip_ok(hipMemset(reordered.base,0xa5,ordered.size()*4u),"weight bucket index reset");
        hip_ok(hipMemset(scratch.base,0xa5,workspace.size()*4u),"weight bucket workspace reset");
        complete_strong_projection();const auto start=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"weight bucket prepare weights");
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"weight bucket prepare inputs");
        const unsigned* active_indices=ids.data();const unsigned* status=nullptr;
        if(variant) {
            const auto plan=plans[variant-1u];
            hip_ok(weight_buckets::launch(ids.data(),selected.size(),unsigned(selected.size()),plan,reordered.data(),selected.size(),scratch.data(),capacity,nullptr),"weight bucket histogram prefix scatter");
            active_indices=reordered.data();status=scratch.data()+weight_buckets::workspace_words(plan)-1u;
        }
        constexpr unsigned candidates_per_dispatch=4096u*64u;
        for(unsigned offset=0u;offset<selected.size();offset+=candidates_per_dispatch) {
            const unsigned count=std::min(candidates_per_dispatch,unsigned(selected.size())-offset);
            hipLaunchKernelGGL(weight_bucket_replay_kernel,dim3((count+63u)/64u),dim3(256u),0u,nullptr,
                pw.data(),pi.data(),active_indices+offset,status,dout.data(),rows,width,count);
            hip_ok(hipGetLastError(),"weight bucket exact replay");
        }
        complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;require(!control.empty(),"weight bucket missing control");
        for(size_t cell=0u;cell<cells;++cell) {
            require(std::isfinite(output[kGuard+cell]),"weight bucket unwritten or nonfinite output");
            require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"weight bucket raw output differs");
            require(bf16(output[kGuard+cell])==reference[kGuard+cell],"weight bucket GB10 output differs");
        }
        reordered.read(ordered);scratch.read(workspace);
        if(variant) {
            const auto plan=plans[variant-1u];const unsigned bins=weight_buckets::bucket_count(plan);
            const unsigned* histogram=workspace.data()+kGuard;const unsigned* starts=histogram+bins;const unsigned* cursors=starts+bins+1u;
            require(cursors[bins]==0u && starts[0]==0u && starts[bins]==selected.size(),"weight bucket device status or span");
            std::vector<unsigned> expected_histogram(bins,0u);for(unsigned cell:selected)++expected_histogram[weight_buckets::bucket(plan,cell)];
            for(unsigned bin=0u;bin<bins;++bin) {
                require(histogram[bin]==expected_histogram[bin] && starts[bin+1u]-starts[bin]==histogram[bin] && cursors[bin]==starts[bin+1u],"weight bucket histogram or cursor differs");
                for(unsigned slot=starts[bin];slot<starts[bin+1u];++slot)require(weight_buckets::bucket(plan,ordered[kGuard+slot])==bin,"weight bucket range membership differs");
            }
            auto permutation=std::vector<unsigned>(ordered.begin()+kGuard,ordered.end()-kGuard);std::sort(permutation.begin(),permutation.end());
            require(permutation==selected,"weight bucket complete permutation differs");
            for(size_t i=weight_buckets::workspace_words(plan);i<capacity;++i)require(workspace[kGuard+i]==marker,"weight bucket unused workspace tail");
        } else {
            require(ordered==std::vector<unsigned>(ordered.size(),marker) && workspace==std::vector<unsigned>(workspace.size(),marker),"weight bucket control changed reorder storage");
        }
        auto after_ids=indices;ids.read(after_ids);auto after_w=weights,after_i=inputs;dw.read(after_w);di.read(after_i);
        require(after_ids==indices && after_w==weights && after_i==inputs,"weight bucket source changed");
        pw.read(wp);pi.read(ip);
        for(unsigned side=0u;side<2u;++side) {
            const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
            for(size_t group=0u;group<count;++group) {
                const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);
                require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"weight bucket prepared operands changed");
            }
            for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&row_guard,sizeof(row_guard)) && !std::memcmp(&packed[kGuard+count+i],&row_guard,sizeof(row_guard)),"weight bucket operand guard");
        }
        for(size_t i=0u;i<kGuard;++i) {
            require(output[i]==kF32Guard && output[kGuard+cells+i]==kF32Guard,"weight bucket output guard");
            require(ordered[i]==marker && ordered[kGuard+selected.size()+i]==marker && workspace[i]==marker && workspace[kGuard+capacity+i]==marker,"weight bucket index or workspace guard");
        }
    }
    for(unsigned variant=0u;variant<4u;++variant) {
        std::array<double,3> ordered_time{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered_time.begin(),ordered_time.end());
        const auto plan=plans[variant?variant-1u:0u];
        std::cout<<"{\"type\":\"weight_bucket_projection_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()
            <<",\"weight_rows_per_bucket\":"<<(variant?plan.weight_rows:0u)<<",\"token_rows_per_bucket\":"<<(variant?plan.token_rows:0u)<<",\"buckets\":"<<(variant?weight_buckets::bucket_count(plan):0u)<<",\"extra_workspace_bytes\":"<<(variant?(selected.size()+weight_buckets::workspace_words(plan))*4u:0u)
            <<",\"preparation_reorder_replay_ms\":"<<ordered_time[1]<<",\"samples_ms\":["<<samples[variant][0]<<','<<samples[variant][1]<<','<<samples[variant][2]<<"],\"warmups\":1,\"measured_attempts\":3,\"candidate_limit_per_dispatch\":262144,\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"complete_permutation_every_attempt\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
    }
}
}
