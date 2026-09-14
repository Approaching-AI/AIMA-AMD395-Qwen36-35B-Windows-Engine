#pragma once
#include "../../native/providers/moe_accumulator/sm121_spatial_indices.h"

namespace projection_safety_test {
template<unsigned TileRows, unsigned TileTokens>
__global__ __launch_bounds__(256) void spatial_projection_collect(
    const unsigned* mask, unsigned* indices, unsigned* count,
    unsigned capacity, unsigned rows, unsigned tokens) {
    __shared__ unsigned local_count, base, local_indices[256];
    if (!threadIdx.x) local_count=0;
    __syncthreads();
    const unsigned cell=qrt_sm121_spatial_indices::index(blockIdx.x*256u+threadIdx.x,
        rows,tokens,TileRows,TileTokens);
    if (cell!=UINT32_MAX && ((mask[cell/32u]>>(cell&31u))&1u)) {
        const unsigned slot=atomicAdd(&local_count,1u);local_indices[slot]=cell;
    }
    __syncthreads();
    if (!threadIdx.x) base=atomicAdd(count,local_count);
    __syncthreads();
    if (threadIdx.x<local_count && base+threadIdx.x<capacity)
        indices[base+threadIdx.x]=local_indices[threadIdx.x];
}

__global__ __launch_bounds__(256) void spatial_projection_replay(
    const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_flags,const unsigned* input_flags,const unsigned* indices,
    float* output,float* raw,unsigned rows,unsigned width,unsigned count) {
    const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;
    if (slot>=count) return;
    const unsigned cell=indices[slot],row=cell%rows,token=cell/rows;
    const float value=qrt_sm121_scalar_projection::validated_dot<4u,1u>(
        inputs+size_t(token)*width,weights+size_t(row)*width,width,
        weight_flags[row] && input_flags[token]);
    if (!(threadIdx.x&3u)) {raw[slot]=value;output[cell]=device_bf16_round_to_float(value);}
}

void run_spatial_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,const std::vector<float>& initial,
    const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width) {
    const size_t cells=size_t(rows)*tokens,mask_words=(cells+31u)/32u;
    constexpr unsigned marker=0xa5a5a5a5u;
    require(!selected.empty() && std::is_sorted(selected.begin(),selected.end()),"spatial original index order");
    std::vector<unsigned> source(selected.size()+2u*kGuard,marker),reordered=source;
    std::copy(selected.begin(),selected.end(),source.begin()+kGuard);
    std::vector<unsigned> wf(rows+2u*kGuard,marker),inf(tokens+2u*kGuard,marker);
    std::vector<unsigned> mask(mask_words+2u*kGuard,marker),expected_mask=mask,count(1u+2u*kGuard,marker);
    std::fill(expected_mask.begin()+kGuard,expected_mask.end()-kGuard,0u);
    for(auto cell:selected) expected_mask[kGuard+cell/32u]|=1u<<(cell&31u);
    std::vector<float> raw_initial(selected.size()+2u*kGuard,kF32Guard);
    std::fill(raw_initial.begin()+kGuard,raw_initial.end()-kGuard,std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<unsigned> ds(source),dr(reordered),dfw(wf),dfi(inf),dm(mask),dc(count);
    DeviceBuffer<float> raw(raw_initial);
    std::vector<float> control;std::vector<uint64_t> raw_control;
    for(auto tile:{std::pair{0u,0u},std::pair{64u,32u},std::pair{128u,64u},std::pair{256u,128u}}) {
        std::array<double,3> times{};
        for(unsigned repetition=0;repetition<4;repetition++) {
            hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*sizeof(float),hipMemcpyHostToDevice),"spatial output reset");
            hip_ok(hipMemcpy(raw.base,raw_initial.data(),raw_initial.size()*sizeof(float),hipMemcpyHostToDevice),"spatial raw reset");
            const auto start=std::chrono::steady_clock::now();
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(rows),dim3(256u),0u,nullptr,
                dw.data(),dfw.data(),rows,width);
            hip_ok(hipGetLastError(),"spatial weight eligibility");
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(tokens),dim3(256u),0u,nullptr,
                di.data(),dfi.data(),tokens,width);
            hip_ok(hipGetLastError(),"spatial input eligibility");
            if(tile.first) {
                hip_ok(qrt_sm121_tiled_projection::mark(ds.data(),unsigned(selected.size()),unsigned(cells),dm.data(),mask_words,nullptr),"spatial membership bitmap");
                hip_ok(hipMemsetAsync(dc.data(),0,sizeof(unsigned),nullptr),"spatial count reset");
                const auto virtual_cells=qrt_sm121_spatial_indices::extent(rows,tokens,tile.first,tile.second);
#define QRT_SPATIAL_COLLECT(r,t) if(tile.first==r) hipLaunchKernelGGL((spatial_projection_collect<r,t>),dim3(virtual_cells/256u),dim3(256u),0u,nullptr,dm.data(),dr.data(),dc.data(),unsigned(selected.size()),rows,tokens)
                QRT_SPATIAL_COLLECT(64u,32u);QRT_SPATIAL_COLLECT(128u,64u);QRT_SPATIAL_COLLECT(256u,128u);
#undef QRT_SPATIAL_COLLECT
                hip_ok(hipGetLastError(),"spatial collection");complete_strong_projection();dc.read(count);
                require(count[kGuard]==selected.size(),"spatial count before replay");
            }
            hipLaunchKernelGGL(spatial_projection_replay,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,
                dw.data(),di.data(),dfw.data(),dfi.data(),tile.first?dr.data():ds.data(),dout.data(),raw.data(),rows,width,unsigned(selected.size()));
            hip_ok(hipGetLastError(),"spatial exact replay");complete_strong_projection();
            if(repetition)times[repetition-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        auto output=initial,unrounded=raw_initial;dout.read(output);raw.read(unrounded);
        auto actual_indices=source;
        if(tile.first){dr.read(reordered);actual_indices=reordered;dm.read(mask);require(mask==expected_mask,"spatial bitmap or redzones");}
        std::vector<uint64_t> keys(selected.size());
        for(size_t i=0;i<selected.size();i++) {
            require(std::isfinite(unrounded[kGuard+i]),"spatial raw accumulator finite");uint32_t bits;
            std::memcpy(&bits,&unrounded[kGuard+i],sizeof(bits));
            keys[i]=(uint64_t(actual_indices[kGuard+i])<<32u)|bits;
        }
        std::sort(keys.begin(),keys.end());
        for(size_t i=0;i<selected.size();i++) require(unsigned(keys[i]>>32u)==selected[i],"spatial complete index permutation");
        if(!tile.first){control=output;raw_control=keys;}
        size_t raw_bad=0,bf16_bad=0,unrounded_bad=0;
        for(size_t i=0;i<keys.size();i++)unrounded_bad+=keys[i]!=raw_control[i];
        for(size_t i=0;i<cells;i++) {
            require(std::isfinite(output[kGuard+i]),"spatial finite output");
            raw_bad+=std::memcmp(&output[kGuard+i],&control[kGuard+i],sizeof(float))!=0;
            bf16_bad+=bf16(output[kGuard+i])!=reference[kGuard+i];
        }
        auto after_w=weights,after_i=inputs;auto after_source=source;
        dw.read(after_w);di.read(after_i);ds.read(after_source);
        require(after_w==weights && after_i==inputs && after_source==source,"spatial immutable operands and membership");
        dfw.read(wf);dfi.read(inf);
        const auto verify_flags=[&](const auto& data,const auto& flags,unsigned n) {
            for(unsigned r=0;r<n;r++) {bool valid=true;for(unsigned k=0;k<width;k++)valid&=qrt_sm121_float_alignment::eligible(data[kGuard+size_t(r)*width+k]);
                require(flags[kGuard+r]==unsigned(valid),"spatial CPU row flags");}
        };
        verify_flags(weights,wf,rows);verify_flags(inputs,inf,tokens);
        for(size_t i=0;i<kGuard;i++) {
            require(output[i]==kF32Guard && output[kGuard+cells+i]==kF32Guard && unrounded[i]==kF32Guard && unrounded[kGuard+selected.size()+i]==kF32Guard,"spatial output redzones");
            require(wf[i]==marker && wf[kGuard+rows+i]==marker && inf[i]==marker && inf[kGuard+tokens+i]==marker,"spatial flag redzones");
            require(actual_indices[i]==marker && actual_indices[kGuard+selected.size()+i]==marker,"spatial index redzones");
            require(count[i]==marker && count[kGuard+1u+i]==marker,"spatial count redzones");
        }
        auto ordered=times;std::sort(ordered.begin(),ordered.end());
        std::cout<<"{\"type\":\"spatial_projection_real_replay\",\"tile_rows\":"<<tile.first<<",\"tile_tokens\":"<<tile.second
            <<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells
            <<",\"candidates\":"<<selected.size()<<",\"raw_bit_mismatches\":"<<raw_bad<<",\"bf16_mismatches\":"<<bf16_bad
            <<",\"unrounded_candidate_bit_mismatches\":"<<unrounded_bad<<",\"preparation_reorder_replay_host_ms\":"<<ordered[1]
            <<",\"completed_host_ms\":["<<times[0]<<','<<times[1]<<','<<times[2]<<"],\"bitmap_bytes\":"<<(tile.first?mask_words*sizeof(unsigned):0)
            <<",\"warmups\":1,\"timed_sequences\":3,\"same_original_four_lane_arithmetic\":true,\"complete_index_permutation\":true,\"cpu_metadata_pass\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"reference_is_compute_input\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
        require(!raw_bad && !bf16_bad && !unrounded_bad,"spatial exact or GB10 mismatch");
    }
}
} // namespace projection_safety_test
