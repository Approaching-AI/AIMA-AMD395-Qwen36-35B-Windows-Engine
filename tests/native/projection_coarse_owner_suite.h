#pragma once
namespace projection_safety_test {
void run_coarse_owner_projection(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
    const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference,unsigned rows,unsigned tokens,unsigned width) {
    require(qrt_coarse_out::applicable(rows,tokens,width,512u,10000u),"coarse owner capture shape");
    require(env_u32_or_default("QRT_PREFILL_DESCRIPTOR_BATCH_FULL_ATTENTION_OUT_HAWKEYE_MIDPOINT_RADIUS",0u)==512u &&
        env_u32_or_default("QRT_PREFILL_DESCRIPTOR_BATCH_FULL_ATTENTION_OUT_HAWKEYE_ABSOLUTE_ERROR_BOUND_PPB",0u)==10000u,
        "coarse owner original production bounds");
    const size_t cells=size_t(rows)*tokens;
    const std::vector<uint16_t> cleared(cells+2u*kGuard,kBf16Guard);
    DeviceBuffer<uint16_t> result(cleared);double samples[2][3]{};
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<2u;++position) {
        const unsigned variant=(attempt+position)%2u;
#ifdef _WIN32
        require(!_putenv_s("QRT_QWEN36_COARSE_OUT_PRODUCER",variant?"1":"0"),"coarse owner environment");
#else
        require(!setenv("QRT_QWEN36_COARSE_OUT_PRODUCER",variant?"1":"0",1),"coarse owner environment");
#endif
        hip_ok(hipMemcpy(result.base,cleared.data(),cleared.size()*2u,hipMemcpyHostToDevice),"coarse_owner_reset");
        hip_ok(hipDeviceSynchronize(),"coarse_owner_before");
        std::string stage,failure;const auto start=std::chrono::steady_clock::now();
        require(full_attention_output_projection_bf16_tile(dw.data(),di.data(),result.data(),rows,width,tokens,
            nullptr,"component_coarse_owner_out",&stage,&failure),(stage+": "+failure).c_str());
        hip_ok(hipDeviceSynchronize(),"coarse_owner_complete");
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(attempt)samples[variant][attempt-1u]=ms;
        auto output=cleared;result.read(output);
        for(size_t cell=0u;cell<cells;++cell)require(output[kGuard+cell]==reference[kGuard+cell],"coarse owner GB10 BF16 cell");
        for(size_t i=0u;i<kGuard;++i)require(output[i]==kBf16Guard&&output[kGuard+cells+i]==kBf16Guard,"coarse owner BF16 guard");
        auto w=weights,x=inputs;dw.read(w);di.read(x);require(w==weights&&x==inputs,"coarse owner immutable inputs");
        std::cout<<"{\"type\":\"coarse_out_owner_attempt\",\"attempt\":"<<attempt<<",\"variant\":"<<variant<<",\"all_bf16_cells_verified\":true,\"redzones_pass\":true,\"immutable_inputs\":true}"<<std::endl;
    }
    for(unsigned variant=0u;variant<2u;++variant){std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
        std::cout<<"{\"type\":\"coarse_out_owner_full_route\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"width\":"<<width<<",\"cells\":"<<cells<<",\"completed_owner_ms\":"<<sorted[1]<<",\"completed_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmups\":1,\"measured_attempts\":3,\"all_attempts_verified\":true,\"bf16_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"actual_full_attention_out_dispatch\":true,\"allocation_and_release_inside_clock\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
    }
}
}
