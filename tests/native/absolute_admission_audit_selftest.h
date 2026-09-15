#pragma once

namespace projection_safety_test {
unsigned run_absolute_admission_audit_suite() {
    constexpr unsigned rows=17u,tokens=3u,first=13u,count=33u,cells=rows*tokens;
    unsigned cases=0u;
    for(unsigned k:{16u,2048u})for(unsigned direction:{0u,1u}) {
        // Artificial producer endpoints and selector metadata isolate both
        // directions of membership change. They are not matrix accuracy data.
        std::vector<uint16_t> weights(size_t(rows)*k+2u*kGuard,kBf16Guard);
        std::vector<uint16_t> inputs(size_t(tokens)*k+2u*kGuard,kBf16Guard);
        std::fill(weights.begin()+kGuard,weights.end()-kGuard,0u);
        std::fill(inputs.begin()+kGuard,inputs.end()-kGuard,0u);
        for(unsigned row=0u;row<rows;++row)weights[kGuard+size_t(row)*k+k-1u]=0x3f80u;
        for(unsigned token=0u;token<tokens;++token)inputs[kGuard+size_t(token)*k+k-1u]=0x3f80u;
        weights[kGuard+5u*k]=1u;inputs[kGuard+k]=0x8001u;
        std::vector<float> output(cells+2u*kGuard,kF32Guard);
        for(unsigned i=0u;i<cells;++i) {
            const uint32_t raw=i%2u?0x3f80c000u:0x3f804000u;
            std::memcpy(&output[kGuard+i],&raw,sizeof(raw));
            const float exact=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                inputs.data()+kGuard+size_t(i/rows)*k,weights.data()+kGuard+size_t(i%rows)*k,k);
            require(bf16(exact)==0x3f80u,"audit fixture independent dot changed");
        }
        std::vector<float> xn(tokens+2u*kGuard,kF32Guard),wn(rows+2u*kGuard,kF32Guard);
        std::fill(xn.begin()+kGuard,xn.end()-kGuard,direction?0.0f:100.0f);
        std::fill(wn.begin()+kGuard,wn.end()-kGuard,direction?0.0f:100.0f);
        std::vector<float> bounds(count+2u*kGuard,kF32Guard);
        std::vector<unsigned> expected;
        unsigned expected_differences=0u,expected_first=UINT32_MAX;
        for(unsigned local=0u;local<count;++local) {
            const bool different=local%3u!=0u;
            bounds[kGuard+local]=(direction?different:!different)?10000.0f:0.0f;
            const unsigned index=first+local;
            if(different && index>=rows) {
                expected.push_back(index);
                if(index%2u){++expected_differences;expected_first=(std::min)(expected_first,index);}
            }
        }
        std::vector<unsigned> counts(2u+2u*kGuard,0x5a5a5a5au),indices(count+2u*kGuard,0x5a5a5a5au);
        counts[kGuard]=counts[kGuard+1u]=0u;
        std::vector<uint16_t> pw(weights.size(),kBf16Guard),px(inputs.size(),kBf16Guard);
        std::vector<unsigned> wf(rows+2u*kGuard,0x5a5a5a5au),xf(tokens+2u*kGuard,0x5a5a5a5au);
        DeviceBuffer<uint16_t> dw(weights),dx(inputs),dpw(pw),dpx(px);
        DeviceBuffer<float> dy(output),dxn(xn),dwn(wn),db(bounds);
        DeviceBuffer<unsigned> dc(counts),di(indices),dwf(wf),dxf(xf);
        hipLaunchKernelGGL(qrt_sm121_prepared_projection::prepare_rows_kernel,
            dim3(rows),dim3(kThreads),0u,nullptr,dw.data(),dpw.data(),dwf.data(),rows,k);
        hip_ok(hipGetLastError(),"audit weight preparation");
        hipLaunchKernelGGL(qrt_sm121_prepared_projection::prepare_rows_kernel,
            dim3(tokens),dim3(kThreads),0u,nullptr,dx.data(),dpx.data(),dxf.data(),tokens,k);
        hip_ok(hipGetLastError(),"audit input preparation");
        hip_ok(hipStreamSynchronize(nullptr),"audit preparation complete");
        dpw.read(pw);dpx.read(px);dwf.read(wf);dxf.read(xf);
        require(!wf[kGuard+5u] && !xf[kGuard+1u] && wf[kGuard] && xf[kGuard],"audit fixture did not cover mixed row eligibility");
        hipLaunchKernelGGL((selected_bf16_projection_hawkeye_compact_kernel<false,true,true>),
            dim3(1u),dim3(256u),0u,nullptr,db.data(),dxn.data(),dwn.data(),dy.data(),rows,512u,1u,1000u,
            dc.data(),di.data(),first,count, nullptr);
        hip_ok(hipGetLastError(),"audit difference collection");
        hip_ok(hipStreamSynchronize(nullptr),"audit collection complete");
        dc.read(counts);di.read(indices);
        require(counts[kGuard]==expected.size(),"audit selector difference count");
        std::vector<unsigned> actual(indices.begin()+kGuard,indices.begin()+kGuard+expected.size());
        std::sort(actual.begin(),actual.end());require(actual==expected,"audit selector difference indices");
        counts[kGuard]=0u;counts[kGuard+1u]=UINT32_MAX;
        hip_ok(hipMemcpy(dc.base,counts.data(),counts.size()*sizeof(unsigned),hipMemcpyHostToDevice),"audit result reset");
        hipLaunchKernelGGL(selected_bf16_projection_hawkeye_admission_audit_kernel,
            dim3((expected.size()+(256u/kSelectedHawkeyeReplayLanes)-1u)/(256u/kSelectedHawkeyeReplayLanes)),
            dim3(256u),0u,nullptr,dw.data(),dx.data(),dpw.data(),dpx.data(),dwf.data(),dxf.data(),dy.data(),rows,k,
            di.data(),0u,unsigned(expected.size()),dc.data());
        hip_ok(hipGetLastError(),"audit exact endpoint comparison");
        hip_ok(hipStreamSynchronize(nullptr),"audit exact comparison complete");dc.read(counts);
        require(counts[kGuard]==expected_differences && counts[kGuard+1u]==expected_first,"audit endpoint differences or first index");
        auto immutable=[&](auto& device,const auto& before) {
            auto after=before;device.read(after);require(after==before,"audit changed an operand, producer, metadata or guard");
        };
        immutable(dw,weights);immutable(dx,inputs);immutable(dpw,pw);immutable(dpx,px);
        immutable(dwf,wf);immutable(dxf,xf);immutable(dy,output);immutable(db,bounds);immutable(dxn,xn);immutable(dwn,wn);immutable(di,indices);
        for(size_t i=0u;i<kGuard;++i)
            require(counts[i]==0x5a5a5a5au && counts[kGuard+2u+i]==0x5a5a5a5au,"audit count redzone");
        std::printf("{\"type\":\"absolute_admission_audit\",\"reduction_size\":%u,\"direction\":\"%s\",\"changed_candidates\":%zu,\"changed_bf16_endpoints\":%u,\"first_index\":%u,\"mixed_row_fallback\":true,\"immutable_inputs_and_outputs\":true,\"redzones_pass\":true,\"inference_acceptance\":false}\n",
            k,direction?"added":"removed",expected.size(),expected_differences,expected_first);
        ++cases;
    }
    return cases;
}
}
