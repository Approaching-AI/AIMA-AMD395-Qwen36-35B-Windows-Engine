#pragma once
#include "absolute_product_bound_suite.h"

namespace projection_safety_test {
unsigned run_absolute_product_hipblaslt_suite() {
    namespace ap = absolute_product_bound_test;
    ap::suite([](const uint16_t* weights,const uint16_t* inputs,const unsigned* wf,const unsigned* xf,
        float* output,unsigned rows,unsigned tokens,unsigned k,size_t first,unsigned count) {
        std::vector<uint16_t> magnitude_weights(size_t(rows)*k+2u*ap::guard,ap::operand_guard);
        std::vector<uint16_t> magnitude_inputs(size_t(tokens)*k+2u*ap::guard,ap::operand_guard);
        ap::Buffer<uint16_t> mw(magnitude_weights),mx(magnitude_inputs);
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel,
            dim3(rows),dim3(256u),0u,nullptr,weights,wf,mw.data(),rows,k);
        ap::check(hipGetLastError());
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel,
            dim3(tokens),dim3(256u),0u,nullptr,inputs,xf,mx.data(),tokens,k);
        ap::check(hipGetLastError());ap::complete();
        mw.read(magnitude_weights);mx.read(magnitude_inputs);
        for(unsigned which=0u;which<2u;++which) {
            const unsigned n=which?tokens:rows;
            const auto& actual=which?magnitude_inputs:magnitude_weights;
            std::vector<uint16_t> original(size_t(n)*k);
            ap::check(hipMemcpy(original.data(),which?inputs:weights,original.size()*sizeof(uint16_t),hipMemcpyDeviceToHost));
            for(unsigned row=0u;row<n;++row) {
                bool valid=true;
                for(unsigned j=0u;j<k;++j)valid&=ap::eligible(original[size_t(row)*k+j]);
                for(unsigned j=0u;j<k;++j) {
                    const size_t index=size_t(row)*k+j;
                    ap::require(actual[ap::guard+index]==(valid?uint16_t(original[index]&0x7fffu):0u),"absolute BF16 view mismatch");
                }
            }
            for(unsigned j=0u;j<ap::guard;++j)
                ap::require(actual[j]==ap::operand_guard && actual[ap::guard+size_t(n)*k+j]==ap::operand_guard,"absolute view redzone");
        }
        const unsigned first_token=unsigned(first/rows),last_token=unsigned((first+count-1u)/rows);
        const unsigned matrix_tokens=last_token-first_token+1u,matrix_cells=matrix_tokens*rows;
        std::vector<float> matrix(matrix_cells+2u*ap::guard,ap::output_guard);
        ap::Buffer<float> dm(matrix);
        std::string stage,failure;
        const bool submitted=resident_bf16_matrix_matmul_f32_output(mw.data(),mx.data()+size_t(first_token)*k,
            dm.data(),rows,k,matrix_tokens,nullptr,"absolute_product_selftest",&stage,&failure);
        ap::require(submitted,failure.c_str());
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::finish_matrix_kernel,
            dim3((matrix_cells+255u)/256u),dim3(256u),0u,nullptr,dm.data(),wf,xf,rows,first_token,k,matrix_cells);
        ap::check(hipGetLastError());ap::complete();
        ap::check(hipMemcpy(output,dm.data()+first%rows,size_t(count)*sizeof(float),hipMemcpyDeviceToDevice));
        dm.read(matrix);
        for(unsigned j=0u;j<ap::guard;++j)
            ap::require(matrix[j]==ap::output_guard && matrix[ap::guard+matrix_cells+j]==ap::output_guard,"matrix backend output redzone");
        mw.unchanged(magnitude_weights);mx.unchanged(magnitude_inputs);
    });
    return 12u;
}
}
