#include "../../native/providers/moe_accumulator/bf16_absolute_product_matrix.h"
#include "absolute_product_bound_suite.h"
int main() try {
    absolute_product_bound_test::suite([](const uint16_t* w,const uint16_t* x,const unsigned* wf,const unsigned* xf,
        float* output,unsigned rows,unsigned tokens,unsigned k,size_t first,unsigned count) {
        const unsigned first_token=unsigned(first/rows),last_token=unsigned((first+count-1u)/rows);
        hipLaunchKernelGGL(qrt_bf16_absolute_product_matrix::window_kernel,
            dim3((rows+127u)/128u,(last_token-first_token+64u)/64u),dim3(256u),0u,nullptr,
            w,x,wf,xf,output,rows,tokens,k,first,count);
    });
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr,"absolute_product_bound_error=%s\n",error.what());return 2;
}
