#pragma once
#include "sm121_q2_recurrent.h"
#include "sm121_q2_linear_layout.h"

namespace qrt_sm121_q2 {
namespace linear_detail {
template<class Element> __global__ void convolution(ConvolutionViews<Element> view) {
    const unsigned feature=blockIdx.x*blockDim.x+threadIdx.x;
    if(feature<8192u)convolution_feature(view.qkv,view.initial_ring,view.weights,view.silu,
        view.first_position,feature,view.staged_rings,view.staged_convolution);
}
} // namespace linear_detail

// Check every cross-stage alias before submitting either producer. Failure
// after submission leaves only private buffers modified. The caller must drain
// the stream or quarantine all borrowed storage before releasing any of it.
template<class Element> inline hipError_t launch_linear(const ConvolutionViews<Element>& conv,
    const RecurrentViews& recurrence,const RecurrentTables& tables,hipStream_t stream=nullptr) {
    if(!valid_linear_views(conv,recurrence,tables))return hipErrorInvalidValue;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(linear_detail::convolution<Element>),dim3(32u),dim3(256u),0u,stream,conv);
    const auto status=hipGetLastError();
    return status==hipSuccess?launch_recurrent(recurrence,tables,stream):status;
}
} // namespace qrt_sm121_q2
