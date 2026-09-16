#pragma once
#if defined(__HIPCC__)
#define QRT_QUERY_LAYOUT __host__ __device__ __forceinline__
#else
#define QRT_QUERY_LAYOUT inline
#endif

// Storage permutations only: every logical query/K feature keeps its value.
// These component layouts make no hardware bank-conflict or speed guarantee.
namespace qrt_query_lds_layout {
constexpr unsigned rows=16u,width=128u;
template<unsigned Layout>
QRT_QUERY_LAYOUT constexpr unsigned words() {
    static_assert(Layout<4u);
    return rows*(width+(Layout==1u?4u:0u));
}
template<unsigned Layout>
QRT_QUERY_LAYOUT constexpr unsigned index(unsigned row,unsigned feature) {
    static_assert(Layout<4u);
    if constexpr(Layout==1u)return row*(width+4u)+feature;
    if constexpr(Layout==2u)return feature*rows+row;
    if constexpr(Layout==3u)return row*width+(feature^((row&1u)*4u));
    return row*width+feature;
}
}
#undef QRT_QUERY_LAYOUT
