#pragma once
#include <cstddef>
#include <cstring>
#include "../sm121_attention_capacity.h"

namespace qrt_long_attention_layout {
constexpr unsigned maximum_tokens=qrt_sm121_attention_capacity::kTokens;
constexpr unsigned maximum_queries=128u;
struct Layout {
    size_t probability=0,scales=0,errors=0,indices=0,count=0,elements=0;
};
// All offsets use four-byte cells. Each BF16 row count is divisible by16,
// so probability ends at a naturally aligned float boundary, even for tails.
constexpr Layout layout(unsigned queries,unsigned stride) {
    if(!queries||queries>maximum_queries||stride<queries||stride>maximum_tokens)return {};
    const size_t rows=size_t(queries)*16u,cells=rows*stride;
    const size_t scales=cells+cells/2u;
    const size_t errors=scales+rows*((stride+31u)/32u+1u);
    const size_t indices=errors+rows*256u,count=indices+rows*256u;
    return {cells,scales,errors,indices,count,count+1u};
}
inline bool select(const char* option,unsigned start,unsigned count,
    bool range_qk,bool transposed_value,bool direct_value,unsigned compact_mode,
    bool all_pv,bool matrix,bool selective,bool& enabled) {
    enabled=false;
    if(option&&*option&&std::strcmp(option,"0")&&std::strcmp(option,"1"))return false;
    if(!option||std::strcmp(option,"1"))return true;
    if(!count||start>=maximum_tokens||count>maximum_tokens-start)return false;
    if(count==1u||start+count<=8192u)return true;
    if(count>8192u||!range_qk||!transposed_value||!direct_value||compact_mode!=1u||
        all_pv||matrix||selective)return false;
    enabled=true;return true;
}
} // namespace qrt_long_attention_layout
