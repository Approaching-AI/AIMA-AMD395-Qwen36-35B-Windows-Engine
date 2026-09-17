#pragma once
#include <cstring>

namespace qrt_fused_probability_pv_policy {
inline bool select(const char* option,unsigned start,unsigned count,
    bool exact_attention,bool& active){
    active=false;
    if(option&&*option&&std::strcmp(option,"0")&&std::strcmp(option,"1"))return false;
    if(!option||std::strcmp(option,"1")||start||count<=1u||count>8192u)return true;
    if(!exact_attention)return false;
    active=true;return true;
}
} // namespace qrt_fused_probability_pv_policy
