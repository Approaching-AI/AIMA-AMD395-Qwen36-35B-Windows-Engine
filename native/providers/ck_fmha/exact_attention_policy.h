#pragma once
#include <cstring>

namespace qrt_exact_attention_policy {
// The combined option is qualified only for the fixed cold owner. Parse every
// call, but preserve existing dispatch for decode, suffix and longer prefill.
inline bool select(const char* option,unsigned start,unsigned count,
    bool prepared_qk,bool exponent_mask,bool register_pv,unsigned compact_mode,bool interpolated,
    bool& active){
    active=false;
    if(option&&*option&&std::strcmp(option,"0")&&std::strcmp(option,"1"))return false;
    if(!option||std::strcmp(option,"1")||start||count<=1u||count>8192u)return true;
    if(!prepared_qk||exponent_mask||!register_pv||compact_mode!=1u||!interpolated)return false;
    active=true;return true;
}
} // namespace qrt_exact_attention_policy
