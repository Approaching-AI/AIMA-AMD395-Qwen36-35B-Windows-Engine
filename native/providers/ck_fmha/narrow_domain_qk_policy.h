#pragma once
#include <cstring>

namespace qrt_narrow_domain_qk_policy {
// Classification is refreshed for the fixed cold owner. Decode, suffix and
// long-history calls retain their existing producer even when requested.
inline bool select(const char* option,unsigned start,unsigned count,
    bool exact_attention,bool prepared_qk,bool exponent_mask,bool rz_tree,
    unsigned matrix_mode,bool& active){
    active=false;
    if(option&&*option&&std::strcmp(option,"0")&&std::strcmp(option,"1"))return false;
    if(!option||std::strcmp(option,"1")||start||count<=1u||count>8192u)return true;
    if(!exact_attention||!prepared_qk||exponent_mask||rz_tree||matrix_mode)return false;
    active=true;return true;
}
} // namespace qrt_narrow_domain_qk_policy
