#include "../../native/providers/ck_fmha/fused_probability_pv_policy.h"
#include <cstdio>
#include <cstdint>
#include <initializer_list>
int main(){
    unsigned cases=0u;
    for(const char* option:{static_cast<const char*>(nullptr),"","0","1","2","01","1 ","-1"})
        for(unsigned start:{0u,1u,7168u,8191u,8192u,UINT32_MAX})
            for(unsigned count:{0u,1u,2u,128u,7169u,8192u,8193u,UINT32_MAX})
                for(bool exact:{false,true}){
                    bool active=true;
                    const bool parsed=!option||!*option||!std::strcmp(option,"0")||!std::strcmp(option,"1");
                    const bool requested=option&&!std::strcmp(option,"1")&&!start&&count>1u&&count<=8192u;
                    const bool accepted=qrt_fused_probability_pv_policy::select(option,start,count,exact,active);
                    if(accepted!=(parsed&&(!requested||exact))||active!=(parsed&&requested&&exact))return 1;
                    ++cases;
                }
    std::printf("fused_probability_pv_policy_pass cases=%u\n",cases);
}
