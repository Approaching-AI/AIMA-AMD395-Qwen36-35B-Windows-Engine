#include "../../native/providers/ck_fmha/exact_attention_policy.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <initializer_list>
int main(){
    unsigned cases=0;
    for(const char* option:{static_cast<const char*>(nullptr),"","0","1","2","01","1 ","-1"}){
        for(unsigned start:{0u,1u,7168u,8191u,8192u,UINT32_MAX}){
            for(unsigned count:{0u,1u,2u,128u,7169u,8192u,8193u,UINT32_MAX}){
                for(unsigned bits=0;bits<16u;++bits)for(unsigned mode=0;mode<4u;++mode){
                    bool active=true;
                    const bool parsed=!option||!*option||!std::strcmp(option,"0")||!std::strcmp(option,"1");
                    const bool requested=option&&!std::strcmp(option,"1")&&!start&&count>1u&&count<=8192u;
                    const bool supported=(bits&1u)&&!(bits&2u)&&(bits&4u)&&(bits&8u)&&mode==1u;
                    const bool accepted=qrt_exact_attention_policy::select(option,start,count,
                        bits&1u,bits&2u,bits&4u,mode,bits&8u,active);
                    if(accepted!=(parsed&&(!requested||supported))||active!=(parsed&&requested&&supported))return 1;
                    ++cases;
                }
            }
        }
    }
    std::printf("exact_attention_policy_pass cases=%u\n",cases);
}
