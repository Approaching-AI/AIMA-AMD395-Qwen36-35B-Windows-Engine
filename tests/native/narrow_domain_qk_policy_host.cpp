#include "../../native/providers/ck_fmha/narrow_domain_qk_policy.h"
#include <cassert>
#include <climits>
#include <cstdio>
#include <initializer_list>

int main(){
    unsigned cases=0u;
    const auto check=[&](const char* option,unsigned start,unsigned count,bool exact,
        bool prepared,bool exponent,bool rz,unsigned matrix,bool accepted,bool selected){
        bool active=true;
        const bool result=qrt_narrow_domain_qk_policy::select(option,start,count,
            exact,prepared,exponent,rz,matrix,active);
        assert(result==accepted&&active==selected);++cases;
    };
    for(const char* option:{static_cast<const char*>(nullptr),"","0"})
        check(option,0u,8192u,false,false,true,true,4u,true,false);
    for(unsigned count:{2u,17u,128u,7169u,8192u})
        check("1",0u,count,true,true,false,false,0u,true,true);
    // A request cannot silently replace an incompatible cold QK owner.
    check("1",0u,8192u,false,true,false,false,0u,false,false);
    check("1",0u,8192u,true,false,false,false,0u,false,false);
    check("1",0u,8192u,true,true,true,false,0u,false,false);
    check("1",0u,8192u,true,true,false,true,0u,false,false);
    check("1",0u,8192u,true,true,false,false,1u,false,false);
    check("1",0u,8192u,true,true,false,false,UINT_MAX,false,false);
    // Inactive scopes must not require the short-owner prerequisites.
    for(unsigned count:{0u,1u,8193u,UINT_MAX})
        check("1",0u,count,false,false,true,true,4u,true,false);
    for(unsigned start:{1u,7168u,8192u,UINT_MAX})
        for(unsigned count:{1u,128u,8192u,UINT_MAX})
            check("1",start,count,false,false,true,true,4u,true,false);
    for(const char* option:{"2","01","1 "," 1","true","-1"}){
        check(option,0u,8192u,true,true,false,false,0u,false,false);
        check(option,UINT_MAX,1u,false,false,true,true,4u,false,false);
    }
    std::printf("narrow_domain_qk_policy_pass cases=%u\n",cases);
}
