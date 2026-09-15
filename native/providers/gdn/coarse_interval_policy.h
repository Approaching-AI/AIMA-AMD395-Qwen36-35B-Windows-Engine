#ifndef QRT_FLA_COARSE_INTERVAL_POLICY_H
#define QRT_FLA_COARSE_INTERVAL_POLICY_H
#include <cstdlib>
#include <cstring>
namespace qrt_fla_coarse_policy {
inline int mode() {
    const char* value=std::getenv("QRT_FLA_GDN_COARSE_INTERVAL");
    if(!value || !*value || !std::strcmp(value,"0"))return 0;
    return !std::strcmp(value,"1")?1:-1;
}
inline bool selected(int mode,bool scalar,unsigned count) {
    return mode==1 && scalar && count && count<=1024u;
}
}
#endif
