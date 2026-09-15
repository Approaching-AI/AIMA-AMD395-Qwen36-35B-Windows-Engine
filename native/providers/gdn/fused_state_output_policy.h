#pragma once
#include <cstdlib>
#include <cstring>
namespace qrt_fla_fused_policy {
inline int columns(){
 const char* value=std::getenv("QRT_FLA_GDN_FUSED_STATE_OUTPUT");
 if(!value || !*value || !std::strcmp(value,"0"))return 0;
 if(!std::strcmp(value,"4"))return 4;
 if(!std::strcmp(value,"8"))return 8;
 return -1;
}
inline bool selected(int columns,bool batched,bool state,bool cooperative,unsigned matrix_lanes,unsigned checkpoints){
 return (columns==4 || columns==8) && batched && state && cooperative && matrix_lanes==1u && !checkpoints;
}
} // namespace qrt_fla_fused_policy
