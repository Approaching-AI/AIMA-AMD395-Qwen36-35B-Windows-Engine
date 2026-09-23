#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using hipError_t=int;using hipStream_t=void*;using hipModule_t=void*;using hipFunction_t=void*;
using hipDeviceptr_t=std::uintptr_t;
constexpr int hipSuccess=0,hipMemcpyHostToDevice=1;
struct Allocation {void* pointer;unsigned char* backing;std::size_t bytes;bool live;};
inline std::vector<Allocation> allocations;
inline std::size_t live_bytes=0,peak_bytes=0;
inline int allocation_attempts=0,fail_allocation=-1,copy_attempts=0,fail_copy=-1,drains=0;
inline int hipMalloc(void** out,std::size_t bytes){
 if(allocation_attempts++==fail_allocation)return 1;
 auto* backing=static_cast<unsigned char*>(std::malloc(bytes+1024));assert(backing);
 std::memset(backing,0xa5,512);std::memset(backing+512+bytes,0xa5,512);
 *out=backing+512;allocations.push_back({*out,backing,bytes,true});live_bytes+=bytes;
 if(live_bytes>peak_bytes)peak_bytes=live_bytes;return 0;
}
inline int hipFree(void* pointer){
 for(auto& a:allocations)if(a.pointer==pointer&&a.live){
  for(unsigned i=0;i<512;++i)assert(a.backing[i]==0xa5&&a.backing[512+a.bytes+i]==0xa5);
  a.live=false;live_bytes-=a.bytes;std::free(a.backing);return 0;
 }
 assert(false);return 1;
}
inline int hipMemcpy(void* destination,const void* source,std::size_t bytes,int kind){
 assert(kind==hipMemcpyHostToDevice&&source);
 if(copy_attempts++==fail_copy)return 1;
 bool found=false;for(const auto& a:allocations)if(a.live&&a.pointer==destination){assert(a.bytes==bytes);found=true;}
 assert(found);std::memcpy(destination,source,bytes);return 0;
}
inline int hipDeviceSynchronize(){++drains;return 0;}
inline const char* hipGetErrorString(int){return "injected HIP failure";}
struct Module {std::string symbol;bool live;};
inline std::vector<Module> modules;
inline int module_attempts=0,fail_module=-1,function_attempts=0,fail_function=-1;
inline int hipModuleLoadData(void** module,const void* bytes){
 assert(bytes&&std::memcmp(bytes,"\177ELF",4)==0);
 if(module_attempts++==fail_module)return 1;
 modules.push_back({{},true});*module=reinterpret_cast<void*>(modules.size());return 0;
}
inline int hipModuleGetFunction(void** function,void* module,const char* name){
 auto i=reinterpret_cast<std::uintptr_t>(module);assert(i&&i<=modules.size()&&modules[i-1].live);
 if(function_attempts++==fail_function)return 1;
 modules[i-1].symbol=name;*function=module;return 0;
}
inline int hipModuleUnload(void* module){
 auto i=reinterpret_cast<std::uintptr_t>(module);assert(i&&i<=modules.size()&&modules[i-1].live);
 modules[i-1].live=false;return 0;
}
struct Launch {std::string symbol;unsigned x,y,z,block,shared;std::vector<std::uintptr_t> pointers;std::vector<std::int32_t> scalars;};
inline std::vector<Launch> launches;
inline int launch_attempts=0,fail_launch=-1;
inline int hipModuleLaunchKernel(void* function,unsigned x,unsigned y,unsigned z,unsigned bx,unsigned by,unsigned bz,
 unsigned shared,void* stream,void** args,void* extra){
 const auto i=reinterpret_cast<std::uintptr_t>(function);assert(i&&i<=modules.size()&&modules[i-1].live);
 const auto& symbol=modules[i-1].symbol;
 unsigned ptrs=0,ints=0;
 if(symbol=="pack_value_kernel"){ptrs=2;ints=1;}
 else if(symbol=="ordered_qk_kernel"){ptrs=3;ints=3;}
 else if(symbol=="probability_kernel"){ptrs=4;ints=3;}
 else if(symbol=="selected_pv_kernel"){ptrs=8;ints=3;}
 else assert(false);
 assert(bx==128&&by==1&&bz==1&&!stream&&!extra);
 Launch item{symbol,x,y,z,bx,shared,{},{}};
 for(unsigned j=0;j<ptrs;++j){std::uintptr_t p;std::memcpy(&p,args[j],sizeof(p));item.pointers.push_back(p);}
 for(unsigned j=0;j<ints;++j)item.scalars.push_back(*static_cast<std::int32_t*>(args[ptrs+j]));
 assert(*static_cast<hipDeviceptr_t*>(args[ptrs+ints])==0&&*static_cast<hipDeviceptr_t*>(args[ptrs+ints+1])==0);
 launches.push_back(item);return launch_attempts++==fail_launch?1:0;
}
