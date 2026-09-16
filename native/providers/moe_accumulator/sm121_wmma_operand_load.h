#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__HIPCC__)
#define QRT_WMMA_LOAD_INLINE __host__ __device__ __forceinline__
#else
#define QRT_WMMA_LOAD_INLINE inline
#endif
namespace qrt_sm121_wmma_operand_load {
// Copy one complete original K16 operand. Valid rows own all16 words; inactive
// rows are zero and need no address calculation. Alignment is checked before
// telling the compiler it may issue aligned vector loads. Arbitrarily aligned
// BF16 rows keep the ordinary representation-preserving memcpy path.
template<class Vector> QRT_WMMA_LOAD_INLINE Vector read(const uint16_t* row,size_t offset,bool valid){
 static_assert(sizeof(Vector)==32u);Vector value{};
 if(valid){
  const auto* source=row+offset;
  if(!(reinterpret_cast<uintptr_t>(source)&15u))
   __builtin_memcpy(&value,__builtin_assume_aligned(source,16u),sizeof(value));
  else __builtin_memcpy(&value,source,sizeof(value));
 }
 return value;
}
}
#undef QRT_WMMA_LOAD_INLINE
