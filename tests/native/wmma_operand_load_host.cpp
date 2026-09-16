#include "../../native/providers/moe_accumulator/sm121_wmma_operand_load.h"
#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>
struct alignas(32) Operand {uint16_t words[16];};
int main(){
 size_t copied=0u;
 for(unsigned value=0u;value<65536u;++value){
  const unsigned offset=value%16u;std::vector<uint16_t> input(offset+16u);
  for(unsigned k=0u;k<input.size();++k)input[k]=uint16_t(value+k*8191u);
  const auto original=input;
  const auto result=qrt_sm121_wmma_operand_load::read<Operand>(input.data(),offset,true);
  for(unsigned k=0u;k<16u;++k){assert(result.words[k]==input[offset+k]);++copied;}
  assert(input==original);
 }
 // ASan checks the exact end of each allocation, including every source
 // alignment and short inactive spans. Inactive addresses are never formed.
 for(unsigned offset=0u;offset<64u;++offset){
  std::vector<uint16_t> input(offset+16u,0xffffu);
  const auto result=qrt_sm121_wmma_operand_load::read<Operand>(input.data(),offset,true);
  for(uint16_t word:result.words)assert(word==0xffffu);
 }
 for(size_t offset:{size_t(0u),std::numeric_limits<size_t>::max()}){
  const auto zero=qrt_sm121_wmma_operand_load::read<Operand>(nullptr,offset,false);
  for(uint16_t word:zero.words)assert(word==0u);
 }
 std::printf("{\"kind\":\"wmma_operand_load_host\",\"bf16_encodings\":65536,\"copied_word_checks\":%zu,\"tail_alignments\":64,\"inactive_null_offsets\":2,\"word_mismatches\":0,\"immutable_inputs\":true}\n",copied);
}
