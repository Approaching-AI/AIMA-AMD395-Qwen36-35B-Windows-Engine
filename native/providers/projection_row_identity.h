#pragma once
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#if defined(__HIPCC__)
#define QRT_ROW_ID_HD __host__ __device__
#else
#define QRT_ROW_ID_HD
#endif
namespace qrt_projection_row_identity {
struct Key {uint32_t first=0u,second=0u;};
QRT_ROW_ID_HD inline uint32_t mix(uint32_t word) {
    word^=word>>16u;word*=0x7feb352du;word^=word>>15u;
    word*=0x846ca68bu;return word^(word>>16u);
}
QRT_ROW_ID_HD inline Key word(uint16_t value,unsigned column) {
    const uint32_t source=uint32_t(value)^((column+1u)*0x9e3779b9u);
    return {mix(source),mix(source^0xa5b35705u)};
}
inline std::vector<unsigned> representatives(const std::vector<Key>& keys) {
    std::vector<unsigned> order(keys.size()),result(keys.size());
    std::iota(order.begin(),order.end(),0u);
    std::sort(order.begin(),order.end(),[&](unsigned a,unsigned b) {
        if(keys[a].first!=keys[b].first)return keys[a].first<keys[b].first;
        if(keys[a].second!=keys[b].second)return keys[a].second<keys[b].second;
        return a<b;
    });
    unsigned previous=0u,canonical=0u;
    for(unsigned i=0u;i<order.size();++i) {
        const unsigned token=order[i];
        if(!i||keys[token].first!=keys[previous].first||keys[token].second!=keys[previous].second)canonical=token;
        result[token]=canonical;previous=token;
    }
    return result;
}
// A hash is only a lookup hint. The caller must compare every original BF16
// word with its representative before counting any row as reusable.
} // namespace qrt_projection_row_identity
#undef QRT_ROW_ID_HD
