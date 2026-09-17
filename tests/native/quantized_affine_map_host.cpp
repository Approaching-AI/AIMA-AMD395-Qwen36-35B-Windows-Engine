#include "checked_quantized_affine_map.h"
#include "../../native/providers/moe_accumulator/sm121_dyadic_carry_scan.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace map = qrt_quantized_affine_map;
namespace modular = qrt_sm121_dyadic_carry_scan;
using Wide = __int128;
Wide floor_wide(Wide value, unsigned shift) {
    const Wide grid = Wide(1) << shift;
    const Wide quotient = value / grid, remainder = value % grid;
    return (quotient - (remainder < 0 ? 1 : 0)) * grid;
}
Wide evaluate_wide(map::Map m, Wide x) {
    return floor_wide(x + m.before, m.shift) + m.after;
}
bool fit(Wide x) { return x >= INT64_MIN && x <= INT64_MAX; }
bool same(map::Map a, map::Map b) {
    return a.before == b.before && a.after == b.after && a.shift == b.shift;
}
modular::Function convert(map::Map input) {
    map::Map canonical;
    if (!map::canonicalize(input,&canonical)) std::abort();
    return {uint64_t(canonical.before),uint64_t(canonical.after),canonical.shift};
}
uint64_t random_state = 0x39581927169ull;
uint64_t next() {
    random_state ^= random_state << 13u; random_state ^= random_state >> 7u;
    random_state ^= random_state << 17u; return random_state;
}
int main() {
    uint64_t pairs = 0u, triples = 0u, rounding = 0u, rejection = 0u;
    const map::Map sentinel{71,-39,17};
    // Exhaustive small positive/negative offsets, all relative grid orders,
    // and values immediately on both sides of zero and quantization edges.
    for (unsigned a = 0u; a <= 6u; ++a) for (unsigned b = 0u; b <= 6u; ++b)
        for (int offset = -16; offset <= 16; ++offset)
            for (int tail = -16; tail <= 16; ++tail) {
                const map::Map first{offset, tail, a}, second{offset - tail, offset + tail, b};
                map::Map combined;
                if (!map::compose(first,second,&combined)) return 1;
                for (int x = -65; x <= 65; ++x) {
                    int64_t actual;
                    const Wide expected = evaluate_wide(second,evaluate_wide(first,x));
                    if (!map::evaluate(combined,x,&actual) || actual != expected ||
                        modular::evaluate(modular::compose(convert(first),convert(second)),uint64_t(x))!=uint64_t(actual)) return 2;
                    ++pairs;
                }
            }
    // Independent random offsets exercise nonzero middle terms, large grids,
    // canonical forms and associativity of representable compositions.
    for (unsigned trial = 0u; trial < 250000u; ++trial) {
        map::Map values[3];
        for (auto& value : values) value={int64_t(next()%2000001u)-1000000,
            int64_t(next()%2000001u)-1000000,unsigned(next()%31u)};
        map::Map ab,bc,left,right;
        if (!map::compose(values[0],values[1],&ab) || !map::compose(values[1],values[2],&bc) ||
            !map::compose(ab,values[2],&left) || !map::compose(values[0],bc,&right)) return 3;
        if (!same(left,right)) return 4;
        for (unsigned sample = 0u; sample < 4u; ++sample) {
            const int64_t x=int64_t(next()%2000000001u)-1000000000;
            const Wide expected=evaluate_wide(values[2],evaluate_wide(values[1],evaluate_wide(values[0],x)));
            int64_t actual;
            if (!map::evaluate(left,x,&actual) || actual!=expected ||
                modular::evaluate(modular::compose(modular::compose(convert(values[0]),convert(values[1])),convert(values[2])),uint64_t(x))!=uint64_t(actual)) return 5;
            ++triples;
        }
    }
    for (unsigned trial=0u;trial<1000000u;++trial) {
        const unsigned a=unsigned(next()%31u),b=unsigned(next()%31u);
        const int64_t x=int64_t(next()%2000000001u)-1000000000;
        const int64_t terms=(int64_t(next()%2001u)-1000)*(int64_t(1)<<a);
        const Wide aligned=(Wide(x)/(Wide(1)<<a))*(Wide(1)<<a),sum=aligned+terms;
        const Wide expected=(sum/(Wide(1)<<b))*(Wide(1)<<b);
        map::Map m;
        int64_t actual;
        if (!map::rounding_step(a,b,x<0,sum<0,terms,&m) ||
            !map::evaluate(m,x,&actual) || actual!=expected) return 6;
        ++rounding;
    }
    // Near-overflow operations must either equal the independent wide result
    // or reject without overwriting the destination. Conservative rejection
    // is allowed; callers must fall back rather than changing a numerical gate.
    const int64_t edges[]={INT64_MIN,INT64_MIN+1,-(int64_t(1)<<60),-1,0,1,
        (int64_t(1)<<60)-1,INT64_MAX-1,INT64_MAX};
    for (unsigned shift : {0u,1u,23u,31u,60u,61u,64u})
        for (int64_t before : edges) for (int64_t after : edges) for (int64_t x : edges) {
            const map::Map m{before,after,shift};
            int64_t actual=12345;
            if (map::evaluate(m,x,&actual)) {
                if (shift>60u || !fit(Wide(x)+before) || evaluate_wide(m,x)!=actual) return 7;
            } else { if(actual!=12345)return 8;++rejection; }
            map::Map result=sentinel;
            if (map::compose(m,{after,before,shift},&result)) {
                int64_t candidate;
                if (map::evaluate(result,x,&candidate) &&
                    candidate!=evaluate_wide({after,before,shift},evaluate_wide(m,x))) return 9;
            } else { if(!same(result,sentinel))return 10;++rejection; }
        }
    if (map::compose({0,0,0},{0,0,0},nullptr) ||
        map::rounding_step(61u,0u,false,false,0u,nullptr)) return 11;
    std::printf("{\"kind\":\"quantized_affine_map_host\",\"pair_evaluations\":%llu,\"triple_evaluations\":%llu,\"signed_rounding_steps\":%llu,\"checked_rejections\":%llu,\"mismatches\":0,\"floating_domain_qualified\":false,\"inference_acceptance\":false}\n",
        (unsigned long long)pairs,(unsigned long long)triples,(unsigned long long)rounding,(unsigned long long)rejection);
    return !rejection;
}
