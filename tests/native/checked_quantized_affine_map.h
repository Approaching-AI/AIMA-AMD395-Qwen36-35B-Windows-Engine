#pragma once
#include <cstdint>
#include <climits>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_QUANTIZED_MAP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_QUANTIZED_MAP_INLINE inline
#endif

// Independent checked-signed test oracle, not a runtime implementation.
// Production experiments already use sm121_dyadic_carry_scan.h. A map represents
// floor_grid(x + before, 2^shift) + after in a caller-declared common unit.
// The caller must separately establish the K16 exponents, signs, operand
// alignment sums and finite domain before using a map for a floating dot.
namespace qrt_quantized_affine_map {
constexpr unsigned maximum_shift = 60u;
struct Map { int64_t before, after; unsigned shift; };
static_assert(sizeof(int64_t) == 8u && ~int64_t(0) == -1);

QRT_QUANTIZED_MAP_INLINE bool add(int64_t a, int64_t b, int64_t* result) {
    if (!result || (b > 0 && a > INT64_MAX - b) ||
        (b < 0 && a < INT64_MIN - b)) return false;
    *result = a + b;
    return true;
}

// Bit clearing is signed floor on the supported two's-complement targets.
// shift is checked by every public operation before this internal helper.
QRT_QUANTIZED_MAP_INLINE int64_t floor_grid(int64_t value, unsigned shift) {
    uint64_t bits; __builtin_memcpy(&bits, &value, sizeof(bits));
    bits &= ~((uint64_t(1) << shift) - 1u);
    int64_t result; __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

QRT_QUANTIZED_MAP_INLINE bool evaluate(Map map, int64_t input, int64_t* output) {
    if (!output || map.shift > maximum_shift) return false;
    int64_t shifted, result;
    if (!add(input, map.before, &shifted) ||
        !add(floor_grid(shifted, map.shift), map.after, &result)) return false;
    *output = result;
    return true;
}

// Canonical offsets keep before in [0,2^shift). This reduces irrelevant
// offset growth but does not promise that every integer map fits int64.
QRT_QUANTIZED_MAP_INLINE bool canonicalize(Map map, Map* output) {
    if (!output || map.shift > maximum_shift) return false;
    const int64_t base = floor_grid(map.before, map.shift);
    int64_t after;
    if (!add(base, map.after, &after)) return false;
    *output = {map.before - base, after, map.shift};
    return true;
}

// Return second(first(x)). For grids A,B and t=first.after+second.before:
// B>=A: floor_B(floor_A(y)+t) = floor_B(y+floor_A(t)).
// B< A: floor_B(floor_A(y)+t) = floor_A(y)+floor_B(t).
// These identities also hold for negative y/t because both use signed floor.
// Arithmetic/representation rejection leaves the caller's output untouched.
QRT_QUANTIZED_MAP_INLINE bool compose(Map first, Map second, Map* output) {
    if (!output || first.shift > maximum_shift || second.shift > maximum_shift)
        return false;
    int64_t middle;
    if (!add(first.after, second.before, &middle)) return false;
    Map result;
    if (second.shift >= first.shift) {
        if (!add(first.before, floor_grid(middle, first.shift), &result.before))
            return false;
        result.after = second.after;
        result.shift = second.shift;
    } else {
        result.before = first.before;
        if (!add(floor_grid(middle, second.shift), second.after, &result.after))
            return false;
        result.shift = first.shift;
    }
    return canonicalize(result, output);
}

// On a declared negative-input domain, ceil(x/q)*q equals
// floor((x+q-1)/q)*q for integer x. Thus the two truncation points of a K16
// update form a composed map once their signs and grids have been certified.
// product_sum must already contain the ORIGINAL individually aligned terms.
// This helper supplies no prediction, floating range or domain certificate.
QRT_QUANTIZED_MAP_INLINE bool rounding_step(unsigned alignment_shift,
    unsigned output_shift, bool negative_input, bool negative_sum,
    int64_t product_sum, Map* output) {
    if (!output || alignment_shift > maximum_shift || output_shift > maximum_shift)
        return false;
    const Map align{negative_input ? (int64_t(1) << alignment_shift) - 1 : 0,
                    product_sum, alignment_shift};
    const Map finish{negative_sum ? (int64_t(1) << output_shift) - 1 : 0,
                     0, output_shift};
    return compose(align, finish, output);
}
} // namespace qrt_quantized_affine_map
#undef QRT_QUANTIZED_MAP_INLINE
