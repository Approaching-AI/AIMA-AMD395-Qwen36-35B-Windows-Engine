#ifndef QRT_Q1_MOE_HAWKEYE_BF16_ACCUMULATOR_H
#define QRT_Q1_MOE_HAWKEYE_BF16_ACCUMULATOR_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_HAWKEYE_HOST_DEVICE __host__ __device__
#else
#define QRT_HAWKEYE_HOST_DEVICE
#endif

/*
 * Source-only adaptation of the BF16 MMA accumulator characterized by
 * Hawkeye / gpu-simulator at commit
 * 30703fcb309c943a6df5eee0277cb81815deb8f4.
 *
 * Upstream copyright (c) 2026 erez, MIT licensed.  The complete notice is in
 * LICENSE.gpu-simulator beside this file.  This adaptation removes Torch,
 * std::vector, and per-dot allocation, specializes the two published BF16
 * parameter sets, and uses a wide signed host sum whose reachable magnitude
 * is still within the upstream int32_t range.
 *
 * These published Ampere/Lovelace and Hopper characterizations are diagnostic
 * candidates for GB10's legacy mma.sync instruction.  They are not a claim
 * about Blackwell arithmetic.
 */

namespace qrt_q1_moe_hawkeye {

struct Value {
    uint32_t significand;
    int16_t exponent;
    bool negative;
};

QRT_HAWKEYE_HOST_DEVICE inline unsigned int bit_width_u64(uint64_t value) {
    if (value == 0u) {
        return 0u;
    }
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return 64u - static_cast<unsigned int>(
        __builtin_clzll(static_cast<unsigned long long>(value))
    );
#elif defined(_MSC_VER)
    unsigned long index = 0u;
#if defined(_M_X64) || defined(_M_ARM64)
    _BitScanReverse64(&index, value);
    return static_cast<unsigned int>(index + 1u);
#else
    const uint32_t high = static_cast<uint32_t>(value >> 32u);
    if (high != 0u) {
        _BitScanReverse(&index, high);
        return static_cast<unsigned int>(index + 33u);
    }
    _BitScanReverse(&index, static_cast<uint32_t>(value));
    return static_cast<unsigned int>(index + 1u);
#endif
#else
    return 64u - static_cast<unsigned int>(
        __builtin_clzll(static_cast<unsigned long long>(value))
    );
#endif
}

QRT_HAWKEYE_HOST_DEVICE inline Value multiply_bf16(
    uint16_t left,
    uint16_t right,
    int16_t zero_exponent
) {
    const bool negative =
        ((left ^ right) & UINT16_C(0x8000)) != UINT16_C(0);
    uint32_t left_exponent =
        (static_cast<uint32_t>(left) >> 7u) & UINT32_C(0xff);
    uint32_t right_exponent =
        (static_cast<uint32_t>(right) >> 7u) & UINT32_C(0xff);
    const uint32_t left_fraction =
        static_cast<uint32_t>(left) & UINT32_C(0x7f);
    const uint32_t right_fraction =
        static_cast<uint32_t>(right) & UINT32_C(0x7f);
    const uint32_t left_significand =
        left_exponent != 0u
            ? left_fraction | UINT32_C(0x80)
            : left_fraction;
    const uint32_t right_significand =
        right_exponent != 0u
            ? right_fraction | UINT32_C(0x80)
            : right_fraction;
    if (left_exponent == 0u) {
        left_exponent = 1u;
    }
    if (right_exponent == 0u) {
        right_exponent = 1u;
    }
    const uint32_t significand =
        (left_significand * right_significand) << 9u;
    if (significand == 0u) {
        return Value{0u, zero_exponent, negative};
    }
    return Value{
        significand,
        static_cast<int16_t>(
            static_cast<int32_t>(left_exponent) +
            static_cast<int32_t>(right_exponent) - 254
        ),
        negative
    };
}

template <int InternalSignificandWidth, int16_t ZeroExponent>
QRT_HAWKEYE_HOST_DEVICE inline Value group_sum(
    const Value *values,
    size_t count
) {
    static_assert(InternalSignificandWidth >= 24, "invalid significand width");
    static_assert(
        InternalSignificandWidth <= 26,
        "uncharacterized significand width"
    );
    constexpr int kInternalToFp32Shift =
        InternalSignificandWidth - 24;
    constexpr int16_t kFp32MinNonzeroExponent = -126;

    int16_t max_exponent = ZeroExponent;
    for (size_t index = 0u; index < count; ++index) {
        if (values[index].exponent > max_exponent) {
            max_exponent = values[index].exponent;
        }
    }

    int64_t signed_significand = 0;
    for (size_t index = 0u; index < count; ++index) {
        const int shift =
            static_cast<int>(max_exponent) -
            static_cast<int>(values[index].exponent);
        if (shift >= 32) {
            continue;
        }
        const uint64_t aligned =
            (static_cast<uint64_t>(values[index].significand)
                << kInternalToFp32Shift) >>
            static_cast<unsigned int>(shift);
        signed_significand += values[index].negative
            ? -static_cast<int64_t>(aligned)
            : static_cast<int64_t>(aligned);
    }

    const bool negative = signed_significand < 0;
    const uint64_t magnitude = negative
        ? static_cast<uint64_t>(-signed_significand)
        : static_cast<uint64_t>(signed_significand);
    const unsigned int width = bit_width_u64(magnitude);
    if (width == 0u) {
        return Value{0u, ZeroExponent, negative};
    }

    int16_t exponent = static_cast<int16_t>(
        static_cast<int>(max_exponent) +
        static_cast<int>(width) - InternalSignificandWidth
    );
    uint64_t normalized = magnitude;
    if (width > static_cast<unsigned int>(InternalSignificandWidth)) {
        normalized >>= width -
            static_cast<unsigned int>(InternalSignificandWidth);
    } else {
        normalized <<= static_cast<unsigned int>(InternalSignificandWidth) -
            width;
    }
    if (exponent < kFp32MinNonzeroExponent) {
        const unsigned int underflow_shift = static_cast<unsigned int>(
            kFp32MinNonzeroExponent - exponent
        );
        normalized = underflow_shift >= 64u
            ? 0u
            : normalized >> underflow_shift;
        exponent = kFp32MinNonzeroExponent;
    }
    normalized >>= kInternalToFp32Shift;
    if (normalized == 0u) {
        return Value{0u, ZeroExponent, negative};
    }
    return Value{
        static_cast<uint32_t>(normalized),
        exponent,
        negative
    };
}

QRT_HAWKEYE_HOST_DEVICE inline float value_to_float(const Value &value) {
    if (value.significand == 0u) {
        return 0.0f;
    }
    uint32_t bits = value.negative
        ? UINT32_C(0x80000000)
        : UINT32_C(0);
    int exponent_bits = static_cast<int>(value.exponent) + 127;
    if ((value.significand & UINT32_C(0x800000)) == 0u) {
        --exponent_bits;
    }
    bits |= (static_cast<uint32_t>(exponent_bits) & UINT32_C(0xff)) << 23u;
    bits |= value.significand & UINT32_C(0x7fffff);
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    union {
        uint32_t bits;
        float value;
    } converted;
    converted.bits = bits;
    return converted.value;
#else
    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
#endif
}

QRT_HAWKEYE_HOST_DEVICE inline Value value_from_float(
    float value,
    int16_t zero_exponent
) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    union {
        float value;
        uint32_t bits;
    } converted;
    converted.value = value;
    const uint32_t bits = converted.bits;
#else
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
#endif
    const bool negative =
        (bits & UINT32_C(0x80000000)) != UINT32_C(0);
    const uint32_t exponent_bits =
        (bits >> 23u) & UINT32_C(0xff);
    const uint32_t fraction =
        bits & UINT32_C(0x7fffff);
    if (exponent_bits == 0u && fraction == 0u) {
        return Value{0u, zero_exponent, negative};
    }
    return Value{
        exponent_bits == 0u
            ? fraction
            : fraction | UINT32_C(0x800000),
        exponent_bits == 0u
            ? static_cast<int16_t>(-126)
            : static_cast<int16_t>(
                  static_cast<int32_t>(exponent_bits) - 127
              ),
        negative
    };
}

template <
    int InternalSignificandWidth,
    int ProductsPerGroup,
    int16_t ZeroExponent
>
QRT_HAWKEYE_HOST_DEVICE inline float accumulate_bf16_impl(
    float initial_accumulator,
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    static_assert(
        ProductsPerGroup == 4 ||
            ProductsPerGroup == 8 ||
            ProductsPerGroup == 16,
                  "uncharacterized BF16 MMA group");
    Value accumulator =
        value_from_float(initial_accumulator, ZeroExponent);
    size_t offset = 0u;
    while (offset < elements) {
        Value group[ProductsPerGroup + 1];
        group[0] = accumulator;
        const size_t remaining = elements - offset;
        const size_t product_count =
            remaining < static_cast<size_t>(ProductsPerGroup)
                ? remaining
                : static_cast<size_t>(ProductsPerGroup);
        for (size_t index = 0u; index < product_count; ++index) {
            group[index + 1u] = multiply_bf16(
                left[offset + index],
                right[offset + index],
                ZeroExponent
            );
        }
        accumulator = group_sum<
            InternalSignificandWidth,
            ZeroExponent
        >(group, product_count + 1u);
        offset += product_count;
    }

    /*
     * The upstream implementation performs one final single-accumulator
     * group_sum even when K is an exact group multiple.  Retain that endpoint
     * so this source-only specialization remains algorithmically identical.
     */
    return value_to_float(
        group_sum<InternalSignificandWidth, ZeroExponent>(&accumulator, 1u)
    );
}

template <
    int InternalSignificandWidth,
    int ProductsPerGroup,
    int16_t ZeroExponent
>
QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_impl(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return accumulate_bf16_impl<
        InternalSignificandWidth,
        ProductsPerGroup,
        ZeroExponent
    >(0.0f, left, right, elements);
}

QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_ampere_lovelace(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return dot_bf16_impl<25, 8, -132>(left, right, elements);
}

QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_hopper(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return dot_bf16_impl<26, 16, -133>(left, right, elements);
}

QRT_HAWKEYE_HOST_DEVICE inline float accumulate_bf16_ampere_lovelace(
    float initial_accumulator,
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return accumulate_bf16_impl<25, 8, -132>(
        initial_accumulator,
        left,
        right,
        elements
    );
}

QRT_HAWKEYE_HOST_DEVICE inline float accumulate_bf16_hopper(
    float initial_accumulator,
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return accumulate_bf16_impl<26, 16, -133>(
        initial_accumulator,
        left,
        right,
        elements
    );
}

QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_group8_width26(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return dot_bf16_impl<26, 8, -133>(left, right, elements);
}

QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_group16_width25(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return dot_bf16_impl<25, 16, -132>(left, right, elements);
}

QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_group4_width25(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return dot_bf16_impl<25, 4, -132>(left, right, elements);
}

QRT_HAWKEYE_HOST_DEVICE inline float dot_bf16_group8_width24(
    const uint16_t *left,
    const uint16_t *right,
    size_t elements
) {
    return dot_bf16_impl<24, 8, -131>(left, right, elements);
}

}  // namespace qrt_q1_moe_hawkeye

#undef QRT_HAWKEYE_HOST_DEVICE

#endif
