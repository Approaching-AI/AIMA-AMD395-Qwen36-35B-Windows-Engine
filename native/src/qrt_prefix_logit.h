#ifndef QRT_PREFIX_LOGIT_H
#define QRT_PREFIX_LOGIT_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Optional prefix-result v1 extension. Old providers leave both words zero;
 * old consumers ignore them. The tag binds the layout, and the second word
 * binds the FP32 logit bits to the actual output-zero token, before decode or
 * shadow rollback can replace the frontier. No ABI size or offset changes. */
#define QRT_PREFIX_FIRST_LOGIT_V1_TAG UINT64_C(0x5152544c4f470001)

static inline void qrt_prefix_first_logit_store(
    uint64_t words[2], uint32_t token_id, float logit
) {
    uint32_t bits;
    words[0] = UINT64_C(0);
    words[1] = UINT64_C(0);
    if (!isfinite((double)logit)) {
        return;
    }
    memcpy(&bits, &logit, sizeof(bits));
    words[1] = ((uint64_t)token_id << 32u) | (uint64_t)bits;
    words[0] = QRT_PREFIX_FIRST_LOGIT_V1_TAG;
}

static inline int qrt_prefix_first_logit_read(
    const uint64_t words[2], uint32_t expected_token, float *out_logit
) {
    uint32_t bits;
    float logit;
    if (out_logit == NULL || words[0] != QRT_PREFIX_FIRST_LOGIT_V1_TAG ||
        (uint32_t)(words[1] >> 32u) != expected_token) {
        return 0;
    }
    bits = (uint32_t)words[1];
    memcpy(&logit, &bits, sizeof(logit));
    if (!isfinite((double)logit)) {
        return 0;
    }
    *out_logit = logit;
    return 1;
}

#endif
