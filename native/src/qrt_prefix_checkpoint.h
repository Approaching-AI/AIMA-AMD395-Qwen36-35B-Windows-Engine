#ifndef QRT_PREFIX_CHECKPOINT_H
#define QRT_PREFIX_CHECKPOINT_H

#include "qrt.h"

/* Optional provider capability. Existing prefix/decode ABI layouts stay
 * unchanged. A miss is successful with prefix_token_count == 0. */
#define QRT_PREFIX_CHECKPOINT_QUERY_VERSION 1u
typedef struct qrt_prefix_checkpoint_query_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t owner_token_count;
    uint32_t input_token_count;
    uint32_t output_token_capacity;
    uint32_t maximum_prefix_tokens;
    uint32_t reserved;
    uint64_t owner_generation;
    uint64_t owner_prompt_digest;
    const qrt_engine_t *owner_engine;
    const uint32_t *input_tokens;
} qrt_prefix_checkpoint_query_v1_t;

typedef struct qrt_prefix_checkpoint_match_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t prefix_token_count;
    uint32_t reserved;
    uint64_t owner_generation;
    uint64_t prefix_digest;
} qrt_prefix_checkpoint_match_v1_t;

typedef int (*qrt_prefix_checkpoint_query_fn)(
    const qrt_prefix_checkpoint_query_v1_t *,
    qrt_prefix_checkpoint_match_v1_t *);
#endif
