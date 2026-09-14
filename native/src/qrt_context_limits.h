#ifndef QRT_CONTEXT_LIMITS_H
#define QRT_CONTEXT_LIMITS_H

/* Runtime storage/request bounds, separate from model configuration metadata.
 * The largest product case is a 256k owner plus 1024 real suffix inputs and
 * 512 generated tokens. Internal attention also reserves the resident tail. */
#define QRT_QWEN36_MAX_PROMPT_TOKENS (262144u + 1024u)
#define QRT_QWEN36_MAX_REQUEST_CONTEXT_TOKENS \
    (QRT_QWEN36_MAX_PROMPT_TOKENS + 512u)
#define QRT_QWEN36_ATTENTION_CAPACITY_TOKENS \
    (((QRT_QWEN36_MAX_PROMPT_TOKENS + 1536u + 1u + 31u) / 32u) * 32u)

#endif
