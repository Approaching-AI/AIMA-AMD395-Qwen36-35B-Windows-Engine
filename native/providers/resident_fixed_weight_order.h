#pragma once
#include "../src/qrt.h"
#include <array>
#include <string>
#include <utility>
#include <vector>

// Shared by the resident loader and its fixed-weight consumers. Keeping the
// exact runtime order here permits one immutable BF16 copy to serve both.
namespace qrt_resident_fixed_order {
inline constexpr std::array<qrt_qwen36_tensor_kind_t, 16> linear_tensor_kinds = {
    QRT_QWEN36_TENSOR_INPUT_NORM,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_QKV,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_Z,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_A,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_B,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_CONV,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_A_LOG,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_DT_BIAS,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_NORM,
    QRT_QWEN36_TENSOR_LINEAR_ATTN_OUT_PROJ,
    QRT_QWEN36_TENSOR_POST_ATTENTION_NORM,
    QRT_QWEN36_TENSOR_MOE_ROUTER,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE_PROJ,
    QRT_QWEN36_TENSOR_MOE_SHARED_UP_PROJ,
    QRT_QWEN36_TENSOR_MOE_SHARED_DOWN
};
inline constexpr std::array<qrt_qwen36_tensor_kind_t, 13>
    full_attention_tensor_kinds = {
    QRT_QWEN36_TENSOR_INPUT_NORM,
    QRT_QWEN36_TENSOR_FULL_ATTN_Q,
    QRT_QWEN36_TENSOR_FULL_ATTN_K,
    QRT_QWEN36_TENSOR_FULL_ATTN_V,
    QRT_QWEN36_TENSOR_FULL_ATTN_Q_NORM,
    QRT_QWEN36_TENSOR_FULL_ATTN_K_NORM,
    QRT_QWEN36_TENSOR_FULL_ATTN_O,
    QRT_QWEN36_TENSOR_POST_ATTENTION_NORM,
    QRT_QWEN36_TENSOR_MOE_ROUTER,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE_PROJ,
    QRT_QWEN36_TENSOR_MOE_SHARED_UP_PROJ,
    QRT_QWEN36_TENSOR_MOE_SHARED_DOWN
    };
inline constexpr std::array<qrt_qwen36_tensor_kind_t, 8>
    fixed_attention_tensor_kinds = {
    QRT_QWEN36_TENSOR_INPUT_NORM,
    QRT_QWEN36_TENSOR_FULL_ATTN_Q,
    QRT_QWEN36_TENSOR_FULL_ATTN_K,
    QRT_QWEN36_TENSOR_FULL_ATTN_V,
    QRT_QWEN36_TENSOR_FULL_ATTN_Q_NORM,
    QRT_QWEN36_TENSOR_FULL_ATTN_K_NORM,
    QRT_QWEN36_TENSOR_FULL_ATTN_O,
    QRT_QWEN36_TENSOR_POST_ATTENTION_NORM
    };
inline constexpr std::array<qrt_qwen36_tensor_kind_t, 5>
    selected_moe_full_v2_tensor_kinds = {
    QRT_QWEN36_TENSOR_MOE_ROUTER,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE,
    QRT_QWEN36_TENSOR_MOE_SHARED_GATE_PROJ,
    QRT_QWEN36_TENSOR_MOE_SHARED_UP_PROJ,
    QRT_QWEN36_TENSOR_MOE_SHARED_DOWN
};
inline bool complete_names(std::vector<std::string>* output) {
    if (!output) return false;
    std::vector<std::string> names;
    auto append = [&](unsigned layer, qrt_qwen36_tensor_kind_t kind) {
        char name[QRT_QWEN36_TENSOR_NAME_CAPACITY] = {};
        if (qrt_qwen36_tensor_name(layer, kind, name, sizeof(name)) != QRT_STATUS_OK)
            return false;
        names.emplace_back(name);
        return true;
    };
    for (unsigned layer = 0; layer < QRT_QWEN36_LAYER_COUNT; ++layer) {
        if (layer % 4u == 3u) {
            for (auto kind : full_attention_tensor_kinds)
                if (!append(layer, kind)) return false;
        } else {
            for (auto kind : linear_tensor_kinds)
                if (!append(layer, kind)) return false;
        }
    }
    if (!append(0u, QRT_QWEN36_TENSOR_LM_HEAD)) return false;
    *output = std::move(names);
    return true;
}
} // namespace qrt_resident_fixed_order
