#!/usr/bin/env python3
"""Generate auditable Windows overlays and COFF inputs for the pinned core.

This prepares a standalone q8192 text experiment, not a replacement runtime.
The imported tree is immutable. Only generated files receive OS/ABI changes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/aima_linux"
INVENTORY_SHA256 = "00737aa8d71b9ade06dd663171cf2e02f45e7de514ce79bfd870cafea97345c2"
CONTEXTS = (1024, 2048, 4096, 7168, 7680, 8191, 8192, 16384, 32768)
EXTRA_AOT = (
    "q1024-text-v151", "vl-unified-attention-v0.1.0",
    "unified-attention-decode-v0.1.0", "vl-recompute-w-u-q131-v0.1.0",
    "packed-linear-decode-v0.1.0", "causal-conv-decode-v0.1.0",
    "linear-gated-norm-decode-v0.1.0", "routed-moe-decode-v0.1.0",
    "routed-moe-exact-hybrid-v0.1.0", "vision-attention-v0.2.0",
)
# Same computation units as build-native-runtime.sh. Media transports,
# tokenizer and server frontends are outside this pretokenized experiment.
EXCLUDED = {
    "main.cpp", "aot_kernel_probe.hip.cpp", "aot_registry_probe.hip.cpp",
    "decode_schedule_probe.cpp", "native_aotriton_fmha_provider.hip.cpp",
    "native_chat_protocol.cpp", "native_doctor.cpp", "native_http_server.cpp",
    "native_http_support.cpp", "native_image_decoder.cpp", "native_media.cpp",
    "native_q16384_hybrid_fmha_provider.hip.cpp", "native_remote_media.cpp",
    "native_tokenizer.cpp", "native_video_decoder.cpp", "native_vl_request.cpp",
    "native_vision_block_stack.hip.cpp",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Upstream adaptation anchor changed: {old[:90]!r}")
    return text.replace(old, new)


def replace_re(text, pattern, replacement, count):
    result, changed = re.subn(pattern, lambda _: replacement, text)
    if changed != count:
        raise ValueError(f"Upstream adaptation count changed: {pattern!r}: {changed}")
    return result


def loader_overlay(text):
    for header in ("fcntl.h", "sys/stat.h", "unistd.h"):
        text = replace(text, f"#include <{header}>\n", "")
    text = replace(text, "#include <hip/hip_runtime.h>",
                   '#include <hip/hip_runtime.h>\n#include "file_io.h"')
    start = text.index("ssize_t pread_retry(")
    end = text.index("\n}\n", start) + 3
    text = replace(text, text[start:end],
        "std::int64_t pread_retry(aima_port::file_handle descriptor, void* buffer,\n"
        "                          std::size_t bytes, std::uint64_t offset) {\n"
        "  return aima_port::read_file_at(descriptor, buffer, bytes, offset);\n}")
    text = replace(text,
        "      struct stat status {};\n"
        "      if (stat(paths.back().c_str(), &status) != 0 ||\n"
        "          static_cast<std::uint64_t>(status.st_size) != shard_bytes[shard]) {",
        "      if (!aima_port::file_size_matches(paths.back().c_str(), shard_bytes[shard])) {")
    text = replace(text, "int descriptor = -1;",
                   "aima_port::file_handle descriptor = aima_port::invalid_file;")
    text = replace_re(text, r"open\(paths\[shard\]\.c_str\(\),\s*O_RDONLY \| O_DIRECT \| O_CLOEXEC\)",
                      "aima_port::open_file(paths[shard].c_str(), true)", 1)
    text = replace_re(text, r"open\(paths\[shard\]\.c_str\(\),\s*O_RDONLY \| O_CLOEXEC\)",
                      "aima_port::open_file(paths[shard].c_str(), false)", 3)
    text = replace(text, "descriptor < 0", "descriptor == aima_port::invalid_file", 4)
    text = replace(text, "descriptor >= 0", "descriptor != aima_port::invalid_file")
    text = replace(text, "descriptor = -1;", "descriptor = aima_port::invalid_file;")
    text = replace(text, "close(descriptor)", "aima_port::close_file(descriptor)", 4)
    text = replace(text, "ssize_t amount", "std::int64_t amount")
    text = replace(text,
        "(void)posix_fadvise(descriptor, static_cast<off_t>(offset),\n"
        "                                    static_cast<off_t>(valid_bytes),\n"
        "                                    POSIX_FADV_DONTNEED);",
        "aima_port::drop_file_cache(descriptor, offset, valid_bytes);")
    return replace(text, "std::ofstream output(output_path);",
                   "std::ofstream output(std::filesystem::u8path(output_path));", 2)


def full_prefill_overlay(text):
    text = replace(text,
        "      if (context_tokens != 8192) {",
        '      dynamic_launch_ = reinterpret_cast<DynamicLaunchFn>(\n'
        '          dlsym(handle_, "qrt_ck_fmha_dynamic_bf16_launch"));\n'
        "      if (context_tokens != 8192 &&\n"
        "          (context_tokens > 8192 || dynamic_launch_ == nullptr)) {")
    text = replace(text,
        "  } else {\n    if (query_tokens != metrics_.context_tokens) {",
        "  } else if (query_tokens != 8192 && dynamic_launch_ != nullptr) {\n"
        "    status = dynamic_launch_(q_bf16, k_bf16, v_bf16, output_f32, stream,\n"
        "                             static_cast<unsigned int>(query_tokens));\n"
        "  } else {\n    if (query_tokens != 8192) {")
    return replace(text, "  legacy_launch_ = nullptr;",
                   "  legacy_launch_ = nullptr;\n  dynamic_launch_ = nullptr;")


def rectangular_ck_overlay(header, source):
    """Opt-in mapping of contiguous Linux KV to the existing Windows ABI."""
    header = replace(header, '#include "aima/native_decode_executor.h"',
                     '#include "aima/native_decode_executor.h"\n#include "ck_suffix_adapter.h"')
    header = replace(header, "  DynamicLaunchFn dynamic_launch_ = nullptr;",
                     "  DynamicLaunchFn dynamic_launch_ = nullptr;\n"
                     "  aima_port::CkSuffixLaunch suffix_launch_ = nullptr;")
    source = replace(source,
        '          dlsym(handle_, "qrt_ck_fmha_dynamic_bf16_launch"));\n'
        "      if (context_tokens != 8192 &&\n"
        "          (context_tokens > 8192 || dynamic_launch_ == nullptr)) {",
        '          dlsym(handle_, "qrt_ck_fmha_dynamic_bf16_launch"));\n'
        "      suffix_launch_ = reinterpret_cast<aima_port::CkSuffixLaunch>(\n"
        '          dlsym(handle_, "qrt_ck_fmha_sm121_suffix_bf16_v1"));\n'
        "      metrics_.rectangular_context_abi = suffix_launch_ != nullptr;\n"
        "      if (context_tokens != 8192 && dynamic_launch_ == nullptr) {")
    source = replace(source,
        "    if (rectangular_launch_ == nullptr) {\n"
        "      throw std::runtime_error(\n"
        '          "native FMHA provider lacks the rectangular context ABI");\n'
        "    }\n"
        "    status = rectangular_launch_(\n"
        "        q_bf16, k_bf16, v_bf16, output_f32,\n"
        "        static_cast<unsigned int>(query_tokens),\n"
        "        static_cast<unsigned int>(kv_tokens), stream);",
        "    if (rectangular_launch_ != nullptr) {\n"
        "      status = rectangular_launch_(\n"
        "          q_bf16, k_bf16, v_bf16, output_f32,\n"
        "          static_cast<unsigned int>(query_tokens),\n"
        "          static_cast<unsigned int>(kv_tokens), stream);\n"
        "    } else if (suffix_launch_ != nullptr) {\n"
        "      const auto views = aima_port::ck_suffix_views(\n"
        "          q_bf16, k_bf16, v_bf16, output_f32, query_tokens, kv_tokens);\n"
        "      status = suffix_launch_(views.q, views.prefix_k, views.prefix_v,\n"
        "          views.suffix_k, views.suffix_v, views.output, stream,\n"
        "          views.prefix_tokens, views.query_tokens);\n"
        "    } else {\n"
        "      throw std::runtime_error(\n"
        '          "native FMHA provider lacks the rectangular context ABI");\n'
        "    }")
    source = replace(source, "  dynamic_launch_ = nullptr;",
                     "  dynamic_launch_ = nullptr;\n  suffix_launch_ = nullptr;")
    return header, source


def current_text_decode_overlay(text):
    """Reuse current upstream decode arithmetic with ordinary text positions.

    The upstream boolean named use_mrope also selects projection, recurrence,
    normalization and MoE arithmetic. Its current path uses in-place state;
    selecting only some of these call sites would leave incompatible swaps.
    No M-RoPE plan or visual request is synthesized, and prefill is unchanged.
    """
    text = replace(text,
        "    const NativeDecodePrepareMetrics prepared =\n"
        "        mrope_plan != nullptr\n"
        "            ? prepare_native_decode_step(\n"
        "                  position, rotary_position, metrics.output_token_ids.back(),\n"
        "                  impl_->weights, impl_->decode_invocations)\n"
        "            : prepare_native_decode_step(\n"
        "                  position, metrics.output_token_ids.back(), impl_->weights,\n"
        "                  impl_->decode_invocations);",
        "    // Windows current-text experiment: ordinary text keeps\n"
        "    // rotary_position == position, with the current BF16 RoPE cache.\n"
        "    const NativeDecodePrepareMetrics prepared =\n"
        "        prepare_native_decode_step(\n"
        "            position, rotary_position, metrics.output_token_ids.back(),\n"
        "            impl_->weights, impl_->decode_invocations);")
    return replace(text,
        "        mrope_plan != nullptr, impl_->decode_shared_gate_plan.get(),\n"
        "        mrope_plan != nullptr ? &impl_->decode_cross_layer_norms : nullptr);",
        "        true, impl_->decode_shared_gate_plan.get(),\n"
        "        &impl_->decode_cross_layer_norms);")


def gb10_gdn_overlays(linear, prefill):
    linear = replace(linear, '#include "gb10_convolution.h"',
                     '#include "gb10_convolution.h"\n#include "gb10_gdn.h"')
    linear = replace(linear,
        "  launch_packed_recurrent(layer_index, workspace, invocations, executor, stream);",
        "  aima_port::gb10_decode_gdn(layer_index, projected_qkv, a_projection,\n"
        "      b_projection, recurrent_output, recurrent_state, stream);\n"
        "  metrics.native_pointwise_launches += 3;")
    linear = replace(linear, "  metrics.aot_launches += 2;\n  launch_bf16_wvsplitk(",
                     "  metrics.aot_launches += 1;\n  launch_bf16_wvsplitk(")
    prefill = replace(prefill, '#include "gb10_convolution.h"',
                      '#include "gb10_convolution.h"\n#include "gb10_gdn.h"')
    prefill = replace(prefill,
        "  const auto started = std::chrono::steady_clock::now();",
        "  // This optional experiment owns complete q8192 GDN arithmetic.\n"
        "  // Imported intermediate-AOT observations/checkpoints no longer\n"
        "  // describe this provider and must not read its retired scratch.\n"
        "  if (!q8192_schedule || tokens != 8192 || comparison_tokens != tokens ||\n"
        "      options.seed_layer_input || options.collect_oracle_comparisons ||\n"
        "      !options.checkpoints.empty() || !fixture.empty() ||\n"
        "      !boundary_fixture.empty() || !tail_fixture.empty() || !sequence_fixture.empty())\n"
        "    throw std::invalid_argument(\"GB10 GDN experiment requires unpadded q8192 without AOT fixtures\");\n"
        "  const auto started = std::chrono::steady_clock::now();")
    begin = prefill.index("  launch_attention_aot(2);")
    end = prefill.index("  launch_attention_aot(8);", begin) + len("  launch_attention_aot(8);")
    prefill = replace(prefill, prefill[begin:end],
        "  aima_port::gb10_prefill_gdn(options.layer_index,\n"
        "      invocations.tensor_pointer(base + 1, \"o_ptr\"),\n"
        "      a, b, core, final_state, tokens, options.has_initial_state);\n"
        "  // Two conversion kernels surround one existing FLA provider call.\n"
        "  result.layer.native_pointwise_launches += 2;")
    prefill = replace(prefill, "  if (q8192_schedule) {\n    aima_port::gb10_prefill_convolution(",
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-input-norm-sampled\", h1, 2048, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-qkv-sampled\", qkv, 8192, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-z-sampled\", z, 4096, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-a-sampled\", a, 32, tokens);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-b-sampled\", b, 32, tokens);\n"
        "  if (q8192_schedule) {\n    aima_port::gb10_prefill_convolution(")
    prefill = replace(prefill,
        "  output_plan.launch(gated, output_weight.device_pointer, attention_output);",
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-gated-sampled\", gated, 4096, tokens);\n"
        "  output_plan.launch(gated, output_weight.device_pointer, attention_output);\n"
        "  aima_port::observe_gdn_prefill(options.layer_index, \"prefill-attention-out-sampled\", attention_output, 2048, tokens);",
        count=2)
    prefill = replace(prefill, "      (q8192_schedule ? 1 : 0);",
                      "      (q8192_schedule ? 8 : 0);")
    return linear, prefill


def gb10_projection_overlays(sources, read):
    engine = "native/src/native_resident_engine.hip.cpp"
    sources[engine] = replace(sources[engine], '#include "aima/native_resident_engine.h"',
        '#include "aima/native_resident_engine.h"\n#include "gb10_projection.h"')
    sources[engine] = replace(sources[engine],
        "    const NativeDecodePrepareMetrics prepared =",
        "    aima_port::set_gb10_decode_token(metrics.output_token_ids.back());\n"
        "    const NativeDecodePrepareMetrics prepared =")
    linear = "native/src/native_linear_layer.hip.cpp"
    sources[linear] = replace(sources[linear], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_projection.h"')
    sources[linear] = replace(sources[linear],
        "    launch_prefill_rmsnorm_2048(\n"
        "        input.device_pointer, input_norm_weight.device_pointer, input_norm, 1,\n"
        "        stream);",
        "    if (layer_index == 0) {\n"
        "      aima_port::gb10_embedding_norm(input.device_pointer,\n"
        "          input_norm_weight.device_pointer, input_norm, 1, stream);\n"
        "    } else {\n"
        "      launch_prefill_rmsnorm_2048(input.device_pointer,\n"
        "          input_norm_weight.device_pointer, input_norm, 1, stream);\n"
        "    }")
    prefill = "native/src/native_linear_prefill.hip.cpp"
    sources[prefill] = replace(sources[prefill], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_projection.h"')
    sources[prefill] = replace(sources[prefill],
        "  if (use_vl_rmsnorm) {\n    launch_prefill_rmsnorm_2048(",
        "  if (options.layer_index == 0) {\n"
        "    aima_port::gb10_embedding_norm(x, input_norm_weight.device_pointer, h1, tokens);\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "  } else if (use_vl_rmsnorm) {\n    launch_prefill_rmsnorm_2048(")
    sources[prefill] = replace(sources[prefill],
        "      (q8192_schedule ? 8 : 0);",
        "      (q8192_schedule ? 8 : 0) -\n"
        "      (options.layer_index == 0 && !use_vl_rmsnorm ? 1 : 0);")
    path = "native/src/bf16_wvsplitk.hip.cpp"
    sources[path] = replace(read(path), '#include "aima/bf16_wvsplitk.h"',
        '#include "aima/bf16_wvsplitk.h"\n#include "gb10_projection.h"')
    sources[path] = replace(sources[path],
        '    throw std::invalid_argument("BF16 wvSplitK dimensions exceed int32");\n'
        "  }\n  const int wave_limit = small_wave_limit(m, k);",
        '    throw std::invalid_argument("BF16 wvSplitK dimensions exceed int32");\n'
        "  }\n  aima_port::gb10_projection(weight_mk, activation_1k, bias_m, output_1m, m, k, stream);\n"
        "  return;\n  const int wave_limit = small_wave_limit(m, k);")
    sources[path] = replace(sources[path],
        "  const int active_waves = minimum_divisor(\n"
        "      static_cast<int>(total_rows), cu_count * kYTile, kGroupedWaves);",
        "  aima_port::gb10_projection_group(projections, projection_count, activation_1k, k, stream);\n"
        "  return;\n  const int active_waves = minimum_divisor(\n"
        "      static_cast<int>(total_rows), cu_count * kYTile, kGroupedWaves);")


def gb10_prefill_projection_overlay(text):
    text = replace(text, '#include "aima/bf16_gemm.h"',
        '#include "aima/bf16_gemm.h"\n#include "gb10_prefill_projection.h"\n#include <cstdio>')
    text = replace(text, "  bool bias_epilogue = false;",
        "  bool bias_epilogue = false;\n  bool gb10_prefill = false;\n  bool gb10_wmma = false;")
    text = replace(text, "  impl_->m = m;",
        "  impl_->gb10_prefill = aima_port::gb10_prefill_projection_shape(m, n, k, bias_epilogue);\n"
        "  // FP32 and BF16 destinations cannot share a selected BLAS algorithm.\n"
        "  // Keep the preceding source-geometry validation, then select a new\n"
        "  // algorithm when deriving a different destination type.\n"
        "  if (algorithm_source != nullptr &&\n"
        "      (algorithm_source->impl_->gb10_prefill != impl_->gb10_prefill ||\n"
        "       algorithm_source->impl_->gb10_wmma))\n"
        "    algorithm_source = nullptr;\n"
        "  impl_->m = m;")
    text = replace(text,
        "      if (selected == heuristics.begin() + impl_->heuristic_count) {",
        "      if (selected == heuristics.begin() + impl_->heuristic_count) {\n"
        "        if (impl_->gb10_prefill) {\n"
        "          impl_->gb10_wmma = true;\n"
        '          std::fprintf(stderr, "GB10_PREFILL_WMMA m=%zu n=%zu k=%zu transposed=%u\\n",\n'
        "              m, n, k, unsigned(right_operand_is_transposed));\n"
        "          return;\n"
        "        }")
    for name in ("c", "d"):
        text = replace(text,
            f"hipblasLtMatrixLayoutCreate(&impl_->{name}_layout, HIP_R_16BF,",
            f"hipblasLtMatrixLayoutCreate(&impl_->{name}_layout,\n"
            "                                            impl_->gb10_prefill ? HIP_R_32F : HIP_R_16BF,")
    begin = text.index("void Bf16GemmPlan::launch(")
    end = text.index("void Bf16GemmPlan::launch_with_bias(", begin)
    method = text[begin:end]
    method = replace(method, "  constexpr float alpha = 1.0f;",
        "  void* destination = impl_->gb10_prefill\n"
        "      ? aima_port::gb10_prefill_projection_buffer(impl_->m, impl_->n, impl_->k, stream) : d;\n"
        "  constexpr float alpha = 1.0f;")
    method = replace(method,
        "                 d, impl_->c_layout, d, impl_->d_layout, &impl_->algorithm,",
        "                 destination, impl_->c_layout, destination, impl_->d_layout, &impl_->algorithm,")
    method = replace(method, "  check_blas(hipblasLtMatmul(",
        "  if (impl_->gb10_prefill && aima_port::gb10_prefill_projection_coarse_enabled()) {\n"
        "    aima_port::gb10_prefill_projection_coarse(a, b, impl_->m, impl_->n,\n"
        "        impl_->k, impl_->right_operand_is_transposed, stream);\n"
        "  } else if (impl_->gb10_wmma || (impl_->gb10_prefill &&\n"
        "      aima_port::gb10_prefill_projection_wmma_enabled(impl_->k))) {\n"
        "    aima_port::gb10_prefill_projection_fallback(a, b, impl_->m, impl_->n,\n"
        "        impl_->k, impl_->right_operand_is_transposed, stream);\n"
        "  } else {\n  check_blas(hipblasLtMatmul(")
    method = replace(method, '             "hipblasLtMatmul");',
        '             "hipblasLtMatmul");\n  }\n'
        "  if (impl_->gb10_prefill)\n"
        "    aima_port::gb10_prefill_projection_finish(a, b, d, impl_->m, impl_->n,\n"
        "        impl_->k, impl_->right_operand_is_transposed, stream);")
    return text[:begin] + method + text[end:]


def gb10_normalization_overlays(sources, read):
    path = "native/src/native_full_attention.hip.cpp"
    text = replace(read(path), '#include "aima/native_full_attention.h"',
        '#include "aima/native_full_attention.h"\n#include "gb10_decode_attention.h"')
    text = replace(text,
        '  hipStream_t stream = static_cast<hipStream_t>(stream_value);\n  auto* k_cache =',
        '  if (aima_port::gb10_decode_attention_enabled() && position + 1 != cache_end)\n'
        '    throw std::invalid_argument("GB10 decode query must be the final cache row");\n'
        '  hipStream_t stream = static_cast<hipStream_t>(stream_value);\n  auto* k_cache =')
    sources[path] = replace(text,
        '  check_hip(hipGetLastError(), "write_kv_kernel");\n',
        '  check_hip(hipGetLastError(), "write_kv_kernel");\n'
        "  if (aima_port::gb10_decode_attention_enabled()) {\n"
        "    aima_port::gb10_decode_attention(q, k_cache, v_cache, attention, cache_end, stream);\n"
        "    NativeFullAttentionCoreMetrics metrics;\n"
        "    metrics.layer_index = layer_index;\n"
        "    metrics.cache_end = cache_end;\n"
        "    metrics.pv_splits = 1;\n"
        "    metrics.native_kernel_launches = 4;\n"
        "    return metrics;\n"
        "  }\n")
    path = "native/src/native_linear_prefill.hip.cpp"
    text = replace(sources[path], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_normalization.h"')
    text = replace(text,
        "    launch_bf16_rowwise_invstd_128(core, invstd, tokens * kLinearHeads);\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "    executor.launch(launches[base + 9]);",
        "    aima_port::gb10_gated_norm(core, z, linear_norm_weight.device_pointer, gated, tokens);\n"
        "    ++result.layer.native_pointwise_launches;")
    text = replace(text,
        "  if (use_vl_rmsnorm) {\n    launch_prefill_add_rmsnorm_2048(",
        "  if (q8192_schedule) {\n"
        "    aima_port::gb10_residual_norm(attention_output, x,\n"
        "        post_attention_norm_weight.device_pointer, after_attention, h2, tokens);\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "  } else if (use_vl_rmsnorm) {\n    launch_prefill_add_rmsnorm_2048(")
    text = replace(text, "      (q8192_schedule ? 8 : 0) -",
        "      (q8192_schedule ? (use_vl_rmsnorm ? 9 : 10) : 0) -")
    sources[path] = text
    path = "native/src/native_pointwise.hip.cpp"
    text = replace(read(path), '#include "aima/native_pointwise.h"',
        '#include "aima/native_pointwise.h"\n#include "gb10_normalization.h"')
    text = replace(text,
        '        "native prefill residual RMSNorm geometry is invalid");\n  }\n',
        '        "native prefill residual RMSNorm geometry is invalid");\n  }\n'
        "  if (token_count <= 8192) {\n"
        "    aima_port::gb10_residual_norm(input_bf16, residual_bf16, weight_bf16,\n"
        "        residual_output_bf16, norm_output_bf16, token_count, stream_value);\n"
        "    return;\n  }\n")
    sources[path] = text
    path = "native/src/native_full_prefill.hip.cpp"
    text = replace(sources[path], '#include "aima/native_full_prefill.h"',
        '#include "aima/native_full_prefill.h"\n#include "gb10_normalization.h"')
    text = replace(text,
        '  diagnostic_stage("after_output_projection");\n  if (use_mrope) {',
        '  diagnostic_stage("after_output_projection");\n'
        "  if (execution_tokens == 8192) {\n"
        "    aima_port::gb10_residual_norm(projected_attention, layer_input,\n"
        "        post_attention_norm_weight.device_pointer, after_attention,\n"
        "        post_attention_norm, execution_tokens);\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "  } else if (use_mrope) {")
    sources[path] = replace(text,
        "  if (use_mrope) {\n    launch_full_attention_head_norm_mrope_prefill(",
        "  if (aima_port::gb10_full_head_norm_rope_enabled()) {\n"
        "    if (use_mrope) throw std::invalid_argument(\"GB10 full-head prototype requires ordinary text positions\");\n"
        "    aima_port::gb10_full_head_norm_rope(q_gate, raw_k, split_projections ? nullptr : raw_v,\n"
        "        q_norm_weight.device_pointer, k_norm_weight.device_pointer, q, normalized_k,\n"
        "        split_projections ? nullptr : normalized_v, execution_tokens,\n"
        "        split_projections ? 8192 : 9216, split_projections ? 512 : 9216,\n"
        "        split_projections ? 0 : 9216, options.cache_position_start);\n"
        "  } else if (use_mrope) {\n    launch_full_attention_head_norm_mrope_prefill(")
    path = "native/src/native_routed_moe.hip.cpp"
    text = replace(read(path), '#include "aima/native_routed_moe.h"',
        '#include "aima/native_routed_moe.h"\n#include "gb10_normalization.h"')
    begin = text.index("  hipLaunchKernelGGL(\n      native_decode_moe_tail_next_rmsnorm_kernel,")
    end = text.index('\n}', begin)
    text = replace(text, text[begin:end],
        "  // Preserve the live residual before the tail writes a possibly aliased carrier.\n"
        "  const void* saved_residual = aima_port::gb10_preserve_decode_residual(residual_bf16, stream);\n"
        "  launch_native_decode_moe_tail(weighted_expert_outputs_bf16,\n"
        "      fused_shared_input_bf16, shared_down_bf16, residual_bf16,\n"
        "      routed_output_bf16, shared_output_bf16, combined_output_bf16, output_bf16, stream);\n"
        "  aima_port::gb10_residual_norm(combined_output_bf16, saved_residual,\n"
        "      next_norm_weight_bf16, output_bf16, next_norm_output_bf16, 1, stream);")
    sources[path] = text
    path = "native/src/native_linear_layer.hip.cpp"
    text = replace(sources[path], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_normalization.h"')
    text = replace(text,
        "  // Current vLLM uses one row per Triton program for decode RMSNormGated.\n"
        "  // The attention-output scratch is still dead here and supplies the 32-value\n"
        "  // FP32 Rstd side output without growing the resident workspace.\n",
        "  // Preserve original short-row reduction and FP32 SiLU arithmetic.\n")
    text = replace(text,
        "  launch_current_linear_gated_norm(\n"
        "      recurrent_output, z_projection, linear_norm_weight.device_pointer,\n"
        "      gated.device_pointer, attention_output.device_pointer, executor, stream);",
        "  aima_port::gb10_gated_norm(recurrent_output, z_projection,\n"
        "      linear_norm_weight.device_pointer, gated.device_pointer, 1, stream);")
    text = replace(text,
        "  metrics.aot_launches += 1;\n  launch_bf16_wvsplitk(",
        "  ++metrics.native_pointwise_launches;\n  launch_bf16_wvsplitk(")
    text = replace(text,
        '    ++metrics.native_pointwise_launches;\n  }\n  observe_boundary(tail_observer, "shared_gate_logits",',
        '    metrics.native_pointwise_launches += next_input_norm != nullptr ? 2 : 1;\n  }\n'
        '  if (next_input_norm != nullptr)\n'
        '    observe_boundary(tail_observer, "next_input_norm", next_input_norm->output_bf16,\n'
        '                     kHidden * sizeof(__hip_bfloat16), DecodeTensorDtype::kBfloat16);\n'
        '  observe_boundary(tail_observer, "shared_gate_logits",')
    sources[path] = text
    path = "native/src/native_full_layer.hip.cpp"
    text = replace(read(path),
        "  ++metrics.native_pointwise_launches;\n  if (attention_observer != nullptr) {",
        "  metrics.native_pointwise_launches += use_mrope && next_input_norm != nullptr ? 2 : 1;\n"
        "  if (attention_observer != nullptr) {")
    text = replace(text, '#include "aima/native_full_layer.h"',
        '#include "aima/native_full_layer.h"\n#include "gb10_normalization.h"')
    sources[path] = replace(text,
        "  if (use_mrope) {\n    if ((mrope_cosine_fp32 == nullptr)",
        "  if (aima_port::gb10_full_head_norm_rope_enabled()) {\n"
        "    aima_port::gb10_full_head_norm_rope(qkv.device_pointer, raw_k, nullptr,\n"
        "        q_norm_weight.device_pointer, k_norm_weight.device_pointer,\n"
        "        q.device_pointer, k.device_pointer, nullptr, 1, 9216, 9216, 0, position, stream);\n"
        "    ++metrics.native_pointwise_launches;\n"
        "  } else if (use_mrope) {\n    if ((mrope_cosine_fp32 == nullptr)")
    # The existing output-only sampler gathers live prefill MoE boundaries.
    # No captured value is an input; GDN's FP32 output scratch is dead here.
    path = "native/src/native_moe_prefill.hip.cpp"
    text = replace(read(path), '#include "aima/native_moe_prefill.h"',
        '#include "aima/native_moe_prefill.h"\n#include "gb10_gdn.h"')
    def observe(name, value, columns):
        return (f'  aima_port::observe_gdn_prefill(options.layer_index, "prefill-{name}-sampled", '
                f'{value}, {columns}, tokens);\n')
    text = replace(text, "  shared_gate_plan.launch(h2, shared_gate_weight.device_pointer,",
        observe("post-attention-norm", "h2", 2048) +
        observe("post-attention-residual", "after_attention", 2048) +
        "  shared_gate_plan.launch(h2, shared_gate_weight.device_pointer,")
    for anchor, samples in (
        ('  diagnostic_stage("after_shared_gate");', [("shared-gate", "shared_gate", 1)]),
        ('  diagnostic_stage("after_shared_gate_projection");', [("shared-gate-projection", "shared_projected_gate", 512)]),
        ('  diagnostic_stage("after_shared_up_projection");', [("shared-up-projection", "shared_projected_up", 512)]),
        ("  shared_down_plan.launch(shared_activated,", [("shared-activation", "shared_activated", 512)]),
        ('  diagnostic_stage("after_shared_down_projection");', [("shared-down", "shared_down", 2048)]),
        ("  if (logical_router_gemm_plans != nullptr) {\n    check_hip(hipMemsetAsync(", [("shared-output", "shared_scaled", 2048)]),
        ('  diagnostic_stage("after_router_projection");', [("router", "router_logits", 256)]),
        ('  diagnostic_stage("after_expert_sum");', [("routed-output", "routed_moe", 2048)]),
        ('  diagnostic_stage("after_output_add");', [("combined-moe", "combined_moe", 2048), ("layer-output", "layer_output", 2048)]),
    ):
        code = "".join(observe(*sample) for sample in samples)
        # Capture a completed stage after its marker, or before its consumer.
        text = replace(text, anchor, anchor + "\n" + code if "diagnostic_stage" in anchor else code + anchor)
    sources[path] = text



def gb10_moe_overlays(sources, read):
    path = "native/src/native_moe_prefill.hip.cpp"
    text = replace(sources[path], '#include "gb10_gdn.h"',
        '#include "gb10_gdn.h"\n#include "gb10_moe.h"')
    text = replace(text, "  const std::size_t bucket_tokens = workspace.context_tokens();",
        "  if (options.seed_post_attention || options.collect_oracle_comparisons ||\n"
        "      options.run_routing_diagnostic || !options.boundary_oracle_dir.empty() ||\n"
        "      !options.chain_output_oracle_dir.empty())\n"
        "    throw std::invalid_argument(\"GB10 MoE requires live unseeded model operands\");\n"
        "  const std::size_t bucket_tokens = workspace.context_tokens();")
    anchor = "  // Match PyTorch's qualified hipBLASLt N=1 preference exactly."
    code = """  const auto provider_started = std::chrono::steady_clock::now();
  aima_port::gb10_prefill_moe(options.layer_index, h2, after_attention,
      router_weight.device_pointer, invocations.tensor_pointer(moe_first, "b_ptr"),
      invocations.tensor_pointer(moe_first + 1, "b_ptr"),
      shared_gate_weight.device_pointer, shared_gate_proj_weight.device_pointer,
      shared_up_proj_weight.device_pointer, shared_down_proj_weight.device_pointer,
      layer_output, tokens);
  // The two conversions are native launches; provider-internal dispatch is
  // opaque here and its completed work stays in the enclosing TTFT interval.
  result.layer.native_pointwise_launches = 2;
  result.layer.wall_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - provider_started).count();
  return result;

"""
    sources[path] = replace(text, anchor, code + anchor)
    path = "native/src/native_linear_prefill.hip.cpp"
    text = replace(sources[path], '#include "gb10_normalization.h"',
        '#include "gb10_normalization.h"\n#include "gb10_moe.h"')
    text = replace(text, "  if (options.layer_index == 0) {\n    aima_port::gb10_embedding_norm",
        "  if (aima_port::gb10_moe_input_norm(options.layer_index, x,\n"
        "          input_norm_weight.device_pointer, h1, tokens)) {\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "  } else if (options.layer_index == 0) {\n    aima_port::gb10_embedding_norm")
    sources[path] = replace(text,
        "      (options.layer_index == 0 && !use_vl_rmsnorm ? 1 : 0);",
        "      (!use_vl_rmsnorm ? 1 : 0);")
    path = "native/src/native_full_prefill.hip.cpp"
    text = replace(sources[path], '#include "gb10_normalization.h"',
        '#include "gb10_normalization.h"\n#include "gb10_moe.h"')
    sources[path] = replace(text, '  diagnostic_stage("before_input_norm");\n  if (use_mrope) {',
        '  diagnostic_stage("before_input_norm");\n'
        "  if (aima_port::gb10_moe_input_norm(options.layer_index, layer_input,\n"
        "          input_norm_weight.device_pointer, normalized_input, execution_tokens)) {\n"
        "    ++result.layer.native_pointwise_launches;\n"
        "  } else if (use_mrope) {")
    path = "native/src/native_resident_engine.hip.cpp"
    text = replace(sources[path], '#include "gb10_projection.h"',
        '#include "gb10_projection.h"\n#include "gb10_moe.h"')
    text = replace(text, "  ~Impl() {", "  ~Impl() {\n    aima_port::gb10_moe_release_weights();")
    text = replace(text, "  impl_->metrics.command_to_ready_wall_ms = elapsed_ms(started);",
        """  const uint16_t* moe_gate_up[40]{};
  const uint16_t* moe_down[40]{};
  for (unsigned layer = 0; layer < 40; ++layer) {
    const auto prefix = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.";
    const auto* gu = impl_->weights.find(prefix + "gate_up_proj");
    const auto* dn = impl_->weights.find(prefix + "down_proj");
    if (!gu || !dn || !gu->device_pointer || !dn->device_pointer ||
        gu->payload_bytes != 256ULL * 1024ULL * 2048ULL * 2ULL ||
        dn->payload_bytes != 256ULL * 2048ULL * 512ULL * 2ULL)
      throw std::runtime_error("GB10 MoE model registration shape differs");
    moe_gate_up[layer] = static_cast<const uint16_t*>(gu->device_pointer);
    moe_down[layer] = static_cast<const uint16_t*>(dn->device_pointer);
  }
  aima_port::gb10_moe_register_weights(moe_gate_up, moe_down, 40);
  impl_->metrics.command_to_ready_wall_ms = elapsed_ms(started);""")
    sources[path] = text
    path = "native/src/native_decode_runner.hip.cpp"
    text = replace(read(path), '#include "aima/native_decode_runner.h"',
        '#include "aima/native_decode_runner.h"\n#include "gb10_moe.h"')
    text = replace(text, "  executor.launch(launches[400], stream);",
        """  const auto* gb10_weight = weights.find("model.language_model.norm.weight");
  const auto* gb10_output = workspace.find("rmsnorm_final_output");
  if (!gb10_weight || gb10_weight->payload_bytes != 4096 || !gb10_output ||
      gb10_output->payload_bytes < 4096 || !gb10_output->device_pointer)
    throw std::runtime_error("GB10 terminal normalization binding differs");
  const bool gb10_terminal = aima_port::gb10_moe_terminal_norm(final_hidden_row,
      gb10_weight->device_pointer, gb10_output->device_pointer, stream);
  if (!gb10_terminal) executor.launch(launches[400], stream);""")
    sources[path] = replace(text, "  metrics.aot_launches = 2;", "  metrics.aot_launches = gb10_terminal ? 1 : 2;")


def gb10_decode_moe_overlays(sources):
    for name, guard, boundary in (
        ("native_linear_layer", "use_current_vllm_projections",
         '  if (next_input_norm != nullptr)\n    observe_boundary(tail_observer, "next_input_norm",'),
        ("native_full_layer", "use_mrope", '  if (attention_observer != nullptr) {')):
        path = f"native/src/{name}.hip.cpp"
        text = replace(sources[path], f'#include "aima/{name}.h"',
            f'#include "aima/{name}.h"\n#include "gb10_decode_moe.h"')
        if '#include "gb10_normalization.h"' not in text:
            text = replace(text, '#include "gb10_decode_moe.h"',
                '#include "gb10_decode_moe.h"\n#include "gb10_normalization.h"')
        begin = text.index("  hipStream_t shared_expert_stream = stream;")
        end = text.index(boundary, begin)
        original = text[begin:end]
        branch = (
            "  if (aima_port::gb10_decode_moe_enabled()) {\n"
            f'    if (!{guard}) throw std::invalid_argument("GB10 decode MoE requires current text decode");\n'
            "    const aima_port::Gb10DecodeMoeWeights weights{\n"
            "        router_weight.device_pointer, shared_expert_gate_weight.device_pointer,\n"
            "        shared_gate_weight.device_pointer, shared_up_weight.device_pointer,\n"
            "        shared_down_weight.device_pointer, routed_gate_up_weight.device_pointer,\n"
            "        routed_down_weight.device_pointer};\n"
            "    const aima_port::Gb10DecodeMoeBuffers buffers{\n"
            "        shared_input.device_pointer, activated.device_pointer, shared_down.device_pointer,\n"
            "        shared_scaled.device_pointer, router_logits.device_pointer,\n"
            "        router_indices.device_pointer, router_weights.device_pointer,\n"
            "        routed_gate_up.device_pointer, routed_activation.device_pointer,\n"
            "        routed_weighted.device_pointer, routed_moe.device_pointer, combined_moe.device_pointer};\n"
            "    aima_port::gb10_decode_moe(layer_index, post_attention_norm, weights, buffers, stream);\n"
            "    metrics.native_projection_launches += 7;\n"
            "    metrics.native_pointwise_launches += 3;\n"
            "    if (next_input_norm != nullptr) {\n"
            "      aima_port::gb10_residual_norm(combined_moe.device_pointer, after_attn.device_pointer,\n"
            "          next_input_norm->weight_bf16, output.device_pointer,\n"
            "          next_input_norm->output_bf16, 1, stream);\n"
            "    } else {\n"
            "      if (layer_index != 39) throw std::invalid_argument(\"GB10 decode MoE requires next-layer normalization\");\n"
            "      aima_port::gb10_decode_moe_save_terminal(combined_moe.device_pointer,\n"
            "          after_attn.device_pointer, output.device_pointer, stream);\n"
            "      launch_bf16_add(combined_moe.device_pointer, after_attn.device_pointer,\n"
            "          output.device_pointer, kHidden, stream);\n"
            "    }\n"
            "    ++metrics.native_pointwise_launches;\n"
            "  } else {\n" + "".join("  " + line for line in original.splitlines(keepends=True)) + "  }\n")
        sources[path] = text[:begin] + branch + text[end:]
    path = "native/src/native_decode_runner.hip.cpp"
    text = replace(sources[path], '#include "gb10_moe.h"',
        '#include "gb10_moe.h"\n#include "gb10_decode_moe.h"')
    sources[path] = replace(text,
        "  const bool gb10_terminal = aima_port::gb10_moe_terminal_norm(final_hidden_row,",
        "  const bool gb10_terminal = aima_port::gb10_decode_moe_terminal_norm(final_hidden_row,\n"
        "      gb10_weight->device_pointer, gb10_output->device_pointer, stream) ||\n"
        "      aima_port::gb10_moe_terminal_norm(final_hidden_row,")


def make_overlays(*, rectangular_ck=False, current_text_decode=False,
                  gb10_convolution=False, gb10_gdn=False, gb10_projections=False,
                  gb10_prefill_projections=False, gb10_normalization=False, gb10_moe=False):
    if gb10_convolution and not current_text_decode:
        raise ValueError("GB10 convolution requires current text decode ownership")
    if gb10_gdn and not gb10_convolution:
        raise ValueError("GB10 GDN requires GB10 convolution and current text decode")
    if gb10_projections and not gb10_gdn:
        raise ValueError("GB10 projections require the GB10 GDN experiment")
    if gb10_prefill_projections and not gb10_projections:
        raise ValueError("GB10 prefill projections require the GB10 projection experiment")
    if gb10_normalization and not gb10_projections:
        raise ValueError("GB10 normalization requires GB10 projection and GDN table ownership")
    if gb10_moe and not gb10_normalization:
        raise ValueError("GB10 MoE requires the GB10 normalization experiment")
    read = lambda p: (UPSTREAM / p).read_text(encoding="utf-8")
    sources = {}
    loader = "benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp"
    sources[loader] = loader_overlay(read(loader))
    header = "native/include/aima/native_resident_engine.h"
    sources[header] = replace(read(header), "  bool first_token_certified = false;",
        "  // Output-only observation of the certified greedy BF16 logit.\n"
        "  float first_token_raw_logit = 0.0f;\n  bool first_token_certified = false;")
    engine = "native/src/native_resident_engine.hip.cpp"
    sources[engine] = replace(read(engine),
        "  metrics.first_token_certified = first.certified;",
        "  metrics.first_token_raw_logit = first.top1_logit;\n"
        "  metrics.first_token_certified = first.certified;")
    sources[engine] = replace(sources[engine],
        "        default_fmha_provider(4096), 4096);",
        "        options.ck_provider.empty() ? default_fmha_provider(4096)\n"
        "                                    : options.ck_provider, 4096);")
    if current_text_decode:
        sources[engine] = current_text_decode_overlay(sources[engine])
    if gb10_convolution:
        linear = "native/src/native_linear_layer.hip.cpp"
        sources[linear] = replace(read(linear),
            '#include "aima/native_linear_layer.h"',
            '#include "aima/native_linear_layer.h"\n#include "gb10_convolution.h"')
        sources[linear] = replace(sources[linear],
            "  launch_current_causal_conv(projected_qkv, conv_weight.device_pointer,\n"
            "                             conv_state_before, direct_conv_state_index,\n"
            "                             executor, stream);",
            "  aima_port::gb10_decode_convolution(\n"
            "      projected_qkv, conv_weight.device_pointer, conv_state_before, stream);\n"
            "  ++metrics.native_pointwise_launches;")
        sources[linear] = replace(sources[linear],
            "  metrics.aot_launches += 3;", "  metrics.aot_launches += 2;")
        linear_prefill = "native/src/native_linear_prefill.hip.cpp"
        sources[linear_prefill] = replace(read(linear_prefill),
            '#include "aima/native_linear_prefill.h"',
            '#include "aima/native_linear_prefill.h"\n#include "gb10_convolution.h"')
        sources[linear_prefill] = replace(sources[linear_prefill],
            "  launch_attention_aot(1);",
            "  if (q8192_schedule) {\n"
            "    aima_port::gb10_prefill_convolution(\n"
            "        qkv, invocations.tensor_pointer(base + 1, \"w_ptr\"),\n"
            "        invocations.tensor_pointer(base + 1, \"initial_states_ptr\"),\n"
            "        invocations.tensor_pointer(base + 1, \"o_ptr\"),\n"
            "        tokens, options.has_initial_state);\n"
            "    result.layer.native_pointwise_launches += 2;\n"
            "  } else {\n"
            "    launch_attention_aot(1);\n"
            "  }")
        sources[linear_prefill] = replace(sources[linear_prefill],
            "      attention_launches - (use_vl_rmsnorm ? 2 : 0);",
            "      attention_launches - (use_vl_rmsnorm ? 2 : 0) -\n"
            "      (q8192_schedule ? 1 : 0);")
        if gb10_gdn:
            sources[linear], sources[linear_prefill] = gb10_gdn_overlays(
                sources[linear], sources[linear_prefill])
    if gb10_projections:
        gb10_projection_overlays(sources, read)
    if gb10_prefill_projections:
        path = "native/src/bf16_gemm.hip.cpp"
        sources[path] = gb10_prefill_projection_overlay(read(path))
        path = "native/src/native_linear_prefill.hip.cpp"
        sources[path] = replace(sources[path], '#include "gb10_projection.h"',
            '#include "gb10_projection.h"\n#include "gb10_prefill_projection.h"')
        sources[path] = replace(sources[path],
            "  output_plan.launch(gated, output_weight.device_pointer, attention_output);",
            "  {\n"
            "    aima_port::Gb10PrefillLinearOutputScope linear_out_scope(tokens, options.layer_index);\n"
            "    output_plan.launch(gated, output_weight.device_pointer, attention_output);\n"
            "  }", count=2)
    weights = "native/src/native_weight_store.hip.cpp"
    sources[weights] = replace(read(weights), "shard_storage.push_back(path.string());",
                             "shard_storage.push_back(path.u8string());")
    sources[weights] = replace(sources[weights], "options.native_report.c_str());",
                             "options.native_report.u8string().c_str());")
    header = "native/include/aima/native_full_prefill.h"
    sources[header] = replace(read(header), "  using ReleaseFn = int (*)();",
        "  using DynamicLaunchFn = int (*)(const void*, const void*, const void*,\n"
        "                                    void*, void*, unsigned int);\n"
        "  using ReleaseFn = int (*)();")
    sources[header] = replace(sources[header], "  LegacyLaunchFn legacy_launch_ = nullptr;",
        "  LegacyLaunchFn legacy_launch_ = nullptr;\n"
        "  DynamicLaunchFn dynamic_launch_ = nullptr;")
    prefill = "native/src/native_full_prefill.hip.cpp"
    sources[prefill] = full_prefill_overlay(read(prefill))
    # The imported resident owner already loads this embedded plan for VL.
    # Trial its identical contiguous Q/K/V ABI on a cold ordinary-text q8192
    # request; keep positional transforms, cache writes and decode separate.
    sources[prefill] = replace(sources[prefill], "#include <cstdio>",
        "#include <cstdio>\n#include <cstdlib>")
    sources[prefill] = replace(sources[prefill],
        "  const bool use_vl_unified_attention =\n"
        "      use_mrope && active_tokens != tokens &&\n"
        "      options.cache_position_start == 0;",
        '  const char* native_text_setting = std::getenv("AIMA_PORT_NATIVE_ATTENTION_PREFILL");\n'
        "  const bool use_native_text_attention = native_text_setting != nullptr &&\n"
        "      native_text_setting[0] == '1' && native_text_setting[1] == '\\0' &&\n"
        "      !use_mrope && tokens == 8192 && active_tokens == tokens &&\n"
        "      options.cache_position_start == 0;\n"
        "  const bool use_vl_unified_attention = use_native_text_attention ||\n"
        "      (use_mrope && active_tokens != tokens &&\n"
        "       options.cache_position_start == 0);")
    sources[prefill] = replace(sources[prefill],
        "    ++result.layer.native_vl_unified_attention_launches;\n",
        "    ++result.layer.native_vl_unified_attention_launches;\n"
        "    if (use_native_text_attention)\n"
        '      std::fprintf(stderr, "{\\\"event\\\":\\\"native_attention_prefill\\\",'
        '\\\"layer\\\":%zu,\\\"queries\\\":%zu,\\\"kv_tokens\\\":%zu,'
        '\\\"embedded_kernel\\\":\\\"85618d461d690f5f7732dfd55b693df8c15642737aa2c1cf66b0674ffd4d7a30\\\"}\\n",\n'
        "          options.layer_index, active_tokens, options.cache_position_start + active_tokens);\n")
    if gb10_prefill_projections:
        sources[prefill] = replace(sources[prefill], '#include "aima/native_full_prefill.h"',
            '#include "aima/native_full_prefill.h"\n#include "gb10_prefill_projection.h"')
        sources[prefill] = replace(sources[prefill],
            "  output_plan.launch(gated, output_weight.device_pointer,\n"
            "                     projected_attention);",
            "  {\n"
            "    aima_port::Gb10PrefillFullOutputScope full_out_scope(execution_tokens, options.layer_index);\n"
            "    output_plan.launch(gated, output_weight.device_pointer, projected_attention);\n"
            "  }")
    if rectangular_ck:
        sources[header], sources[prefill] = rectangular_ck_overlay(
            sources[header], sources[prefill])
    if gb10_normalization:
        gb10_normalization_overlays(sources, read)
    if gb10_moe:
        gb10_moe_overlays(sources, read)
        gb10_decode_moe_overlays(sources)
    # These two upstream enum-to-string functions have no media I/O dependency.
    media = read("native/src/native_media.cpp")
    names = media[media.index("std::string_view native_media_kind_name("):]
    sources["native/src/media_names.cpp"] = (
        '// SPDX-License-Identifier: Apache-2.0\n'
        '// Copyright 2026 Approaching AI Authors\n'
        '#include "aima/native_media.h"\nnamespace aima {\n' + names)
    return sources


def verify_import():
    raw = (UPSTREAM / "UPSTREAM.json").read_bytes()
    if digest(raw) != INVENTORY_SHA256:
        raise ValueError("Pinned upstream inventory changed")
    inventory = json.loads(raw)
    for item in inventory["files"]:
        data = (UPSTREAM / item["path"]).read_bytes()
        if len(data) != item["bytes"] or digest(data) != item["sha256"]:
            raise ValueError("Imported upstream file changed: " + item["path"])
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--windows-rectangular-ck", action="store_true",
                        help="Generate the optional Windows suffix ABI mapping; not model-qualified")
    parser.add_argument("--current-text-decode", action="store_true",
                        help="Use current upstream decode arithmetic for text; not model-qualified")
    parser.add_argument("--gb10-convolution", action="store_true",
                        help="Use RNE BF16 convolution products and the qualified SiLU table")
    parser.add_argument("--gb10-gdn", action="store_true",
                        help="Use existing Windows FLA and original GB10 Q2 decode arithmetic")
    parser.add_argument("--gb10-projections", action="store_true",
                        help="Use SM121 decode projections and full-vocabulary embedding RMS scales")
    parser.add_argument("--gb10-prefill-projections", action="store_true",
                        help="Use FP32 q8192 GEMM outputs with existing SM121 staged exact replay")
    parser.add_argument("--gb10-normalization", action="store_true",
                        help="Use GB10 prefill/decode gated norm and unrounded residual variance")
    parser.add_argument("--gb10-moe", action="store_true",
                        help="Use the qualified Windows prefill MoE provider and live FP32 carriers")
    args = parser.parse_args()
    inventory = verify_import()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Output already exists; preserve it and choose a fresh directory")
    overlays = make_overlays(rectangular_ck=args.windows_rectangular_ck,
                             current_text_decode=args.current_text_decode,
                             gb10_convolution=args.gb10_convolution, gb10_gdn=args.gb10_gdn,
                             gb10_projections=args.gb10_projections,
                             gb10_prefill_projections=args.gb10_prefill_projections,
                             gb10_normalization=args.gb10_normalization, gb10_moe=args.gb10_moe)
    out.mkdir(parents=True)
    adapted = []
    for relative, text in overlays.items():
        target = out / "overlay" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        original = UPSTREAM / relative
        adapted.append(dict(path=relative, output_sha256=digest(target.read_bytes()),
                            original_sha256=digest(original.read_bytes()) if original.exists() else None))
    def run_generator(name, arguments):
        result = subprocess.run([sys.executable, str(UPSTREAM / "scripts" / name), *map(str, arguments)],
                                check=True, capture_output=True, text=True, timeout=90)
        return json.loads(result.stdout)
    aot = UPSTREAM / "native/aot/gfx1151"
    contexts = [aot / (f"q{n}-output2" if n == 8192 else f"q{n}-output1") for n in CONTEXTS]
    manifests = [d / "manifest.json" for d in contexts + [aot / name for name in EXTRA_AOT]]
    aot_args = [argument for p in manifests for argument in ("--manifest", p)]
    aot_report = run_generator("generate-native-aot-registry.py", aot_args + [
        "--output-cpp", out / "aot_registry.cpp", "--output-plan", out / "aot_objects.tsv"])
    decode = aot / "q8192-output2"
    reports = [run_generator("generate-native-decode-registry.py", [
        "--schedule", decode / "decode-schedule.json", "--aot-manifest", decode / "manifest.json",
        "--output-cpp", out / "decode_registry.cpp"])]
    prefill_args = [argument for d in contexts for argument in (
        "--schedule", d / "prefill-schedule.json", "--aot-manifest", d / "manifest.json")]
    reports.append(run_generator("generate-native-decode-registry.py", [
        "--phase", "prefill", *prefill_args, "--output-cpp", out / "prefill_registry.cpp"]))
    frozen = aot / "q1024-text-v151"
    reports.append(run_generator("generate-native-decode-registry.py", [
        "--phase", "prefill", "--prefill-registry", "frozen-text",
        "--schedule", frozen / "prefill-schedule.json", "--aot-manifest", frozen / "manifest.json",
        "--output-cpp", out / "frozen_text_registry.cpp"]))
    images, symbols = [], set()
    assembly = ['// Generated COFF read-only image data. No executable host code.', '.section .rdata,"dr"']
    for line in (out / "aot_objects.tsv").read_text().splitlines():
        image_path, object_name, filename = line.split("\t")
        path = Path(image_path)
        symbol = "_binary_" + re.sub(r"[^A-Za-z0-9_]", "_", filename) + "_start"
        if symbol in symbols:
            raise ValueError("Duplicate binary image symbol")
        symbols.add(symbol)
        # Forward slashes and quoted .incbin paths are accepted by LLVM's
        # Windows assembler. Refuse characters that could become directives.
        incbin = path.as_posix()
        if any(c in incbin for c in '\"\r\n\t'):
            raise ValueError("Unrepresentable assembler input path")
        assembly.extend([".p2align 8", f".globl {symbol}", f"{symbol}:", f'.incbin "{incbin}"'])
        images.append(dict(path=path.relative_to(UPSTREAM).as_posix(), symbol=symbol,
                           bytes=path.stat().st_size, sha256=digest(path.read_bytes())))
    (out / "aot_images.S").write_text("\n".join(assembly) + "\n", encoding="utf-8", newline="\n")
    sources = []
    for p in sorted((UPSTREAM / "native/src").glob("*.cpp")):
        if p.name in EXCLUDED:
            continue
        relative = p.relative_to(UPSTREAM).as_posix()
        sources.append(str(out / "overlay" / relative if relative in overlays else p))
    sources.extend(str(out / "overlay" / p) for p in (
        "benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp",
        "native/src/media_names.cpp"))
    sources.extend(str(out / p) for p in (
        "aot_registry.cpp", "decode_registry.cpp", "prefill_registry.cpp", "frozen_text_registry.cpp"))
    sources.append(str(ROOT / "native/linux_core_port/probe.cpp"))
    if args.gb10_gdn:
        sources.append(str(ROOT / "native/linux_core_port/gb10_gdn.hip.cpp"))
    if args.gb10_projections:
        sources.append(str(ROOT / "native/linux_core_port/gb10_projection.hip.cpp"))
    if args.gb10_prefill_projections:
        sources.append(str(ROOT / "native/linux_core_port/gb10_prefill_projection.hip.cpp"))
    if args.gb10_normalization:
        sources.append(str(ROOT / "native/linux_core_port/gb10_normalization.hip.cpp"))
        sources.append(str(ROOT / "native/linux_core_port/gb10_decode_attention.hip.cpp"))
    if args.gb10_moe:
        sources.append(str(ROOT / "native/linux_core_port/gb10_moe.hip.cpp"))
        sources.append(str(ROOT / "native/linux_core_port/gb10_decode_moe.hip.cpp"))
    generated = [dict(path=p.relative_to(out).as_posix(), bytes=p.stat().st_size,
                      sha256=digest(p.read_bytes())) for p in sorted(out.rglob("*")) if p.is_file()]
    report = dict(schema=1, upstream_revision=inventory["revision"],
        upstream_inventory_sha256=INVENTORY_SHA256, imported_files=len(inventory["files"]),
        imported_bytes=sum(p["bytes"] for p in inventory["files"]),
        overlays=adapted, sources=sources, images=images, generated=generated,
        aot_registry=aot_report, schedule_registries=reports,
        image_bytes=sum(x["bytes"] for x in images),
        vision_image=str(aot / "vision-attention-v0.3.0/kernels/d09fefdcb1ddb6cb-_fwd_kernel.hsaco"),
        windows_build_qualified=False, model_correctness_qualified=False, performance_qualified=False)
    report["optional_native_attention_prefill"] = dict(
        environment="AIMA_PORT_NATIVE_ATTENTION_PREFILL", enabled_value="1",
        scope="ordinary text; exact8192 active queries; cold cache position zero",
        backend="existing resident NativeVlUnifiedAttentionPlan; embedded kernel_unified_attention_2d",
        query_output_layout="BF16 [8192,16,256]", kv_layout="resident token-major BF16 [capacity,2,256]",
        positional_transform="unchanged GB10 head norm and ordinary RoPE when enabled",
        gate="existing BF16-input sigmoid gate", decode_changed=False,
        additional_device_bytes=0, additional_artifact_bytes=0, model_qualified=False)
    if args.windows_rectangular_ck:
        adapter = ROOT / "native/linux_core_port/ck_suffix_adapter.h"
        report["optional_adaptations"] = dict(windows_rectangular_ck=True,
            adapter_path=adapter.relative_to(ROOT).as_posix(),
            adapter_sha256=digest(adapter.read_bytes()),
            maximum_suffix_queries=8192, maximum_total_kv_tokens=262144)
    if args.current_text_decode:
        report.setdefault("optional_adaptations", {})["current_text_decode"] = dict(
            arithmetic="upstream current-vLLM decode",
            rotary_positions="ordinary text position, existing M-RoPE plan when supplied",
            rotary_cache="BF16 rounded", linear_state="in place; no historical ping-pong swaps",
            prefill_changed=False, model_qualified=False)
    if args.gb10_convolution:
        report["optional_adaptations"]["gb10_convolution"] = dict(
            product_rounding="BF16 round-to-nearest ties-to-even",
            accumulation="sequential FP32", prefill_tokens=8192, decode_tokens=1,
            prefill_changed=True,
            silu_table_sha256="673f8dd1280700578c1e8743afd2e3b4da134b1fbd463c890527e1c4d9f796b8",
            model_qualified=False)
    if args.gb10_gdn:
        report["optional_adaptations"]["gb10_gdn"] = dict(
            prefill="existing Windows FLA provider 1d11bf7 with exact q8192 arithmetic",
            decode="original GB10 Q2 recurrence applied to one accepted token",
            state_layout="value-head, value, key; FP32 and in place",
            decode_beta="FP32", prefill_beta="BF16",
            prefill_tokens=8192, decode_tokens=1, intermediate_aot_observations=False,
            provider_calls_per_linear_prefill=1, model_qualified=False)
    if args.gb10_projections:
        report["optional_adaptations"]["gb10_projections"] = dict(
            decode="existing Windows K16 width-26 SM121 projection arithmetic",
            windows_integer_lowering="DPP lane reductions and compact canonical normalization; same integer result",
            scope="singleton wvSplitK call sites, including grouped projections",
            embedding_norm="live token IDs select full-vocabulary model inverse-RMS scales",
            embedding_table_bytes=993280, embedding_device_bytes=1026048,
            embedding_table_sha256="f4e37f759c586bfc8fcc4d74cefdd89235f0f0c0c90cd286147e331e87509e67",
            prefill_dense_changed=False, model_qualified=False)
    if args.gb10_prefill_projections:
        report["optional_adaptations"]["gb10_prefill_projections"] = dict(
            tokens=8192, maximum_rows=12352, reductions=[512, 2048, 4096],
            producer="hipBLASLt BF16 inputs, FP32 destination; existing WMMA K16 geometry when no solution exists",
            replay="existing lossless scaled-half staged SM121 K16 arithmetic",
            selector="radius512 plus L2 upper bounds, 1000 ppb / K4096 10000 ppb",
            maximum_window_cells=1048576, candidate_counts="device-owned; no host count copy unless optional profiling is armed",
            shared_scratch_stream="default stream only; nondefault streams rejected",
            device_scratch_bytes=598360324,
            optional_wmma=dict(environment="AIMA_PORT_PREFILL_WMMA", enabled_value="1",
                producer="existing M64/N128 ascending-K16 BF16 WMMA; all eligible dense plans",
                output_only_environment="AIMA_PORT_PREFILL_WMMA_OUTPUT_ONLY",
                output_only_enabled_value="1", output_only_reduction=4096,
                additional_device_bytes=0),
            optional_linear_bound=dict(environment="AIMA_PORT_PREFILL_LINEAR_BOUND", enabled_value="1",
                scope="actual linear output projection call only; ordered synchronous host scope",
                linear_output_ppb=1000, full_attention_output_ppb=10000,
                radius=512, exact_replay="unchanged", additional_device_bytes=0),
            optional_full_coarse=dict(environment="AIMA_PORT_PREFILL_FULL_COARSE", enabled_value="1",
                scope="actual full-attention output call only; contiguous N2048/K4096",
                producer="existing C64 vector/domain SM121 interval matrix, one fragment, native coefficient 2^-19",
                selector="same-BF16 interval endpoints; ineligible rows and ambiguous cells use complete original replay",
                flags="reuse dead norm-bound buffers", additional_device_bytes=67108864,
                native_error_bound_universal_proof=False),
            optional_profile=dict(environment="AIMA_PORT_PREFILL_PROJECTION_PROFILE", enabled_value="1",
                arm="after load and READY; warmup excluded", completed_gpu_events=295,
                maximum_windows=97, additional_device_count_bytes=388,
                stages=["producer", "operands", "norm_bound", "selection", "replay"],
                candidate_reads="one completed bounded count array per projection; never controls arithmetic",
                invalid_elapsed_intervals="reported without clamping negative HIP intervals; timing_valid=false",
                timing="diagnostic; event, copy and synchronization overhead included in request"),
            model_qualified=False)
    if args.gb10_normalization:
        report["optional_adaptations"]["gb10_normalization"] = dict(
            gated="original q8192 sixteen-lane/eight-value prefill reduction; separate thirty-two-lane/four-value short decode; FP32 SiLU",
            residual="FP32 unrounded sum variance; BF16 residual numerator",
            gated_prefill_tokens=8192, gated_decode_tokens=1, residual_maximum_tokens=8192,
            decode_gated_call_site="current linear decode replaces the imported AOT gated-normalization call",
            dense_prefill="SM121 selected replay" if args.gb10_prefill_projections else "imported hipBLASLt BF16 producer",
            silu_table_bytes=262144, additional_device_bytes=266240,
            cross_layer_residual="one 4096-byte live row snapshot before the MoE tail; default stream only",
            prefill_moe_observations=12, decode_next_norm_observations=1,
            silu_table_sha256="f8b4983266a2d26f64a154c0c53c6acd6616e3298e7eb2e128c7431be586c97c",
            rsqrt_table="borrowed from the live GB10 GDN owner",
            optional_full_head_norm_rope=dict(
                environment="AIMA_PORT_FULL_ATTENTION_ROPE_TABLE", tokens=[1, 8192],
                head_norm="Q two stride64 warps; K four stride128 warps; SM121 reciprocal root",
                rope="rounded BF16 sine product and single-round BF16 FMA",
                layout="position_cos32_sin32", positions=262144,
                table_bytes=33554432, device_bytes=33554432,
                table_sha256="ba12ce218327d4cf23aac7dfacd8e9efbc99fd207611a8466227089838ef0e80",
                ordinary_text_only=True),
            optional_decode_attention=dict(
                environment="AIMA_PORT_DECODE_ATTENTION", enabled_value="1",
                cache="borrowed separate token-major K/V planes including the current row",
                arithmetic="SM121 K16 QK, original Q2 softmax/PV with denominator FMA",
                exp2="borrowed from live GB10 GDN owner", query_rows=1,
                reciprocal_environment="AIMA_PORT_ATTENTION_RCP_TABLE", reciprocal_bytes=8388640,
                reciprocal_sha256="d2e557543f6bc51f5141ba6414000cd8ed892e2e915eda19245c3cae22c16b39",
                scratch_bytes_at_probe_capacity=606208, default_stream_only=True),
            model_qualified=False)
    if args.gb10_moe:
        report["optional_adaptations"]["gb10_moe"] = dict(
            prefill="existing Windows MoE provider 9235750, live q8192 inputs and raw model weights",
            registered_weight_layers=40, preparation_before_ready=True,
            norm="unrounded FP32 MoE carrier variance, BF16 normalized numerator",
            carrier_bytes=201326592, provider_internal_dispatch="opaque; completed inside TTFT",
            scope="cold q8192; ordered layers and final row consumed exactly once",
            optional_decode=dict(environment="AIMA_PORT_DECODE_MOE", enabled_value="1",
                arithmetic="existing original SM121 complete MoE primitives, singleton live rows",
                resident_buffers=True, cache_copies=0, weight_copies=0,
                sigmoid="borrowed from GDN owner", additional_device_bytes=33693724,
                layers_per_token=40, flag_check="after layer39 before token publication",
                shared_weights="separate original gate/up planes", default_stream_only=True,
                terminal="8192-byte operand snapshot preserves unrounded final RMS variance"),
            model_qualified=False)
    (out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("upstream_revision", "imported_files", "imported_bytes", "image_bytes")}
                     | dict(images=len(images), compilation_units=len(sources), overlays=len(adapted))))


if __name__ == "__main__":
    main()
