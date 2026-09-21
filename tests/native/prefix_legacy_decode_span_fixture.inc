// Negative control copied verbatim from whole_provider.cpp at 6137e029dd9273f028c156cc23a707796894ed42.
// Decode-span SHA256: 2f35198cd4a1bda4e6c66466a3754dad5c35d2d176b6cb93d13a2505fe704a08
// Frozen fixture: source archives can run this regression without git history.
    const uint64_t decode_start_ns = qrt_now_ns();
    uint64_t provider_decode_end_ns = UINT64_C(0);
    while (completed_output_tokens < request->output_token_capacity) {
        const uint32_t remaining =
            request->output_token_capacity - completed_output_tokens;
        const uint32_t span_decode_tokens =
            gb10_continuation_teacher_forced
                ? 1u
                : (std::min)(
                      remaining,
                      QRT_QWEN36_WHOLE_PROVIDER_DECODE_MAX_OUTPUT_TOKENS - 1u
                  );
        decode_request =
            qrt_qwen36_whole_provider_decode_request_v1_t{};
        decode_result =
            qrt_qwen36_whole_provider_decode_result_v1_t{};
        decode_request.struct_size =
            static_cast<uint32_t>(sizeof(decode_request));
        decode_request.abi_version =
            QRT_QWEN36_WHOLE_PROVIDER_DECODE_ABI_VERSION;
        decode_request.flags =
            QRT_QWEN36_WHOLE_PROVIDER_DECODE_FLAG_NONE;
        decode_request.batch_size = 1u;
        decode_request.expected_prefix_token_count =
            request->expected_prefix_token_count;
        decode_request.output_token_capacity =
            span_decode_tokens + 1u;
        decode_request.initial_output_token_id =
            gb10_continuation_teacher_forced
                ? gb10_teacher_forced_continuation_tokens[
                      completed_output_tokens - 1u
                  ]
                : out_result->output_tokens[completed_output_tokens - 1u];
        decode_request.expected_session_generation =
            request->expected_session_generation;
        decode_request.expected_prompt_token_ids_fnv1a64 =
            request->expected_prompt_token_ids_fnv1a64;
        if (gb10_continuation_teacher_forced) {
            std::lock_guard<std::recursive_mutex> lock(
                g_qwen36_resident_session_mutex
            );
            if (!g_qwen36_resident_session.valid ||
                g_qwen36_resident_session.generation !=
                    request->expected_session_generation ||
                g_qwen36_resident_session.prompt_token_ids_fnv1a64 !=
                    request->expected_prompt_token_ids_fnv1a64) {
                return rollback_failure(
                    QRT_STATUS_UNSUPPORTED,
                    "qwen36_resident_prefix_gb10_teacher_forced_session",
                    "the gb10 teacher-forced diagnostic lost its shadow session"
                );
            }
            g_qwen36_resident_session.current_token_id =
                decode_request.initial_output_token_id;
            g_qwen36_resident_session.current_token_valid = true;
        }
        const int decode_ok = qrt_qwen36_whole_provider_decode_v1(
            &decode_request,
            &decode_result
        );
        if (!decode_ok || decode_result.completed == 0u ||
            decode_result.status !=
                static_cast<int32_t>(QRT_STATUS_OK) ||
            decode_result.output_token_count !=
                span_decode_tokens + 1u ||
            decode_result.timing_count != span_decode_tokens + 1u ||
            decode_result.output_tokens[0] !=
                decode_request.initial_output_token_id ||
            decode_result.token_step_elapsed_ns[0] != UINT64_C(0) ||
            decode_result.token_end_elapsed_ns[0] != UINT64_C(0) ||
            decode_result.tpot_elapsed_ns == UINT64_C(0)) {
            return rollback_failure(
                QRT_STATUS_UNSUPPORTED,
                decode_result.failure_stage[0] != '\0'
                    ? decode_result.failure_stage
                    : "qwen36_resident_prefix_decode",
                decode_result.failure[0] != '\0'
                    ? decode_result.failure
                    : "resident prefix request could not generate its requested continuation with exact per-token timing"
            );
        }
        const uint32_t output_base = completed_output_tokens;
        std::copy_n(
            decode_result.output_tokens + 1u,
            span_decode_tokens,
            out_result->output_tokens + output_base
        );
        for (uint32_t span_index = 1u;
             span_index <= span_decode_tokens;
             ++span_index) {
            const uint32_t output_index =
                output_base + span_index - 1u;
            const uint64_t step_ns =
                decode_result.token_step_elapsed_ns[span_index];
            const uint64_t local_end_ns =
                decode_result.token_end_elapsed_ns[span_index];
            if (step_ns == UINT64_C(0) ||
                local_end_ns < step_ns ||
                provider_decode_end_ns >
                    UINT64_MAX - local_end_ns) {
                return rollback_failure(
                    QRT_STATUS_UNSUPPORTED,
                    "qwen36_resident_prefix_decode_timing",
                    "resident prefix request received an invalid or overflowing provider token clock"
                );
            }
            out_result->output_token_step_elapsed_ns[output_index] =
                step_ns;
            out_result->output_token_end_elapsed_ns[output_index] =
                provider_decode_end_ns + local_end_ns;
        }
        if (provider_decode_end_ns >
            UINT64_MAX - decode_result.tpot_elapsed_ns) {
            return rollback_failure(
                QRT_STATUS_UNSUPPORTED,
                "qwen36_resident_prefix_decode_timing",
                "resident prefix request provider decode clock overflowed"
            );
        }
        provider_decode_end_ns += decode_result.tpot_elapsed_ns;
        if (request->emit_callback != nullptr) {
            for (uint32_t span_index = 0u;
                 span_index < span_decode_tokens;
                 ++span_index) {
                const uint32_t output_index =
                    output_base + span_index;
                if (!request->emit_callback(
                        request->emit_user_data,
                        request->expected_session_generation,
                        output_index,
                        out_result->output_tokens[output_index],
                        out_result
                            ->output_token_step_elapsed_ns[output_index],
                        out_result
                            ->output_token_end_elapsed_ns[output_index]
                    )) {
                    return rollback_failure(
                        QRT_STATUS_UNSUPPORTED,
                        "qwen36_resident_prefix_decode_callback_cancelled",
                        "resident prefix request callback cancelled a copy-on-write decode span"
                    );
                }
            }
        }
        completed_output_tokens += span_decode_tokens;
    }
    out_result->decode_elapsed_ns =
        qrt_elapsed_ns(decode_start_ns, qrt_now_ns());
    out_result->tpot_elapsed_ns = provider_decode_end_ns;
    out_result->tpot_sample_count =
        request->output_token_capacity - 1u;
    out_result->output_token_count = completed_output_tokens;
    out_result->output_token_ids_fnv1a64 = qrt_fnv1a64_bytes(
        out_result->output_tokens,
        static_cast<size_t>(out_result->output_token_count) *
            sizeof(out_result->output_tokens[0])
    );
