// Included after run_qwen36_resident_batch_suffix. Cold chunks own the session
// mutex until all real inputs have advanced every layer and the final sample is
// ready. Intermediate samples never cross the streaming ABI.
#pragma once

hipError_t reserve_qwen36_prefill_chunk_attention(
    Qwen36ResidentSessionFullAttentionLayer &layer, size_t capacity
) {
    if (!layer.valid || layer.decode_tail_contiguous || layer.decode_tail_token_count ||
        layer.prefill_reserved_tokens || !layer.history_tokens ||
        capacity <= layer.history_tokens || capacity > qrt_sm121_attention_capacity::kTokens ||
        layer.element_kind != Qwen36ResidentSessionElementKind::kBf16 ||
        layer.k_bytes != layer.history_tokens * 1024u || layer.v_bytes != layer.k_bytes ||
        !layer.device_allocation || layer.device_k != layer.device_allocation ||
        layer.device_v != static_cast<unsigned char *>(layer.device_k) + layer.k_bytes)
        return hipErrorInvalidValue;
    const size_t bytes = capacity * 1024u;
    void *next = nullptr;
    hipError_t status = qrt_unpooled_device_malloc(&next, bytes * 2u);
    if (status != hipSuccess) return status;
    auto *next_v = static_cast<unsigned char *>(next) + bytes;
    status = hipMemcpy(next, layer.device_k, layer.k_bytes, hipMemcpyDeviceToDevice);
    if (status == hipSuccess)
        status = hipMemcpy(next_v, layer.device_v, layer.v_bytes, hipMemcpyDeviceToDevice);
    if (status == hipSuccess) status = hipFree(layer.device_allocation);
    if (status != hipSuccess) { (void)hipFree(next); return status; }
    layer.device_allocation = layer.device_k = next;
    layer.device_v = next_v;
    layer.prefill_reserved_tokens = capacity;
    return hipSuccess;
}

hipError_t resize_qwen36_prefill_chunk_tail(
    Qwen36ResidentSessionFullAttentionLayer &layer, size_t capacity
) {
    if (!layer.valid || layer.decode_tail_contiguous || layer.decode_tail_token_count ||
        !layer.device_decode_tail_allocation || capacity == 0u || capacity > 8192u ||
        layer.element_kind != Qwen36ResidentSessionElementKind::kBf16)
        return hipErrorInvalidValue;
    if (layer.decode_tail_capacity_tokens == capacity) return hipSuccess;
    const size_t bytes = capacity * 1024u;
    void *next = nullptr;
    hipError_t status = qrt_unpooled_device_malloc(&next, bytes * 2u);
    if (status != hipSuccess) return status;
    status = hipFree(layer.device_decode_tail_allocation);
    if (status != hipSuccess) { (void)hipFree(next); return status; }
    layer.device_decode_tail_allocation = layer.device_decode_tail_k = next;
    layer.device_decode_tail_v = static_cast<unsigned char *>(next) + bytes;
    layer.decode_tail_k_bytes = layer.decode_tail_v_bytes = bytes;
    layer.decode_tail_capacity_tokens = capacity;
    return hipSuccess;
}

hipError_t promote_qwen36_prefill_chunk_attention(
    Qwen36ResidentSessionFullAttentionLayer &layer, size_t prefix, size_t tokens
) {
    if (!layer.valid || layer.decode_tail_contiguous || !prefix || !tokens || tokens > 8192u ||
        prefix > qrt_sm121_attention_capacity::kTokens - tokens ||
        layer.element_kind != Qwen36ResidentSessionElementKind::kBf16 ||
        layer.history_tokens != prefix || layer.decode_tail_token_count != tokens ||
        layer.k_bytes != prefix * 1024u || layer.v_bytes != prefix * 1024u ||
        layer.decode_tail_k_bytes < tokens * 1024u || layer.decode_tail_v_bytes < tokens * 1024u ||
        !layer.device_allocation || !layer.device_k || !layer.device_v ||
        !layer.device_decode_tail_k || !layer.device_decode_tail_v)
        return hipErrorInvalidValue;
    const size_t bytes = (prefix + tokens) * 1024u;
    if (layer.prefill_reserved_tokens) {
        if (layer.prefill_reserved_tokens > qrt_sm121_attention_capacity::kTokens ||
            prefix + tokens > layer.prefill_reserved_tokens ||
            layer.device_k != layer.device_allocation ||
            layer.device_v != static_cast<unsigned char *>(layer.device_k) +
                layer.prefill_reserved_tokens * 1024u)
            return hipErrorInvalidValue;
        // The preceding chunk is fenced. Append only to uncommitted storage;
        // either copy may fail without altering the published history. A retry
        // overwrites both spans, and the coordinator discards a failed owner.
        auto status = hipMemcpy(static_cast<unsigned char *>(layer.device_k) + prefix * 1024u,
            layer.device_decode_tail_k, tokens * 1024u, hipMemcpyDeviceToDevice);
        if (status == hipSuccess)
            status = hipMemcpy(static_cast<unsigned char *>(layer.device_v) + prefix * 1024u,
                layer.device_decode_tail_v, tokens * 1024u, hipMemcpyDeviceToDevice);
        if (status != hipSuccess) return status;
        layer.k_bytes = layer.v_bytes = bytes;
        layer.history_tokens = prefix + tokens;
        layer.decode_tail_token_count = 0u;
        if (layer.history_tokens == layer.prefill_reserved_tokens)
            layer.prefill_reserved_tokens = 0u;
        return hipSuccess;
    }
    void *next = nullptr;
    hipError_t status = qrt_unpooled_device_malloc(&next, bytes * 2u);
    if (status != hipSuccess) return status;
    auto *next_k = static_cast<unsigned char *>(next);
    auto *next_v = next_k + bytes;
    // Synchronous copies make the old owner and all counters unchanged on a
    // failed copy. The caller has already fenced the completed chunk kernels.
    void *destinations[] = {next_k, next_v, next_k + prefix * 1024u, next_v + prefix * 1024u};
    const void *sources[] = {layer.device_k, layer.device_v, layer.device_decode_tail_k, layer.device_decode_tail_v};
    const size_t sizes[] = {prefix * 1024u, prefix * 1024u, tokens * 1024u, tokens * 1024u};
    for (unsigned i = 0u; i < 4u && status == hipSuccess; ++i)
        status = hipMemcpy(destinations[i], sources[i], sizes[i], hipMemcpyDeviceToDevice);
    if (status == hipSuccess) status = hipFree(layer.device_allocation);
    if (status != hipSuccess) { (void)hipFree(next); return status; }
    layer.device_allocation = layer.device_k = next;
    layer.device_v = next_v;
    layer.k_bytes = layer.v_bytes = bytes;
    layer.history_tokens = prefix + tokens;
    layer.decode_tail_token_count = 0u;
    return hipSuccess;
}

int run_qwen36_chunked_prefill(
    const qrt_qwen36_whole_provider_request_t &request,
    qrt_qwen36_whole_provider_result_t *result, uint64_t start_ns
) {
    std::lock_guard<std::recursive_mutex> lock(g_qwen36_resident_session_mutex);
    const uint64_t preload_ns = result->preload_wall_clock_ns;
    auto fail_request = [&](const std::string &stage, const std::string &message) {
        qrt_qwen36_whole_provider_set_failure(result, stage, message, start_ns);
        result->resident_session_valid = 0u;
        return 0;
    };
    const size_t total = request.input_token_count;
    const bool native_mtp = env_flag_enabled("QRT_QWEN36_MTP_NATIVE_DECODE");
    // The two supported transaction sizes have original-input seeded FLA
    // evidence. Other tail sizes continue to use the ordinary route when this
    // experimental mode is off; they are not padded or silently substituted.
    if (!request.resident_engine || total <= 8192u || total > qrt_sm121_attention_capacity::kTokens ||
        (total % 8192u != 0u && total % 8192u != 1024u) ||
        g_qwen36_chunked_prefill_total_tokens || ScopedQwen36PrefixBatchSuffix::active ||
        raw_env_flag_enabled("QRT_QWEN36_PREFIX_CHECKPOINTS") ||
        !env_flag_enabled("QRT_QWEN36_WHOLE_PROVIDER_RESIDENT_SESSION") ||
        !env_flag_enabled("QRT_QWEN36_WHOLE_PROVIDER_DIRECT_ORCHESTRATION") ||
        !env_flag_enabled("QRT_QWEN36_WHOLE_PROVIDER_FUSED_LAYER_STACK"))
        return fail_request("qwen36_chunked_prefill_request",
            "cold chunks require an 8192-aligned prompt, optionally 1024 final inputs, and a resident fused owner without checkpoint capture");
    if (native_mtp && total >= qrt_mtp_draft_schedule::reference_drafter_limit)
        return fail_request("mtp_chunked_prefill_retirement",
            "native MTP cold chunks currently require a prompt before drafter retirement");
    const uint64_t prompt_digest = qrt_fnv1a64_bytes(request.input_tokens, total * sizeof(uint32_t));
    if (request.expected_prompt_token_ids_fnv1a64 && request.expected_prompt_token_ids_fnv1a64 != prompt_digest)
        return fail_request("qwen36_chunked_prefill_prompt", "actual prompt tokens do not match the request digest");
    struct TotalScope {
        explicit TotalScope(size_t tokens) { g_qwen36_chunked_prefill_total_tokens = tokens; }
        ~TotalScope() { g_qwen36_chunked_prefill_total_tokens = 0u; }
    } total_scope(total);
    auto fail = [&](const std::string &stage, const std::string &message) {
        // A failed cold replacement never publishes a partially advanced owner.
        // Existing release logic fences or quarantines all resident allocations.
        g_qwen36_resident_session.valid = false;
        (void)release_qwen36_resident_session_locked();
        return fail_request(stage, message);
    };
    try {
        qrt_sm121_mtp_runtime::ChunkedPrefillSeed mtp_seed;
        std::vector<uint32_t> mtp_processed_inputs;
        if (native_mtp) mtp_processed_inputs.assign(request.input_tokens,request.input_tokens+total);
        const auto mtp_frontier = [&](size_t processed) {
            const auto& session=g_qwen36_resident_session;
            return qrt_sm121_mtp::TargetFrontier{session.owner_engine,session.generation,mtp_seed.epoch(),
                mtp_processed_inputs.data(),processed,session.current_token_id};
        };
        const auto mtp_failure = [&](const qrt_sm121_mtp::PromptStep& step) {
            if (step.completion_unknown) g_qwen36_resident_completion_unknown=true;
            return fail(step.stage,"native MTP could not complete the actual cold prefill chunks");
        };
        qrt_qwen36_whole_provider_request_t seed = request;
        seed.input_token_count = 8192u;
        seed.output_token_capacity = 1u;
        seed.flags = QRT_QWEN36_WHOLE_PROVIDER_FLAG_ARBITRARY_PREFILL |
            QRT_QWEN36_WHOLE_PROVIDER_FLAG_RESIDENT_DECODE_V1_RESULT |
            QRT_QWEN36_WHOLE_PROVIDER_FLAG_PREFIX_SEED_CAPTURE;
        seed.prefill_emit_callback = nullptr;
        seed.prefill_emit_user_data = nullptr;
        seed.expected_prompt_token_ids_fnv1a64 = qrt_fnv1a64_bytes(seed.input_tokens, 8192u * sizeof(uint32_t));
        seed.expected_output_token_id = UINT_MAX;
        auto chunk_result = std::make_unique<qrt_qwen36_whole_provider_result_t>();
        std::unique_ptr<qrt_mtp_target_rows::PrefillRows> first_mtp_rows;
        if(native_mtp)first_mtp_rows=std::make_unique<qrt_mtp_target_rows::PrefillRows>(request.input_tokens,total,0u,8192u);
        int seed_ok=0;
        {
            qrt_mtp_target_rows::Scope rows_scope(first_mtp_rows.get());
            seed_ok=qrt_qwen36_whole_provider_prefill_v1(&seed,chunk_result.get());
        }
        if (!seed_ok ||
            !chunk_result->completed || chunk_result->output_token_count != 1u ||
            !g_qwen36_resident_session.valid || g_qwen36_resident_session.prefix_tokens != 8192u ||
            g_qwen36_resident_session.committed_decode_token_count ||
            g_qwen36_resident_session.prompt_token_ids_fnv1a64 != seed.expected_prompt_token_ids_fnv1a64)
            return fail("qwen36_chunked_prefill_seed",
                chunk_result->failure[0] ? chunk_result->failure : "first real 8192 inputs did not create a complete resident seed");
        auto &owner = g_qwen36_resident_session;
        if(native_mtp){
            if(owner.owner_engine!=request.resident_engine || !owner.current_token_valid ||
                owner.current_token_id!=chunk_result->output_tokens[0])
                return fail("mtp_chunked_prefill_seed_frontier","first cold chunk did not preserve its actual target engine and sample");
            std::string stage,failure;
            const auto source=acquire_qwen36_mtp_model_weight_source(request.model_dir,&stage,&failure);
            if(!source)return fail(stage,failure);
            auto actual=mtp_frontier(8192u);actual.model_epoch=source->epoch();
            const unsigned capacity=static_cast<unsigned>((std::min)(size_t(262144u),total+request.output_token_capacity));
            const auto begun=mtp_seed.begin(*first_mtp_rows,source,actual,capacity);
            if(begun.status!=hipSuccess)return mtp_failure(begun);
            first_mtp_rows.reset();
        }
        // The complete prompt size is already known. Keep one KV allocation
        // per layer throughout the cold transaction instead of repeatedly
        // allocating and copying the growing history at every chunk boundary.
        for (unsigned i = 3u; i < QRT_QWEN36_LAYER_COUNT; i += 4u) {
            const auto status = reserve_qwen36_prefill_chunk_attention(
                owner.full_attention_layers[i], total);
            if (status != hipSuccess)
                return fail("qwen36_chunked_prefill_kv_reserve", hipGetErrorString(status));
        }
        std::cerr << "BATCH_MARK qwen36_chunked_prefill_kv_reservation layers="
                  << QRT_QWEN36_LAYER_COUNT / 4u << " capacity_tokens=" << total
                  << " allocation_bytes=" << total * 2048u * (QRT_QWEN36_LAYER_COUNT / 4u)
                  << " history_copy_on_append=0 completed=1" << std::endl;
        auto resize_tails = [&](size_t capacity) {
            for (unsigned i = 3u; i < QRT_QWEN36_LAYER_COUNT; i += 4u) {
                auto &layer = owner.full_attention_layers[i];
                const size_t before = layer.decode_tail_k_bytes + layer.decode_tail_v_bytes;
                const auto status = resize_qwen36_prefill_chunk_tail(layer, capacity);
                if (status != hipSuccess) return status;
                owner.full_attention_decode_tail_bytes -= before;
                owner.full_attention_decode_tail_bytes += layer.decode_tail_k_bytes + layer.decode_tail_v_bytes;
            }
            return hipSuccess;
        };
        hipError_t status = resize_tails(8192u);
        if (status != hipSuccess) return fail("qwen36_chunked_prefill_tail", hipGetErrorString(status));
        unsigned chunks = 1u;
        std::cerr << "BATCH_MARK qwen36_chunked_prefill_chunk index=0 first_position=0 input_tokens=8192"
                  << " history_tokens=8192 sampled_to_caller=0" << std::endl;
        {
            // The seed and persistent KV/tail reservations are complete. Reuse the
            // same disposable buffers across cold suffix chunks under the session
            // lock. Each suffix drains its kernels and returns every temporary
            // handoff before advancing the owner. Release the pool before final
            // decode-tail allocation, publication or any caller callback.
            ScopedDescriptorProductDeviceAllocationReuse chunk_scratch_scope(true);
            for (size_t prefix = 8192u; prefix < total;) {
                const size_t count = (std::min)(size_t(8192u), total - prefix);
                qrt_qwen36_whole_provider_prefix_request_v1_t suffix{};
                suffix.expected_prefix_token_count = static_cast<uint32_t>(prefix);
                suffix.suffix_token_count = static_cast<uint32_t>(count);
                suffix.suffix_tokens = request.input_tokens + prefix;
                std::string stage, failure;
                const uint64_t chunk_start = qrt_now_ns();
                std::unique_ptr<qrt_mtp_target_rows::PrefillRows> mtp_rows;
                if(native_mtp)mtp_rows=std::make_unique<qrt_mtp_target_rows::PrefillRows>(request.input_tokens,total,prefix,count);
                bool suffix_ok=false;
                {
                    qrt_mtp_target_rows::Scope rows_scope(mtp_rows.get());
                    suffix_ok=run_qwen36_resident_batch_suffix(suffix,nullptr,&stage,&failure,true,chunk_result.get());
                }
                if (!suffix_ok)
                    return fail(stage, failure);
                bool scratch_idle = true;
                {
                    std::lock_guard<std::mutex> scratch_lock(g_descriptor_device_allocation_pool_mutex);
                    for (const auto &block : g_descriptor_device_allocation_pool) {
                        if (block.in_use) scratch_idle = false;
                    }
                }
                if (!scratch_idle)
                    return fail("qwen36_chunked_prefill_scratch_handoff",
                        "completed cold chunk still owns a temporary allocation");
                // Validate all recurrent counters before promoting any layer. Their
                // original FP32 states and four-slot convolution rings stay in place.
                for (unsigned i = 0u; i < QRT_QWEN36_LAYER_COUNT; ++i) {
                    if (i % 4u == 3u) continue;
                    const auto &layer = owner.linear_layers[i];
                    if (!layer.valid || layer.prefix_tokens != prefix ||
                        layer.decode_qkv_token_count != count || layer.decode_recurrent_token_count != count)
                        return fail("qwen36_chunked_prefill_linear_commit", "a linear layer did not consume the complete original chunk");
                }
                for (unsigned i = 3u; i < QRT_QWEN36_LAYER_COUNT; i += 4u) {
                    status = promote_qwen36_prefill_chunk_attention(owner.full_attention_layers[i], prefix, count);
                    if (status != hipSuccess) return fail("qwen36_chunked_prefill_kv_commit", hipGetErrorString(status));
                    owner.full_attention_kv_bytes += count * 2048u;
                }
                for (unsigned i = 0u; i < QRT_QWEN36_LAYER_COUNT; ++i) {
                    if (i % 4u == 3u) continue;
                    auto &layer = owner.linear_layers[i];
                    layer.prefix_tokens = prefix + count;
                    layer.decode_qkv_token_count = layer.decode_recurrent_token_count = 0u;
                }
                owner.prefix_tokens = prefix + count;
                owner.committed_decode_token_count = 0u;
                owner.prompt_token_ids_fnv1a64 = qrt_fnv1a64_bytes(request.input_tokens, owner.prefix_tokens * sizeof(uint32_t));
                owner.mtp_target_hidden_valid = false;
                if(native_mtp){
                    const auto appended=mtp_seed.append(*mtp_rows,mtp_frontier(prefix+count));
                    if(appended.status!=hipSuccess)return mtp_failure(appended);
                }
                std::cerr << "BATCH_MARK qwen36_chunked_prefill_chunk index=" << chunks++
                          << " first_position=" << prefix << " input_tokens=" << count
                          << " history_tokens=" << owner.prefix_tokens << " sampled_to_caller=0 elapsed_ms="
                          << double(qrt_elapsed_ns(chunk_start, qrt_now_ns())) / 1000000.0 << std::endl;
                prefix += count;
            }
            std::cerr << "BATCH_MARK qwen36_chunked_prefill_scratch_reuse input_tokens=" << total
                      << " continued_chunks=" << chunks - 1u
                      << " in_use_blocks=0 persistent_kv_unpooled=1" << std::endl;
        }
        status = resize_tails(kQwen36ResidentDecodeTailCapacityTokens);
        if (status != hipSuccess) return fail("qwen36_chunked_prefill_decode_tail", hipGetErrorString(status));
        for (unsigned i = 3u; i < QRT_QWEN36_LAYER_COUNT; i += 4u) {
            const auto &layer = owner.full_attention_layers[i];
            if (layer.prefill_reserved_tokens || layer.history_tokens != total ||
                layer.k_bytes != total * 1024u || layer.v_bytes != layer.k_bytes)
                return fail("qwen36_chunked_prefill_kv_publication",
                    "the final KV append did not fill the reserved compact owner");
        }
        if (owner.prefix_tokens != total || owner.prompt_token_ids_fnv1a64 != prompt_digest ||
            !owner.current_token_valid || !owner.last_decode_top2_valid ||
            owner.last_decode_top2_position != total - 1u ||
            !qwen36_resident_decode_activation_workspace_layout_valid(owner.activation_workspace) ||
            owner.activation_workspace.full_attention_score_scratch_token_capacity < total + kQwen36ResidentDecodeTailCapacityTokens + 1u)
            return fail("qwen36_chunked_prefill_publication", "the completed chunk owner does not cover the requested prompt and decode workspace");
        if(native_mtp){
            qrt_sm121_mtp::RequestCheckpoint checkpoint;
            const auto saved=mtp_seed.save(&checkpoint,mtp_frontier(total));
            if(saved.status!=hipSuccess)return mtp_failure(saved);
            owner.native_mtp_checkpoint=std::move(checkpoint);
            owner.native_mtp_processed_inputs.swap(mtp_processed_inputs);
            std::cerr << "BATCH_MARK qwen36_mtp_native_request_seed generation=" << owner.generation
                      << " processed_tokens=" << owner.native_mtp_processed_inputs.size()
                      << " exact_target_frontier=1 immutable_checkpoint=1 chunked_prefill=1"
                      << " mtp_acceptance_enabled=0 numerical_correctness_claimed=0" << std::endl;
        }
        *result = *chunk_result;
        result->output_token_capacity = request.output_token_capacity;
        result->preload_wall_clock_ns = preload_ns;
        result->prompt_token_ids_fnv1a64 = prompt_digest;
        result->resident_session_valid = 1u;
        result->resident_session_prefix_token_count = static_cast<uint32_t>(total);
        result->resident_session_generation = owner.generation;
        result->resident_session_prompt_token_ids_fnv1a64 = prompt_digest;
        result->provider_digest_fnv1a64 = qrt_fnv1a64_update_bytes(prompt_digest,
            &result->output_tokens_fnv1a64, sizeof(result->output_tokens_fnv1a64));
        result->wall_clock_ns = qrt_elapsed_ns(start_ns, qrt_now_ns());
        // The descriptor diagnostics describe the final real chunk. The ABI
        // wall and callback clock cover every chunk, allocation and KV promotion.
        std::cerr << "BATCH_MARK qwen36_chunked_prefill_complete input_tokens=" << total
                  << " chunks=" << chunks << " maximum_chunk_tokens=8192 decode_committed_tokens=0"
                  << " descriptor_metrics_scope=last_chunk first_token=" << result->output_tokens[0]
                  << " raw_logit=" << result->continuation.output_logit
                  << " elapsed_ms=" << double(result->wall_clock_ns) / 1000000.0 << std::endl;
        if (request.prefill_emit_callback) {
            result->prefill_emit_attempted = 1u;
            result->prefill_emit_elapsed_ns = result->wall_clock_ns;
            const bool accepted = request.prefill_emit_callback(request.prefill_emit_user_data,
                result->output_tokens[0], result->wall_clock_ns) != 0;
            result->prefill_emit_completed = accepted ? 1u : 0u;
            result->prefill_emit_rejected = accepted ? 0u : 1u;
            if (!accepted) return fail("qwen36_chunked_prefill_emit", "stream callback rejected the completed cold prompt sample");
        }
        return 1;
    } catch (const std::exception &error) {
        return fail("qwen36_chunked_prefill_exception", error.what());
    } catch (...) {
        return fail("qwen36_chunked_prefill_exception", "unknown cold chunk failure");
    }
}
