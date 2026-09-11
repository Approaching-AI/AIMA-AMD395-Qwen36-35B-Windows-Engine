"""Read-only target-model boundaries for the frozen cold token matrix.

The pinned runner supplies actual positions, input IDs and logits indices.
Speculative rows are retained with their transaction and input identity; a
matching generated history is required before using one as a reference.
"""
from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
import time

from capture_gb10_token_matrix import GPU_MODEL_RUNNER_SHA, TokenMatrixCapture
from capture_sm121_exp2_table import file_sha


def target_rows(positions, input_ids, logits_indices, selected_positions):
    if (not positions or len(positions) != len(input_ids) or
            any(type(x) is not int or x < 0 for x in positions + input_ids) or
            any(b != a + 1 for a, b in zip(positions, positions[1:])) or
            any(type(i) is not int or not 0 <= i < len(positions) for i in logits_indices) or
            len(set(logits_indices)) != len(logits_indices)):
        raise ValueError("invalid batch-one target row identity")
    return [dict(row=i, position=p, input_token_id=input_ids[i],
                 logit_row=logits_indices.index(i) if i in logits_indices else None)
            for i, p in enumerate(positions) if p in selected_positions]


def qualify_transaction(transaction, history):
    """All scheduled inputs through a row must match the generated history."""
    start = transaction["first_position"]
    ids = transaction["input_token_ids"]
    return [dict(row, matches_generated_history=(
        start + row["row"] < len(history) and
        ids[:row["row"] + 1] == history[start:start + row["row"] + 1]))
        for row in transaction["rows"]]


def prepared_token_ids(forward_ids, runner_ids, token_count, embedding_shape):
    # The pinned multimodal runner embeds even text-only requests before
    # _model_forward. Its input_ids argument is then None; the scheduled IDs
    # remain in the same runner.input_ids.gpu buffer used by embed_input_ids.
    if len(runner_ids) != token_count:
        raise ValueError("prepared token buffer length changed")
    if forward_ids is None:
        if embedding_shape != [token_count, 2048]:
            raise ValueError("embedded target input shape changed")
    elif forward_ids != runner_ids:
        raise ValueError("forward IDs differ from the prepared token buffer")
    return runner_ids


def recurrent_state_selection(indices, accepted, token_count, cache_slots):
    """Resolve the pinned fused kernel's batch-one input and output slots."""
    if not 1 <= token_count <= 2 or not indices:
        raise ValueError("unsupported recurrent observation batch")
    if isinstance(indices[0], list):
        if len(indices) != 1:
            raise ValueError("recurrent observation is batch one")
        slots = indices[0]
    else:
        if token_count != 1:
            raise ValueError("non-speculative recurrent observation is singleton")
        slots = indices
    previous = 0
    if accepted is not None:
        if len(accepted) != 1 or type(accepted[0]) is not int or accepted[0] < 1:
            raise ValueError("invalid accepted-token count")
        previous = accepted[0] - 1
    if (previous >= len(slots) or token_count > len(slots) or
            any(type(i) is not int or not 0 <= i < cache_slots for i in slots)):
        raise ValueError("invalid recurrent cache slot")
    return dict(initial_slot=slots[previous], final_slots=slots[:token_count],
                accepted_tokens=accepted, state_indices=indices)


def observation_layers(name, default, *, linear=False):
    value = os.environ.get(name)
    if value is None:
        return list(default)
    layers = [int(item) for item in value.split(',')]
    if (not 1 <= len(layers) <= 3 or len(set(layers)) != len(layers) or
            any(not 0 <= index < 40 or (linear and index % 4 == 3) for index in layers)):
        raise ValueError('invalid bounded observation layers: ' + name)
    return layers


def full_cache_observation_offset(case):
    controls = {
        'q8191-out32': ('QRT_GB10_Q8191_FULL_CACHE_OFFSET', 32),
        'q7169-out512': ('QRT_GB10_Q7169_FULL_CACHE_OFFSET', 512),
        'q8192-out512': ('QRT_GB10_Q8192_FULL_CACHE_OFFSET', 512),
    }
    if case not in controls:
        return 0
    name, continuation = controls[case]
    offset = int(os.environ.get(name, '0'))
    if not 0 <= offset < continuation:
        raise ValueError('full-attention observation offset exceeds the frozen continuation')
    return offset


def full_cache_observation_row(case):
    if case not in {'q8191-out32', 'q7169-out512', 'q8192-out512'}:
        return 0
    name = 'QRT_GB10_' + case.split('-')[0].upper() + '_FULL_CACHE_ROW'
    row = int(os.environ.get(name, '0'))
    if row not in (0, 1) or row > full_cache_observation_offset(case):
        raise ValueError('invalid original full-attention observation row')
    return row


def full_cache_row_is_qualified(cache, transactions):
    if cache is None:
        return True
    return any(
        transaction['ordinal'] == cache['transaction'] and
        any(row['row'] == cache['row'] and row['position'] + 1 == cache['tokens'] and
            row['input_token_id'] == cache['input_token_id'] and row['matches_generated_history']
            for row in transaction['qualified_rows'])
        for transaction in transactions)


def observation_positions(case, prompt_tokens):
    selected = {prompt_tokens - 1, prompt_tokens,
                prompt_tokens + full_cache_observation_offset(case)}
    defaults = {
        'q8191-out32': (32, (1, 5)),
        'q7169-out512': (512, (30, 31, 32, 118, 119, 120)),
        'q8192-out512': (512, (107, 108, 109)),
    }
    if case in defaults:
        continuation, offsets = defaults[case]
        name = 'QRT_GB10_' + case.split('-')[0].upper() + '_BOUNDARY_OFFSETS'
        value = os.environ.get(name)
        if value is not None:
            offsets = [int(item) for item in value.split(',')]
            if (not 1 <= len(offsets) <= 3 or len(set(offsets)) != len(offsets) or
                    any(not 0 <= offset < continuation for offset in offsets)):
                raise ValueError('invalid bounded continuation observation offsets')
        selected.update(prompt_tokens + offset for offset in offsets)
    return selected


class RuntimeBoundaryCapture(TokenMatrixCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import torch

        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        runner = self.model_runner
        if file_sha(Path(inspect.getsourcefile(type(runner)))) != GPU_MODEL_RUNNER_SHA:
            raise ValueError("target runner source changed")
        root = Path(directory)
        case = root.name
        full_cache_offset = full_cache_observation_offset(case)
        full_cache_row = full_cache_observation_row(case)
        # Position selection is separate from the cache's actual MTP row.
        # Capture both scheduled identities, then qualify against real tokens.
        selected = observation_positions(case, prompt_tokens)
        self._qrt_boundary_handles = []
        self._qrt_boundary_transactions = []
        self._qrt_boundary_files = {}
        self._qrt_boundary_norms = {}
        self._qrt_boundary_stages = {}
        self._qrt_boundary_decode_states = []
        self._qrt_boundary_bytes = 0
        self._qrt_boundary_started = time.monotonic()
        self._qrt_boundary_active = None
        self._qrt_boundary_selected = selected
        self._qrt_boundary_prompt_tokens = prompt_tokens
        self._qrt_boundary_linear_layers = observation_layers(
            'QRT_GB10_BOUNDARY_LINEAR_LAYERS',
            [0, 2, 4] if case == "q8191-out32" else [0, 2], linear=True)
        self._qrt_boundary_current = None
        self._qrt_boundary_indices = None
        model = runner.model
        containers = [(name, module) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.ModuleList) and len(module) == 40
                      and hasattr(module[3], "self_attn")]
        if len(containers) != 1:
            raise ValueError("ambiguous target layer container")
        layer_container, layers = containers[0]
        parent = model.get_submodule(layer_container.rsplit(".", 1)[0])
        original_prepare = runner._prepare_inputs
        original_forward = runner._model_forward
        original_logits = model.compute_logits
        self._qrt_boundary_restores = [(runner, "_prepare_inputs", original_prepare),
                                       (runner, "_model_forward", original_forward)]

        def prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            self._qrt_boundary_indices = result[0].detach().cpu().tolist()
            return result

        def forward(*args, **kwargs):
            if runner.input_batch.num_reqs != 1 or args:
                raise ValueError("unexpected target forward request layout")
            positions = kwargs["positions"].detach().cpu()
            # Text-only MRoPE has three identical position axes.
            if positions.ndim == 2:
                if positions.shape[0] != 3 or not torch.equal(positions, positions[:1].expand_as(positions)):
                    raise ValueError("non-text target positions")
                positions = positions[0]
            if positions.ndim != 1:
                raise ValueError("invalid target position dimensions")
            pos = positions.tolist()
            forward_ids = kwargs["input_ids"]
            embeddings = kwargs["inputs_embeds"]
            ids = prepared_token_ids(
                forward_ids.detach().cpu().tolist() if forward_ids is not None else None,
                runner.input_ids.gpu[:len(pos)].detach().cpu().tolist(), len(pos),
                list(embeddings.shape) if embeddings is not None else None)
            rows = target_rows(pos, ids, self._qrt_boundary_indices, selected)
            transaction = dict(ordinal=len(self._qrt_boundary_transactions), first_position=pos[0],
                               input_token_ids=ids, logits_indices=self._qrt_boundary_indices,
                               input_id_source="pinned_runner.input_ids.gpu",
                               forward_uses_embeddings=forward_ids is None,
                               rows=rows, token_count=len(pos), discarded=bool(runner.discard_request_mask.np[0]))
            self._qrt_boundary_transactions.append(transaction)
            self._qrt_boundary_active = transaction
            self._qrt_boundary_current = transaction
            try:
                return original_forward(*args, **kwargs)
            finally:
                self._qrt_boundary_active = None

        def save(label, value, transaction):
            if time.monotonic() - self._qrt_boundary_started > 180:
                raise ValueError("boundary observation deadline exceeded")
            if value.dtype not in (torch.bfloat16, torch.float32, torch.int32):
                raise ValueError("boundary dtype changed")
            value = value.detach().contiguous().cpu()
            payload = value.view(torch.uint8).numpy().tobytes()
            if self._qrt_boundary_bytes + len(payload) > 128 << 20:
                raise ValueError("boundary artifact ceiling exceeded")
            suffix = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int32: "i32"}[value.dtype]
            key = f'txn{transaction["ordinal"]:04d}-{label}-{suffix}'
            path = root / (key + ".bin")
            with path.open("xb") as stream:
                stream.write(payload)
            self._qrt_boundary_bytes += len(payload)
            self._qrt_boundary_files[key] = dict(file=path.name, dtype=suffix,
                shape=list(value.shape), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                transaction=transaction["ordinal"], label=label)

        def selected_tensor(value, transaction, width=2048):
            if (value.ndim != 2 or value.shape != (transaction["token_count"], width)):
                raise ValueError("target hidden boundary shape changed")
            indices = [row["row"] for row in transaction["rows"]]
            if indices == list(range(indices[0], indices[-1] + 1)):
                return value[indices[0]:indices[-1] + 1].detach().cpu()
            return torch.cat([value[index:index + 1].detach().cpu() for index in indices])

        def norm_hook(label):
            def observe(module, args, output):
                transaction = self._qrt_boundary_active
                if transaction is None or not transaction["rows"]:
                    return
                value = output[0] if isinstance(output, tuple) else output
                if (case == "q8191-out32" and transaction["first_position"] == 0 and
                        transaction["token_count"] == prompt_tokens):
                    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    self._qrt_boundary_norms[label] = dict(shape=list(value.shape), dtype=str(value.dtype),
                        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                save(label, selected_tensor(value, transaction), transaction)
            return observe

        def layer_hook(index):
            def observe(module, args, output):
                transaction = self._qrt_boundary_active
                if transaction is None or not transaction["rows"]:
                    return
                hidden, residual = (selected_tensor(value, transaction) for value in output)
                save(f"layer-{index:02d}-hidden", hidden, transaction)
                save(f"layer-{index:02d}-residual", residual, transaction)
                save(f"layer-{index:02d}-combined", hidden.float() + residual.float(), transaction)
            return observe

        for index, layer in enumerate(layers):
            for module, hook in ((layer, layer_hook(index)),
                                 (layer.input_layernorm, norm_hook(f"layer-{index:02d}-input-rmsnorm")),
                                 (layer.post_attention_layernorm, norm_hook(f"layer-{index:02d}-post-attention-rmsnorm"))):
                self._qrt_boundary_handles.append(module.register_forward_hook(hook))
        self._qrt_boundary_handles.append(parent.norm.register_forward_hook(norm_hook("final-norm")))

        # Preserve the original first-layer MoE call and selector. These
        # small decode-only endpoints distinguish projection, routing and
        # shared/routed rounding after the now-qualified GDN boundary.
        self._qrt_boundary_moe_layers = observation_layers(
            'QRT_GB10_BOUNDARY_MOE_LAYERS', [0, 2])
        if 39 in self._qrt_boundary_moe_layers:
            raise ValueError('MoE next-norm observation requires a following layer')
        def attach_moe(index):
            mlp = layers[index].mlp
            if mlp.tp_size != 1 or mlp.shared_expert is None:
                raise ValueError("decode MoE observation requires the original shared expert")
            self._qrt_boundary_moe_labels = {
                "input", "router", "output", "shared", "shared-gate-up",
                "shared-activated", "shared-down", "shared-gate", "topk-weights",
                "topk-ids", "expert-part-0", "expert-part-1", "next-hidden", "residual"}

            def moe_stage(label, value, width):
                transaction = self._qrt_boundary_active
                if (transaction is None or not transaction["rows"] or
                        transaction["first_position"] < prompt_tokens):
                    return
                if transaction["token_count"] > 2:
                    raise ValueError("decode MoE observation exceeds the original MTP batch")
                if value.dtype == torch.int64:
                    if torch.any((value < 0) | (value >= 256)):
                        raise ValueError("decode MoE expert index out of range")
                    value = value.to(torch.int32)
                save(f"moe-{index:02d}-" + label, selected_tensor(value, transaction, width), transaction)

            def moe_input(label, width):
                def observe(module, args):
                    moe_stage(label, args[0], width)
                return observe

            def moe_output(label, width):
                def observe(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    moe_stage(label, value, width)
                return observe

            for module, hook, pre in (
                    (mlp, moe_input("input", 2048), True),
                    (mlp.gate, moe_output("router", 256), False),
                    (mlp.shared_expert, moe_output("shared", 2048), False),
                    (mlp.shared_expert.gate_up_proj, moe_output("shared-gate-up", 1024), False),
                    (mlp.shared_expert.down_proj, moe_input("shared-activated", 512), True),
                    (mlp.shared_expert.down_proj, moe_output("shared-down", 2048), False),
                    (mlp.shared_expert_gate, moe_output("shared-gate", 1), False),
                    (mlp, moe_output("output", 2048), False)):
                register = module.register_forward_pre_hook if pre else module.register_forward_hook
                self._qrt_boundary_handles.append(register(hook))
            router = mlp.experts.router
            original_select = router.select_experts
            self._qrt_boundary_restores.append((router, "select_experts", original_select))

            def observe_selected(*args, **kwargs):
                result = original_select(*args, **kwargs)
                moe_stage("topk-weights", result[0], 8)
                moe_stage("topk-ids", result[1], 8)
                return result

            router.select_experts = observe_selected

            def expert_outputs(module, args, output):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError("original shared/routed expert tuple changed")
                for part, value in enumerate(output):
                    moe_stage(f"expert-part-{part}", value, 2048)

            def next_inputs(module, args):
                if len(args) != 2:
                    raise ValueError("original residual normalization arguments changed")
                moe_stage("next-hidden", args[0], 2048)
                moe_stage("residual", args[1], 2048)

            self._qrt_boundary_handles.append(mlp.experts.register_forward_hook(expert_outputs))
            self._qrt_boundary_handles.append(layers[index + 1].input_layernorm.register_forward_pre_hook(next_inputs))

        for index in self._qrt_boundary_moe_layers:
            attach_moe(index)

        # Observe the first full-attention layer after the linear/MoE boundary.
        # Hooks retain original qkv/norm/RoPE/attention/output results, including
        # the BF16 sigmoid-product endpoint at the output projection input.
        attention = layers[3].self_attn
        original_attention_forward = attention.forward
        self._qrt_boundary_restores.append((attention, "forward", original_attention_forward))
        self._qrt_boundary_full_active = False
        def observe_full_attention(*args, **kwargs):
            # get_rope() caches module instances across layers. Scope their
            # hooks to this owner's forward rather than overwriting another
            # layer's rotary observation under the same transaction key.
            self._qrt_boundary_full_active = True
            try:
                return original_attention_forward(*args, **kwargs)
            finally:
                self._qrt_boundary_full_active = False
        attention.forward = observe_full_attention
        self._qrt_boundary_full_cache = None
        capture_full_cache = case in {"q8191-out32", "q7169-out512", "q8192-out512"}
        self._qrt_boundary_full_cache_required = capture_full_cache
        def full_cache():
            transaction = self._qrt_boundary_active
            if (not capture_full_cache or self._qrt_boundary_full_cache is not None or
                    not self._qrt_boundary_full_active or transaction is None or
                    transaction["first_position"] + full_cache_row != prompt_tokens + full_cache_offset):
                return
            if transaction['token_count'] <= full_cache_row:
                raise ValueError('original full-attention observation row missing')
            from vllm.model_executor.layers.attention.attention import get_attention_context
            metadata, owner, cache, _ = get_attention_context(attention.attn.layer_name)
            if (owner is not attention.attn or cache.dtype != torch.bfloat16 or
                    cache.ndim != 5 or cache.shape[1] != 2 or tuple(cache.shape[3:]) != (2, 256)):
                raise ValueError("original full-attention cache layout changed")
            tokens = prompt_tokens + full_cache_offset + 1
            if not tokens <= int(metadata.seq_lens[0].item()) <= tokens + 1:
                raise ValueError("original full-attention cache length changed")
            block_size = cache.shape[2]
            blocks = metadata.block_table[0, :(tokens + block_size - 1) // block_size].long()
            if torch.any((blocks < 0) | (blocks >= cache.shape[0])):
                raise ValueError("original full-attention block index invalid")
            for label, tensor in zip(("cache-k", "cache-v"), cache.unbind(1)):
                logical = tensor.index_select(0, blocks).reshape(-1, 2, 256)[:tokens]
                save("full-03-" + label, logical, transaction)
            self._qrt_boundary_full_cache = dict(transaction=transaction["ordinal"],
                layer=3, tokens=tokens, decode_offset=full_cache_offset,
                row=full_cache_row, input_token_id=transaction['input_token_ids'][full_cache_row],
                block_size=block_size, block_indices=blocks.cpu().tolist(),
                cache_shape=list(cache.shape), max_query_len=metadata.max_query_len,
                seq_lens=metadata.seq_lens.cpu().tolist())
        self._qrt_boundary_full_labels = {
            "qkv", "q-norm", "k-norm", "q-rope", "k-rope", "context", "gated", "output"}
        def full_stage(label, value, width):
            transaction = self._qrt_boundary_active
            if (not self._qrt_boundary_full_active or
                    transaction is None or not transaction["rows"] or
                    transaction["first_position"] < prompt_tokens):
                return
            if not 1 <= transaction["token_count"] <= 2:
                raise ValueError("full-attention decode observation exceeds the original batch")
            value = value.reshape(transaction["token_count"], width)
            save("full-03-" + label, selected_tensor(value, transaction, width), transaction)
        def full_output(label, width):
            def observe(module, args, output):
                full_stage(label, output[0] if isinstance(output, tuple) else output, width)
                if label == "context":
                    full_cache()
            return observe
        def full_rope(module, args, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("original full-attention rotary result changed")
            full_stage("q-rope", output[0], 4096)
            full_stage("k-rope", output[1], 512)
        def full_gated(module, args):
            full_stage("gated", args[0], 4096)
        for module, hook in (
                (attention.qkv_proj, full_output("qkv", 9216)),
                (attention.q_norm, full_output("q-norm", 4096)),
                (attention.k_norm, full_output("k-norm", 512)),
                (attention.rotary_emb, full_rope),
                (attention.attn, full_output("context", 4096)),
                (attention.o_proj, full_output("output", 2048))):
            self._qrt_boundary_handles.append(module.register_forward_hook(hook))
        self._qrt_boundary_handles.append(attention.o_proj.register_forward_pre_hook(full_gated))

        def linear_stage(index, label, value, width):
            transaction = self._qrt_boundary_active
            if transaction is None or not transaction["rows"]:
                return
            if value.numel() != transaction["token_count"] * width:
                raise ValueError("linear boundary element count changed: " + label)
            value = value.reshape(transaction["token_count"], width)
            name = f"linear-{index:02d}-{label}"
            if case == "q8191-out32" and index == 4 and transaction["first_position"] == 0:
                payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                self._qrt_boundary_stages[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            save(name, selected_tensor(value, transaction, width), transaction)

        def attach_linear(index):
            linear = layers[index].linear_attn
            if linear.gqa_interleaved_layout or linear.tp_size != 1:
                raise ValueError("linear observation requires the original single-device layout")

            def projection(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                linear_stage(index, "qkv", value[:, :8192], 8192)
                linear_stage(index, "z-projection", value[:, 8192:], 4096)

            def ba(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                linear_stage(index, "b-projection", value[:, :32], 32)
                linear_stage(index, "a-projection", value[:, 32:], 32)

            def core_inputs(module, args, kwargs):
                if kwargs.get("q") is not None:
                    for label, width in (("q", 2048), ("k", 2048), ("v", 4096), ("g", 32), ("beta", 32)):
                        linear_stage(index, label + "-core-input", kwargs[label], width)

            def prefill_state(module, args, output):
                transaction = self._qrt_boundary_active
                if transaction is not None and transaction["rows"]:
                    state = output[1]
                    if state.shape != (1, 32, 128, 128):
                        raise ValueError("prefill final recurrent state shape changed")
                    save(f"linear-{index:02d}-prefill-state-after", state, transaction)

            def gated_inputs(module, args):
                linear_stage(index, "core", args[0], 4096)
                linear_stage(index, "z", args[1], 4096)

            def gated(module, args):
                linear_stage(index, "gated", args[0], 4096)

            def out(module, args, output):
                linear_stage(index, "output-projection",
                             output[0] if isinstance(output, tuple) else output, 2048)

            self._qrt_boundary_handles.extend((
                linear.in_proj_qkvz.register_forward_hook(projection),
                linear.in_proj_ba.register_forward_hook(ba),
                linear.chunk_gated_delta_rule.register_forward_pre_hook(core_inputs, with_kwargs=True),
                linear.chunk_gated_delta_rule.register_forward_hook(prefill_state),
                linear.norm.register_forward_pre_hook(gated_inputs),
                linear.out_proj.register_forward_pre_hook(gated),
                linear.out_proj.register_forward_hook(out)))

        for index in self._qrt_boundary_linear_layers:
            attach_linear(index)

        # Observe the actual cache slots selected by the original fused decode
        # call. MTP may accept either one or two tokens from its previous call;
        # slot zero is therefore not always the recurrent input state.
        active_linear = []
        core_module = inspect.getmodule(layers[0].linear_attn._forward_core)
        observed_sources = set()

        def selected_decode():
            transaction = self._qrt_boundary_active
            if (not active_linear or transaction is None or not transaction["rows"] or
                    transaction["first_position"] < prompt_tokens):
                return None
            if not 1 <= transaction["token_count"] <= 2:
                raise ValueError("decode observation token count changed")
            return active_linear[-1], transaction

        def wrap_core(index):
            linear = layers[index].linear_attn
            original = linear._forward_core
            if inspect.getmodule(original) is not core_module:
                raise ValueError("selected recurrent implementations differ")
            self._qrt_boundary_restores.append((linear, "_forward_core", original))

            def core(*args, **kwargs):
                active_linear.append(index)
                try:
                    return original(*args, **kwargs)
                finally:
                    active_linear.pop()
            linear._forward_core = core

        def wrap_operator(name):
            original = getattr(core_module, name)
            observed_sources.add(Path(inspect.getsourcefile(original)))
            signature = inspect.signature(original)
            self._qrt_boundary_restores.append((core_module, name, original))

            def operator(*args, **kwargs):
                selected = selected_decode()
                if selected is None:
                    return original(*args, **kwargs)
                index, transaction = selected
                values = signature.bind(*args, **kwargs)
                values.apply_defaults()
                values = values.arguments
                prefix = f"linear-{index:02d}"
                if name == "causal_conv1d_update":
                    state = values["conv_state"]
                    slots = values["conv_state_indices"].detach().cpu().tolist()
                    accepted = values["num_accepted_tokens"]
                    accepted = accepted.detach().cpu().tolist() if accepted is not None else None
                    if (len(slots) != 1 or type(slots[0]) is not int or
                            not 0 <= slots[0] < state.shape[0] or state.ndim != 3 or
                            state.shape[1] != 8192 or not 3 <= state.shape[2] <= 5):
                        raise ValueError("convolution cache observation shape changed")
                    slot = slots[0]
                    save(prefix + "-conv-history-before", state[slot:slot + 1], transaction)
                    result = original(*args, **kwargs)
                    linear_stage(index, "conv-decode-output", result, 8192)
                    save(prefix + "-conv-history-after", state[slot:slot + 1], transaction)
                    self._qrt_boundary_decode_states.append(dict(transaction=transaction["ordinal"],
                        layer=index, operator=name, cache_slot=slot, accepted_tokens=accepted,
                        history_token_offset=accepted[0] - 1 if accepted is not None else 0))
                    return result
                state = values["initial_state"]
                slots = values["ssm_state_indices"].detach().cpu().tolist()
                accepted = values.get("num_accepted_tokens")
                accepted = accepted.detach().cpu().tolist() if accepted is not None else None
                if state.ndim != 4 or tuple(state.shape[1:]) != (32, 128, 128):
                    raise ValueError("decode recurrent state shape changed")
                selection = recurrent_state_selection(slots, accepted,
                    transaction["token_count"], state.shape[0])
                slot = selection["initial_slot"]
                save(prefix + "-decode-state-before", state[slot:slot + 1], transaction)
                if name == "fused_sigmoid_gating_delta_rule_update":
                    for label, width in (("q", 2048), ("k", 2048), ("v", 4096)):
                        linear_stage(index, label + "-decode-input", values[label], width)
                result = original(*args, **kwargs)
                save(prefix + "-decode-state-after",
                     torch.cat([state[i:i + 1] for i in selection["final_slots"]]), transaction)
                self._qrt_boundary_decode_states.append(dict(transaction=transaction["ordinal"],
                    layer=index, operator=name, **selection))
                return result
            setattr(core_module, name, operator)

        for index in self._qrt_boundary_linear_layers:
            wrap_core(index)
        for name in ("causal_conv1d_update", "fused_sigmoid_gating_delta_rule_update",
                     "fused_recurrent_gated_delta_rule_packed_decode"):
            wrap_operator(name)
        observed_sources.add(Path(inspect.getsourcefile(core_module)))
        observed_sources.add(Path(inspect.getsourcefile(type(attention.attn))))
        observed_sources.add(Path(inspect.getsourcefile(type(attention.attn.impl))))

        def logits(*args, **kwargs):
            output = original_logits(*args, **kwargs)
            transaction = self._qrt_boundary_current
            if transaction is None or output.shape[0] != len(transaction["logits_indices"]):
                raise ValueError("target logits mapping changed")
            transaction["logit_rows"] = []
            for row in transaction["rows"]:
                index = row["logit_row"]
                if index is None:
                    continue
                value = output[index:index + 1].detach().cpu()
                save(f'position-{row["position"]}-logits', value, transaction)
                values = value[0].float()
                if torch.isnan(values).any() or torch.isposinf(values).any():
                    raise ValueError("invalid target logits")
                argmax = int(values.argmax().item())
                transaction["logit_rows"].append(dict(row, argmax_token=argmax,
                                                       raw_logit=float(values[argmax].item())))
            return output

        runner._prepare_inputs = prepare
        runner._model_forward = forward
        model.compute_logits = logits
        first_mlp = layers[self._qrt_boundary_moe_layers[0]].mlp
        record["runtime_boundaries"] = dict(layer_container=layer_container, selected_positions=sorted(selected),
            model_sources=[dict(file=str(path), sha256=file_sha(path)) for path in sorted({
                Path(inspect.getsourcefile(type(layers[0]))),
                Path(inspect.getsourcefile(layers[0].forward)),
                Path(inspect.getsourcefile(type(layers[0].input_layernorm)))})],
            maximum_saved_bytes=128 << 20, maximum_observation_seconds=180,
            decode_operator_sources=[dict(file=str(path), sha256=file_sha(path))
                                     for path in sorted(observed_sources)],
            linear_stage_layers=list(self._qrt_boundary_linear_layers),
            decode_moe_layers=list(self._qrt_boundary_moe_layers), decode_moe_source=dict(
                file=str(Path(inspect.getsourcefile(type(first_mlp)))),
                sha256=file_sha(Path(inspect.getsourcefile(type(first_mlp)))),
                internal_router=first_mlp.experts.is_internal_router,
                router_is_original_module=first_mlp.experts.gate is first_mlp.gate),
            decode_full_attention_layers=[3],
            decode_full_attention_cache=capture_full_cache,
            all_prefill_norm_hashes=case == "q8191-out32", original_methods_returned_unchanged=True)
        return record

    def qrt_finish_token_matrix(self):
        for owner, name, original in self._qrt_boundary_restores:
            setattr(owner, name, original)
        for handle in self._qrt_boundary_handles:
            handle.remove()
        record = super().qrt_finish_token_matrix()
        if self._qrt_boundary_full_cache_required and self._qrt_boundary_full_cache is None:
            raise ValueError("missing original first-decode full-attention cache")
        required = {f"layer-{i:02d}-{surface}" for i in range(40)
                    for surface in ("hidden", "residual", "combined", "input-rmsnorm", "post-attention-rmsnorm")}
        required.add("final-norm")
        required.update(f"linear-{layer:02d}-{stage}" for layer in self._qrt_boundary_linear_layers
                        for stage in ("qkv", "z-projection", "b-projection", "a-projection",
                                      "core", "z", "gated", "output-projection"))
        for transaction in self._qrt_boundary_transactions:
            if transaction["rows"]:
                observed = {value["label"] for value in self._qrt_boundary_files.values()
                            if value["transaction"] == transaction["ordinal"]}
                if not required <= observed:
                    raise ValueError("incomplete selected target layer observations")
                if transaction["first_position"] >= self._qrt_boundary_prompt_tokens:
                    decode_required = {f"linear-{layer:02d}-{stage}"
                        for layer in self._qrt_boundary_linear_layers
                        for stage in ("conv-history-before", "conv-history-after", "conv-decode-output",
                                      "decode-state-before", "decode-state-after")}
                    if not decode_required <= observed:
                        raise ValueError("incomplete original decode state observations")
                    if not {f"moe-{layer:02d}-" + label for layer in self._qrt_boundary_moe_layers
                            for label in self._qrt_boundary_moe_labels} <= observed:
                        raise ValueError("incomplete original decode MoE observations")
                    if not {"full-03-" + label for label in self._qrt_boundary_full_labels} <= observed:
                        raise ValueError("incomplete original decode full-attention observations")
        boundaries = dict(files=self._qrt_boundary_files, bytes=self._qrt_boundary_bytes,
            transactions=self._qrt_boundary_transactions, full_prefill_norms=self._qrt_boundary_norms,
            full_prefill_linear_stages=self._qrt_boundary_stages,
            decode_state_selections=self._qrt_boundary_decode_states,
            full_attention_cache=self._qrt_boundary_full_cache,
            selected_positions=sorted(self._qrt_boundary_selected),
            original_methods_returned_unchanged=True, diagnostic_only=True)
        record["runtime_boundaries"] = boundaries
        return record
