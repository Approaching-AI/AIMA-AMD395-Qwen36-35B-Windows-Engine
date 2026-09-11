"""Read-only target-model boundaries for the frozen cold token matrix.

The pinned runner supplies actual positions, input IDs and logits indices.
Speculative rows are retained with their transaction and input identity; a
matching generated history is required before using one as a reference.
"""
from __future__ import annotations

import hashlib
import inspect
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


class RuntimeBoundaryCapture(TokenMatrixCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import torch

        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        runner = self.model_runner
        if file_sha(Path(inspect.getsourcefile(type(runner)))) != GPU_MODEL_RUNNER_SHA:
            raise ValueError("target runner source changed")
        root = Path(directory)
        case = root.name
        selected = {prompt_tokens - 1, prompt_tokens}
        if case == "q8191-out32":
            selected.update(range(prompt_tokens - 64, prompt_tokens))
            selected.add(8196)
        elif case == "q7169-out512":
            selected.update((7287, 7288, 7289))
        elif case == "q8192-out512":
            selected.update((8299, 8300, 8301))
        self._qrt_boundary_handles = []
        self._qrt_boundary_transactions = []
        self._qrt_boundary_files = {}
        self._qrt_boundary_norms = {}
        self._qrt_boundary_bytes = 0
        self._qrt_boundary_started = time.monotonic()
        self._qrt_boundary_active = None
        self._qrt_boundary_selected = selected
        self._qrt_boundary_current = None
        self._qrt_boundary_indices = None
        model = runner.model
        containers = [(name, module) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.ModuleList) and len(module) == 40
                      and hasattr(module[3], "self_attn")]
        if len(containers) != 1:
            raise ValueError("ambiguous target layer container")
        name, layers = containers[0]
        parent = model.get_submodule(name.rsplit(".", 1)[0])
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
            ids = kwargs["input_ids"].detach().cpu().tolist()
            rows = target_rows(pos, ids, self._qrt_boundary_indices, selected)
            transaction = dict(ordinal=len(self._qrt_boundary_transactions), first_position=pos[0],
                               input_token_ids=ids, logits_indices=self._qrt_boundary_indices,
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
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("boundary dtype changed")
            value = value.detach().contiguous().cpu()
            payload = value.view(torch.uint8).numpy().tobytes()
            if self._qrt_boundary_bytes + len(payload) > 128 << 20:
                raise ValueError("boundary artifact ceiling exceeded")
            suffix = "bf16" if value.dtype == torch.bfloat16 else "f32"
            key = f'txn{transaction["ordinal"]:04d}-{label}-{suffix}'
            path = root / (key + ".bin")
            with path.open("xb") as stream:
                stream.write(payload)
            self._qrt_boundary_bytes += len(payload)
            self._qrt_boundary_files[key] = dict(file=path.name, dtype=suffix,
                shape=list(value.shape), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                transaction=transaction["ordinal"], label=label)

        def selected_tensor(value, transaction):
            if (value.ndim != 2 or value.shape != (transaction["token_count"], 2048)):
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
        record["runtime_boundaries"] = dict(layer_container=name, selected_positions=sorted(selected),
            model_sources=[dict(file=str(path), sha256=file_sha(path)) for path in sorted({
                Path(inspect.getsourcefile(type(layers[0]))),
                Path(inspect.getsourcefile(layers[0].forward)),
                Path(inspect.getsourcefile(type(layers[0].input_layernorm)))})],
            maximum_saved_bytes=128 << 20, maximum_observation_seconds=180,
            all_prefill_norm_hashes=case == "q8191-out32", original_methods_returned_unchanged=True)
        return record

    def qrt_finish_token_matrix(self):
        for owner, name, original in self._qrt_boundary_restores:
            setattr(owner, name, original)
        for handle in self._qrt_boundary_handles:
            handle.remove()
        record = super().qrt_finish_token_matrix()
        required = {f"layer-{i:02d}-{surface}" for i in range(40)
                    for surface in ("hidden", "residual", "combined", "input-rmsnorm", "post-attention-rmsnorm")}
        required.add("final-norm")
        for transaction in self._qrt_boundary_transactions:
            if transaction["rows"]:
                observed = {value["label"] for value in self._qrt_boundary_files.values()
                            if value["transaction"] == transaction["ordinal"]}
                if not required <= observed:
                    raise ValueError("incomplete selected target layer observations")
        boundaries = dict(files=self._qrt_boundary_files, bytes=self._qrt_boundary_bytes,
            transactions=self._qrt_boundary_transactions, full_prefill_norms=self._qrt_boundary_norms,
            selected_positions=sorted(self._qrt_boundary_selected),
            original_methods_returned_unchanged=True, diagnostic_only=True)
        record["runtime_boundaries"] = boundaries
        return record
