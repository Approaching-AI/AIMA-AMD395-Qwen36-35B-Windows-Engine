"""Bind original MTP norm calls to their selected, unchanged Triton launchers."""
from __future__ import annotations

import inspect
from pathlib import Path
import re

from capture_gb10_mtp_boundaries import MtpBoundaryCapture
from capture_sm121_exp2_table import file_sha


LAUNCHER_SHA = "a8a32f9ae3f24f4f29d998e80c0393fa10b7b9b2ef2179841b9f5b2938b3f916"


def norm_call_identity(transaction, label, tensor_layouts, launcher_keys):
    if (type(transaction) is not int or transaction < 0 or not label or
            not tensor_layouts or not launcher_keys or len(launcher_keys) > 4):
        raise ValueError("incomplete original MTP norm launch identity")
    for layout in tensor_layouts:
        if (layout['dtype'] not in ('torch.bfloat16', 'torch.float32') or
                not layout['shape'] or len(layout['shape']) != len(layout['stride']) or
                any(type(n) is not int or n < 1 for n in layout['shape']) or
                any(type(n) is not int or n < 0 for n in layout['stride']) or
                type(layout['storage_offset']) is not int or layout['storage_offset'] < 0):
            raise ValueError("invalid original MTP norm tensor layout")
    return dict(transaction=transaction, label=label, inputs=tensor_layouts,
                kernels=list(launcher_keys), original_call_returned_unchanged=True)


class MtpKernelBoundaryCapture(MtpBoundaryCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import torch
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        source = Path(inspect.getsourcefile(CachingAutotuner))
        if file_sha(source) != LAUNCHER_SHA:
            raise ValueError("original Inductor launcher source changed")
        self._qrt_mtp_norm_current = None
        self._qrt_mtp_norm_calls = []
        self._qrt_mtp_norm_kernels = {}
        self._qrt_mtp_norm_handles = []
        self._qrt_mtp_launcher_owner = CachingAutotuner
        self._qrt_mtp_launcher_original = CachingAutotuner.run
        self._qrt_mtp_launcher_source = dict(file=str(source), sha256=LAUNCHER_SHA)
        root = Path(directory)

        def layouts(values):
            return [dict(dtype=str(value.dtype), shape=list(value.shape),
                stride=list(value.stride()), storage_offset=value.storage_offset())
                for value in values if isinstance(value, torch.Tensor)]

        def enter(label):
            def observe(module, args):
                transaction = self._qrt_mtp_active
                if transaction is None:
                    return
                if self._qrt_mtp_norm_current is not None:
                    raise ValueError("nested original MTP norm observation")
                self._qrt_mtp_norm_current = dict(transaction=transaction['ordinal'],
                    label=label, inputs=layouts(args), kernels=[])
            return observe

        def leave(module, args, output):
            if self._qrt_mtp_active is None:
                return
            current = self._qrt_mtp_norm_current
            if current is None:
                raise ValueError("original MTP norm observation lacks entry")
            row = norm_call_identity(current['transaction'], current['label'],
                                     current['inputs'], current['kernels'])
            row['outputs'] = layouts(output if isinstance(output, tuple) else (output,))
            self._qrt_mtp_norm_calls.append(row)
            self._qrt_mtp_norm_current = None

        original_run = CachingAutotuner.run

        def run(autotuner, *args, **kwargs):
            result = original_run(autotuner, *args, **kwargs)
            current = self._qrt_mtp_norm_current
            if current is None:
                return result
            if len(autotuner.launchers) != 1:
                raise ValueError("original MTP norm launcher selection is incomplete")
            launcher = autotuner.launchers[0]
            cache_hash = launcher.cache_hash
            kernel_name = autotuner.inductor_meta['kernel_name']
            if (not re.fullmatch(r'[A-Z2-7]{52}', cache_hash) or
                    not re.fullmatch(r'triton_red_[A-Za-z0-9_]+', kernel_name)):
                raise ValueError("unexpected original MTP norm launcher identity")
            key = cache_hash + '/' + kernel_name
            if key not in self._qrt_mtp_norm_kernels:
                if len(self._qrt_mtp_norm_kernels) >= 32:
                    raise ValueError("MTP kernel observation count ceiling exceeded")
                artifacts = []
                for suffix in ('.ptx', '.ttgir', '.json'):
                    path = Path('/tmp/torchinductor_root/triton/0') / (key + suffix)
                    size = path.stat().st_size
                    if size > 256 << 10:
                        raise ValueError("original MTP kernel observation size ceiling exceeded")
                    destination = root / ('norm-kernel-' + cache_hash + suffix)
                    with destination.open('xb') as stream:
                        stream.write(path.read_bytes())
                    artifacts.append(dict(original_path=str(path), file=destination.name,
                        bytes=size, sha256=file_sha(destination)))
                self._qrt_mtp_norm_kernels[key] = dict(cache_hash=cache_hash, name=kernel_name,
                    config=dict(launcher.config.kwargs), num_warps=launcher.config.num_warps,
                    num_stages=launcher.config.num_stages, artifacts=artifacts)
            current['kernels'].append(key)
            return result

        parent = self.model_runner.drafter.model.model
        layer = parent.layers[0]
        modules = ((parent.pre_fc_norm_embedding, 'embedding-norm'),
            (parent.pre_fc_norm_hidden, 'hidden-norm'), (layer.input_layernorm, 'input-norm'),
            (layer.self_attn.q_norm, 'q-norm'), (layer.self_attn.k_norm, 'k-norm'),
            (layer.post_attention_layernorm, 'post-attention-norm'), (parent.norm, 'final-norm'))
        self._qrt_mtp_norm_labels = {label for _, label in modules}
        for module, label in modules:
            self._qrt_mtp_norm_handles.append(module.register_forward_pre_hook(enter(label)))
            self._qrt_mtp_norm_handles.append(module.register_forward_hook(leave))
        CachingAutotuner.run = run
        record['mtp_norm_launchers'] = dict(source=self._qrt_mtp_launcher_source,
            original_launcher_returned_unchanged=True, maximum_kernels=32)
        return record

    def qrt_finish_token_matrix(self):
        self._qrt_mtp_launcher_owner.run = self._qrt_mtp_launcher_original
        for handle in self._qrt_mtp_norm_handles:
            handle.remove()
        record = super().qrt_finish_token_matrix()
        if self._qrt_mtp_norm_current is not None:
            raise ValueError("unfinished original MTP norm call")
        for transaction in record['mtp_boundaries']['transactions']:
            calls = [row for row in self._qrt_mtp_norm_calls if row['transaction'] == transaction['ordinal']]
            if len(calls) != 7 or {row['label'] for row in calls} != self._qrt_mtp_norm_labels:
                raise ValueError("original MTP norm transaction lacks a selected launcher")
        record['mtp_boundaries']['norm_launchers'] = dict(calls=self._qrt_mtp_norm_calls,
            kernels=self._qrt_mtp_norm_kernels, source=self._qrt_mtp_launcher_source,
            original_launcher_returned_unchanged=True, diagnostic_only=True)
        return record
