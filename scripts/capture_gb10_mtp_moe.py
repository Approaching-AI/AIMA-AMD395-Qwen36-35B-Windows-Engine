"""Observe original MTP MoE internals and the actual sampled draft logits."""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
import struct
import time

from capture_gb10_mtp_launchers import MtpKernelBoundaryCapture
from capture_sm121_exp2_table import file_sha


MAXIMUM_BYTES = 64 << 20
MAXIMUM_WEIGHT_BYTES = 16 << 20
FRONTIERS = {
    'router': (256, 'torch.bfloat16'), 'shared-gate': (1, 'torch.bfloat16'),
    'shared-gate-up': (1024, 'torch.bfloat16'), 'shared-activated': (512, 'torch.bfloat16'),
    'shared-down': (2048, 'torch.bfloat16'), 'shared': (2048, 'torch.bfloat16'),
    'expert-part-0': (2048, 'torch.bfloat16'), 'expert-part-1': (2048, 'torch.bfloat16'),
    'topk-weights': (8, 'torch.float32'), 'topk-ids': (8, 'torch.int32'),
    'routed-gate-up': (8192, 'torch.bfloat16'), 'routed-activated': (4096, 'torch.bfloat16'),
    'routed-weighted': (16384, 'torch.bfloat16'),
    'final-hidden': (2048, 'torch.bfloat16'), 'final-residual': (2048, 'torch.bfloat16'),
}
WEIGHTS = {'router': (256, 2048), 'shared-gate': (1, 2048),
           'shared-gate-up': (1024, 2048), 'shared-down': (2048, 512), 'final-norm': (2048,)}


def observe_original_moe_routed(forward, module, observe, tokens, *args, **kwargs):
    """Keep the separately pinned target observer unchanged during MTP capture.

    This uses the same two-call observation as the qualified target-MoE
    observer, including restoration when an original call or copy fails.
    """
    if type(tokens) is not int or not 1 <= tokens <= 8192:
        raise ValueError('routed MoE observation chunk extent changed')
    original_dispatch = module.invoke_fused_moe_triton_kernel
    dispatch_count = 0

    def dispatch(*pos, **kw):
        nonlocal dispatch_count
        if dispatch_count >= 2 or len(pos) < 3:
            raise ValueError('original routed MoE projection call changed')
        width = 1024 if dispatch_count == 0 else 2048
        if tuple(pos[2].shape) != (tokens, 8, width):
            raise ValueError('original routed MoE projection shape changed')
        if dispatch_count == 1:
            if tuple(pos[0].shape) != (tokens * 8, 512):
                raise ValueError('original routed MoE activation shape changed')
            observe('routed-activated', pos[0], 8 * 512)
        result = original_dispatch(*pos, **kw)
        observe('routed-gate-up' if dispatch_count == 0 else 'routed-weighted', pos[2], 8 * width)
        dispatch_count += 1
        return result

    module.invoke_fused_moe_triton_kernel = dispatch
    try:
        result = forward(*args, **kwargs)
        if dispatch_count != 2:
            raise ValueError('incomplete original routed MoE observations')
        return result
    finally:
        module.invoke_fused_moe_triton_kernel = original_dispatch


def qualify_moe_capture(worker, directory):
    """Check file identities and bind sampled logits to unchanged draft results."""
    root = Path(directory)
    mtp, moe = worker['mtp_boundaries'], worker['mtp_moe']
    transactions = mtp['transactions']
    if (not mtp['original_history_qualified'] or not transactions or
            not moe['original_results_returned_unchanged'] or
            not 0 < moe['bytes'] <= MAXIMUM_BYTES or
            len(moe['draft_logits']) != len(transactions)):
        raise ValueError('incomplete original MTP MoE observations')
    total = 0

    def checked(meta, shape, dtype, directory=root, count=True):
        nonlocal total
        width = 2 if dtype == 'torch.bfloat16' else 4
        name = meta['file']
        path = directory / name
        size = math.prod(shape) * width
        if (not isinstance(name, str) or Path(name).name != name or path.is_symlink() or
                meta['shape'] != shape or meta['dtype'] != dtype or meta['bytes'] != size or
                path.stat().st_size != size or file_sha(path) != meta['sha256']):
            raise ValueError('original MTP MoE file identity or layout changed')
        if count:
            total += size
        return path.read_bytes()

    if set(moe['weights']) != set(WEIGHTS) or moe['weights_directory'] != '../mtp-moe-original-weights':
        raise ValueError('original MTP MoE weight layout changed')
    weight_bytes = 0
    for label, shape in WEIGHTS.items():
        weight_bytes += len(checked(moe['weights'][label], list(shape), 'torch.bfloat16',
            root.parent / 'mtp-moe-original-weights', count=False))
    if weight_bytes > MAXIMUM_WEIGHT_BYTES:
        raise ValueError('original MTP MoE weight byte bound')
    required_keys = set()
    draft_checks = []
    for transaction in transactions:
        ordinal = transaction['ordinal']
        count = len(transaction['rows'])
        data = {}
        for label, (width, dtype) in FRONTIERS.items():
            key = f'mtp-moe{ordinal:04d}-{label}'
            required_keys.add(key)
            meta = moe['files'][key]
            if meta['transaction'] != ordinal or meta['label'] != label:
                raise ValueError('original MTP MoE transaction changed')
            data[label] = checked(meta, [count, width], dtype)
        if data['shared'] != data['expert-part-0']:
            raise ValueError('original shared expert tuple ordering changed')
        original = mtp['files'][f'mtp{ordinal:04d}-moe-output']
        if data['final-hidden'] != checked(original, [count, 2048], 'torch.bfloat16', count=False):
            raise ValueError('original final normalization input differs from MoE output')
        ids = struct.unpack('<' + str(count * 8) + 'i', data['topk-ids'])
        weights = struct.unpack('<' + str(count * 8) + 'f', data['topk-weights'])
        if (any(not 0 <= value < 256 for value in ids) or
                any(not math.isfinite(value) or not 0 <= value <= 1 for value in weights) or
                any(len(set(ids[row * 8:row * 8 + 8])) != 8 or
                    abs(sum(weights[row * 8:row * 8 + 8]) - 1) > 2e-6 for row in range(count))):
            raise ValueError('original MTP routing identities or probabilities changed')
        draft = moe['draft_logits'][str(ordinal)]
        sampled = [i for i, row in enumerate(transaction['rows']) if row['selected_for_sampling']]
        if (len(sampled) != 1 or draft['transaction'] != ordinal or
                draft['sampled_row'] != transaction['sampled_rows'][0] or
                not draft['original_result_returned_unchanged']):
            raise ValueError('original MTP draft sampling row changed')
        hidden = checked(draft['hidden'], [1, 2048], 'torch.bfloat16')
        normalized = checked(mtp['files'][f'mtp{ordinal:04d}-final-norm'],
            [count, 2048], 'torch.bfloat16', count=False)
        offset = sampled[0] * 4096
        if hidden != normalized[offset:offset + 4096]:
            raise ValueError('draft LM-head input differs from original sampled final norm')
        logits = struct.unpack('<248320f', checked(draft['logits'], [1, 248320], 'torch.float32'))
        if any(not math.isfinite(value) for value in logits):
            raise ValueError('non-finite original MTP draft logits')
        token = max(range(len(logits)), key=logits.__getitem__)
        if transaction['draft_token_ids'] != [[token]]:
            raise ValueError('original draft token differs from copied original logits')
        draft_checks.append(dict(transaction=ordinal, sampled_row=draft['sampled_row'],
            draft_token_id=token, raw_logit=logits[token], raw_logits_sha256=draft['logits']['sha256'],
            sampled_hidden_matches=True, argmax_matches_original_draft=True))
    if set(moe['files']) != required_keys or set(moe['draft_logits']) != {str(t['ordinal']) for t in transactions} or total != moe['bytes']:
        raise ValueError('original MTP MoE file set or byte accounting changed')
    moe.update(original_frontiers_qualified=True, sampled_draft_logits_qualified=True,
               draft_checks=draft_checks, weight_bytes=weight_bytes)


class MtpMoeBoundaryCapture(MtpKernelBoundaryCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import importlib
        import torch

        if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 8192:
            raise ValueError('MTP MoE capture requires a bounded short control')
        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        root = Path(directory)
        model = self.model_runner.drafter.model
        parent = model.model
        mlp = parent.layers[0].mlp
        router = mlp.experts.router
        routed_module = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
        sources = []
        for value, expected in (
                (type(mlp), '0f7c2df8fa972a193922bad89260d6cf1b6acb97a63b9c6675bc0be362d0a1e5'),
                (type(mlp.shared_expert), '58899ae017336a4ea00e37788788969e17a6178c113961486acca9e6569f4a8f'),
                (type(mlp.experts), 'c6b929944dfab05216164844882adbd3a5266094eec8db71d875a3a993e3b1f0'),
                (type(router), '411faeb99079135084bef8734e82ff1e3d0c82c913198bb35d8140e77c41cbfa'),
                (routed_module, '607c0a459306a71ff7d01445772494367f3924098739bbd3b4f43020738297d4')):
            path = Path(inspect.getsourcefile(value))
            if file_sha(path) != expected:
                raise ValueError('original MTP MoE implementation changed')
            sources.append(dict(file=str(path), sha256=expected))
        if (mlp.tp_size != 1 or mlp.ep_size != 1 or mlp.shared_expert is None or
                mlp.enable_eplb or mlp.is_sequence_parallel or mlp.experts.use_overlapped or
                mlp.experts.is_internal_router or mlp.shared_expert.expert_gate is not mlp.shared_expert_gate):
            raise ValueError('original MTP MoE configuration changed')
        self._qrt_mtp_moe_files = {}
        self._qrt_mtp_moe_drafts = {}
        self._qrt_mtp_moe_bytes = 0
        self._qrt_mtp_moe_handles = []
        self._qrt_mtp_moe_restores = []
        self._qrt_mtp_moe_sources = sources

        def payload(path, tensor):
            data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            with path.open('xb') as stream:
                stream.write(data)
            return dict(file=path.name, dtype=str(tensor.dtype), shape=list(tensor.shape),
                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

        owner = (id(model), str(root.parent))
        if not hasattr(self, '_qrt_mtp_moe_weights'):
            weights = {'router': mlp.gate.weight, 'shared-gate': mlp.shared_expert_gate.weight,
                'shared-gate-up': mlp.shared_expert.gate_up_proj.weight,
                'shared-down': mlp.shared_expert.down_proj.weight, 'final-norm': parent.norm.weight}
            if (any(value.dtype != torch.bfloat16 or tuple(value.shape) != WEIGHTS[label]
                    for label, value in weights.items()) or
                    sum(value.numel() * value.element_size() for value in weights.values()) > MAXIMUM_WEIGHT_BYTES):
                raise ValueError('original MTP MoE weight shape or byte bound changed')
            weights_root = root.parent / 'mtp-moe-original-weights'
            weights_root.mkdir()
            self._qrt_mtp_moe_weights = {label: payload(weights_root / (label + '.bf16.bin'), value)
                for label, value in weights.items()}
            self._qrt_mtp_moe_weight_owner = owner
        elif self._qrt_mtp_moe_weight_owner != owner:
            raise ValueError('original MTP MoE weights changed owners')

        def bounded_payload(name, value):
            size = value.numel() * value.element_size()
            if (self._qrt_mtp_moe_bytes + size > MAXIMUM_BYTES or
                    time.monotonic() - self._qrt_boundary_started > self._qrt_boundary_timeout):
                raise ValueError('MTP MoE observation byte or time bound exceeded')
            result = payload(root / (name + '.bin'), value)
            self._qrt_mtp_moe_bytes += result['bytes']
            return result

        def save(label, value, width=None):
            transaction = self._qrt_mtp_active
            if transaction is None:
                return
            columns, dtype = FRONTIERS[label]
            if width is not None and width != columns:
                raise ValueError('original routed MTP MoE width changed')
            if label == 'topk-ids' and value.dtype == torch.int64:
                if torch.any((value < 0) | (value >= 256)):
                    raise ValueError('original MTP expert index out of range')
                value = value.to(torch.int32)
            if str(value.dtype) != dtype or value.numel() != transaction['token_count'] * columns:
                raise ValueError('original MTP MoE tensor layout changed: ' + label)
            value = value.reshape(transaction['token_count'], columns)
            selected = torch.cat([value[row['row']:row['row'] + 1].detach().cpu()
                                  for row in transaction['rows']])
            key = f"mtp-moe{transaction['ordinal']:04d}-{label}"
            if key in self._qrt_mtp_moe_files:
                raise ValueError('repeated original MTP MoE observation')
            self._qrt_mtp_moe_files[key] = dict(bounded_payload(key, selected),
                label=label, transaction=transaction['ordinal'])

        def output_hook(label):
            def observe(module, args, output):
                save(label, output[0] if isinstance(output, tuple) else output)
            return observe

        for module, label in ((mlp.gate, 'router'), (mlp.shared_expert_gate, 'shared-gate'),
                (mlp.shared_expert.gate_up_proj, 'shared-gate-up'),
                (mlp.shared_expert.down_proj, 'shared-down'), (mlp.shared_expert, 'shared')):
            self._qrt_mtp_moe_handles.append(module.register_forward_hook(output_hook(label)))

        def activated(module, args):
            save('shared-activated', args[0])

        def final_inputs(module, args):
            if len(args) != 2:
                raise ValueError('original MTP final norm arguments changed')
            save('final-hidden', args[0])
            save('final-residual', args[1])

        def expert_outputs(module, args, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError('original MTP shared/routed expert tuple changed')
            save('expert-part-0', output[0])
            save('expert-part-1', output[1])

        self._qrt_mtp_moe_handles.extend((
            mlp.shared_expert.down_proj.register_forward_pre_hook(activated),
            parent.norm.register_forward_pre_hook(final_inputs),
            mlp.experts.register_forward_hook(expert_outputs)))
        original_select = router.select_experts
        original_experts = mlp.experts.forward
        original_logits = model.compute_logits
        self._qrt_mtp_moe_restores.extend(((router, 'select_experts', original_select),
            (mlp.experts, 'forward', original_experts), (model, 'compute_logits', original_logits)))

        def select(*args, **kwargs):
            result = original_select(*args, **kwargs)
            save('topk-weights', result[0])
            save('topk-ids', result[1])
            return result

        def experts(*args, **kwargs):
            transaction = self._qrt_mtp_active
            if transaction is None:
                return original_experts(*args, **kwargs)
            return observe_original_moe_routed(original_experts, routed_module, save,
                transaction['token_count'], *args, **kwargs)

        def logits(*args, **kwargs):
            result = original_logits(*args, **kwargs)
            transaction = self._qrt_mtp_active
            if transaction is None:
                return result
            hidden = args[0] if args else kwargs['hidden_states']
            key = str(transaction['ordinal'])
            if (key in self._qrt_mtp_moe_drafts or list(hidden.shape) != [1, 2048] or
                    hidden.dtype != torch.bfloat16 or result.dtype != torch.float32 or
                    list(result.shape) != [1, 248320]):
                raise ValueError('original MTP sampled logits layout changed')
            prefix = f"mtp-draft{transaction['ordinal']:04d}-"
            self._qrt_mtp_moe_drafts[key] = dict(transaction=transaction['ordinal'],
                sampled_row=transaction['sampled_rows'][0], original_result_returned_unchanged=True,
                hidden=bounded_payload(prefix + 'hidden', hidden),
                logits=bounded_payload(prefix + 'logits', result))
            return result

        router.select_experts = select
        mlp.experts.forward = experts
        model.compute_logits = logits
        record['mtp_moe'] = dict(sources=sources, maximum_saved_bytes=MAXIMUM_BYTES,
            maximum_weight_bytes=MAXIMUM_WEIGHT_BYTES, original_results_returned_unchanged=True)
        return record

    def qrt_finish_token_matrix(self):
        for owner, name, original in self._qrt_mtp_moe_restores:
            setattr(owner, name, original)
        for handle in self._qrt_mtp_moe_handles:
            handle.remove()
        record = super().qrt_finish_token_matrix()
        record['mtp_moe'] = dict(files=self._qrt_mtp_moe_files, bytes=self._qrt_mtp_moe_bytes,
            draft_logits=self._qrt_mtp_moe_drafts, weights=self._qrt_mtp_moe_weights,
            weights_directory='../mtp-moe-original-weights', sources=self._qrt_mtp_moe_sources,
            original_results_returned_unchanged=True, diagnostic_only=True)
        return record
