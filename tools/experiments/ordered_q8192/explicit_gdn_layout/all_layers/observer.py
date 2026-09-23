"""Observe original q8192 GDN calls; candidate results never enter the model."""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import sys
import time

from capture_gb10_token_matrix import TokenMatrixCapture


ROOT = Path('/work')
LAYERS = [i for i in range(40) if i % 4 != 3]


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def tensor_record(value):
    import torch
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def difference(actual, expected):
    import torch
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    unequal = actual.contiguous().view(integer) != expected.contiguous().view(integer)
    count = int(unequal.sum())
    result = dict(values=actual.numel(), bit_mismatches=count,
                  actual=tensor_record(actual), original=tensor_record(expected))
    if count:
        indices = unequal.reshape(-1).nonzero()[:32, 0]
        result.update(maximum_absolute_error=float((actual.float() - expected.float()).abs().max()),
                      first_differences=[dict(flat_index=int(i), actual=float(a), original=float(e))
                          for i, a, e in zip(indices.cpu(), actual.reshape(-1)[indices].cpu(),
                                             expected.reshape(-1)[indices].cpu())])
    return result


class Pipeline:
    def __init__(self):
        import torch
        import triton
        assert torch.cuda.get_device_capability() == (12, 1) and triton.__version__ == '3.6.0'
        self.manifest = json.loads((ROOT / 'source-inputs.json').read_text())
        for name, digest in self.manifest['files'].items():
            assert sha(ROOT / name) == digest, name
        self.variants = {}
        previous = {key: sys.modules.get(key) for key in ('integer_u', 'group16', 'exp2')}
        try:
            directory = ROOT / 'candidate/u64'
            arithmetic = module('ordered_u64_arithmetic', directory / 'integer_u.py')
            sys.modules['integer_u'] = arithmetic
            pipeline = module('ordered_u64_pipeline', directory / 'ordered_pipeline.py')
            self.variants['u64'] = (pipeline, arithmetic.integer_u_kernel)
            directory = ROOT / 'candidate/explicit_layout'
            sys.modules['group16'] = module('ordered_explicit_group16', directory / 'group16.py')
            sys.modules['exp2'] = module('ordered_explicit_exp2', directory / 'exp2.py')
            pipeline = module('ordered_explicit_pipeline', directory / 'ordered_pipeline.py')
            self.variants['explicit_layout'] = (pipeline, pipeline.integer_u_kernel)
        finally:
            for key, prior in previous.items():
                if prior is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = prior
        self.inverse = module('ordered_inverse_all_layers', ROOT / 'candidate/u64/ordered_inverse.py').native_ordered_64_inverse_kernel
        self.cumsum = module('ordered_cumsum_all_layers', ROOT / 'candidate/cumsum.py').chunk_local_cumsum_scalar_kernel
        path = Path('/table/exp2.bin')
        assert path.stat().st_size == 183174448 and sha(path) == 'f490940df2bd80421159a96424c3e922330b7ca120d5ae7b629a973b9183730b'
        self.table = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).cuda()
        self.table_hash = tensor_record(self.table)
        self.original_sources = []
        from vllm.model_executor.layers.fla.ops import l2norm, cumsum, chunk
        expected = self.manifest['original_operator_sources']
        for source in (l2norm, cumsum, chunk):
            path = Path(inspect.getsourcefile(source))
            assert sha(path) == expected[str(path)], str(path)
            self.original_sources.append(dict(file=str(path), sha256=sha(path)))
        assert not l2norm.USE_DEFAULT_FLA_NORM
        self.normalize = l2norm.l2norm_fwd
        self.original_cumsum = cumsum.chunk_local_cumsum

    def run(self, variant, q, k, v, beta, g, initial):
        import torch
        import triton
        pipeline, u_kernel = self.variants[variant]
        T = 8192
        guarded = []

        def output(shape, dtype, clear=False):
            n = 1
            for extent in shape:
                n *= extent
            item = 2 if dtype == torch.bfloat16 else 4
            backing = torch.full((n * item + 1024,), 0xa5, dtype=torch.uint8, device='cuda')
            tensor = backing[512:-512].view(dtype).reshape(shape)
            if clear:
                tensor.zero_()
            guarded.append(backing)
            return tensor

        common = dict(BM=8, BN=8, num_warps=4, num_stages=2, enable_fp_fusion=False)
        a = output((1, T, 32, 64), torch.float32)
        inverse = output(a.shape, torch.bfloat16, True)
        w = output((1, T, 32, 128), torch.bfloat16)
        u = output(w.shape, torch.bfloat16)
        scores = output(a.shape, torch.bfloat16)
        vn = output((1, 64, 32, 128), torch.bfloat16)
        residual = output(vn.shape, torch.bfloat16)
        core = output(w.shape, torch.bfloat16)
        states = [output(initial.shape, torch.float32) for _ in range(2)]
        states[0].copy_(initial)
        table = self.table
        dummy = torch.empty(1, device='cuda')
        pipeline.gram_kernel[(T // 8, 8, 32)](k, k, beta, g, table, a, T, SCORE=False, **common)
        self.inverse[(T // 64, 32)](a, inverse, None, None, T, H=32, BT=64, USE_TMA=False,
            IS_VARLEN=False, DOT_PRECISION='ieee', num_warps=4, num_stages=2, enable_fp_fusion=False)
        pipeline.w_kernel[(T // 8, 16, 32)](inverse, k, beta, g, table, w, T, **common)
        u_kernel[(T // 8, 16, 32)](inverse, v, beta, u, dummy, T, H=32, NV=128, CAPTURE_F32=False, **common)
        pipeline.gram_kernel[(T // 8, 8, 32)](q, k, beta, g, table, scores, T, SCORE=True, **common)
        current = states[0]
        for chunk, start in enumerate(range(0, T, 64)):
            end = start + 64
            gate = g[:, start:end]
            pipeline.residual_kernel[(8, 16, 32)](w[:, start:end], u[:, start:end], current,
                gate, table, vn, residual, 64, **common)
            next_state = states[(chunk + 1) % 2]
            pipeline.state_kernel[(16, 16, 32)](k[:, start:end], residual, current, gate,
                table, next_state, 64, **common)
            pipeline.output_kernel[(8, 16, 32)](q[:, start:end], vn, current, gate,
                scores[:, start:end], table, core[:, start:end], 64, **common)
            current = next_state
        torch.cuda.synchronize()
        guards = all(bool((b[:512] == 0xa5).all()) and bool((b[-512:] == 0xa5).all()) for b in guarded)
        return core, current, guards

    def compare(self, values, original):
        import torch
        q = self.normalize(values['q'])
        k = self.normalize(values['k'])
        beta = values['beta'].float()
        g = torch.empty_like(values['g'])
        self.cumsum[(128, 32)](values['g'], g, None, None, 8192, B=1, H=32, BT=64,
            REVERSE=False, IS_VARLEN=False, HEAD_FIRST=False, num_warps=2,
            num_stages=3, enable_fp_fusion=True)
        original_g = self.original_cumsum(values['g'], chunk_size=64, cu_seqlens=values['cu_seqlens'])
        prep = dict(q_normalized=tensor_record(q), k_normalized=tensor_record(k),
                    cumsum=difference(g, original_g), normalization='unchanged original l2norm_fwd')
        assert prep['cumsum']['bit_mismatches'] == 0
        rows = []
        for variant in ('u64', 'explicit_layout'):
            begun = time.monotonic()
            core, state, guards = self.run(variant, q, k, values['v'], beta, g, values['initial_state'])
            rows.append(dict(variant=variant, core=difference(core, original[0]),
                final_state=difference(state, original[1]), guards_pass=guards,
                wall_seconds=time.monotonic() - begun, performance_acceptance=False))
            del core, state
        return prep, rows


class OrderedGdnCapture(TokenMatrixCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import torch
        result = super().qrt_arm_token_matrix(directory, prompt_tokens)
        self._ordered_handles = []
        self._ordered_pipeline = None
        self._ordered_rows = []
        self._ordered_pending = {}
        self._ordered_root = Path(directory)
        self._ordered_enabled = self._ordered_root.name == 'q8192-out512'
        if not self._ordered_enabled:
            return result
        assert prompt_tokens == 8192
        self._ordered_pipeline = Pipeline()
        self._ordered_started = time.monotonic()
        candidates = [(name, layer) for name, layer in self.model_runner.model.named_modules()
            if isinstance(layer, torch.nn.ModuleList) and len(layer) == 40 and hasattr(layer[3], 'self_attn')]
        assert len(candidates) == 1
        name, layers = candidates[0]

        def attach(index):
            linear = layers[index].linear_attn
            assert not linear.gqa_interleaved_layout and linear.tp_size == 1
            op = linear.chunk_gated_delta_rule
            assert op._forward_method.__name__ == 'forward_native'

            def before(module, args, kwargs):
                assert index not in self._ordered_pending
                q = kwargs['q']
                assert tuple(q.shape) == (1, 8192, 16, 128) and q.dtype == torch.bfloat16
                assert index not in [row['layer'] for row in self._ordered_rows]
                assert kwargs['use_qk_l2norm_in_kernel'] and kwargs['output_final_state']
                assert kwargs['cu_seqlens'].tolist() == [0, 8192]
                values = {key: kwargs[key].detach().contiguous().clone() for key in
                          ('q', 'k', 'v', 'g', 'beta', 'initial_state', 'cu_seqlens')}
                assert tuple(values['v'].shape) == (1, 8192, 32, 128)
                assert tuple(values['initial_state'].shape) == (1, 32, 128, 128)
                assert values['initial_state'].dtype == torch.float32
                assert int(torch.count_nonzero(values['initial_state'])) == 0
                self._ordered_pending[index] = dict(values=values,
                    inputs={key: tensor_record(value) for key, value in values.items()})

            def after(module, args, kwargs, output):
                pending = self._ordered_pending.pop(index)
                values = pending['values']
                assert len(output) == 2 and output[0].dtype == torch.bfloat16 and output[1].dtype == torch.float32
                original_hashes = [tensor_record(value) for value in output]
                preparation, variants = self._ordered_pipeline.compare(values, output)
                unchanged = all(tensor_record(value) == pending['inputs'][key] for key, value in values.items())
                original_unchanged = original_hashes == [tensor_record(value) for value in output]
                actual_inputs_unchanged = all(tensor_record(kwargs[key]) == pending['inputs'][key] for key in values)
                row = dict(layer=index, tokens=8192, chunks=128, original_inputs=pending['inputs'],
                    original_outputs=original_hashes, preparation=preparation, variants=variants,
                    copied_inputs_unchanged=unchanged, original_inputs_unchanged=actual_inputs_unchanged,
                    original_outputs_unchanged=original_unchanged, candidate_outputs_fed_to_model=False)
                self._ordered_rows.append(row)
                (self._ordered_root / ('ordered-layer-%02d.json' % index)).write_text(json.dumps(row, indent=2) + '\n')
                print(json.dumps(dict(event='ordered_gdn_layer_comparison', layer=index,
                    variants=[dict(variant=v['variant'], core_mismatches=v['core']['bit_mismatches'],
                        state_mismatches=v['final_state']['bit_mismatches'], guards_pass=v['guards_pass'],
                        wall_seconds=v['wall_seconds']) for v in variants],
                    original_inputs_unchanged=actual_inputs_unchanged,
                    elapsed_seconds=time.monotonic() - self._ordered_started)), flush=True)
                assert unchanged and actual_inputs_unchanged and original_unchanged
                assert all(v['guards_pass'] for v in variants)
                # Return None: PyTorch passes the exact original result onward.

            self._ordered_handles += [op.register_forward_pre_hook(before, with_kwargs=True),
                op.register_forward_hook(after, with_kwargs=True)]

        for index in LAYERS:
            attach(index)
        result['ordered_gdn'] = dict(layer_container=name, layers=LAYERS, tokens=8192,
            variants=['u64', 'explicit_layout'], source_inputs_sha256=sha(ROOT / 'source-inputs.json'),
            original_operator_sources=self._ordered_pipeline.original_sources,
            candidate_outputs_fed_to_model=False, model_compute_sources_modified=False)
        return result

    def qrt_finish_token_matrix(self):
        for handle in self._ordered_handles:
            handle.remove()
        result = super().qrt_finish_token_matrix()
        if self._ordered_enabled:
            table_unchanged = tensor_record(self._ordered_pipeline.table) == self._ordered_pipeline.table_hash
            rows = self._ordered_rows
            all_match = (len(rows) == 30 and [r['layer'] for r in rows] == LAYERS and
                all(r['original_inputs_unchanged'] and r['copied_inputs_unchanged'] and r['original_outputs_unchanged']
                    and all(v['guards_pass'] and v['core']['bit_mismatches'] == 0 and
                            v['final_state']['bit_mismatches'] == 0 for v in r['variants']) for r in rows))
            report = dict(layers=rows, layers_complete=[r['layer'] for r in rows] == LAYERS,
                all_original_gdn_results_match=all_match, exp2_table_unchanged=table_unchanged,
                pending_calls=len(self._ordered_pending), candidate_outputs_fed_to_model=False,
                model_compute_sources_modified=False, windows_acceptance=False, performance_acceptance=False)
            (self._ordered_root / 'ordered-summary.json').write_text(json.dumps(report, indent=2) + '\n')
            result['ordered_gdn'] = report
        self._ordered_pipeline = None
        self._ordered_handles = []
        self._ordered_pending = {}
        return result
