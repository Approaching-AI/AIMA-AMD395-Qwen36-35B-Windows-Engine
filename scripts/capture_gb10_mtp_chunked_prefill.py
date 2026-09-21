"""Copy actual cold MTP chunk inputs, KV write values and final draft logits."""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
import struct
import time

from capture_gb10_mtp_launchers import MtpKernelBoundaryCapture
from capture_sm121_exp2_table import file_sha


FRONTIERS = {'target-hidden': 2048, 'k-rope': 512, 'v': 512}
MAXIMUM_PROMPT = 32768
MAXIMUM_BYTES = 224 << 20
LOGITS_PROCESSOR_SHA = '53216955b40bafd63b27137162d04e6fa4eff6620807a44684907ddd67f57521'


def chunk_layout(prompt_tokens, transaction):
    first, rows = transaction.get('first_position'), transaction.get('token_count')
    if (type(prompt_tokens) is not int or not 1 <= prompt_tokens <= MAXIMUM_PROMPT or
            type(first) is not int or first < 0 or type(rows) is not int or not 1 <= rows <= 8192 or
            first + rows > prompt_tokens or
            transaction.get('discarded_target') is not (first + rows < prompt_tokens)):
        raise ValueError('original MTP cold chunk extent changed')
    return rows * (2 * sum(FRONTIERS.values()) + 4)


def observe_original_prefill_logits(original, transaction, prompt_tokens, observe, *args, **kwargs):
    result = original(*args, **kwargs)
    if (transaction is not None and transaction.get('first_position', prompt_tokens) < prompt_tokens and
            transaction['first_position'] + transaction['token_count'] == prompt_tokens):
        chunk_layout(prompt_tokens, transaction)
        hidden = args[0] if args else kwargs['hidden_states']
        if (list(hidden.shape) != [1, 2048] or str(hidden.dtype) != 'torch.bfloat16' or
                list(result.shape) != [1, 248320] or str(result.dtype) != 'torch.bfloat16'):
            raise ValueError('original final MTP draft logit layout changed')
        observe(transaction, hidden, result)
    return result


def qualify_chunked_prefill_capture(worker, prompt, outputs, directory):
    full = worker['mtp_chunked_prefill']
    mtp = worker['mtp_boundaries']
    if (not mtp['original_history_qualified'] or not outputs or
            not full['original_results_returned_unchanged'] or full['prompt_tokens'] != len(prompt) or
            full['logits_processor_source']['sha256'] != LOGITS_PROCESSOR_SHA or
            not 0 < full['bytes'] <= MAXIMUM_BYTES):
        raise ValueError('incomplete original MTP cold chunks')
    root = Path(directory)
    total = 0
    seen = set()

    def checked(meta, shape, dtype, element_bytes):
        nonlocal total
        path = root / meta['file']
        expected = math.prod(shape) * element_bytes
        if (path.parent != root or path.is_symlink() or meta['file'] in seen or
                meta['shape'] != shape or meta['dtype'] != dtype or meta['bytes'] != expected or
                path.stat().st_size != expected or file_sha(path) != meta['sha256']):
            raise ValueError('original MTP chunk file identity or layout changed')
        seen.add(meta['file']); total += expected
        return path

    transactions = [t for t in mtp['transactions'] if t['first_position'] < len(prompt)]
    if set(full['chunks']) != {str(t['ordinal']) for t in transactions}:
        raise ValueError('missing or extra original MTP cold chunk')
    selected = {(m['transaction'], m['label']): m for m in mtp['files'].values()}
    cursor = compared = 0
    for transaction in transactions:
        expected_bytes = chunk_layout(len(prompt), transaction)
        first, rows, ordinal = transaction['first_position'], transaction['token_count'], transaction['ordinal']
        chunk = full['chunks'][str(ordinal)]
        if (first != cursor or chunk['first_position'] != first or chunk['rows'] != rows or
                chunk['transaction'] != ordinal or chunk['bytes'] != expected_bytes or
                not transaction['original_result_returned_unchanged'] or set(chunk['files']) != set(FRONTIERS)):
            raise ValueError('original MTP cold chunks are not a complete ordered prompt')
        ids = list(struct.unpack('<' + str(rows) + 'I',
            checked(chunk['shifted_input_ids'], [rows], 'torch.int32', 4).read_bytes()))
        expected_ids = prompt[first + 1:first + rows] + [prompt[-1] if transaction['discarded_target'] else outputs[0]]
        if ids != expected_ids:
            raise ValueError('original MTP chunk shift differs from the actual prompt')
        for label, width in FRONTIERS.items():
            complete = checked(chunk['files'][label], [rows, width], 'torch.bfloat16', 2)
            partial_meta = selected[ordinal, label]
            # Selected observations have their own byte count outside the new
            # full-chunk ledger, but their exact bytes must still be checked.
            before = total
            partial = checked(partial_meta, [len(transaction['rows']), width], 'torch.bfloat16', 2).read_bytes()
            total = before
            with complete.open('rb') as stream:
                for i, row in enumerate(transaction['rows']):
                    index = row['row']
                    if (type(index) is not int or not 0 <= index < rows or
                            row['position'] != first + index or row['input_token_id'] != ids[index]):
                        raise ValueError('selected original MTP chunk row identity changed')
                    stream.seek(index * width * 2)
                    if stream.read(width * 2) != partial[i * width * 2:(i + 1) * width * 2]:
                        raise ValueError('original MTP chunk and selected observations differ')
                    compared += 1
        cursor += rows
    if cursor != len(prompt) or not transactions:
        raise ValueError('incomplete original MTP cold prompt')
    draft = full['final_draft']; final = transactions[-1]
    if (draft['transaction'] != final['ordinal'] or draft['sampled_row'] != final['token_count'] - 1 or
            final['sampled_rows'] != [draft['sampled_row']] or not draft['original_result_returned_unchanged']):
        raise ValueError('final MTP draft is not paired with the complete prompt')
    hidden = checked(draft['hidden'], [1, 2048], 'torch.bfloat16', 2).read_bytes()
    before = total
    selected_norm = checked(selected[final['ordinal'], 'final-norm'],
        [len(final['rows']), 2048], 'torch.bfloat16', 2).read_bytes()
    total = before
    sampled = [i for i, row in enumerate(final['rows']) if row['row'] == draft['sampled_row']]
    if len(sampled) != 1 or hidden != selected_norm[sampled[0] * 4096:(sampled[0] + 1) * 4096]:
        raise ValueError('original MTP final draft hidden differs from the sampled norm row')
    raw = checked(draft['logits'], [1, 248320], 'torch.bfloat16', 2).read_bytes()
    words = struct.unpack('<248320H', raw)
    logits = struct.unpack('<248320f', b''.join(struct.pack('<I', word << 16) for word in words))
    if not all(math.isfinite(value) for value in logits):
        raise ValueError('nonfinite original MTP final draft logits')
    token = max(range(len(logits)), key=logits.__getitem__)
    if final['draft_token_ids'] != [[token]] or total != full['bytes']:
        raise ValueError('original MTP final sample or byte accounting differs')
    full.update(original_chunk_frontiers_qualified=True, actual_shifted_prompt_ids_qualified=True,
        selected_observation_rows_compared=compared, final_draft_token=token, final_draft_logit=logits[token],
        final_draft_logits_qualified=True)


class MtpChunkedPrefillBoundaryCapture(MtpKernelBoundaryCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= MAXIMUM_PROMPT:
            raise ValueError('bounded cold MTP prompt required')
        self._qrt_mtp_chunk_root = Path(directory)
        self._qrt_mtp_chunk_prompt = prompt_tokens
        self._qrt_mtp_chunks = {}
        self._qrt_mtp_chunk_bytes = 0
        self._qrt_mtp_final_draft = None
        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        model = self.model_runner.drafter.model
        path = Path(inspect.getsourcefile(type(model.logits_processor)))
        if file_sha(path) != LOGITS_PROCESSOR_SHA:
            raise ValueError('original MTP logits processor changed')
        self._qrt_mtp_chunk_logit_source = dict(file=str(path), sha256=LOGITS_PROCESSOR_SHA)
        original = model.compute_logits
        self._qrt_mtp_restores.append((model, 'compute_logits', original))

        def observe(transaction, hidden, logits):
            if self._qrt_mtp_final_draft is not None:
                raise ValueError('repeated original MTP final prefill logits')
            self._qrt_mtp_final_draft = dict(transaction=transaction['ordinal'],
                sampled_row=transaction['sampled_rows'][0], original_result_returned_unchanged=True,
                hidden=self._qrt_save_mtp_chunk('final-draft-hidden', hidden),
                logits=self._qrt_save_mtp_chunk('final-draft-logits', logits))

        def logits(*args, **kwargs):
            return observe_original_prefill_logits(original, self._qrt_mtp_active,
                prompt_tokens, observe, *args, **kwargs)

        model.compute_logits = logits
        record['mtp_chunked_prefill'] = dict(maximum_saved_bytes=MAXIMUM_BYTES,
            maximum_prompt_tokens=MAXIMUM_PROMPT, original_results_returned_unchanged=True)
        return record

    def _qrt_save_mtp_chunk(self, name, tensor):
        import torch

        size = tensor.numel() * tensor.element_size()
        if (self._qrt_mtp_chunk_bytes + size > MAXIMUM_BYTES or
                time.monotonic() - self._qrt_boundary_started > self._qrt_boundary_timeout):
            raise ValueError('original MTP chunk byte or time bound exceeded')
        data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        path = self._qrt_mtp_chunk_root / ('full-mtp-' + name + '.bin')
        with path.open('xb') as stream:
            stream.write(data)
        self._qrt_mtp_chunk_bytes += len(data)
        return dict(file=path.name, dtype=str(tensor.dtype), shape=list(tensor.shape),
            bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def qrt_observe_mtp_full_frontier(self, label, value, width):
        import torch

        transaction = self._qrt_mtp_active
        if (transaction is None or transaction['first_position'] >= self._qrt_mtp_chunk_prompt or label not in FRONTIERS):
            return
        required = chunk_layout(self._qrt_mtp_chunk_prompt, transaction)
        rows, ordinal = transaction['token_count'], transaction['ordinal']
        if width != FRONTIERS[label] or value.dtype != torch.bfloat16 or list(value.shape) != [rows, width]:
            raise ValueError('original MTP complete chunk tensor layout changed')
        key = str(ordinal)
        if key not in self._qrt_mtp_chunks:
            if label != 'target-hidden':
                raise ValueError('original MTP chunk lacks its actual target hidden input')
            ids = self.model_runner.drafter.input_ids[:rows].detach().cpu().to(torch.int32)
            self._qrt_mtp_chunks[key] = dict(first_position=transaction['first_position'], rows=rows,
                transaction=ordinal, files={}, bytes=required,
                shifted_input_ids=self._qrt_save_mtp_chunk(f'chunk{ordinal:04d}-shifted-input-ids', ids))
        chunk = self._qrt_mtp_chunks[key]
        if label in chunk['files']:
            raise ValueError('repeated original MTP chunk tensor')
        chunk['files'][label] = self._qrt_save_mtp_chunk(f'chunk{ordinal:04d}-' + label, value)

    def qrt_finish_token_matrix(self):
        record = super().qrt_finish_token_matrix()
        if not self._qrt_mtp_chunks or self._qrt_mtp_final_draft is None:
            raise ValueError('incomplete original MTP chunk capture')
        record['mtp_chunked_prefill'] = dict(chunks=self._qrt_mtp_chunks, final_draft=self._qrt_mtp_final_draft,
            bytes=self._qrt_mtp_chunk_bytes, prompt_tokens=self._qrt_mtp_chunk_prompt,
            logits_processor_source=self._qrt_mtp_chunk_logit_source, maximum_saved_bytes=MAXIMUM_BYTES,
            original_results_returned_unchanged=True, diagnostic_only=True)
        return record
