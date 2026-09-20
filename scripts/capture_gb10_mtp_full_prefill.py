"""Copy the complete first MTP cache frontend without changing its execution."""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import time

from capture_gb10_mtp_launchers import MtpKernelBoundaryCapture
from capture_sm121_exp2_table import file_sha


# Original label: (saved label, original row width, first feature, saved width).
FULL_FRONTIERS = {
    'target-hidden': ('target-hidden', 2048, 0, 2048),
    'embedding': ('embedding', 2048, 0, 2048),
    'embedding-norm': ('embedding-norm', 2048, 0, 2048),
    'hidden-norm': ('hidden-norm', 2048, 0, 2048),
    'fusion-input': ('fusion-input', 4096, 0, 4096),
    'fusion': ('fusion', 2048, 0, 2048),
    'input-norm': ('input-norm', 2048, 0, 2048),
    'qkv': ('kv-projection', 9216, 8192, 1024),
    'k-norm': ('k-norm', 512, 0, 512),
    'k-rope': ('k-rope', 512, 0, 512),
    'v': ('v', 512, 0, 512),
}
MAXIMUM_FULL_BYTES = 320 << 20


def full_prefill_layout(prompt_tokens, transaction):
    if (type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 8192 or
            transaction.get('ordinal') != 0 or transaction.get('first_position') != 0 or
            transaction.get('token_count') != prompt_tokens or transaction.get('discarded_target') is not False):
        raise ValueError('full MTP frontier requires one complete first prefill batch')
    bytes_required = prompt_tokens * (sum(v[3] for v in FULL_FRONTIERS.values()) * 2 + 4)
    if bytes_required > MAXIMUM_FULL_BYTES:
        raise ValueError('full MTP frontier byte bound')
    return bytes_required


def qualify_full_prefill_capture(worker, prompt_token_ids, output_token_ids, directory):
    full = worker['mtp_full_prefill']
    mtp = worker['mtp_boundaries']
    transaction = mtp['transactions'][0]
    expected_bytes = full_prefill_layout(len(prompt_token_ids), transaction)
    if (not mtp['original_history_qualified'] or not output_token_ids or
            not full['original_results_returned_unchanged'] or full['bytes'] != expected_bytes or
            full['rows'] != len(prompt_token_ids) or
            full['first_position'] != 0 or full['transaction'] != 0 or not transaction['rows'] or
            set(full['files']) != {rule[0] for rule in FULL_FRONTIERS.values()}):
        raise ValueError('incomplete original full MTP frontier')
    root = Path(directory)

    def checked(meta):
        path = root / meta['file']
        if (path.parent != root or path.is_symlink() or path.stat().st_size != meta['bytes'] or
                file_sha(path) != meta['sha256']):
            raise ValueError('full MTP frontier file identity changed')
        return path

    ids_meta = full['shifted_input_ids']
    if (ids_meta['bytes'] != len(prompt_token_ids) * 4 or ids_meta['dtype'] != 'torch.int32' or
            ids_meta['shape'] != [len(prompt_token_ids)]):
        raise ValueError('full MTP input-ID layout changed')
    ids = list(struct.unpack('<' + str(len(prompt_token_ids)) + 'I', checked(ids_meta).read_bytes()))
    if ids != prompt_token_ids[1:] + [output_token_ids[0]]:
        raise ValueError('full MTP shifted inputs differ from the actual request and original first sample')
    selected = {(meta['transaction'], meta['label']): meta for meta in mtp['files'].values()}
    compared_rows = 0
    for original_label, (label, original_width, begin, width) in FULL_FRONTIERS.items():
        meta = full['files'][label]
        if (meta['dtype'] != 'torch.bfloat16' or meta['shape'] != [len(prompt_token_ids), width] or
                meta['bytes'] != len(prompt_token_ids) * width * 2 or meta['source_label'] != original_label or
                meta['source_width'] != original_width or meta['first_feature'] != begin):
            raise ValueError('full MTP tensor layout changed')
        complete_path = checked(meta)
        partial_meta = selected[0, original_label]
        partial = checked(partial_meta).read_bytes()
        if (partial_meta['dtype'] != 'torch.bfloat16' or
                partial_meta['shape'] != [len(transaction['rows']), original_width] or
                partial_meta['bytes'] != len(transaction['rows']) * original_width * 2):
            raise ValueError('selected MTP observation layout changed')
        with complete_path.open('rb') as stream:
            for index, row in enumerate(transaction['rows']):
                if (type(row['row']) is not int or not 0 <= row['row'] < len(prompt_token_ids) or
                        row['position'] != row['row'] or ids[row['row']] != row['input_token_id']):
                    raise ValueError('selected MTP identity differs from full prefill')
                stream.seek(row['row'] * width * 2)
                offset = (index * original_width + begin) * 2
                if stream.read(width * 2) != partial[offset:offset + width * 2]:
                    raise ValueError('full and selected original MTP observations differ')
                compared_rows += 1
    full.update(actual_shifted_prompt_ids_qualified=True, selected_observation_rows_compared=compared_rows,
                all_selected_observations_match=True, original_full_frontiers_qualified=True)


class MtpFullPrefillBoundaryCapture(MtpKernelBoundaryCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 8192:
            raise ValueError('full MTP capture requires at most 8192 prompt tokens')
        self._qrt_mtp_full_root = Path(directory)
        self._qrt_mtp_full_prompt = prompt_tokens
        self._qrt_mtp_full_files = {}
        self._qrt_mtp_full_ids = None
        self._qrt_mtp_full_bytes = 0
        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        record['mtp_full_prefill'] = dict(maximum_saved_bytes=MAXIMUM_FULL_BYTES,
            maximum_rows=8192, first_transaction_only=True, original_results_returned_unchanged=True)
        return record

    def qrt_observe_mtp_full_frontier(self, label, value, width):
        import torch

        transaction = self._qrt_mtp_active
        if transaction is None or transaction['ordinal'] != 0 or label not in FULL_FRONTIERS:
            return
        full_prefill_layout(self._qrt_mtp_full_prompt, transaction)
        name, original_width, begin, columns = FULL_FRONTIERS[label]
        if (width != original_width or value.dtype != torch.bfloat16 or
                list(value.shape) != [self._qrt_mtp_full_prompt, original_width] or
                name in self._qrt_mtp_full_files or
                time.monotonic() - self._qrt_boundary_started > self._qrt_boundary_timeout):
            raise ValueError('full MTP frontier shape, uniqueness or deadline changed')

        def save(name, tensor):
            size = tensor.numel() * tensor.element_size()
            if self._qrt_mtp_full_bytes + size > MAXIMUM_FULL_BYTES:
                raise ValueError('full MTP frontier byte bound exceeded')
            data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            path = self._qrt_mtp_full_root / ('full-mtp-' + name + '.bin')
            with path.open('xb') as stream:
                stream.write(data)
            self._qrt_mtp_full_bytes += len(data)
            return dict(file=path.name, dtype=str(tensor.dtype), shape=list(tensor.shape),
                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

        if self._qrt_mtp_full_ids is None:
            if label != 'target-hidden':
                raise ValueError('full MTP observation lacks its original target hidden input')
            ids = self.model_runner.drafter.input_ids[:self._qrt_mtp_full_prompt].detach().cpu().to(torch.int32)
            self._qrt_mtp_full_ids = save('shifted-input-ids', ids)
        meta = save(name, value[:, begin:begin + columns])
        self._qrt_mtp_full_files[name] = dict(meta, source_label=label,
            source_width=original_width, first_feature=begin, transaction=0)

    def qrt_finish_token_matrix(self):
        record = super().qrt_finish_token_matrix()
        if (set(self._qrt_mtp_full_files) != {v[0] for v in FULL_FRONTIERS.values()} or
                self._qrt_mtp_full_ids is None):
            raise ValueError('incomplete full MTP prefill capture')
        record['mtp_full_prefill'] = dict(files=self._qrt_mtp_full_files,
            shifted_input_ids=self._qrt_mtp_full_ids, bytes=self._qrt_mtp_full_bytes,
            rows=self._qrt_mtp_full_prompt, first_position=0, transaction=0,
            maximum_saved_bytes=MAXIMUM_FULL_BYTES, original_results_returned_unchanged=True,
            diagnostic_only=True)
        return record
