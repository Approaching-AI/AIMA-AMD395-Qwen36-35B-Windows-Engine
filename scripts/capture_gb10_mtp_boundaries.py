"""Observe the original MTP inputs, projection boundaries and draft decisions.

All hooks copy original results and return None. Wrapped methods return the
original object. These observations are offline evidence, never inference data.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import time

from capture_gb10_runtime_boundaries import RuntimeBoundaryCapture
from capture_sm121_exp2_table import file_sha


EAGLE_SHA = "0e81b97de2426cef270e19e1a9ac53a31e028be9b39f9d2373cbf34a7bb61f5d"
MTP_SHA = "cf97664e82371425df14e412c9d351405d05ae8db3622fc817813ddda6858622"


def selected_draft_rows(positions, token_ids, sample_indices):
    if (not positions or len(positions) != len(token_ids) or
            len(sample_indices) != 1 or type(sample_indices[0]) is not int or
            not 0 <= sample_indices[0] < len(positions) or
            any(type(p) is not int or p < 0 for p in positions) or
            any(type(t) is not int or not 0 <= t < 248320 for t in token_ids) or
            any(b != a + 1 for a, b in zip(positions, positions[1:]))):
        raise ValueError("invalid original batch-one MTP row identity")
    selected = range(len(positions)) if len(positions) <= 2 else sorted(
        {0, len(positions) - 1, sample_indices[0]})
    return [dict(row=i, position=positions[i], input_token_id=token_ids[i],
                 selected_for_sampling=i == sample_indices[0],
                 at_or_before_sample=i <= sample_indices[0]) for i in selected]


def qualify_mtp_capture(worker, prompt_token_ids, output_token_ids):
    """Attach original generated history and next-target identities to copies."""
    capture = worker['mtp_boundaries']
    targets = worker['runtime_boundaries']['transactions']
    history = prompt_token_ids + output_token_ids
    qualified = 0
    backup_rows = 0
    for proposal in capture['transactions']:
        target = targets[proposal['target_transaction']]
        if (proposal['original_max_seq_len'] + 1 > 262144 or
                proposal['first_position'] != target['first_position'] or
                proposal['token_count'] != target['token_count'] or
                proposal['discarded_target'] != target['discarded']):
            raise ValueError("MTP proposal does not follow the original target extent")
        if proposal['discarded_target']:
            # The pinned padded drafter uses the request's last known token as
            # backup during partial prefill, rather than the next chunk token.
            if (proposal['prefill_backup_position'] != len(prompt_token_ids) - 1 or
                    proposal['prefill_backup_token_id'] != prompt_token_ids[-1] or
                    proposal['next_token_ids'] != [proposal['prefill_backup_token_id']]):
                raise ValueError("discarded prefill MTP backup differs from the original prompt")
        for row in proposal['rows']:
            position = row['position'] + 1
            sampled = row['row'] in proposal['sampled_rows']
            if (sampled != row['selected_for_sampling'] or
                    row['at_or_before_sample'] != (row['row'] <= proposal['sampled_rows'][0]) or
                    sampled and row['input_token_id'] != proposal['next_token_ids'][0]):
                raise ValueError("MTP row does not follow the original sample selection")
            row['input_matches_generated_history'] = (
                row['input_token_id'] == history[position] if position < len(history) else None)
            backup = proposal['discarded_target'] and sampled
            row['input_provenance'] = ('discarded_prefill_backup' if backup else
                'accepted_history' if row['at_or_before_sample'] else 'rejected_padding')
            if row['at_or_before_sample'] and not backup:
                if row['input_matches_generated_history'] is False:
                    raise ValueError("accepted MTP input differs from original generated history")
                qualified += row['input_matches_generated_history'] is True
            backup_rows += backup
        next_index = proposal['target_transaction'] + 1
        proposal['next_target_draft_check'] = None
        if not proposal['discarded_target'] and next_index < len(targets):
            following = targets[next_index]
            if following['first_position'] >= len(prompt_token_ids):
                first_matches = following['input_token_ids'][0] == proposal['next_token_ids'][0]
                draft_matches = (following['input_token_ids'][1] == proposal['draft_token_ids'][0][0]
                                 if following['token_count'] == 2 else None)
                if not first_matches or draft_matches is False:
                    raise ValueError("next original target batch differs from its MTP proposal")
                proposal['next_target_draft_check'] = dict(transaction=next_index,
                    first_input_matches=True, scheduled_draft_matches=draft_matches)
    if not qualified:
        raise ValueError("MTP capture lacks original generated-history rows")
    capture['qualified_original_input_rows'] = qualified
    capture['qualified_prefill_backup_rows'] = backup_rows
    capture['original_history_qualified'] = True


class MtpBoundaryCapture(RuntimeBoundaryCapture):
    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import torch

        record = super().qrt_arm_token_matrix(directory, prompt_tokens)
        runner = self.model_runner
        drafter = runner.drafter
        model = drafter.model
        parent = model.model
        sources = []
        for value, expected in ((drafter, EAGLE_SHA), (parent, MTP_SHA)):
            path = Path(inspect.getsourcefile(type(value)))
            if file_sha(path) != expected:
                raise ValueError("original MTP implementation changed")
            sources.append(dict(file=str(path), sha256=expected))
        if (drafter.method != "mtp" or drafter.num_speculative_tokens != 1 or
                drafter.needs_extra_input_slots or not drafter.pass_hidden_states_to_model or
                len(parent.layers) != 1 or parent.config.hidden_size != 2048):
            raise ValueError("unsupported original MTP configuration")
        root = Path(directory)
        self._qrt_mtp_transactions = []
        self._qrt_mtp_files = {}
        self._qrt_mtp_bytes = 0
        self._qrt_mtp_active = None
        self._qrt_mtp_handles = []
        self._qrt_mtp_restores = []

        def payload(path, tensor):
            data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            with path.open('xb') as stream:
                stream.write(data)
            return dict(file=path.name, dtype=str(tensor.dtype), shape=list(tensor.shape),
                        bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

        weights_root = root.parent / 'mtp-original-weights'
        if not hasattr(self, '_qrt_mtp_shared_weights'):
            weights_root.mkdir()
            attention = parent.layers[0].self_attn
            if tuple(parent.fc.weight.shape) != (2048, 4096) or tuple(attention.qkv_proj.weight.shape) != (9216, 2048):
                raise ValueError("original MTP projection shape changed")
            weights = {
                'fc': parent.fc.weight,
                'pre-fc-embedding-norm': parent.pre_fc_norm_embedding.weight,
                'pre-fc-hidden-norm': parent.pre_fc_norm_hidden.weight,
                'input-norm': parent.layers[0].input_layernorm.weight,
                'key-norm': attention.k_norm.weight,
                'kv-projection': attention.qkv_proj.weight[8192:],
            }
            if any(value.dtype != torch.bfloat16 for value in weights.values()):
                raise ValueError("original MTP weights are not BF16")
            self._qrt_mtp_shared_weights = {name: payload(weights_root / (name + '.bf16.bin'), value)
                                           for name, value in weights.items()}
            self._qrt_mtp_shared_owner = (id(model), str(root.parent))
            if sum(row['bytes'] for row in self._qrt_mtp_shared_weights.values()) > 32 << 20:
                raise ValueError("MTP weight observation ceiling exceeded")
        elif self._qrt_mtp_shared_owner != (id(model), str(root.parent)):
            raise ValueError("MTP shared weight observation changed owners")

        def save(label, value, width):
            transaction = self._qrt_mtp_active
            if transaction is None or 'rows' not in transaction:
                return
            if (time.monotonic() - self._qrt_boundary_started > self._qrt_boundary_timeout or
                    value.dtype not in (torch.bfloat16, torch.float32)):
                raise ValueError("MTP observation deadline or dtype changed")
            if value.numel() != transaction['token_count'] * width:
                raise ValueError("MTP boundary shape changed: " + label)
            value = value.reshape(transaction['token_count'], width)
            selected = torch.cat([value[row['row']:row['row'] + 1].detach().cpu()
                                  for row in transaction['rows']])
            size = selected.numel() * selected.element_size()
            if self._qrt_mtp_bytes + size > 64 << 20:
                raise ValueError("MTP observation byte ceiling exceeded")
            key = f"mtp{transaction['ordinal']:04d}-{label}"
            meta = payload(root / (key + '.bin'), selected)
            self._qrt_mtp_bytes += meta['bytes']
            self._qrt_mtp_files[key] = dict(meta, label=label, transaction=transaction['ordinal'])

        def output_hook(label, width):
            def observe(module, args, output):
                save(label, output[0] if isinstance(output, tuple) else output, width)
            return observe

        def input_hook(label, width):
            def observe(module, args):
                save(label, args[0], width)
            return observe

        layer = parent.layers[0]
        attention = layer.self_attn
        for module, label, width in (
                (parent.pre_fc_norm_embedding, 'embedding-norm', 2048),
                (parent.pre_fc_norm_hidden, 'hidden-norm', 2048),
                (parent.fc, 'fusion', 2048),
                (layer.input_layernorm, 'input-norm', 2048),
                (attention.qkv_proj, 'qkv', 9216),
                (attention.q_norm, 'q-norm', 4096),
                (attention.k_norm, 'k-norm', 512),
                (attention.attn, 'context', 4096),
                (attention.o_proj, 'attention-output', 2048),
                (layer.post_attention_layernorm, 'post-attention-norm', 2048),
                (layer.mlp, 'moe-output', 2048),
                (parent.norm, 'final-norm', 2048)):
            self._qrt_mtp_handles.append(module.register_forward_hook(output_hook(label, width)))
        for module, label, width in ((parent.pre_fc_norm_embedding, 'embedding', 2048),
                (parent.fc, 'fusion-input', 4096), (attention.o_proj, 'gated', 4096)):
            self._qrt_mtp_handles.append(module.register_forward_pre_hook(input_hook(label, width)))

        def attention_inputs(module, args):
            if self._qrt_mtp_active is not None:
                for label, value, width in zip(('q-rope', 'k-rope', 'v'), args, (4096, 512, 512)):
                    save(label, value, width)
        self._qrt_mtp_handles.append(attention.attn.register_forward_pre_hook(attention_inputs))
        original_propose = drafter.propose
        original_inputs = drafter.set_inputs_first_pass
        self._qrt_mtp_restores.extend(((drafter, 'propose', original_propose),
                                      (drafter, 'set_inputs_first_pass', original_inputs)))

        def first_pass(*args, **kwargs):
            result = original_inputs(*args, **kwargs)
            count, sampled, metadata = result
            transaction = self._qrt_mtp_active
            if transaction is None or not 1 <= count <= 8192 or metadata.batch_size() != 1:
                raise ValueError("original MTP transaction layout changed")
            positions = drafter._get_positions(count).detach().cpu()
            if positions.ndim == 2:
                if positions.shape[0] != 3 or not torch.equal(positions, positions[:1].expand_as(positions)):
                    raise ValueError("MTP positions are not text-only MRoPE")
                positions = positions[0]
            ids = drafter.input_ids[:count].detach().cpu().tolist()
            selected = selected_draft_rows(positions.tolist(), ids, sampled.detach().cpu().tolist())
            transaction.update(token_count=count, first_position=int(positions[0]),
                rows=selected, sampled_rows=sampled.detach().cpu().tolist(),
                seq_lens=metadata.seq_lens.detach().cpu().tolist(), max_seq_len=metadata.max_seq_len,
                query_start_loc=metadata.query_start_loc.detach().cpu().tolist())
            save('target-hidden', drafter.hidden_states[:count], 2048)
            return result

        def propose(*args, **kwargs):
            if args or self._qrt_mtp_active is not None or len(self._qrt_mtp_transactions) >= 1024:
                raise ValueError("unexpected original MTP proposal call")
            target = self._qrt_boundary_current
            if target is None:
                raise ValueError("MTP proposal lacks its original target transaction")
            metadata = kwargs['common_attn_metadata']
            rejected = kwargs.get('num_rejected_tokens_gpu')
            transaction = dict(ordinal=len(self._qrt_mtp_transactions), target_transaction=target['ordinal'],
                discarded_target=target['discarded'], target_first_position=target['first_position'],
                target_token_count=target['token_count'], original_max_seq_len=metadata.max_seq_len,
                original_num_rejected_tokens=None if rejected is None else rejected.detach().cpu().tolist(),
                next_token_ids=kwargs['next_token_ids'].detach().cpu().tolist())
            if target['discarded']:
                transaction.update(
                    prefill_backup_position=int(runner.input_batch.num_tokens_no_spec[0]) - 1,
                    prefill_backup_token_id=int(drafter.backup_next_token_ids.np[0]))
            self._qrt_mtp_transactions.append(transaction)
            self._qrt_mtp_active = transaction
            try:
                result = original_propose(*args, **kwargs)
                if result.shape != (1, 1):
                    raise ValueError("original MTP draft result shape changed")
                transaction['draft_token_ids'] = result.detach().cpu().tolist()
                transaction['original_result_returned_unchanged'] = True
                return result
            finally:
                self._qrt_mtp_active = None

        drafter.propose = propose
        drafter.set_inputs_first_pass = first_pass
        self._qrt_mtp_sources = sources
        record['mtp_boundaries'] = dict(sources=sources, maximum_saved_bytes=64 << 20,
            maximum_transactions=1024, target_limit=runner.max_model_len,
            drafter_limit=runner.effective_drafter_max_model_len, original_methods_returned_unchanged=True)
        return record

    def qrt_finish_token_matrix(self):
        for owner, name, original in self._qrt_mtp_restores:
            setattr(owner, name, original)
        for handle in self._qrt_mtp_handles:
            handle.remove()
        record = super().qrt_finish_token_matrix()
        required = {'target-hidden', 'embedding', 'embedding-norm', 'hidden-norm', 'fusion-input',
                    'fusion', 'input-norm', 'qkv', 'q-norm', 'k-norm', 'q-rope', 'k-rope', 'v',
                    'context', 'gated', 'attention-output', 'post-attention-norm', 'moe-output', 'final-norm'}
        for transaction in self._qrt_mtp_transactions:
            labels = {meta['label'] for meta in self._qrt_mtp_files.values()
                      if meta['transaction'] == transaction['ordinal']}
            if labels != required or not transaction.get('original_result_returned_unchanged'):
                raise ValueError("incomplete original MTP observations")
        record['mtp_boundaries'] = dict(files=self._qrt_mtp_files, bytes=self._qrt_mtp_bytes,
            transactions=self._qrt_mtp_transactions, sources=self._qrt_mtp_sources,
            weights_directory='../mtp-original-weights', weights=self._qrt_mtp_shared_weights,
            original_methods_returned_unchanged=True, diagnostic_only=True)
        return record
