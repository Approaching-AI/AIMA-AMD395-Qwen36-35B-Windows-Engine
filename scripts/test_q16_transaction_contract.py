#!/usr/bin/env python3
"""Host-only transaction contract for a q16 target verifier.

Pins completion[0:16] -> completion[1:17], whole-block commit, whole-block
discard, resident-state immutability on mismatch, and deterministic telemetry.
This is an executable design contract, not an inference-success claim.

Run: python scripts/test_q16_transaction_contract.py
"""

import copy
import hashlib
import json


CONTRACT = "qrt.q16.transaction.v1"
Q = 16
MISMATCH_SLOTS = (0, 1, 7, 14)

# First 64 token IDs from gb10-q8192-speculative64-20260712T230000Z-342cd4c1.
# Source response SHA-256:
# ed0456ee8487056f8bd759928132924a002caf18853d2e31e887d213749549a8
COMPLETION = (
    16, 15, 15, 220, 16, 15, 16, 220, 16, 15, 17, 220, 16, 15, 18, 220,
    16, 15, 19, 220, 16, 15, 20, 220, 16, 15, 21, 220, 16, 15, 22, 220,
    16, 15, 23, 220, 16, 15, 24, 220, 16, 16, 15, 220, 16, 16, 16, 220,
    16, 16, 17, 220, 16, 16, 18, 220, 16, 16, 19, 220, 16, 16, 20, 220,
)
PINNED_HASHES = {
    "oracle_sha256": "d4b58d2f4715c173b276250afe7190ee33af33022eb051f0cc35ec8bd3e0394c",
    "input_sha256": "e849959afa8404e6e0e817c38ed3978499537b8d3630c8c5ed91dfb69774c86c",
    "expected_output_sha256": "ae4d6d28ca9efc9990ed3df330031e366c181409deadc17893055712651b3c00",
    "state_before_sha256": "ed47a37a8b1a0224ea7000328dd3d219e21a1cfa6899dfa065f0bc0a83ee80a5",
    "state_after_sha256": "c307bd1ba5d50a39ea1c8bd131dc7c58018c9ed401401470b31171c733dbd758",
}


def sha256(kind, value):
    payload = {"contract": CONTRACT, "kind": kind, "value": value}
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def initial_resident():
    # completion[0] has been streamed by prefill but has not been processed by
    # decode. The resident wrapper lets commit publish shadow state by one
    # reference swap; discard performs no assignment at all.
    return {
        "state": {
            "position": 8192,
            "processed_token_ids": [248068, 271, 248069, 271],
            "emitted_token_ids": [COMPLETION[0]],
            "linear_words": [0x10203040, 0x55667788, 0x90ABCDEF, 0x13579BDF],
            "full_kv_tail_words": [
                0x00001FFE0003C904,
                0x00001FFF0000000F,
                0x00001FFF00000010,
                0x00001FFF000000DC,
            ],
        }
    }


def state_sha256(resident):
    return sha256("resident_state", resident["state"])


def oracle_window(completion):
    completion = tuple(completion)
    if len(completion) < Q + 1:
        raise ValueError("q16 verification requires completion[0:17]")
    return completion[0:Q], completion[1 : Q + 1]


def target_shadow(resident, inputs, outputs):
    shadow = copy.deepcopy(resident["state"])
    start = shadow["position"]
    for offset, token_id in enumerate(inputs):
        position = start + offset
        word_index = position % len(shadow["linear_words"])
        old_word = shadow["linear_words"][word_index]
        shadow["linear_words"][word_index] = (
            (old_word * 0x9E3779B1) ^ token_id ^ (position * 0x85EBCA77)
        ) & 0xFFFFFFFF
        shadow["full_kv_tail_words"].append(
            ((position & 0xFFFFFFFF) << 32) | (token_id & 0xFFFFFFFF)
        )
    shadow["position"] += Q
    shadow["processed_token_ids"].extend(inputs)
    shadow["emitted_token_ids"].extend(outputs)
    shadow["full_kv_tail_words"] = shadow["full_kv_tail_words"][-32:]
    return shadow


def transaction_counters(first_mismatch):
    commit = first_mismatch is None
    return {
        "attempted_blocks": 1,
        "attempted_input_tokens": Q,
        "verified_output_tokens": Q,
        "matching_prefix_tokens": Q if commit else first_mismatch,
        "committed_blocks": 1 if commit else 0,
        "committed_input_tokens": Q if commit else 0,
        "emitted_tokens": Q if commit else 0,
        "discarded_blocks": 0 if commit else 1,
        "discarded_input_tokens": 0 if commit else Q,
        "mismatch_count": 0 if commit else 1,
    }


def verify_transaction(resident, completion, candidates):
    inputs, expected = oracle_window(completion)
    candidates = tuple(candidates)
    if len(candidates) != Q:
        raise ValueError("q16 candidate output must contain exactly 16 tokens")

    before_sha = state_sha256(resident)
    shadow = target_shadow(resident, inputs, expected)
    shadow_sha = sha256("resident_state", shadow)
    mismatch = next(
        (
            slot
            for slot, pair in enumerate(zip(expected, candidates, strict=True))
            if pair[0] != pair[1]
        ),
        None,
    )
    if mismatch is None:
        resident["state"] = shadow

    return {
        "decision": "commit" if mismatch is None else "discard",
        "input_completion_indices": list(range(0, Q)),
        "output_completion_indices": list(range(1, Q + 1)),
        "input_token_ids": list(inputs),
        "expected_output_token_ids": list(expected),
        "candidate_output_token_ids": list(candidates),
        "first_mismatch_slot": mismatch,
        "first_mismatch_expected": None if mismatch is None else expected[mismatch],
        "first_mismatch_candidate": (
            None if mismatch is None else candidates[mismatch]
        ),
        "oracle_sha256": sha256("gb10_completion_token_ids", list(completion)),
        "input_sha256": sha256("verifier_input_token_ids", list(inputs)),
        "expected_output_sha256": sha256("verifier_output_token_ids", list(expected)),
        "candidate_output_sha256": sha256(
            "verifier_output_token_ids", list(candidates)
        ),
        "state_before_sha256": before_sha,
        "shadow_state_sha256": shadow_sha,
        "state_after_sha256": state_sha256(resident),
        "counters": transaction_counters(mismatch),
    }


def mismatch_candidates(slot):
    _, expected = oracle_window(COMPLETION)
    candidates = list(expected)
    candidates[slot] = 1_000_000 + slot
    return candidates


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def test_indexing():
    inputs, outputs = oracle_window(COMPLETION)
    require(inputs == COMPLETION[0:16], "input window is not completion[0:16]")
    require(outputs == COMPLETION[1:17], "output window is not completion[1:17]")
    require((inputs[0], inputs[15]) == (COMPLETION[0], COMPLETION[15]), "input endpoints moved")
    require((outputs[0], outputs[15]) == (COMPLETION[1], COMPLETION[16]), "output endpoints moved")
    return True


def test_all_accept():
    resident = initial_resident()
    before = copy.deepcopy(resident)
    inputs, outputs = oracle_window(COMPLETION)
    report = verify_transaction(resident, COMPLETION, outputs)
    require(report["decision"] == "commit", "exact q16 block did not commit")
    require(report["first_mismatch_slot"] is None, "exact block has mismatch")
    require(resident != before, "commit did not publish shadow state")
    require(resident["state"]["position"] == 8208, "commit did not advance 16")
    require(
        resident["state"]["processed_token_ids"]
        == before["state"]["processed_token_ids"] + list(inputs),
        "commit processed the wrong inputs",
    )
    require(
        resident["state"]["emitted_token_ids"] == list(COMPLETION[:17]),
        "commit emitted the wrong outputs",
    )
    require(
        report["state_after_sha256"] == report["shadow_state_sha256"],
        "committed state differs from shadow",
    )
    require(
        report["state_after_sha256"] != report["state_before_sha256"],
        "commit state hash did not move",
    )
    require(
        report["counters"]
        == {
            "attempted_blocks": 1,
            "attempted_input_tokens": 16,
            "verified_output_tokens": 16,
            "matching_prefix_tokens": 16,
            "committed_blocks": 1,
            "committed_input_tokens": 16,
            "emitted_tokens": 16,
            "discarded_blocks": 0,
            "discarded_input_tokens": 0,
            "mismatch_count": 0,
        },
        "commit counters",
    )
    return True


def test_mismatch(slot):
    resident = initial_resident()
    before = copy.deepcopy(resident)
    candidates = mismatch_candidates(slot)
    _, expected = oracle_window(COMPLETION)
    report = verify_transaction(resident, COMPLETION, candidates)
    require(report["decision"] == "discard", f"slot {slot} did not discard")
    require(report["first_mismatch_slot"] == slot, f"slot {slot} misreported")
    require(report["first_mismatch_expected"] == expected[slot], "wrong expected")
    require(report["first_mismatch_candidate"] == candidates[slot], "wrong candidate")
    require(
        report["counters"]
        == {
            "attempted_blocks": 1,
            "attempted_input_tokens": 16,
            "verified_output_tokens": 16,
            "matching_prefix_tokens": slot,
            "committed_blocks": 0,
            "committed_input_tokens": 0,
            "emitted_tokens": 0,
            "discarded_blocks": 1,
            "discarded_input_tokens": 16,
            "mismatch_count": 1,
        },
        "discard counters",
    )
    require(resident == before, f"slot {slot} mutated resident state")
    require(
        report["state_after_sha256"] == report["state_before_sha256"],
        f"slot {slot} changed the state hash",
    )
    require(
        report["shadow_state_sha256"] != report["state_before_sha256"],
        "test did not exercise scratch writes",
    )
    return True


def test_hash_surface():
    _, expected = oracle_window(COMPLETION)
    accepted = verify_transaction(initial_resident(), COMPLETION, expected)
    require(
        {key: accepted[key] for key in PINNED_HASHES} == PINNED_HASHES,
        "canonical transaction hashes moved",
    )
    rejected_hashes = set()
    for slot in MISMATCH_SLOTS:
        rejected = verify_transaction(
            initial_resident(), COMPLETION, mismatch_candidates(slot)
        )
        for key in ("oracle_sha256", "input_sha256", "expected_output_sha256"):
            require(rejected[key] == accepted[key], f"stable {key} changed")
        rejected_hashes.add(rejected["candidate_output_sha256"])
    require(
        accepted["candidate_output_sha256"] == accepted["expected_output_sha256"],
        "equal output sequences do not have equal hashes",
    )
    require(len(rejected_hashes) == len(MISMATCH_SLOTS), "candidate hashes collide")
    require(
        accepted["candidate_output_sha256"] not in rejected_hashes,
        "mismatch hash equals accepted hash",
    )
    return True


CASES = [
    ("completion_0_through_15_predicts_completion_1_through_16", test_indexing),
    ("all_accept_commits_one_whole_q16_block", test_all_accept),
    *[
        (
            f"mismatch_at_slot_{slot}_discards_without_state_mutation",
            lambda slot=slot: test_mismatch(slot),
        )
        for slot in MISMATCH_SLOTS
    ],
    ("digest_and_counter_surface_is_deterministic", test_hash_surface),
]


def contract_summary():
    _, expected = oracle_window(COMPLETION)
    accepted = verify_transaction(initial_resident(), COMPLETION, expected)
    rejected = {}
    for slot in MISMATCH_SLOTS:
        report = verify_transaction(
            initial_resident(), COMPLETION, mismatch_candidates(slot)
        )
        rejected[str(slot)] = {
            "candidate_output_sha256": report["candidate_output_sha256"],
            "state_unchanged": (
                report["state_before_sha256"] == report["state_after_sha256"]
            ),
            "counters": report["counters"],
        }
    return {
        "contract": CONTRACT,
        "q": Q,
        "input_completion_indices": [0, 15],
        "output_completion_indices": [1, 16],
        "oracle_sha256": accepted["oracle_sha256"],
        "input_sha256": accepted["input_sha256"],
        "expected_output_sha256": accepted["expected_output_sha256"],
        "all_accept": {
            "decision": accepted["decision"],
            "state_before_sha256": accepted["state_before_sha256"],
            "state_after_sha256": accepted["state_after_sha256"],
            "counters": accepted["counters"],
        },
        "forced_mismatch": rejected,
    }


def main():
    failures = []
    for name, fn in CASES:
        try:
            ok = bool(fn())
        except Exception as exc:  # Surface all errors as contract failures.
            ok = False
            name = f"{name} (raised {type(exc).__name__}: {exc})"
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} FAILED")
        raise SystemExit(1)
    print(f"\nall {len(CASES)} q16 transaction contract cases passed")
    print(json.dumps(contract_summary(), separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
