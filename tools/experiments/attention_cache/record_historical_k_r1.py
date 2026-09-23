"""Publish compact, source-bound evidence for the full256k layer-19 fault."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "build/recovery-20260910"
PUBLIC = ROOT.parent / "AIMA-public-r1191-candidate"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def main() -> None:
    native_path = B / "q1-cache-capture-prefix256k-long-prefix262144-suffix1024-out512-mode1-r1/run-record.json"
    comparison_path = B / "q1-layer19-logical-comparison-20260923-r1/collected/result/comparison.json"
    replay_dir = B / "runtime-attention-images-windows-r2"
    replay_path = replay_dir / "outputs/result.json"
    run_path = replay_dir / "run/run-record.json"
    plan_path = replay_dir / "manifest.json"
    native, comparison, replay, run, plan = map(read, (native_path, comparison_path, replay_path, run_path, plan_path))
    comparisons = {item["name"]: item for item in comparison["comparisons"]}
    assert native["host"].lower() == "baiying" and native["exit_code"] == 6
    assert native["host_checks_pass"] and not native["after_processes"]
    assert comparison["complete"] and comparison["all_inputs_unchanged"]
    assert comparison["provenance"]["native_run_sha256"] == digest(native_path)
    assert comparison["provenance"]["source_commit"] == replay["execution_checkout_commit"]
    assert replay["host"] == "baiying" and replay["manifest_sha256"] == digest(plan_path)
    assert replay["all_components_match"] and replay["inputs_and_tables_unchanged"]
    assert replay["sources_unchanged"] and replay["guards_pass"] and replay["cleanup_pass"]
    assert run["host_checks_pass"] and not run["after_processes"] and run["exit_code"] == 0
    assert len(replay["qk_records"]) == 16 and len(replay["pv_records"]) == 48
    assert all(item["scores"]["bit_exact"] and item["padding_negative_infinity"] for item in replay["qk_records"])
    assert all(item["score_input_unchanged"] and item["guards_pass"] and
               all(value["bit_exact"] for value in item["comparisons"].values())
               for item in replay["pv_records"])
    key = comparisons["cache-k"]
    assert key["bit_mismatches"] == 1 and key["first_32_differences"] == [{
        "index": 134800895, "coordinates": [263282, 1, 255],
        "actual_bits": 49013, "reference_bits": 49012}]
    assert comparisons["cache-v"]["bit_mismatches"] == 0
    assert comparisons["scores"]["bit_mismatches"] == 8
    assert comparisons["context"]["bit_mismatches"] == 2044
    for name in ("query", "current-k", "current-v"):
        assert comparisons[name]["bit_mismatches"] == 0
    history = comparison["provenance"]["generated_history"]
    assert history["position"] == 263291 and history["history_matches"]
    assert history["first_wrong_output_index"] == 157
    kinds = Counter(item["producer"] for item in replay["qk_records"])
    pv_inputs = Counter((item["pv_image"], item["score_producer"]) for item in replay["pv_records"])
    assert kinds == {"observed": 8, "qualified_probe": 8}
    assert set(pv_inputs.values()) == {8} and len(pv_inputs) == 6
    prior = ROOT / "benchmarks/correctness/layer19-runtime-attention-images-20260923.json"
    report = {
        "schema": 1,
        "classification": "full256k_historical_key_single_bit_and_exact_runtime_attention_replay",
        "controller_host": "JiaweiMnideMini.lan",
        "runtime_host": "baiying",
        "reference_host": comparison["provenance"]["reference_host"],
        "source_commit": replay["execution_checkout_commit"],
        "model": comparison["provenance"]["model"],
        "native_command_file": native["command_file"],
        "native_output": {
            "first_wrong_output_index": history["first_wrong_output_index"],
            "native_token": history["actual_512_output_ids"][157],
            "gb10_token": history["original_512_output_ids"][157],
            "wrong_tokens": history["output_mismatches"],
            "native_exit_code": native["exit_code"],
            "host_checks_pass": native["host_checks_pass"],
        },
        "compared_position": history["position"],
        "logical_history": {
            "all_inputs_unchanged": comparison["all_inputs_unchanged"],
            "compared_bytes": comparison["compared_bytes"],
            "cache_k_elements": key["elements"],
            "cache_k_bit_mismatches": key["bit_mismatches"],
            "first_cache_k_difference": key["first_32_differences"][0],
            "cache_v_elements": comparisons["cache-v"]["elements"],
            "cache_v_bit_mismatches": comparisons["cache-v"]["bit_mismatches"],
            "query_current_k_current_v_bit_mismatches": {
                name: comparisons[name]["bit_mismatches"] for name in ("query", "current-k", "current-v")},
            "score_bit_mismatches": comparisons["scores"]["bit_mismatches"],
            "segment_output_bit_mismatches": comparisons["segment-output"]["bit_mismatches"],
            "segment_max_bit_mismatches": comparisons["segment-max"]["bit_mismatches"],
            "segment_sum_bit_mismatches": comparisons["segment-sum"]["bit_mismatches"],
            "context_bit_mismatches": comparisons["context"]["bit_mismatches"],
        },
        "actual_windows_attention_replay": {
            "host": replay["host"], "command": replay["command"],
            "model_reference": replay["model_reference"],
            "original_model_loaded": replay["original_model_loaded"],
            "qk_cases": len(replay["qk_records"]),
            "qk_cases_per_image": dict(kinds),
            "pv_cases": len(replay["pv_records"]),
            "pv_cases_per_image_and_score_source": [
                {"image": image, "score_source": score, "cases": count}
                for (image, score), count in sorted(pv_inputs.items())],
            "all_qk_scores_bit_exact": True,
            "all_pv_context_and_segment_values_bit_exact": True,
            "all_components_match": replay["all_components_match"],
            "guards_pass": replay["guards_pass"],
            "cleanup_pass": replay["cleanup_pass"],
            "hip_dll": replay["hip_dll"],
        },
        "interpretation": (
            "For the captured original-model position, one historical K BF16 word is the first "
            "identified divergent attention operand. The unrotated channel 255 is upstream of QK. "
            "The actual DLL attention images reproduce gb10 QK and segmented PV bit-for-bit on "
            "gb10 operands, so those images do not explain this captured operand difference. "
            "The stage that produced the historical K word remains unresolved."
        ),
        "sources": {"native_run": source(native_path), "logical_comparison": source(comparison_path),
                    "replay_manifest": source(plan_path), "replay_result": source(replay_path),
                    "replay_run": source(run_path), "prior_binary_inspection": source(prior)},
        "inference_acceptance": False, "performance_acceptance": False, "release_qualified": False,
    }
    out = Path("benchmarks/correctness/layer19-historical-k-20260924.json")
    raw = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    for repo in (ROOT, PUBLIC):
        (repo / out).write_text(raw)
    print(json.dumps({"report_sha256": digest(ROOT / out), "bytes": len(raw.encode()),
                      "historical_key_mismatches": 1, "actual_runtime_replay_exact": True}))


if __name__ == "__main__":
    main()
