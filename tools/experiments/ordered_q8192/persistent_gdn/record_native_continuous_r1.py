"""Summarize actual AMD GDN controls without treating components as inference."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
B = ROOT / "build/recovery-20260910"
PUBLIC = ROOT.parent / "AIMA-public-r1191-candidate"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def collect(name):
    directory = B / name
    manifest_path = directory / "manifest.json"
    run_path = directory / "run/run-record.json"
    result_path = directory / "outputs/result.json"
    manifest, run, result = map(read, (manifest_path, run_path, result_path))
    assert result["host"].lower() == "baiying"
    assert run["host_checks_pass"] and all(run["host_checks"].values()) and not run["after_processes"]
    assert run["exit_code"] == 6 and run["reason"] == "completed"
    assert result["manifest_sha256"] == sha(manifest_path) == run["spec"]["source_manifest_sha256"]
    assert result["guards_pass"] and result["cleanup_pass"] and result["exp2_table_unchanged"]
    assert all(item["inputs_unchanged"] and item["guards_pass"] for item in result["cases"])
    cases = []
    for item in result["cases"]:
        checkpoints = item.get("checkpoints", [])
        cases.append({
            "name": item["name"], "tokens": item["tokens"],
            "all_values_match": item["all_values_match"],
            "failed_surfaces": [record["surface"] for record in item["records"] if not record["bit_exact"]],
            "checkpoint_count": len(checkpoints),
            "first_wrong_checkpoint": next((r["chunk"] for r in checkpoints if not r["bit_exact"]), None),
            "first_wrong_checkpoint_actual_sha256": next((r["incoming_bf16_sha256"] for r in checkpoints if not r["bit_exact"]), None),
            "first_wrong_checkpoint_expected_sha256": next((r["expected_bf16_sha256"] for r in checkpoints if not r["bit_exact"]), None),
        })
    return {
        "manifest_sha256": sha(manifest_path), "result_sha256": sha(result_path),
        "run_sha256": sha(run_path), "source_commit": result["source_commit"],
        "execution_checkout_commit": result["execution_checkout_commit"],
        "command": result["command"], "model_reference": result["source_model_reference"],
        "actual_model_loaded": result["actual_model_loaded"],
        "cases": cases,
    }


def main():
    layout = collect("native-gdn-layout-controls-windows-r2")
    compact = collect("native-gdn-compact-windows-r2")
    persistent = collect("native-gdn-persistent-windows-r2")
    for family in (layout, compact, persistent):
        assert all(x["all_values_match"] for x in family["cases"] if "unfused_u64" in x["name"] or
                   x["name"].startswith("first64-u64") or x["name"] == "continuous7169-u64")
    assert len(layout["cases"]) == 24 and len(compact["cases"]) == len(persistent["cases"]) == 15
    for family in (compact, persistent):
        bad = [x for x in family["cases"] if "continuous7169-persistent-capture" in x["name"]]
        assert len(bad) == 2 and all(x["first_wrong_checkpoint"] == 1 for x in bad)
    first = next(x for x in compact["cases"] if x["name"] == "continuous7169-persistent-capture-repeat0")
    second = next(x for x in persistent["cases"] if x["name"] == "continuous7169-persistent-capture-repeat0")
    assert first["first_wrong_checkpoint_actual_sha256"] == second["first_wrong_checkpoint_actual_sha256"]
    assert first["first_wrong_checkpoint_expected_sha256"] == second["first_wrong_checkpoint_expected_sha256"]
    layout_passes = Counter(x["name"].split("-")[1] if x["name"].startswith("first64") else
                            x["name"].split("-")[-1] for x in layout["cases"] if x["all_values_match"])
    report = {
        "schema": 1,
        "classification": "actual_amd_gdn_continuous_component_controls",
        "controller_host": "JiaweiMnideMini.lan", "runtime_host": "baiying",
        "model_reference": "D:/models/Qwen3.6-35B-A3B; original gb10 first64 and q7169 operands",
        "layout_cases": layout, "compact_persistent_cases": compact,
        "unrolled_persistent_cases": persistent,
        "summary": {
            "layout_control_pass_counts": dict(layout_passes),
            "compact_first64_passed": sum(x["all_values_match"] for x in compact["cases"] if x["tokens"] == 64),
            "compact_first64_total": sum(x["tokens"] == 64 for x in compact["cases"]),
            "unrolled_first64_passed": sum(x["all_values_match"] for x in persistent["cases"] if x["tokens"] == 64),
            "unrolled_first64_total": sum(x["tokens"] == 64 for x in persistent["cases"]),
            "both_persistent_capture_first_wrong_checkpoint": 1,
            "both_persistent_capture_checkpoint1_actual_sha256": first["first_wrong_checkpoint_actual_sha256"],
            "checkpoint1_expected_sha256": first["first_wrong_checkpoint_expected_sha256"],
            "original_u64_continuous_controls_pass": True,
            "persistent_continuous_pass": False,
            "selection": "Keep persistent options disabled; investigate the first carried-state boundary.",
        },
        "upstream_original_all_layer_control": {
            "path": "benchmarks/correctness/persistent-gdn-compact-all-layers-20260923.json",
            "sha256": sha(ROOT / "benchmarks/correctness/persistent-gdn-compact-all-layers-20260923.json"),
        },
        "component_only": True, "inference_acceptance": False,
        "performance_acceptance": False, "release_qualified": False,
    }
    output = Path("benchmarks/correctness/gdn-native-continuous-controls-20260924.json")
    raw = json.dumps(report, indent=2) + "\n"
    for repo in (ROOT, PUBLIC):
        (repo / output).write_text(raw)
    print(json.dumps({"sha256": sha(ROOT / output), "bytes": len(raw),
                      "first_wrong_checkpoint": 1, "persistent_continuous_pass": False}))


if __name__ == "__main__":
    main()
