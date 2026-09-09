#!/usr/bin/env python3
"""Compare saved native GDN surfaces with fingerprinted GB10 captures.

This is a component diagnostic, never an inference/performance acceptance.
Only the Python standard library is required; comparison uses bounded blocks.
"""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(candidate: Path, reference: Path, dtype: str) -> dict:
    size = candidate.stat().st_size
    if size != reference.stat().st_size or size % (2 if dtype == "bf16" else 4):
        raise ValueError(f"invalid surface sizes: {candidate} / {reference}")
    total = mismatches = nonfinite = 0
    maximum = error2 = norm2 = 0.0
    first = None
    with candidate.open("rb") as left, reference.open("rb") as right:
        while payload := left.read(1 << 20):
            other = right.read(len(payload))
            if dtype == "bf16":
                bits_a, bits_b = array("H"), array("H")
                bits_a.frombytes(payload)
                bits_b.frombytes(other)
                if sys.byteorder != "little":
                    bits_a.byteswap()
                    bits_b.byteswap()
                a, b = array("f"), array("f")
                a.frombytes(array("I", (x << 16 for x in bits_a)).tobytes())
                b.frombytes(array("I", (x << 16 for x in bits_b)).tobytes())
            else:
                a, b = array("f"), array("f")
                a.frombytes(payload)
                b.frombytes(other)
                if sys.byteorder != "little":
                    a.byteswap()
                    b.byteswap()
            for x, y in zip(a, b):
                if not math.isfinite(x) or not math.isfinite(y):
                    nonfinite += 1
                elif x != y:
                    mismatches += 1
                    delta = x - y
                    maximum = max(maximum, abs(delta))
                    error2 += delta * delta
                    if first is None:
                        first = {"index": total, "candidate": x, "reference": y}
                if math.isfinite(y):
                    norm2 += y * y
                total += 1
    return dict(elements=total, mismatch_count=mismatches, nonfinite_count=nonfinite,
                maximum_absolute_error=maximum,
                relative_l2=math.sqrt(error2 / max(norm2, 1.0e-300)),
                first_mismatch=first, exact_match=mismatches == 0 and nonfinite == 0,
                candidate_sha256=fingerprint(candidate), reference_sha256=fingerprint(reference))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.reference_manifest.read_text())
    reference = manifest["results"][0]
    if reference["forward_method"] != "forward_native":
        raise ValueError("reference must identify the actual Triton/FLA worker route")
    results = {}
    for name, dtype in (("output-bf16", "bf16"), ("state-f32", "f32")):
        expected = reference["files"][name]
        source = args.reference_dir / (name + ".bin")
        if source.stat().st_size != expected["bytes"] or fingerprint(source) != expected["sha256"]:
            raise ValueError(f"GB10 fingerprint mismatch: {source}")
        native = Path(str(args.candidate_prefix) + "-" + name + ".bin")
        results[name] = compare(native, source, dtype)
    record = dict(schema_version=1, kind="gdn_component_diagnostic",
                  tokens=reference["tokens"], inference_acceptance=False,
                  all_surfaces_exact=all(x["exact_match"] for x in results.values()),
                  reference_manifest_sha256=fingerprint(args.reference_manifest), surfaces=results)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2, allow_nan=False))
    raise SystemExit(0 if record["all_surfaces_exact"] else 3)


if __name__ == "__main__":
    main()
