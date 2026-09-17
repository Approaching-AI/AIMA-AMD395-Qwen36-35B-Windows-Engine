"""Bounded native build, including an unchanged extracted convolution control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from linear_input_control import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    control = args.output.parent / "linear_input_control.generated.h"
    source = root / "native/providers/whole_provider.cpp"
    with control.open("x", encoding="utf-8") as output:
        output.write(extract(source.read_text(encoding="utf-8")))
    command = [args.compiler, "-std=c++17", "-O3", "-DQRT_SM121_DPP_REDUCTION=1", "-DQRT_SM121_COMPACT_NORMALIZE=1",
               "--offload-arch=gfx1151", "-I" + str(args.output.parent),
               str(root / "tests/native/linear_input_preparation_selftest.cpp"),
               str(root / "native/providers/gdn/blackwell_l2norm.cpp"), "-o", str(args.output)]
    result = subprocess.run(command, timeout=150, check=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(dict(kind="linear_input_preparation_build", command=command, returncode=result.returncode,
                         whole_source_sha256=sha(source), extracted_control_sha256=sha(control))), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
