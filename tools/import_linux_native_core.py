#!/usr/bin/env python3
"""Import the pinned public Linux compute core without changing its bytes."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import tarfile

REVISION = "ec9934446911fdf376da8eebcd83e7b137efbb7c"
REPOSITORY = "https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine"
ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md",
    "third_party/licenses",
    "native/include", "native/src", "native/generated", "native/aot",
    "benchmarks/shape-lab/native/src/torch_owned_safetensors_loader.hip.cpp",
    "scripts/generate-native-aot-registry.py",
    "scripts/generate-native-decode-registry.py",
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--verify", action="store_true",
                        help="Compare the imported bytes and inventory with the pinned Git tree")
    args = parser.parse_args()
    actual = subprocess.check_output(
        ["git", "rev-parse", REVISION + "^{commit}"],
        cwd=args.source_repo, text=True, timeout=15).strip()
    if actual != REVISION:
        raise SystemExit("Pinned upstream revision is unavailable")
    destination = ROOT / "third_party/aima_linux"
    if destination.exists() and not args.verify:
        raise SystemExit("Import already exists; verify its inventory instead of replacing it")
    payload = subprocess.check_output(
        ["git", "archive", "--format=tar", REVISION, *PATHS],
        cwd=args.source_repo, timeout=60)
    files = []
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        members = archive.getmembers()
        if sum(m.size for m in members) > 64 * 1024 * 1024:
            raise SystemExit("Unexpected upstream source extent")
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise SystemExit("Unexpected upstream archive member")
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                data = stream.read()
                files.append(dict(path=member.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if not args.verify:
            destination.mkdir(parents=True)
            archive.extractall(destination, filter="data")
    for entry in files:
        path = destination / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    manifest = dict(schema=1, repository=REPOSITORY, revision=REVISION,
        release="v1.5.1-native-vl.10 declared native source",
        license="Apache-2.0 with bundled component notices",
        upstream_bytes_unchanged=True, files=sorted(files, key=lambda x: x["path"]),
        runtime_enabled=False, windows_build_qualified=False,
        model_correctness_qualified=False, performance_qualified=False)
    if args.verify:
        if json.loads((destination / "UPSTREAM.json").read_text()) != manifest:
            raise SystemExit("Upstream inventory differs from the pinned Git tree")
        expected = {entry["path"] for entry in files} | {"UPSTREAM.json"}
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
        if actual != expected:
            raise SystemExit("Unexpected or missing files in the imported tree")
    else:
        (destination / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(revision=REVISION, verified=args.verify, files=len(files),
                         bytes=sum(x["bytes"] for x in files))))

if __name__ == "__main__":
    main()
