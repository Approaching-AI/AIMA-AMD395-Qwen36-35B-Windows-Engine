#!/usr/bin/env python3
"""Verify a pinned Windows runtime ZIP and both of its file inventories.

This caller/developer tool never extracts or executes the runtime. Inventory
integrity is separate from native, numerical, performance and release gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
import zipfile

PROJECT = "AIMA-AMD395-Qwen36-35B-Windows-Engine"
TARGET = "windows-x86_64-gfx1151"
RELEASE_MANIFEST = "FILE-SHA256SUMS.json"
RUNTIME_MANIFEST = "runtime-manifest.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
REQUIRED_RUNTIME = {
    "engine/qrt.exe", "product-cli/qrt-product.exe",
    "whole-provider/qrt_qwen36_whole_provider.dll", "runtime.env",
}


class InventoryError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def digest_stream(stream) -> tuple[int, str]:
    count, digest = 0, hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        count += len(chunk)
        digest.update(chunk)
    return count, digest.hexdigest()


def portable_path(value: object) -> str:
    require(isinstance(value, str) and bool(value), "empty or non-string inventory path")
    require(not any(ord(c) < 32 or c in '\\:<>"|?*' for c in value),
            f"non-portable inventory path: {value!r}")
    parts = value.split("/")
    require(all(part and part not in (".", "..") and not part.endswith((".", " "))
                and not RESERVED.match(part) for part in parts),
            f"non-portable inventory path: {value!r}")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def invalid_constant(value):
    raise InventoryError(f"non-finite JSON constant: {value}")


def read_manifest(archive, member) -> dict:
    require(member.file_size <= MAX_MANIFEST_BYTES, "inventory JSON exceeds size limit")
    value = json.loads(archive.read(member).decode("utf-8-sig"),
                       object_pairs_hook=unique_object, parse_constant=invalid_constant)
    require(isinstance(value, dict), "inventory JSON must be an object")
    return value


def records(value, label: str) -> dict[str, dict]:
    require(isinstance(value, list) and bool(value), f"{label} inventory is empty")
    result, folded = {}, set()
    for row in value:
        require(isinstance(row, dict), f"invalid {label} inventory record")
        path = portable_path(row.get("path"))
        require(path.casefold() not in folded, f"duplicate Windows path in {label}: {path}")
        require(type(row.get("bytes")) is int and row["bytes"] >= 0,
                f"invalid byte count for {path}")
        require(isinstance(row.get("sha256"), str) and SHA256.fullmatch(row["sha256"]),
                f"invalid SHA256 for {path}")
        result[path] = {key: row[key] for key in ("path", "bytes", "sha256")}
        folded.add(path.casefold())
    return result


def checksum_sidecar(path: Path, archive: Path) -> str:
    require(path.stat().st_size <= 4096, "checksum sidecar exceeds size limit")
    match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *]([^\r\n]+)\r?\n?",
                         path.read_text(encoding="ascii"))
    require(match is not None and match[2] == archive.name,
            "checksum sidecar must name this exact archive")
    return match[1].lower()


def verify_archive(path: Path, expected_sha256: str, *,
                   expected_runtime_manifest_sha256: str | None = None,
                   maximum_bytes: int = 8 * 1024**3) -> dict:
    require(isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256),
            "an explicit lowercase archive SHA256 is required")
    require(type(maximum_bytes) is int and maximum_bytes > 0, "invalid archive size limit")
    if expected_runtime_manifest_sha256 is not None:
        require(SHA256.fullmatch(expected_runtime_manifest_sha256) is not None,
                "invalid expected runtime manifest SHA256")
    require(path.stat().st_size <= maximum_bytes, "archive exceeds size limit")
    with path.open("rb") as stream:
        archive_bytes, archive_digest = digest_stream(stream)
        require(archive_digest == expected_sha256, "archive SHA256 mismatch")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            files, directories, folded, total = {}, set(), set(), 0
            for member in archive.infolist():
                require(member.orig_filename == member.filename, "ZIP entry contains a NUL filename")
                raw = member.filename[:-1] if member.is_dir() else member.filename
                name = portable_path(raw)
                require(name.casefold() not in folded, f"duplicate Windows ZIP path: {name}")
                folded.add(name.casefold())
                kind = stat.S_IFMT(member.external_attr >> 16)
                require(kind in (0, stat.S_IFDIR if member.is_dir() else stat.S_IFREG),
                        f"ZIP entry is not a regular file/directory: {name}")
                require(not member.flag_bits & 1, f"encrypted ZIP entry: {name}")
                require(member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                        f"unsupported portable ZIP compression: {name}")
                total += member.file_size
                require(total <= maximum_bytes, "uncompressed archive exceeds size limit")
                if member.is_dir():
                    require(member.file_size == 0, f"nonempty directory entry: {name}")
                    directories.add(name)
                else:
                    files[name] = member
            roots = [name[:-len(RELEASE_MANIFEST) - 1] for name in files
                     if name.endswith("/" + RELEASE_MANIFEST)]
            require(len(roots) == 1 and "/" not in roots[0],
                    "archive must contain one top-level release manifest")
            root = roots[0]
            require(all(name.startswith(root + "/") for name in files),
                    "file lies outside the release directory")
            require(all(name == root or name.startswith(root + "/") for name in directories),
                    "directory lies outside the release directory")
            # Reject a file being used as another entry's parent, including
            # case-only collisions that extract differently on Windows.
            file_names = {name.casefold() for name in files}
            for name in files.keys() | directories:
                parts = name.split("/")
                require(all("/".join(parts[:i]).casefold() not in file_names
                            for i in range(1, len(parts))), f"file/directory collision: {name}")
            relative = {name[len(root) + 1:]: member for name, member in files.items()}
            release = read_manifest(archive, relative[RELEASE_MANIFEST])
            require(release.get("schema_version") == 1 and release.get("project") == PROJECT
                    and release.get("target") == TARGET, "release identity/target mismatch")
            require(root == PROJECT + "-v" + str(release.get("version")),
                    "release directory/version mismatch")
            source = release.get("source_commit")
            require(isinstance(source, str) and COMMIT.fullmatch(source), "invalid release source commit")
            inventory = records(release.get("files"), "release")
            require(set(relative) == set(inventory) | {RELEASE_MANIFEST},
                    "ZIP files differ from complete release inventory")
            require(RELEASE_MANIFEST not in inventory and RUNTIME_MANIFEST in inventory,
                    "invalid release/runtime manifest ownership")
            actual = {}
            for name, member in relative.items():
                with archive.open(member) as content:
                    count, digest = digest_stream(content)
                require(count == member.file_size, f"ZIP byte count mismatch: {name}")
                actual[name] = {"path": name, "bytes": count, "sha256": digest}
                if name in inventory:
                    require(actual[name] == inventory[name], f"release hash/size mismatch: {name}")
            runtime = read_manifest(archive, relative[RUNTIME_MANIFEST])
            require(runtime.get("schema_version") == 1 and runtime.get("dirty_tree") is False
                    and runtime.get("offload_arch") == "gfx1151"
                    and runtime.get("repo_commit") == source, "runtime source/target mismatch")
            runtime_inventory = records(runtime.get("artifacts"), "runtime")
            require(REQUIRED_RUNTIME <= runtime_inventory.keys(), "required runtime artifact is absent")
            require(any(name.startswith("ck-fmha/") and name.count("/") == 1
                        and name.endswith(".dll") for name in runtime_inventory), "CK DLL is absent")
            for name, record in runtime_inventory.items():
                require(name not in (RELEASE_MANIFEST, RUNTIME_MANIFEST)
                        and actual.get(name) == record, f"runtime hash/size mismatch: {name}")
            components = runtime.get("source_component_builds", {})
            require(isinstance(components, dict), "invalid component inventory")
            commits = runtime.get("component_commits", {})
            require(isinstance(commits, dict), "invalid component commit inventory")
            for name, component in components.items():
                require(isinstance(component, dict), f"invalid component: {name}")
                commit = component.get("commit")
                require(isinstance(commit, str) and COMMIT.fullmatch(commit)
                        and commits.get(name) == commit, f"component commit mismatch: {name}")
                packaged = portable_path(component.get("package_path"))
                artifact = component.get("artifact")
                require(isinstance(artifact, dict) and packaged in runtime_inventory
                        and artifact.get("sha256") == actual[packaged]["sha256"],
                        f"component artifact mismatch: {name}")
                if "bytes" in artifact:
                    require(type(artifact["bytes"]) is int and artifact["bytes"] == actual[packaged]["bytes"],
                            f"component byte count mismatch: {name}")
            runtime_digest = actual[RUNTIME_MANIFEST]["sha256"]
            if expected_runtime_manifest_sha256 is not None:
                require(runtime_digest == expected_runtime_manifest_sha256,
                        "pinned runtime manifest SHA256 mismatch")
            result = dict(
                schema_version=1, kind="portable_archive_inventory", archive=path.name,
                archive_bytes=archive_bytes, archive_sha256=archive_digest, version=release["version"],
                source_commit=source, target=TARGET, release_files=len(inventory), archive_files=len(actual),
                runtime_artifacts=len(runtime_inventory), uncompressed_bytes=total,
                release_manifest_sha256=actual[RELEASE_MANIFEST]["sha256"],
                runtime_manifest_sha256=runtime_digest,
                source_component_builds=components, declared_component_artifacts_checked=len(components),
                archive_sha256_pass=True, crc_and_release_inventory_pass=True,
                runtime_inventory_pass=True, windows_paths_pass=True,
                pinned_runtime_manifest_checked=expected_runtime_manifest_sha256 is not None,
                runtime_executed=False, inference_acceptance=False,
                performance_acceptance=False, release_qualified=False,
            )
        stream.seek(0)
        require(digest_stream(stream) == (archive_bytes, archive_digest),
                "archive changed during verification")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sha256")
    group.add_argument("--checksum-file", type=Path)
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--maximum-bytes", type=int, default=8 * 1024**3)
    args = parser.parse_args()
    try:
        expected = args.sha256 or checksum_sidecar(args.checksum_file, args.archive)
        result = verify_archive(args.archive, expected,
                                expected_runtime_manifest_sha256=args.runtime_manifest_sha256,
                                maximum_bytes=args.maximum_bytes)
    except (OSError, ValueError, KeyError, UnicodeError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"archive verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
