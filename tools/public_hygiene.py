#!/usr/bin/env python3
"""High-signal checks for secrets and private deployment details."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str


_BYTE_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private-key",
        re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    ),
    (
        "github-token",
        re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    ),
    ("aws-access-key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "private-home-path",
        re.compile(rb"/(?:Users|home|data/home)/[A-Za-z0-9._-]+(?:/|\b)"),
    ),
    (
        "private-windows-home-path",
        re.compile(
            rb"\b[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"
            rb"[A-Za-z0-9._ -]+(?:[\\/]|\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "private-ipv4",
        re.compile(
            rb"\b(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}"
            rb"|192\.168\.[0-9]{1,3}\.[0-9]{1,3}"
            rb"|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})\b"
        ),
    ),
    (
        "credential-in-url",
        re.compile(rb"\b[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@", re.IGNORECASE),
    ),
    ("sshpass-password", re.compile(rb"\bsshpass\s+-p\s+\S+", re.IGNORECASE)),
)

_LITERAL_CREDENTIAL = re.compile(
    rb"\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*"
    rb"(?P<quote>['\"])(?P<value>[^'\"\r\n]+)(?P=quote)",
    re.IGNORECASE,
)
_SAFE_LITERAL_VALUES = {
    b"",
    b"changeme",
    b"change-me",
    b"example",
    b"placeholder",
    b"redacted",
    b"test-only-placeholder",
}


def scan_bytes(path: str, payload: bytes) -> list[Finding]:
    findings: list[Finding] = []
    for rule, pattern in _BYTE_RULES:
        for match in pattern.finditer(payload):
            findings.append(
                Finding(path, payload.count(b"\n", 0, match.start()) + 1, rule)
            )
    for match in _LITERAL_CREDENTIAL.finditer(payload):
        value = match.group("value").strip().lower()
        if value not in _SAFE_LITERAL_VALUES and not value.startswith((b"${", b"$", b"<")):
            findings.append(
                Finding(
                    path,
                    payload.count(b"\n", 0, match.start()) + 1,
                    "literal-credential",
                )
            )
    return findings


def candidate_paths(root: Path) -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return [root / item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]

    excluded = {".git", "__pycache__", "build", "dist", "output", "state"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in excluded for part in path.relative_to(root).parts)
    ]


def scan_public_tree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in candidate_paths(root):
        if not path.is_file():
            continue
        findings.extend(scan_bytes(path.relative_to(root).as_posix(), path.read_bytes()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan_public_tree(root)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.rule}")
    if findings:
        print(f"public_hygiene=fail finding_count={len(findings)}")
        return 1
    print("public_hygiene=pass finding_count=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
