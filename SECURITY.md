# Security policy

## Supported versions

Security fixes are provided for the latest v1.x release. Older snapshots are
unsupported after a newer patched release is available.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for this repository. Do not open a
public issue for a suspected vulnerability that could expose hosts, model
artifacts, prompts, service users, or credentials. Include the affected
version/commit, Windows version, reproduction steps, and impact.

## Deployment boundary

The resident HTTP server is an inference transport, not an internet-facing TLS
gateway. Keep it on loopback or behind an authenticated reverse proxy. The CLI
rejects unauthenticated non-loopback binds unless the operator explicitly
overrides the guard. Use `--api-key` for bearer authentication, protect the
state/log files with appropriate Windows ACLs, and avoid logging sensitive
prompts at surrounding infrastructure layers.

The queue bounds protect memory and produce explicit overload responses, but
they are not a tenant quota or denial-of-service defense. Apply network-level
rate limits for untrusted clients. Model outputs are untrusted text and tool
arguments; validate them before executing any tool.

Never commit credentials, private keys, private deployment paths, model
weights, or prompt/output artifacts containing private data. Treat any secret
that entered Git history as compromised and rotate it.
