# Security policy

## Supported versions

There is no supported runtime release yet. The current public revision is a
release-staging repository and contains no installable service or binary.

## Reporting a vulnerability

Use GitHub private vulnerability reporting for this repository. Do not open a
public issue for a suspected vulnerability that could expose hosts, model
artifacts, prompts, service users, or credentials. Include the affected commit
or release, Windows version, reproduction steps, and impact.

Never commit credentials, private keys, private host addresses, personal
deployment paths, model weights, or prompt/output artifacts containing private
data. The repository's hygiene check is a prevention layer, not a Git-history
sanitizer. Treat any credential that entered history as compromised and rotate
it before coordinating a history rewrite.

## Future deployment boundary

The planned resident HTTP server is an inference transport, not an
internet-facing gateway. A public runtime release must document authentication,
TLS termination, remote-bind behavior, shutdown controls, request timeouts,
prompt logging, output retention, and Windows service permissions before it is
declared supported.
