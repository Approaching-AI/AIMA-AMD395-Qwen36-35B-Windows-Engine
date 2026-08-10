# Windows packaging

The release archive contains project-built executables/providers, generated
kernel directories, the qualified `aot/gfx1151` base inventory, `runtime.env`,
a SHA256 manifest, and all project/upstream
notices. It excludes model weights, ROCm SDK/runtime files, drivers, compiler
toolchains, caches, logs, service state, build-only executables/import libraries,
and component provenance files containing machine-local build paths. The
included runtime manifest uses paths relative to the archive root.

Run `scripts/package-runtime.ps1` against a successful `build-runtime.ps1`
output. Packaging refuses incomplete manifests and existing output unless the
operator explicitly requests replacement.
