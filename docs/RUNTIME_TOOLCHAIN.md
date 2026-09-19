# Observed Windows runtime

At 2026-09-19T18:39:59Z, a read-only inspection of the original full256k
process on baiying found the following. PID12488 and its exact launch/start
times match that active run. This observation changes no environment and
qualifies no inference or performance result.

| Surface | Actual path/version |
| --- | --- |
| Compiler and math SDK | `C:\Program Files\AMD\ROCm\7.1`; HIP header7.1.51803, clang21.0.0git |
| Loaded HIP runtime | `C:\WINDOWS\SYSTEM32\amdhip64_7.dll`; PE version10.0.3679.0, description HIP7.2 Runtime |
| Loaded code-object manager | System32 `amd_comgr_3.dll`, version3.0.0.0 |
| Loaded rocBLAS/hipBLASLt | SDK7.1 `bin\rocblas.dll` and `bin\libhipblaslt.dll` |
| AMD display driver | `32.0.31036.15`, dated2026-08-12 |

The compiler directory alone does not identify the mapped HIP runtime.
AMD's [HIP SDK7.2 notes](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/about/releasenotes.html)
describe a Strix Halo slowdown affecting HIP6 applications with26.10-branch
drivers and explicitly exclude HIP7. That issue does not apply to this observed
process. The driver number matches the Windows Store version listed for
RDNA3/newer in [PRO26.Q3](https://www.amd.com/en/resources/support-articles/release-notes/RN-PRO-WIN-26-Q3.html);
the observation does not identify the complete installed driver package.

AMD documents a newer [Windows gfx1151 tarball](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html).
Its [compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html)
lists Windows11 25H2 and specific Adrenalin/CDE driver versions. This does not
establish compatibility with the observed PRO driver or a benefit for the
engine. That process inspection did not install or test a new SDK.

The [subsequent static archive review](../benchmarks/correctness/rocm10-windows-archive-static-review-20260920.json)
records the completed 1,684,865,477-byte Windows gfx1151 download, SHA256
`1293927b06b3b8d4bd7e0265823fb998bc9e0d83c68f33dcfa5d32663b30ce38`.
Its 9,995 entries contain 8,715,298,580 logical regular-file bytes. Actual
Clang and amdclang PE resources, `Version.inc`, CMake metadata and compiler
resource directories identify **23.0.0git**; HIP metadata identifies
**7.15.26333**. This Windows artifact must not be described as LLVM24 solely
from the general ROCm10 release notes. Direct imports include the Windows and
Visual C++ runtimes. These static imports are not a complete packaging test.

The archive remains on the controller. No SDK was extracted to disk, installed
or executed on Windows. An isolated compiler invocation, ABI/driver checks and
correctness-attached model measurements remain necessary before selecting this
toolchain. The first PE reader missed raw archive path spellings; the corrected
reader uses the inventory's normalized names and retains the failed attempt.

[Structured observation](../benchmarks/correctness/resident-runtime-toolchain-review-20260920.json)
retains the raw DLL descriptions, registry fields, image sizes, observer/source
hashes and matching wholeb3/CK370/FLA1d/CLI6d source identities. Image sizes
are not file hashes. The failed first encoded-command transport produced no
snapshot; the same observer succeeded using a command file in revision2.
