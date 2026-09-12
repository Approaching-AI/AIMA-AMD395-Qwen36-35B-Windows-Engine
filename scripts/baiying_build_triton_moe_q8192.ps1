param(
    [Parameter(Mandatory = $false)][string]$WslDistribution = "Ubuntu-24.04",
    [Parameter(Mandatory = $false)][string]$TritonPython = "/opt/qwen36-vllm/bin/python",
    [Parameter(Mandatory = $false)][string]$HipccPath = "",
    [Parameter(Mandatory = $false)][string]$BuildDir = "",
    [Parameter(Mandatory = $false)][string]$OutDir = "",
    [Parameter(Mandatory = $false)][string]$OffloadArch = "gfx1151",
    [Parameter(Mandatory = $false)][int]$Repetitions = 5,
    [Parameter(Mandatory = $false)][int]$BlockM = 64,
    [Parameter(Mandatory = $false)][int]$BlockN = 64,
    [Parameter(Mandatory = $false)][int]$GateBlockN = 0,
    [Parameter(Mandatory = $false)][int]$DownBlockN = 0,
    [Parameter(Mandatory = $false)][int]$GateBlockK = 64,
    [Parameter(Mandatory = $false)][int]$DownBlockK = 64,
    [Parameter(Mandatory = $false)][int]$GroupM = 8,
    [Parameter(Mandatory = $false)][int]$NumWarps = 4,
    [Parameter(Mandatory = $false)][int]$NumStages = 1,
    [Parameter(Mandatory = $false)][int]$WavesPerEu = 0,
    [Parameter(Mandatory = $false)][int]$GateNumWarps = 0,
    [Parameter(Mandatory = $false)][int]$GateNumStages = 0,
    [Parameter(Mandatory = $false)][int]$GateWavesPerEu = -1,
    [Parameter(Mandatory = $false)][int]$DownNumWarps = 0,
    [Parameter(Mandatory = $false)][int]$DownNumStages = 0,
    [Parameter(Mandatory = $false)][int]$DownWavesPerEu = -1,
    [Parameter(Mandatory = $false)]
        [ValidateRange(0, 1)]
        [int]$VllmSiluBf16Intermediate = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaRouted = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$RoutedProjectionDebug = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$BatchedHawkeye = 0,
    [Parameter(Mandatory = $false)][ValidateSet(4, 8, 16)][int]$RoutedReplayLanes = 16,
    [Parameter(Mandatory = $false)][ValidateSet(0, 1)][int]$DppReduction = 0,
    [ValidateSet(0, 1)][int]$CompactNormalize = 0,
    [Parameter(Mandatory = $false)][ValidateSet(1, 4, 8)][int]$DotStagingGroups = 1,
    [ValidateSet(0, 1)][int]$CertifiedDotTiles = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$FullSharedHawkeye = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$ExactShared = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$ConditionalExactGate = 0,
    [Parameter(Mandatory = $false)]
        [ValidateRange(0, 1)]
        [int]$SortedConditionalExactGate = 0,
    [Parameter(Mandatory = $false)]
        [ValidateRange(0, 1)]
        [int]$RowMajorSortedConditionalExactGate = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(4, 8, 16, 32, 64, 128, 256, 512)]
        [int]$ConditionalExactGateRows = 64,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$ConditionalExactDown = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(4, 8, 16, 32, 64, 128, 256, 512)]
        [int]$ConditionalExactDownRows = 4,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaGate = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaDown = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeFusedRouteLayout = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaAdaptiveTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaBucketedTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaParallelBucketedTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaCompactSplitTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaTail32 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaPrune16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaWideN = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsB = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBNarrowN = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM96 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM64 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM80LoadWave = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(0, 192, 224, 256)]
        [int]$NativeWmmaLdsBM80LoadThreads = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(0, 128, 160, 192, 224, 256)]
        [int]$NativeWmmaLdsBM64LoadThreads = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(0, 128, 160, 192, 224, 256)]
        [int]$NativeWmmaLdsBM64GateLoadThreads = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(0, 128, 160, 192, 224, 256)]
        [int]$NativeWmmaLdsBM64DownLoadThreads = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBSplitGatePasses = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBSerialGateN32 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBParallelGateN64 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBParallelGateUpN32 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBSerialDownN32 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBSerialWideN64 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBCompactM96Waves = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBSkipInactiveAStores = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBFusedOverflow16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM64FusedOverflow32 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLosslessPalette = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLosslessRowPalette = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaWeightInt8 = 0,
    [Parameter(Mandatory = $false)]
        [ValidateSet(32, 64, 128)]
        [int]$NativeWmmaWeightInt8GroupValues = 128,
    [Parameter(Mandatory = $false)]
        [ValidateRange(0, 1)]
        [int]$NativeWmmaWeightInt8ScaleFp16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM64DirectM16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM64DualM16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBGroupedSoleM16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBM64QuadGateM16 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBAdaptiveM128 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBHybridM96M128 = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBAdaptiveDirectGrid = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBFullDirectTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBCompactSoleTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaLdsBParallelCompactSoleTail = 0,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$NativeWmmaTallM96 = 0,
    [Parameter(Mandatory = $false)][ValidateSet(32, 64, 128)][int]$NativeWmmaKStage = 64,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$TransposedRouter = 0,
    [Parameter(Mandatory = $false)][ValidateSet(16, 32, 64, 128, 256)][int]$RouterThreads = 256,
    [Parameter(Mandatory = $false)][ValidateSet(1, 2, 4, 8)][int]$RouterTokenTile = 1,
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$FullV3FusedCombine = 0,
    [Parameter(Mandatory = $false)][ValidateSet(1, 4)][int]$FusedCombineWidth = 1,
    [Parameter(Mandatory = $false)][ValidateRange(1, 256)][int]$FullV3EventSlots = 16,
    [Parameter(Mandatory = $false)]
        [string]$WslLlvmStrip = "/opt/rocm/llvm/bin/llvm-strip",
    [Parameter(Mandatory = $false)][string]$ReuseAotDir = "",
    [Parameter(Mandatory = $false)][ValidateRange(0, 1)][int]$RequireExpectedFullProviderHash = 0
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding -ArgumentList $false
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $repo "build\triton-moe-aot"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = $BuildDir
}
if ([string]::IsNullOrWhiteSpace($HipccPath)) {
    $HipccPath = (Get-Command hipcc.exe -ErrorAction Stop).Source
}
if ($Repetitions -le 0) {
    throw "Repetitions must be positive"
}
if ($GateBlockN -le 0) { $GateBlockN = $BlockN }
if ($DownBlockN -le 0) { $DownBlockN = $BlockN }
if ($GateNumWarps -le 0) { $GateNumWarps = $NumWarps }
if ($DownNumWarps -le 0) { $DownNumWarps = $NumWarps }
if ($GateNumStages -le 0) { $GateNumStages = $NumStages }
if ($DownNumStages -le 0) { $DownNumStages = $NumStages }
if ($GateWavesPerEu -lt 0) { $GateWavesPerEu = $WavesPerEu }
if ($DownWavesPerEu -lt 0) { $DownWavesPerEu = $WavesPerEu }
if ($NativeWmmaRouted -ne 0) {
    $NativeWmmaGate = 1
    $NativeWmmaDown = 1
}
if ($FullSharedHawkeye -ne 0 -and $BatchedHawkeye -eq 0) {
    throw "Full shared Hawkeye requires -BatchedHawkeye 1."
}
if ($FullSharedHawkeye -ne 0 -and $ExactShared -ne 0) {
    throw "Full shared Hawkeye and exact-shared AOT are mutually exclusive."
}
if (($ConditionalExactGate -ne 0 -or $ConditionalExactDown -ne 0) -and
        $BatchedHawkeye -eq 0) {
    throw "Conditional-exact gate/down requires -BatchedHawkeye 1."
}
if ($ConditionalExactGate -ne 0 -and $RoutedProjectionDebug -ne 0) {
    throw "Conditional-exact gate and routed projection debug are mutually exclusive."
}
if ($SortedConditionalExactGate -ne 0 -and $ConditionalExactGate -eq 0) {
    throw "Sorted conditional-exact gate requires -ConditionalExactGate 1."
}
if ($SortedConditionalExactGate -ne 0 -and $BlockM -ne 64) {
    throw "Sorted conditional-exact gate AOT requires -BlockM 64."
}
if ($RowMajorSortedConditionalExactGate -ne 0 -and
        $SortedConditionalExactGate -eq 0) {
    throw "Row-major sorted conditional-exact gate requires -SortedConditionalExactGate 1."
}
if ($NativeWmmaLdsBHybridM96M128 -ne 0) {
    $NativeWmmaLdsBAdaptiveM128 = 1
}
if ($NativeWmmaLdsBAdaptiveDirectGrid -ne 0) {
    $NativeWmmaLdsBAdaptiveM128 = 1
}
if ($NativeFusedRouteLayout -ne 0 -and
        ($NativeWmmaLdsB -eq 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0)) {
    throw "Fused route layout requires the plain native LDS-B route layout."
}
$nativeWmmaLdsBM80LoadThreadsResolved = $NativeWmmaLdsBM80LoadThreads
if ($NativeWmmaLdsBM80LoadWave -ne 0) {
    if ($nativeWmmaLdsBM80LoadThreadsResolved -ne 0) {
        throw "Use either the legacy M80 load-wave switch or an explicit M80 load-thread count, not both."
    }
    $nativeWmmaLdsBM80LoadThreadsResolved = 192
}
$nativeWmmaLdsBSerialM16Layout =
    $NativeWmmaLdsB -ne 0 -and
    $NativeWmmaLdsBSplitGatePasses -ne 0 -and
    $NativeWmmaLdsBSerialGateN32 -ne 0 -and
    $NativeWmmaLdsBSerialDownN32 -ne 0 -and
    $NativeWmmaLdsBM64 -eq 0 -and
    $NativeWmmaLdsBNarrowN -eq 0 -and
    $BlockM -ge 48 -and $BlockM -le 256 -and $BlockM % 16 -eq 0 -and
    (($BlockM -eq 96 -and $NativeWmmaLdsBM96 -ne 0) -or
     ($BlockM -ne 96 -and $NativeWmmaLdsBM96 -eq 0))
if ($NativeWmmaCompactSplitTail -ne 0 -and
        $NativeWmmaAdaptiveTail -eq 0 -and $NativeWmmaTail32 -eq 0 -and
        $NativeWmmaBucketedTail -eq 0) {
    $NativeWmmaAdaptiveTail = 1
}
if (($NativeWmmaGate -ne 0 -or $NativeWmmaDown -ne 0) -and
        $NativeWmmaWideN -eq 0 -and $NativeWmmaLdsB -eq 0 -and
        $NativeWmmaTallM96 -eq 0 -and
        $BlockM -notin @(32, 64)) {
    throw "Direct native WMMA selected-MoE kernels require -BlockM 32 or 64."
}
if ($NativeWmmaWideN -ne 0 -and
        ($BlockM -lt 64 -or $BlockM -gt 128 -or $BlockM % 16 -ne 0)) {
    throw "Native-WMMA wide-N requires BlockM 64..128 in increments of 16."
}
if ($NativeWmmaLdsB -ne 0 -and $NativeWmmaLdsBM96 -eq 0 -and
        $NativeWmmaLdsBM64 -eq 0 -and
        -not $nativeWmmaLdsBSerialM16Layout -and
        $BlockM -ne 128) {
    throw "Native-WMMA LDS-B requires M128, M96, M64, or the serial-N32 M48..M256 layout."
}
if ($NativeWmmaLdsBNarrowN -ne 0 -and $NativeWmmaLdsB -eq 0) {
    throw "Native-WMMA LDS-B narrow-N requires -NativeWmmaLdsB 1."
}
if ($NativeWmmaLdsBM96 -ne 0 -and
        ($NativeWmmaLdsB -eq 0 -or $BlockM -ne 96)) {
    throw "Native-WMMA LDS-B M96 requires -NativeWmmaLdsB 1 and -BlockM 96."
}
if ($NativeWmmaLdsBM96 -ne 0 -and $NativeWmmaLdsBNarrowN -ne 0) {
    throw "Native-WMMA LDS-B M96 and narrow-N are mutually exclusive."
}
if ($NativeWmmaLdsBM64 -ne 0 -and
        ($NativeWmmaLdsB -eq 0 -or $BlockM -ne 64)) {
    throw "Native-WMMA LDS-B M64 requires -NativeWmmaLdsB 1 and -BlockM 64."
}
if ($NativeWmmaLdsBM64 -ne 0 -and
        ($NativeWmmaLdsBM96 -ne 0 -or $NativeWmmaLdsBNarrowN -ne 0)) {
    throw "Native-WMMA LDS-B M64, M96, and narrow-N are mutually exclusive."
}
if ($nativeWmmaLdsBM80LoadThreadsResolved -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 80 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaCompactSplitTail -ne 0)) {
    throw "M80 cooperative load threads require the plain serial-N32 M80 LDS-B layout."
}
if ($NativeWmmaLdsBM64LoadThreads -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         ($NativeWmmaCompactSplitTail -ne 0 -and
          $NativeWmmaLdsBCompactSoleTail -eq 0))) {
    throw "M64 cooperative load threads require the plain serial-N32 M64 LDS-B layout."
}
if (($NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
        $NativeWmmaLdsBM64DownLoadThreads -ne 0) -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         ($NativeWmmaCompactSplitTail -ne 0 -and
          $NativeWmmaLdsBCompactSoleTail -eq 0))) {
    throw "Split M64 load threads require the plain serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLdsBSplitGatePasses -ne 0 -and
        ($NativeWmmaLdsB -eq 0 -or
         ($BlockM -ne 128 -and -not $nativeWmmaLdsBSerialM16Layout) -or
         $NativeWmmaLdsBM64 -ne 0 -or $NativeWmmaLdsBNarrowN -ne 0 -or
         ($BlockM -eq 96 -and $NativeWmmaLdsBM96 -eq 0) -or
         ($BlockM -eq 96 -and $NativeWmmaLdsBSerialGateN32 -eq 0))) {
    throw "Native-WMMA LDS-B split gate passes require M128 or a serial-N32 M48..M256 layout."
}
if ($NativeWmmaLdsBSerialGateN32 -ne 0 -and
        $NativeWmmaLdsBSplitGatePasses -eq 0) {
    throw "Native-WMMA LDS-B serial gate N32 requires split gate passes."
}
if ($NativeWmmaLdsBParallelGateN64 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -gt 128 -or
         $NativeWmmaLdsBParallelGateUpN32 -ne 0)) {
    throw "Native-WMMA LDS-B parallel gate N64 requires a serial-N32 M64..M128 layout."
}
if ($NativeWmmaLdsBParallelGateUpN32 -ne 0 -and
        ($NativeWmmaLdsB -eq 0 -or $BlockM -ne 128 -or
         $NativeWmmaLdsBM96 -ne 0 -or $NativeWmmaLdsBM64 -ne 0 -or
         $NativeWmmaLdsBNarrowN -ne 0 -or
         $NativeWmmaLdsBSplitGatePasses -ne 0 -or
         $NativeWmmaLdsBSerialGateN32 -ne 0)) {
    throw "Native-WMMA LDS-B parallel gate/up N32 requires the unsplit M128 LDS-B layout."
}
if ($NativeWmmaLdsBSerialDownN32 -ne 0 -and
        -not $nativeWmmaLdsBSerialM16Layout) {
    throw "Native-WMMA LDS-B serial down N32 requires a serial-N32 M16 layout."
}
if ($NativeWmmaLdsBSerialWideN64 -ne 0 -and
        ((($BlockM -ne 64 -or $NativeWmmaLdsBM96 -ne 0) -and
          ($BlockM -ne 96 -or $NativeWmmaLdsBM96 -eq 0)) -or
         $NativeWmmaLdsBSerialGateN32 -eq 0 -or
         $NativeWmmaLdsBSerialDownN32 -eq 0)) {
    throw "Native-WMMA LDS-B serial wide N64 requires the M64 or M96 serial gate/down layout."
}
if ($NativeWmmaLdsBCompactM96Waves -ne 0 -and
        ($BlockM -ne 96 -or $NativeWmmaLdsB -eq 0 -or
         $NativeWmmaLdsBM96 -eq 0 -or
         $NativeWmmaLdsBSerialGateN32 -eq 0 -or
         $NativeWmmaLdsBSerialDownN32 -eq 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0)) {
    throw "Compact M96 waves require the non-wide M96 serial gate/down layout."
}
if ($NativeWmmaLdsBSkipInactiveAStores -ne 0 -and
        $NativeWmmaLdsB -eq 0) {
    throw "Skipping inactive A stores requires -NativeWmmaLdsB 1."
}
if ($NativeWmmaLdsBFusedOverflow16 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 80 -or
         $NativeWmmaCompactSplitTail -ne 0)) {
    throw "Fused overflow16 requires the serial-N32 LDS-B M80 layout without compact tails."
}
if ($NativeWmmaLdsBM64FusedOverflow32 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0 -or
         $NativeWmmaTail32 -ne 0)) {
    throw "M64 fused overflow32 requires the plain 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLosslessPalette -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64 -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBM64DirectM16 -ne 0 -or
         $NativeWmmaLdsBM64DualM16 -ne 0 -or
         $NativeWmmaLdsBM64QuadGateM16 -ne 0)) {
    throw "Lossless-palette WMMA requires the 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLosslessRowPalette -ne 0 -and
        $NativeWmmaLosslessPalette -eq 0) {
    throw "Row-palette WMMA requires -NativeWmmaLosslessPalette 1."
}
if ($NativeWmmaWeightInt8 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64 -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBM64DirectM16 -ne 0 -or
         $NativeWmmaLdsBM64DualM16 -ne 0 -or
         $NativeWmmaLdsBM64QuadGateM16 -ne 0 -or
         $NativeWmmaLosslessPalette -ne 0)) {
    throw "Weight-int8 WMMA requires the 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaWeightInt8ScaleFp16 -ne 0 -and
        $NativeWmmaWeightInt8 -eq 0) {
    throw "FP16 weight-int8 scales require -NativeWmmaWeightInt8 1."
}
if ($NativeWmmaLdsBM64DirectM16 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBM64FusedOverflow32 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0 -or
         $NativeWmmaTail32 -ne 0)) {
    throw "M64 direct-M16 requires the plain 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLdsBM64DualM16 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBM64DirectM16 -ne 0 -or
         $NativeWmmaLdsBM64FusedOverflow32 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0 -or
         $NativeWmmaTail32 -ne 0)) {
    throw "M64 dual-M16 requires the plain 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLdsBGroupedSoleM16 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBParallelGateUpN32 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBM64FusedOverflow32 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0 -or
         $NativeWmmaTail32 -ne 0 -or
         $NativeWmmaLdsBM64DirectM16 -ne 0 -or
         $NativeWmmaLdsBM64DualM16 -ne 0 -or
         $NativeWmmaLosslessPalette -ne 0)) {
    throw "Grouped sole-M16 requires the plain 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLdsBM64QuadGateM16 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaLdsBM64LoadThreads -ne 192 -or
         $NativeWmmaLdsBM64GateLoadThreads -ne 0 -or
         $NativeWmmaLdsBM64DownLoadThreads -ne 0 -or
         $NativeWmmaLdsBParallelGateN64 -ne 0 -or
         $NativeWmmaLdsBParallelGateUpN32 -ne 0 -or
         $NativeWmmaLdsBSerialWideN64 -ne 0 -or
         $NativeWmmaLdsBM64FusedOverflow32 -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBCompactM96Waves -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0 -or
         $NativeWmmaTail32 -ne 0 -or
         $NativeWmmaLdsBM64DirectM16 -ne 0 -or
         $NativeWmmaLdsBM64DualM16 -ne 0 -or
         $NativeWmmaLdsBGroupedSoleM16 -ne 0 -or
         $NativeWmmaLosslessPalette -ne 0)) {
    throw "M64 quad-gate M16 requires the plain 192-thread serial-N32 M64 LDS-B layout."
}
if ($NativeWmmaLdsBHybridM96M128 -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 96 -or
         $NativeWmmaLdsBM96 -eq 0 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0)) {
    throw "Hybrid M96/M128 requires the M96 serial-N32 LDS-B route layout without other tail variants."
}
if ($NativeWmmaLdsBAdaptiveDirectGrid -ne 0 -and
        ($NativeWmmaLdsBHybridM96M128 -ne 0 -or $BlockM -ne 80)) {
    throw "Adaptive direct-grid requires the exact-fragment M80 adaptive route."
}
if ($NativeWmmaLdsBAdaptiveM128 -ne 0 -and
        $NativeWmmaLdsBHybridM96M128 -eq 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 80 -or
         $NativeWmmaCompactSplitTail -ne 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0)) {
    throw "Adaptive M128 requires the serial-N32 LDS-B M80 route layout without other tail variants."
}
if ($NativeWmmaLdsBFullDirectTail -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 80 -or
         $NativeWmmaAdaptiveTail -eq 0 -or
         $NativeWmmaCompactSplitTail -eq 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0)) {
    throw "Full direct tail requires compact adaptive LDS-B M80 without other LDS-B tail variants."
}
if ($NativeWmmaLdsBCompactSoleTail -ne 0 -and
        (-not $nativeWmmaLdsBSerialM16Layout -or $BlockM -ne 64 -or
         $NativeWmmaAdaptiveTail -eq 0 -or
         $NativeWmmaCompactSplitTail -eq 0 -or
         $NativeWmmaLdsBFusedOverflow16 -ne 0 -or
         $NativeWmmaLdsBAdaptiveM128 -ne 0)) {
    throw "Compact sole tails require compact adaptive LDS-B M64 without other LDS-B tail variants."
}
if ($NativeWmmaLdsBParallelCompactSoleTail -ne 0 -and
        $NativeWmmaLdsBCompactSoleTail -eq 0) {
    throw "Parallel compact sole tails require compact sole-tail routing."
}
if ($NativeWmmaTallM96 -ne 0 -and $BlockM -ne 96) {
    throw "Native-WMMA tall-M96 requires -BlockM 96."
}
if ($NativeWmmaTallM96 -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Native-WMMA tall-M96 requires both gate and down kernels."
}
if ($NativeWmmaTallM96 -ne 0 -and
        ($NativeWmmaWideN -ne 0 -or $NativeWmmaLdsB -ne 0 -or
         $NativeWmmaPrune16 -ne 0 -or $NativeWmmaTail32 -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0)) {
    throw "Native-WMMA tall-M96 cannot be combined with other native tail/layout experiments."
}
if ($NativeWmmaLdsB -ne 0 -and $NativeWmmaKStage -notin @(32, 64)) {
    throw "Native-WMMA LDS-B requires -NativeWmmaKStage 32 or 64."
}
if ($NativeWmmaAdaptiveTail -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Adaptive native-WMMA tails require both gate and down kernels."
}
if ($NativeWmmaBucketedTail -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Bucketed native-WMMA tails require both gate and down kernels."
}
if ($NativeWmmaParallelBucketedTail -ne 0 -and
        ($NativeWmmaBucketedTail -eq 0 -or
         $NativeWmmaCompactSplitTail -eq 0)) {
    throw "Parallel bucketed tails require compact bucketed tail kernels."
}
if ($NativeWmmaCompactSplitTail -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Compact split-tail native-WMMA requires both gate and down kernels."
}
if ($NativeWmmaTail32 -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Native-WMMA tail32 requires both gate and down kernels."
}
if (($NativeWmmaTail32 + $NativeWmmaAdaptiveTail +
        $NativeWmmaBucketedTail) -gt 1) {
    throw "Native-WMMA split-tail variants are mutually exclusive."
}
if ($NativeWmmaPrune16 -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Native-WMMA prune16 requires both gate and down kernels."
}
if ($NativeWmmaPrune16 -ne 0 -and
        ($NativeWmmaTail32 -ne 0 -or $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0)) {
    throw "Native-WMMA prune16 cannot be combined with split tail kernels."
}
if ($NativeWmmaWideN -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Native-WMMA wide-N requires both gate and down kernels."
}
if ($NativeWmmaLdsB -ne 0 -and
        ($NativeWmmaGate -eq 0 -or $NativeWmmaDown -eq 0)) {
    throw "Native-WMMA LDS-B requires both gate and down kernels."
}
if ($NativeWmmaLdsB -ne 0 -and
        ($NativeWmmaWideN -ne 0 -or $NativeWmmaPrune16 -ne 0 -or
         $NativeWmmaTail32 -ne 0 -or $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0) -and
        -not ($nativeWmmaLdsBSerialM16Layout -and
              $NativeWmmaCompactSplitTail -ne 0 -and
              $NativeWmmaAdaptiveTail -ne 0 -and
              $NativeWmmaWideN -eq 0 -and
              $NativeWmmaPrune16 -eq 0 -and
              $NativeWmmaTail32 -eq 0 -and
              $NativeWmmaBucketedTail -eq 0)) {
    throw "Native-WMMA LDS-B cannot be combined with other native tail/layout experiments."
}
if ($NativeWmmaWideN -ne 0 -and
        ($NativeWmmaPrune16 -ne 0 -or $NativeWmmaTail32 -ne 0 -or
         $NativeWmmaAdaptiveTail -ne 0 -or
         $NativeWmmaBucketedTail -ne 0)) {
    throw "Native-WMMA wide-N cannot be combined with experimental tail kernels."
}
if ($TransposedRouter -eq 0 -and
        ($RouterThreads -ne 256 -or $RouterTokenTile -ne 1)) {
    throw "RouterThreads below 256 or token tiles above 1 require -TransposedRouter 1."
}
if ($FullV3FusedCombine -eq 0 -and $FusedCombineWidth -ne 1) {
    throw "FusedCombineWidth above 1 requires -FullV3FusedCombine 1."
}

New-Item -ItemType Directory -Force -Path $BuildDir, $OutDir | Out-Null

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "cannot convert non-drive Windows path to WSL: $resolved"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace('\', '/')
    return "/mnt/$drive/$tail"
}

$generator = Join-Path $repo (
    "native\generators\compile_q8192_triton_selected_moe.py"
)
$providerSource = Join-Path $repo (
    "native\providers\triton_moe\qrt_triton_moe_q8192_provider.cpp"
)
$smokeSource = Join-Path $repo (
    "native\providers\triton_moe\q8192_triton_selected_moe_smoke.cpp"
)
foreach ($required in @($generator, $providerSource, $smokeSource)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "required selected-MoE provider input not found: $required"
    }
}

$aotFiles = @(
    "q8192_selected_moe_route_count.hsaco",
    "q8192_selected_moe_route_prefix_by_program.hsaco",
    "q8192_selected_moe_route_padded_prefix.hsaco",
    "q8192_selected_moe_route_scatter.hsaco",
    "q8192_selected_moe_gate_up_silu.hsaco",
    "q8192_selected_moe_down.hsaco"
)
$aotReused = -not [string]::IsNullOrWhiteSpace($ReuseAotDir)
$resolvedReuseAotDir = $null
if ($aotReused) {
    $resolvedReuseAotDir = (Resolve-Path -LiteralPath $ReuseAotDir).Path
    $reuseMetadataCandidates = @(
        (Join-Path $resolvedReuseAotDir "metadata.json"),
        (Join-Path $resolvedReuseAotDir "q8192_triton_selected_moe_f32out.json")
    )
    $reuseMetadata = $reuseMetadataCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if ($null -eq $reuseMetadata) {
        throw "reused AOT directory has no selected-MoE metadata: $resolvedReuseAotDir"
    }
    foreach ($name in $aotFiles) {
        $sourceAot = Join-Path $resolvedReuseAotDir $name
        if (-not (Test-Path -LiteralPath $sourceAot -PathType Leaf)) {
            throw "reused AOT directory is missing $name"
        }
        $destinationAot = Join-Path $BuildDir $name
        if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
                [IO.Path]::GetFullPath($sourceAot),
                [IO.Path]::GetFullPath($destinationAot)
            )) {
            Copy-Item -LiteralPath $sourceAot -Destination $destinationAot -Force
        }
    }
    Copy-Item -LiteralPath $reuseMetadata `
        -Destination (Join-Path $BuildDir "metadata.json") -Force
} else {
    $wslGenerator = Convert-ToWslPath $generator
    $wslBuildDir = Convert-ToWslPath $BuildDir
    $wslMetadata = "$wslBuildDir/metadata.json"
    $generatorArgs = @(
        $wslGenerator,
        "--output-dir", $wslBuildDir,
        "--metadata", $wslMetadata,
        "--block-m", $BlockM,
        "--block-n", $BlockN,
        "--gate-block-n", $GateBlockN,
        "--down-block-n", $DownBlockN,
        "--gate-block-k", $GateBlockK,
        "--down-block-k", $DownBlockK,
        "--group-m", $GroupM,
        "--num-warps", $NumWarps,
        "--num-stages", $NumStages,
        "--waves-per-eu", $WavesPerEu,
        "--gate-num-warps", $GateNumWarps,
        "--gate-num-stages", $GateNumStages,
        "--gate-waves-per-eu", $GateWavesPerEu,
        "--down-num-warps", $DownNumWarps,
        "--down-num-stages", $DownNumStages,
        "--down-waves-per-eu", $DownWavesPerEu
    )
    if ($VllmSiluBf16Intermediate -ne 0) {
        $generatorArgs += "--vllm-silu-bf16-intermediate"
    }
    & wsl.exe -d $WslDistribution -- $TritonPython @generatorArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Triton AOT generation exited $LASTEXITCODE"
    }
}

$exactSharedAotSourceDir = $null
$exactSharedAotFiles = @(
    "q1024_triton_0626_exact_shared_gate_up_silu.hsaco",
    "q1024_triton_0626_exact_shared_gate_logit.hsaco",
    "q1024_triton_0626_exact_shared_down.hsaco",
    "q1024_triton_0626_exact_metadata.json"
)
if ($ExactShared -ne 0) {
    $exactSharedAotSourceDir = Join-Path $repo "native\aot\$OffloadArch"
    if (-not (Test-Path -LiteralPath $exactSharedAotSourceDir -PathType Container)) {
        throw "exact-shared AOT source directory not found: $exactSharedAotSourceDir"
    }
    foreach ($name in $exactSharedAotFiles) {
        $sourceAot = Join-Path $exactSharedAotSourceDir $name
        if (-not (Test-Path -LiteralPath $sourceAot -PathType Leaf)) {
            throw "exact-shared AOT source is missing $name"
        }
        Copy-Item -LiteralPath $sourceAot `
            -Destination (Join-Path $BuildDir $name) -Force
    }
}

$conditionalExactAotSourceDir = $null
$conditionalExactAotFiles = @()
if ($ConditionalExactGate -ne 0) {
    if ($RowMajorSortedConditionalExactGate -ne 0) {
        $conditionalExactAotFiles +=
            "q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows$ConditionalExactGateRows.hsaco"
    } elseif ($SortedConditionalExactGate -ne 0) {
        $conditionalExactAotFiles +=
            "q8192_triton_0626_sorted_conditional_exact_gate_rows$ConditionalExactGateRows.hsaco"
    } else {
        $conditionalExactAotFiles +=
            "q8192_triton_0626_conditional_exact_gate_rows$ConditionalExactGateRows.hsaco"
    }
    $conditionalExactAotFiles +=
        "q8192_triton_0626_zero_correction_gate_finalize.hsaco"
    $conditionalExactAotFiles +=
        "q8192_triton_0626_zero_correction_gate_finalize_retained_fused_f32_silu.hsaco"
}
if ($ConditionalExactDown -ne 0) {
    $conditionalExactAotFiles +=
        "q8192_triton_0626_conditional_exact_down_rows$ConditionalExactDownRows.hsaco"
}
if ($conditionalExactAotFiles.Count -ne 0) {
    $conditionalExactAotSourceDir = Join-Path $repo "native\aot\$OffloadArch"
    if (-not (Test-Path -LiteralPath $conditionalExactAotSourceDir -PathType Container)) {
        throw "conditional-exact AOT source directory not found: $conditionalExactAotSourceDir"
    }
    foreach ($name in $conditionalExactAotFiles) {
        $sourceAot = Join-Path $conditionalExactAotSourceDir $name
        if (-not (Test-Path -LiteralPath $sourceAot -PathType Leaf)) {
            throw "conditional-exact AOT source is missing $name"
        }
        Copy-Item -LiteralPath $sourceAot `
            -Destination (Join-Path $BuildDir $name) -Force
    }
}

$aotFilesToStrip = @($aotFiles)
if ($conditionalExactAotFiles.Count -ne 0) {
    $aotFilesToStrip += $conditionalExactAotFiles
}
$aotFilesToStrip = @($aotFilesToStrip | Sort-Object -Unique)
& wsl.exe -d $WslDistribution -- /usr/bin/test -x $WslLlvmStrip
if ($LASTEXITCODE -ne 0) {
    throw "ROCm llvm-strip is not executable in WSL: $WslLlvmStrip"
}
foreach ($name in $aotFilesToStrip) {
    $aotPath = Join-Path $BuildDir $name
    if (-not (Test-Path -LiteralPath $aotPath -PathType Leaf)) {
        throw "AOT debug-strip input is missing: $aotPath"
    }
    $wslAotPath = Convert-ToWslPath $aotPath
    & wsl.exe -d $WslDistribution -- $WslLlvmStrip --strip-debug $wslAotPath
    if ($LASTEXITCODE -ne 0) {
        throw "ROCm llvm-strip failed for $name with exit code $LASTEXITCODE"
    }
}

$baseMetadataPath = Join-Path $BuildDir "metadata.json"
$baseMetadata = Get-Content -Raw -LiteralPath $baseMetadataPath |
    ConvertFrom-Json
foreach ($kernel in @($baseMetadata.kernels)) {
    $kernelPath = Join-Path $BuildDir ([string]$kernel.file)
    if (-not (Test-Path -LiteralPath $kernelPath -PathType Leaf)) {
        throw "selected-MoE metadata references missing AOT: $kernelPath"
    }
    $kernel.bytes = (Get-Item -LiteralPath $kernelPath).Length
    $kernel.sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $kernelPath
    ).Hash.ToLowerInvariant()
}
$aotPostprocess = [ordered]@{
    debug_sections_stripped = $true
    tool = $WslLlvmStrip
    arguments = @("--strip-debug")
}
if ($null -eq $baseMetadata.PSObject.Properties["postprocess"]) {
    $baseMetadata | Add-Member -NotePropertyName "postprocess" `
        -NotePropertyValue $aotPostprocess
} else {
    $baseMetadata.postprocess = $aotPostprocess
}
[IO.File]::WriteAllText(
    $baseMetadataPath,
    ($baseMetadata | ConvertTo-Json -Depth 20) + "`n",
    $utf8
)

$aotFilesToScan = @($aotFilesToStrip)
if ($ExactShared -ne 0) {
    $aotFilesToScan += @(
        $exactSharedAotFiles | Where-Object { $_.EndsWith(".hsaco") }
    )
}
$privateHomePatterns = @(
    '/(?:Users|home|data/home)/[A-Za-z0-9._-]+(?:/|\b)',
    '\b[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]'
)
foreach ($name in @($aotFilesToScan | Sort-Object -Unique)) {
    $aotPath = Join-Path $BuildDir $name
    $aotAscii = [Text.Encoding]::ASCII.GetString(
        [IO.File]::ReadAllBytes($aotPath)
    )
    foreach ($pattern in $privateHomePatterns) {
        if ([regex]::IsMatch($aotAscii, $pattern)) {
            throw "AOT contains a private home path after debug stripping: $name"
        }
    }
}

$compiledMetadata = Get-Content -Raw -LiteralPath (Join-Path $BuildDir "metadata.json") | ConvertFrom-Json
$aotBlockM = [int]$compiledMetadata.shape.block_m
if ($aotBlockM -le 0) {
    throw "Triton metadata reported a non-positive block-M"
}
if ($aotBlockM -ne $BlockM -and
        $NativeWmmaWideN -eq 0 -and $NativeWmmaLdsB -eq 0 -and
        $NativeWmmaTallM96 -eq 0) {
    throw "provider/AOT block-M mismatch requires a native route layout"
}
$routeMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "route_count" }
$scatterMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "route_scatter" }
$gateMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "gate_up_silu" }
$downMetadata = $compiledMetadata.kernels | Where-Object { $_.name -eq "down" }
if ($null -eq $routeMetadata -or $null -eq $scatterMetadata -or
        $null -eq $gateMetadata -or $null -eq $downMetadata) {
    throw "Triton metadata is missing route, gate/up, or down kernel records"
}
if (@($routeMetadata.abi) -notcontains "logical_routes" -or
        @($scatterMetadata.abi) -notcontains "logical_routes") {
    throw "Triton route kernels are missing the dynamic logical-routes ABI"
}
$routeThreads = [int]$routeMetadata.threads
$routeKernelNames = @("route_count", "route_prefix_by_program", "route_padded_prefix", "route_scatter")
$routeKernels = @($compiledMetadata.kernels | Where-Object { $_.name -in $routeKernelNames })
if ($routeKernels.Count -ne $routeKernelNames.Count) {
    throw "Triton metadata is missing one or more route kernels"
}
foreach ($routeKernel in $routeKernels) {
    if ([int]$routeKernel.threads -ne $routeThreads) {
        throw "route kernels disagree on workgroup size"
    }
}
$gateThreads = [int]$gateMetadata.threads
$downThreads = [int]$downMetadata.threads
if ($routeThreads -le 0 -or $gateThreads -le 0 -or $downThreads -le 0) {
    throw "Triton metadata reported a non-positive workgroup size"
}
$variantDefines = @(
    "-DQRT_TRITON_MOE_BLOCK_M=$BlockM",
    "-DQRT_TRITON_MOE_GATE_BLOCK_N=$GateBlockN",
    "-DQRT_TRITON_MOE_DOWN_BLOCK_N=$DownBlockN",
    "-DQRT_TRITON_MOE_GROUP_M=$GroupM",
    "-DQRT_TRITON_MOE_ROUTE_THREADS=$routeThreads",
    "-DQRT_TRITON_MOE_GATE_THREADS=$gateThreads",
    "-DQRT_TRITON_MOE_DOWN_THREADS=$downThreads",
    "-DQRT_TRITON_MOE_GATE_SHARED_BYTES=$([int]$gateMetadata.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_DOWN_SHARED_BYTES=$([int]$downMetadata.dynamic_shared_bytes)",
    "-DQRT_TRITON_MOE_ROUTER_THREADS=$RouterThreads",
    "-DQRT_TRITON_MOE_ROUTER_TOKEN_TILE=$RouterTokenTile",
    "-DQRT_TRITON_MOE_FUSED_COMBINE_WIDTH=$FusedCombineWidth",
    "-DQRT_TRITON_MOE_FULL_V3_EVENT_SLOTS=$FullV3EventSlots",
    "-DQRT_MOE_ROUTED_REPLAY_LANES=$RoutedReplayLanes",
    "-DQRT_SM121_DPP_REDUCTION=$DppReduction",
    "-DQRT_SM121_COMPACT_NORMALIZE=$CompactNormalize",
    "-DQRT_SM121_DOT_STAGING_GROUPS=$DotStagingGroups",
    "-DQRT_SM121_CERTIFIED_DOT_TILES=$CertifiedDotTiles",
    "-DQRT_TRITON_MOE_NATIVE_WMMA_K_STAGE=$NativeWmmaKStage"
)
if ($NativeWmmaGate -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_GATE=1" }
if ($RoutedProjectionDebug -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_ROUTED_PROJECTION_DEBUG=1"
}
if ($BatchedHawkeye -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_BATCHED_HAWKEYE=1"
}
if ($FullSharedHawkeye -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_FULL_SHARED_HAWKEYE=1"
}
if ($ExactShared -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_Q1024_EXACT_SHARED=1"
}
if ($ConditionalExactGate -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_CONDITIONAL_EXACT_GATE=1"
    $variantDefines += "-DQRT_TRITON_MOE_CONDITIONAL_EXACT_GATE_ROWS=$ConditionalExactGateRows"
}
if ($SortedConditionalExactGate -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_SORTED_CONDITIONAL_EXACT_GATE=1"
}
if ($RowMajorSortedConditionalExactGate -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_ROW_MAJOR_SORTED_CONDITIONAL_EXACT_GATE=1"
}
if ($ConditionalExactDown -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_CONDITIONAL_EXACT_DOWN=1"
    $variantDefines += "-DQRT_TRITON_MOE_CONDITIONAL_EXACT_DOWN_ROWS=$ConditionalExactDownRows"
}
if ($NativeWmmaDown -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_DOWN=1" }
if ($NativeWmmaAdaptiveTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_ADAPTIVE_TAIL=1"
}
if ($NativeWmmaBucketedTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_BUCKETED_TAIL=1"
}
if ($NativeWmmaParallelBucketedTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_PARALLEL_BUCKETED_TAIL=1"
}
if ($NativeWmmaCompactSplitTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_COMPACT_SPLIT_TAIL=1"
}
if ($NativeWmmaTail32 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_TAIL32=1"
}
if ($NativeWmmaPrune16 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_PRUNE16=1"
}
if ($NativeWmmaWideN -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_WIDE_N=1"
}
if ($NativeWmmaLdsB -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B=1"
}
if ($NativeWmmaLdsBNarrowN -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_NARROW_N=1"
}
if ($NativeWmmaLdsBM96 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M96=1"
}
if ($NativeWmmaLdsBM64 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64=1"
}
if ($NativeWmmaLdsBM80LoadWave -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M80_LOAD_WAVE=1"
}
if ($nativeWmmaLdsBM80LoadThreadsResolved -ne 0) {
    $variantDefines += (
        "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M80_LOAD_THREADS=" +
        $nativeWmmaLdsBM80LoadThreadsResolved
    )
}
if ($NativeWmmaLdsBM64LoadThreads -ne 0) {
    $variantDefines += (
        "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_LOAD_THREADS=" +
        $NativeWmmaLdsBM64LoadThreads
    )
}
if ($NativeWmmaLdsBM64GateLoadThreads -ne 0) {
    $variantDefines += (
        "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_GATE_LOAD_THREADS=" +
        $NativeWmmaLdsBM64GateLoadThreads
    )
}
if ($NativeWmmaLdsBM64DownLoadThreads -ne 0) {
    $variantDefines += (
        "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_DOWN_LOAD_THREADS=" +
        $NativeWmmaLdsBM64DownLoadThreads
    )
}
if ($NativeFusedRouteLayout -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_FUSED_ROUTE_LAYOUT=1"
}
if ($NativeWmmaLdsBSplitGatePasses -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SPLIT_GATE_PASSES=1"
}
if ($NativeWmmaLdsBSerialGateN32 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_GATE_N32=1"
}
if ($NativeWmmaLdsBParallelGateN64 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_PARALLEL_GATE_N64=1"
}
if ($NativeWmmaLdsBParallelGateUpN32 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_PARALLEL_GATE_UP_N32=1"
}
if ($NativeWmmaLdsBSerialDownN32 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_DOWN_N32=1"
}
if ($NativeWmmaLdsBSerialWideN64 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_WIDE_N64=1"
}
if ($NativeWmmaLdsBCompactM96Waves -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_COMPACT_M96_WAVES=1"
}
if ($NativeWmmaLdsBSkipInactiveAStores -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SKIP_INACTIVE_A_STORES=1"
}
if ($NativeWmmaLdsBFusedOverflow16 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_FUSED_OVERFLOW16=1"
}
if ($NativeWmmaLdsBM64FusedOverflow32 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_FUSED_OVERFLOW32=1"
}
if ($NativeWmmaLosslessPalette -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_PALETTE=1"
}
if ($NativeWmmaLosslessRowPalette -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LOSSLESS_ROW_PALETTE=1"
}
if ($NativeWmmaWeightInt8 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8=1"
    $variantDefines += (
        "-DQRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_GROUP_VALUES=" +
        $NativeWmmaWeightInt8GroupValues
    )
    if ($NativeWmmaWeightInt8ScaleFp16 -ne 0) {
        $variantDefines +=
            "-DQRT_TRITON_MOE_NATIVE_WMMA_WEIGHT_INT8_FP16_SCALES=1"
    }
}
if ($NativeWmmaLdsBM64DirectM16 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_DIRECT_M16=1"
}
if ($NativeWmmaLdsBM64DualM16 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_DUAL_M16=1"
}
if ($NativeWmmaLdsBGroupedSoleM16 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_GROUPED_SOLE_M16=1"
}
if ($NativeWmmaLdsBM64QuadGateM16 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_QUAD_GATE_M16=1"
}
if ($NativeWmmaLdsBAdaptiveM128 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_ADAPTIVE_M128=1"
}
if ($NativeWmmaLdsBHybridM96M128 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_HYBRID_M96_M128=1"
}
if ($NativeWmmaLdsBAdaptiveDirectGrid -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_ADAPTIVE_DIRECT_GRID=1"
}
if ($NativeWmmaLdsBFullDirectTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_FULL_DIRECT_TAIL=1"
}
if ($NativeWmmaLdsBCompactSoleTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_COMPACT_SOLE_TAIL=1"
}
if ($NativeWmmaLdsBParallelCompactSoleTail -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_LDS_B_PARALLEL_COMPACT_SOLE_TAIL=1"
}
if ($NativeWmmaTallM96 -ne 0) {
    $variantDefines += "-DQRT_TRITON_MOE_NATIVE_WMMA_TALL_M96=1"
}
if ($TransposedRouter -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_TRANSPOSED_ROUTER=1" }
if ($FullV3FusedCombine -ne 0) { $variantDefines += "-DQRT_TRITON_MOE_FULL_V3_FUSED_COMBINE=1" }
$smokeVariantDefines = @(
    $variantDefines | ForEach-Object {
        if ($_ -like "-DQRT_TRITON_MOE_BLOCK_M=*") {
            "-DQRT_TRITON_MOE_BLOCK_M=$aotBlockM"
        } else {
            $_
        }
    }
)

$providerDll = Join-Path $BuildDir "qrt_triton_moe_q8192_provider.dll"
$smokeExe = Join-Path $BuildDir "q8192_triton_selected_moe_smoke.exe"
$rocmRoot = Split-Path -Parent (Split-Path -Parent $HipccPath)
$rocmInclude = Join-Path $rocmRoot "include"
$hipblasLtImport = Join-Path $rocmRoot "lib\libhipblaslt.dll.a"
if (-not (Test-Path -LiteralPath $hipblasLtImport -PathType Leaf)) {
    throw "libhipblaslt.dll.a not found at $hipblasLtImport"
}
$hipblasLtLink = Join-Path $BuildDir "hipblaslt.lib"
Copy-Item -LiteralPath $hipblasLtImport -Destination $hipblasLtLink -Force
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $variantDefines -I $rocmInclude -L $BuildDir -shared $providerSource -o $providerDll `
    -lhipblaslt
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the selected-MoE provider"
}
& $HipccPath -std=c++17 -O3 "--offload-arch=$OffloadArch" `
    $smokeVariantDefines $smokeSource -o $smokeExe
if ($LASTEXITCODE -ne 0) {
    throw "hipcc exited $LASTEXITCODE while building the selected-MoE smoke"
}

$smokeOutput = & $smokeExe $BuildDir $Repetitions $providerDll
$smokeExitCode = $LASTEXITCODE
$smokeText = $smokeOutput -join "`n"
if ($smokeExitCode -ne 0) {
    throw "selected-MoE full-shape smoke exited $smokeExitCode"
}
$routerDebugPass =
    $smokeText -match '(?m)\brouter_debug_id_mismatches=0\b' -and
    $smokeText -match '(?m)\brouter_debug_weight_mismatches=0\b'
if (-not $routerDebugPass) {
    throw "selected-MoE smoke did not report exact router-debug parity"
}
$dynamicLogicalQ8192ExactPass =
    $smokeText -match '(?m)\bdynamic_logical_q8192_mismatches=0\b' -and
    $smokeText -match '(?m)\bdynamic_logical_q8192_max_abs_diff=0\b'
$expectedDynamicLogicalTokens = @(
    2073, 2156, 2560, 3073, 4609, 6145, 2049, 2175,
    2176, 2177, 2559, 2561, 3071, 3072, 3583, 3584,
    3585, 4095, 4096, 4097, 4607, 4608, 6143, 6144,
    7167, 7168, 7169, 7679, 7680, 7681, 8191, 8192
)
$dynamicLogicalCaseMatches = [regex]::Matches(
    $smokeText,
    '(?m)^dynamic_logical_case index=([0-9]+) tokens=([0-9]+) total_ms=([0-9.]+) submit_ms=([0-9.]+) device_completion_ms=([0-9.]+) timing_samples=3 timing_stat=median first_sample_ms=([0-9.]+) min_ms=([0-9.]+) max_ms=([0-9.]+) reset_included=0 component_only=1 inference_success_claimed=0$'
)
$dynamicLogicalCases = @()
for ($caseIndex = 0; $caseIndex -lt $dynamicLogicalCaseMatches.Count; `
        $caseIndex++) {
    $caseMatch = $dynamicLogicalCaseMatches[$caseIndex]
    $totalMs = [double]$caseMatch.Groups[3].Value
    $submitMs = [double]$caseMatch.Groups[4].Value
    $deviceCompletionMs = [double]$caseMatch.Groups[5].Value
    $firstSampleMs = [double]$caseMatch.Groups[6].Value
    $minMs = [double]$caseMatch.Groups[7].Value
    $maxMs = [double]$caseMatch.Groups[8].Value
    $dynamicLogicalCases += [ordered]@{
        index = [int]$caseMatch.Groups[1].Value
        tokens = [int]$caseMatch.Groups[2].Value
        total_ms = $totalMs
        submit_ms = $submitMs
        device_completion_ms = $deviceCompletionMs
        timing_samples = 3
        timing_stat = 'median'
        first_sample_ms = $firstSampleMs
        min_ms = $minMs
        max_ms = $maxMs
        reset_included = $false
        submit_fraction = if ($totalMs -gt 0.0) {
            $submitMs / $totalMs
        } else {
            0.0
        }
    }
}
$dynamicLogicalTimingPass =
    $dynamicLogicalCases.Count -eq $expectedDynamicLogicalTokens.Count
if ($dynamicLogicalTimingPass) {
    for ($caseIndex = 0; $caseIndex -lt $dynamicLogicalCases.Count; `
            $caseIndex++) {
        $case = $dynamicLogicalCases[$caseIndex]
        if ([int]$case.index -ne $caseIndex -or
            [int]$case.tokens -ne $expectedDynamicLogicalTokens[$caseIndex] -or
            [double]$case.total_ms -le 0.0 -or
            [double]$case.submit_ms -lt 0.0 -or
            [double]$case.device_completion_ms -lt 0.0 -or
            [double]$case.submit_ms -gt [double]$case.total_ms -or
            [int]$case.timing_samples -ne 3 -or
            [string]$case.timing_stat -ne 'median' -or
            [double]$case.first_sample_ms -le 0.0 -or
            [double]$case.min_ms -le 0.0 -or
            [double]$case.max_ms -lt [double]$case.total_ms -or
            [double]$case.min_ms -gt [double]$case.total_ms) {
            $dynamicLogicalTimingPass = $false
            break
        }
    }
}
$dynamicLogicalPass =
    $smokeText -match '(?m)\bdynamic_logical_q3073_ms=[0-9.]+' -and
    $smokeText -match '(?m)\bdynamic_logical_q4609_ms=[0-9.]+' -and
    $smokeText -match '(?m)\bdynamic_logical_q6145_ms=[0-9.]+' -and
    $smokeText -match '(?m)\bdynamic_logical_case_count=32\b' -and
    $smokeText -match '(?m)\bdynamic_logical_timing_samples=3\b' -and
    $smokeText -match '(?m)\bdynamic_logical_timing_stat=median\b' -and
    $smokeText -match '(?m)\bdynamic_logical_min_tokens=2049\b' -and
    $smokeText -match '(?m)\bdynamic_logical_max_tokens=8192\b' -and
    $smokeText -match '(?m)\bdynamic_logical_nonfinite=0\b' -and
    $smokeText -match '(?m)\bdynamic_logical_max_abs_diff=[0-9.]+' -and
    $smokeText -match '(?m)\bdynamic_logical_tail_guard_pass=1\b' -and
    $smokeText -match '(?m)\bdynamic_logical_reset_included=0\b' -and
    $smokeText -match '(?m)\bdynamic_logical_component_only=1\b' -and
    $smokeText -match '(?m)\binference_success_claimed=0\b' -and
    $dynamicLogicalTimingPass -and
    $dynamicLogicalQ8192ExactPass
if (-not $dynamicLogicalPass) {
    throw "selected-MoE smoke did not validate dynamic logical-token launches"
}
$syncHashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_sync_hash=([0-9a-f]+)\b'
)
$asyncHashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_async_hash=([0-9a-f]+)\b'
)
$v3HashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_hash=([0-9a-f]+)\b'
)
$v3AsyncHashMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_async_hash=([0-9a-f]+)\b'
)
$asyncParityPass =
    $smokeText -match '(?m)\bfull_provider_async_mismatches=0\b' -and
    $syncHashMatch.Success -and $asyncHashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $asyncHashMatch.Groups[1].Value
if (-not $asyncParityPass) {
    throw "selected-MoE smoke did not report exact sync/async full-v2 parity"
}
$v3ParityPass =
    $smokeText -match '(?m)\bfull_provider_v3_mismatches=0\b' -and
    $syncHashMatch.Success -and $v3HashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $v3HashMatch.Groups[1].Value
if (-not $v3ParityPass) {
    throw "selected-MoE smoke did not report exact full-v2/full-v3 parity"
}
$expectedAsyncChainCalls = $FullV3EventSlots + 1
$v3AsyncParityPass =
    $smokeText -match "(?m)\bfull_provider_v3_async_chain_calls=$expectedAsyncChainCalls\b" -and
    $smokeText -match "(?m)\bfull_provider_v3_async_checked_outputs=$expectedAsyncChainCalls\b" -and
    $smokeText -match '(?m)\bfull_provider_v3_async_mismatches=0\b' -and
    $syncHashMatch.Success -and $v3AsyncHashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $v3AsyncHashMatch.Groups[1].Value
if (-not $v3AsyncParityPass) {
    throw "selected-MoE smoke did not report exact $expectedAsyncChainCalls-call full-v3-async parity"
}
$expectedFullProviderHash = "c8b9f2290b8bbd3"
$expectedFullProviderHashPass =
    $syncHashMatch.Success -and
    $syncHashMatch.Groups[1].Value -eq $expectedFullProviderHash
$v3AsyncTotalMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_async_chain_total_ms=([0-9]+(?:\.[0-9]+)?)\b'
)
$v3AsyncPerCallMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_async_per_call_ms=([0-9]+(?:\.[0-9]+)?)\b'
)
if (-not $v3AsyncTotalMatch.Success -or -not $v3AsyncPerCallMatch.Success) {
    throw "selected-MoE smoke did not report full-v3-async chain timing"
}
$v3AsyncTotalMs = [double]::Parse(
    $v3AsyncTotalMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture
)
$v3AsyncPerCallMs = [double]::Parse(
    $v3AsyncPerCallMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture
)
$v3TimingMatch = [regex]::Match(
    $smokeText,
    '(?m)\bfull_provider_v3_reset_included_ms=([0-9]+(?:\.[0-9]+)?)\b'
)
if (-not $v3TimingMatch.Success) {
    throw "selected-MoE smoke did not report reset-included full-v3 timing"
}
$v3ResetIncludedMs = [double]::Parse(
    $v3TimingMatch.Groups[1].Value,
    [Globalization.CultureInfo]::InvariantCulture
)
$expectedProviderBackendMask =
    $(if ($NativeWmmaGate -ne 0) { 1 } else { 0 }) -bor
    $(if ($NativeWmmaDown -ne 0) { 2 } else { 0 }) -bor
    $(if ($TransposedRouter -ne 0) { 4 } else { 0 }) -bor
    $(if ($FullV3FusedCombine -ne 0) { 8 } else { 0 })
if ($smokeText -notmatch
    "(?m)\bprovider_backend_mask=$expectedProviderBackendMask\b" -or
    $smokeText -notmatch '(?m)\bprovider_backend_mask_pass=1\b') {
    throw "selected-MoE smoke did not confirm provider backend mask $expectedProviderBackendMask"
}

$runtimeFiles = @(
    "q8192_selected_moe_route_count.hsaco",
    "q8192_selected_moe_route_prefix_by_program.hsaco",
    "q8192_selected_moe_route_padded_prefix.hsaco",
    "q8192_selected_moe_route_scatter.hsaco",
    "q8192_selected_moe_gate_up_silu.hsaco",
    "q8192_selected_moe_down.hsaco",
    "qrt_triton_moe_q8192_provider.dll"
)
if ($ExactShared -ne 0) {
    $runtimeFiles += $exactSharedAotFiles
}
if ($conditionalExactAotFiles.Count -ne 0) {
    $runtimeFiles += $conditionalExactAotFiles
}
$artifacts = foreach ($name in $runtimeFiles) {
    $source = Join-Path $BuildDir $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "selected-MoE build output is missing: $source"
    }
    $destination = Join-Path $OutDir $name
    $sourceFullPath = [IO.Path]::GetFullPath($source)
    $destinationFullPath = [IO.Path]::GetFullPath($destination)
    if (-not [StringComparer]::OrdinalIgnoreCase.Equals(
            $sourceFullPath,
            $destinationFullPath
        )) {
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    $item = Get-Item -LiteralPath $destination
    [ordered]@{
        file = $name
        bytes = $item.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
    }
}

$record = [ordered]@{
    schema_version = 1
    host = [Environment]::MachineName
    execution = "local_windows_process"
    repo_commit = (& git -C $repo rev-parse HEAD).Trim()
    dirty_tree = @(& git -C $repo status --porcelain).Count -ne 0
    command_file = $PSCommandPath
    command_file_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath
    ).Hash.ToLowerInvariant()
    offload_arch = $OffloadArch
    wsl_distribution = $WslDistribution
    triton_python = $TritonPython
    wsl_llvm_strip = $WslLlvmStrip
    aot_debug_sections_stripped = $true
    aot_private_home_path_count = 0
    hipcc = $HipccPath
    aot_reused = $aotReused
    aot_source_dir = $resolvedReuseAotDir
    exact_shared = ($ExactShared -ne 0)
    exact_shared_aot_source_dir = $exactSharedAotSourceDir
    conditional_exact_gate = ($ConditionalExactGate -ne 0)
    sorted_conditional_exact_gate = ($SortedConditionalExactGate -ne 0)
    row_major_sorted_conditional_exact_gate = (
        $RowMajorSortedConditionalExactGate -ne 0
    )
    conditional_exact_gate_rows = $ConditionalExactGateRows
    conditional_exact_down = ($ConditionalExactDown -ne 0)
    conditional_exact_down_rows = $ConditionalExactDownRows
    conditional_exact_aot_source_dir = $conditionalExactAotSourceDir
    provider_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $providerSource
    ).Hash.ToLowerInvariant()
    sm121_prefill_projection_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        (Join-Path $repo 'native\providers\moe_accumulator\sm121_prefill_projection.h')).Hash.ToLowerInvariant()
    sm121_wave16_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_wave16.h')).Hash.ToLowerInvariant()
    sm121_group16_modulo_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_group16_modulo.h')).Hash.ToLowerInvariant()
    midpoint_selector_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\bf16_midpoint_selector.h')
    ).Hash.ToLowerInvariant()
    scaled_l2_header_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (
        Join-Path $repo 'native\providers\moe_accumulator\bf16_scaled_l2.h'
    )).Hash.ToLowerInvariant()
    generator_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $generator
    ).Hash.ToLowerInvariant()
    smoke_source_sha256 = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $smokeSource
    ).Hash.ToLowerInvariant()
    dynamic_logical_pass = $dynamicLogicalPass
    dynamic_logical_q8192_exact_pass = $dynamicLogicalQ8192ExactPass
    dynamic_logical_case_count = 32
    dynamic_logical_timing_samples = 3
    dynamic_logical_timing_stat = 'median'
    dynamic_logical_min_tokens = 2049
    dynamic_logical_max_tokens = 8192
    dynamic_logical_nonfinite = 0
    dynamic_logical_reset_included = $false
    dynamic_logical_component_only = $true
    inference_success_claimed = $false
    dynamic_logical_cases = $dynamicLogicalCases
    tile = [ordered]@{
        block_m = $BlockM
        aot_block_m = $aotBlockM
        block_n = $BlockN
        gate_block_n = $GateBlockN
        down_block_n = $DownBlockN
        gate_block_k = $GateBlockK
        down_block_k = $DownBlockK
        group_m = $GroupM
        num_warps = $NumWarps
        num_stages = $NumStages
        waves_per_eu = $WavesPerEu
        threads = $routeThreads
        route_num_warps = $NumWarps
        route_num_stages = $NumStages
        route_waves_per_eu = $WavesPerEu
        gate_num_warps = $GateNumWarps
        gate_num_stages = $GateNumStages
        gate_waves_per_eu = $GateWavesPerEu
        down_num_warps = $DownNumWarps
        down_num_stages = $DownNumStages
        down_waves_per_eu = $DownWavesPerEu
        vllm_silu_bf16_intermediate =
            ($VllmSiluBf16Intermediate -ne 0)
        route_threads = $routeThreads
        gate_threads = $gateThreads
        down_threads = $downThreads
        gate_dynamic_shared_bytes = [int]$gateMetadata.dynamic_shared_bytes
        down_dynamic_shared_bytes = [int]$downMetadata.dynamic_shared_bytes
    }
    build_dir = (Resolve-Path -LiteralPath $BuildDir).Path
    out_dir = (Resolve-Path -LiteralPath $OutDir).Path
    async_parity_pass = $asyncParityPass
    v3_parity_pass = $v3ParityPass
    v3_async_parity_pass = $v3AsyncParityPass
    router_debug_parity_pass = $routerDebugPass
    native_wmma_routed = ($NativeWmmaRouted -ne 0)
    routed_projection_debug = ($RoutedProjectionDebug -ne 0)
    batched_hawkeye = ($BatchedHawkeye -ne 0)
    routed_replay_lanes = $RoutedReplayLanes
    sm121_dpp_reduction = ($DppReduction -ne 0)
    sm121_compact_normalize = ($CompactNormalize -ne 0)
    sm121_canonical_normalize_header_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_canonical_normalize.h')).Hash.ToLowerInvariant()
    sm121_dot_staging_groups = $DotStagingGroups
    sm121_certified_dot_tiles = ($CertifiedDotTiles -ne 0)
    sm121_dot_certificate_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_dot_certificate.h')).Hash.ToLowerInvariant()
    sm121_lane_reduce_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_lane_reduce.h')).Hash.ToLowerInvariant()
    parallel_gate_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\triton_moe\routed_parallel_gate.h')).Hash.ToLowerInvariant()
    sm121_subgroup_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_subgroup.h')).Hash.ToLowerInvariant()
    sm121_paired_products_header_sha256 = (Get-FileHash -Algorithm SHA256 `
        -LiteralPath (Join-Path $repo 'native\providers\moe_accumulator\sm121_paired_products.h')).Hash.ToLowerInvariant()
    full_shared_hawkeye = ($FullSharedHawkeye -ne 0)
    native_wmma_gate = ($NativeWmmaGate -ne 0)
    native_wmma_down = ($NativeWmmaDown -ne 0)
    native_fused_route_layout = ($NativeFusedRouteLayout -ne 0)
    native_wmma_adaptive_tail = ($NativeWmmaAdaptiveTail -ne 0)
    native_wmma_bucketed_tail = ($NativeWmmaBucketedTail -ne 0)
    native_wmma_parallel_bucketed_tail = ($NativeWmmaParallelBucketedTail -ne 0)
    native_wmma_compact_split_tail = ($NativeWmmaCompactSplitTail -ne 0)
    native_wmma_tail32 = ($NativeWmmaTail32 -ne 0)
    native_wmma_prune16 = ($NativeWmmaPrune16 -ne 0)
    native_wmma_wide_n = ($NativeWmmaWideN -ne 0)
    native_wmma_lds_b = ($NativeWmmaLdsB -ne 0)
    native_wmma_lds_b_narrow_n = ($NativeWmmaLdsBNarrowN -ne 0)
    native_wmma_lds_b_m96 = ($NativeWmmaLdsBM96 -ne 0)
    native_wmma_lds_b_m64 = ($NativeWmmaLdsBM64 -ne 0)
    native_wmma_lds_b_m80_load_wave = ($NativeWmmaLdsBM80LoadWave -ne 0)
    native_wmma_lds_b_m80_load_threads = $nativeWmmaLdsBM80LoadThreadsResolved
    native_wmma_lds_b_m64_load_threads = $NativeWmmaLdsBM64LoadThreads
    native_wmma_lds_b_m64_gate_load_threads = $NativeWmmaLdsBM64GateLoadThreads
    native_wmma_lds_b_m64_down_load_threads = $NativeWmmaLdsBM64DownLoadThreads
    native_wmma_lds_b_split_gate_passes = ($NativeWmmaLdsBSplitGatePasses -ne 0)
    native_wmma_lds_b_serial_gate_n32 = ($NativeWmmaLdsBSerialGateN32 -ne 0)
    native_wmma_lds_b_parallel_gate_n64 = ($NativeWmmaLdsBParallelGateN64 -ne 0)
    native_wmma_lds_b_parallel_gate_up_n32 = ($NativeWmmaLdsBParallelGateUpN32 -ne 0)
    native_wmma_lds_b_serial_down_n32 = ($NativeWmmaLdsBSerialDownN32 -ne 0)
    native_wmma_lds_b_serial_wide_n64 = ($NativeWmmaLdsBSerialWideN64 -ne 0)
    native_wmma_lds_b_compact_m96_waves = ($NativeWmmaLdsBCompactM96Waves -ne 0)
    native_wmma_lds_b_skip_inactive_a_stores = ($NativeWmmaLdsBSkipInactiveAStores -ne 0)
    native_wmma_lds_b_fused_overflow16 = ($NativeWmmaLdsBFusedOverflow16 -ne 0)
    native_wmma_lds_b_m64_fused_overflow32 = ($NativeWmmaLdsBM64FusedOverflow32 -ne 0)
    native_wmma_lossless_palette = ($NativeWmmaLosslessPalette -ne 0)
    native_wmma_lossless_row_palette = ($NativeWmmaLosslessRowPalette -ne 0)
    native_wmma_weight_int8 = ($NativeWmmaWeightInt8 -ne 0)
    native_wmma_weight_int8_group_values =
        $NativeWmmaWeightInt8GroupValues
    native_wmma_weight_int8_scale_fp16 =
        ($NativeWmmaWeightInt8ScaleFp16 -ne 0)
    native_wmma_lds_b_m64_direct_m16 = ($NativeWmmaLdsBM64DirectM16 -ne 0)
    native_wmma_lds_b_m64_dual_m16 = ($NativeWmmaLdsBM64DualM16 -ne 0)
    native_wmma_lds_b_grouped_sole_m16 = ($NativeWmmaLdsBGroupedSoleM16 -ne 0)
    native_wmma_lds_b_m64_quad_gate_m16 = ($NativeWmmaLdsBM64QuadGateM16 -ne 0)
    native_wmma_lds_b_adaptive_m128 = ($NativeWmmaLdsBAdaptiveM128 -ne 0)
    native_wmma_lds_b_hybrid_m96_m128 = ($NativeWmmaLdsBHybridM96M128 -ne 0)
    native_wmma_lds_b_adaptive_direct_grid = ($NativeWmmaLdsBAdaptiveDirectGrid -ne 0)
    native_wmma_lds_b_full_direct_tail = ($NativeWmmaLdsBFullDirectTail -ne 0)
    native_wmma_lds_b_compact_sole_tail = ($NativeWmmaLdsBCompactSoleTail -ne 0)
    native_wmma_lds_b_parallel_compact_sole_tail = ($NativeWmmaLdsBParallelCompactSoleTail -ne 0)
    native_wmma_tall_m96 = ($NativeWmmaTallM96 -ne 0)
    native_wmma_k_stage = $NativeWmmaKStage
    transposed_router = ($TransposedRouter -ne 0)
    router_threads = $RouterThreads
    router_token_tile = $RouterTokenTile
    full_v3_fused_combine = ($FullV3FusedCombine -ne 0)
    fused_combine_width = $FusedCombineWidth
    full_v3_event_slots = $FullV3EventSlots
    provider_backend_mask = $expectedProviderBackendMask
    expected_full_provider_hash = $expectedFullProviderHash
    expected_full_provider_hash_pass = $expectedFullProviderHashPass
    expected_full_provider_hash_diagnostic_only = $true
    expected_full_provider_hash_requirement_requested = `
        ($RequireExpectedFullProviderHash -ne 0)
    full_provider_v3_async_under_29ms = ($v3AsyncPerCallMs -le 29.0)
    v3_async_chain_total_ms = $v3AsyncTotalMs
    v3_async_per_call_ms = $v3AsyncPerCallMs
    v3_async_chain_calls = $expectedAsyncChainCalls
    v3_reset_included_ms = $v3ResetIncludedMs
    smoke = $smokeText
    artifacts = $artifacts
}
$json = $record | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText(
    (Join-Path $OutDir "build-provenance.json"),
    $json + [Environment]::NewLine,
    $utf8
)
Write-Output $json
