$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'baiying_boot_identity.ps1')
function Identity([string]$Kind, [string]$Value, [string]$Clock) {
    return [pscustomobject]@{kind=$Kind;value=$Value;cim_boot=$Clock}
}
$event = Identity 'kernel_general_boot_event' '12:1000' 'clock-before'
$cases = @(
    @{name='clock_adjustment_keeps_boot';before=$event;after=(Identity 'kernel_general_boot_event' '12:1000' 'clock-after');expected=$true},
    @{name='new_boot_record_rejected';before=$event;after=(Identity 'kernel_general_boot_event' '13:1001' 'clock-before');expected=$false},
    @{name='reused_record_number_rejected';before=$event;after=(Identity 'kernel_general_boot_event' '12:1001' 'clock-before');expected=$false},
    @{name='identity_source_change_rejected';before=$event;after=(Identity 'cim_boot_time' '12:1000' 'clock-before');expected=$false},
    @{name='fallback_stable_clock';before=(Identity 'cim_boot_time' 'clock' 'clock');after=(Identity 'cim_boot_time' 'clock' 'clock');expected=$true},
    @{name='fallback_changed_clock_rejected';before=(Identity 'cim_boot_time' 'before' 'before');after=(Identity 'cim_boot_time' 'after' 'after');expected=$false},
    @{name='missing_identity_rejected';before=$null;after=$event;expected=$false}
)
foreach($case in $cases) {
    if ((Test-QrtSameBoot $case.before $case.after) -ne $case.expected) { throw $case.name }
}
$clock = (Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5).LastBootUpTime.ToString('o')
$before = Get-QrtBootIdentity $clock
$after = Get-QrtBootIdentity $clock
if ($before.kind -ne 'kernel_general_boot_event' -or -not (Test-QrtSameBoot $before $after)) { throw 'Live boot event control failed' }
[ordered]@{kind='qrt_boot_identity_control';cases=$cases.Count;pass=$true;live_before=$before;live_after=$after;inference_acceptance=$false}|ConvertTo-Json -Depth 4
