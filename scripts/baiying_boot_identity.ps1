# Win32_OperatingSystem.LastBootUpTime can shift after wall-clock corrections.
# A recorded kernel boot event remains the identity of that boot. Preserve the
# CIM value separately for diagnostics and as a strict fallback if the event
# log is unavailable; a changed identity or identity source never passes.
function Get-QrtBootIdentity([string]$CimBootTime) {
    try {
        $event = Get-WinEvent -FilterHashtable @{
            LogName = 'System'; ProviderName = 'Microsoft-Windows-Kernel-General'; Id = 12
        } -MaxEvents 1 -ErrorAction Stop
        if ($null -eq $event -or $null -eq $event.TimeCreated) { throw 'Missing kernel boot event' }
        return [pscustomobject]@{
            kind = 'kernel_general_boot_event'
            value = ([string]$event.RecordId) + ':' + $event.TimeCreated.ToUniversalTime().Ticks
            record_id = [long]$event.RecordId
            utc = $event.TimeCreated.ToUniversalTime().ToString('o')
            cim_boot = $CimBootTime
        }
    } catch {
        return [pscustomobject]@{
            kind = 'cim_boot_time'; value = $CimBootTime
            cim_boot = $CimBootTime; event_lookup_error = $_.Exception.Message
        }
    }
}

function Test-QrtSameBoot($Before, $After) {
    return $null -ne $Before -and $null -ne $After -and
        -not [string]::IsNullOrEmpty([string]$Before.value) -and
        $Before.kind -eq $After.kind -and $Before.value -eq $After.value
}
