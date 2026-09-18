$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$path = Join-Path $repo 'scripts\baiying_guarded_inference.ps1'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'Guard script parse error' }
# Exercise the actual parameter declarations without executing a guard body,
# creating a process, or acquiring the GPU mutex. The 64k acceptance run needs
# an explicit 1800-second deadline; the previous 900 ceiling rejected it before
# preflight. Default duration and rejection of out-of-range values stay.
$parameterCheck = [ScriptBlock]::Create($ast.ParamBlock.Extent.Text + "`nreturn $" + 'TimeoutSeconds')
$commonParameters = @{SpecPath='unused-spec.json';OutDir='unused-output'}
if ((& $parameterCheck @commonParameters) -ne 90) { throw 'Default timeout changed' }
foreach ($seconds in @(1, 90, 900, 1800)) {
    if ((& $parameterCheck @commonParameters -TimeoutSeconds $seconds) -ne $seconds) {
        throw 'Explicit bounded timeout was not admitted'
    }
}
foreach ($seconds in @(0, -1, 1801, [int]::MaxValue)) {
    $rejected = $false
    try { $null = & $parameterCheck @commonParameters -TimeoutSeconds $seconds }
    catch {
        if ($_.FullyQualifiedErrorId -notlike 'ParameterArgumentValidationError*') { throw }
        $rejected = $true
    }
    if (-not $rejected) { throw 'Out-of-range timeout was admitted' }
}
$quote = $ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Quote-Argument'}, $true)
if ($null -eq $quote) { throw 'Missing original quoting function' }
Invoke-Expression $quote.Extent.Text
$source = [IO.File]::ReadAllText($path)
$start = $source.IndexOf('        $argumentLine = ')
$end = $source.IndexOf('        try { [QrtRunGuard]::Assign', $start)
if ($start -lt 0 -or $end -lt 0) { throw 'Missing actual process-launch block' }
$launch = [ScriptBlock]::Create($source.Substring($start, $end - $start))
function Start-Process {
    [CmdletBinding()]
    param([string]$FilePath, [AllowEmptyString()][string]$ArgumentList,
          [string]$WorkingDirectory, [switch]$PassThru, [switch]$NoNewWindow,
          [string]$RedirectStandardOutput, [string]$RedirectStandardError)
    return @{arguments_present=$PSBoundParameters.ContainsKey('ArgumentList');arguments=$ArgumentList;
        file=$FilePath;directory=$WorkingDirectory;pass_thru=[bool]$PassThru;no_window=[bool]$NoNewWindow;
        stdout=$RedirectStandardOutput;stderr=$RedirectStandardError}
}
$stdoutPath = 'stdout.jsonl'; $stderrPath = 'stderr.log'
$cases = @(
    @{args=@();present=$false;expected=''},
    @{args=@('');present=$true;expected='""'},
    @{args=@('--flag', 'path with spaces');present=$true;expected='"--flag" "path with spaces"'},
    @{args=@('quote"inside', 'trailing\');present=$true;expected='"quote\"inside" "trailing\\"'}
)
foreach ($case in $cases) {
    $spec = [pscustomobject]@{executable='test.exe';working_directory='D:\test space';arguments=$case.args}
    . $launch
    if ($process.arguments_present -ne $case.present -or $process.arguments -cne $case.expected -or
        $process.file -cne $spec.executable -or $process.directory -cne $spec.working_directory -or
        -not $process.pass_thru -or -not $process.no_window -or $process.stdout -cne $stdoutPath -or
        $process.stderr -cne $stderrPath) { throw 'Actual process parameter transport changed' }
}
[ordered]@{kind='guarded_process_arguments';cases=$cases.Count;timeout_cases=9;maximum_timeout_seconds=1800;pass=$true;actual_guard_block=$true;actual_parameter_declarations=$true;process_started=$false} | ConvertTo-Json -Compress
