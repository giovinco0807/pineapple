Set-StrictMode -Version Latest

$baseCommon = if ((Get-Variable -Name Audit50DevelopmentRunDir -Scope Script -ErrorAction SilentlyContinue) -and $script:Audit50DevelopmentRunDir) {
    Join-Path $script:Audit50DevelopmentRunDir 'package_src/scripts/HuM43Attempt08Spot.Common.ps1'
} else {
    Join-Path $PSScriptRoot 'HuM43Attempt08Spot.Common.ps1'
}
if (-not (Test-Path -LiteralPath $baseCommon -PathType Leaf)) {
    throw "Attempt08 audit50 frozen base common module is missing: $baseCommon"
}
. $baseCommon

$script:M43A8TotalShards = 50
$script:M43A8MaxWaveShards = 25

function Expand-M43A8Audit50ShardSelection {
    param([Parameter(Mandatory = $true)][string[]]$Values)
    $selected = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) {
        foreach ($token in ([string]$raw -split ',')) {
            $value = $token.Trim()
            if (-not $value) { continue }
            if ($value -match '^(\d+)-(\d+)$') {
                $first = [int]$Matches[1]; $last = [int]$Matches[2]
                if ($first -gt $last) { throw "Invalid audit50 shard range: $value" }
                for ($index = $first; $index -le $last; $index++) { [void]$selected.Add($index) }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid audit50 shard selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge 50) { throw "Attempt08 audit50 shard is outside 0..49: $index" }
    }
    $result = @($selected)
    if ($result.Count -eq 0) { throw 'Attempt08 audit50 selected no shards' }
    if ($result.Count -gt 25) { throw 'Attempt08 audit50 wave may contain at most 25 shards' }
    return $result
}

function Invoke-M43A8Audit50Python {
    param(
        [Parameter(Mandatory = $true)][string]$ModuleFile,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$TimeoutSeconds = 3600
    )
    $pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath -Arguments (@('-B',$ModuleFile) + $Arguments) `
        -Label $Label -TimeoutSeconds $TimeoutSeconds `
        -Environment @{ PYTHONDONTWRITEBYTECODE = '1' }
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "$Label failed: $($result.stderr)$($result.stdout)"
    }
    return [string]$result.stdout
}

function Publish-M43A8Audit50ImmutableObject {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ProjectId
    )
    return Publish-M43A8ImmutableObject -Source $Source -Uri $Uri -ProjectId $ProjectId
}
