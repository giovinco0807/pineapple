Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'HuM43Attempt04Spot.Common.ps1')

$script:M43A8TotalShards = 200
$script:M43A8MaxWaveShards = 25

function Expand-M43A8ShardSelection {
    param(
        [Parameter(Mandatory = $true)][string[]]$Values,
        [int]$MaxCount = $script:M43A8MaxWaveShards
    )
    $selected = [Collections.Generic.SortedSet[int]]::new()
    foreach ($raw in $Values) {
        foreach ($token in ([string]$raw -split ',')) {
            $value = $token.Trim()
            if (-not $value) { continue }
            if ($value -match '^(\d+)-(\d+)$') {
                $first = [int]$Matches[1]
                $last = [int]$Matches[2]
                if ($first -gt $last) { throw "Invalid Attempt08 shard range: $value" }
                for ($index = $first; $index -le $last; $index++) {
                    [void]$selected.Add($index)
                }
            }
            elseif ($value -match '^\d+$') { [void]$selected.Add([int]$value) }
            else { throw "Invalid Attempt08 shard selection: $value" }
        }
    }
    foreach ($index in $selected) {
        if ($index -lt 0 -or $index -ge $script:M43A8TotalShards) {
            throw "Attempt08 shard is outside 0..199: $index"
        }
    }
    $result = @($selected)
    if ($result.Count -eq 0) { throw 'Attempt08 selected no shards' }
    if ($MaxCount -gt 0 -and $result.Count -gt $MaxCount) {
        throw "Attempt08 wave may contain at most $MaxCount shards"
    }
    return $result
}

function ConvertTo-M43A8VmPrefix {
    param([Parameter(Mandatory = $true)][string]$Value)
    $result = (($Value.ToLowerInvariant() -replace '[^a-z0-9-]', '-') -replace '-+', '-').Trim('-')
    if (-not $result) { throw 'Attempt08 RunName cannot produce a VM prefix' }
    if ($result.Length -gt 54) { $result = $result.Substring(0, 54).TrimEnd('-') }
    return $result
}

function Invoke-M43A8Python {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$TimeoutSeconds = 1800
    )
    $pythonPath = (Get-Command python -ErrorAction Stop | Select-Object -First 1).Source
    $result = Invoke-M43A4ProcessBounded `
        -FilePath $pythonPath -Arguments $Arguments -Label $Label `
        -TimeoutSeconds $TimeoutSeconds `
        -Environment @{ PYTHONPATH = (Join-Path $RepoRoot 'src') }
    if ($result.timed_out -or $result.exit_code -ne 0) {
        throw "$Label failed: $($result.stderr)$($result.stdout)"
    }
    return [string]$result.stdout
}

function Publish-M43A8ImmutableObject {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$ProjectId
    )
    $upload = Invoke-M43A4GcloudProcess `
        -Arguments @('storage','cp',$Source,$Uri,'--project',$ProjectId,'--if-generation-match=0') `
        -TimeoutSeconds 600 -Label "publish immutable Attempt08 object $Uri"
    if (-not $upload.timed_out -and $upload.exit_code -eq 0) { return 'created' }
    $temporary = Join-Path ([IO.Path]::GetTempPath()) ('m43a8-existing-' + [guid]::NewGuid().ToString('N'))
    try {
        Copy-M43A4RemoteFileExact -Uri $Uri -Destination $temporary -ProjectId $ProjectId
        if ((Get-M43A4Sha256 $temporary) -ne (Get-M43A4Sha256 $Source)) {
            throw "Immutable Attempt08 object differs: $Uri"
        }
        return 'existing-identical'
    }
    finally { Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue }
}

