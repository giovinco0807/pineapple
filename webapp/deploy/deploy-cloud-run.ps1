[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z][a-z0-9-]{4,28}[a-z0-9]$')]
    [string]$ProjectId,

    [ValidatePattern('^[a-z0-9-]+$')]
    [string]$Region = 'asia-northeast1',

    [ValidatePattern('^[a-z][a-z0-9-]{0,62}$')]
    [string]$ServiceName = 'ofc-pineapple-webapp',

    [ValidatePattern('^[a-z][a-z0-9-]{2,62}$')]
    [string]$ArtifactRepository = 'ofc-webapp',

    [string]$RecordsBucket = '',

    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$SecretName = 'ofc-webapp-shared-token',

    [ValidatePattern('^[a-z][a-z0-9-]{4,28}[a-z0-9]$')]
    [string]$ServiceAccountName = 'ofc-webapp-runner',

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$ImageTag = (Get-Date -Format 'yyyyMMdd-HHmmss')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Gcloud {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GcloudArguments
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & gcloud @GcloudArguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "gcloud failed with exit code $exitCode`: gcloud $($GcloudArguments -join ' ')"
    }
}

function Test-GcloudResource {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$GcloudArguments
    )

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & gcloud @GcloudArguments *> $null
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return $exitCode -eq 0
}

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    throw 'Google Cloud CLI (gcloud) is required.'
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot 'webapp\Dockerfile'))) {
    throw "Run this script from an intact regular-ofc-pineapple checkout: $RepoRoot"
}

if ([string]::IsNullOrWhiteSpace($RecordsBucket)) {
    $RecordsBucket = "ofc-webapp-records-$ProjectId".ToLowerInvariant()
}
if (
    $RecordsBucket.Length -gt 63 -or
    $RecordsBucket -notmatch '^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$'
) {
    throw "RecordsBucket is not a valid Cloud Storage bucket name: $RecordsBucket"
}

$ServiceAccount = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"
$Image = "$Region-docker.pkg.dev/$ProjectId/$ArtifactRepository/$ServiceName`:$ImageTag"
$BucketUri = "gs://$RecordsBucket"

Push-Location $RepoRoot
try {
    Invoke-Gcloud -GcloudArguments @(
        'services', 'enable',
        'artifactregistry.googleapis.com',
        'cloudbuild.googleapis.com',
        'iam.googleapis.com',
        'run.googleapis.com',
        'secretmanager.googleapis.com',
        'storage.googleapis.com',
        '--project', $ProjectId
    )

    if (-not (Test-GcloudResource -GcloudArguments @(
        'artifacts', 'repositories', 'describe', $ArtifactRepository,
        '--project', $ProjectId,
        '--location', $Region
    ))) {
        Invoke-Gcloud -GcloudArguments @(
            'artifacts', 'repositories', 'create', $ArtifactRepository,
            '--project', $ProjectId,
            '--location', $Region,
            '--repository-format', 'docker',
            '--description', 'OFC Pineapple Web application images'
        )
    }

    if (-not (Test-GcloudResource -GcloudArguments @(
        'iam', 'service-accounts', 'describe', $ServiceAccount,
        '--project', $ProjectId
    ))) {
        Invoke-Gcloud -GcloudArguments @(
            'iam', 'service-accounts', 'create', $ServiceAccountName,
            '--project', $ProjectId,
            '--display-name', 'OFC Pineapple Web runtime'
        )
    }

    if (-not (Test-GcloudResource -GcloudArguments @(
        'secrets', 'describe', $SecretName,
        '--project', $ProjectId
    ))) {
        throw (
            "Secret '$SecretName' does not exist. Create it with a SHARED_TOKEN " +
            'version before running this deployment script; see webapp/README.md.'
        )
    }
    if (-not (Test-GcloudResource -GcloudArguments @(
        'secrets', 'versions', 'describe', 'latest',
        '--secret', $SecretName,
        '--project', $ProjectId
    ))) {
        throw (
            "Secret '$SecretName' has no enabled latest version. Add a " +
            'SHARED_TOKEN version before deploying.'
        )
    }

    if (-not (Test-GcloudResource -GcloudArguments @(
        'storage', 'buckets', 'describe', $BucketUri,
        '--project', $ProjectId
    ))) {
        Invoke-Gcloud -GcloudArguments @(
            'storage', 'buckets', 'create', $BucketUri,
            '--project', $ProjectId,
            '--location', $Region,
            '--uniform-bucket-level-access',
            '--public-access-prevention'
        )
    }

    Invoke-Gcloud -GcloudArguments @(
        'storage', 'buckets', 'add-iam-policy-binding', $BucketUri,
        '--project', $ProjectId,
        '--member', "serviceAccount:$ServiceAccount",
        '--role', 'roles/storage.objectAdmin'
    )
    Invoke-Gcloud -GcloudArguments @(
        'secrets', 'add-iam-policy-binding', $SecretName,
        '--project', $ProjectId,
        '--member', "serviceAccount:$ServiceAccount",
        '--role', 'roles/secretmanager.secretAccessor'
    )

    Invoke-Gcloud -GcloudArguments @(
        'builds', 'submit', '.',
        '--project', $ProjectId,
        '--config', 'webapp/cloudbuild.yaml',
        '--ignore-file', 'webapp/deploy/cloudbuild.gcloudignore',
        '--substitutions', "_IMAGE=$Image"
    )

    $RuntimeEnvironment = @(
        "OFC_ASSEMBLY_PATH=/app/assembly.json",
        "OFC_ASSEMBLY_ROOT=/app",
        "OFC_DB_PATH=/tmp/ofc-webapp.sqlite3",
        "OFC_GCS_BUCKET=$RecordsBucket",
        "OFC_GCS_OBJECT=sqlite/ofc-webapp.sqlite3",
        "OFC_STATIC_DIR=/app/static"
    ) -join ','

    Invoke-Gcloud -GcloudArguments @(
        'run', 'deploy', $ServiceName,
        '--project', $ProjectId,
        '--region', $Region,
        '--platform', 'managed',
        '--image', $Image,
        '--service-account', $ServiceAccount,
        '--allow-unauthenticated',
        '--ingress', 'all',
        '--cpu', '1',
        '--memory', '1Gi',
        '--min', '0',
        '--max', '1',
        '--concurrency', '4',
        '--timeout', '300',
        '--port', '8080',
        '--set-env-vars', $RuntimeEnvironment,
        '--set-secrets', "SHARED_TOKEN=$SecretName`:latest",
        '--quiet'
    )

    $ServiceUrl = & gcloud run services describe $ServiceName `
        --project $ProjectId `
        --region $Region `
        --format 'value(status.url)'
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ServiceUrl)) {
        throw 'Deployment succeeded but the Cloud Run service URL could not be read.'
    }

    Write-Host "Cloud Run service: $ServiceUrl"
    Write-Host "Authenticated metadata endpoint: $ServiceUrl/api/meta"
    Write-Host "Records bucket: $BucketUri"
    Write-Host "Image: $Image"
}
finally {
    Pop-Location
}
