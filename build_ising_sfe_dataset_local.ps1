# Build Ising/SFE graph-level total-energy PyG dataset.
# Requires jara-ovito (OVITO + torch + torch_geometric) + 20261020_data.
#
# Smoke:  .\build_ising_sfe_dataset_local.ps1 -Limit 50 -Materials ac-ca
# Full:   .\build_ising_sfe_dataset_local.ps1

param(
    [int]$Limit = 0,
    [string[]]$Materials = @(),
    [string[]]$Kinds = @("ising_polytype", "supercell")
)

$ErrorActionPreference = "Stop"
$Python = "D:\Programme\miniconda3\envs\jara-ovito\python.exe"
if (-not (Test-Path $Python)) { throw "Missing: $Python" }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$argsList = @(
    (Join-Path $Root "ising_sfe_graph_maker.py"),
    "--kinds"
) + $Kinds

if ($Limit -gt 0) { $argsList += @("--limit", "$Limit") }
if ($Materials.Count -gt 0) {
    $argsList += @("--materials") + $Materials
}

Push-Location $Root
try {
    Write-Host "[build] $Python $($argsList -join ' ')"
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) { throw "build failed with exit $LASTEXITCODE" }
} finally {
    Pop-Location
}
