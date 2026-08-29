# Build point graph-fraction datasets locally (requires SIMULATIONS/ + jara-ovito env).
# Usage:
#   .\build_graph_fractions_local.ps1
#   .\build_graph_fractions_local.ps1 -Tier p32 -Force

param(
    [string]$Tier = "",
    [switch]$Force,
    [switch]$BaselineOnly,
    [switch]$SkipBaseline
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\Programme\miniconda3\envs\jara-ovito\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing conda env python: $Python (expected env: jara-ovito)"
}
if (-not (Test-Path (Join-Path $Root "SIMULATIONS"))) {
    throw "SIMULATIONS/ not found under $Root"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

$argsList = @("-u", (Join-Path $Root "build_point_graph_fraction_datasets.py"))
if ($Force) { $argsList += "--force" }
if ($BaselineOnly) { $argsList += "--baseline-only" }
if ($SkipBaseline) { $argsList += "--skip-baseline" }
if ($Tier) { $argsList += @("--tier", $Tier) }

Write-Host "[build] $Python $($argsList -join ' ')"
Push-Location $Root
try {
    & $Python @argsList
} finally {
    Pop-Location
}
