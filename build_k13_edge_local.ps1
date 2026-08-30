# Build k=13 edge_k wiring datasets (e03..e06). Requires jara-ovito + SIMULATIONS/.
#   .\build_k13_edge_local.ps1
#   .\build_k13_edge_local.ps1 -Tier e04 -Force

param(
    [string]$Tier = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\Programme\miniconda3\envs\jara-ovito\python.exe"

if (-not (Test-Path $Python)) { throw "Missing: $Python" }
if (-not (Test-Path (Join-Path $Root "SIMULATIONS"))) { throw "Missing SIMULATIONS/" }

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$argsList = @("-u", (Join-Path $Root "build_point_k13_edge_datasets.py"))
if ($Force) { $argsList += "--force" }
if ($Tier) { $argsList += @("--tier", $Tier) }

Write-Host "[build] $Python $($argsList -join ' ')"
Push-Location $Root
try { & $Python @argsList } finally { Pop-Location }
