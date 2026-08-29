# Build point graph-size datasets (shell cutoff_k). Requires jara-ovito + SIMULATIONS/.
#   .\build_graph_size_local.ps1
#   .\build_graph_size_local.ps1 -Tier k10 -Force

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

$argsList = @("-u", (Join-Path $Root "build_point_graph_size_datasets.py"))
if ($Force) { $argsList += "--force" }
if ($Tier) { $argsList += @("--tier", $Tier) }

Write-Host "[build] $Python $($argsList -join ' ')"
Push-Location $Root
try { & $Python @argsList } finally { Pop-Location }
