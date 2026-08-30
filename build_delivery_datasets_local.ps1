# Build delivery datasets (k13 point + planar). Requires jara-ovito + SIMULATIONS/.
#   .\build_delivery_datasets_local.ps1
#   .\build_delivery_datasets_local.ps1 -Force

param(
    [switch]$Force,
    [switch]$PointOnly,
    [switch]$PlanarOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\Programme\miniconda3\envs\jara-ovito\python.exe"

if (-not (Test-Path $Python)) { throw "Missing: $Python" }

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$argsList = @("-u", (Join-Path $Root "build_delivery_datasets.py"))
if ($Force) { $argsList += "--force" }
if ($PointOnly) { $argsList += "--point-only" }
if ($PlanarOnly) { $argsList += "--planar-only" }

Write-Host "[build] $Python $($argsList -join ' ')"
Push-Location $Root
try { & $Python @argsList } finally { Pop-Location }
