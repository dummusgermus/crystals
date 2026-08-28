# Local step: merge cluster inference outputs + LAMMPS dumps -> predictions_new/
#
# Requires:
#   predictions_new_inference/   (from cluster)
#   predictions_new/delivery_split_indices.json
#   SIMULATIONS/ + Laves_* dump trees
#
# Usage:
#   .\run_predictions_new_restore.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$SplitJson = if ($env:SPLIT_JSON) { $env:SPLIT_JSON } else { "predictions_new/delivery_split_indices.json" }
$InferRoot = if ($env:INFER_ROOT) { $env:INFER_ROOT } else { "predictions_new_inference" }
$OutRoot = if ($env:OUT_ROOT) { $env:OUT_ROOT } else { "predictions_new" }

if (-not (Test-Path $SplitJson)) {
    Write-Error "[FATAL] Missing $SplitJson (run cluster training first)."
}
if (-not (Test-Path (Join-Path $InferRoot "point_cgcnn"))) {
    Write-Error "[FATAL] Missing $InferRoot/ (pull cluster inference outputs)."
}

python -u crystal_prediction_export.py `
  --mode restore-merged `
  --job all `
  --job-profile global_v2 `
  --inference-root $InferRoot `
  --output-root $OutRoot `
  --split-json $SplitJson

Write-Host "[done] Merged CSVs -> $OutRoot/point/ and $OutRoot/planar/"
