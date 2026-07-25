# Predict a slate of games. Usage:
#   .\predict.ps1                # today
#   .\predict.ps1 2026-08-01     # a specific date
#   .\predict.ps1 today -Fast    # skip the data refresh (uses cached data, ~2s)
#
# Writes reports\predictions_<date>.csv, overwriting any earlier run for that date.
param(
    [string]$Date = "today",
    [switch]$Fast
)

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "No virtualenv at $py - see README Quickstart for setup." -ForegroundColor Red
    exit 1
}

# Progress logging goes to stderr; PowerShell 5.1 would otherwise flag it as failure.
$cmd = @("-m", "mlbpred.predict", "--date", $Date)
if ($Fast) { $cmd += "--no-refresh" }

& $py @cmd 2>&1 | ForEach-Object { "$_" }

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nPrediction failed (exit $LASTEXITCODE)." -ForegroundColor Red
    exit $LASTEXITCODE
}
