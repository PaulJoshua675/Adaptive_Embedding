param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PidFile = Join-Path $Root "data\api-$Port.pid"

if (-not (Test-Path $PidFile)) {
    Write-Output "No managed API PID file found for port $Port."
    exit 0
}

$ApiProcessId = [int](Get-Content $PidFile -Raw)
$ApiProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ApiProcessId" -ErrorAction SilentlyContinue
if ($ApiProcess -and $ApiProcess.CommandLine -match "uvicorn main_api:app") {
    Stop-Process -Id $ApiProcessId -Force
    Write-Output "Stopped API process $ApiProcessId."
} else {
    Write-Output "PID $ApiProcessId is not the managed Uvicorn process; leaving it untouched."
}
Remove-Item $PidFile -Force