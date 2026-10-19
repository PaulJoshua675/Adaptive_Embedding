param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000,
    [int]$StartupTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PidFile = Join-Path $Root "data\api-$Port.pid"
$StdoutLog = Join-Path $Root "data\api-$Port.stdout.log"
$StderrLog = Join-Path $Root "data\api-$Port.stderr.log"

if (-not (Test-Path $Python)) {
    throw "Project virtualenv Python was not found at $Python. Create .venv and install requirements.txt first."
}
New-Item -ItemType Directory -Force -Path (Join-Path $Root "data") | Out-Null

$ExistingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($ExistingListener) {
    throw "Port $Port is already listening (PID $($ExistingListener.OwningProcess))."
}

$Process = Start-Process -FilePath $Python `
    -ArgumentList @("-m", "uvicorn", "main_api:app", "--host", $HostAddress, "--port", "$Port") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru
Set-Content -Path $PidFile -Value $Process.Id -NoNewline

$Deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
do {
    if ($Process.HasExited) {
        Get-Content $StderrLog -Tail 60 -ErrorAction SilentlyContinue
        throw "Uvicorn exited with code $($Process.ExitCode). See $StderrLog."
    }
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($Health.status -eq "ok") {
            Write-Output "API ready at http://127.0.0.1:$Port (PID $($Process.Id)); listening on $HostAddress."
            exit 0
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
} while ((Get-Date) -lt $Deadline)

Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
Get-Content $StderrLog -Tail 60 -ErrorAction SilentlyContinue
throw "API did not become healthy within $StartupTimeoutSeconds seconds."