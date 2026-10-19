param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSeconds = 10
)

$ErrorActionPreference = "Stop"
$Health = Invoke-RestMethod -Uri "$($BaseUrl.TrimEnd('/'))/health" -TimeoutSec $TimeoutSeconds
if ($Health.status -ne "ok") {
    throw "Unexpected API health status: $($Health.status)"
}
$Health | ConvertTo-Json -Depth 4