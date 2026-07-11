# Loads credentials from .env and configures the shell for Modal on Windows.
# Usage:  . .\load-env.ps1   (dot-source so the variables persist in your session)

$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "No .env file found at $envFile"
    return
}

Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $key, $value = $line.Split("=", 2)
        $key = $key.Trim()
        $value = $value.Trim().Trim('"')
        Set-Item -Path ("Env:" + $key) -Value $value
    }
}

# UTF-8 so Modal's progress output (checkmarks etc.) does not crash the console.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host "Loaded Modal + HF credentials from .env (UTF-8 console enabled)."
if ($env:MODAL_TOKEN_ID) {
    Write-Host ("MODAL_TOKEN_ID = " + $env:MODAL_TOKEN_ID.Substring(0, [Math]::Min(8, $env:MODAL_TOKEN_ID.Length)) + "...")
}
