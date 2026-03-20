$ErrorActionPreference = "Stop"

param(
    [string]$Python = "python",
    [string]$VenvDir = ".venv"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot $VenvDir

Push-Location $repoRoot
try {
    & $Python -m venv $venvPath

    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        throw "Virtualenv python not found: $venvPython"
    }

    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
    & $venvPython -m pip install -e .

    Write-Host "Bootstrap complete."
    Write-Host "Use: $venvPython -m factorlab --help"
} finally {
    Pop-Location
}
