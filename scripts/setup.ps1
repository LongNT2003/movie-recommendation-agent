$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir '.venv\Scripts\python.exe'

Push-Location $projectDir
try {
    if (-not (Test-Path -LiteralPath $python)) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment' }
    }
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install requirements' }
    & $python -m recommender.train
    if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
} finally {
    Pop-Location
}
