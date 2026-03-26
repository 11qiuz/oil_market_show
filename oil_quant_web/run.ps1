param(
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "Installing Python dependencies (may take a while)..."
py -m pip install -r "$PSScriptRoot\backend\requirements.txt"

Write-Host "Starting server..."
py -m uvicorn backend.app:app --reload --port $Port

