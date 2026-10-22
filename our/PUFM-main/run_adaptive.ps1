param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ModelArgs)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:ADAPTIVE_PYTHON) { $env:ADAPTIVE_PYTHON } else { 'D:\Anaconda\envs\pufm\python.exe' }
& $pythonPath -B (Join-Path $PSScriptRoot 'adaptive_v3.py') @ModelArgs
if ($LASTEXITCODE -ne 0) { throw "Adaptive PUFM command failed with exit code $LASTEXITCODE" }
