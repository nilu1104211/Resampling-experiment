# All dependencies are isolated from existing Conda environments.
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('pufmpp', 'spu-imr', 'ounet')]
    [string]$ModelName,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ModelArgs
)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:V3_PYTHON) { $env:V3_PYTHON } else { 'D:\Anaconda\envs\dianyun\python.exe' }
$dependencyPath = if ($env:V3_DEPENDENCIES) { $env:V3_DEPENDENCIES } else { 'D:\dkwork\scdata\network_adaptation\deps' }
$folders = @{ 'pufmpp'='Enhanced_PUFM-main'; 'spu-imr'='spu-imr-main'; 'ounet'='upsample-clean-master' }
$entryPath = Join-Path (Join-Path $PSScriptRoot $folders[$ModelName]) 'v3.py'
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$dependencyPath;$previousPythonPath"
    & $pythonPath -B $entryPath @ModelArgs
    if ($LASTEXITCODE -ne 0) { throw "V3 command failed with exit code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
