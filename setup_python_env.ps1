Param(
    [string]$PythonDir = "",
    [string]$Workspace = "C:\code\learningExpts\learningExpts",
    [switch]$AddScripts = $true,
    [switch]$WhatIf = $false
)

function Write-Info($m) { Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Write-Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }

Write-Info "Starting setup_python_env.ps1 (WhatIf=$WhatIf)"

# Try to detect python if not provided
if ([string]::IsNullOrEmpty($PythonDir)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        $pythonExe = $cmd.Source
        $detected = Split-Path -Parent $pythonExe
        Write-Info "Detected python executable: $pythonExe"
        $PythonDir = $detected
    } else {
        Write-Warn "No 'python' found in PATH. You can pass -PythonDir or install Python first."
    }
}

if ($PythonDir) {
    # Detect if this looks like the Microsoft Store alias
    if ($PythonDir -like "*WindowsApps*" -or $PythonDir -match "AppExecutionAlias") {
        Write-Warn "Detected Python App Execution Alias (Microsoft Store). Please disable App execution aliases in Settings > Apps > Advanced app settings > App execution aliases and re-run."
    }

    $scriptsDir = Join-Path $PythonDir 'Scripts'

    # Read current user PATH
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    if (-not $userPath) { $userPath = "" }
    $parts = $userPath -split ';' | Where-Object { $_ -ne '' }

    $toAdd = @()
    if (-not ([string]::IsNullOrEmpty($PythonDir))) { $toAdd += $PythonDir }
    if ($AddScripts -and (Test-Path $scriptsDir)) { $toAdd += $scriptsDir }

    foreach ($p in $toAdd) {
        if (-not ($parts -contains $p)) {
            Write-Info "Will add to USER PATH: $p"
            $parts += $p
        } else {
            Write-Info "Already in PATH: $p"
        }
    }

    $newUserPath = ($parts -join ';')

    if ($WhatIf) {
        Write-Info "[WhatIf] New USER PATH would be:`n$newUserPath"
    } else {
        # setx has a length limit; warn if long
        if ($newUserPath.Length -gt 1024) { Write-Warn "Resulting PATH is long (>1024 chars); setx may fail or truncate." }
        setx PATH "$newUserPath" | Out-Null
        Write-Info "Updated USER PATH. Open a new terminal to see changes."
    }
} else {
    Write-Warn "PythonDir not provided and python not detected; skipping PATH update."
}

# Set PYTHONPATH to include workspace
$currentPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH','User')
if (-not $currentPythonPath) { $currentPythonPath = "" }

$entries = $currentPythonPath -split ';' | Where-Object { $_ -ne '' }
if (-not ($entries -contains $Workspace)) {
    $entries += $Workspace
    $newPythonPath = ($entries -join ';')
    if ($WhatIf) {
        Write-Info "[WhatIf] New PYTHONPATH would be:`n$newPythonPath"
    } else {
        setx PYTHONPATH "$newPythonPath" | Out-Null
        Write-Info "Updated PYTHONPATH to include: $Workspace"
    }
} else {
    Write-Info "Workspace already in PYTHONPATH: $Workspace"
}

Write-Info "Done. If you changed PATH or PYTHONPATH, please open a new terminal window to pick up changes."
