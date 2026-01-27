# ==========================================
# run_gpu_opt.ps1
# Compile + run GPU matmul and save results
# (robust to nvcc warnings on stderr)
# ==========================================

# Don't treat stderr text from native tools as terminating errors
$ErrorActionPreference = "Continue"

# Optional: switch console code page to UTF-8 to reduce encoding warnings
# (won't affect nvcc itself, but helps display)
try { chcp 65001 | Out-Null } catch {}

# Sizes to test
$Ns = @(256, 512, 1024, 2048, 4096)

# Paths
$src    = "..\src\matrix_gpu_with_optimize.cu"
$outDir = "..\results"
$exe    = "..\bin\matrix_gpu_with_optimize.exe"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $exe) | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $outDir "gpu_results_$timestamp.txt"

Write-Host "========================================"
Write-Host "Compiling GPU code..."
Write-Host "Source: $src"
Write-Host "Output: $exe"
Write-Host "Log:    $log"
Write-Host "========================================"

# ---- Compile (capture ALL output but only fail on exit code) ----
& nvcc -O3 $src -o $exe 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) {
    Write-Host "nvcc failed with exit code $LASTEXITCODE. See log: $log"
    exit $LASTEXITCODE
}

Write-Host "`n========================================" | Tee-Object -FilePath $log -Append
Write-Host "Running benchmarks..." | Tee-Object -FilePath $log -Append
Write-Host "========================================" | Tee-Object -FilePath $log -Append

foreach ($N in $Ns) {
    Write-Host "`n----------------------------------------" | Tee-Object -FilePath $log -Append
    Write-Host "Running GPU matrix multiplication, N=$N"  | Tee-Object -FilePath $log -Append
    Write-Host "----------------------------------------" | Tee-Object -FilePath $log -Append

    & $exe $N 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Program failed at N=$N with exit code $LASTEXITCODE. See log: $log"
        exit $LASTEXITCODE
    }
}

Write-Host "`nDone. Results saved to: $log"
