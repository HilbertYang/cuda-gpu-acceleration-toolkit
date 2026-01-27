# ==========================================
# run_cpu.ps1
# Run CPU matrix multiplication benchmarks
# ==========================================

$Ns = @(256, 512, 1024, 2048)

$exe = "..\bin\matmul_cpu.exe"
$resultFile = "..\results\cpu_results.txt"

# Check executable exists
if (!(Test-Path $exe)) {
    Write-Host "ERROR: matmul_cpu.exe not found in bin/"
    exit 1
}

# Write header once
"CPU Matrix Multiplication Results" | Out-File $resultFile
"=================================" | Out-File $resultFile -Append

foreach ($N in $Ns) {
    Write-Host "----------------------------------------"
    Write-Host "Running CPU matrix multiplication, N=$N"
    Write-Host "----------------------------------------"

    # Run program and save output
    & $exe $N | Tee-Object -Append $resultFile
}
