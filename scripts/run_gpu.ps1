# scripts/run_gpu.ps1
$Ns = @(256, 512, 1024, 2048)
$exe = "..\src\matrix_gpu.exe"

New-Item -ItemType Directory -Force -Path "..\results" | Out-Null
$out = "..\results\gpu_results.txt"

"==== GPU results $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====" | Out-File -FilePath $out -Append

foreach ($N in $Ns) {
  Write-Host "Running GPU, N=$N"
  & $exe $N | Tee-Object -FilePath $out -Append
}
"" | Out-File -FilePath $out -Append
