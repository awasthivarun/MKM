param(
    [switch]$IncludeNumba,
    [switch]$SkipFullModel
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $repoRoot "mkm_frumkin_performance_$stamp.txt"
$profiler = Join-Path $PSScriptRoot "profile_agpd_frumkin.py"

"=== Frumkin profiling started $(Get-Date -Format o) ===" | Tee-Object -FilePath $log
"Repository: $repoRoot" | Tee-Object -FilePath $log -Append
"Git revision: $(git rev-parse --short HEAD)" | Tee-Object -FilePath $log -Append
"Python: $(python --version)" | Tee-Object -FilePath $log -Append
"All stages are diagnostic only: no sampling, no source edits, no result overwrites." |
    Tee-Object -FilePath $log -Append

$stages = @(
    @{ Name = "Saved posterior chain diagnostics"; Args = @("stats", "--model", "c_ER") },
    @{ Name = "c_ER adaptive solver, posterior chain 0 (forward)"; Args = @(
        "coverage", "--model", "c_ER", "--source", "posterior", "--chain", "0", "--draw", "50",
        "--points", "1", "16", "492", "--draws", "10", "30", "50", "70", "90") },
    @{ Name = "c_ER adaptive solver, posterior chain 1 (forward)"; Args = @(
        "coverage", "--model", "c_ER", "--source", "posterior", "--chain", "1", "--draw", "50",
        "--points", "1", "16", "492", "--draws", "10", "30", "50", "70", "90") },
    @{ Name = "Adaptive solver physical parameter gradients"; Args = @(
        "coverage", "--model", "c_ER", "--source", "posterior", "--chain", "0", "--draw", "50",
        "--points", "1", "16", "--draws", "10", "50", "90", "--gradient") },
    @{ Name = "Old fixed 96-step solver comparison"; Args = @(
        "coverage", "--model", "c_ER", "--source", "posterior", "--chain", "0", "--draw", "50",
        "--solver", "fixed", "--points", "1", "16", "492", "--draws", "10", "30", "50", "70", "90") }
)

if (-not $SkipFullModel) {
    $stages += @(
        @{ Name = "Fully coupled prior draw 2 (adaptive)"; Args = @(
            "coverage", "--model", "cxo_CO_BF_ER_LH_3_pOH_sOH", "--source", "prior",
            "--chain", "0", "--draw", "2", "--points", "1", "16", "492") },
        @{ Name = "Fully coupled prior draw 30 (adaptive)"; Args = @(
            "coverage", "--model", "cxo_CO_BF_ER_LH_3_pOH_sOH", "--source", "prior",
            "--chain", "0", "--draw", "30", "--points", "16", "492") }
    )
}

$stages += @{ Name = "Full PyMC c_ER joint logp and gradient (FAST_RUN)"; Args = @(
    "logp", "--model", "c_ER", "--backend", "FAST_RUN", "--repeats", "3") }

if ($IncludeNumba) {
    $stages += @{ Name = "Experimental PyMC c_ER joint logp and gradient (NUMBA)"; Args = @(
        "logp", "--model", "c_ER", "--backend", "NUMBA", "--repeats", "2") }
}

$results = @()
for ($index = 0; $index -lt $stages.Count; $index++) {
    $stage = $stages[$index]
    $heading = "`n=== STAGE $($index + 1)/$($stages.Count): $($stage.Name) === $(Get-Date -Format o)"
    $heading | Tee-Object -FilePath $log -Append
    $start = Get-Date
    $stageArgs = [string[]]$stage.Args
    & python -u $profiler @stageArgs 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $seconds = [math]::Round(((Get-Date) - $start).TotalSeconds, 2)
    "=== STAGE RESULT: exit=$code, seconds=$seconds ===" | Tee-Object -FilePath $log -Append
    $results += [PSCustomObject]@{ Stage = $stage.Name; Exit = $code; Seconds = $seconds }
}

"`n=== FINAL STAGE SUMMARY ===" | Tee-Object -FilePath $log -Append
$results | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $log -Append
"Finished $(Get-Date -Format o). Full log: $log" | Tee-Object -FilePath $log -Append
