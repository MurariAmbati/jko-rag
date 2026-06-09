# Sequential pipeline for all remaining JKO-RAG experiments.
# Runs: DPP (scifact, nfcorpus, fiqa) → BGE geometry (all) → SCIDOCS stability → SCIDOCS distractors → report.
# DPP starts after iter_retgen finishes; BGE geometry and SCIDOCS follow-ons start after SCIDOCS Stage 1.
# This script is safe to kill and resume: each step checks its output JSON before running.

Set-Location C:\Users\murar\jko-rag
$venv = ".venv\Scripts\python.exe"
$results = "results"

function Wait-File {
    param([string]$path, [int]$max_minutes = 300)
    $waited = 0
    while (-not (Test-Path $path)) {
        Start-Sleep -Seconds 60
        $waited++
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Waiting for $path ... ($waited min elapsed)"
        if ($waited -ge $max_minutes) {
            Write-Error "Timeout waiting for $path after $max_minutes minutes"; exit 1
        }
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Found $path, proceeding."
}

function Run-If-Missing {
    param([string]$output_json, [string]$log_path, [scriptblock]$cmd)
    if (Test-Path $output_json) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] SKIP: $output_json already exists."
        return
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] RUNNING -> log: $log_path"
    & {
        $now = Get-Date -Format 'HH:mm:ss'
        "=== START $now ===" | Tee-Object -FilePath $log_path -Append
        & $cmd 2>&1 | Tee-Object -FilePath $log_path -Append
        $now2 = Get-Date -Format 'HH:mm:ss'
        "=== END $now2 ===" | Tee-Object -FilePath $log_path -Append
    }
}

# -----------------------------------------------------------------------
# PHASE 1: Wait for iter_retgen, then run DPP experiments
# -----------------------------------------------------------------------
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Phase 1: Waiting for Iter-RetGen to complete..."
Wait-File "$results\iter_retgen_scifact.json" 120

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === DPP experiments ==="

if (-not (Test-Path "$results\dpp_scifact.json")) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] DPP SciFact (config: T=5/inner=40)..."
    & $venv src\dpp_retrieval.py --dataset scifact --split test `
        --config-file results\best_hparams_scifact.json `
        2>&1 | Tee-Object -FilePath "$results\dpp_scifact_v2.log"
} else { Write-Host "SKIP dpp_scifact.json (exists)" }

if (-not (Test-Path "$results\dpp_nfcorpus.json")) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] DPP NFCorpus (config: T=5/inner=40)..."
    & $venv src\dpp_retrieval.py --dataset nfcorpus --split test `
        --config-file results\best_hparams_nfcorpus.json `
        2>&1 | Tee-Object -FilePath "$results\dpp_nfcorpus_v2.log"
} else { Write-Host "SKIP dpp_nfcorpus.json (exists)" }

if (-not (Test-Path "$results\dpp_fiqa.json")) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] DPP FiQA (config: T=3/inner=15)..."
    & $venv src\dpp_retrieval.py --dataset fiqa --split test `
        --config-file results\best_hparams_fiqa.json `
        2>&1 | Tee-Object -FilePath "$results\dpp_fiqa_v2.log"
} else { Write-Host "SKIP dpp_fiqa.json (exists)" }

# -----------------------------------------------------------------------
# PHASE 2: Wait for SCIDOCS Stage 1, then run BGE geometry + SCIDOCS follow-ons
# -----------------------------------------------------------------------
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Phase 2: Waiting for SCIDOCS Stage 1 to complete..."
Wait-File "$results\stage1_scidocs_fast.json" 300

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === BGE geometry experiments (fast config T=2/inner=15) ==="

if (-not (Test-Path "$results\bge_geometry_scifact.json")) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BGE geometry SciFact..."
    & $venv src\run_bge_geometry.py --dataset scifact --split test `
        --config-file results\scidocs_cfg.json `
        2>&1 | Tee-Object -FilePath "$results\bge_geometry_scifact_v2.log"
} else { Write-Host "SKIP bge_geometry_scifact.json (exists)" }

if (-not (Test-Path "$results\bge_geometry_nfcorpus.json")) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BGE geometry NFCorpus..."
    & $venv src\run_bge_geometry.py --dataset nfcorpus --split test `
        --config-file results\scidocs_cfg.json `
        2>&1 | Tee-Object -FilePath "$results\bge_geometry_nfcorpus_v2.log"
} else { Write-Host "SKIP bge_geometry_nfcorpus.json (exists)" }

if (-not (Test-Path "$results\bge_geometry_fiqa.json")) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] BGE geometry FiQA..."
    & $venv src\run_bge_geometry.py --dataset fiqa --split test `
        --config-file results\scidocs_cfg.json `
        2>&1 | Tee-Object -FilePath "$results\bge_geometry_fiqa_v2.log"
} else { Write-Host "SKIP bge_geometry_fiqa.json (exists)" }

# -----------------------------------------------------------------------
# PHASE 3: SCIDOCS stability and distractors
# -----------------------------------------------------------------------
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === SCIDOCS stability (n=60 queries) ==="

if (-not (Test-Path "$results\stability_scidocs.json")) {
    & $venv src\run_stability_multi.py --dataset scidocs --n-queries 60 `
        2>&1 | Tee-Object -FilePath "$results\stability_scidocs.log"
} else { Write-Host "SKIP stability_scidocs.json (exists)" }

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === SCIDOCS distractors ==="

if (-not (Test-Path "$results\distractors_scidocs.json")) {
    & $venv src\run_distractors.py --dataset scidocs --n-queries 100 `
        --config-file results\scidocs_cfg.json `
        2>&1 | Tee-Object -FilePath "$results\distractors_scidocs.log"
} else { Write-Host "SKIP distractors_scidocs.json (exists)" }

# -----------------------------------------------------------------------
# PHASE 4: Final report
# -----------------------------------------------------------------------
Write-Host "[$(Get-Date -Format 'HH:mm:ss')] === Generating final report ==="
& $venv src\final_report.py 2>&1 | Tee-Object -FilePath "$results\final_report_regen.log"

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ===== ALL DONE ====="
