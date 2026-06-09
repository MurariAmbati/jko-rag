#!/usr/bin/env bash
# Sequential pipeline for remaining JKO-RAG experiments.
# - Idempotent: each step skips if its output JSON already exists.
# - Fail-soft: a single step's failure does NOT abort the pipeline.
# - Verbose: every action and timing is logged to results/pipeline_v2.log AND stdout.
# - Health-checked: validates Python + all configs/scripts before starting.
#
# Run with:   bash src/run_pipeline.sh
# Designed to be launched as a harness-tracked background task so it cannot die silently.

set -u   # unset vars = error, but NOT -e (we want soft-fail per step)

cd /c/Users/murar/jko-rag || { echo "FATAL: cannot cd to project root"; exit 99; }

PY=".venv/Scripts/python.exe"
PIPELINE_LOG="results/pipeline_v2.log"

ts() { date +"%H:%M:%S"; }
log() { local msg="[$(ts)] $1"; echo "$msg"; echo "$msg" >> "$PIPELINE_LOG"; }

# Per-step results (parallel arrays): names, statuses, durations
declare -a STEP_NAMES=()
declare -a STEP_STATUS=()
declare -a STEP_SECS=()

run_step() {
    # run_step <label> <output_json> <log_file> <cmd...>
    local label="$1"; local out_json="$2"; local log_file="$3"; shift 3
    STEP_NAMES+=("$label")

    if [[ -f "$out_json" ]]; then
        log "SKIP  $label  (output exists: $out_json)"
        STEP_STATUS+=("SKIP")
        STEP_SECS+=("0")
        return 0
    fi

    log "START $label  -> log: $log_file"
    local t0; t0=$(date +%s)
    "$@" > "$log_file" 2>&1
    local rc=$?
    local t1; t1=$(date +%s)
    local secs=$((t1 - t0))
    STEP_SECS+=("$secs")

    if [[ $rc -ne 0 ]]; then
        log "FAIL  $label  exit=$rc  duration=${secs}s  -- last 5 log lines:"
        tail -n 5 "$log_file" 2>/dev/null | sed 's/^/      /' | tee -a "$PIPELINE_LOG"
        STEP_STATUS+=("FAIL(exit=$rc)")
        return $rc
    fi

    if [[ ! -f "$out_json" ]]; then
        log "WARN  $label  exit=0 but $out_json NOT created  duration=${secs}s"
        STEP_STATUS+=("NO_JSON")
        return 1
    fi

    log "DONE  $label  duration=${secs}s  -> $out_json"
    STEP_STATUS+=("OK")
    return 0
}

# ---------- header ----------
echo "================================================================" > "$PIPELINE_LOG"
log "==== Pipeline v2 starting (pid $$) ===="

# ---------- pre-flight ----------
log "Pre-flight checks..."
if [[ ! -x "$PY" ]]; then
    log "FATAL: python interpreter not found at $PY"; exit 2
fi
log "  python: $($PY --version 2>&1)"

REQUIRED_FILES=(
    "src/dpp_retrieval.py"
    "src/run_bge_geometry.py"
    "src/run_stability_multi.py"
    "src/run_distractors.py"
    "src/final_report.py"
    "results/best_hparams_scifact.json"
    "results/best_hparams_nfcorpus.json"
    "results/best_hparams_fiqa.json"
    "results/scidocs_cfg.json"
    "indices/candidates_test.npz"
    "indices/nfcorpus/candidates_test.npz"
    "indices/fiqa/candidates_test.npz"
    "indices/scidocs/candidates_test.npz"
    "indices_bge/scifact/embeddings.npy"
    "indices_bge/nfcorpus/embeddings.npy"
    "indices_bge/fiqa/embeddings.npy"
)
missing=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        log "  MISSING: $f"
        missing=$((missing + 1))
    fi
done
if [[ $missing -gt 0 ]]; then
    log "FATAL: $missing required file(s) missing -- aborting"; exit 3
fi
log "  All $(echo "${#REQUIRED_FILES[@]}") required files present."

# ---------- PHASE 1: DPP experiments ----------
log "==== PHASE 1: DPP-MAP comparison ===="

run_step "DPP/scifact"  "results/dpp_scifact.json"  "results/dpp_scifact_v2.log" \
    "$PY" src/dpp_retrieval.py --dataset scifact  --split test \
    --config-file results/best_hparams_scifact.json

run_step "DPP/nfcorpus" "results/dpp_nfcorpus.json" "results/dpp_nfcorpus_v2.log" \
    "$PY" src/dpp_retrieval.py --dataset nfcorpus --split test \
    --config-file results/best_hparams_nfcorpus.json

run_step "DPP/fiqa"     "results/dpp_fiqa.json"     "results/dpp_fiqa_v2.log" \
    "$PY" src/dpp_retrieval.py --dataset fiqa     --split test \
    --config-file results/best_hparams_fiqa.json

# ---------- PHASE 2: BGE geometry ablation ----------
log "==== PHASE 2: BGE geometry ablation (fast T=2/inner=15) ===="

run_step "BGE/scifact"  "results/bge_geometry_scifact.json"  "results/bge_geometry_scifact_v2.log" \
    "$PY" src/run_bge_geometry.py --dataset scifact  --split test \
    --config-file results/scidocs_cfg.json

run_step "BGE/nfcorpus" "results/bge_geometry_nfcorpus.json" "results/bge_geometry_nfcorpus_v2.log" \
    "$PY" src/run_bge_geometry.py --dataset nfcorpus --split test \
    --config-file results/scidocs_cfg.json

run_step "BGE/fiqa"     "results/bge_geometry_fiqa.json"     "results/bge_geometry_fiqa_v2.log" \
    "$PY" src/run_bge_geometry.py --dataset fiqa     --split test \
    --config-file results/scidocs_cfg.json

# ---------- PHASE 3: SCIDOCS follow-ons ----------
log "==== PHASE 3: SCIDOCS stability + distractors ===="

run_step "SCIDOCS/stability" "results/stability_scidocs.json" "results/stability_scidocs.log" \
    "$PY" src/run_stability_multi.py --dataset scidocs --n-queries 60

run_step "SCIDOCS/distractors" "results/distractors_scidocs.json" "results/distractors_scidocs.log" \
    "$PY" src/run_distractors.py --dataset scidocs --n-queries 100 \
    --config-file results/scidocs_cfg.json

# ---------- PHASE 4: Final report ----------
log "==== PHASE 4: Regenerate REPORT.md ===="
"$PY" src/final_report.py > results/final_report_regen.log 2>&1
rc=$?
if [[ $rc -eq 0 ]]; then
    log "DONE  Final report  -> results/REPORT.md"
    STEP_NAMES+=("final_report"); STEP_STATUS+=("OK"); STEP_SECS+=("0")
else
    log "FAIL  Final report  exit=$rc"
    STEP_NAMES+=("final_report"); STEP_STATUS+=("FAIL"); STEP_SECS+=("0")
fi

# ---------- summary ----------
log ""
log "==== Pipeline v2 SUMMARY ===="
log "$(printf '%-25s %-15s %s' "STEP" "STATUS" "DURATION")"
for i in "${!STEP_NAMES[@]}"; do
    log "$(printf '%-25s %-15s %ss' "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}" "${STEP_SECS[$i]}")"
done
log "==== Pipeline v2 complete ===="
