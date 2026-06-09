#!/usr/bin/env bash
# Pipeline for Directions D1a, D1b, D2, D3 (all 4 follow-ups to C1-C4).
# Idempotent + fail-soft.

set -u
cd /c/Users/murar/jko-rag

PY=".venv/Scripts/python.exe"
LOG="results/directions_pipeline.log"

ts() { date +"%H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "$LOG"; }

run_step() {
    local label="$1"; local out_json="$2"; local log_file="$3"; shift 3
    if [[ -f "$out_json" ]]; then log "SKIP $label"; return 0; fi
    log "START $label"
    local t0; t0=$(date +%s)
    "$@" > "$log_file" 2>&1
    local rc=$?
    local t1; t1=$(date +%s)
    local secs=$((t1 - t0))
    if [[ $rc -ne 0 ]]; then
        log "FAIL $label exit=$rc duration=${secs}s -- last 5 lines:"
        tail -n 5 "$log_file" | sed 's/^/      /' | tee -a "$LOG"
        return $rc
    fi
    log "DONE $label duration=${secs}s -> $out_json"
}

run_step_no_json() {
    # Same as run_step but checks file presence instead of JSON.
    local label="$1"; local check_path="$2"; local log_file="$3"; shift 3
    if [[ -f "$check_path" ]]; then log "SKIP $label"; return 0; fi
    log "START $label"
    local t0; t0=$(date +%s)
    "$@" > "$log_file" 2>&1
    local rc=$?
    local t1; t1=$(date +%s)
    local secs=$((t1 - t0))
    if [[ $rc -ne 0 ]]; then
        log "FAIL $label exit=$rc duration=${secs}s"
        tail -n 5 "$log_file" | sed 's/^/      /' | tee -a "$LOG"
        return $rc
    fi
    log "DONE $label duration=${secs}s -> $check_path"
}

echo "" > "$LOG"
log "==== Directions pipeline starting ===="

# === D2: SAM-JKO (already ran via bench launch -- check if json exists) ===
log "==== D2: SAM-JKO benchmark ===="
if [[ -f results/mr_jko_bench.json ]]; then
    log "SKIP mr-jko-bench (exists)"
else
    log "  Note: SAM-JKO benchmark needs to be launched separately"
fi

# === D3: end-to-end NM-JKO training for scifact/nfcorpus/fiqa ===
log "==== D3: end-to-end NM-JKO training ===="
for ds in scifact nfcorpus fiqa; do
    base="indices"
    [[ "$ds" != "scifact" ]] && base="indices/$ds"
    nt=80; [[ "$ds" != "scifact" ]] && nt=60
    run_step_no_json "E2E-train/$ds" "$base/learned_metric_e2e.pt" "results/e2e_train_${ds}.log" \
        "$PY" src/nm_jko_end2end.py --dataset "$ds" --n-train "$nt" --n-epochs 12 --T-unroll 1 --n-inner 6
done

# === D1b: DUAL-RANK selective coverage ===
log "==== D1b: DUAL-RANK selective coverage ===="
for ds in scifact nfcorpus fiqa scidocs; do
    run_step "dual-sel/$ds" "results/dual_selective_${ds}.json" "results/dual_selective_${ds}.log" \
        "$PY" src/dual_rank_selective.py --dataset "$ds" --split test \
        --config-file results/contrib_cfg.json
done

# === D1a: stability eval with new methods ===
log "==== D1a: stability eval with new methods ===="
for ds in scifact nfcorpus fiqa; do
    run_step "stability-new/$ds" "results/stability_new_${ds}.json" "results/stability_new_${ds}.log" \
        "$PY" src/run_stability_new.py --dataset "$ds" --n-queries 50
done

log "==== Directions pipeline complete ===="
