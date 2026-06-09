#!/usr/bin/env bash
# Pipeline for the 4 new method contributions (C1 NM-JKO, C2 BW-JKO,
# C3 DUAL-RANK, C4 MR-JKO).  Idempotent.

set -u
cd /c/Users/murar/jko-rag

PY=".venv/Scripts/python.exe"
LOG="results/contrib_pipeline.log"

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

echo "" > "$LOG"
log "==== Contributions pipeline starting ===="

# PHASE A: train learned metrics (if not already trained)
log "==== Phase A: training learned metrics ===="
for ds in scifact nfcorpus fiqa; do
    base="indices"
    [[ "$ds" != "scifact" ]] && base="indices/$ds"
    if [[ -f "$base/learned_metric.pt" ]]; then
        log "SKIP NM-train/$ds (metric exists)"
    else
        nt=80; [[ "$ds" != "scifact" ]] && nt=60
        run_step "NM-train/$ds" "$base/learned_metric.pt" "results/nm_train_${ds}.log" \
            "$PY" src/learned_metric.py --dataset "$ds" --split train --r 64 --n-train "$nt" --n-epochs 150
    fi
done

# PHASE B: contributions evaluation (per dataset)
log "==== Phase B: contributions evaluation ===="
run_step "contrib/scifact"  "results/contrib_scifact.json"  "results/contrib_scifact.log" \
    "$PY" src/run_contributions.py --dataset scifact  --split test \
    --config-file results/contrib_cfg.json

run_step "contrib/nfcorpus" "results/contrib_nfcorpus.json" "results/contrib_nfcorpus.log" \
    "$PY" src/run_contributions.py --dataset nfcorpus --split test \
    --config-file results/contrib_cfg.json

run_step "contrib/fiqa"     "results/contrib_fiqa.json"     "results/contrib_fiqa.log" \
    "$PY" src/run_contributions.py --dataset fiqa     --split test \
    --config-file results/contrib_cfg.json

# SCIDOCS has no train split so no learned metric -- run with --no-nm
run_step "contrib/scidocs"  "results/contrib_scidocs.json"  "results/contrib_scidocs.log" \
    "$PY" src/run_contributions.py --dataset scidocs  --split test \
    --config-file results/contrib_cfg.json --no-nm

# PHASE C: MR-JKO benchmark (synthetic + scifact at M=200)
log "==== Phase C: MR-JKO benchmark ===="
run_step "mr-jko/bench" "results/mr_jko_bench.json" "results/mr_jko_bench.log" \
    "$PY" src/run_mr_jko_bench.py

log "==== Contributions pipeline complete ===="
