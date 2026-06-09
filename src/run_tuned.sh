#!/bin/bash
PY="/c/Users/murar/jko-rag/.venv/Scripts/python.exe"
SRC="/c/Users/murar/jko-rag/src"
RES="/c/Users/murar/jko-rag/results"
CFG="$RES/best_hparams_scifact.json"

# Build a small wrapper script that uses the tuned config in run_full_dataset
# but routes the legacy scifact indices correctly.

# SciFact (uses legacy flat indices)
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" "$SRC/run_full_dataset_legacy.py" \
  --dataset scifact --split test --config-file "$CFG" --out-suffix "_tuned" \
  2>&1 | tee "$RES/stage1_scifact_tuned.log"
echo "SCIFACT_TUNED_DONE"

# NFCorpus
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" "$SRC/run_full_dataset.py" \
  --dataset nfcorpus --split test --config-file "$CFG" --out-suffix "_tuned" \
  2>&1 | tee "$RES/stage1_nfcorpus_tuned.log"
echo "NFCORPUS_TUNED_DONE"

# TREC-COVID
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" "$SRC/run_full_dataset.py" \
  --dataset trec-covid --split test --config-file "$CFG" --out-suffix "_tuned" \
  2>&1 | tee "$RES/stage1_trec_tuned.log"
echo "TREC_TUNED_DONE"
