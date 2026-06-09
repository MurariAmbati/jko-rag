#!/bin/bash
# Wait for nfcorpus candidates file to exist, then run stage 1
# Then wait for fiqa, then run stage 1
PY="/c/Users/murar/jko-rag/.venv/Scripts/python.exe"
SRC="/c/Users/murar/jko-rag/src"
RES="/c/Users/murar/jko-rag/results"

# nfcorpus
until [ -f "/c/Users/murar/jko-rag/indices/nfcorpus/candidates_test.npz" ]; do
  sleep 30
done
echo "NFCORPUS_PRECOMPUTE_READY"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" "$SRC/run_full_dataset.py" --dataset nfcorpus --split test 2>&1 | tee "$RES/stage1_nfcorpus.log"
echo "NFCORPUS_STAGE1_DONE"

# fiqa
until [ -f "/c/Users/murar/jko-rag/indices/fiqa/candidates_test.npz" ]; do
  sleep 60
done
echo "FIQA_PRECOMPUTE_READY"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PY" "$SRC/run_full_dataset.py" --dataset fiqa --split test 2>&1 | tee "$RES/stage1_fiqa.log"
echo "FIQA_STAGE1_DONE"
