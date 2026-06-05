#!/usr/bin/env bash
# 阶段一完整重跑（修正 LLM_MODEL 后）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/memory_agent"

[ -f env.sh ] && source env.sh
source env.example.sh

LOG="$ROOT/memory_agent/experiments/results/phase1_run.log"
EXPS=(01_no_memory 02_full_context 03_vanilla_rag 04_dense_only 05_threefactor 06_no_forgetting 07_with_forgetting 08_full_system 09_forgetting_joint)

echo "=== Phase 1 FULL RERUN $(date) ===" | tee "$LOG"
for exp in "${EXPS[@]}"; do
  echo "--- [$exp] START $(date) ---" | tee -a "$LOG"
  if python eval/run_eval.py --full --experiment "$exp" >> "$LOG" 2>&1; then
    echo "--- [$exp] OK $(date) ---" | tee -a "$LOG"
  else
    echo "--- [$exp] FAIL $(date) ---" | tee -a "$LOG"
  fi
done

echo "=== Judge $(date) ===" | tee -a "$LOG"
python eval/batch_judge.py >> "$LOG" 2>&1 || true
python eval/summarize_all.py experiments/results/ >> "$LOG" 2>&1 || true
echo "=== Phase 1 COMPLETE $(date) ===" | tee -a "$LOG"
