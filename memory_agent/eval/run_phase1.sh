#!/usr/bin/env bash
# 阶段一：10 conv / 40 QA 全部 9 组实验
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/memory_agent"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
export INGEST_CACHE=1
export WRITER_CACHE_VERSION=v3
export LLM_MODEL="${LLM_MODEL:-/mnt/d/models/Qwen2.5-3B-AWQ}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export JUDGE_LLM_BASE_URL="${JUDGE_LLM_BASE_URL:-https://api.deepseek.com/v1}"
export JUDGE_LLM_MODEL="${JUDGE_LLM_MODEL:-deepseek-v4-flash}"

EXPS=(
  01_no_memory
  02_full_context
  03_vanilla_rag
  04_dense_only
  05_threefactor
  06_no_forgetting
  07_with_forgetting
  08_full_system
  09_forgetting_joint
)

LOG="$ROOT/memory_agent/experiments/results/phase1_run.log"
echo "=== Phase 1 full experiments $(date) ===" | tee -a "$LOG"

for exp in "${EXPS[@]}"; do
  echo "--- [$exp] $(date) ---" | tee -a "$LOG"
  python eval/run_eval.py --full --experiment "$exp" 2>&1 | tee -a "$LOG"
done

echo "=== Judge batch $(date) ===" | tee -a "$LOG"
python eval/batch_judge.py 2>&1 | tee -a "$LOG"

echo "=== Done $(date) ===" | tee -a "$LOG"
