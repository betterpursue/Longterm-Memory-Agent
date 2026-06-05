#!/usr/bin/env bash
# 续跑阶段一（跳过已完成的 01）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/memory_agent"

# shellcheck source=/dev/null
[ -f "$ROOT/memory_agent/env.sh" ] && source "$ROOT/memory_agent/env.sh"
[ -f "$ROOT/memory_agent/env.example.sh" ] && source "$ROOT/memory_agent/env.example.sh"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
export INGEST_CACHE=1
export WRITER_CACHE_VERSION=v3
export LLM_MODEL="${LLM_MODEL:-/mnt/d/models/Qwen2.5-3B-AWQ}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export JUDGE_LLM_BASE_URL="${JUDGE_LLM_BASE_URL:-https://api.deepseek.com/v1}"
export JUDGE_LLM_MODEL="${JUDGE_LLM_MODEL:-deepseek-v4-flash}"

LOG="$ROOT/memory_agent/experiments/results/phase1_run.log"
EXPS=(02_full_context 03_vanilla_rag 04_dense_only 05_threefactor 06_no_forgetting 07_with_forgetting 08_full_system 09_forgetting_joint)

echo "=== Phase 1 resume $(date) ===" >> "$LOG"
for exp in "${EXPS[@]}"; do
  echo "--- [$exp] START $(date) ---" | tee -a "$LOG"
  if python eval/run_eval.py --full --experiment "$exp" >> "$LOG" 2>&1; then
    echo "--- [$exp] OK $(date) ---" | tee -a "$LOG"
  else
    echo "--- [$exp] FAIL $(date) ---" | tee -a "$LOG"
  fi
done

echo "=== Judge $(date) ===" | tee -a "$LOG"
python eval/batch_judge.py >> "$LOG" 2>&1
echo "=== Phase 1 complete $(date) ===" | tee -a "$LOG"
