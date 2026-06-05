#!/usr/bin/env bash
# 阶段二：eval_set_full.json（10 conv / 200 QA）关键实验
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

EVAL_SET="$ROOT/eval_kit/eval_set_full.json"
LOG="$ROOT/memory_agent/experiments/results/phase2_run.log"

EXPS=(
  03_vanilla_rag
  05_threefactor
  07_with_forgetting
  08_full_system
)

echo "=== Phase 2 key experiments $(date) ===" | tee -a "$LOG"
echo "Eval set: $EVAL_SET" | tee -a "$LOG"

for exp in "${EXPS[@]}"; do
  echo "--- [$exp] $(date) ---" | tee -a "$LOG"
  python eval/run_eval.py --full --experiment "$exp" \
    --eval_set "$EVAL_SET" \
    --output_dir experiments/results/full_200 \
    2>&1 | tee -a "$LOG"
done

echo "=== Phase 2 Judge $(date) ===" | tee -a "$LOG"
for pred in experiments/results/full_200/0*.json; do
  [ -f "$pred" ] || continue
  [[ "$pred" == *"_judge"* ]] && continue
  base=$(basename "$pred" .json)
  out="experiments/results/full_200/${base}_judge.json"
  if [ -f "$out" ]; then
    echo "[skip] $out"
    continue
  fi
  python ../eval_kit/run_judge.py --predictions "$pred" --output "$out"
done

python eval/summarize_all.py experiments/results/full_200/

echo "=== Phase 2 Done $(date) ===" | tee -a "$LOG"
