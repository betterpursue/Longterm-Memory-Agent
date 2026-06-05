# 生成阶段（本地 vLLM）与 Judge 阶段（云端 API）环境变量示例
# 用法: source env.example.sh  （复制为 env.sh 并填入 API Key，勿提交 git）

# ---- 生成 LLM（本地 vLLM）----
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
# vLLM serve 的 model id（curl http://localhost:8000/v1/models 查看）
export LLM_MODEL="${LLM_MODEL:-/mnt/d/models/Qwen2.5-3B-AWQ}"

# ---- Embedding ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"

# ---- Ingest 缓存 ----
export INGEST_CACHE="${INGEST_CACHE:-1}"
export WRITER_CACHE_VERSION="${WRITER_CACHE_VERSION:-v3}"

# ---- Judge LLM（DeepSeek，与生成分离）----
export JUDGE_LLM_BASE_URL="${JUDGE_LLM_BASE_URL:-https://api.deepseek.com/v1}"
export JUDGE_LLM_MODEL="${JUDGE_LLM_MODEL:-deepseek-v4-flash}"
# export JUDGE_LLM_API_KEY="sk-..."  # 或使用 LLM_API_KEY 指向 DeepSeek
