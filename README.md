# 长期记忆对话 Agent — MemoryAgent

基于 LoCoMo 评测集的长期记忆对话 Agent 实验项目。
探索方向：**A（检索策略）** + **B（记忆遗忘）**。

## 项目结构

```
memory_agent/
├── memory/
│   ├── store.py           # 记忆存储与 FAISS 索引
│   ├── writer.py          # LLM 记忆提取
│   ├── retriever.py       # 检索策略（稠密 / 三因子 / 带遗忘三因子）
│   └── updater.py         # 遗忘曲线（Ebbinghaus / NoOp）
├── agent/
│   └── controller.py      # 主流程编排 + 消融子类
├── eval/
│   └── run_eval.py        # 一键评测入口
├── experiments/
│   └── results/           # 各组实验结果 JSON
├── README.md
└── requirements.txt
```

## 环境要求

- Python 3.10+
- 单卡 NVIDIA 3070 8G（或兼容 GPU）
- 推荐系统：Ubuntu 22.04 / Windows 11

## 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 部署生成模型（本地 vLLM）
建议使用虚拟环境来配置：

# 创建一个名为 vllm-env 的虚拟环境
python3 -m venv vllm-env

# 激活虚拟环境
source vllm-env/bin/activate

#设置国内镜像
export HF_ENDPOINT="https://hf-mirror.com"

vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75

若使用阿里魔搭下载到本地，则可以使用命令：
vllm serve /mnt/d/models/Qwen2.5-3B-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75

若出现OOM问题，可考虑降低swap分配空间:
vllm serve /mnt/d/models/Qwen2.5-3B-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --swap-space 1


注意如果使用该方法，需要下载相关的包：
pip install modelscope

若以上命令无法成功，考虑降级算子
vllm serve /mnt/d/models/Qwen2.5-3B-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --quantization awq

若出现显存不足，则可以降低gpu-memory-utilization参数数值
vllm serve /mnt/d/models/Qwen2.5-3B-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6 \
    --quantization awq

# 3. 配置 Judge 云端 API（评测用）
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_API_KEY="sk-your-key"
export LLM_MODEL="deepseek-v4-flash"
```

## 使用云服务器
如果选择使用云服务器进行模型部署，可以按照以下步骤实现：
1.首先找到云服务器的远程链接ssh与password，在本地shell终端输入ssh和password进行链接
2.建立本地到云服务器的channel：
```bash
在代码文件夹目录新建一个终端，输入以下命令：
这行命令的意思是：把你本地的 8000 端口，映射到云服务器的 localhost:8000 上
ssh -N -L 8000:localhost:8000 root@你的云服务器IP -p 你的高级SSH端口

```
3.在云服务器部署大语言模型
```bash
注意要设置国内镜像，否则容易连接不上huggingface
#设置国内镜像(注意该命令只在当前会话生效，每次开启终端都需要重新设置一次)
export HF_ENDPOINT="https://hf-mirror.com"

vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75
```

## 快速开始

### 1. 准备评测集

```bash
cd eval_kit
python prepare_eval_set.py --output eval_set.json --per_category 10 --seed 42
```

### 2. 单次评测（烟雾测试）

```bash
cd memory_agent
python -m agent.controller ...   # 见下方详细命令

# 或用 run_generation.py 跑单组实验：
python ../eval_kit/run_generation.py \
    --eval_set ../eval_kit/eval_set.json \
    --agent agent.controller:DenseOnlyAgent \
    --output experiments/results/test.json \
    --limit_conversations 2
```

如果出现网络连不上 HuggingFace的情况，可以选择以下方案：
方案一：用镜像源（推荐）
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 eval_kit/run_generation.py \
    --eval_set eval_kit/eval_set.json \
    --agent memory_agent.agent.controller:MyMemoryAgent \
    --output memory_agent/experiments/results/smoke_test.json \
    --limit_conversations 1

```

先把镜像源设成环境变量，再跑。bge-small-en-v1.5 只有 ~100MB，一般一分钟内下完。

方案二：如果还是不行，手动下载
从镜像下载模型到本地
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir ./models/bge-small-en-v1.5
```

然后指定本地路径跑
```bash
export EMBED_MODEL=./models/bge-small-en-v1.5
python3 eval_kit/run_generation.py \
    --eval_set eval_kit/eval_set.json \
    --agent memory_agent.agent.controller:MyMemoryAgent \
    --output memory_agent/experiments/results/smoke_test.json \
    --limit_conversations 1
```
### 3. 一键运行所有实验

```bash
cd memory_agent
python eval/run_eval.py --quick          # 快速：2 个对话
python eval/run_eval.py                  # 默认：关键实验
python eval/run_eval.py --full           # 全集：所有实验
```

### 4. Judge 评测

```bash
cd eval_kit
python run_judge.py \
    --predictions ../memory_agent/experiments/results/08_full_system.json \
    --output ../memory_agent/experiments/results/08_full_system_judged.json
```

## 实验清单

| # | 实验名 | Agent 类 | 检索策略 | 遗忘 | 说明 |
|---|--------|---------|---------|------|------|
| 基线 | | | | | |
| 01 | no_memory | — | — | — | 仅 query（由 DenseOnlyAgent 在没有记忆时回退） |
| 02 | full_context | FullContextAgent | — | — | 全部历史塞入 prompt |
| 03 | vanilla_rag | VanillaRAGAgent | 稠密 | ✗ | 原始对话切片检索 |
| 方向 A 消融 | | | | | |
| 04 | dense_only | DenseOnlyAgent | 稠密 | ✗ | 记忆提取 + 稠密检索 |
| 05 | threefactor | ThreeFactorAgent | 三因子 | ✗ | 记忆提取 + 三因子检索 |
| 方向 B 消融 | | | | | |
| 06 | no_forgetting | ThreeFactorAgent | 三因子 | ✗ | 无遗忘 |
| 07 | with_forgetting | ThreeFactorForgettingAgent | 三因子 | ✔ | B2：Ebbinghaus 遗忘（三因子检索） |
| 主方案 | | | | | |
| 08 | full_system | DenseOnlyAgent | 稠密 | ✗ | **主评测路径**：Writer + Dense |
| B2 联合 | | | | | |
| 09 | forgetting_joint | MyMemoryAgent | 三因子+遗忘 | ✔ | B2 消融：ForgettingRetriever（非主方案） |

## 自定义配置

通过环境变量配置：

| 变量 | 默认值 | 说明 |
|------|-------|------|
| `RETRIEVAL_TOP_K` | 20 | 检索返回的记忆数 |
| `INGEST_CACHE` | 1 | 是否缓存 Writer ingest 结果（`experiments/ingest_cache/`） |
| `WRITER_CACHE_VERSION` | v3 | bump 后使 ingest 缓存失效 |
| `RETRIEVER_ALPHA` | 0.7 | 三因子：Relevance 权重 |
| `RETRIEVER_BETA` | 0.15 | 三因子：Recency 权重 |
| `RETRIEVER_GAMMA` | 0.15 | 三因子：Importance 权重 |
| `RECENCY_HALFLIFE_HOURS` | 2160 | Recency 半衰期（小时，90 天） |
| `ENABLE_FORGETTING` | 1 | 是否启用遗忘曲线 |
| `EMBED_MODEL` | BAAI/bge-small-en-v1.5 | Embedding 模型（LoCoMo 为英文语料，用 en 版；） |

## 参考

- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior", UIST 2023
- Zhong et al., "MemoryBank: Enhancing Large Language Models with Long-Term Memory", AAAI 2024
- Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents", ACL 2024
- Chhikara et al., "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory", ECAI 2025
