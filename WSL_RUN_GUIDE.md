# WSL 运行指南（快速复现实验）

## 概述

生成模型（Qwen2.5-3B-AWQ）通过 vLLM 在 WSL Ubuntu 中运行。  
Judge 评测通过云端 DeepSeek API（与生成模型分离）。  
Embedding 模型在 Windows Python 中 CPU 运行（~100MB）。

## 前置条件

- WSL Ubuntu-22.04 已启用、vllm-env 虚拟环境已创建
- DeepSeek API Key 已获取（platform.deepseek.com）
- 模型权重已本地下载（`/mnt/d/models/Qwen2.5-3B-AWQ`）

---

## 步骤 1：启动 WSL 和 vLLM

```bash
# 在 PowerShell（管理员）中启动 WSL
wsl --distribution Ubuntu-22.04

# 在 WSL 内部：
source ~/Desktop/vllm-env/bin/activate

# 启动 vLLM（调整 gpu-memory-utilization 按显存）
vllm serve /mnt/d/models/Qwen2.5-3B-AWQ \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.75 \
    --quantization awq

# 验证 vLLM 正常（新开一个 WSL 终端）
curl http://localhost:8000/v1/models
# 应返回模型列表，model id 为 /mnt/d/models/Qwen2.5-3B-AWQ
```

**常见问题**：
- OOM → 降低 `--gpu-memory-utilization` 为 0.6
- model id 不匹配 → `curl localhost:8000/v1/models` 确认，然后更新 `LLM_MODEL`

---

## 步骤 2：配置环境变量

```bash
# 在 WSL 内，进入项目目录
cd "/mnt/d/A new start/自然语言处理/大作业 长期记忆agent/memory_agent"

# 编辑 env.sh，填入 DeepSeek API Key
nano env.sh
# 将 JUDGE_LLM_API_KEY 改为自己的 key

# 加载环境变量
source env.sh

# 验证 LLM 可用
python eval/test_llm.py
```

---

## 步骤 3：烟雾测试（2 conv / 11 QA，~5 min）

```bash
cd "/mnt/d/A new start/自然语言处理/大作业 长期记忆agent/memory_agent"

# 运行主方案（08_full_system）+ Vanilla RAG 基线（03）
python eval/run_eval.py --quick
```

**预期结果**：08_full_system 得分 ~95.5%（参考 `08_full_system_judge_v2.json`）  
03_vanilla_rag 多数 unknown（3B 基线表现弱）

---

## 步骤 4：阶段一全集（9 实验 × 40 QA，~数小时）

### 方式 A：逐实验运行（推荐，避免 GPU 占满崩溃）

```bash
# 先备份已有的 v2 可信结果
cp experiments/results/08_full_system_judge_v2.json experiments/results/08_full_system_judge_v2.json.bak

# 逐个运行
python eval/run_eval.py --full --experiment 08_full_system
python eval/run_eval.py --full --experiment 04_dense_only
python eval/run_eval.py --full --experiment 05_threefactor
python eval/run_eval.py --full --experiment 06_no_forgetting
python eval/run_eval.py --full --experiment 07_with_forgetting
python eval/run_eval.py --full --experiment 09_forgetting_joint
python eval/run_eval.py --full --experiment 01_no_memory
python eval/run_eval.py --full --experiment 02_full_context
python eval/run_eval.py --full --experiment 03_vanilla_rag
```

### 方式 B：一键运行（含 Judge）

```bash
bash eval/run_phase1_full.sh
```

---

## 步骤 5：Judge + 汇总

```bash
cd "/mnt/d/A new start/自然语言处理/大作业 长期记忆agent/memory_agent"

# 对所有未评的 predictions 跑 Judge
python eval/batch_judge.py --skip_existing false

# 生成汇总表
python eval/summarize_all.py experiments/results/

# 查看汇总
cat experiments/results/summary_table.md
```

---

## 步骤 6：阶段二（200 QA）

```bash
bash eval/run_phase2.sh
```

---

## 步骤 7：生成报告

```bash
python eval/build_report_pdf.py
# → 生成 report.html（浏览器打开 → 打印 → 另存 PDF）
# 若安装了 pandoc+xelatex，自动生成 report.pdf
```

---

## 步骤 8：恢复 11 题可信预测（如被覆盖）

若 `08_full_system.json` 被失败 run 覆盖：

```bash
python eval/restore_from_judge.py \
    --judge experiments/results/08_full_system_judge_v2.json \
    --output experiments/results/08_full_system_restored.json
```

---

## 关键注意事项

| 问题 | 说明 |
|------|------|
| **LLM_MODEL 必须准确** | 用 `curl localhost:8000/v1/models` 确认 vLLM 暴露的 model id |
| **Judge 与生成分离** | Judge 用 DeepSeek（云端），生成用本地 3B；勿混淆 |
| **INGEST_CACHE 必开** | 省 ~8-10 min/conv；改 Writer 规则后 bump `WRITER_CACHE_VERSION` |
| **单实验串行** | 不要同时跑多个 vLLM 实验，8G 显存会 OOM |
| **WSL 路径空格** | 项目路径含中文和空格，在 WSL 内用引号包裹 |
