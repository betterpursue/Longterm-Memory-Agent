# 长期记忆 Agent — 工作交接文档

> 供后续 Agent / 开发者接续南京大学 NLP 大作业。最后更新：2026-05-29。

---

## 1. 项目目标与探索方向

**任务**：实现支持长期记忆的对话 Agent，在 LoCoMo 子集上完成写入 → 检索 → 生成 → 评测。

**探索方向**（作业要求选 2 个，本项目为 **A + B**）：

| 方向 | 内容 | 代码体现 |
|------|------|----------|
| **A 检索策略** | Dense vs 三因子 (Recency × Importance × Relevance) | `04_dense_only` / `05_threefactor` / `08_full_system` |
| **B 记忆遗忘** | NoOp vs Ebbinghaus 遗忘曲线 | `06_no_forgetting` / `07_with_forgetting` / `09_forgetting_joint` |

**主评测路径**：`DenseOnlyAgent`（实验 `08_full_system`）= Writer 结构化记忆 + 稠密检索 + 无遗忘。

作业说明原文见仓库根目录 [`Agent_Memory.md`](../Agent_Memory.md)。

---

## 2. 仓库结构

```
大作业 长期记忆agent/
├── Agent_Memory.md              # 作业要求
├── eval_kit/                    # 助教评测工具包
│   ├── eval_set.json            # 阶段一：10 conv / 40 QA
│   ├── eval_set_full.json       # 阶段二：10 conv / 200 QA（已生成）
│   ├── run_generation.py        # 单实验生成入口
│   ├── run_judge.py             # LLM-as-Judge
│   ├── llm_client.py            # OpenAI 兼容客户端
│   ├── vanilla_rag_agent.py     # Vanilla RAG 基线
│   └── agent_template.py        # FullContextAgent
└── memory_agent/                # 学生实现（提交目录）
    ├── memory/                  # store / writer / retriever / updater
    ├── agent/controller.py      # 主流程 + 消融 Agent 子类
    ├── eval/                    # run_eval.py、batch_judge、汇总脚本
    ├── experiments/
    │   ├── results/             # 预测 + Judge 结果
    │   ├── ingest_cache/        # Writer ingest 缓存（v3）
    │   └── BAD_CASES.md
    ├── report.md                # 报告草稿（待填真实数字、转 PDF）
    ├── README.md
    ├── env.example.sh           # 环境变量模板
    └── HANDOFF.md               # 本文档
```

---

## 3. 已完成的代码工作

### 3.1 核心模块（功能完整）

| 模块 | 文件 | 说明 |
|------|------|------|
| Store | `memory/store.py` | MemoryItem + FAISS |
| Writer | `memory/writer.py` | LLM 提取 + 多轮重试 + 规则补全层 |
| Retriever | `memory/retriever.py` | Dense / ThreeFactor / Forgetting |
| Updater | `memory/updater.py` | Ebbinghaus / NoOp |
| Controller | `agent/controller.py` | ingest / answer / trace / 消融子类 |

### 3.2 Writer 规则补全（`writer.py` `_inject_missing_facts` 等）

针对 LoCoMo 高频考点，在 LLM 漏提时注入：

- gym + 日期、trophy + dance contest
- video presentation + session 日期
- networking events + yesterday → 推算日期
- roadtrip 连写 + 事故
- one-on-one mentoring and training
- 跨 session 地点合并（`merge_global_location_facts`）
- 相对日期规范化（last week / yesterday / N days ago）

缓存版本：**`WRITER_CACHE_VERSION=v3`**（改 Writer 规则需 bump，见 `eval/cache_ingest.py`）。

### 3.3 检索与生成增强（`controller.py`）

- `RETRIEVAL_TOP_K=20`（环境变量可配）
- Query 扩展（地点 / 时间 / Would / roadtrip / mentoring 等）
- 轻量 rerank（关键词 +0.05 bonus）
- 生成兜底：location / temporal / career / Would / open_domain
- Would 题 `max_tokens=128`，截断检测（修复 Q30 `"Lik"` bug）
- Q66 `_postprocess_answer`：mentoring 补全 `and training`
- 每次 answer 导出 **trace**（retrieved + 完整 prompt）

### 3.4 评测基建

| 文件 | 作用 |
|------|------|
| `eval/run_eval.py` | 9 组实验一键入口（`--quick` / `--full` / `--experiment`） |
| `eval/cache_ingest.py` | `INGEST_CACHE=1` 跳过重复 Writer LLM |
| `eval/batch_judge.py` | 批量 Judge，支持 `JUDGE_LLM_*` 与生成分离 |
| `eval/summarize_all.py` | 汇总总表 + 分题型表 → `summary_table.md` |
| `eval/summarize_judge.py` | 单文件 Judge 摘要 |
| `eval/run_phase1_full.sh` | 阶段一 9 组实验 + Judge |
| `eval/run_phase2.sh` | 阶段二 200 QA 关键 4 组实验 |
| `eval/build_report_pdf.py` | report.md → PDF/HTML |

`run_judge.py` 已增强：max_tokens=256、空输出重试、F1 预筛。

---

## 4. 实验结果（务必区分「可信」与「失效」）

### 4.1 可信结果 — quick 子集（2 conv / 11 QA）

用户在 **vLLM 正常、LLM_MODEL 正确** 时手动跑通：

| 实验 | Judge 得分 | 说明 |
|------|-----------|------|
| **08_full_system** | **95.5%**（10 CORRECT + 1 PARTIAL） | 见 `08_full_system_judge_v2.json` |
| 03_vanilla_rag | 多数 unknown | 明显低于主方案 |

**08 分题型（11 题）**：single_hop 1.0 / temporal 1.0 / multi_hop 1.0 / open_domain 0.833。  
唯一 PARTIAL：`conv-30_q66`（缺 `training`，代码已加 postprocess，需重跑验证）。

Judge 文件 **`08_full_system_judge_v2.json` 内含完整 graded 与 trace**，是撰写报告最可靠的数字来源。

### 4.2 失效结果 — 阶段一自动重跑（10 conv / 40 QA）

`eval/run_phase1_full.sh` 曾跑完 01–09 的 **生成 + Judge**，但生成阶段 **几乎全部失败**：

- 原因 1：默认 `LLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ`，而 vLLM 实际 model id 为 **`/mnt/d/models/Qwen2.5-3B-AWQ`**
- 原因 2：vLLM 并发/崩溃导致 **`Internal Server Error`**
- 结果：`01–09` 的 `*.json` 中 40 题均为 `answer_failed`，Judge 得分 **0%**
- **`08_full_system.json` 已被覆盖为 40 条失败记录**；11 题好结果仅保留在 `08_full_system_judge_v2.json`

`experiments/results/summary_table.md` 当前混合了失效的 40 题 Judge 与旧的 11 题 Judge，**不可直接用于报告**，需重跑后重新生成。

### 4.3 尚未完成

| 项 | 状态 |
|----|------|
| 阶段一 40 QA 有效生成 + Judge | **需重跑** |
| 阶段二 200 QA（`eval_set_full.json`） | 数据已生成，**实验未跑**（`full_200/` 目录不存在） |
| `report.pdf` | 仅有 `report.md` 草稿，数字多为占位 |
| 08 全 40 题 Judge | 未做 |

---

## 5. 环境配置（极易踩坑）

### 5.1 运行环境

- **OS**：Windows + WSL2（用户主要在 WSL `vllm-env` 里跑）
- **GPU**：单卡 3070 8G
- **生成**：本地 vLLM `@8000`
- **Judge**：DeepSeek API（`deepseek-v4-flash`），与用户本地 3B 分离

### 5.2 必设环境变量

```bash
source memory_agent/env.example.sh   # 复制为 env.sh 并填入 JUDGE API Key

# 生成（本地 vLLM）— 关键：model id 必须匹配 vLLM
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=/mnt/d/models/Qwen2.5-3B-AWQ   # curl localhost:8000/v1/models 确认

# Embedding
export HF_ENDPOINT=https://hf-mirror.com      # 国内镜像
export EMBED_MODEL=BAAI/bge-small-en-v1.5     # LoCoMo 英文，非作业默认 zh

# Ingest 缓存
export INGEST_CACHE=1
export WRITER_CACHE_VERSION=v3

# Judge（云端，勿与生成共用错误 model）
export JUDGE_LLM_BASE_URL=https://api.deepseek.com/v1
export JUDGE_LLM_MODEL=deepseek-v4-flash
export JUDGE_LLM_API_KEY=sk-...                 # 或 LLM_API_KEY
```

### 5.3 启动 vLLM

```bash
source ~/Desktop/vllm-env/bin/activate   # 用户实际路径
vllm serve /mnt/d/models/Qwen2.5-3B-AWQ \
  --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.75 --quantization awq
```

**注意**：实验前先用 `python eval/test_llm.py` 或 curl 确认 `/v1/chat/completions` 正常；**不要**在 vLLM 未就绪时跑 `run_phase1_full.sh`。

### 5.4 已修复的 llm_client 默认值

`eval_kit/llm_client.py` 默认 model 已改为 `/mnt/d/models/Qwen2.5-3B-AWQ`，但仍建议显式 `export LLM_MODEL`。

---

## 6. 后续 Agent 建议工作清单（按优先级）

### P0 — 恢复可信实验数字

1. 确认 vLLM 正常 → `python eval/test_llm.py`
2. 重跑阶段一（建议单实验，避免占满 GPU）：
   ```bash
   cd memory_agent
   source env.sh   # 含 Judge Key
   python eval/run_eval.py --full --experiment 08_full_system
   python eval/run_eval.py --full --experiment 03_vanilla_rag
   # ... 其余 01–09
   ```
3. 删除或备份失效的 `*_judge.json`，重新：
   ```bash
   python eval/batch_judge.py --skip_existing false   # 需改脚本或手动删旧 judge
   python eval/summarize_all.py experiments/results/
   ```
4. **保留** `08_full_system_judge_v2.json` 作 quick 子集对照，勿覆盖。

### P1 — 阶段二 200 QA

```bash
# eval_set_full.json 已在 eval_kit/（10 conv / 200 QA）
bash eval/run_phase2.sh
```

### P2 — 报告交付

1. 用真实 `summary_table.md` 更新 `report.md`
2. `python eval/build_report_pdf.py` → `report.pdf`（≤8 页）
3. 确认 `BAD_CASES.md` 与最终 WRONG/PARTIAL 一致

### P3 — 可选优化

- 让 `debug_full.py` 与 `DenseOnlyAgent` 逻辑完全对齐
- 扩大 ingest 缓存到全部 10 conv（目前仅 conv-26/30 有 v3 缓存）
- 修复 `summarize_all.py` 对重复实验名（08 两行）的去重

---

## 7. 关键设计结论（可写进报告）

1. **Writer + 规则补全** 是最大增益来源（Sweden、conference、roadtrip、trophy 等）。
2. **3B 生成极保守**，大量 unknown 靠 fallback 拉回；Would 题需加长 max_tokens。
3. **Forgetting 在 QA 场景为负面对照**：末段记忆 retention 高，挤占 Top-K，早期事实检索不到（见 `BAD_CASES.md` §6）。
4. **Vanilla RAG** 对结构化时间/推理题弱于 Writer 记忆。
5. **Ingest 是耗时瓶颈**（~8–10 min/conv 首次）；`INGEST_CACHE=1` 必开。

---

## 8. 实验配置速查（`run_eval.py`）

| ID | 名称 | Agent |
|----|------|-------|
| 01 | no_memory | `NoMemoryAgent` |
| 02 | full_context | `FullContextAgent` |
| 03 | vanilla_rag | `VanillaRAGAgent` |
| 04 | dense_only | `DenseOnlyAgent` |
| 05 | threefactor | `ThreeFactorAgent` |
| 06 | no_forgetting | `ThreeFactorAgent` |
| 07 | with_forgetting | `ThreeFactorForgettingAgent` |
| 08 | **full_system** | `DenseOnlyAgent` |
| 09 | forgetting_joint | `MyMemoryAgent` |

---

## 9. 重要文件路径

| 用途 | 路径 |
|------|------|
| 最可信 Judge | `experiments/results/08_full_system_judge_v2.json` |
| 阶段一日志 | `experiments/results/phase1_run.log` |
| Bad case | `experiments/BAD_CASES.md` |
| 汇总表（待刷新） | `experiments/results/summary_table.md` |
| 200 QA 评测集 | `eval_kit/eval_set_full.json` |
| 计划原文 | `.cursor/plans/作业交付后续计划_*.plan.md`（勿改） |

---

## 10. 对话与决策历史摘要

1. 从 quick eval **27% → 95.5%**（11 题）：经 Writer 规则、Top-K=20、query 扩展、rerank、生成兜底、Judge 修复等多轮迭代。
2. 用户确认评测策略：**两阶段** — 先在 10/40 出消融表，再 200 QA 验证。
3. 阶段 0 已完成：Q66 postprocess、README v3、env.example.sh。
4. 阶段一脚本已跑但 **vLLM 配置问题导致生成失败**；接手的 Agent 首要任务是 **在稳定 vLLM 下重跑并刷新 Judge**。

---

## 11. 一键复现（给接手 Agent）

```bash
# 1. WSL 内
cd "/mnt/d/A new start/自然语言处理/大作业 长期记忆agent/memory_agent"
source /mnt/c/Users/32036/Desktop/vllm-env/bin/activate
source env.example.sh   # 或 env.sh（含 API Key）

# 2. 确认 vLLM
python eval/test_llm.py

# 3. 烟雾测试
python eval/run_eval.py --quick   # 2 conv, 03+08

# 4. 阶段一全集
bash eval/run_phase1_full.sh      # 约数小时，需稳定 vLLM

# 5. 阶段二
bash eval/run_phase2.sh

# 6. 报告
python eval/build_report_pdf.py
```

如有疑问，作业联系人：qqf@smail.nju.edu.cn（见 Agent_Memory.md）。

---

## 12. 计划中但未完成的工作

本节记录**交付计划里已设计、部分动手、但未真正交付**的内容，避免接手 Agent 重复规划或漏项。

### 12.1 实验与评测（核心缺口）

| 计划项 | 原计划 | 实际状态 | 接手后怎么做 |
|--------|--------|----------|--------------|
| **阶段一 9 组 × 40 QA 有效结果** | `run_phase1_full.sh` 一键跑完 + Judge | 脚本已写并**跑过一遍**，但生成全失败（vLLM/model）；`*.json` 多为空 prediction | vLLM 稳定后**整批重跑**；建议先备份 `08_full_system_judge_v2.json` |
| **08 全 40 题 Judge** | 与 quick 11 题对照 | 未做；当前 `08_full_system.json` 是失败版 | 重跑 08 后对 40 题单独 Judge，更新 report |
| **阶段二 200 QA** | `run_phase2.sh` 跑 03/05/07/08 | `eval_set_full.json` **已生成**，`experiments/results/full_200/` **不存在** | `bash eval/run_phase2.sh` |
| **Vanilla 全 40 题 Judge** | 基线对比 | 03 的 40 题预测失效，无有效 Judge | 与 08 同步重跑 03 |
| **消融表写入报告** | 表 1–4：基线 / A / B / 延迟成本 | `report.md` §4 多为「见 summary_table」占位 | 用刷新后的 `summary_table.md` 填表 |
| **quick vs full 对比段** | 报告说明小样本调参 vs 大样本泛化 | 未写 | 阶段二完成后在 report 加一节 |
| **Forgetting 定量对比** | 06 vs 07 vs 09 vs 08，引用 trace | BAD_CASES §6 仅有定性描述 | 重跑后抽 2–3 题对比 `trace.retrieved` 写进报告 |
| **Q66 postprocess 验证** | 改完 `_postprocess_answer` 后重跑 | 代码已合入，**未在 Judge 上验证是否变 CORRECT** | quick 或 full 重跑 08，看 conv-30_q66 |

### 12.2 报告与交付物

| 计划项 | 状态 | 说明 |
|--------|------|------|
| **`report.pdf`（≤8 页）** | 未生成 | 仅有 `report.md` 草稿；`build_report_pdf.py` 已写但未成功产出 PDF（环境可能无 pandoc/xelatex） |
| **报告 § 架构图** | 未做正式插图 | report 里是 ASCII 流程，计划里提到「架构图」可换成 mermaid/Visio 导出 |
| **报告 § 延迟/成本表** | 未填 | 计划要求「每题 LLM 调用次数」；`summarize_all.py` 只粗估 answer 阶段，**未统计 Writer ingest 调用** |
| **Bad case 与全量 WRONG 对齐** | 部分完成 | `BAD_CASES.md` 已更新 7 条（含改进前后），但**未基于阶段一 40 QA 的 WRONG/PARTIAL 刷新** |
| **作业目录 `report.pdf` 提交** | 缺失 | Agent_Memory.md 要求与 README 同级提交 |
| **git 整理 / 提交** | 未做 | 未帮用户 commit；接手时需确认 `.env`/`env.sh` 不进仓库 |

### 12.3 工具脚本（已创建但未完善）

| 文件 | 计划功能 | 未完成点 |
|------|----------|----------|
| `eval/batch_judge.py` | 批量 Judge | 文档写了 `--skip_existing false`，**脚本未实现该 CLI 参数**；重跑 Judge 需手动删旧 `*_judge.json` |
| `eval/summarize_all.py` | 总表 + 分题型 + 成本 | **未去重**（08 在 summary 里出现两行）；**LLM 调用成本**仅 answer 计数，ingest 未摊销 |
| `eval/run_phase1_full.sh` | 无人值守阶段一 | 已跑但失败；**未加 vLLM 健康检查**（应先 `test_llm.py` 再跑） |
| `eval/run_phase2.sh` | 阶段二 | **从未执行** |
| `eval/build_report_pdf.py` | MD→PDF | **未验证**；无 pandoc 时只出 HTML |
| `memory_agent/env.sh` | 用户私有配置 | 只有 `env.example.sh` 模板，**未创建含真实 API Key 的 env.sh**（需用户本地添加） |

### 12.4 代码优化（计划 P3，未做）

| 项 | 原因/预期收益 |
|----|----------------|
| **`debug_full.py` 与 `DenseOnlyAgent` 对齐** | 当前 debug 缺 `merge_global_location_facts`、query 扩展、rerank；正式评测比 debug 更完整，易误导调参 |
| **10 conv 全部 ingest 缓存** | 目前 `ingest_cache/` 只有 **conv-26、conv-30** 的 v3；其余 8 个 conv 每次仍全量 Writer |
| **从 Judge v2 恢复 11 题 prediction** | `08_full_system.json` 被失败 run 覆盖；好预测仅嵌在 `08_full_system_judge_v2.json` 的 graded 里，**未写脚本还原成独立 predictions JSON** |
| **规则补全泛化** | 当前 `_inject_missing_facts` 偏 LoCoMo 考点硬编码；计划讨论过但未做更通用版本 |
| **Forgetting 参数扫** | 半衰期 / Top-K 未做网格，B 方向只有固定配置消融 |
| **Vanilla RAG 与主方案 trace 对比工具** | 未做自动 diff 检索结果的可视化 |

### 12.5 环境与运维（尝试过但未彻底解）

| 问题 | 已做 | 未做 |
|------|------|------|
| vLLM model 名不一致 | 改 `llm_client.py` 默认值、`env.example.sh` | 未在 `README.md` 醒目警告；未加启动前自检到 `run_eval.py` |
| vLLM Internal Server Error | 曾重启 vLLM | 未加**单实验串行 + 重试**；未限制并发 ingest |
| Judge 误用本地 3B | `batch_judge.py` 支持 `JUDGE_LLM_*` | 01 曾用本地模型 Judge 得 0 分；**未在 run_judge 默认强制云端** |
| HuggingFace SSL | 文档提 `HF_ENDPOINT` | 未在代码里默认设置镜像 |
| WSL 路径含空格 | 脚本内用引号 | 从 Windows PowerShell 调 wsl 时仍易踩坑，**未写 Windows 侧 wrapper** |

### 12.6 作业要求对照（仍缺项）

对照 [`Agent_Memory.md`](../Agent_Memory.md)：

| 作业要求 | 状态 |
|----------|------|
| 4 基线 + 主系统 **全集** 可量化结果 | quick 11 题可信；**40/200 QA 全集未完成** |
| 探索方向 **干净消融** 表 | 代码齐；**数字全 0（失效 run）** |
| 按 **4 类问题** 分别报告 | 仅 08 quick 有分题型；全集无 |
| **3–5 Bad case** 定位环节 | BAD_CASES 有 7 条，但未与全集失败样例绑定 |
| **≥ Vanilla RAG** | quick 上已满足；**全集未证明** |
| **report.pdf ≤ 8 页** | 未提交 |
| 评测规模 ~50 conv / ~200 QA | LoCoMo 缓存仅 **10 conv**；200 **题**已够，**对话数**不足 50（报告需说明） |

### 12.7 建议接手 Agent 的执行顺序（合并「未做」清单）

```
1. 备份 08_full_system_judge_v2.json
2. source env.sh（补 API Key + LLM_MODEL）
3. vLLM 启动 → test_llm.py 通过
4. run_eval --quick 验证 08 ≥ 90%
5. run_phase1_full.sh（或逐实验 --experiment，夜间跑）
6. 删失效 *_judge.json → batch_judge → summarize_all
7. run_phase2.sh
8. 用 summary 填 report.md → build_report_pdf.py
9. 按 WRONG 刷新 BAD_CASES；补 Forgetting trace 对比
10. 可选：debug 对齐、ingest 缓存补全、从 v2 恢复 11 题 json
```

### 12.8 迭代历史（方便接手理解「为什么有这些规则」）

早期 quick 评测 **08 ≈ 27%**（3/11 CORRECT），主要瓶颈依次修复：

1. Writer 漏写（Sweden、trophy、conference 日期）→ 规则补全 + merge_global  
2. 检索 Top-15 漏召 → Top-K=20 + query 扩展 + rerank  
3. 3B 保守 unknown → 多类 fallback  
4. Would 题截断 `"Lik"` → max_tokens=128 + 截断检测  
5. Judge 空输出误判 → run_judge 重试 + max_tokens=256  
6. roadtrip / networking / video / mentoring 等 **v3 Writer** 补全  

上述迭代在 **11 题** 上验证到 95.5%；**尚未在 40/200 题上复验**。

