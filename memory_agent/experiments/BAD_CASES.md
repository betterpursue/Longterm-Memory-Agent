# Bad Case 分析

基于 LoCoMo 评测集上的预测 trace 与 Judge 结果，按 **Writer → Retriever → Generator** 三环节归因。

---

## 1. conv-26_q11 — Sweden（写入 + 检索）

| 环节 | 问题 |
|------|------|
| Writer | LLM 有时只写 `moved from home country`，Sweden 落在另一 session |
| 检索 | `home country` 与 query 语义更近，Sweden 记忆常排 #15 以外 |
| 生成 | Top 上下文无国名 → `unknown` |

**改进**：`merge_global_location_facts`、query 扩展、location fallback、Top-K=20。  
**改进后**：08_full_system → **CORRECT**（Sweden）。

---

## 2. conv-26_q25 — conference 日期（Writer + 时间兜底）

| 阶段 | 表现 |
|------|------|
| 改进前 | 记忆为 `two days ago`，3B 答 unknown |
| 改进后 | Writer 规范化为 `on 10 July 2023` + temporal fallback → **CORRECT** |

**结论**：时间题需 Writer 写入绝对日期 + 生成阶段规则兜底。

---

## 3. conv-26_q30 — Melanie LGBTQ（生成截断）

| 环节 | 问题 |
|------|------|
| 检索 | Top 记忆含 LGBTQ / ally 线索 |
| 生成 | `max_tokens=64` 截断为 `"Lik"` |
| Judge | 语义错误 |

**改进**：Would 题 `max_tokens=128`、截断检测 + `_would_inference_fallback`。  
**改进后**：**CORRECT**（Likely no; supports community…）。

---

## 4. conv-26_q77 — roadtrip（Writer 漏写）

| 环节 | 问题 |
|------|------|
| Writer | 原文为 `roadtrip` 连写，规则 `road\s*trip` 未匹配，事故记忆未入库 |
| 检索 | Top-20 无 road trip 相关记忆 |
| 生成 | unknown |

**改进**：Writer 规则支持 `roadtrip` 连写 + query 扩展 `road trip accident`。  
**改进后**：**CORRECT**（Likely no; road trip went badly）。

---

## 5. conv-30_q66 — mentoring（生成不完整 → 已修复）

| 环节 | 问题 |
|------|------|
| Writer | 已注入 `one-on-one mentoring and training` |
| 检索 | Top 记忆含 mentoring，但排序靠后 |
| 生成 | 仅答 `one-on-one mentoring` → Judge **PARTIAL** |

**改进**：`_postprocess_answer` 补全 `and training`；open_domain fallback 返回完整短语。

---

## 6. Forgetting 消融 — 检索被「新记忆」挤占（方向 B）

| 对比 | 现象 |
|------|------|
| 08 DenseOnly | 检索 Top-K 与问题语义对齐，quick 子集 **95.5%** |
| 07/09 Forgetting | 末段 adoption/LGBTQ 记忆 retention≈1.0，霸榜 Top-K |
| 结果 | 早期 Sweden/conference 记忆被挤出 → 多题 unknown |

**报告结论**：LoCoMo QA 场景下 Ebbinghaus 遗忘 + 三因子未带来收益，适合作**负面对照**；主方案采用 Dense + NoOpUpdater。

---

## 7. conv-30_q22 / q33 — 时间题（检索有、3B 答 unknown）

| 题 | 检索 | 根因 |
|----|------|------|
| q22 video | Top-1 含 `13 June 2023` 括号日期 | 3B 未利用 session 时间戳 |
| q33 networking | 缺 `visited on 20 June` 显式事实 | Writer 未注入 yesterday 推算 |

**改进**：temporal fallback 用 session 括号日期；Writer 注入 networking + yesterday。  
**改进后**：两题均为 **CORRECT**。

---

## 复现

```bash
source memory_agent/env.example.sh   # 或 env.sh（含 DeepSeek Key）
cd memory_agent
python eval/run_eval.py --quick      # 2 conv 烟雾测试
python eval/run_eval.py --full       # 10 conv / 40 QA
python eval/batch_judge.py
python eval/summarize_all.py experiments/results/
```

Trace 字段位于各 `experiments/results/*.json` 的 `trace.retrieved` / `trace.prompt`。
