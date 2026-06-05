"""
评测入口 — 一键运行所有实验。

用法：
    # 先确保 eval_set.json 已生成（在 eval_kit 目录下）
    # 然后从 project root 运行：
    python -m agent.controller ...          # 单次评测
    python eval/run_eval.py --quick         # 快速调试（5 条样本）
    python eval/run_eval.py --full          # 全集实验
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 确定项目根目录
_PROJ_ROOT = Path(__file__).resolve().parent.parent
_PROJ_PARENT = _PROJ_ROOT.parent
_EVAL_KIT = _PROJ_PARENT / "eval_kit"

# 默认评测集路径
_DEFAULT_EVAL_SET = str(_EVAL_KIT / "eval_set.json")

# 默认 CWD：项目根目录（memory_agent 和 eval_kit 的公共父目录）
_DEFAULT_CWD = str(_PROJ_PARENT)

# 实验配置
EXPERIMENTS = {
    # ---- 基线 ----
    "01_no_memory": {
        "agent": "memory_agent.agent.controller:NoMemoryAgent",
        "description": "Baseline: No-memory (无检索, LLM直接回答)",
        "env": {},
    },
    "02_full_context": {
        "agent": "agent_template:FullContextAgent",
        "description": "Baseline: Full-context (全部对话历史塞入 prompt)",
        "env": {},
    },
    "03_vanilla_rag": {
        "agent": "vanilla_rag_agent:VanillaRAGAgent",
        "description": "Baseline: Vanilla RAG (原始对话切片 + 稠密检索)",
        "env": {},
    },
    # ---- 方向 A 消融 ----
    "04_dense_only": {
        "agent": "memory_agent.agent.controller:DenseOnlyAgent",
        "description": "Ablation A1: 纯稠密检索 (Dense Only, no forgetting)",
        "env": {"ENABLE_FORGETTING": "0"},
    },
    "05_threefactor": {
        "agent": "memory_agent.agent.controller:ThreeFactorAgent",
        "description": "Ablation A2: 三因子检索 (ThreeFactor, no forgetting)",
        "env": {"ENABLE_FORGETTING": "0"},
    },
    # ---- 方向 B 消融 ----
    "06_no_forgetting": {
        "agent": "memory_agent.agent.controller:ThreeFactorAgent",
        "description": "Ablation B1: 无遗忘 (ThreeFactor, NoOpUpdater)",
        "env": {"ENABLE_FORGETTING": "0"},
    },
    "07_with_forgetting": {
        "agent": "memory_agent.agent.controller:ThreeFactorForgettingAgent",
        "description": "Ablation B2: 遗忘曲线 (ThreeFactor + Ebbinghaus)",
        "env": {"ENABLE_FORGETTING": "1"},
    },
    # ---- 主方案（Writer + 稠密检索，无遗忘） ----
    "08_full_system": {
        "agent": "memory_agent.agent.controller:DenseOnlyAgent",
        "description": "Primary: Writer + Dense retrieval (主评测路径)",
        "env": {"ENABLE_FORGETTING": "0"},
    },
    # ---- B2 联合消融（ForgettingRetriever，非主方案） ----
    "09_forgetting_joint": {
        "agent": "memory_agent.agent.controller:MyMemoryAgent",
        "description": "Ablation B2: 三因子 + 遗忘曲线 (ForgettingRetriever)",
        "env": {"ENABLE_FORGETTING": "1"},
    },
}


def run_experiment(exp_name: str, exp_cfg: dict, eval_set: str,
                   output_dir: str, limit: int = None):
    """运行一个实验配置。"""
    out_path = os.path.join(output_dir, f"{exp_name}.json")

    # 构建环境变量
    env = os.environ.copy()
    for k, v in exp_cfg.get("env", {}).items():
        env[k] = v
    # 确保离线模式，避免 HuggingFace Hub 受代理干扰
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    # 把项目根目录和 eval_kit 目录都加入 PYTHONPATH，
    # 这样 import agent_template / memory_agent.agent.controller 等都能找到
    _paths = os.pathsep.join([_DEFAULT_CWD, str(_EVAL_KIT)])
    env["PYTHONPATH"] = _paths + os.pathsep + env.get("PYTHONPATH", "")

    # 构建命令：用 python path/to/run_generation.py 而非 -m，避免包导入问题
    run_gen_py = os.path.join(_DEFAULT_CWD, "eval_kit", "run_generation.py")
    cmd = [
        sys.executable, run_gen_py,
        "--eval_set", eval_set,
        "--agent", exp_cfg["agent"],
        "--output", out_path,
    ]
    if limit is not None:
        cmd.extend(["--limit_conversations", str(limit)])

    print(f"\n{'='*60}")
    print(f"[{exp_name}] {exp_cfg['description']}")
    print(f"  Agent: {exp_cfg['agent']}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")

    t0 = time.time()
    # CWD 设到项目根目录（memory_agent 和 eval_kit 的公共父目录）
    result = subprocess.run(cmd, cwd=_DEFAULT_CWD, env=env,
                            capture_output=True, text=True)
    elapsed = time.time() - t0

    # 打印输出
    for line in result.stdout.split("\n"):
        print(f"  {line}")
    if result.returncode != 0:
        print(f"  [ERROR] Return code: {result.returncode}")
        if result.stderr:
            # 打印完整 stderr
            for line in result.stderr.split("\n"):
                print(f"  {line}")
        else:
            print("  (no stderr)")

    print(f"  Done in {elapsed:.1f}s")

    # 计算简单统计
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            preds = json.load(f)
        errors = [p for p in preds if p.get("error")]
        avg_latency = sum(p.get("latency_sec", 0) for p in preds) / len(preds) if preds else 0
        print(f"  Predictions: {len(preds)}, errors: {len(errors)}, "
              f"avg latency: {avg_latency:.2f}s")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run all memory agent experiments")
    parser.add_argument("--eval_set", default=_DEFAULT_EVAL_SET,
                        help=f"Path to eval_set.json (default: {_DEFAULT_EVAL_SET})")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for predictions (default: experiments/results)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: run only 2 conversations and 1 experiment")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: run all experiments")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit conversations per experiment")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run a single experiment by name")
    args = parser.parse_args()

    output_dir = args.output_dir or str(_PROJ_ROOT / "experiments" / "results")
    os.makedirs(output_dir, exist_ok=True)

    # 选择实验集
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            return
        experiments = {args.experiment: EXPERIMENTS[args.experiment]}
    elif args.quick:
        # 快速调试：Vanilla RAG 基线 + 主方案
        experiments = {
            "03_vanilla_rag": EXPERIMENTS["03_vanilla_rag"],
            "08_full_system": EXPERIMENTS["08_full_system"],
        }
    elif args.full:
        experiments = EXPERIMENTS
    else:
        # 默认：只跑关键实验
        experiments = {
            "03_vanilla_rag": EXPERIMENTS["03_vanilla_rag"],
            "04_dense_only": EXPERIMENTS["04_dense_only"],
            "05_threefactor": EXPERIMENTS["05_threefactor"],
            "08_full_system": EXPERIMENTS["08_full_system"],
            "07_with_forgetting": EXPERIMENTS["07_with_forgetting"],
            "09_forgetting_joint": EXPERIMENTS["09_forgetting_joint"],
        }

    if args.limit:
        limit = args.limit
    elif args.quick:
        limit = 2
    else:
        limit = None

    print(f"Evaluation set: {args.eval_set}")
    print(f"Experiments to run: {len(experiments)}")
    if limit:
        print(f"Conversation limit: {limit}")

    results = {}
    for exp_name, exp_cfg in experiments.items():
        out = run_experiment(exp_name, exp_cfg, args.eval_set,
                             output_dir, limit=limit)
        results[exp_name] = out

    print(f"\n{'='*60}")
    print("All experiments completed. Results saved to:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
