"""
批量 Judge — 对 experiments/results/ 下所有 predictions JSON 跑 Judge。

用法：
    cd memory_agent
    python eval/batch_judge.py
    python eval/batch_judge.py --results_dir experiments/results
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent
_PARENT = _PROJ.parent
_EVAL_KIT = _PARENT / "eval_kit"
_RUN_JUDGE = _EVAL_KIT / "run_judge.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(_PROJ / "experiments" / "results"))
    parser.add_argument("--skip_existing", type=lambda x: x.lower() == "true",
                        default=True,
                        help="跳过已存在的 Judge 文件。设为 false 强制重跑。用法: --skip_existing false")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    preds = sorted(results_dir.glob("0*.json"))
    preds = [p for p in preds if "_judge" not in p.name and p.name != "smoke_test.json"]

    for pred in preds:
        out = results_dir / f"{pred.stem}_judge.json"
        if args.skip_existing and out.exists():
            print(f"[skip] {out.name} 已存在")
            continue
        print(f"[judge] {pred.name} -> {out.name}")
        cmd = [
            sys.executable, str(_RUN_JUDGE),
            "--predictions", str(pred),
            "--output", str(out),
        ]
        judge_url = os.environ.get("JUDGE_LLM_BASE_URL")
        judge_model = os.environ.get("JUDGE_LLM_MODEL", "deepseek-v4-flash")
        if judge_url:
            cmd.extend(["--judge_base_url", judge_url, "--judge_model", judge_model])
        env = os.environ.copy()
        if judge_url and os.environ.get("JUDGE_LLM_API_KEY"):
            env["LLM_API_KEY"] = os.environ["JUDGE_LLM_API_KEY"]
        r = subprocess.run(cmd, cwd=str(_EVAL_KIT), env=env)
        if r.returncode != 0:
            print(f"[ERROR] Judge failed for {pred.name}")
            sys.exit(r.returncode)

    # 汇总
    summarize = _PROJ / "eval" / "summarize_all.py"
    judge_files = sorted(results_dir.glob("*_judge*.json"))
    if judge_files and summarize.exists():
        subprocess.run(
            [sys.executable, str(summarize), str(results_dir)],
            cwd=str(_PROJ),
        )


if __name__ == "__main__":
    main()
