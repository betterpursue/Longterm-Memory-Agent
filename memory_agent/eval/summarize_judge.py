"""
汇总 Judge 结果 — 按题型打印表格。

用法：
    python eval/summarize_judge.py experiments/results/08_full_system_judge.json
"""

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_judge.py <judge_results.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    overall = data.get("overall", {})
    by_cat = data.get("by_category", {})
    print(f"\n=== {path.name} ===")
    print(f"Judge model: {data.get('judge_model', '?')}")
    print(f"{'类别':<14}{'题数':>5}{'得分':>9}{'正确':>6}{'部分':>6}{'错误':>6}")
    for name in sorted(by_cat.keys()):
        d = by_cat[name]
        print(f"{name:<14}{d['n']:>5}{d['score']:>9.3f}"
              f"{d.get('correct', 0):>6}{d.get('partial', 0):>6}{d.get('wrong', 0):>6}")
    print(f"{'总体':<14}{overall.get('n', 0):>5}{overall.get('score', 0):>9.3f}")
    print(f"平均延迟: {overall.get('avg_latency_sec', '?')}s\n")

    wrong = [g for g in data.get("graded", []) if g.get("judge_label") == "WRONG"]
    if wrong:
        print("WRONG 样例:")
        for g in wrong[:10]:
            print(f"  {g['qa_id']}: pred={g['prediction']!r} ref={g['reference']!r}")


if __name__ == "__main__":
    main()
