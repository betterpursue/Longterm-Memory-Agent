"""
从 Judge JSON 中恢复 predictions（当 predictions JSON 被失败 run 覆盖时使用）。

用法:
    python eval/restore_from_judge.py \\
        --judge experiments/results/08_full_system_judge_v2.json \\
        --output experiments/results/08_full_system_restored.json

输出: 与 run_generation.py 输出格式一致的 predictions JSON。
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="从 Judge JSON 恢复 predictions（当原始 JSON 被覆盖时）"
    )
    parser.add_argument("--judge", required=True, help="Judge JSON 文件路径")
    parser.add_argument(
        "--output",
        default=None,
        help="输出路径（默认与 judge 同目录，文件名去掉 _judge 后缀）",
    )
    args = parser.parse_args()

    judge_path = Path(args.judge)
    if not judge_path.exists():
        print(f"[ERROR] Judge 文件不存在: {judge_path}")
        return

    with open(judge_path, encoding="utf-8") as f:
        judge_data = json.load(f)

    graded = judge_data.get("graded", [])
    if not graded:
        print(f"[ERROR] Judge JSON 中没有 graded 字段")
        return

    predictions = []
    for g in graded:
        entry = {
            "qa_id": g["qa_id"],
            "question": g["question"],
            "reference": g["reference"],
            "category": g["category"],
            "category_name": g["category_name"],
            "prediction": g.get("prediction", ""),
            "error": g.get("error"),
            "latency_sec": g.get("latency_sec", 0),
            "trace": g.get("trace"),
        }
        predictions.append(entry)

    # 默认输出路径：judge 同目录 + 去掉 _judge 后缀
    if args.output:
        out_path = Path(args.output)
    else:
        stem = judge_path.stem
        for suffix in ["_judge_v2", "_judge"]:
            stem = stem.replace(suffix, "")
        out_name = f"{stem}.json"
        # 避免覆盖原有文件
        if out_name == judge_path.name:
            out_name = f"{stem}_restored.json"
        out_path = judge_path.parent / out_name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    categories = {}
    for p in predictions:
        cat = p["category_name"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"从 {judge_path.name} 恢复了 {len(predictions)} 条 predictions")
    print(f"输出: {out_path}")
    print(f"分类: {categories}")
    print(f"错误: {sum(1 for p in predictions if p['error'])}")
    print(f"有效答案: {sum(1 for p in predictions if p.get('prediction'))}")


if __name__ == "__main__":
    main()
