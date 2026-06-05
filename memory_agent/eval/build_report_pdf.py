#!/usr/bin/env python3
"""将 report.md 转为 report.pdf（需 pandoc + xelatex，或降级为 HTML）。"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "report.md"
PDF = ROOT / "report.pdf"
HTML = ROOT / "report.html"


def main():
    if not MD.exists():
        print(f"缺少 {MD}")
        sys.exit(1)

    if shutil.which("pandoc"):
        cmd = [
            "pandoc", str(MD), "-o", str(PDF),
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=2cm",
            "-V", "CJKmainfont=Noto Sans CJK SC",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"已生成 {PDF}")
            return
        print("pandoc PDF 失败，尝试 HTML:", r.stderr[:200])

    # 降级：简单 HTML，可浏览器打印为 PDF
    text = MD.read_text(encoding="utf-8")
    body = "<pre style='font-family:sans-serif;white-space:pre-wrap;line-height:1.5'>"
    body += text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    body += "</pre>"
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>长期记忆 Agent 报告</title></head><body>{body}</body></html>"
    HTML.write_text(html, encoding="utf-8")
    print(f"已生成 {HTML}（浏览器打开 → 打印 → 另存为 PDF）")
    # 若 wkhtmltopdf 可用
    if shutil.which("wkhtmltopdf"):
        subprocess.run(["wkhtmltopdf", str(HTML), str(PDF)], check=False)
        if PDF.exists():
            print(f"已生成 {PDF}")


if __name__ == "__main__":
    main()
