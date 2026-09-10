#!/usr/bin/env python3
"""统一的 YOLO 原始输出 ONNX 推理入口。

脚本根据输出节点自动识别检测或实例分割任务，并兼容 P2-P5 与 P3-P5 检测头。
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def configure_repo_imports() -> None:
    """确保脚本方式运行时也能导入仓库内的 pipeline 包。"""
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def detect_task(model_path: str) -> str:
    """根据 ONNX 输出节点判断 detect 或 segment。"""
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    output_names = {output.name for output in model.graph.output}
    if "proto" in output_names and any(name.startswith("mask_coeff_p") for name in output_names):
        return "segment"
    if any(name.startswith("box_p") for name in output_names) and any(
        name.startswith("score_p") for name in output_names
    ):
        return "detect"
    raise ValueError(f"无法识别 ONNX 输出结构: {sorted(output_names)}")


def main() -> None:
    """识别模型任务并转交统一入口对应的内部实现。"""
    configure_repo_imports()
    if "--help" in sys.argv and "--onnx-model" not in sys.argv:
        print(
            "用法: python tools/inference.py --onnx-model MODEL [任务参数]\n\n"
            "自动支持 detect/segment 和 P2-P5/P3-P5 原始输出 ONNX。\n"
            "检测常用参数: --img-dir、--yaml、--output-dir、--manifest。\n"
            "分割常用参数: --single、--eval、--output、--manifest、--pidnet-onnx。\n"
            "使用具体模型加 --help 可查看对应任务的完整参数。"
        )
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--onnx-model", required=True)
    known, _ = parser.parse_known_args()
    task = detect_task(known.onnx_model)
    print(f"[ONNX] 自动识别任务: {task}")
    module = "pipeline.workers.onnx_segment" if task == "segment" else "pipeline.workers.onnx_detect"
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
