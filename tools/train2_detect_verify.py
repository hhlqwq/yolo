"""独立训练、验证、导出和测试 liquid/debris 两类检测模型。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export import export_yolov11_detect_p2  # noqa: E402
from ultralytics import YOLO  # noqa: E402


DEFAULT_DATA_YAML = Path(
    "/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/"
    "算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/"
    "模型训练/experiments/train2_detect.yaml"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/"
    "算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/"
    "模型训练/experiments/liquid_debris_verify_20260902"
)
DEFAULT_INITIAL_WEIGHT = Path(
    "/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/"
    "算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/"
    "模型训练/V021_20260901_add0901/runs/yolo_detect_p2/train/weights/best.pt"
)
EXPECTED_CLASS_NAMES = {0: "liquid", 1: "debris"}


def parse_args() -> argparse.Namespace:
    """解析独立实验的数据、权重、输出和训练测试参数。"""
    parser = argparse.ArgumentParser(
        description="独立完成 liquid/debris 两类检测训练、验证、ONNX 导出和测试。"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=REPO_ROOT / "ultralytics/cfg/models/11/yolo11s_p2.yaml",
    )
    parser.add_argument("--initial-weight", type=Path, default=DEFAULT_INITIAL_WEIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def _normalize_names(value: Any) -> dict[int, str]:
    """将数据 YAML 中列表或字典形式的类别名称标准化为字典。"""
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(class_id): str(name) for class_id, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("数据 YAML 的 names 包含非法类别 ID。") from exc
    raise ValueError("数据 YAML 的 names 必须是列表或字典。")


def _validate_dataset_yaml(data_yaml: Path) -> None:
    """确认训练配置存在且严格定义 liquid/debris 两类。"""
    try:
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"数据 YAML 无法读取: {data_yaml}, 原因: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"数据 YAML 根节点必须是对象: {data_yaml}")
    if payload.get("nc") != 2:
        raise ValueError(f"数据 YAML 必须设置 nc=2: {data_yaml}")
    names = _normalize_names(payload.get("names"))
    if names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"数据 YAML 类别必须为 {EXPECTED_CLASS_NAMES}，实际为 {names}。"
        )
    for split in ("train", "val"):
        values = payload.get(split)
        paths = values if isinstance(values, list) else [values]
        if not values or not all(Path(str(path)).expanduser().is_dir() for path in paths):
            raise ValueError(f"数据 YAML 的 {split} 图片目录不存在或为空: {paths}")


def _validate_arguments(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """检查实验所需文件、参数和防覆盖输出目录。"""
    data_yaml = args.data.expanduser().resolve()
    model_yaml = args.model_yaml.expanduser().resolve()
    initial_weight = args.initial_weight.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"数据 YAML 不存在: {data_yaml}")
    if not model_yaml.is_file():
        raise FileNotFoundError(f"P2 模型结构不存在: {model_yaml}")
    if not initial_weight.is_file():
        raise FileNotFoundError(f"初始权重不存在: {initial_weight}")
    if args.epochs <= 0 or args.batch <= 0 or args.workers < 0:
        raise ValueError("epochs、batch 必须大于 0，workers 不能小于 0。")
    if len(args.imgsz) != 2 or any(size <= 0 for size in args.imgsz):
        raise ValueError("--imgsz 必须提供两个大于 0 的值。")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf 和 --iou 必须在 0 和 1 之间。")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise RuntimeError(
                f"实验输出目录已存在且不可复用，为避免覆盖已停止: {output_dir}"
            )
    _validate_dataset_yaml(data_yaml)
    return data_yaml, model_yaml, initial_weight, output_dir


def _json_value(value: Any) -> Any:
    """将训练库返回值转换为可写入 JSON 的基础类型。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _run_onnx_test(
    onnx_path: Path,
    data_yaml: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """调用现有 ONNX 检测程序评估全部验证集并保存三视图。"""
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/inference.py"),
        "--onnx-model",
        str(onnx_path),
        "--yaml",
        str(data_yaml),
        "--output-dir",
        str(output_dir),
        "--imgsz",
        str(args.imgsz[0]),
        str(args.imgsz[1]),
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--max-det",
        str(args.max_det),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    """依次执行独立训练、PT 验证、P2 ONNX 导出和 ONNX 测试。"""
    args = parse_args()
    data_yaml, model_yaml, initial_weight, output_dir = _validate_arguments(args)
    runs_dir = output_dir / "runs"
    onnx_dir = output_dir / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[实验] 数据配置: {data_yaml}")
    print(f"[实验] 初始权重: {initial_weight}")
    print(f"[实验] 输出目录: {output_dir}")
    model = YOLO(str(model_yaml))
    model.load(str(initial_weight))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=list(args.imgsz),
        project=str(runs_dir),
        name="train",
        exist_ok=False,
        device=args.device,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        amp=args.amp,
        workers=args.workers,
        resume=False,
    )

    best_path = runs_dir / "train" / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"训练完成后缺少 best.pt: {best_path}")
    best_model = YOLO(str(best_path))
    validation = best_model.val(
        data=str(data_yaml),
        split="val",
        batch=args.batch,
        imgsz=list(args.imgsz),
        device=args.device,
        workers=args.workers,
        project=str(runs_dir),
        name="val",
        exist_ok=False,
        plots=True,
    )

    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "liquid_debris_yolo11s_p2.onnx"
    exported = Path(
        export_yolov11_detect_p2(
            str(best_path),
            str(onnx_path),
            imgsz=list(args.imgsz),
            device="cpu",
            opset=11,
        )
    )
    if not exported.is_file():
        raise RuntimeError(f"ONNX 导出完成后文件不存在: {exported}")
    _run_onnx_test(
        exported,
        data_yaml,
        output_dir / "onnx_test",
        args,
    )

    summary = {
        "data": str(data_yaml),
        "model_yaml": str(model_yaml),
        "initial_weight": str(initial_weight),
        "best_pt": str(best_path),
        "onnx": str(exported),
        "onnx_test": str(output_dir / "onnx_test"),
        "validation_metrics": _json_value(getattr(validation, "results_dict", {})),
        "parameters": {
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": list(args.imgsz),
            "device": args.device,
            "workers": args.workers,
            "amp": args.amp,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
        },
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[完成] 最佳权重: {best_path}")
    print(f"[完成] ONNX: {exported}")
    print(f"[完成] ONNX 测试: {output_dir / 'onnx_test'}")


if __name__ == "__main__":
    main()
