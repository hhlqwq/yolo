"""各命令在昂贵操作开始前的一次性完整检查."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .errors import PipelineError
from .logging_utils import conda_python_command, find_conda_executable
from ..steps.reporting import validate_excel_template


def _require_directory(title: str, path: Path) -> None:
    """要求目录存在并给出配置项名称."""
    if not path.expanduser().is_dir():
        raise PipelineError(f"预检失败,{title}目录不存在:{path.expanduser()}")


def _require_file(title: str, path: Path) -> None:
    """要求文件存在并给出配置项名称."""
    if not path.expanduser().is_file():
        raise PipelineError(f"预检失败,{title}文件不存在:{path.expanduser()}")


def _require_weight_or_history(config: PipelineConfig, task: str, title: str, fallback: Path) -> None:
    """要求某模型存在历史成功权重或日期脚本兜底权重."""
    current_candidates = {
        "detect": [
            config.work_dir / "runs/yolo_detect_p2/train/weights/last.pt",
            config.work_dir / "runs/yolo_detect_p2/train/weights/best.pt",
        ],
        "segment": [
            config.work_dir / "runs/yolo_segment_p2/train/weights/last.pt",
            config.work_dir / "runs/yolo_segment_p2/train/weights/best.pt",
        ],
        "pidnet": [config.work_dir / "runs/pidnet"],
    }[task]
    if task == "pidnet":
        if any(current_candidates[0].glob("**/checkpoint.pth.tar")):
            return
    elif any(path.is_file() for path in current_candidates):
        return
    history = config.registry_dir / "model_history.json"
    if config.auto_finetune and history.is_file():
        try:
            payload = json.loads(history.read_text(encoding="utf-8"))
            records = payload.get("models", {}).get(task, [])
            if any(Path(str(item.get("best_path", ""))).expanduser().is_file() for item in records):
                return
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    if fallback.expanduser().is_file():
        return
    raise PipelineError(f"预检失败,{title}没有历史模型且兜底权重不存在:{fallback.expanduser()}")


def _require_configured_path(title: str, path: Path) -> None:
    """阻止缺省 Path('.') 被误认为用户已配置目录."""
    if str(path) in {"", "."}:
        raise PipelineError(f"预检失败,日期脚本没有配置{title}")


def _overlaps(first: Path, second: Path) -> bool:
    """判断两个解析后的目录相同或互相包含."""
    first, second = first.expanduser().resolve(), second.expanduser().resolve()
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _validate_path_boundaries(config: PipelineConfig, command: str) -> None:
    """阻止数据、模型产物和预标注目录发生危险嵌套."""
    processes_data = command == "prepare" or (command == "all" and config.enable_data_update)
    for index, (input_dir, output_dir) in enumerate(zip(config.input_dirs, config.output_dirs), start=1):
        if processes_data and _overlaps(input_dir, output_dir):
            raise PipelineError(f"预检失败,第{index}组原始数据与输出目录不能相同或互相包含")
        for title, path in (("原始数据", input_dir), ("标准数据集输出", output_dir)):
            if _overlaps(config.work_dir, path):
                raise PipelineError(f"预检失败,运行目录不能与第{index}组{title}目录相同或互相包含")
    if config.should_run_prelabel(command):
        for prelabel_index, prelabel_dir in enumerate(config.prelabel_dirs, start=1):
            for index, (input_dir, output_dir) in enumerate(zip(config.input_dirs, config.output_dirs), start=1):
                for title, path in (("原始数据", input_dir), ("标准数据集输出", output_dir)):
                    if _overlaps(prelabel_dir, path):
                        raise PipelineError(
                            f"预检失败,第{prelabel_index}个预标注目录不能与第{index}组{title}目录相同或互相包含"
                        )


def _check_python_environment(environment: str, imports: list[str], cwd: Path) -> None:
    """在指定 Conda 环境中检查本命令需要的 Python 模块."""
    code = ";".join(f"import {name}" for name in imports)
    command = conda_python_command(environment, ["-c", code])
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        summary = (result.stderr or result.stdout).strip().splitlines()
        raise PipelineError(
            f"预检失败,Conda环境{environment}缺少运行依赖:{imports};"
            f"错误:{' | '.join(summary[-5:])}"
        )


def run_preflight(config: PipelineConfig, command: str) -> None:
    """根据命令检查路径、仓库、模型配置、环境入口和 Excel."""
    _validate_path_boundaries(config, command)
    if command in {"train", "export", "test", "prelabel", "all"}:
        _require_directory("Ultralytics仓库", config.repo_root)
        if find_conda_executable() is None:
            raise PipelineError("预检失败,找不到conda命令")
        yolo_imports = ["torch", "numpy", "cv2", "yaml", "ultralytics"]
        if command in {"export", "test", "all"}:
            yolo_imports.append("onnx")
        if command in {"test", "all"}:
            yolo_imports.append("onnxruntime")
        _check_python_environment(config.yolo_env, yolo_imports, config.repo_root)
        if command in {"train", "export", "prelabel", "all"}:
            _require_directory("PIPELINE_PIDNET_ROOT", config.pidnet_root)
            pidnet_imports = ["torch", "numpy", "cv2", "yaml", "models", "configs"]
            if command in {"export", "all"}:
                pidnet_imports.append("onnx")
            _check_python_environment(config.pidnet_env, pidnet_imports, config.pidnet_root)
    if command == "prepare" or (command == "all" and config.enable_data_update):
        for index, (input_dir, output_dir) in enumerate(zip(config.input_dirs, config.output_dirs), start=1):
            _require_directory(f"PIPELINE_INPUT_DIR[{index}]", input_dir)
            if not output_dir.expanduser().parent.is_dir():
                raise PipelineError(f"预检失败,PIPELINE_OUTPUT_DIR[{index}]父目录不存在:{output_dir.expanduser().parent}")
    if command == "all" and not config.enable_data_update:
        _require_file("累计数据注册表", config.registry_dir / "datasets.yaml")
    elif command == "train" and (config.registry_dir / "datasets.yaml").is_file():
        _require_file("累计数据注册表", config.registry_dir / "datasets.yaml")
    elif command == "train":
        for index, output_dir in enumerate(config.output_dirs, start=1):
            _require_directory(f"标准数据集[{index}]", output_dir)
    if command in {"train", "all"}:
        _require_file("YOLO检测模型配置", config.detect_model_yaml)
        _require_file("YOLO分割模型配置", config.segment_model_yaml)
        _require_directory("PIPELINE_PIDNET_ROOT", config.pidnet_root)
        _require_file("PIDNet训练脚本", config.pidnet_root / "tools" / "train.py")
        _require_file("PIDNet配置", config.pidnet_config)
        _require_weight_or_history(config, "detect", "YOLO检测", config.detect_fallback_weight)
        _require_weight_or_history(config, "segment", "YOLO分割", config.segment_fallback_weight)
        _require_weight_or_history(config, "pidnet", "PIDNet", config.pidnet_fallback_weight)
    if config.should_run_prelabel(command):
        for index, prelabel_dir in enumerate(config.prelabel_dirs, start=1):
            _require_configured_path(f"PIPELINE_PRELABEL_DIR[{index}]", prelabel_dir)
            _require_directory(f"PIPELINE_PRELABEL_DIR[{index}]", prelabel_dir)
        _require_directory("PIPELINE_PIDNET_ROOT", config.pidnet_root)
    if command in {"report", "all"}:
        validate_excel_template(config.excel_path.expanduser())
