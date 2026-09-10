"""三模型训练、断点恢复、历史最佳权重选择和指标整理."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.auto_batch import select_batch
from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.io_utils import atomic_write_text
from ..core.logging_utils import conda_python_command, run_subprocess
from ..core.stage_runner import StageOutcome
from ..core.state import StageStatus


MODEL_TASKS = ("detect", "segment", "pidnet")


def sha256_file(path: Path) -> str:
    """流式计算模型文件 SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_model_history() -> dict[str, Any]:
    """返回空的模型历史注册表."""
    return {"version": 1, "models": {task: [] for task in MODEL_TASKS}}


def load_model_history(context: RunContext) -> dict[str, Any]:
    """加载模型历史,并兼容旧 latest_models.json."""
    path = context.config.registry_dir / "model_history.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(f"模型历史注册表损坏:{path},原因:{exc}") from exc
        if payload.get("version") != 1 or not isinstance(payload.get("models"), dict):
            raise PipelineError(f"模型历史注册表版本或结构不支持:{path}")
        for task in MODEL_TASKS:
            payload["models"].setdefault(task, [])
        return payload

    history = empty_model_history()
    legacy = context.config.registry_dir / "latest_models.json"
    if legacy.is_file():
        try:
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            run_name = str(payload.get("run_name", "legacy"))
            models = payload.get("models", {})
            for task in MODEL_TASKS:
                value = models.get(task)
                if value and Path(value).is_file():
                    history["models"][task].append(
                        {
                            "run_name": run_name,
                            "best_path": str(Path(value).resolve()),
                            "completed_at": payload.get("updated_at"),
                            "best_epoch": None,
                            "metrics_path": None,
                            "sha256": None,
                        }
                    )
            save_model_history(context, history)
            context.logger.info("已迁移旧版最近模型注册表:%s", legacy)
        except (OSError, json.JSONDecodeError):
            context.logger.warning("旧版最近模型注册表无法迁移:%s", legacy)
    return history


def save_model_history(context: RunContext, history: dict[str, Any]) -> Path:
    """原子保存模型历史注册表."""
    path = context.config.registry_dir / "model_history.json"
    atomic_write_text(path, json.dumps(history, ensure_ascii=False, indent=2))
    return path


def register_best_model(
    context: RunContext,
    task: str,
    best_path: Path,
    metrics_path: Path | None,
    best_epoch: int | None,
) -> None:
    """幂等记录一个成功完成的最佳模型."""
    history = load_model_history(context)
    records = history["models"][task]
    checksum = sha256_file(best_path)
    record = {
        "run_name": context.config.run_name,
        "best_path": str(best_path.resolve()),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best_epoch,
        "metrics_path": None if metrics_path is None else str(metrics_path.resolve()),
        "sha256": checksum,
    }
    records[:] = [item for item in records if item.get("run_name") != context.config.run_name]
    records.append(record)
    save_model_history(context, history)


def select_previous_best(context: RunContext, task: str, fallback: Path) -> Path:
    """选择最近成功模型;不存在时使用日期脚本配置的兜底权重."""
    if context.config.auto_finetune:
        history = load_model_history(context)
        for record in reversed(history["models"][task]):
            candidate = Path(str(record.get("best_path", ""))).expanduser()
            if candidate.is_file():
                expected = record.get("sha256")
                if expected and sha256_file(candidate) != expected:
                    context.logger.warning("历史模型校验值不匹配,跳过:%s", candidate)
                    continue
                return candidate.resolve()
    fallback = fallback.expanduser()
    if fallback.is_file():
        return fallback.resolve()
    raise PipelineError(f"{task}没有有效历史best.pt,兜底权重也不存在:{fallback}")


def backup_run_directory(run_dir: Path, title: str, logger: Any) -> Path | None:
    """强制重训前备份模型目录并返回备份路径."""
    if not run_dir.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_dir.with_name(f"{run_dir.name}_backup_{timestamp}")
    suffix = 1
    while backup.exists():
        backup = run_dir.with_name(f"{run_dir.name}_backup_{timestamp}_{suffix}")
        suffix += 1
    run_dir.rename(backup)
    logger.warning("FORCE_TRAIN_BACKUP | %s | %s", title, backup)
    return backup


def read_yolo_results(run_dir: Path) -> list[dict[str, str]]:
    """读取 YOLO results.csv 并清理列名空格."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return [
                {str(key).strip(): str(value).strip() for key, value in row.items()}
                for row in csv.DictReader(file)
            ]
    except OSError as exc:
        raise PipelineError(f"YOLO results.csv读取失败:{path},原因:{exc}") from exc


def yolo_actual_epoch(run_dir: Path) -> int | None:
    """返回 YOLO 已完成的 1-based Epoch 数."""
    epochs: list[int] = []
    for row in read_yolo_results(run_dir):
        try:
            epochs.append(int(round(float(row["epoch"]))))
        except (KeyError, ValueError):
            continue
    return max(epochs) if epochs else None


def yolo_training_seconds(run_dir: Path) -> float | None:
    """从 results.csv 返回累计训练秒数."""
    values: list[float] = []
    for row in read_yolo_results(run_dir):
        try:
            values.append(float(row["time"]))
        except (KeyError, ValueError):
            continue
    return max(values) if values else None


def infer_best_epoch(run_dir: Path, metrics: dict[str, Any]) -> int | None:
    """使用 best.pt 验证指标匹配 results.csv 中的最佳 Epoch."""
    metric_names = [
        name
        for name, value in metrics.items()
        if name != "fitness" and isinstance(value, (int, float, np.number))
    ]
    candidates: list[tuple[float, int]] = []
    for row in read_yolo_results(run_dir):
        try:
            epoch = int(round(float(row["epoch"])))
            names = [name for name in metric_names if name in row]
            score = sum((float(row[name]) - float(metrics[name])) ** 2 for name in names)
        except (KeyError, TypeError, ValueError):
            continue
        if names:
            candidates.append((score, epoch))
    return min(candidates)[1] if candidates else None


def load_yolo_metrics(run_dir: Path) -> dict[str, Any]:
    """读取并校正 YOLO 最佳权重指标."""
    path = run_dir / "best_metrics.json"
    if not path.is_file():
        return {"best_epoch": None, "fitness": None, "metrics": {}, "class_metrics": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"YOLO最佳指标文件损坏:{path},原因:{exc}") from exc
    metrics = payload.get("metrics", {})
    inferred = infer_best_epoch(run_dir, metrics) if isinstance(metrics, dict) else None
    if inferred is not None:
        payload["best_epoch"] = inferred
    return payload


def ensure_yolo_metrics(
    context: RunContext,
    task: str,
    run_dir: Path,
    best_path: Path,
    data_yaml: Path,
) -> tuple[Path, dict[str, Any]]:
    """确保旧版或新训练 YOLO best.pt 都有结构化最佳指标."""
    metrics_path = run_dir / "best_metrics.json"
    existing = load_yolo_metrics(run_dir)
    if existing.get("metrics") and existing.get("class_metrics"):
        return metrics_path, existing
    code = f'''import json
from pathlib import Path
from ultralytics import YOLO
model = YOLO({str(best_path)!r})
result = model.val(
    data={str(data_yaml)!r},
    imgsz={context.config.model_image_size!r},
    batch=1,
    device={context.config.gpu_device!r},
    plots=False,
)
metrics = {{key: float(value) for key, value in result.results_dict.items()}}
summary = result.summary()
if hasattr(result, "seg"):
    for index, item in enumerate(summary):
        values = result.seg.class_result(index)
        item["Mask-mAP50"] = float(values[2])
        item["Mask-mAP50-95"] = float(values[3])
payload = {{
    "best_epoch": None,
    "fitness": float(result.fitness),
    "metrics": metrics,
    "class_metrics": summary,
}}
Path({str(metrics_path)!r}).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("PIPELINE_METRICS_SAVED", {str(metrics_path)!r})
'''
    run_subprocess(
        conda_python_command(context.config.yolo_env, ["-c", code]),
        context.config.repo_root,
        f"YOLO-{task} best.pt指标评估",
        context.logger,
        key_line_filter=lambda line: "PIPELINE_METRICS_SAVED" in line,
    )
    payload = load_yolo_metrics(run_dir)
    atomic_write_text(metrics_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return metrics_path, payload


def yolo_completion_marker(run_dir: Path) -> Path:
    """返回 YOLO 正常完成标记路径."""
    return run_dir / ".pipeline_model.json"


def yolo_checkpoint_batch(path: Path, default: int) -> int:
    """从 YOLO checkpoint 读取训练器最终使用的实际 Batch."""
    if not path.is_file():
        return default
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        train_args = payload.get("train_args", {}) if isinstance(payload, dict) else {}
        value = train_args.get("batch", default) if isinstance(train_args, dict) else default
        return max(1, int(value))
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return default


def update_pidnet_batch(config_path: Path, batch: int) -> None:
    """把自动选择的实际 Batch 写入本次 PIDNet 配置快照."""
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"PIDNet配置读取失败:{config_path},原因:{exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"PIDNet配置必须是字典:{config_path}")
    train = payload.setdefault("TRAIN", {})
    if not isinstance(train, dict):
        raise PipelineError(f"PIDNet配置TRAIN必须是字典:{config_path}")
    train["BATCH_SIZE_PER_GPU"] = batch
    atomic_write_text(config_path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def train_yolo(context: RunContext, task: str, data_yaml: Path) -> StageOutcome:
    """训练、恢复或复用一个 YOLO 检测/实例分割模型."""
    if task not in {"detect", "segment"}:
        raise PipelineError(f"不支持的YOLO任务:{task}")
    config = context.config
    model_yaml = config.detect_model_yaml if task == "detect" else config.segment_model_yaml
    epochs = config.detect_epochs if task == "detect" else config.segment_epochs
    configured_batch = config.detect_batch if task == "detect" else config.segment_batch
    batch = configured_batch
    fallback = config.detect_fallback_weight if task == "detect" else config.segment_fallback_weight
    project_dir = context.work_dir / "runs" / f"yolo_{task}_p2"
    run_dir = project_dir / "train"
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    forced_initial: Path | None = None
    initial_weight: Path | None = None
    if config.force_train:
        backup = backup_run_directory(run_dir, f"YOLO-{task}", context.logger)
        if backup is not None and (backup / "weights" / "best.pt").is_file():
            forced_initial = backup / "weights" / "best.pt"

    actual_epoch = yolo_actual_epoch(run_dir)
    marker = yolo_completion_marker(run_dir)
    marker_payload: dict[str, Any] = {}
    if marker.is_file():
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker_payload = {}
    complete = best_path.is_file() and (
        marker_payload.get("completed") is True
        or (actual_epoch is not None and actual_epoch >= epochs)
    )
    if complete and not config.force_train:
        mode = "skipped"
        batch = int(marker_payload.get("batch", configured_batch))
        if marker_payload.get("initial_weight"):
            initial_weight = Path(str(marker_payload["initial_weight"]))
        context.logger.info(
            "TRAIN_INIT | model=YOLO-%s | mode=skip | best=%s | epoch=%s/%d",
            task,
            best_path,
            actual_epoch,
            epochs,
        )
    elif last_path.is_file() and run_dir.exists():
        mode = "resume"
        batch = select_batch(
            context,
            task,
            configured_batch,
            config.yolo_env,
            [
                "--imgsz",
                *map(str, config.model_image_size),
                "--model",
                str(model_yaml),
                "--weight",
                str(last_path),
                *(["--amp"] if config.yolo_amp else []),
            ],
            config.repo_root,
        )
        initial_weight = last_path
        context.logger.info(
            "TRAIN_INIT | model=YOLO-%s | mode=resume | checkpoint=%s | epoch=%s/%d",
            task,
            last_path,
            actual_epoch,
            epochs,
        )
        code = (
            "from ultralytics import YOLO\n"
            f"model = YOLO({str(last_path)!r})\n"
            f"model.train(resume=True, epochs={epochs!r}, batch={batch!r}, imgsz={config.model_image_size!r})\n"
        )
        run_subprocess(
            conda_python_command(config.yolo_env, ["-c", code]),
            config.repo_root,
            f"YOLO-{task}断点恢复",
            context.logger,
        )
    else:
        mode = "finetune"
        initial = forced_initial or select_previous_best(context, task, fallback)
        initial_weight = initial
        worker_arguments = [
            "--imgsz",
            *map(str, config.model_image_size),
            "--model",
            str(model_yaml),
            "--weight",
            str(initial),
        ]
        if config.yolo_amp:
            worker_arguments.append("--amp")
        batch = select_batch(
            context,
            task,
            configured_batch,
            config.yolo_env,
            worker_arguments,
            config.repo_root,
        )
        context.logger.info(
            "TRAIN_INIT | model=YOLO-%s | mode=finetune | source_best=%s | epoch=1/%d",
            task,
            initial,
            epochs,
        )
        arguments = {
            "data": str(data_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": config.model_image_size,
            "name": "train",
            "project": str(project_dir),
            "exist_ok": True,
            "device": config.gpu_device,
            "cos_lr": True,
            "warmup_epochs": 3,
            "close_mosaic": 10,
            "amp": config.yolo_amp,
            "workers": config.yolo_workers,
            "resume": False,
        }
        code = (
            "from ultralytics import YOLO\n"
            f"model = YOLO({str(model_yaml)!r})\n"
            f"model.load({str(initial)!r})\n"
            f"model.train(**{arguments!r})\n"
        )
        run_subprocess(
            conda_python_command(config.yolo_env, ["-c", code]),
            config.repo_root,
            f"YOLO-{task}训练",
            context.logger,
        )

    if not best_path.is_file():
        raise PipelineError(f"YOLO-{task}结束后缺少best.pt:{best_path}")
    actual_epoch = yolo_actual_epoch(run_dir)
    if mode != "skipped":
        batch = yolo_checkpoint_batch(last_path, batch)
        marker_payload = {
            "completed": True,
            "task": task,
            "planned_epochs": epochs,
            "batch": batch,
            "image_size": config.model_image_size,
            "force_train": config.force_train,
            "initial_weight": "" if initial_weight is None else str(initial_weight),
            "actual_epoch": actual_epoch,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_text(marker, json.dumps(marker_payload, ensure_ascii=False, indent=2))
    metrics_path, metrics = ensure_yolo_metrics(context, task, run_dir, best_path, data_yaml)
    register_best_model(context, task, best_path, metrics_path, metrics.get("best_epoch"))
    training_seconds = yolo_training_seconds(run_dir)
    return StageOutcome(
        status=StageStatus.SKIPPED if mode == "skipped" else StageStatus.SUCCEEDED,
        message=f"mode={mode},best_epoch={metrics.get('best_epoch')}",
        artifacts={
            "best": str(best_path),
            "last": str(last_path),
            "run_dir": str(run_dir),
            "metrics": str(metrics_path),
            "mode": mode,
            "initial_weight": "" if initial_weight is None else str(initial_weight),
        },
        metrics={
            **metrics,
            "actual_epoch": actual_epoch,
            "training_seconds": training_seconds,
            "planned_epochs": marker_payload.get("planned_epochs", epochs),
            "batch": marker_payload.get("batch", batch),
            "image_size": marker_payload.get("image_size", config.model_image_size),
            "force_train": marker_payload.get("force_train", config.force_train),
        },
        value=best_path,
    )


def load_pidnet_checkpoint(path: Path) -> dict[str, Any]:
    """在CPU上读取PIDNet checkpoint;不存在时返回空字典."""
    if not path.is_file():
        return {}
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PipelineError(f"PIDNet checkpoint读取失败:{path},原因:{exc}") from exc


def load_pidnet_metrics(output_dir: Path) -> dict[str, Any]:
    """读取PIDNet最佳指标JSON."""
    path = output_dir / "best_metrics.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"PIDNet最佳指标读取失败:{path},原因:{exc}") from exc


def ensure_pidnet_metrics(
    context: RunContext,
    output_dir: Path,
    best_path: Path,
    config_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """确保PIDNet best.pt具有完整总体和逐类别指标."""
    metrics_path = output_dir / "best_metrics.json"
    payload = load_pidnet_metrics(output_dir)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if metrics.get("class_metrics") and metrics.get("pixel_accuracy") is not None:
        return metrics_path, payload
    eval_script = context.config.pidnet_root / "tools" / "eval.py"
    if not eval_script.is_file():
        raise PipelineError(f"PIDNet评估脚本不存在:{eval_script}")
    command = conda_python_command(
        context.config.pidnet_env,
        [
            str(eval_script),
            "--cfg",
            str(config_path),
            "--metrics-json",
            str(metrics_path),
            "TEST.MODEL_FILE",
            str(best_path),
        ],
    )
    run_subprocess(command, context.config.pidnet_root, "PIDNet best.pt指标评估", context.logger)
    payload = load_pidnet_metrics(output_dir)
    checkpoint = load_pidnet_checkpoint(output_dir / "checkpoint.pth.tar")
    payload["best_epoch"] = payload.get("best_epoch", checkpoint.get("best_epoch"))
    payload["actual_epoch"] = payload.get("actual_epoch", checkpoint.get("epoch"))
    atomic_write_text(metrics_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return metrics_path, payload


def train_pidnet(context: RunContext, config_path: Path) -> StageOutcome:
    """训练、恢复或复用PIDNet语义分割模型."""
    config = context.config
    train_script = config.pidnet_root / "tools" / "train.py"
    if not train_script.is_file():
        raise PipelineError(f"PIDNet训练脚本不存在:{train_script}")
    output_root = context.work_dir / "runs" / "pidnet"
    output_dir = output_root / "train"
    legacy_root = output_root / "liquid_metal"
    legacy_candidates = [path for path in legacy_root.glob("*") if path.is_dir()] if legacy_root.is_dir() else []
    if len(legacy_candidates) > 1:
        raise PipelineError(f"PIDNet发现多个旧输出目录,无法自动判断:{legacy_candidates}")
    legacy_output_dir = legacy_candidates[0] if legacy_candidates else None
    if legacy_output_dir is not None:
        if output_dir.exists():
            raise PipelineError(
                f"PIDNet新旧输出目录同时存在,请人工确认后保留一个:"
                f"new={output_dir},legacy={legacy_output_dir}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        legacy_output_dir.rename(output_dir)
        if legacy_root.is_dir() and not any(legacy_root.iterdir()):
            legacy_root.rmdir()
        context.logger.warning("PIDNet旧输出目录已迁移:%s -> %s", legacy_output_dir, output_dir)
    best_path = output_dir / "best.pt"
    checkpoint_path = output_dir / "checkpoint.pth.tar"
    final_state_path = output_dir / "final_state.pt"
    marker_path = output_dir / ".pipeline_model.json"
    marker_payload: dict[str, Any] = {}
    if marker_path.is_file():
        try:
            marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker_payload = {}
    forced_initial: Path | None = None
    initial_weight: Path | None = None
    batch = config.pidnet_batch
    if config.force_train:
        backup = backup_run_directory(output_dir, "PIDNet", context.logger)
        if backup is not None and (backup / "best.pt").is_file():
            forced_initial = backup / "best.pt"

    complete = best_path.is_file() and (final_state_path.is_file() or marker_path.is_file())
    command = [
        str(train_script),
        "--cfg",
        str(config_path),
        "OUTPUT_DIR",
        str(output_root),
        "LOG_DIR",
        str(context.work_dir / "tensorboard"),
        "DATASET.ROOT",
        f"{context.work_dir}{os.sep}",
        "DATASET.TRAIN_SET",
        "pidnet_list/train.lst",
        "DATASET.TEST_SET",
        "pidnet_list/val.lst",
        "OUTPUT_FLAT",
        "True",
    ]
    if complete and not config.force_train:
        mode = "skipped"
        batch = int(marker_payload.get("batch", config.pidnet_batch))
        if marker_payload.get("initial_weight"):
            initial_weight = Path(str(marker_payload["initial_weight"]))
        context.logger.info("TRAIN_INIT | model=PIDNet | mode=skip | best=%s", best_path)
    elif checkpoint_path.is_file():
        mode = "resume"
        batch = select_batch(
            context,
            "pidnet",
            config.pidnet_batch,
            config.pidnet_env,
            [
                "--imgsz",
                *map(str, config.model_image_size),
                "--config",
                str(config_path),
                "--pidnet-root",
                str(config.pidnet_root),
                "--weight",
                str(checkpoint_path),
            ],
            config.pidnet_root,
        )
        update_pidnet_batch(config_path, batch)
        initial_weight = checkpoint_path
        checkpoint = load_pidnet_checkpoint(checkpoint_path)
        context.logger.info(
            "TRAIN_INIT | model=PIDNet | mode=resume | checkpoint=%s | epoch=%s/%d",
            checkpoint_path,
            checkpoint.get("epoch"),
            config.pidnet_epochs,
        )
        command.extend(["TRAIN.RESUME", "True"])
        run_subprocess(
            conda_python_command(config.pidnet_env, command),
            config.pidnet_root,
            "PIDNet断点恢复",
            context.logger,
        )
    else:
        mode = "finetune"
        initial = forced_initial or select_previous_best(context, "pidnet", config.pidnet_fallback_weight)
        initial_weight = initial
        batch = select_batch(
            context,
            "pidnet",
            config.pidnet_batch,
            config.pidnet_env,
            [
                "--imgsz",
                *map(str, config.model_image_size),
                "--config",
                str(config_path),
                "--pidnet-root",
                str(config.pidnet_root),
                "--weight",
                str(initial),
            ],
            config.pidnet_root,
        )
        update_pidnet_batch(config_path, batch)
        context.logger.info(
            "TRAIN_INIT | model=PIDNet | mode=finetune | source_best=%s | epoch=1/%d",
            initial,
            config.pidnet_epochs,
        )
        command.extend(["MODEL.PRETRAINED", str(initial), "TRAIN.RESUME", "False"])
        run_subprocess(
            conda_python_command(config.pidnet_env, command),
            config.pidnet_root,
            "PIDNet训练",
            context.logger,
        )

    if not best_path.is_file():
        raise PipelineError(f"PIDNet结束后缺少best.pt:{best_path}")
    checkpoint = load_pidnet_checkpoint(checkpoint_path)
    if mode != "skipped":
        marker_payload = {
            "completed": True,
            "planned_epochs": config.pidnet_epochs,
            "batch": batch,
            "image_size": config.model_image_size,
            "force_train": config.force_train,
            "initial_weight": "" if initial_weight is None else str(initial_weight),
            "actual_epoch": checkpoint.get("epoch"),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        atomic_write_text(marker_path, json.dumps(marker_payload, ensure_ascii=False, indent=2))
    metrics_path, payload = ensure_pidnet_metrics(context, output_dir, best_path, config_path)
    register_best_model(context, "pidnet", best_path, metrics_path, payload.get("best_epoch"))
    return StageOutcome(
        status=StageStatus.SKIPPED if mode == "skipped" else StageStatus.SUCCEEDED,
        message=f"mode={mode},best_epoch={payload.get('best_epoch')}",
        artifacts={
            "best": str(best_path),
            "checkpoint": str(checkpoint_path),
            "run_dir": str(output_dir),
            "metrics": str(metrics_path),
            "mode": mode,
            "initial_weight": "" if initial_weight is None else str(initial_weight),
        },
        metrics={
            **payload,
            "actual_epoch": payload.get("actual_epoch", checkpoint.get("epoch")),
            "planned_epochs": marker_payload.get("planned_epochs", config.pidnet_epochs),
            "batch": marker_payload.get("batch", batch),
            "image_size": marker_payload.get("image_size", config.model_image_size),
            "force_train": marker_payload.get("force_train", config.force_train),
        },
        value=best_path,
    )
