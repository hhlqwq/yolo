"""三模型自动 Batch 探测、持久化和断点复用."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .context import RunContext
from .io_utils import atomic_write_text
from .logging_utils import conda_python_command


RESULT_PREFIX = "PIPELINE_AUTO_BATCH_RESULT="
TARGET_MEMORY_FRACTION = 0.80


def _manifest_path(context: RunContext) -> Path:
    """返回本次运行的自动 Batch 记录文件."""
    return context.work_dir / "training_configs" / "auto_batch.json"


def _load_manifest(context: RunContext) -> dict[str, Any]:
    """读取已有自动 Batch 记录,损坏时返回空结构."""
    path = _manifest_path(context)
    if not path.is_file():
        return {"version": 1, "tasks": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        context.logger.warning("AUTO_BATCH记录损坏，将重新探测:%s", path)
        return {"version": 1, "tasks": {}}
    if payload.get("version") != 1 or not isinstance(payload.get("tasks"), dict):
        context.logger.warning("AUTO_BATCH记录版本不支持，将重新探测:%s", path)
        return {"version": 1, "tasks": {}}
    return payload


def _save_manifest(context: RunContext, payload: dict[str, Any]) -> Path:
    """原子保存自动 Batch 记录."""
    path = _manifest_path(context)
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def _query_gpu_memory(device: int) -> dict[str, Any]:
    """通过 nvidia-smi 查询指定 GPU 的总显存和实时空闲显存."""
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=name,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    fields = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 4:
        return {}
    try:
        return {
            "name": fields[0],
            "total_mb": int(fields[1]),
            "free_mb": int(fields[2]),
            "used_mb": int(fields[3]),
        }
    except ValueError:
        return {}


def _memory_fallback(configured_max: int, gpu: dict[str, Any]) -> int:
    """探测失败时,按 GPU 当前占用和 80% 上限回退 Batch."""
    total = int(gpu.get("total_mb", 0))
    free = int(gpu.get("free_mb", 0))
    if total <= 0 or free <= 0:
        return configured_max
    target_used = max(1, math.floor(total * TARGET_MEMORY_FRACTION))
    current_used = max(0, total - free)
    remaining = max(0, target_used - current_used)
    ratio = min(1.0, remaining / target_used)
    return max(1, min(configured_max, math.floor(configured_max * ratio)))


def select_batch(
    context: RunContext,
    task: str,
    configured_max: int,
    environment: str,
    worker_arguments: list[str],
    cwd: Path,
) -> int:
    """实际探测一个模型的安全 Batch,并在同一 run-name 中固定结果."""
    if not context.config.auto_batch:
        context.logger.info(
            "AUTO_BATCH | model=%s | enabled=false | selected=%d",
            task,
            configured_max,
        )
        return configured_max

    manifest = _load_manifest(context)
    existing = manifest["tasks"].get(task)
    try:
        existing_value = int(existing.get("selected_batch", 0)) if isinstance(existing, dict) else 0
    except (TypeError, ValueError):
        existing_value = 0
    if isinstance(existing, dict) and existing_value > 0:
        selected = max(1, min(configured_max, existing_value))
        if selected != existing_value:
            existing["selected_batch"] = selected
            existing["configured_max"] = configured_max
            _save_manifest(context, manifest)
        context.logger.info(
            "AUTO_BATCH | model=%s | mode=reuse | configured_max=%d | selected=%d",
            task,
            configured_max,
            selected,
        )
        return selected

    gpu = _query_gpu_memory(context.config.gpu_device)
    if gpu:
        context.logger.info(
            "GPU_MEMORY | model=%s | device=%d | name=%s | total=%.2fGB | free=%.2fGB | used=%.2fGB",
            task,
            context.config.gpu_device,
            gpu["name"],
            gpu["total_mb"] / 1024,
            gpu["free_mb"] / 1024,
            gpu["used_mb"] / 1024,
        )
    worker = context.config.repo_root / "pipeline/workers/auto_batch.py"
    command = conda_python_command(
        environment,
        [
            str(worker),
            "--task",
            task,
            "--device",
            str(context.config.gpu_device),
            "--max-batch",
            str(configured_max),
            *worker_arguments,
        ],
    )
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result = subprocess.CompletedProcess(command, 1, stdout="", stderr=str(exc))
    probe: dict[str, Any] = {}
    for line in reversed((result.stdout + "\n" + result.stderr).splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                probe = json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                probe = {}
            break
    if result.returncode == 0 and probe.get("selected_batch"):
        selected = max(1, min(configured_max, int(probe["selected_batch"])))
        mode = "probe"
    else:
        selected = _memory_fallback(configured_max, gpu)
        mode = "memory_fallback"
        summary = " | ".join((result.stderr or result.stdout).strip().splitlines()[-5:])
        context.logger.warning(
            "AUTO_BATCH探测失败，使用显存比例回退 | model=%s | selected=%d | error=%s",
            task,
            selected,
            summary or "无子进程错误输出",
        )

    record = {
        "configured_max": configured_max,
        "selected_batch": selected,
        "target_memory_fraction": TARGET_MEMORY_FRACTION,
        "mode": mode,
        "gpu": gpu,
        "probe": probe,
    }
    manifest["tasks"][task] = record
    path = _save_manifest(context, manifest)
    context.logger.info(
        "AUTO_BATCH | model=%s | mode=%s | configured_max=%d | selected=%d | "
        "target_memory=%.0f%% | selected_peak_memory=%s | record=%s",
        task,
        mode,
        configured_max,
        selected,
        TARGET_MEMORY_FRACTION * 100,
        (
            f"{float(probe['selected_peak_memory_fraction']) * 100:.1f}%"
            if probe.get("selected_peak_memory_fraction") is not None
            else "N/A"
        ),
        path,
    )
    return selected
