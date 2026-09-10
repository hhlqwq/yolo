"""调用现有 ONNX 推理脚本完成验证集抽样评测."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.io_utils import atomic_write_text
from ..core.logging_utils import conda_python_command, run_subprocess
from ..core.stage_runner import StageOutcome
from .registry import collect_yolo_images, load_registry, yolo_label_path


def _has_target(path: Path) -> bool:
    """判断 YOLO 标签是否至少包含一个有效目标."""
    if not path.is_file():
        return False
    return any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())


def _sample_id(image_path: Path) -> str:
    """为跨目录同名图片生成稳定且可读的唯一 ID."""
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}"


def build_test_manifest(context: RunContext) -> Path:
    """从累计验证集筛选有目标图片并固定随机抽样."""
    registry = load_registry(context)
    pidnet_masks = {
        str(Path(item["image"]).expanduser().resolve()): Path(item["mask"]).expanduser().resolve()
        for item in registry["pidnet"]["val"]
    }
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_text in registry["yolo"]["val"]:
        for image_path in collect_yolo_images(Path(source_text).expanduser()):
            image_path = image_path.resolve()
            key = str(image_path)
            if key in seen:
                continue
            detect_label = yolo_label_path(image_path, "labels_detect")
            segment_label = yolo_label_path(image_path, "labels_segment")
            pidnet_mask = pidnet_masks.get(key)
            if not _has_target(detect_label):
                continue
            if not segment_label.is_file() or pidnet_mask is None or not pidnet_mask.is_file():
                raise PipelineError(f"测试样本缺少分割GT:image={image_path}")
            candidates.append(
                {
                    "id": _sample_id(image_path),
                    "image": key,
                    "detect_label": str(detect_label.resolve()),
                    "segment_label": str(segment_label.resolve()),
                    "pidnet_mask": str(pidnet_mask),
                }
            )
            seen.add(key)
    if not candidates:
        raise PipelineError("累计验证集中没有检测GT非空且三套GT完整的测试图片")
    generator = random.Random(context.config.test_seed)
    generator.shuffle(candidates)
    samples = candidates[: context.config.test_images]
    payload: dict[str, Any] = {
        "seed": context.config.test_seed,
        "requested_images": context.config.test_images,
        "candidate_images": len(candidates),
        "selected_images": len(samples),
        "samples": samples,
    }
    path = context.work_dir / "results" / "test_samples.json"
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def _load_metrics(path: Path) -> dict[str, Any]:
    """读取推理脚本生成的指标 JSON."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"测试指标读取失败:{path},原因:{exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"测试指标必须是JSON字典:{path}")
    return payload


def run_model_tests(
    context: RunContext,
    detect_onnx: Path,
    segment_onnx: Path,
    pidnet_onnx: Path,
) -> StageOutcome:
    """生成统一清单并调用统一 ONNX 推理入口完成检测和分割测试."""
    manifest = build_test_manifest(context)
    results_dir = context.work_dir / "results"
    detect_dir = results_dir / "detect"
    segment_dir = results_dir / "segment"
    inference_script = context.config.repo_root / "tools" / "inference.py"
    detect_command = conda_python_command(
        context.config.yolo_env,
        [
            str(inference_script), "--manifest", str(manifest), "--onnx-model", str(detect_onnx),
            "--yaml", str(context.work_dir / "training_configs" / "yolo_detect.yaml"),
            "--output-dir", str(detect_dir), "--imgsz", *map(str, context.config.requested_image_size),
            "--conf", str(context.config.test_confidence), "--iou", str(context.config.test_iou),
        ],
    )
    run_subprocess(
        detect_command, context.config.repo_root, "验证集检测测试", context.logger,
        key_line_filter=lambda line: "Summary" in line or "Metrics" in line,
    )
    segment_command = conda_python_command(
        context.config.yolo_env,
        [
            str(inference_script), "--manifest", str(manifest), "--onnx-model", str(segment_onnx),
            "--pidnet-onnx", str(pidnet_onnx), "--output", str(segment_dir),
            "--imgsz", *map(str, context.config.requested_image_size), "--conf", str(context.config.test_confidence),
            "--iou", str(context.config.test_iou),
        ],
    )
    run_subprocess(
        segment_command, context.config.repo_root, "验证集融合分割测试", context.logger,
        key_line_filter=lambda line: "PIPELINE_TEST_" in line,
    )
    detect_metrics_path = detect_dir / "metrics.json"
    segment_metrics_path = segment_dir / "metrics.json"
    metrics = {
        "images": len(json.loads(manifest.read_text(encoding="utf-8"))["samples"]),
        "detect": _load_metrics(detect_metrics_path),
        "segment": _load_metrics(segment_metrics_path),
    }
    context.logger.info("PIPELINE_TEST_SUMMARY | %s", json.dumps(metrics, ensure_ascii=False))
    return StageOutcome(
        message=f"验证集抽样测试完成:images={metrics['images']}",
        artifacts={
            "manifest": str(manifest), "detect_dir": str(detect_dir), "segment_dir": str(segment_dir),
            "detect_metrics": str(detect_metrics_path), "segment_metrics": str(segment_metrics_path),
        },
        metrics=metrics,
        value=metrics,
    )
