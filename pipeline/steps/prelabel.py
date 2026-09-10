"""融合 YOLO paper 实例分割与 PIDNet 液体/金属语义分割生成 LabelMe JSON."""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.logging_utils import conda_python_command, run_subprocess
from ..core.stage_runner import StageOutcome


def collect_images(image_dir: Path) -> list[Path]:
    """收集预标注目录当前层级的支持图片."""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in supported)


def simplify_polygon(points: np.ndarray, epsilon: float) -> list[list[float]]:
    """适度简化轮廓并保留自然边界."""
    contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if epsilon > 0 and len(contour) >= 3:
        contour = cv2.approxPolyDP(contour, epsilon, True)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in contour.reshape(-1, 2)]


def mask_to_pidnet_shapes(context: RunContext, mask_path: Path) -> list[dict[str, Any]]:
    """把 PIDNet mask 中 liquid/metal 的连通区域转换为多边形."""
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise PipelineError(f"无法读取PIDNet预标注mask:{mask_path}")
    shapes: list[dict[str, Any]] = []
    for class_id, label in ((1, "liquid"), (2, "metal")):
        binary = np.where(mask == class_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < context.config.pidnet_prelabel_min_area:
                continue
            points = simplify_polygon(contour.reshape(-1, 2), context.config.prelabel_polygon_epsilon)
            if len(points) >= 3:
                shapes.append(
                    {
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "description": f"模型预标注,来源=PIDNet,类别ID={class_id}",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )
    return shapes


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子保存一个 LabelMe JSON."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _run_pidnet_masks(
    context: RunContext,
    model_path: Path,
    config_path: Path,
    image_dir: Path,
    pending: list[Path],
    temporary_root: Path,
) -> tuple[Path, dict[str, Path]]:
    """在 pid 环境生成临时 mask;返回临时目录和图片映射."""
    mask_root = temporary_root / "pidnet_masks"
    manifest_path = temporary_root / "manifest.json"
    entries: list[dict[str, str]] = []
    mapping: dict[str, Path] = {}
    for image_path in pending:
        relative = image_path.relative_to(image_dir).with_suffix(".png")
        mask_path = mask_root / relative
        entries.append({"image": str(image_path), "mask": str(mask_path)})
        mapping[str(image_path.resolve())] = mask_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    helper = context.config.repo_root / "pipeline" / "workers" / "pidnet_infer.py"
    command = conda_python_command(
        context.config.pidnet_env,
        [
            str(helper),
            "--pidnet-root", str(context.config.pidnet_root),
            "--config", str(config_path),
            "--weights", str(model_path),
            "--manifest", str(manifest_path),
            "--device", f"cuda:{context.config.gpu_device}",
        ],
    )
    run_subprocess(
        command,
        context.config.pidnet_root,
        "PIDNet预标注推理",
        context.logger,
        key_line_filter=lambda line: "PIPELINE_PIDNET_PRELABEL_" in line,
    )
    return temporary_root, mapping


def generate_prelabels(
    context: RunContext,
    yolo_model_path: Path,
    pidnet_model_path: Path,
    pidnet_config_path: Path,
) -> StageOutcome:
    """为每个预标注图片目录生成两模型融合预标注."""
    directory_metrics: list[dict[str, Any]] = []
    for directory_index, prelabel_dir in enumerate(context.config.prelabel_dirs, start=1):
        outcome = _generate_prelabels_for_dir(
            context,
            yolo_model_path,
            pidnet_model_path,
            pidnet_config_path,
            prelabel_dir.expanduser().resolve(),
            directory_index,
        )
        directory_metrics.append(outcome.metrics)
    metrics = {
        "directories": directory_metrics,
        "images": sum(int(item["images"]) for item in directory_metrics),
        "generated": sum(int(item["generated"]) for item in directory_metrics),
        "skipped_existing": sum(int(item["skipped_existing"]) for item in directory_metrics),
        "yolo_paper_shapes": sum(int(item.get("yolo_paper_shapes", 0)) for item in directory_metrics),
        "pidnet_shapes": sum(int(item.get("pidnet_shapes", 0)) for item in directory_metrics),
        "shapes": sum(int(item["shapes"]) for item in directory_metrics),
    }
    return StageOutcome(message="融合预标注完成", metrics=metrics, value=metrics)


def _generate_prelabels_for_dir(
    context: RunContext,
    yolo_model_path: Path,
    pidnet_model_path: Path,
    pidnet_config_path: Path,
    image_dir: Path,
    directory_index: int,
) -> StageOutcome:
    """为一个预标注图片目录中缺少 JSON 的图片生成融合预标注."""
    yolo_model_path = yolo_model_path.expanduser().resolve()
    pidnet_model_path = pidnet_model_path.expanduser().resolve()
    for title, path, is_dir in (
        ("预标注图片目录", image_dir, True),
        ("YOLO分割best.pt", yolo_model_path, False),
        ("PIDNet best.pt", pidnet_model_path, False),
        ("PIDNet配置", pidnet_config_path, False),
    ):
        valid = path.is_dir() if is_dir else path.is_file()
        if not valid:
            raise PipelineError(f"{title}不存在:{path}")
    images = collect_images(image_dir)
    if not images:
        raise PipelineError(f"预标注目录中没有支持的图片:{image_dir}")
    pending = [path for path in images if context.config.prelabel_overwrite or not path.with_suffix(".json").exists()]
    skipped = len(images) - len(pending)
    context.logger.info(
        "PRELABEL_SCAN | directory=%s | images=%d | existing_skipped=%d | pending=%d | overwrite=%s",
        image_dir, len(images), skipped, len(pending), context.config.prelabel_overwrite,
    )
    if not pending:
        return StageOutcome(
            message="全部图片已有同名JSON,未执行推理",
            metrics={
                "directory": str(image_dir),
                "images": len(images),
                "generated": 0,
                "skipped_existing": skipped,
                "shapes": 0,
            },
        )

    temporary_root = context.work_dir / f".prelabel_tmp_{directory_index}"
    if temporary_root.exists():
        if temporary_root.is_symlink():
            raise PipelineError(f"拒绝清理符号链接临时目录:{temporary_root}")
        shutil.rmtree(temporary_root)
    yolo_json_root = temporary_root / "yolo_json"
    yolo_json_root.mkdir(parents=True)

    from ultralytics import YOLO
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(yolo_model_path))
    yolo_shapes = 0
    for index, image_path in enumerate(pending, start=1):
        results = model.predict(
            source=str(image_path),
            stream=False,
            batch=context.config.prelabel_batch,
            save=False,
            imgsz=context.config.model_image_size,
            conf=context.config.prelabel_confidence,
            iou=context.config.prelabel_iou,
            device=context.config.gpu_device,
            half=context.config.prelabel_half,
            retina_masks=context.config.prelabel_retina_masks,
            verbose=False,
        )
        if len(results) != 1:
            raise PipelineError(f"YOLO分割未返回唯一结果:{image_path}")
        result = results[0]
        shapes: list[dict[str, Any]] = []
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for polygon, class_id, confidence in zip(result.masks.xy, classes, confidences):
                points = simplify_polygon(polygon, context.config.prelabel_polygon_epsilon)
                if len(points) < 3:
                    continue
                names = result.names
                label = str(names[class_id] if isinstance(names, dict) else names[class_id])
                shapes.append(
                    {
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "description": f"模型预标注,来源=YOLO,置信度={float(confidence):.6f}",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )
        height, width = result.orig_shape
        relative_json = image_path.relative_to(image_dir).with_suffix(".json")
        json_path = yolo_json_root / relative_json
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "5.0.1",
            "flags": {"pipeline_prelabel": True},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": int(height),
            "imageWidth": int(width),
        }
        _atomic_write_json(json_path, payload)
        yolo_shapes += len(shapes)
        if index % 100 == 0 or index == len(pending):
            context.logger.info("PRELABEL_YOLO_PROGRESS | %d/%d", index, len(pending))
        del result, results

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    temporary_root, masks = _run_pidnet_masks(
        context, pidnet_model_path, pidnet_config_path, image_dir, pending, temporary_root,
    )
    pidnet_shapes = 0
    try:
        for index, image_path in enumerate(pending, start=1):
            mask_path = masks.get(str(image_path.resolve()))
            if mask_path is None or not mask_path.is_file():
                raise PipelineError(f"PIDNet未生成对应mask:{image_path}")
            intermediate_path = yolo_json_root / image_path.relative_to(image_dir).with_suffix(".json")
            output_path = image_path.with_suffix(".json")
            payload = json.loads(intermediate_path.read_text(encoding="utf-8"))
            added = mask_to_pidnet_shapes(context, mask_path)
            payload["shapes"].extend(added)
            _atomic_write_json(output_path, payload)
            pidnet_shapes += len(added)
            if index % 100 == 0 or index == len(pending):
                context.logger.info("PRELABEL_PIDNET_PROGRESS | %d/%d", index, len(pending))
    finally:
        if temporary_root.is_symlink():
            raise PipelineError(f"拒绝清理符号链接临时目录:{temporary_root}")
        if temporary_root.is_dir():
            shutil.rmtree(temporary_root)
    metrics = {
        "directory": str(image_dir),
        "images": len(images),
        "generated": len(pending),
        "skipped_existing": skipped,
        "yolo_paper_shapes": yolo_shapes,
        "pidnet_shapes": pidnet_shapes,
        "shapes": yolo_shapes + pidnet_shapes,
    }
    return StageOutcome(message="融合预标注完成", metrics=metrics, value=metrics)
