"""检测 TXT 数据集处理、校验和累计注册表维护。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import cv2

from pipeline.core.errors import PipelineError
from pipeline.core.io_utils import atomic_write_text
from pipeline.core.stage_runner import StageOutcome

from ..core.config import DetectConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")
CLASS_NAMES = {0: "paper", 1: "liquid", 2: "metal"}
COCO_CLASS_IDS = {name: class_id for class_id, name in CLASS_NAMES.items()}
COCO_IGNORE_NAME = "ignore"


@dataclass(frozen=True, slots=True)
class SourceSample:
    """保存已解析的检测样本和独立 ignore 涂黑区域."""

    image: Path
    source_label: Path
    retained_lines: tuple[str, ...]
    ignore_boxes: tuple[tuple[float, float, float, float], ...]


def _inside(path: Path, parent: Path) -> bool:
    """判断 path 是否位于 parent 中。"""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _collect_txt_pairs(source: Path, output: Path) -> list[tuple[Path, Path]]:
    """按三种兼容布局收集唯一的同名图片与 TXT 标签对。"""
    pairs: dict[str, tuple[Path, Path]] = {}
    conflicts: list[str] = []

    def add(image_dir: Path, label_dir: Path) -> None:
        """从一对图片和标签目录读取可配对样本。"""
        if _inside(image_dir, output) or not label_dir.is_dir():
            return
        labels = {item.stem.casefold(): item for item in label_dir.glob("*.txt") if item.is_file()}
        for image in image_dir.iterdir():
            if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            label = labels.get(image.stem.casefold())
            if label is None:
                continue
            key = str(image.resolve()).casefold()
            if key in pairs and pairs[key][1].resolve() != label.resolve():
                conflicts.append(f"{image}: {pairs[key][1]} / {label}")
            else:
                pairs[key] = (image.resolve(), label.resolve())

    candidates = [source, *sorted(item for item in source.rglob("*") if item.is_dir())]
    for directory in candidates:
        if _inside(directory, output):
            continue
        for image_name, label_name in (("images", "labels"), ("img", "yolo")):
            add(directory / image_name, directory / label_name)
        add(directory, directory)
    if conflicts:
        raise PipelineError("同一图片匹配到多个标签:\n" + "\n".join(conflicts[:20]))
    return sorted(pairs.values(), key=lambda item: str(item[0]).casefold())


def collect_pairs(source: Path, output: Path) -> list[tuple[Path, Path]]:
    """收集 TXT 图片标签对,未找到时保留历史报错行为."""
    pairs = _collect_txt_pairs(source, output)
    if not pairs:
        raise PipelineError(f"未找到检测数据对:{source}；支持images/labels、img/yolo或图片和TXT直存")
    return pairs


def _parse_label(
    path: Path,
    ignore_class_ids: tuple[int, ...] = (),
) -> tuple[Counter[int], list[tuple[float, float, float, float]], list[str]]:
    """验证 YOLO 标签，并提取 ignore 框和过滤后的训练标签行。"""
    counts: Counter[int] = Counter()
    ignore_boxes: list[tuple[float, float, float, float]] = []
    retained_lines: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise PipelineError(f"标签无法读取:{path}") from exc
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise PipelineError(f"检测标签格式错误:{path}:{index}，应为5列")
        try:
            class_id = int(fields[0])
            values = [float(item) for item in fields[1:]]
        except ValueError as exc:
            raise PipelineError(f"检测标签含非数值字段:{path}:{index}") from exc
        if not all(0.0 <= value <= 1.0 for value in values):
            raise PipelineError(f"检测标签类别或坐标越界:{path}:{index}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise PipelineError(f"检测框宽高必须大于0:{path}:{index}")
        if class_id in ignore_class_ids:
            ignore_boxes.append(tuple(values))
            continue
        if class_id not in CLASS_NAMES:
            raise PipelineError(f"检测标签类别或坐标越界:{path}:{index}")
        counts[class_id] += 1
        retained_lines.append(" ".join(fields))
    return counts, ignore_boxes, retained_lines


def validate_label(path: Path, ignore_class_ids: tuple[int, ...] = ()) -> Counter[int]:
    """验证一个 YOLO 检测标签，并返回非 ignore 类别的实例统计。"""
    counts, _, _ = _parse_label(path, ignore_class_ids)
    return counts


def _txt_samples(
    pairs: list[tuple[Path, Path]],
    ignore_class_ids: tuple[int, ...],
) -> list[SourceSample]:
    """解析传统 YOLO TXT 数据对并分离 ignore 区域."""
    samples: list[SourceSample] = []
    for image, label in pairs:
        _, ignore_boxes, retained_lines = _parse_label(label, ignore_class_ids)
        samples.append(
            SourceSample(
                image=image,
                source_label=label,
                retained_lines=tuple(retained_lines),
                ignore_boxes=tuple(ignore_boxes),
            )
        )
    return samples


def _load_coco_payload(path: Path) -> dict[str, Any]:
    """读取并验证一个场景级 COCO 检测 JSON 根结构."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"COCO JSON无法读取:{path},原因:{exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"COCO JSON根节点必须是对象:{path}")
    for key in ("categories", "images", "annotations"):
        if not isinstance(payload.get(key), list):
            raise PipelineError(f"COCO JSON缺少数组字段{key}:{path}")
    return payload


def _coco_category_map(categories: list[Any], path: Path) -> dict[int, int | None]:
    """按类别名称构建 COCO ID 到训练类别或 ignore 的映射."""
    mapping: dict[int, int | None] = {}
    names: set[str] = set()
    allowed_names = {*COCO_CLASS_IDS, COCO_IGNORE_NAME}
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise PipelineError(f"COCO第{index}个category不是对象:{path}")
        category_id = category.get("id")
        name = str(category.get("name", "")).strip().casefold()
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise PipelineError(f"COCO category.id必须是整数:{path}:{index}")
        if name not in allowed_names:
            raise PipelineError(f"COCO包含未知类别{name!r}:{path}:{index}")
        if category_id in mapping or name in names:
            raise PipelineError(f"COCO category.id或name重复:{path}:{index}")
        mapping[category_id] = None if name == COCO_IGNORE_NAME else COCO_CLASS_IDS[name]
        names.add(name)
    missing_names = sorted(allowed_names - names)
    if missing_names:
        raise PipelineError(f"COCO缺少类别定义:{missing_names}:{path}")
    return mapping


def _scene_image_index(image_dir: Path, path: Path) -> dict[str, Path]:
    """按不区分大小写的文件名索引一个 COCO 场景图片目录."""
    if not image_dir.is_dir():
        raise PipelineError(f"COCO场景缺少images目录:{image_dir};JSON={path}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for image in sorted(image_dir.rglob("*")):
        if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        key = image.name.casefold()
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(image)
        else:
            index[key] = image.resolve()
    if duplicates:
        details = "\n".join(
            " / ".join(str(item) for item in items)
            for items in list(duplicates.values())[:20]
        )
        raise PipelineError(f"COCO场景images存在重名图片:{path}\n{details}")
    return index


def _coco_box(
    bbox: Any,
    width: int,
    height: int,
    description: str,
) -> tuple[float, float, float, float]:
    """校验 COCO 像素 bbox 并转换为归一化中心点格式."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise PipelineError(f"COCO bbox必须是[x,y,width,height]:{description}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        raise PipelineError(f"COCO bbox必须是数值数组:{description}")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        raise PipelineError(f"COCO bbox包含非有限数值:{description}")
    if box_width <= 0.0 or box_height <= 0.0:
        raise PipelineError(f"COCO bbox宽高必须大于0:{description}")
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(float(width), x + box_width)
    bottom = min(float(height), y + box_height)
    if right <= left or bottom <= top:
        raise PipelineError(f"COCO bbox位于图片范围外:{description}")
    return (
        ((left + right) / 2.0) / width,
        ((top + bottom) / 2.0) / height,
        (right - left) / width,
        (bottom - top) / height,
    )


def _collect_coco_samples(
    source: Path,
    output: Path,
) -> tuple[list[SourceSample], dict[str, int]]:
    """递归读取场景级 COCO JSON 并生成内存检测样本."""
    json_paths = [
        path
        for path in sorted(source.rglob("*.json"))
        if path.is_file()
        and path.parent.name.casefold() == "labels"
        and not _inside(path, output)
    ]
    samples: list[SourceSample] = []
    seen_images: dict[str, Path] = {}
    statistics = {
        "json": len(json_paths),
        "json_images": 0,
        "annotations": 0,
        "disk_images": 0,
        "unlisted_images": 0,
    }
    for json_path in json_paths:
        payload = _load_coco_payload(json_path)
        category_map = _coco_category_map(payload["categories"], json_path)
        image_index = _scene_image_index(json_path.parent.parent / "images", json_path)
        statistics["disk_images"] += len(image_index)
        coco_images: dict[int, tuple[Path, int, int]] = {}
        listed_names: set[str] = set()
        for index, item in enumerate(payload["images"]):
            if not isinstance(item, dict):
                raise PipelineError(f"COCO第{index}个image不是对象:{json_path}")
            image_id = item.get("id")
            file_name = str(item.get("file_name", "")).strip()
            width, height = item.get("width"), item.get("height")
            if isinstance(image_id, bool) or not isinstance(image_id, int):
                raise PipelineError(f"COCO image.id必须是整数:{json_path}:{index}")
            if image_id in coco_images:
                raise PipelineError(f"COCO image.id重复:{json_path}:{image_id}")
            if not file_name:
                raise PipelineError(f"COCO image.file_name为空:{json_path}:{image_id}")
            if (
                not isinstance(width, int)
                or not isinstance(height, int)
                or width <= 0
                or height <= 0
            ):
                raise PipelineError(f"COCO图片尺寸无效:{json_path}:{image_id}")
            file_key = Path(file_name.replace("\\", "/")).name.casefold()
            image_path = image_index.get(file_key)
            if image_path is None:
                raise PipelineError(f"COCO图片不存在:{json_path}:{file_name}")
            resolved_key = str(image_path).casefold()
            if resolved_key in seen_images:
                raise PipelineError(
                    f"同一图片被多个COCO JSON登记:{image_path};"
                    f"JSON={seen_images[resolved_key]} / {json_path}"
                )
            listed_names.add(file_key)
            seen_images[resolved_key] = json_path
            coco_images[image_id] = (image_path, width, height)
        statistics["json_images"] += len(coco_images)
        statistics["unlisted_images"] += len(set(image_index) - listed_names)
        annotations_by_image: dict[int, list[dict[str, Any]]] = {
            image_id: [] for image_id in coco_images
        }
        for index, annotation in enumerate(payload["annotations"]):
            if not isinstance(annotation, dict):
                raise PipelineError(f"COCO第{index}个annotation不是对象:{json_path}")
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            if image_id not in coco_images:
                raise PipelineError(f"COCO annotation引用未知image_id:{json_path}:{index}")
            if category_id not in category_map:
                raise PipelineError(f"COCO annotation引用未知category_id:{json_path}:{index}")
            annotations_by_image[image_id].append(annotation)
        statistics["annotations"] += len(payload["annotations"])
        for image_id, (image_path, width, height) in coco_images.items():
            image = cv2.imread(str(image_path))
            if image is None:
                raise PipelineError(f"COCO图片无法读取:{image_path}")
            actual_height, actual_width = image.shape[:2]
            if (actual_width, actual_height) != (width, height):
                raise PipelineError(
                    f"COCO图片尺寸不一致:{image_path};"
                    f"JSON={width}x{height},实际={actual_width}x{actual_height}"
                )
            retained_lines: list[str] = []
            ignore_boxes: list[tuple[float, float, float, float]] = []
            for annotation_index, annotation in enumerate(annotations_by_image[image_id]):
                description = f"{json_path}:image_id={image_id}:annotation={annotation_index}"
                normalized = _coco_box(annotation.get("bbox"), width, height, description)
                class_id = category_map[annotation["category_id"]]
                if class_id is None:
                    ignore_boxes.append(normalized)
                    continue
                retained_lines.append(
                    f"{class_id} {normalized[0]:.6f} {normalized[1]:.6f} "
                    f"{normalized[2]:.6f} {normalized[3]:.6f}"
                )
            samples.append(
                SourceSample(
                    image=image_path,
                    source_label=json_path,
                    retained_lines=tuple(retained_lines),
                    ignore_boxes=tuple(ignore_boxes),
                )
            )
    return samples, statistics


def collect_source_samples(
    source: Path,
    output: Path,
    ignore_class_ids: tuple[int, ...],
) -> tuple[list[SourceSample], dict[str, Any]]:
    """同时收集传统 YOLO TXT 和场景级 COCO JSON 检测样本."""
    txt_pairs = _collect_txt_pairs(source, output)
    txt_samples = _txt_samples(txt_pairs, ignore_class_ids)
    coco_samples, coco_statistics = _collect_coco_samples(source, output)
    samples = [*txt_samples, *coco_samples]
    seen: dict[str, Path] = {}
    for sample in samples:
        key = str(sample.image.resolve()).casefold()
        if key in seen:
            raise PipelineError(
                f"同一图片匹配到多个检测标签:{sample.image};"
                f"labels={seen[key]} / {sample.source_label}"
            )
        seen[key] = sample.source_label
    if not samples:
        raise PipelineError(
            f"未找到检测数据:{source};支持YOLO TXT或场景目录images配labels/COCO JSON"
        )
    return samples, {
        "format": (
            "mixed"
            if txt_samples and coco_samples
            else ("coco" if coco_samples else "yolo_txt")
        ),
        "txt_samples": len(txt_samples),
        "coco_samples": len(coco_samples),
        "coco": coco_statistics,
    }


def _blackout_ignore_boxes(image_path: Path, ignore_boxes: list[tuple[float, float, float, float]]) -> None:
    """将 YOLO 归一化 ignore 框区域涂黑并原地保存图片。"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise PipelineError(f"无法读取图片以处理ignore区域:{image_path}")
    height, width = image.shape[:2]
    for center_x, center_y, box_width, box_height in ignore_boxes:
        left = max(0, int((center_x - box_width / 2.0) * width))
        top = max(0, int((center_y - box_height / 2.0) * height))
        right = min(width - 1, int((center_x + box_width / 2.0) * width))
        bottom = min(height - 1, int((center_y + box_height / 2.0) * height))
        if right >= left and bottom >= top:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), thickness=-1)
    if not cv2.imwrite(str(image_path), image):
        raise PipelineError(f"保存ignore处理后的图片失败:{image_path}")


def _validate_dataset(dataset: Path, include_negative_samples: bool = True) -> dict[str, Any]:
    """校验规范化数据集的目录、图片标签一一对应和类别统计。"""
    totals: dict[str, Any] = {
        "images": {},
        "negative_images": {},
        "instances": {name: {} for name in CLASS_NAMES.values()},
    }
    for split in SPLITS:
        image_dir, label_dir = dataset / split / "images", dataset / split / "labels_detect"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise PipelineError(f"数据集缺少目录:{dataset}/{split}")
        images = sorted(item for item in image_dir.iterdir() if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES)
        image_stems, label_stems = {item.stem for item in images}, {item.stem for item in label_dir.glob("*.txt")}
        if image_stems != label_stems:
            raise PipelineError(f"{dataset}/{split}图片与标签未一一对应:images={len(image_stems)},labels={len(label_stems)}")
        counts: Counter[int] = Counter()
        negative_images = 0
        for label in label_dir.glob("*.txt"):
            label_counts = validate_label(label)
            counts.update(label_counts)
            if not label_counts:
                negative_images += 1
        if negative_images and not include_negative_samples:
            raise PipelineError(
                f"数据集包含{negative_images}个空标签负样本，"
                f"但PIPELINE_INCLUDE_NEGATIVE_SAMPLES=false:{dataset}/{split}"
            )
        totals["images"][split] = len(images)
        totals["negative_images"][split] = negative_images
        for class_id, name in CLASS_NAMES.items():
            totals["instances"][name][split] = counts[class_id]
    if not totals["images"]["train"] or not totals["images"]["val"]:
        raise PipelineError(f"训练集和验证集均必须至少包含一张图片:{dataset}")
    return totals


def _validate_registered_source(image_dir: Path, split: str) -> int:
    """校验注册表中一个独立 split 的图片和检测标签来源。"""
    label_dir = image_dir.parent / "labels_detect"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise PipelineError(
            f"注册表{split}检测来源无效:images={image_dir},labels_detect={label_dir}"
        )
    images = sorted(
        item
        for item in image_dir.rglob("*")
        if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
    )
    if not images:
        raise PipelineError(f"注册表{split}图片目录为空:{image_dir}")
    missing_labels: list[Path] = []
    for image in images:
        relative = image.relative_to(image_dir).with_suffix(".txt")
        label = label_dir / relative
        if not label.is_file():
            missing_labels.append(label)
            continue
        validate_label(label)
    if missing_labels:
        examples = "\n".join(str(item) for item in missing_labels[:20])
        raise PipelineError(
            f"注册表{split}图片缺少同名labels_detect标签:count={len(missing_labels)}\n{examples}"
        )
    return len(images)


def _collect_registered_statistics(yolo: dict[str, list[str]]) -> dict[str, Any]:
    """汇总注册表中实际参与训练的图片和逐类别检测实例数。"""
    statistics: dict[str, Any] = {
        "images": {split: 0 for split in SPLITS},
        "detect": {
            split: {name: 0 for name in CLASS_NAMES.values()}
            for split in SPLITS
        },
    }
    for split in SPLITS:
        for value in yolo[split]:
            image_dir = Path(value)
            label_dir = image_dir.parent / "labels_detect"
            statistics["images"][split] += _validate_registered_source(image_dir, split)
            images = (
                item for item in image_dir.rglob("*")
                if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
            )
            for image in images:
                label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
                for class_id, count in validate_label(label).items():
                    statistics["detect"][split][CLASS_NAMES[class_id]] += count
    return statistics


def prepare_dataset(context: Any) -> StageOutcome:
    """用 staging 原子生成每个输入目录对应的标准检测数据集。"""
    config: DetectConfig = context.config
    datasets: list[dict[str, Any]] = []
    for source, output in zip(config.input_dirs, config.output_dirs, strict=True):
        source, output = source.expanduser().resolve(), output.expanduser().resolve()
        if not source.is_dir():
            raise PipelineError(f"输入目录不存在:{source}")
        if output.exists():
            try:
                summary = _validate_dataset(output, config.include_negative_samples)
                datasets.append({"source": str(source), "output": str(output), "status": "reused", **summary})
                context.logger.info(
                    "DATASET_OUTPUT | status=reused | source=%s | output=%s | "
                    "train_images=%d | val_images=%d | detect_instances=%s",
                    source, output, summary["images"]["train"], summary["images"]["val"],
                    json.dumps(summary["instances"], ensure_ascii=False, sort_keys=True),
                )
                continue
            except PipelineError:
                if any(output.iterdir()):
                    raise PipelineError(f"输出目录已存在但不完整，请人工处理后重试:{output}")
        samples, input_statistics = collect_source_samples(
            source,
            output,
            config.ignore_class_ids,
        )
        positive_samples: list[SourceSample] = []
        negative_pairs = 0
        for sample in samples:
            if sample.retained_lines:
                positive_samples.append(sample)
            else:
                negative_pairs += 1
                if config.include_negative_samples:
                    positive_samples.append(sample)
        if not positive_samples:
            raise PipelineError(
                f"过滤空标签负样本后没有可用检测数据:{source};"
                "请设置PIPELINE_INCLUDE_NEGATIVE_SAMPLES=true或提供有目标标签"
            )
        skipped_negative = negative_pairs if not config.include_negative_samples else 0
        context.logger.info(
            "DATA_INPUT_FORMAT | source=%s | format=%s | txt_samples=%d | "
            "coco_samples=%d | coco=%s",
            source,
            input_statistics["format"],
            input_statistics["txt_samples"],
            input_statistics["coco_samples"],
            json.dumps(input_statistics["coco"], ensure_ascii=False, sort_keys=True),
        )
        context.logger.info(
            "DATA_NEGATIVE_FILTER | source=%s | paired=%d | positive=%d | "
            "negative=%d | include_negative=%s | skipped_negative=%d",
            source,
            len(samples),
            len(samples) - negative_pairs,
            negative_pairs,
            config.include_negative_samples,
            skipped_negative,
        )
        shuffled = list(positive_samples)
        random.Random(config.random_seed).shuffle(shuffled)
        val_count = min(max(1, round(len(shuffled) * config.val_ratio)), len(shuffled) - 1) if len(shuffled) > 1 else 0
        if val_count == 0:
            raise PipelineError(f"数据量不足，至少需要两张可配对图片:{source}")
        staging = output.with_name(f".{output.name}.{config.run_name}.staging")
        if staging.exists():
            raise PipelineError(f"发现未清理的 staging 目录，请检查后删除:{staging}")
        for split in SPLITS:
            (staging / split / "images").mkdir(parents=True)
            (staging / split / "labels_detect").mkdir(parents=True)
        manifest: list[dict[str, str]] = []
        ignore_images = 0
        ignore_boxes = 0
        try:
            for index, sample in enumerate(shuffled):
                split = "val" if index < val_count else "train"
                image = sample.image
                digest = hashlib.sha1(str(image.relative_to(source)).encode("utf-8")).hexdigest()[:10]
                stem = f"{image.stem}_{digest}"
                target_image = staging / split / "images" / f"{stem}{image.suffix.casefold()}"
                target_label = staging / split / "labels_detect" / f"{stem}.txt"
                shutil.copy2(image, target_image)
                if sample.ignore_boxes:
                    _blackout_ignore_boxes(target_image, list(sample.ignore_boxes))
                    ignore_images += 1
                    ignore_boxes += len(sample.ignore_boxes)
                target_label.write_text(
                    "\n".join(sample.retained_lines) + ("\n" if sample.retained_lines else ""),
                    encoding="utf-8",
                )
                manifest.append(
                    {
                        "split": split,
                        "source_image": str(image),
                        "source_label": str(sample.source_label),
                        "image": str(target_image.relative_to(staging)),
                        "label": str(target_label.relative_to(staging)),
                    }
                )
            atomic_write_text(staging / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            summary = _validate_dataset(staging, config.include_negative_samples)
            summary["ignore_images"] = ignore_images
            summary["ignore_boxes"] = ignore_boxes
            staging.replace(output)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        datasets.append({"source": str(source), "output": str(output), "status": "prepared", **summary})
        context.logger.info(
            "DATASET_OUTPUT | status=prepared | source=%s | output=%s | "
            "train_images=%d | val_images=%d | detect_instances=%s",
            source, output, summary["images"]["train"], summary["images"]["val"],
            json.dumps(summary["instances"], ensure_ascii=False, sort_keys=True),
        )
        context.logger.info(
            "DATA_IGNORE_PROCESS | source=%s | ignore_class_ids=%s | "
            "ignore_images=%d | ignore_boxes=%d",
            source,
            list(config.ignore_class_ids),
            ignore_images,
            ignore_boxes,
        )
    context.logger.info(
        "DATASET_OUTPUT_SUMMARY | datasets=%d | train_images=%d | val_images=%d",
        len(datasets),
        sum(item["images"]["train"] for item in datasets),
        sum(item["images"]["val"] for item in datasets),
    )
    return StageOutcome(message=f"检测数据处理完成:{len(datasets)}个目录", artifacts={"datasets": [item["output"] for item in datasets]}, metrics={"datasets": datasets}, value=datasets)


def _load_registry(path: Path) -> dict[str, Any]:
    """读取共用注册表，保持 PIDNet 等无关区段完全不变。"""
    if not path.is_file():
        return {"version": 2, "yolo": {"train": [], "val": []}, "pidnet": {"train": [], "val": []}}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"累计数据注册表读取失败:{path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("yolo"), dict):
        raise PipelineError(f"累计数据注册表缺少yolo区段:{path}")
    for split in SPLITS:
        if not isinstance(payload["yolo"].get(split, []), list):
            raise PipelineError(f"累计数据注册表yolo.{split}必须是列表")
    return payload


def _save_registry(context: Any, path: Path, payload: dict[str, Any]) -> None:
    """先备份再原子写入发生变化的共用注册表。"""
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    if path.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"datasets_backup_{stamp}.yaml")
        suffix = 1
        while backup.exists():
            backup = path.with_name(f"datasets_backup_{stamp}_{suffix}.yaml")
            suffix += 1
        shutil.copy2(path, backup)
        context.logger.info("REGISTRY_BACKUP | %s", backup)
    atomic_write_text(path, text)


def register_dataset(context: Any) -> StageOutcome:
    """仅更新或复用共享注册表的 yolo.train 与 yolo.val。"""
    config: DetectConfig = context.config
    path = config.registry_dir / "datasets.yaml"
    readonly = context.command == "train" or (context.command == "all" and not config.enable_data_update)
    if readonly and not path.is_file():
        raise PipelineError(f"PIPELINE_ENABLE_DATA_UPDATE=false时必须存在注册表:{path}")
    payload = _load_registry(path)
    yolo = payload["yolo"]
    for split in SPLITS:
        values = [str(Path(item).expanduser()) for item in yolo[split]]
        yolo[split] = list(dict.fromkeys(values))
    if not readonly:
        for output in config.output_dirs:
            _validate_dataset(output.expanduser().resolve(), config.include_negative_samples)
            for split in SPLITS:
                value = str((output.expanduser().resolve() / split / "images"))
                if value not in yolo[split]:
                    yolo[split].append(value)
        _save_registry(context, path, payload)
    for split in SPLITS:
        for value in yolo[split]:
            image_dir = Path(value)
            _validate_registered_source(image_dir, split)
    config_dir = context.work_dir / "training_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir / "yolo_detect.yaml"
    training_payload = {"path": str(context.work_dir), "train": yolo["train"], "val": yolo["val"], "labels_dir": "labels_detect", "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    atomic_write_text(yaml_path, yaml.safe_dump(training_payload, allow_unicode=True, sort_keys=False))
    stats = _collect_registered_statistics(yolo)
    labels = {
        split: sum(stats["detect"][split].values())
        for split in SPLITS
    }
    classes = {
        name: {split: stats["detect"][split][name] for split in SPLITS}
        for name in CLASS_NAMES.values()
    }
    context.logger.info(
        "TRAINING_DATASET_SUMMARY | train_images=%d | val_images=%d | "
        "train_labels=%d | val_labels=%d | classes=%s",
        stats["images"]["train"], stats["images"]["val"], labels["train"], labels["val"],
        json.dumps(classes, ensure_ascii=False, sort_keys=True),
    )
    atomic_write_text(context.work_dir / "reports" / "dataset_statistics.json", json.dumps(stats, ensure_ascii=False, indent=2))
    return StageOutcome(message="累计检测注册表已" + ("只读复用" if readonly else "更新"), artifacts={"registry": str(path), "detect": str(yaml_path)}, metrics=stats, value={"config": yaml_path, "registry": payload})
