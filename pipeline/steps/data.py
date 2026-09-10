"""原始 LabelMe 数据检查、复制、划分和三类标签生成."""

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

import cv2
import numpy as np

from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.io_utils import atomic_write_text
from ..core.stage_runner import StageOutcome
from ..core.state import StageStatus


SPLITS = ("train", "val")
DATA_SUBDIRECTORIES = ("img_src", "images", "json", "labels_detect", "labels_segment", "Seg")
DETECT_CLASSES = {"paper": 0, "liquid": 1, "metal": 2}
SEGMENT_CLASSES = {"paper": 0}
PIDNET_CLASSES = {"background": 0, "liquid": 1, "metal": 2}
BACKGROUND_LABELS = {"background", "__background__", "negative", "ignore"}
SOURCE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class SourcePair:
    """一对文件名相同的图片与 LabelMe JSON."""

    stem: str
    image_path: Path
    json_path: Path


@dataclass(frozen=True)
class CheckedPair:
    """完成图片、JSON、尺寸和 shape 检查的数据对."""

    source: SourcePair
    width: int
    height: int
    dimensions_repaired: bool


def read_image(path: Path) -> np.ndarray:
    """读取图片,兼容中文路径."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise PipelineError(f"无法读取图片:{path},原因:{exc}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError(f"图片损坏或格式错误:{path}")
    return image


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    """将图片以指定质量写为 JPG,兼容中文路径."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise PipelineError(f"JPG 编码失败:{path}")
    encoded.tofile(path)


def load_json(path: Path) -> dict[str, Any]:
    """读取并验证 LabelMe JSON 根结构."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"JSON 无法读取:{path},原因:{exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("shapes", []), list):
        raise PipelineError(f"JSON 根节点必须是对象且 shapes 必须是列表:{path}")
    return payload


def shape_to_polygon(shape: dict[str, Any]) -> np.ndarray | None:
    """将 LabelMe polygon、rectangle 和 circle 统一转换为多边形."""
    raw_points = shape.get("points", [])
    if not isinstance(raw_points, list):
        return None
    try:
        points = np.asarray(raw_points, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points[:, :2]).all():
        return None
    points = points[:, :2]
    shape_type = str(shape.get("shape_type", "polygon")).casefold()
    if shape_type == "rectangle":
        if len(points) != 2:
            return None
        (x1, y1), (x2, y2) = points
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
    if shape_type == "circle":
        if len(points) != 2:
            return None
        center, edge = points
        radius = float(np.linalg.norm(edge - center))
        if radius <= 0:
            return None
        angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
        return np.column_stack(
            (center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles))
        )
    return points if len(points) >= 3 else None


def clip_polygon(polygon: np.ndarray, width: int, height: int, yolo: bool = False) -> np.ndarray:
    """将多边形限制到图片边界或 YOLO 归一化边界."""
    clipped = polygon.astype(np.float64, copy=True)
    upper_x = width if yolo else max(width - 1, 0)
    upper_y = height if yolo else max(height - 1, 0)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, upper_x)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, upper_y)
    return clipped


def collect_unique_files(root: Path, suffixes: str | set[str], excluded: Path) -> dict[str, Path]:
    """递归收集文件并拒绝扁平化后重名的数据."""
    files: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    excluded_resolved = excluded.resolve()
    accepted_suffixes = {suffixes} if isinstance(suffixes, str) else suffixes
    accepted_suffixes = {suffix.casefold() for suffix in accepted_suffixes}
    suffix_description = "/".join(sorted(accepted_suffixes))
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in accepted_suffixes:
            continue
        try:
            path.resolve().relative_to(excluded_resolved)
            continue
        except ValueError:
            pass
        key = path.stem.casefold()
        if key in files:
            duplicates.setdefault(key, [files[key]]).append(path)
        else:
            files[key] = path
    if duplicates:
        details = ["、".join(map(str, paths)) for paths in list(duplicates.values())[:20]]
        raise PipelineError(f"发现 {len(duplicates)} 组同名{suffix_description}文件:\n" + "\n".join(details))
    return files


def collect_pairs(input_dir: Path, output_dir: Path) -> tuple[list[SourcePair], dict[str, int]]:
    """收集同名图片和 JSON 的交集,跳过没有配对的原始文件."""
    images = collect_unique_files(input_dir, SOURCE_IMAGE_SUFFIXES, output_dir)
    labels: dict[str, Path] = {}
    duplicate_labels: dict[str, list[Path]] = {}
    json_count = 0
    excluded_resolved = output_dir.resolve()
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() != ".json":
            continue
        try:
            path.resolve().relative_to(excluded_resolved)
            continue
        except ValueError:
            pass
        json_count += 1
        key = path.stem.casefold()
        if key not in images:
            continue
        if key in labels:
            duplicate_labels.setdefault(key, [labels[key]]).append(path)
        else:
            labels[key] = path
    if duplicate_labels:
        details = ["、".join(map(str, paths)) for paths in list(duplicate_labels.values())[:20]]
        raise PipelineError(f"发现 {len(duplicate_labels)} 组同名且可配对的 JSON 文件:\n" + "\n".join(details))
    paired_stems = sorted(set(images) & set(labels))
    statistics = {
        "images": len(images),
        "json": json_count,
        "paired": len(paired_stems),
        "images_skipped": len(images) - len(paired_stems),
        "json_skipped": json_count - len(paired_stems),
    }
    if not paired_stems:
        raise PipelineError(
            "输入目录中没有同名的图片与 JSON 数据对:"
            f"{input_dir};图片数量={statistics['images']},JSON数量={statistics['json']}"
        )
    return (
        [SourcePair(images[key].stem, images[key], labels[key]) for key in paired_stems],
        statistics,
    )


def check_pairs(pairs: list[SourcePair]) -> tuple[list[CheckedPair], Counter[str]]:
    """复制前集中检查全部图片、尺寸、类别和标注点."""
    checked: list[CheckedPair] = []
    labels: Counter[str] = Counter()
    errors: list[str] = []
    for pair in pairs:
        try:
            image = read_image(pair.image_path)
            height, width = image.shape[:2]
            payload = load_json(pair.json_path)
            json_width = payload.get("imageWidth")
            json_height = payload.get("imageHeight")
            repaired = json_width is None or json_height is None
            if json_width is not None and int(json_width) != width:
                raise PipelineError(f"图片宽度={width},JSON宽度={json_width}")
            if json_height is not None and int(json_height) != height:
                raise PipelineError(f"图片高度={height},JSON高度={json_height}")
            for index, shape in enumerate(payload.get("shapes", [])):
                if not isinstance(shape, dict):
                    raise PipelineError(f"第{index}个shape不是对象")
                label = str(shape.get("label", "")).strip()
                if not label:
                    raise PipelineError(f"第{index}个shape缺少label")
                if shape_to_polygon(shape) is None:
                    raise PipelineError(f"第{index}个shape({label})不是有效多边形")
                labels[label] += 1
            checked.append(CheckedPair(pair, width, height, repaired))
        except (PipelineError, TypeError, ValueError) as exc:
            errors.append(f"{pair.stem}:{exc}")
    if errors:
        raise PipelineError(
            f"完整性检查失败,共{len(errors)}个数据对存在问题:\n" + "\n".join(errors[:100])
        )
    return checked, labels


def split_stems(stems: list[str], val_ratio: float, seed: int) -> dict[str, list[str]]:
    """使用固定种子将文件名划分为训练集和验证集."""
    shuffled = sorted(stems)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        val_count = 0
    else:
        val_count = min(max(1, int(round(len(shuffled) * val_ratio))), len(shuffled) - 1)
    return {
        "train": sorted(shuffled[val_count:]),
        "val": sorted(shuffled[:val_count]),
    }


def normalized_polygon_line(class_id: int, polygon: np.ndarray, width: int, height: int) -> str:
    """生成一行 YOLO 实例分割标签."""
    clipped = clip_polygon(polygon, width, height, yolo=True)
    values = [f"{class_id}"]
    for x, y in clipped:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
    return " ".join(values)


def detection_box_line(
    class_id: int,
    polygon: np.ndarray,
    width: int,
    height: int,
) -> str | None:
    """根据多边形外接矩形生成一行 YOLO 检测标签."""
    clipped = clip_polygon(polygon, width, height, yolo=True)
    x_min, y_min = clipped.min(axis=0)
    x_max, y_max = clipped.max(axis=0)
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    return (
        f"{class_id} {center_x / width:.6f} {center_y / height:.6f} "
        f"{box_width / width:.6f} {box_height / height:.6f}"
    )


def write_label(path: Path, lines: list[str]) -> None:
    """写入 YOLO 标签;负样本保留空文件."""
    content = "\n".join(lines)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def process_pair(
    item: CheckedPair,
    split_dir: Path,
    jpg_quality: int,
) -> dict[str, Any]:
    """复制一个数据对并生成 ignore 图片、YOLO 标签和 PIDNet mask."""
    stem = item.source.stem
    source_image = read_image(item.source.image_path)
    payload = load_json(item.source.json_path)
    payload["imageWidth"] = item.width
    payload["imageHeight"] = item.height
    payload["imagePath"] = f"{stem}.jpg"

    source_suffix = item.source.image_path.suffix.casefold()
    shutil.copy2(item.source.image_path, split_dir / "img_src" / f"{stem}{source_suffix}")
    json_path = split_dir / "json" / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    processed_image = source_image.copy()
    segment_lines: list[str] = []
    detect_lines: list[str] = []
    positive_masks: list[tuple[int, np.ndarray]] = []
    background_masks: list[np.ndarray] = []
    result: dict[str, Any] = {
        "ignore_regions": 0,
        "detect": Counter(),
        "segment": Counter(),
        "pidnet_pixels": Counter(),
    }

    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        polygon = shape_to_polygon(shape)
        if polygon is None:
            continue
        if label == "ignore":
            points = np.rint(clip_polygon(polygon, item.width, item.height)).astype(np.int32)
            cv2.fillPoly(processed_image, [points], (0, 0, 0))
            result["ignore_regions"] += 1
        if label in SEGMENT_CLASSES:
            segment_lines.append(
                normalized_polygon_line(SEGMENT_CLASSES[label], polygon, item.width, item.height)
            )
            result["segment"][label] += 1
        if label in DETECT_CLASSES:
            line = detection_box_line(DETECT_CLASSES[label], polygon, item.width, item.height)
            if line is not None:
                detect_lines.append(line)
                result["detect"][label] += 1
        if label in PIDNET_CLASSES and label not in BACKGROUND_LABELS:
            positive_masks.append((PIDNET_CLASSES[label], polygon))
        else:
            background_masks.append(polygon)

    write_jpg(split_dir / "images" / f"{stem}.jpg", processed_image, jpg_quality)
    write_label(split_dir / "labels_segment" / f"{stem}.txt", segment_lines)
    write_label(split_dir / "labels_detect" / f"{stem}.txt", detect_lines)

    mask = np.zeros((item.height, item.width), dtype=np.uint8)
    for class_id, polygon in positive_masks:
        points = np.rint(clip_polygon(polygon, item.width, item.height)).astype(np.int32)
        cv2.fillPoly(mask, [points], int(class_id))
    for polygon in background_masks:
        points = np.rint(clip_polygon(polygon, item.width, item.height)).astype(np.int32)
        cv2.fillPoly(mask, [points], 0)
    for class_name, class_id in PIDNET_CLASSES.items():
        result["pidnet_pixels"][class_name] = int(np.count_nonzero(mask == class_id))
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise PipelineError(f"PIDNet mask 编码失败:{stem}")
    encoded.tofile(split_dir / "Seg" / f"{stem}.png")
    return result


def verify_prepared_dataset(dataset_dir: Path) -> dict[str, Any]:
    """验证标准数据集的目录、文件名集合和 mask 类别."""
    summary: dict[str, Any] = {"images": {}, "files": {}}
    for split in SPLITS:
        split_dir = dataset_dir / split
        missing_dirs = [name for name in DATA_SUBDIRECTORIES if not (split_dir / name).is_dir()]
        if missing_dirs:
            raise PipelineError(f"{split}缺少目录:{missing_dirs}")
        stems_by_dir: dict[str, set[str]] = {}
        suffixes = {
            "images": ".jpg",
            "json": ".json",
            "labels_detect": ".txt",
            "labels_segment": ".txt",
            "Seg": ".png",
        }
        stems_by_dir["img_src"] = {
            path.stem
            for path in (split_dir / "img_src").iterdir()
            if path.is_file() and path.suffix.casefold() in SOURCE_IMAGE_SUFFIXES
        }
        for name, suffix in suffixes.items():
            stems_by_dir[name] = {path.stem for path in (split_dir / name).glob(f"*{suffix}")}
        expected = stems_by_dir["images"]
        mismatched = {name: len(stems) for name, stems in stems_by_dir.items() if stems != expected}
        if mismatched:
            raise PipelineError(f"{split}文件未能一一对应:{mismatched}")
        for mask_path in (split_dir / "Seg").glob("*.png"):
            encoded = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise PipelineError(f"PIDNet mask不是单通道PNG:{mask_path}")
            invalid = set(np.unique(mask).tolist()) - set(PIDNET_CLASSES.values())
            if invalid:
                raise PipelineError(f"PIDNet mask包含非法类别{sorted(invalid)}:{mask_path}")
        summary["images"][split] = len(expected)
        summary["files"][split] = {name: len(stems) for name, stems in stems_by_dir.items()}
    return summary


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    """读取数据处理清单;不存在时返回 None."""
    path = dataset_dir / ".pipeline_dataset.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"数据处理清单损坏:{path},原因:{exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise PipelineError(f"数据处理清单版本不支持:{path}")
    return payload


def _files_equal(first: Path, second: Path) -> bool:
    """通过文件大小和 SHA256 判断两个原始图片是否完全一致."""
    if first.stat().st_size != second.stat().st_size:
        return False
    digests = []
    for path in (first, second):
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        digests.append(digest.digest())
    return digests[0] == digests[1]


def source_matches_prepared_dataset(input_dir: Path, dataset_dir: Path) -> bool:
    """判断改名后的原始目录是否仍对应现有数据集中的原图副本."""
    pairs, _ = collect_pairs(input_dir, dataset_dir)
    source = {pair.stem.casefold(): pair for pair in pairs}
    prepared_images = {
        path.stem.casefold(): path
        for split in SPLITS
        for path in (dataset_dir / split / "img_src").iterdir()
        if path.is_file() and path.suffix.casefold() in SOURCE_IMAGE_SUFFIXES
    }
    prepared_json = {
        path.stem.casefold(): path
        for split in SPLITS
        for path in (dataset_dir / split / "json").glob("*.json")
    }
    if set(source) != set(prepared_images) or set(source) != set(prepared_json):
        return False
    return all(
        _files_equal(source[stem].image_path, prepared_images[stem])
        and load_json(source[stem].json_path).get("shapes", [])
        == load_json(prepared_json[stem]).get("shapes", [])
        for stem in source
    )


def prepare_dataset(context: RunContext) -> StageOutcome:
    """按输入输出列表一对一处理或复用当前批次数据集。"""
    outcomes = [
        prepare_dataset_pair(context, input_dir, output_dir)
        for input_dir, output_dir in zip(context.config.input_dirs, context.config.output_dirs)
    ]
    train_images = 0
    val_images = 0
    for input_dir, output_dir, outcome in zip(
        context.config.input_dirs, context.config.output_dirs, outcomes
    ):
        statistics = outcome.metrics
        images = statistics.get("images", {})
        train_count = int(images.get("train", 0))
        val_count = int(images.get("val", 0))
        train_images += train_count
        val_images += val_count
        context.logger.info(
            "DATASET_OUTPUT | status=%s | source=%s | output=%s | "
            "train_images=%d | val_images=%d | detect_instances=%s",
            "reused" if outcome.status == StageStatus.SKIPPED else "prepared",
            input_dir,
            output_dir,
            train_count,
            val_count,
            json.dumps(statistics.get("detect", {}), ensure_ascii=False, sort_keys=True),
        )
    context.logger.info(
        "DATASET_OUTPUT_SUMMARY | datasets=%d | train_images=%d | val_images=%d",
        len(outcomes),
        train_images,
        val_images,
    )
    return StageOutcome(
        message=f"已处理或复用{len(outcomes)}个标准数据集",
        artifacts={
            f"dataset_{index}": str(output_dir)
            for index, output_dir in enumerate(context.config.output_dirs, start=1)
        },
        metrics={"datasets": [outcome.metrics for outcome in outcomes]},
        value=[outcome.value for outcome in outcomes],
    )


def prepare_dataset_pair(context: RunContext, input_dir: Path, dataset_dir: Path) -> StageOutcome:
    """处理或严格复用一组原始输入与标准数据集输出目录。"""
    config = context.config
    input_dir = input_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise PipelineError(f"原始数据目录不存在:{input_dir}")

    manifest = load_dataset_manifest(dataset_dir) if dataset_dir.is_dir() else None
    if manifest is not None:
        if Path(str(manifest.get("input_dir", ""))).resolve() != input_dir:
            verify_prepared_dataset(dataset_dir)
            if not source_matches_prepared_dataset(input_dir, dataset_dir):
                raise PipelineError(
                    f"数据集属于其他输入目录且原图内容不一致:清单={manifest.get('input_dir')},"
                    f"本次={input_dir}"
                )
            previous_input = manifest.get("input_dir")
            manifest["input_dir"] = str(input_dir)
            manifest["input_dir_rebound_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_text(
                dataset_dir / ".pipeline_dataset.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
            context.logger.warning("原始目录路径变化但文件完全一致,已安全重新绑定:%s -> %s", previous_input, input_dir)
        verify_prepared_dataset(dataset_dir)
        return StageOutcome(
            status=StageStatus.SKIPPED,
            message="标准数据集已完整存在",
            artifacts={"dataset_dir": str(dataset_dir), "manifest": str(dataset_dir / ".pipeline_dataset.json")},
            metrics=manifest.get("statistics", {}),
            value=manifest,
        )

    if dataset_dir.is_dir() and any(dataset_dir.iterdir()):
        try:
            verification = verify_prepared_dataset(dataset_dir)
        except PipelineError as exc:
            raise PipelineError(f"数据输出目录存在半成品且无法安全接管:{dataset_dir},原因:{exc}") from exc
        adopted = {
            "version": 1,
            "run_name": config.run_name,
            "input_dir": str(input_dir),
            "dataset_dir": str(dataset_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "adopted_existing": True,
            "statistics": verification,
        }
        atomic_write_text(
            dataset_dir / ".pipeline_dataset.json",
            json.dumps(adopted, ensure_ascii=False, indent=2),
        )
        return StageOutcome(
            status=StageStatus.SKIPPED,
            message="已验证并接管旧版标准数据集",
            artifacts={"dataset_dir": str(dataset_dir)},
            metrics=verification,
            value=adopted,
        )

    pairs, pair_statistics = collect_pairs(input_dir, dataset_dir)
    context.logger.info(
        "DATA_SOURCE_PAIRING | images=%d | json=%d | paired=%d | images_skipped=%d | json_skipped=%d",
        pair_statistics["images"],
        pair_statistics["json"],
        pair_statistics["paired"],
        pair_statistics["images_skipped"],
        pair_statistics["json_skipped"],
    )
    checked, source_labels = check_pairs(pairs)
    split_map = split_stems([item.source.stem for item in checked], config.val_ratio, config.random_seed)
    split_lookup = {stem: split for split, stems in split_map.items() for stem in stems}
    staging_dir = dataset_dir.with_name(f".{dataset_dir.name}.{config.run_name}.staging")
    if staging_dir.exists():
        if not staging_dir.is_dir() or staging_dir.is_symlink():
            raise PipelineError(f"准备阶段临时路径异常:{staging_dir}")
        shutil.rmtree(staging_dir)
        context.logger.warning("清理上次中断留下的数据准备临时目录:%s", staging_dir)
    staging_dir.mkdir(parents=True)
    for split in SPLITS:
        for subdir in DATA_SUBDIRECTORIES:
            (staging_dir / split / subdir).mkdir(parents=True)

    statistics: dict[str, Any] = {
        "images": {split: len(split_map[split]) for split in SPLITS},
        "detect": {split: {name: 0 for name in DETECT_CLASSES} for split in SPLITS},
        "segment": {split: {name: 0 for name in SEGMENT_CLASSES} for split in SPLITS},
        "pidnet_pixels": {split: {name: 0 for name in PIDNET_CLASSES} for split in SPLITS},
        "source_labels": dict(sorted(source_labels.items())),
        "repaired_dimensions": sum(item.dimensions_repaired for item in checked),
        "ignore_regions": 0,
    }
    try:
        for index, item in enumerate(checked, start=1):
            split = split_lookup[item.source.stem]
            result = process_pair(item, staging_dir / split, config.jpg_quality)
            statistics["ignore_regions"] += result["ignore_regions"]
            for task in ("detect", "segment", "pidnet_pixels"):
                for class_name, count in result[task].items():
                    statistics[task][split][class_name] += int(count)
            if index % 100 == 0 or index == len(checked):
                context.logger.info("DATA_PREPARE_PROGRESS | %d/%d", index, len(checked))
        verification = verify_prepared_dataset(staging_dir)
        statistics["verification"] = verification
        manifest = {
            "version": 1,
            "run_name": config.run_name,
            "input_dir": str(input_dir),
            "dataset_dir": str(dataset_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "adopted_existing": False,
            "statistics": statistics,
        }
        (staging_dir / ".pipeline_dataset.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if dataset_dir.exists():
            dataset_dir.rmdir()
        os.replace(staging_dir, dataset_dir)
    except Exception:
        context.logger.error("数据准备失败,临时目录保留用于排查:%s", staging_dir)
        raise
    return StageOutcome(
        message=f"数据处理完成:train={len(split_map['train'])},val={len(split_map['val'])}",
        artifacts={"dataset_dir": str(dataset_dir), "manifest": str(dataset_dir / ".pipeline_dataset.json")},
        metrics=statistics,
        value=manifest,
    )
