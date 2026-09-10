"""从多个 YOLO 检测数据集提取 OBJ 并合成检测训练数据."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_DIRECTORY_NAME = "images"
LABEL_DIRECTORY_NAME = "labels_detect"
LIQUID_MIN_INTERIOR_VISIBILITY = 4.0
LIQUID_MIN_EDGE_VISIBILITY = 18.0
LIQUID_VISIBILITY_BOOSTS = (1.0, 1.25, 1.50, 1.80)
OBJECT_VARIANT_ATTEMPTS = 5

# 只需要编辑此配置并直接运行脚本;首次运行或需要重建 OBJ 时设置 update_objects=True.
CONFIG: dict[str, Any] = {
    "update_objects": True,
    # 更新 OBJ 时可以填写多个 YOLO 检测数据集根目录.
    "source_datasets": [
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/shunyu260717/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/shunyu260717/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260721/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260721/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260729/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260729/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260730/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260730/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260731/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260731/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260812/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260812/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260813/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260813/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260819/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260819/val"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260820/train"),
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260820/val"),
    ],
    # OBJ 更新输出和 MixUp 输入共用同一个素材库目录.
    "object_library": Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/mixup_indoor_260825/objs"),
    # MixUp 可以填写多个背景目录或带 YOLO 标签的背景数据集.
    "background_datasets": [
        Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/mixup_indoor_260825/bg"),
    ],
    "output": Path("/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/mixup_indoor_260825/mixup"),
    "num_images": 500,
    "seed": 42,
    # None 表示使用全部类别;填写列表时,extract 和 compose 都只使用这些类别.
    # 液体不参与实例贴图,仅使用真实采集数据训练.
    "class_ids": [0, 2],
    "extract": {
        "bbox_padding": 0.12,
        "grabcut_iterations": 5,
        "min_object_area": 200,
    },
    "compose": {
        # 普通尺寸目标优先生成更密集的场景.
        "min_objects": 4,
        "max_objects": 8,
        # 实际目标框达到画面该比例时,该图停止继续贴入目标.
        "large_object_area_ratio": 0.50,
        "scale_range": (0.70, 1.30),
        "rotation_range": (-35.0, 35.0),
        # 液体实例贴图已停用,避免程序化边缘成为伪特征.
        "liquid_class_ids": [],
        "liquid_opacity_range": (0.55, 0.80),
        "max_overlap_iou": 0.05,
        "placement_attempts": 80,
        "image_attempts": 20,
        "jpeg_quality": 95,
    },
}


@dataclass(frozen=True, slots=True)
class DatasetPair:
    """保存一组同名图片和 YOLO 检测标签路径."""

    image: Path
    label: Path | None
    dataset_root: Path


@dataclass(frozen=True, slots=True)
class Detection:
    """保存一行归一化 YOLO 检测标注."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class ObjectRecord:
    """描述 OBJ 素材库中的一个透明目标."""

    object_id: str
    class_id: int
    rgba_path: str
    mask_path: str
    source_dataset: str
    source_image: str
    source_label: str
    source_object_index: int
    source_bbox: list[float]


def log_info(message: str) -> None:
    """立即刷新一条关键运行信息,避免长任务看起来无响应."""
    print(f"INFO:{message}", flush=True)


def validate_directory(path: Path, description: str) -> Path:
    """验证目录存在并返回绝对路径."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description}不存在:{resolved}")
    return resolved


def collect_images(root: Path, allow_direct_images: bool = False) -> list[Path]:
    """收集数据集 images 图片,并按需支持背景目录中的直接图片."""
    if allow_direct_images:
        direct_images = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        if direct_images:
            return direct_images
    if root.name.casefold() == IMAGE_DIRECTORY_NAME:
        image_directories = [root]
    else:
        image_directories = sorted(
            path
            for path in root.rglob("*")
            if path.is_dir() and path.name.casefold() == IMAGE_DIRECTORY_NAME
        )
    if not image_directories:
        if allow_direct_images:
            raise RuntimeError(f"背景目录中没有直接图片或 images 目录:{root}")
        raise RuntimeError(f"数据集未找到 images 目录:{root}")
    return sorted(
        image
        for directory in image_directories
        for image in directory.rglob("*")
        if image.is_file() and image.suffix.casefold() in IMAGE_SUFFIXES
    )


def _candidate_label_paths(image: Path, root: Path) -> list[Path]:
    """根据 images/labels_detect 结构生成图片标签候选路径."""
    if root.name.casefold() == IMAGE_DIRECTORY_NAME:
        return [
            root.parent
            / LABEL_DIRECTORY_NAME
            / image.relative_to(root).with_suffix(".txt")
        ]
    if image.parent == root:
        return [root / LABEL_DIRECTORY_NAME / image.with_suffix(".txt").name]
    relative = image.relative_to(root)
    candidates: list[Path] = []
    parts = relative.parts
    for index, part in enumerate(parts[:-1]):
        if part.casefold() != IMAGE_DIRECTORY_NAME:
            continue
        label_relative = Path(
            *parts[:index],
            LABEL_DIRECTORY_NAME,
            *parts[index + 1 :],
        ).with_suffix(".txt")
        candidates.append(root / label_relative)
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def find_label_path(image: Path, root: Path) -> Path | None:
    """查找图片唯一对应的 YOLO TXT 标签."""
    matches = [candidate for candidate in _candidate_label_paths(image, root) if candidate.is_file()]
    if len(matches) > 1:
        raise RuntimeError(f"图片匹配到多个标签:{image};labels={matches}")
    return matches[0] if matches else None


def discover_dataset_pairs(
    root: Path,
    require_labels: bool,
    allow_direct_images: bool = False,
) -> list[DatasetPair]:
    """发现一个数据集中的图片和 YOLO 标签配对."""
    images = collect_images(root, allow_direct_images=allow_direct_images)
    if not images:
        raise RuntimeError(f"输入目录中没有可用图片:{root}")
    pairs: list[DatasetPair] = []
    missing: list[Path] = []
    for image in images:
        label = find_label_path(image, root)
        if require_labels and label is None:
            missing.append(image)
            continue
        pairs.append(DatasetPair(image=image, label=label, dataset_root=root))
    if missing:
        examples = ",".join(str(path.relative_to(root)) for path in missing[:5])
        raise RuntimeError(f"数据集有 {len(missing)} 张图片缺少同名 YOLO 标签:{root};示例:{examples}")
    return pairs


def parse_yolo_label(path: Path) -> list[Detection]:
    """严格解析 YOLO 检测格式标签,拒绝分割格式."""
    detections: list[Detection] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"仅支持 YOLO 检测格式 class x y w h:{path}:{line_number}")
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"YOLO 标签字段不是有效数字:{path}:{line_number}") from exc
        if class_id < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"YOLO 标签类别或坐标无效:{path}:{line_number}")
        x_center, y_center, width, height = values
        if not all(0.0 <= value <= 1.0 for value in values) or width <= 0.0 or height <= 0.0:
            raise ValueError(f"YOLO 标签坐标必须归一化且宽高大于 0:{path}:{line_number}")
        detections.append(Detection(class_id, x_center, y_center, width, height))
    return detections


def read_rgb_image(path: Path) -> np.ndarray:
    """读取图片,应用 EXIF 方向并转换为 RGB 数组."""
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8).copy()


def detection_to_pixel_box(detection: Detection, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """将归一化检测框转换为裁剪安全的像素坐标."""
    left = math.floor((detection.x_center - detection.width / 2.0) * image_width)
    top = math.floor((detection.y_center - detection.height / 2.0) * image_height)
    right = math.ceil((detection.x_center + detection.width / 2.0) * image_width)
    bottom = math.ceil((detection.y_center + detection.height / 2.0) * image_height)
    left = min(max(left, 0), image_width - 1)
    top = min(max(top, 0), image_height - 1)
    right = min(max(right, left + 1), image_width)
    bottom = min(max(bottom, top + 1), image_height)
    return left, top, right, bottom


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """只保留二值 mask 中面积最大的连通区域."""
    component_count, component_map, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return np.zeros_like(mask)
    component_index = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    return np.where(component_map == component_index, 255, 0).astype(np.uint8)


def extract_object_rgba(
    rgb: np.ndarray,
    detection: Detection,
    bbox_padding: float,
    grabcut_iterations: int,
    min_object_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """使用 YOLO 检测框引导 GrabCut 提取透明目标和二值 mask."""
    image_height, image_width = rgb.shape[:2]
    left, top, right, bottom = detection_to_pixel_box(detection, image_width, image_height)
    box_width, box_height = right - left, bottom - top
    padding = max(2, int(round(max(box_width, box_height) * bbox_padding)))
    crop_left = max(0, left - padding)
    crop_top = max(0, top - padding)
    crop_right = min(image_width, right + padding)
    crop_bottom = min(image_height, bottom + padding)
    crop = rgb[crop_top:crop_bottom, crop_left:crop_right].copy()
    crop_height, crop_width = crop.shape[:2]
    rect_left = max(0, left - crop_left)
    rect_top = max(0, top - crop_top)
    rect_right = min(crop_width, right - crop_left)
    rect_bottom = min(crop_height, bottom - crop_top)
    if rect_right - rect_left < 2 or rect_bottom - rect_top < 2:
        return None
    grabcut_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    rectangle = (rect_left, rect_top, rect_right - rect_left, rect_bottom - rect_top)
    try:
        cv2.grabCut(
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            grabcut_mask,
            rectangle,
            background_model,
            foreground_model,
            grabcut_iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None
    mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _largest_component(mask)
    if int(np.count_nonzero(mask)) < min_object_area:
        return None
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    trimmed_rgb = crop[y : y + height, x : x + width]
    trimmed_mask = mask[y : y + height, x : x + width]
    rgba = np.dstack((trimmed_rgb, trimmed_mask))
    return rgba, trimmed_mask


def prepare_empty_directories(root: Path, names: Iterable[str]) -> dict[str, Path]:
    """创建空输出目录,并阻止覆盖已有文件."""
    resolved = root.expanduser().resolve()
    if resolved.is_dir():
        existing = next((path for path in resolved.rglob("*") if path.is_file()), None)
        if existing is not None:
            raise RuntimeError(f"输出目录必须为空,避免覆盖已有数据:{existing}")
    directories: dict[str, Path] = {}
    for name in names:
        directory = resolved / name
        directory.mkdir(parents=True, exist_ok=True)
        directories[name] = directory
    return directories


def _object_id(dataset_index: int, relative_image: Path, object_index: int) -> str:
    """生成跨数据集稳定且不冲突的 OBJ ID."""
    source = f"{dataset_index}:{relative_image.as_posix()}:{object_index}"
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", relative_image.stem).strip("_") or "object"
    return f"d{dataset_index:03d}_{stem}_{object_index:03d}_{digest}"


def run_extract(config: dict[str, Any]) -> None:
    """从多个 YOLO 检测数据集提取并落盘 OBJ 素材库."""
    extract_config = config["extract"]
    bbox_padding = float(extract_config["bbox_padding"])
    grabcut_iterations = int(extract_config["grabcut_iterations"])
    min_object_area = int(extract_config["min_object_area"])
    class_ids = config["class_ids"]
    if bbox_padding < 0.0:
        raise ValueError("CONFIG.extract.bbox_padding 必须大于等于 0")
    if grabcut_iterations <= 0 or min_object_area <= 0:
        raise ValueError("GrabCut 迭代次数和最小目标面积必须大于 0")
    if class_ids is not None and any(class_id < 0 for class_id in class_ids):
        raise ValueError("CONFIG.class_ids 不能包含负数")
    source_paths = [Path(path) for path in config["source_datasets"]]
    object_library = Path(config["object_library"])
    log_info(
        f"开始 OBJ 提取,数据集={len(source_paths)},输出={object_library},"
        f"GrabCut迭代={grabcut_iterations},最小面积={min_object_area}"
    )
    roots: list[Path] = []
    for dataset_index, path in enumerate(source_paths, start=1):
        log_info(f"验证源数据集 {dataset_index}/{len(source_paths)}:{path}")
        roots.append(validate_directory(path, "源数据集"))
    log_info(f"检查 OBJ 输出目录是否为空:{object_library}")
    prepare_empty_directories(object_library, ("objects", "masks"))
    output_root = object_library.expanduser().resolve()
    class_filter = set(class_ids) if class_ids is not None else None
    procedural_class_ids = {
        int(class_id) for class_id in config["compose"]["liquid_class_ids"]
    }
    records: list[ObjectRecord] = []
    stats: Counter[str] = Counter()
    for dataset_index, root in enumerate(roots, start=1):
        log_info(f"扫描源数据集 {dataset_index}/{len(roots)}:{root}")
        pairs = discover_dataset_pairs(root, require_labels=True)
        log_info(f"数据集扫描完成 {dataset_index}/{len(roots)},图片={len(pairs)}:{root}")
        dataset_stats: Counter[str] = Counter()
        progress = tqdm(
            pairs,
            desc=f"Extract {dataset_index}/{len(roots)}",
            unit="image",
            dynamic_ncols=True,
        )
        for pair in progress:
            detections = parse_yolo_label(pair.label) if pair.label is not None else []
            dataset_stats["images"] += 1
            dataset_stats["boxes"] += len(detections)
            if not detections:
                stats["negative_images"] += 1
                dataset_stats["negative_images"] += 1
            else:
                dataset_stats["positive_images"] += 1
                rgb = read_rgb_image(pair.image)
                for object_index, detection in enumerate(detections, start=1):
                    if class_filter is not None and detection.class_id not in class_filter:
                        stats["filtered_class"] += 1
                        dataset_stats["filtered_class"] += 1
                        continue
                    relative_image = pair.image.relative_to(root)
                    object_id = _object_id(dataset_index, relative_image, object_index)
                    source_bbox = [
                        detection.x_center,
                        detection.y_center,
                        detection.width,
                        detection.height,
                    ]
                    if detection.class_id in procedural_class_ids:
                        records.append(
                            ObjectRecord(
                                object_id=object_id,
                                class_id=detection.class_id,
                                rgba_path="",
                                mask_path="",
                                source_dataset=str(root),
                                source_image=str(pair.image),
                                source_label=str(pair.label),
                                source_object_index=object_index,
                                source_bbox=source_bbox,
                            )
                        )
                        stats[f"class_{detection.class_id}"] += 1
                        stats["procedural_shape_references"] += 1
                        stats["accepted"] += 1
                        dataset_stats[f"class_{detection.class_id}"] += 1
                        dataset_stats["procedural_shape_references"] += 1
                        dataset_stats["accepted"] += 1
                        continue
                    extracted = extract_object_rgba(
                        rgb,
                        detection,
                        bbox_padding,
                        grabcut_iterations,
                        min_object_area,
                    )
                    if extracted is None:
                        stats["extract_failed"] += 1
                        dataset_stats["extract_failed"] += 1
                        continue
                    rgba, mask = extracted
                    class_directory = str(detection.class_id)
                    rgba_relative = Path("objects") / class_directory / f"{object_id}.png"
                    mask_relative = Path("masks") / class_directory / f"{object_id}.png"
                    rgba_path = output_root / rgba_relative
                    mask_path = output_root / mask_relative
                    rgba_path.parent.mkdir(parents=True, exist_ok=True)
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba, mode="RGBA").save(rgba_path)
                    Image.fromarray(mask, mode="L").save(mask_path)
                    records.append(
                        ObjectRecord(
                            object_id=object_id,
                            class_id=detection.class_id,
                            rgba_path=rgba_relative.as_posix(),
                            mask_path=mask_relative.as_posix(),
                            source_dataset=str(root),
                            source_image=str(pair.image),
                            source_label=str(pair.label),
                            source_object_index=object_index,
                            source_bbox=source_bbox,
                        )
                    )
                    stats[f"class_{detection.class_id}"] += 1
                    stats["accepted"] += 1
                    dataset_stats[f"class_{detection.class_id}"] += 1
                    dataset_stats["accepted"] += 1
            progress.set_postfix(
                accepted=dataset_stats["accepted"],
                failed=dataset_stats["extract_failed"],
                filtered=dataset_stats["filtered_class"],
                refresh=False,
            )
        progress.close()
        log_info(
            f"数据集处理完成 {dataset_index}/{len(roots)},图片={dataset_stats['images']},"
            f"正样本={dataset_stats['positive_images']},空标签={dataset_stats['negative_images']},"
            f"检测框={dataset_stats['boxes']},接受={dataset_stats['accepted']},"
            f"失败={dataset_stats['extract_failed']},过滤={dataset_stats['filtered_class']}"
        )
    if not records:
        raise RuntimeError("没有成功提取任何 OBJ,请检查检测框、图片或 GrabCut 参数")
    metadata_text = "\n".join(json.dumps(asdict(record), ensure_ascii=False) for record in records) + "\n"
    (output_root / "metadata.jsonl").write_text(metadata_text, encoding="utf-8")
    summary = {"datasets": [str(root) for root in roots], "objects": len(records), "statistics": dict(stats)}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_info(f"OBJ 提取完成,数据集={len(roots)},目标={len(records)},输出={output_root}")
    log_info(f"提取统计={dict(stats)}")


def load_object_library(
    root: Path,
    procedural_class_ids: set[int] | None = None,
) -> list[ObjectRecord]:
    """读取 OBJ 素材库,并允许程序化类别只保留尺寸元数据."""
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OBJ 素材库缺少 metadata.jsonl:{root}")
    records: list[ObjectRecord] = []
    metadata_count = 0
    procedural_reference_count = 0
    removed_object_ids: list[str] = []
    missing_mask_ids: list[str] = []
    procedural_ids = procedural_class_ids or set()
    for line_number, raw_line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        metadata_count += 1
        try:
            record = ObjectRecord(**json.loads(raw_line))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"OBJ 元数据无效:{metadata_path}:{line_number}") from exc
        if record.class_id in procedural_ids:
            records.append(record)
            procedural_reference_count += 1
            continue
        if not (root / record.rgba_path).is_file():
            removed_object_ids.append(record.object_id)
            continue
        if not (root / record.mask_path).is_file():
            missing_mask_ids.append(record.object_id)
        records.append(record)
    log_info(
        f"OBJ 人工筛选统计,元数据={metadata_count},可用={len(records)},"
        f"程序化尺寸参考={procedural_reference_count},"
        f"已删除={len(removed_object_ids)},mask缺失={len(missing_mask_ids)}"
    )
    if removed_object_ids:
        log_info(f"已跳过删除的 OBJ,示例={removed_object_ids[:5]}")
    if missing_mask_ids:
        log_info(f"OBJ mask 缺失但不影响检测合成,示例={missing_mask_ids[:5]}")
    if not records:
        raise RuntimeError(f"人工筛选后没有可用 OBJ:{root}")
    return records


def _pixel_box_from_detection(detection: Detection, width: int, height: int) -> tuple[float, float, float, float]:
    """将检测标注转换为浮点像素框."""
    return (
        (detection.x_center - detection.width / 2.0) * width,
        (detection.y_center - detection.height / 2.0) * height,
        (detection.x_center + detection.width / 2.0) * width,
        (detection.y_center + detection.height / 2.0) * height,
    )


def _box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    """计算两个 xyxy 像素框的 IoU."""
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def augment_rgba(
    rgba_path: Path,
    scale_range: tuple[float, float],
    rotation_range: tuple[float, float],
    max_width: int,
    max_height: int,
    rng: random.Random,
) -> np.ndarray | None:
    """对透明 OBJ 执行缩放、翻转、旋转和轻微颜色增强."""
    with Image.open(rgba_path) as source:
        image = source.convert("RGBA")
    scale = rng.uniform(*scale_range)
    fit_scale = min(max_width / max(image.width, 1), max_height / max(image.height, 1))
    scale = min(scale, fit_scale)
    if scale <= 0.05:
        return None
    width = max(2, int(round(image.width * scale)))
    height = max(2, int(round(image.height * scale)))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.90, 1.10))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.92, 1.08))
    rgb.putalpha(image.getchannel("A"))
    image = rgb.rotate(
        rng.uniform(*rotation_range),
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )
    alpha_box = image.getchannel("A").getbbox()
    if alpha_box is None:
        return None
    return np.asarray(image.crop(alpha_box), dtype=np.uint8).copy()


def generate_procedural_liquid_rgba(
    record: ObjectRecord,
    scale_range: tuple[float, float],
    max_width: int,
    max_height: int,
    rng: random.Random,
) -> np.ndarray | None:
    """根据原检测框尺寸生成不规则透明水渍,不携带原背景像素."""
    if len(record.source_bbox) != 4:
        return None
    normalized_width = float(record.source_bbox[2])
    normalized_height = float(record.source_bbox[3])
    if normalized_width <= 0.0 or normalized_height <= 0.0:
        return None
    scale = rng.uniform(*scale_range)
    width = max(8, int(round(max_width * normalized_width * scale)))
    height = max(8, int(round(max_height * normalized_height * scale)))
    width = min(width, max_width)
    height = min(height, max_height)
    if width < 8 or height < 8:
        return None

    point_count = rng.randint(18, 30)
    radial_values = np.asarray(
        [rng.uniform(0.72, 1.0) for _ in range(point_count)],
        dtype=np.float32,
    )
    for _ in range(3):
        radial_values = (
            np.roll(radial_values, 1)
            + 2.0 * radial_values
            + np.roll(radial_values, -1)
        ) / 4.0
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius_x = max(2.0, width * rng.uniform(0.43, 0.49))
    radius_y = max(2.0, height * rng.uniform(0.43, 0.49))
    angle_offset = rng.uniform(0.0, 2.0 * math.pi)
    points: list[list[int]] = []
    for point_index, radial_value in enumerate(radial_values):
        angle = angle_offset + 2.0 * math.pi * point_index / point_count
        x = center_x + math.cos(angle) * radius_x * float(radial_value)
        y = center_y + math.sin(angle) * radius_y * float(radial_value)
        points.append([int(round(x)), int(round(y))])

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    blur_sigma = max(0.8, min(width, height) * rng.uniform(0.012, 0.025))
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    support = np.where(mask > 16, 255, 0).astype(np.uint8)
    support_points = cv2.findNonZero(support)
    if support_points is None:
        return None
    x, y, crop_width, crop_height = cv2.boundingRect(support_points)
    if crop_width < 8 or crop_height < 8:
        return None
    cropped_mask = mask[y : y + crop_height, x : x + crop_width]
    neutral_rgb = np.full((crop_height, crop_width, 3), 128, dtype=np.uint8)
    return np.dstack((neutral_rgb, cropped_mask))


def _liquid_transmission_rgb(
    roi: np.ndarray,
    rgba: np.ndarray,
    strength: float,
) -> np.ndarray:
    """根据背景亮度生成自适应液体透射图像."""
    source_rgb = rgba[:, :, :3].astype(np.float32)
    source_alpha = rgba[:, :, 3] > 16
    source_pixels = source_rgb[source_alpha]
    if source_pixels.size == 0:
        return roi.astype(np.float32)
    median_color = np.median(source_pixels, axis=0)
    luminance = float(np.dot(median_color, np.array([0.299, 0.587, 0.114])))
    if luminance <= 1.0:
        color_ratio = np.ones(3, dtype=np.float32)
    else:
        color_ratio = np.clip(median_color / luminance, 0.75, 1.25)
    roi_float = roi.astype(np.float32)
    roi_luminance = np.dot(roi_float, np.array([0.299, 0.587, 0.114]))
    local_luminance = float(np.mean(roi_luminance[source_alpha]))
    if local_luminance >= 128.0:
        brightness_ratio = max(0.85, 1.0 - 0.08 * strength)
    else:
        brightness_ratio = min(1.15, 1.0 + 0.08 * strength)
    tint_ratio = 1.0 + 0.25 * strength * (color_ratio - 1.0)
    # 仅保留 OBJ 的低频液体明暗,过滤其中携带的原背景细纹.
    blur_sigma = max(2.0, min(rgba.shape[:2]) / 12.0)
    source_low_frequency = cv2.GaussianBlur(source_rgb, (0, 0), blur_sigma)
    source_luminance = np.dot(source_low_frequency, np.array([0.299, 0.587, 0.114]))
    median_source_luminance = max(float(np.median(source_luminance[source_alpha])), 1.0)
    shape_ratio = np.clip(source_luminance / median_source_luminance, 0.90, 1.10)
    softened = cv2.GaussianBlur(roi_float, (5, 5), 0)
    transmitted = roi_float * 0.80 + softened * 0.20
    transmitted *= 1.0 + 0.25 * strength * (shape_ratio[:, :, None] - 1.0)
    return np.clip(transmitted * brightness_ratio * tint_ratio, 0, 255)


def _liquid_rim_mask(rgba: np.ndarray) -> np.ndarray:
    """根据液体尺寸生成位于目标内部的柔和边缘权重."""
    mask = np.where(rgba[:, :, 3] > 16, 255, 0).astype(np.uint8)
    min_dimension = min(mask.shape)
    rim_width = max(2, min(7, int(round(min_dimension * 0.04))))
    kernel_size = rim_width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask, kernel, iterations=1)
    inner_rim = (mask.astype(np.float32) - eroded.astype(np.float32)) / 255.0
    blurred = cv2.GaussianBlur(inner_rim, (0, 0), max(0.8, rim_width / 2.0))
    return np.clip(blurred * (mask.astype(np.float32) / 255.0), 0.0, 1.0)


def _add_liquid_rim(
    candidate: np.ndarray,
    roi: np.ndarray,
    rgba: np.ndarray,
    opacity: float,
    strength: float,
) -> np.ndarray:
    """根据液体色调和背景亮度生成柔和湿润边缘."""
    source_rgb = rgba[:, :, :3].astype(np.float32)
    support = rgba[:, :, 3] > 16
    median_color = np.median(source_rgb[support], axis=0)
    source_luminance = max(float(np.dot(median_color, np.array([0.299, 0.587, 0.114]))), 1.0)
    color_ratio = np.clip(median_color / source_luminance, 0.75, 1.25)
    local_luminance = float(
        np.mean(np.dot(roi.astype(np.float32), np.array([0.299, 0.587, 0.114]))[support])
    )
    edge_luminance = (
        max(18.0, local_luminance - 55.0 * strength)
        if local_luminance >= 128.0
        else min(237.0, local_luminance + 55.0 * strength)
    )
    edge_color = np.clip(edge_luminance * color_ratio, 0, 255)
    if float(np.max(color_ratio) - np.min(color_ratio)) < 0.08:
        edge_color = np.clip(edge_color * np.array([0.94, 1.01, 1.08]), 0, 255)
    rim = _liquid_rim_mask(rgba)
    height, width = rim.shape
    y_weight = np.linspace(1.0, 0.0, max(height, 1), dtype=np.float32)[:, None]
    x_weight = np.linspace(1.0, 0.0, max(width, 1), dtype=np.float32)[None, :]
    highlight_weight = (x_weight + y_weight) * 0.5
    rim_alpha = rim * min(0.90, (0.45 + 0.18 * strength) * opacity)
    rim_alpha_3d = rim_alpha[:, :, None]
    colored = candidate * (1.0 - rim_alpha_3d) + edge_color * rim_alpha_3d
    highlight = rim_alpha * highlight_weight * 12.0
    return np.clip(colored + highlight[:, :, None], 0, 255)


def _liquid_visibility(
    candidate: np.ndarray,
    roi: np.ndarray,
    rgba: np.ndarray,
) -> tuple[float, float]:
    """计算液体内部和柔和边缘相对原背景的平均像素差."""
    support = rgba[:, :, 3] > 16
    if not np.any(support):
        return 0.0, 0.0
    difference = np.mean(
        np.abs(candidate.astype(np.float32) - roi.astype(np.float32)),
        axis=2,
    )
    rim_support = _liquid_rim_mask(rgba) > 0.10
    interior_visibility = float(np.mean(difference[support]))
    edge_visibility = (
        float(np.mean(difference[rim_support]))
        if np.any(rim_support)
        else interior_visibility
    )
    return interior_visibility, edge_visibility


def paste_rgba(
    canvas: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int,
    liquid_opacity: float | None = None,
) -> tuple[float, float] | None:
    """融合 OBJ 并返回可见性,不可见液体不修改背景."""
    height, width = rgba.shape[:2]
    roi = canvas[y : y + height, x : x + width]
    base_alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    base_alpha = cv2.GaussianBlur(base_alpha, (3, 3), 0)
    if liquid_opacity is None:
        foreground = rgba[:, :, :3].astype(np.float32)
        alpha = np.clip(base_alpha, 0.0, 1.0)[:, :, None]
        blended = foreground * alpha + roi.astype(np.float32) * (1.0 - alpha)
        candidate = np.clip(blended, 0, 255).astype(np.uint8)
        visibility = _liquid_visibility(candidate, roi, rgba)
        canvas[y : y + height, x : x + width] = candidate
        return visibility
    roi_float = roi.astype(np.float32)
    for strength in LIQUID_VISIBILITY_BOOSTS:
        opacity = min(1.0, liquid_opacity * strength)
        alpha = np.clip(base_alpha * opacity, 0.0, 1.0)[:, :, None]
        foreground = _liquid_transmission_rgb(roi, rgba, strength)
        blended = foreground * alpha + roi_float * (1.0 - alpha)
        candidate = _add_liquid_rim(blended, roi, rgba, opacity, strength).astype(np.uint8)
        interior_visibility, edge_visibility = _liquid_visibility(candidate, roi, rgba)
        if (
            interior_visibility >= LIQUID_MIN_INTERIOR_VISIBILITY
            and edge_visibility >= LIQUID_MIN_EDGE_VISIBILITY
        ):
            canvas[y : y + height, x : x + width] = candidate
            return interior_visibility, edge_visibility
    return None


def place_object(
    canvas: np.ndarray,
    rgba: np.ndarray,
    occupied_boxes: list[tuple[float, float, float, float]],
    max_overlap_iou: float,
    placement_attempts: int,
    rng: random.Random,
    liquid_opacity: float | None = None,
) -> tuple[float, float, float, float] | None:
    """寻找低重叠位置,贴入目标并返回实际前景像素框."""
    image_height, image_width = canvas.shape[:2]
    object_height, object_width = rgba.shape[:2]
    if object_width > image_width or object_height > image_height:
        return None
    alpha_points = cv2.findNonZero(np.where(rgba[:, :, 3] > 16, 255, 0).astype(np.uint8))
    if alpha_points is None:
        return None
    mask_x, mask_y, mask_width, mask_height = cv2.boundingRect(alpha_points)
    for _ in range(placement_attempts):
        x = rng.randint(0, image_width - object_width)
        y = rng.randint(0, image_height - object_height)
        box = (
            float(x + mask_x),
            float(y + mask_y),
            float(x + mask_x + mask_width),
            float(y + mask_y + mask_height),
        )
        if any(_box_iou(box, occupied) > max_overlap_iou for occupied in occupied_boxes):
            continue
        visibility = paste_rgba(canvas, rgba, x, y, liquid_opacity=liquid_opacity)
        if visibility is None:
            continue
        return box
    return None


def is_large_placed_object(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    area_ratio: float,
) -> bool:
    """判断已贴入目标的实际包围框是否属于大尺寸目标."""
    box_width = max(0.0, box[2] - box[0])
    box_height = max(0.0, box[3] - box[1])
    image_area = max(1, image_width * image_height)
    return box_width * box_height / image_area >= area_ratio


def _detection_from_pixel_box(
    class_id: int,
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Detection:
    """将最终 OBJ 像素框转换为 YOLO 检测标注."""
    left, top, right, bottom = box
    return Detection(
        class_id=class_id,
        x_center=((left + right) / 2.0) / image_width,
        y_center=((top + bottom) / 2.0) / image_height,
        width=(right - left) / image_width,
        height=(bottom - top) / image_height,
    )


def format_detection(detection: Detection) -> str:
    """将检测标注格式化为稳定的 YOLO TXT 行."""
    return (
        f"{detection.class_id} {detection.x_center:.6f} {detection.y_center:.6f} "
        f"{detection.width:.6f} {detection.height:.6f}"
    )


def _records_by_class(records: list[ObjectRecord]) -> dict[int, list[ObjectRecord]]:
    """按类别 ID 组织 OBJ 素材并保持稳定顺序."""
    grouped: dict[int, list[ObjectRecord]] = {}
    for record in records:
        grouped.setdefault(record.class_id, []).append(record)
    return grouped


def _is_carpet_background(pair: DatasetPair) -> bool:
    """判断背景图片文件名是否包含 Carpet 标记."""
    return "carpet" in pair.image.name.casefold()


def _balanced_class_plan(
    class_ids: list[int],
    pasted_counts: Counter[int],
    target_count: int,
    rng: random.Random,
) -> list[int]:
    """优先选择累计数量最少的类别并生成本图粘贴计划."""
    planned_counts = Counter({class_id: pasted_counts[class_id] for class_id in class_ids})
    plan: list[int] = []
    for _ in range(target_count):
        minimum = min(planned_counts[class_id] for class_id in class_ids)
        candidates = [class_id for class_id in class_ids if planned_counts[class_id] == minimum]
        class_id = rng.choice(candidates)
        plan.append(class_id)
        planned_counts[class_id] += 1
    return plan


def _balanced_background_plan(
    background_pairs: list[DatasetPair],
    target_count: int,
    rng: random.Random,
) -> list[DatasetPair]:
    """按轮次打乱背景素材,使每张背景的使用次数尽量接近."""
    plan: list[DatasetPair] = []
    while len(plan) < target_count:
        round_pairs = list(background_pairs)
        rng.shuffle(round_pairs)
        remaining = target_count - len(plan)
        plan.extend(round_pairs[:remaining])
    return plan


def synthesize_image(
    background: DatasetPair,
    object_root: Path,
    objects_by_class: dict[int, list[ObjectRecord]],
    pasted_counts: Counter[int],
    compose_config: dict[str, Any],
    rng: random.Random,
) -> tuple[np.ndarray, list[Detection], dict[str, object]] | None:
    """尝试生成一张带 YOLO 检测标注的合成图片."""
    class_ids = sorted(objects_by_class)
    liquid_class_ids = {int(class_id) for class_id in compose_config["liquid_class_ids"]}
    liquid_opacity_range = tuple(float(value) for value in compose_config["liquid_opacity_range"])
    large_object_area_ratio = float(compose_config["large_object_area_ratio"])
    eligible_class_ids = [
        class_id
        for class_id in class_ids
        if not (_is_carpet_background(background) and class_id in liquid_class_ids)
    ]
    if not eligible_class_ids:
        return None
    placement_failures = 0
    procedural_liquid_failures = 0
    for image_attempt in range(1, int(compose_config["image_attempts"]) + 1):
        target_count = rng.randint(
            int(compose_config["min_objects"]),
            int(compose_config["max_objects"]),
        )
        planned_classes = _balanced_class_plan(
            eligible_class_ids,
            pasted_counts,
            target_count,
            rng,
        )
        canvas = read_rgb_image(background.image)
        image_height, image_width = canvas.shape[:2]
        existing_detections = parse_yolo_label(background.label) if background.label is not None else []
        occupied_boxes = [
            _pixel_box_from_detection(detection, image_width, image_height)
            for detection in existing_detections
        ]
        generated: list[Detection] = []
        used_objects: list[str] = []
        plan_failed = False
        for class_id in planned_classes:
            placed: tuple[ObjectRecord, tuple[float, float, float, float]] | None = None
            for _ in range(OBJECT_VARIANT_ATTEMPTS):
                record = rng.choice(objects_by_class[class_id])
                if class_id in liquid_class_ids:
                    rgba = generate_procedural_liquid_rgba(
                        record,
                        tuple(compose_config["scale_range"]),
                        max(2, image_width - 2),
                        max(2, image_height - 2),
                        rng,
                    )
                else:
                    rgba = augment_rgba(
                        object_root / record.rgba_path,
                        tuple(compose_config["scale_range"]),
                        tuple(compose_config["rotation_range"]),
                        max(2, image_width - 2),
                        max(2, image_height - 2),
                        rng,
                    )
                if rgba is None:
                    placement_failures += 1
                    if class_id in liquid_class_ids:
                        procedural_liquid_failures += 1
                    continue
                liquid_opacity = (
                    rng.uniform(*liquid_opacity_range)
                    if class_id in liquid_class_ids
                    else None
                )
                box = place_object(
                    canvas,
                    rgba,
                    occupied_boxes,
                    float(compose_config["max_overlap_iou"]),
                    int(compose_config["placement_attempts"]),
                    rng,
                    liquid_opacity=liquid_opacity,
                )
                if box is not None:
                    placed = record, box
                    break
                placement_failures += 1
                if class_id in liquid_class_ids:
                    procedural_liquid_failures += 1
            if placed is None:
                plan_failed = True
                break
            record, box = placed
            occupied_boxes.append(box)
            generated.append(_detection_from_pixel_box(class_id, box, image_width, image_height))
            used_objects.append(record.object_id)
            if is_large_placed_object(
                box,
                image_width,
                image_height,
                large_object_area_ratio,
            ):
                target_count = len(generated)
                break
        if not plan_failed and len(generated) == target_count:
            provenance: dict[str, object] = {
                "background": str(background.image),
                "background_label": str(background.label) if background.label is not None else None,
                "background_is_carpet": _is_carpet_background(background),
                "objects": used_objects,
                "generated_class_ids": [detection.class_id for detection in generated],
                "procedural_liquid_count": sum(
                    detection.class_id in liquid_class_ids for detection in generated
                ),
                "procedural_liquid_failures": procedural_liquid_failures,
                "large_object_count": sum(
                    is_large_placed_object(
                        _pixel_box_from_detection(detection, image_width, image_height),
                        image_width,
                        image_height,
                        large_object_area_ratio,
                    )
                    for detection in generated
                ),
                "image_attempt": image_attempt,
                "placement_failures": placement_failures,
            }
            return canvas, [*existing_detections, *generated], provenance
    return None


def run_compose(config: dict[str, Any]) -> None:
    """读取 OBJ 素材库并生成 YOLO 检测合成数据集."""
    compose_config = config["compose"]
    num_images = int(config["num_images"])
    min_objects = int(compose_config["min_objects"])
    max_objects = int(compose_config["max_objects"])
    scale_range = tuple(compose_config["scale_range"])
    rotation_range = tuple(compose_config["rotation_range"])
    liquid_class_ids = [int(class_id) for class_id in compose_config["liquid_class_ids"]]
    liquid_opacity_range = tuple(float(value) for value in compose_config["liquid_opacity_range"])
    large_object_area_ratio = float(compose_config["large_object_area_ratio"])
    max_overlap_iou = float(compose_config["max_overlap_iou"])
    placement_attempts = int(compose_config["placement_attempts"])
    image_attempts = int(compose_config["image_attempts"])
    jpeg_quality = int(compose_config["jpeg_quality"])
    if num_images <= 0:
        raise ValueError("CONFIG.num_images 必须大于 0")
    if min_objects <= 0 or max_objects < min_objects:
        raise ValueError("目标数量范围无效")
    if not 0.0 < large_object_area_ratio <= 1.0:
        raise ValueError("CONFIG.compose.large_object_area_ratio 必须在 0 到 1 之间")
    if not 0.0 <= max_overlap_iou <= 1.0:
        raise ValueError("CONFIG.compose.max_overlap_iou 必须在 0 和 1 之间")
    if scale_range[0] <= 0.0 or scale_range[1] < scale_range[0]:
        raise ValueError("CONFIG.compose.scale_range 无效")
    if rotation_range[1] < rotation_range[0]:
        raise ValueError("CONFIG.compose.rotation_range 无效")
    if any(class_id < 0 for class_id in liquid_class_ids):
        raise ValueError("CONFIG.compose.liquid_class_ids 不能包含负数")
    if (
        len(liquid_opacity_range) != 2
        or not 0.0 < liquid_opacity_range[0] <= liquid_opacity_range[1] <= 1.0
    ):
        raise ValueError("CONFIG.compose.liquid_opacity_range 必须是 0 到 1 之间的递增范围")
    if placement_attempts <= 0 or image_attempts <= 0:
        raise ValueError("目标放置和图片重试次数必须大于 0")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("CONFIG.compose.jpeg_quality 必须在 1 和 100 之间")
    log_info(f"开始合成,目标图片={num_images},OBJ素材库={config['object_library']}")
    log_info("加载并验证 OBJ 素材库")
    object_root = validate_directory(Path(config["object_library"]), "OBJ 素材库")
    records = load_object_library(object_root, set(liquid_class_ids))
    objects_by_class = _records_by_class(records)
    configured_class_ids = config["class_ids"]
    if configured_class_ids is not None:
        selected_class_ids = sorted({int(class_id) for class_id in configured_class_ids})
        missing_class_ids = [
            class_id
            for class_id in selected_class_ids
            if class_id not in objects_by_class
        ]
        if missing_class_ids:
            raise RuntimeError(f"OBJ 素材库缺少配置类别:{missing_class_ids}")
        objects_by_class = {class_id: objects_by_class[class_id] for class_id in selected_class_ids}
    if not objects_by_class:
        raise RuntimeError("没有可用于合成的 OBJ 类别")
    unknown_liquid_class_ids = sorted(set(liquid_class_ids) - set(objects_by_class))
    if unknown_liquid_class_ids:
        raise RuntimeError(f"液体类别不在合成类别中:{unknown_liquid_class_ids}")
    class_summary = {class_id: len(items) for class_id, items in sorted(objects_by_class.items())}
    log_info(f"OBJ 素材加载完成,总数={len(records)},类别统计={class_summary}")
    log_info(f"类别均衡配置,类别={sorted(objects_by_class)}")
    if liquid_class_ids:
        log_info(
            f"液体透射配置,类别={liquid_class_ids},透明度范围={liquid_opacity_range},"
            f"内部可见性阈值={LIQUID_MIN_INTERIOR_VISIBILITY},"
            f"边缘可见性阈值={LIQUID_MIN_EDGE_VISIBILITY}"
        )
    else:
        log_info("液体实例贴图已停用,液体类别仅应使用真实采集数据")
    log_info(
        f"单图目标数量={min_objects}-{max_objects},大目标阈值={large_object_area_ratio:.0%}"
    )
    background_paths = [Path(path) for path in config["background_datasets"]]
    background_roots: list[Path] = []
    for dataset_index, path in enumerate(background_paths, start=1):
        log_info(f"验证背景数据集 {dataset_index}/{len(background_paths)}:{path}")
        background_roots.append(validate_directory(path, "背景数据集"))
    background_pairs: list[DatasetPair] = []
    for dataset_index, root in enumerate(background_roots, start=1):
        log_info(f"扫描背景数据集 {dataset_index}/{len(background_roots)}:{root}")
        pairs = discover_dataset_pairs(
            root,
            require_labels=False,
            allow_direct_images=True,
        )
        background_pairs.extend(pairs)
        labeled_images = sum(pair.label is not None for pair in pairs)
        log_info(
            f"背景扫描完成 {dataset_index}/{len(background_roots)},"
            f"图片={len(pairs)},带标签={labeled_images}:{root}"
        )
    if not background_pairs:
        raise RuntimeError("没有可用背景图片")
    carpet_background_pairs = [pair for pair in background_pairs if _is_carpet_background(pair)]
    non_carpet_background_pairs = [pair for pair in background_pairs if not _is_carpet_background(pair)]
    log_info(
        f"背景统计,总数={len(background_pairs)},Carpet={len(carpet_background_pairs)},"
        f"非Carpet={len(non_carpet_background_pairs)}"
    )
    if liquid_class_ids and not non_carpet_background_pairs:
        raise RuntimeError("液体类别已启用,但没有文件名不含 Carpet 的可用背景")
    output = Path(config["output"])
    log_info(f"检查合成输出目录是否为空:{output}")
    directories = prepare_empty_directories(output, ("images", "labels_detect"))
    output_root = output.expanduser().resolve()
    rng = random.Random(int(config["seed"]))
    background_plan = _balanced_background_plan(background_pairs, num_images, rng)
    pasted_counts: Counter[int] = Counter()
    final_counts: Counter[int] = Counter()
    used_background_counts: Counter[str] = Counter()
    used_background_kinds: Counter[str] = Counter()
    output_stem_counts: Counter[str] = Counter()
    provenance_lines: list[str] = []
    progress = tqdm(
        enumerate(background_plan, start=1),
        total=num_images,
        desc="Compose",
        unit="image",
        dynamic_ncols=True,
    )
    total_objects = 0
    total_retries = 0
    total_placement_failures = 0
    total_procedural_liquids = 0
    total_procedural_liquid_failures = 0
    for index, background in progress:
        result = synthesize_image(
            background,
            object_root,
            objects_by_class,
            pasted_counts,
            compose_config,
            rng,
        )
        if result is None:
            raise RuntimeError(f"第 {index} 张图片无法在重试次数内完成目标放置")
        rgb, detections, provenance = result
        background_stem = background.image.stem
        output_stem_counts[background_stem] += 1
        filename = f"{background_stem}_mixup_{output_stem_counts[background_stem]:03d}"
        image_path = directories["images"] / f"{filename}.jpg"
        label_path = directories["labels_detect"] / f"{filename}.txt"
        Image.fromarray(rgb, mode="RGB").save(
            image_path,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
        )
        label_path.write_text("\n".join(format_detection(item) for item in detections) + "\n", encoding="utf-8")
        for detection in detections:
            final_counts[detection.class_id] += 1
        for class_id in provenance["generated_class_ids"]:
            pasted_counts[int(class_id)] += 1
        background_key = str(background.image)
        background_kind = "carpet" if _is_carpet_background(background) else "non_carpet"
        used_background_counts[background_key] += 1
        used_background_kinds[background_kind] += 1
        provenance.update(output_image=image_path.name, output_label=label_path.name)
        provenance_lines.append(json.dumps(provenance, ensure_ascii=False))
        total_objects += len(provenance["objects"])
        total_retries += int(provenance["image_attempt"]) - 1
        total_placement_failures += int(provenance["placement_failures"])
        total_procedural_liquids += int(provenance["procedural_liquid_count"])
        total_procedural_liquid_failures += int(provenance["procedural_liquid_failures"])
        progress.set_postfix(
            objects=total_objects,
            balance="/".join(str(pasted_counts[class_id]) for class_id in sorted(objects_by_class)),
            retries=total_retries,
            place_fail=total_placement_failures,
            refresh=False,
        )
    progress.close()
    (output_root / "provenance.jsonl").write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")
    summary = {
        "images": num_images,
        "object_library": str(object_root),
        "background_datasets": [str(root) for root in background_roots],
        "balanced_class_ids": sorted(objects_by_class),
        "pasted_label_counts": {str(key): value for key, value in sorted(pasted_counts.items())},
        "final_label_counts": {str(key): value for key, value in sorted(final_counts.items())},
        "available_background_counts": {
            "carpet": len(carpet_background_pairs),
            "non_carpet": len(non_carpet_background_pairs),
        },
        "output_background_counts": {
            key: value for key, value in sorted(used_background_kinds.items())
        },
        "background_usage_range": {
            "min": min(used_background_counts.values(), default=0),
            "max": max(used_background_counts.values(), default=0),
        },
        "procedural_liquid_count": total_procedural_liquids,
        "procedural_liquid_failures": total_procedural_liquid_failures,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_info(
        f"YOLO 检测合成完成,图片={num_images},新增目标={total_objects},"
        f"新增类别统计={dict(sorted(pasted_counts.items()))},"
        f"换背景重试={total_retries},放置失败={total_placement_failures},输出={output_root}"
    )


def rebuild_object_library(config: dict[str, Any]) -> None:
    """在临时同级目录重建 OBJ,成功后备份并替换现有素材库."""
    object_root = Path(config["object_library"]).expanduser().resolve()
    staging_root = object_root.with_name(f"{object_root.name}.building")
    if staging_root.exists():
        existing = next((path for path in staging_root.rglob("*") if path.is_file()), None)
        if existing is not None:
            raise RuntimeError(f"OBJ 临时构建目录非空,请先人工处理:{existing}")
    extract_config = dict(config)
    extract_config["object_library"] = staging_root
    run_extract(extract_config)

    backup_root: Path | None = None
    if object_root.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = object_root.with_name(f"{object_root.name}.backup_{timestamp}")
        if backup_root.exists():
            raise RuntimeError(f"OBJ 备份目录已存在:{backup_root}")
        object_root.rename(backup_root)
    try:
        staging_root.rename(object_root)
    except OSError:
        if backup_root is not None and backup_root.exists() and not object_root.exists():
            backup_root.rename(object_root)
        raise
    if backup_root is not None:
        log_info(f"旧 OBJ 素材库已保留为备份:{backup_root}")
    log_info(f"新 OBJ 素材库已启用:{object_root}")


def main() -> None:
    """脚本入口,按需更新 OBJ 后执行检测数据合成."""
    try:
        update_objects = CONFIG["update_objects"]
        if not isinstance(update_objects, bool):
            raise ValueError("CONFIG.update_objects 必须是 bool")
        log_info(f"脚本启动,update_objects={update_objects}")
        if update_objects:
            rebuild_object_library(CONFIG)
        else:
            validate_directory(Path(CONFIG["object_library"]), "OBJ 素材库")
        run_compose(CONFIG)
    except (FileNotFoundError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR:{exc}") from exc


if __name__ == "__main__":
    main()
