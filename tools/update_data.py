#!/usr/bin/env python3
"""按 JSON 修改日期独立更新训练数据根目录下的子数据集。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SOURCE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DETECT_CLASSES = {"paper": 0, "liquid": 1, "metal": 2}
SEGMENT_CLASSES = {"paper": 0}
PIDNET_CLASSES = {"background": 0, "liquid": 1, "metal": 2}
BACKGROUND_LABELS = {"background", "__background__", "negative", "ignore"}
GENERATED_SUBDIRECTORIES = ("images", "labels_detect", "labels_segment", "Seg")


class UpdateDataError(RuntimeError):
    """表示独立数据更新过程中的可读错误。"""


@dataclass(frozen=True)
class SourcePair:
    """表示一对同名的源图片与 LabelMe JSON。"""

    stem: str
    image_path: Path
    json_path: Path


def parse_args() -> argparse.Namespace:
    """解析唯一的 JSON 更新起始时间参数。"""
    parser = argparse.ArgumentParser(description="增量更新当前训练数据根目录下的子数据集。")
    parser.add_argument("time", help="处理此时间及之后修改的 JSON 文件,格式 YYYY-MM-DD HH:MM:SS。")
    return parser.parse_args()


def parse_since(value: str) -> float:
    """把 YYYY-MM-DD HH:MM:SS 转换为本地时间戳。"""
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError as exc:
        raise UpdateDataError(f"时间格式错误,应为 YYYY-MM-DD HH:MM:SS: {value!r}") from exc


def collect_files(directory: Path, suffixes: Iterable[str]) -> dict[str, Path]:
    """收集当前目录文件,并拒绝忽略大小写后重名的文件。"""
    accepted = {suffix.casefold() for suffix in suffixes}
    files: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in accepted:
            continue
        key = path.stem.casefold()
        if key in files:
            duplicates.setdefault(key, [files[key]]).append(path)
        else:
            files[key] = path
    if duplicates:
        details = [", ".join(map(str, paths)) for paths in duplicates.values()]
        raise UpdateDataError(f"目录存在同名文件:{directory}\n" + "\n".join(details[:20]))
    return files


def load_image(path: Path) -> np.ndarray:
    """读取图片,兼容中文路径。"""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise UpdateDataError(f"图片损坏或无法读取:{path}")
    return image


def load_json(path: Path) -> dict[str, Any]:
    """读取并检查 LabelMe JSON 的根结构。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdateDataError(f"JSON 无法读取:{path},原因:{exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("shapes", []), list):
        raise UpdateDataError(f"JSON 缺少 shapes 列表:{path}")
    return payload


def shape_to_polygon(shape: dict[str, Any]) -> np.ndarray | None:
    """将 LabelMe polygon、rectangle 和 circle 统一转换为多边形。"""
    try:
        points = np.asarray(shape.get("points", []), dtype=np.float64)
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
        return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))
    return points if len(points) >= 3 else None


def clip_polygon(polygon: np.ndarray, width: int, height: int, yolo: bool = False) -> np.ndarray:
    """将多边形限制到图像边界或 YOLO 归一化边界。"""
    clipped = polygon.astype(np.float64, copy=True)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width if yolo else max(width - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height if yolo else max(height - 1, 0))
    return clipped


def write_label(path: Path, lines: list[str]) -> None:
    """写入 YOLO 标签,负样本保留空 TXT 文件。"""
    content = "\n".join(lines)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def render_pair(pair: SourcePair, staging_dir: Path) -> None:
    """用独立处理逻辑生成一张图片的全部训练产物。"""
    source_image = load_image(pair.image_path)
    height, width = source_image.shape[:2]
    payload = load_json(pair.json_path)
    if payload.get("imageWidth") is not None and int(payload["imageWidth"]) != width:
        raise UpdateDataError(f"JSON 宽度与图片不一致:{width}!={payload['imageWidth']}")
    if payload.get("imageHeight") is not None and int(payload["imageHeight"]) != height:
        raise UpdateDataError(f"JSON 高度与图片不一致:{height}!={payload['imageHeight']}")

    image = source_image.copy()
    detect_lines: list[str] = []
    segment_lines: list[str] = []
    positive_masks: list[tuple[int, np.ndarray]] = []
    background_masks: list[np.ndarray] = []
    for index, shape in enumerate(payload["shapes"]):
        if not isinstance(shape, dict):
            raise UpdateDataError(f"shape 不是对象:index={index}")
        label = str(shape.get("label", "")).strip()
        polygon = shape_to_polygon(shape)
        if not label or polygon is None:
            raise UpdateDataError(f"shape 标签或多边形无效:index={index}")
        clipped = clip_polygon(polygon, width, height)
        if label == "ignore":
            cv2.fillPoly(image, [np.rint(clipped).astype(np.int32)], (0, 0, 0))
        if label in SEGMENT_CLASSES:
            yolo_polygon = clip_polygon(polygon, width, height, yolo=True)
            coordinates = " ".join(
                f"{value:.6f}" for point in yolo_polygon for value in (point[0] / width, point[1] / height)
            )
            segment_lines.append(f"{SEGMENT_CLASSES[label]} {coordinates}")
        if label in DETECT_CLASSES:
            yolo_polygon = clip_polygon(polygon, width, height, yolo=True)
            x_min, y_min = yolo_polygon.min(axis=0)
            x_max, y_max = yolo_polygon.max(axis=0)
            box_width, box_height = x_max - x_min, y_max - y_min
            if box_width > 0 and box_height > 0:
                detect_lines.append(
                    f"{DETECT_CLASSES[label]} {(x_min + x_max) / (2 * width):.6f} "
                    f"{(y_min + y_max) / (2 * height):.6f} {box_width / width:.6f} {box_height / height:.6f}"
                )
        if label in PIDNET_CLASSES and label not in BACKGROUND_LABELS:
            positive_masks.append((PIDNET_CLASSES[label], polygon))
        else:
            background_masks.append(polygon)

    jpg_path = staging_dir / "images" / f"{pair.stem}.jpg"
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise UpdateDataError(f"JPG 编码失败:{pair.stem}")
    encoded.tofile(jpg_path)
    write_label(staging_dir / "labels_detect" / f"{pair.stem}.txt", detect_lines)
    write_label(staging_dir / "labels_segment" / f"{pair.stem}.txt", segment_lines)

    mask = np.zeros((height, width), dtype=np.uint8)
    for class_id, polygon in positive_masks:
        cv2.fillPoly(mask, [np.rint(clip_polygon(polygon, width, height)).astype(np.int32)], class_id)
    for polygon in background_masks:
        cv2.fillPoly(mask, [np.rint(clip_polygon(polygon, width, height)).astype(np.int32)], 0)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise UpdateDataError(f"Seg 编码失败:{pair.stem}")
    encoded.tofile(staging_dir / "Seg" / f"{pair.stem}.png")


def iter_splits(dataset_dir: Path) -> list[Path]:
    """返回子数据集内直接包含 json 目录的 train、val 或其他 split。"""
    return [path for path in sorted(dataset_dir.iterdir()) if path.is_dir() and (path / "json").is_dir()]


def create_staging_split(split_dir: Path) -> Path:
    """创建只包含本次增量样本的临时输出目录。"""
    staging_dir = split_dir / f".update_data.{os.getpid()}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    for name in GENERATED_SUBDIRECTORIES:
        (staging_dir / name).mkdir(parents=True, exist_ok=True)
    return staging_dir


def replace_generated_files(staging_dir: Path, split_dir: Path, stem: str) -> None:
    """用临时生成结果原子替换当前 split 中同名的训练产物。"""
    for name in GENERATED_SUBDIRECTORIES:
        destination_dir = split_dir / name
        destination_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".png" if name == "Seg" else ".jpg" if name == "images" else ".txt"
        source = staging_dir / name / f"{stem}{suffix}"
        if not source.is_file():
            raise UpdateDataError(f"临时输出缺少文件:{source}")
        os.replace(source, destination_dir / source.name)


def update_split(split_dir: Path, since_timestamp: float, statistics: Counter[str]) -> None:
    """更新一个 split 中日期满足条件且存在同名 img_src 图片的 JSON。"""
    json_files = collect_files(split_dir / "json", {".json"})
    changed_json = [path for path in json_files.values() if path.stat().st_mtime >= since_timestamp]
    statistics["json_total"] += len(json_files)
    statistics["skipped_old"] += len(json_files) - len(changed_json)
    if not changed_json:
        return
    image_dir = split_dir / "img_src"
    if not image_dir.is_dir():
        for json_path in changed_json:
            print(f"UPDATE_DATA_FAILED | {split_dir.parent.name}/{split_dir.name}/{json_path.stem} | 缺少 img_src 目录")
            statistics["failed"] += 1
        return
    images = collect_files(image_dir, SOURCE_IMAGE_SUFFIXES)
    staging_dir = create_staging_split(split_dir)
    try:
        for json_path in changed_json:
            image_path = images.get(json_path.stem.casefold())
            label = f"{split_dir.parent.name}/{split_dir.name}/{json_path.stem}"
            if image_path is None:
                print(f"UPDATE_DATA_FAILED | {label} | img_src 缺少同名图片")
                statistics["failed"] += 1
                continue
            try:
                pair = SourcePair(image_path.stem, image_path, json_path)
                render_pair(pair, staging_dir)
                replace_generated_files(staging_dir, split_dir, pair.stem)
            except (OSError, UpdateDataError, ValueError) as exc:
                print(f"UPDATE_DATA_FAILED | {label} | {exc}")
                statistics["failed"] += 1
                continue
            print(f"UPDATE_DATA_UPDATED | {label}")
            statistics["updated"] += 1
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    """遍历当前训练数据根目录,按 JSON 修改日期更新所有子数据集。"""
    args = parse_args()
    since_timestamp = parse_since(args.time)
    root_dir = Path.cwd().resolve()
    datasets = [path for path in sorted(root_dir.iterdir()) if path.is_dir()]
    if not datasets:
        raise UpdateDataError(f"训练数据根目录没有子数据集:{root_dir}")
    statistics: Counter[str] = Counter(datasets=len(datasets))
    print(f"UPDATE_DATA | root={root_dir} | since={args.time}")
    for dataset_dir in datasets:
        splits = iter_splits(dataset_dir)
        if not splits:
            print(f"UPDATE_DATA_SKIPPED | {dataset_dir.name} | 未找到包含 json 的 split")
            statistics["skipped_dataset"] += 1
            continue
        for split_dir in splits:
            statistics["splits"] += 1
            update_split(split_dir, since_timestamp, statistics)
    print(
        "UPDATE_DATA_SUMMARY | "
        f"datasets={statistics['datasets']} | splits={statistics['splits']} | "
        f"json_total={statistics['json_total']} | updated={statistics['updated']} | "
        f"failed={statistics['failed']} | skipped_old={statistics['skipped_old']} | "
        f"skipped_dataset={statistics['skipped_dataset']}"
    )


if __name__ == "__main__":
    main()
