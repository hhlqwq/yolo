#!/usr/bin/env python3
"""合并 YOLO 检测标签中相交或接近的同类别检测框。"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Detection:
    """保存一条 YOLO 归一化检测框标签。"""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def pixel_box(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        """将归一化 YOLO 框转换为像素 left/top/right/bottom。"""
        half_width = self.width * image_width / 2.0
        half_height = self.height * image_height / 2.0
        center_x = self.center_x * image_width
        center_y = self.center_y * image_height
        return center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height


@dataclass
class Statistics:
    """保存批量合并过程的统计数据。"""

    label_files: int = 0
    changed_files: int = 0
    unchanged_files: int = 0
    failed_files: int = 0
    missing_images: int = 0
    merged_boxes: int = 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="合并 YOLO 检测标签中接近的同类别检测框。")
    parser.add_argument("labels_dir", type=Path, help="输入 labels_detect 目录，会递归处理 TXT。")
    parser.add_argument("--image-dir", type=Path, default=None, help="图片目录，默认同级 img 或 images。")
    parser.add_argument("--output-dir", type=Path, default=None, help="默认同级 labels_detect_merge。")
    parser.add_argument("--overwrite", action="store_true", help="删除已有输出目录后重新生成。")
    parser.add_argument("--distance", type=float, default=20.0, help="最大合并距离，单位像素，默认 20。")
    return parser.parse_args()


def resolve_directories(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """解析输入图片、标签和输出目录，并阻止覆盖已有结果。"""
    labels_dir = args.labels_dir.expanduser().resolve()
    if not labels_dir.is_dir():
        raise ValueError(f"输入标签目录不存在:{labels_dir}")
    if args.distance < 0:
        raise ValueError(f"--distance 不能小于 0:{args.distance}")
    if args.image_dir is not None:
        image_dir = args.image_dir.expanduser().resolve()
    else:
        image_dir = labels_dir.parent / "img"
        if not image_dir.is_dir():
            image_dir = labels_dir.parent / "images"
    if not image_dir.is_dir():
        raise ValueError(f"图片目录不存在，请通过 --image-dir 指定:{image_dir}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else labels_dir.parent / f"{labels_dir.name}_merge"
    if output_dir == labels_dir or output_dir.is_relative_to(labels_dir):
        raise ValueError("输出目录不能是输入标签目录本身或其子目录")
    if output_dir.exists():
        if not args.overwrite:
            raise ValueError(f"输出目录已存在，为避免覆盖已停止:{output_dir}")
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError(f"拒绝删除非普通输出目录:{output_dir}")
        print(f"REMOVE_EXISTING_OUTPUT | {output_dir}")
        shutil.rmtree(output_dir)
    return labels_dir, image_dir, output_dir


def collect_images(image_dir: Path) -> dict[str, Path]:
    """按不含扩展名的文件名收集图片，并拒绝重名图片。"""
    images: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.casefold()
        if key in images:
            duplicates.setdefault(key, [images[key]]).append(path)
        else:
            images[key] = path
    if duplicates:
        details = ["、".join(map(str, paths)) for paths in list(duplicates.values())[:20]]
        raise ValueError(f"图片目录中存在 {len(duplicates)} 组重名图片:\n" + "\n".join(details))
    return images


def read_image_size(path: Path) -> tuple[int, int]:
    """读取图片宽高，兼容包含中文的路径。"""
    try:
        with Image.open(path) as image:
            return image.size
    except OSError as exc:
        raise ValueError(f"图片无法读取:{path}") from exc


def parse_detections(path: Path) -> list[Detection]:
    """读取并校验一个 YOLO 检测标签文件。"""
    detections: list[Detection] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"第 {line_number} 行必须有 5 列，实际为 {len(values)} 列")
        try:
            class_id = int(values[0])
            center_x, center_y, width, height = (float(value) for value in values[1:])
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行存在非数值字段") from exc
        if class_id < 0 or width <= 0 or height <= 0:
            raise ValueError(f"第 {line_number} 行类别或宽高非法")
        detections.append(Detection(class_id, center_x, center_y, width, height))
    return detections


def box_distance(first: Detection, second: Detection, image_width: int, image_height: int) -> float:
    """计算两个检测框的最短像素距离，相交时返回 0。"""
    first_left, first_top, first_right, first_bottom = first.pixel_box(image_width, image_height)
    second_left, second_top, second_right, second_bottom = second.pixel_box(image_width, image_height)
    horizontal = max(first_left - second_right, second_left - first_right, 0.0)
    vertical = max(first_top - second_bottom, second_top - first_bottom, 0.0)
    return math.hypot(horizontal, vertical)


def merge_group(group: list[Detection]) -> Detection:
    """将同一类别的一组检测框合并为最小外接检测框。"""
    left = min(item.center_x - item.width / 2.0 for item in group)
    top = min(item.center_y - item.height / 2.0 for item in group)
    right = max(item.center_x + item.width / 2.0 for item in group)
    bottom = max(item.center_y + item.height / 2.0 for item in group)
    return Detection(group[0].class_id, (left + right) / 2.0, (top + bottom) / 2.0, right - left, bottom - top)


def merge_once(detections: list[Detection], image_width: int, image_height: int, distance: float) -> list[Detection]:
    """执行一轮同类别检测框合并，并返回本轮外接框结果。"""
    parent = list(range(len(detections)))

    def find(index: int) -> int:
        """查找并压缩并查集根节点。"""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        """合并两个并查集集合。"""
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(detections)):
        for second in range(first + 1, len(detections)):
            if detections[first].class_id == detections[second].class_id and box_distance(detections[first], detections[second], image_width, image_height) <= distance:
                union(first, second)
    groups: dict[int, list[Detection]] = {}
    for index, detection in enumerate(detections):
        groups.setdefault(find(index), []).append(detection)
    merged = [merge_group(group) for group in groups.values()]
    merged.sort(key=lambda item: (item.class_id, item.center_y, item.center_x, item.width, item.height))
    return merged


def merge_detections(detections: list[Detection], image_width: int, image_height: int, distance: float) -> tuple[list[Detection], int]:
    """重复按框边缘距离合并，直到新外接框不再触发新的合并。"""
    current = detections
    while True:
        merged = merge_once(current, image_width, image_height, distance)
        if len(merged) == len(current):
            return merged, len(detections) - len(merged)
        current = merged


def format_detection(detection: Detection) -> str:
    """将检测框格式化为 YOLO TXT 的一行。"""
    return f"{detection.class_id} {detection.center_x:.6f} {detection.center_y:.6f} {detection.width:.6f} {detection.height:.6f}"


def copy_label(path: Path, source_dir: Path, output_dir: Path) -> None:
    """将无法处理的原始标签原样复制到输出目录。"""
    target = output_dir / path.relative_to(source_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def main() -> int:
    """递归合并检测标签并打印每个文件和总计统计。"""
    args = parse_args()
    try:
        labels_dir, image_dir, output_dir = resolve_directories(args)
        images = collect_images(image_dir)
    except ValueError as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    statistics = Statistics()
    for label_path in sorted(labels_dir.rglob("*.txt")):
        if not label_path.is_file():
            continue
        statistics.label_files += 1
        image_path = images.get(label_path.stem.casefold())
        if image_path is None:
            statistics.missing_images += 1
            copy_label(label_path, labels_dir, output_dir)
            print(f"MERGE_DETECT_SKIPPED_NO_IMAGE | {label_path}")
            continue
        try:
            image_width, image_height = read_image_size(image_path)
            detections = parse_detections(label_path)
            merged, merged_boxes = merge_detections(detections, image_width, image_height, args.distance)
        except (OSError, ValueError) as exc:
            statistics.failed_files += 1
            print(f"MERGE_DETECT_FAILED | {label_path} | {exc}", file=sys.stderr)
            continue
        target_path = output_dir / label_path.relative_to(labels_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("\n".join(format_detection(item) for item in merged) + ("\n" if merged else ""), encoding="utf-8")
        statistics.merged_boxes += merged_boxes
        if merged_boxes:
            statistics.changed_files += 1
            print(f"MERGE_DETECT_UPDATED | {label_path} | source_boxes={len(detections)} | output_boxes={len(merged)} | merged_boxes={merged_boxes}")
        else:
            statistics.unchanged_files += 1
    print(
        "MERGE_DETECT_SUMMARY"
        f" | label_files={statistics.label_files} | changed_files={statistics.changed_files}"
        f" | unchanged_files={statistics.unchanged_files} | missing_images={statistics.missing_images}"
        f" | failed_files={statistics.failed_files} | merged_boxes={statistics.merged_boxes}"
        f" | output_dir={output_dir}"
    )
    return 1 if statistics.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
