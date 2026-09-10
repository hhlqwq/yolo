"""校验 Train2 原始检测数据，并按批次划分两类训练集和验证集。"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import uuid
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm


DEFAULT_SOURCE_ROOT = Path(
    "/data/users/hailong.he/nas_smb/Datasets/internal/"
    "P000_SHUNYU_2026/3_Train2"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_LABEL_DIR_NAMES = ("labels_detect", "label_detect")
SOURCE_CLASS_MAP = {1: 0, 2: 1}
TARGET_CLASS_NAMES = {0: "liquid", 1: "debris"}
DROP_CLASS_IDS = {0}


@dataclass(frozen=True)
class ValidSample:
    """保存一组已校验的原始图片、标签和转换后标签内容。"""

    image: Path
    label: Path
    converted_lines: tuple[str, ...]
    class_counts: dict[int, int]


@dataclass(frozen=True)
class BatchPlan:
    """保存一个批次的有效样本、跳过记录和划分结果。"""

    batch_dir: Path
    label_dirs: tuple[Path, ...]
    source_image_count: int
    source_label_count: int
    train_samples: tuple[ValidSample, ...]
    val_samples: tuple[ValidSample, ...]
    skipped: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    """解析数据根目录、划分比例和随机种子。"""
    parser = argparse.ArgumentParser(
        description="校验 3_Train2 原始数据，并按批次划分 liquid/debris 两类数据。"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="包含多个原始数据批次的 3_Train2 根目录。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="每个批次划入验证集的比例，默认 0.2。",
    )
    parser.add_argument("--seed", type=int, default=42, help="固定划分随机种子。")
    parser.add_argument(
        "--copy-workers",
        type=int,
        default=8,
        help="NAS 图片并发复制线程数，默认 8。",
    )
    return parser.parse_args()


def _group_by_stem(paths: list[Path]) -> dict[str, list[Path]]:
    """按不区分大小写的文件名主干分组。"""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[path.stem.casefold()].append(path)
    return groups


def _skip_record(
    batch_dir: Path,
    stem: str,
    reason: str,
    images: list[Path],
    labels: list[Path],
) -> dict[str, Any]:
    """构建一条可写入清单的跳过记录。"""
    return {
        "batch": batch_dir.name,
        "stem": stem,
        "reason": reason,
        "images": [str(path) for path in images],
        "labels": [str(path) for path in labels],
    }


def _convert_label(label_path: Path) -> tuple[tuple[str, ...], dict[int, int]]:
    """校验一个旧三类 YOLO 标签，并转换为 liquid/debris 两类标签。"""
    converted_lines: list[str] = []
    class_counts: Counter[int] = Counter()
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"标签无法读取: {exc}") from exc

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"第 {line_number} 行不是五列 YOLO 检测标签")
        try:
            source_class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行包含非数值字段") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"第 {line_number} 行包含非有限数值")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"第 {line_number} 行坐标超出 [0, 1]")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(f"第 {line_number} 行检测框宽高必须大于 0")
        if source_class_id in DROP_CLASS_IDS:
            continue
        if source_class_id not in SOURCE_CLASS_MAP:
            raise ValueError(
                f"第 {line_number} 行包含不支持的旧类别 ID {source_class_id}"
            )
        target_class_id = SOURCE_CLASS_MAP[source_class_id]
        converted_lines.append(" ".join([str(target_class_id), *fields[1:]]))
        class_counts[target_class_id] += 1
    return tuple(converted_lines), dict(class_counts)


def _collect_batch_samples(
    batch_dir: Path,
) -> tuple[list[ValidSample], list[dict[str, Any]], tuple[Path, ...]]:
    """按文件名配对一个批次，并跳过缺失、冲突或非法的样本。"""
    image_dir = batch_dir / "images"
    if not image_dir.is_dir():
        return [], [
            _skip_record(batch_dir, "", "批次缺少 images 目录", [], [])
        ], ()

    label_dirs = tuple(
        batch_dir / name
        for name in SOURCE_LABEL_DIR_NAMES
        if (batch_dir / name).is_dir()
    )
    if not label_dirs:
        images = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ]
        return [], [
            _skip_record(
                batch_dir,
                "",
                "批次缺少 labels_detect 或 label_detect 目录",
                images,
                [],
            )
        ], ()

    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    labels = sorted(
        (
            path
            for label_dir in label_dirs
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".txt"
        ),
        key=lambda path: str(path).casefold(),
    )
    image_groups = _group_by_stem(images)
    label_groups = _group_by_stem(labels)
    valid_samples: list[ValidSample] = []
    skipped: list[dict[str, Any]] = []

    stems = sorted(set(image_groups) | set(label_groups))
    for stem in tqdm(
        stems,
        desc=f"{batch_dir.name} 校验标签",
        unit="组",
        dynamic_ncols=True,
    ):
        matching_images = image_groups.get(stem, [])
        matching_labels = label_groups.get(stem, [])
        if not matching_images:
            skipped.append(
                _skip_record(
                    batch_dir,
                    stem,
                    "标签没有同名图片",
                    matching_images,
                    matching_labels,
                )
            )
            continue
        if not matching_labels:
            skipped.append(
                _skip_record(
                    batch_dir,
                    stem,
                    "图片没有同名标签",
                    matching_images,
                    matching_labels,
                )
            )
            continue
        if len(matching_images) != 1 or len(matching_labels) != 1:
            skipped.append(
                _skip_record(
                    batch_dir,
                    stem,
                    "同名图片或标签发生冲突",
                    matching_images,
                    matching_labels,
                )
            )
            continue
        try:
            converted_lines, class_counts = _convert_label(matching_labels[0])
        except ValueError as exc:
            skipped.append(
                _skip_record(
                    batch_dir,
                    stem,
                    f"标签校验失败: {exc}",
                    matching_images,
                    matching_labels,
                )
            )
            continue
        valid_samples.append(
            ValidSample(
                image=matching_images[0],
                label=matching_labels[0],
                converted_lines=converted_lines,
                class_counts=class_counts,
            )
        )
    return valid_samples, skipped, label_dirs


def _split_samples(
    samples: list[ValidSample], val_ratio: float, seed: int
) -> tuple[tuple[ValidSample, ...], tuple[ValidSample, ...]]:
    """使用固定随机种子划分一个批次的训练样本和验证样本。"""
    shuffled = sorted(samples, key=lambda sample: sample.image.name.casefold())
    random.Random(seed).shuffle(shuffled)
    val_count = min(max(1, round(len(shuffled) * val_ratio)), len(shuffled) - 1)
    val_samples = tuple(shuffled[:val_count])
    train_samples = tuple(shuffled[val_count:])
    return train_samples, val_samples


def _preflight_output(batch_dirs: list[Path], source_root: Path) -> None:
    """在写入前确认所有目标目录和元数据不会覆盖已有结果。"""
    conflicts: list[Path] = []
    for batch_dir in batch_dirs:
        for split in ("train", "val"):
            output_dir = batch_dir / split
            if output_dir.exists():
                if not output_dir.is_dir() or any(output_dir.iterdir()):
                    conflicts.append(output_dir)
    for metadata in (
        source_root / "train2_detect.yaml",
        source_root / "split_manifest.json",
    ):
        if metadata.exists():
            conflicts.append(metadata)
    if conflicts:
        examples = "\n".join(str(path) for path in conflicts)
        raise RuntimeError(f"发现已有划分结果，为避免覆盖已停止:\n{examples}")


def _write_sample(
    sample: ValidSample,
    split_dir: Path,
) -> dict[str, Any]:
    """复制一组有效图片和转换后标签，并返回清单记录。"""
    batch_name = split_dir.parent.name
    output_stem = f"{batch_name}__{sample.image.stem}"
    image_output = split_dir / "images" / f"{output_stem}{sample.image.suffix.lower()}"
    label_output = split_dir / "labels" / f"{output_stem}.txt"
    shutil.copyfile(sample.image, image_output)
    label_text = "\n".join(sample.converted_lines)
    if label_text:
        label_text += "\n"
    label_output.write_text(label_text, encoding="utf-8")
    return {
        "source_image": str(sample.image),
        "source_label": str(sample.label),
        "output_image": str(image_output),
        "output_label": str(label_output),
        "class_counts": sample.class_counts,
        "negative": not sample.converted_lines,
    }


def _validate_staged_split(
    split_dir: Path,
    expected_count: int,
    batch_name: str,
    split: str,
) -> None:
    """校验 staging 中图片、标签文件名集合、内容和预期数量。"""
    images = {
        path.stem.casefold()
        for path in (split_dir / "images").iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }
    labels = {
        path.stem.casefold()
        for path in (split_dir / "labels").glob("*.txt")
        if path.is_file()
    }
    if images != labels or len(images) != expected_count:
        raise RuntimeError(
            f"划分结果校验失败: {split_dir}, images={len(images)}, "
            f"labels={len(labels)}, expected={expected_count}"
        )
    label_paths = sorted((split_dir / "labels").glob("*.txt"))
    for label_path in tqdm(
        label_paths,
        desc=f"{batch_name} 校验{split}输出",
        unit="标签",
        dynamic_ncols=True,
    ):
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or fields[0] not in {"0", "1"}:
                raise RuntimeError(
                    f"转换后标签校验失败: {label_path}:{line_number}"
                )


def _stage_batches(
    plans: list[BatchPlan], staging_root: Path, copy_workers: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """将全部批次先写入统一 staging，避免校验前暴露半成品。"""
    manifest_samples: list[dict[str, Any]] = []
    batch_statistics: dict[str, dict[str, int]] = {}
    for plan in plans:
        batch_staging = staging_root / plan.batch_dir.name
        for split in ("train", "val"):
            (batch_staging / split / "images").mkdir(parents=True)
            (batch_staging / split / "labels").mkdir(parents=True)
        class_counts: Counter[int] = Counter()
        negative_images = 0
        copy_items = [
            (split, sample)
            for split, samples in (
                ("train", plan.train_samples),
                ("val", plan.val_samples),
            )
            for sample in samples
        ]
        executor = ThreadPoolExecutor(max_workers=copy_workers)
        futures: dict[Future[dict[str, Any]], tuple[str, ValidSample]] = {
            executor.submit(_write_sample, sample, batch_staging / split): (
                split,
                sample,
            )
            for split, sample in copy_items
        }
        try:
            completed = as_completed(futures)
            for future in tqdm(
                completed,
                total=len(futures),
                desc=f"{plan.batch_dir.name} 复制图片",
                unit="张",
                dynamic_ncols=True,
            ):
                split, sample = futures[future]
                record = future.result()
                record.update({"batch": plan.batch_dir.name, "split": split})
                manifest_samples.append(record)
                class_counts.update(sample.class_counts)
                negative_images += int(not sample.converted_lines)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        for split, samples in (
            ("train", plan.train_samples),
            ("val", plan.val_samples),
        ):
            _validate_staged_split(
                batch_staging / split,
                len(samples),
                plan.batch_dir.name,
                split,
            )
        batch_statistics[plan.batch_dir.name] = {
            "source_images": plan.source_image_count,
            "source_labels": plan.source_label_count,
            "valid_pairs": len(plan.train_samples) + len(plan.val_samples),
            "train_images": len(plan.train_samples),
            "val_images": len(plan.val_samples),
            "negative_images": negative_images,
            "liquid_boxes": class_counts[0],
            "debris_boxes": class_counts[1],
            "skipped": len(plan.skipped),
        }
    manifest_samples.sort(
        key=lambda item: (item["batch"], item["split"], item["output_image"])
    )
    return manifest_samples, batch_statistics


def _commit_batches(plans: list[BatchPlan], staging_root: Path) -> None:
    """将已校验的 staging train/val 目录提交到各原始批次下。"""
    for plan in plans:
        for split in ("train", "val"):
            destination = plan.batch_dir / split
            if destination.exists():
                destination.rmdir()
            (staging_root / plan.batch_dir.name / split).replace(destination)


def _write_metadata(
    source_root: Path,
    plans: list[BatchPlan],
    samples: list[dict[str, Any]],
    statistics: dict[str, dict[str, int]],
    val_ratio: float,
    seed: int,
    extra_skipped: list[dict[str, Any]],
) -> None:
    """写入两类数据 YAML 和包含全部跳过项的划分清单。"""
    training_paths = [str((plan.batch_dir / "train" / "images").resolve()) for plan in plans]
    validation_paths = [str((plan.batch_dir / "val" / "images").resolve()) for plan in plans]
    dataset = {
        "path": str(source_root.resolve()),
        "train": training_paths,
        "val": validation_paths,
        "nc": len(TARGET_CLASS_NAMES),
        "names": TARGET_CLASS_NAMES,
    }
    (source_root / "train2_detect.yaml").write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "source_root": str(source_root.resolve()),
        "val_ratio": val_ratio,
        "seed": seed,
        "source_class_map": SOURCE_CLASS_MAP,
        "drop_class_ids": sorted(DROP_CLASS_IDS),
        "target_class_names": TARGET_CLASS_NAMES,
        "statistics": statistics,
        "source_label_dirs": {
            plan.batch_dir.name: [str(path) for path in plan.label_dirs]
            for plan in plans
        },
        "samples": samples,
        "skipped": [
            *[record for plan in plans for record in plan.skipped],
            *extra_skipped,
        ],
    }
    (source_root / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_summary(
    plans: list[BatchPlan], statistics: dict[str, dict[str, int]]
) -> None:
    """输出各批次划分、类别和跳过数量摘要。"""
    print("\n划分完成，原始 images 和标签目录均已保留。")
    for plan in plans:
        values = statistics[plan.batch_dir.name]
        print(
            f"[{plan.batch_dir.name}] images={values['source_images']}, "
            f"labels={values['source_labels']}, valid={values['valid_pairs']}, "
            f"train={values['train_images']}, "
            f"val={values['val_images']}, negative={values['negative_images']}, "
            f"liquid_boxes={values['liquid_boxes']}, "
            f"debris_boxes={values['debris_boxes']}, skipped={values['skipped']}"
        )
        for record in plan.skipped[:5]:
            print(
                f"  [跳过] stem={record['stem'] or '-'}, "
                f"reason={record['reason']}"
            )
    print(f"数据配置: {plans[0].batch_dir.parent / 'train2_detect.yaml'}")
    print(f"划分清单: {plans[0].batch_dir.parent / 'split_manifest.json'}")


def main() -> None:
    """校验全部原始批次，生成两类 train/val 数据和元数据。"""
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"数据根目录不存在: {source_root}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio 必须在 0 和 1 之间。")
    if args.copy_workers <= 0:
        raise ValueError("--copy-workers 必须大于 0。")

    batch_dirs = sorted(
        (
            path
            for path in source_root.iterdir()
            if path.is_dir() and (path / "images").is_dir()
        ),
        key=lambda path: path.name.casefold(),
    )
    if not batch_dirs:
        raise RuntimeError(f"未找到包含 images 目录的原始批次: {source_root}")
    print(f"发现 {len(batch_dirs)} 个原始批次，开始检查图片和标签。", flush=True)
    _preflight_output(batch_dirs, source_root)

    plans: list[BatchPlan] = []
    ignored_batch_records: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        print(f"\n[批次] 开始校验 {batch_dir.name}。", flush=True)
        valid_samples, skipped, label_dirs = _collect_batch_samples(batch_dir)
        if len(valid_samples) < 2:
            ignored_batch_records.extend(skipped)
            ignored_batch_records.append(
                _skip_record(
                    batch_dir,
                    "",
                    f"有效样本不足 2 个，整个批次跳过: {len(valid_samples)}",
                    [sample.image for sample in valid_samples],
                    [sample.label for sample in valid_samples],
                )
            )
            continue
        print(
            f"[批次] {batch_dir.name} 校验完成: "
            f"有效={len(valid_samples)}, 跳过={len(skipped)}。",
            flush=True,
        )
        train_samples, val_samples = _split_samples(
            valid_samples,
            args.val_ratio,
            args.seed,
        )
        plans.append(
            BatchPlan(
                batch_dir=batch_dir,
                label_dirs=label_dirs,
                source_image_count=sum(
                    1
                    for path in (batch_dir / "images").iterdir()
                    if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
                ),
                source_label_count=sum(
                    1
                    for label_dir in label_dirs
                    for path in label_dir.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".txt"
                ),
                train_samples=train_samples,
                val_samples=val_samples,
                skipped=tuple(skipped),
            )
        )
    if not plans:
        raise RuntimeError("所有批次均没有足够的有效样本，未生成划分结果。")

    staging_root = source_root / f".train2_split_{uuid.uuid4().hex}"
    print(
        f"\n开始复制划分结果，线程数={args.copy_workers}。",
        flush=True,
    )
    print(f"临时目录: {staging_root}", flush=True)
    try:
        samples, statistics = _stage_batches(
            plans,
            staging_root,
            args.copy_workers,
        )
        print("全部批次复制和校验完成，正在提交 train/val 目录。", flush=True)
        _commit_batches(plans, staging_root)
        print("train/val 目录提交完成，正在写入 YAML 和清单。", flush=True)
        _write_metadata(
            source_root,
            plans,
            samples,
            statistics,
            args.val_ratio,
            args.seed,
            ignored_batch_records,
        )
    finally:
        if staging_root.exists():
            print(f"正在清理临时目录: {staging_root}", flush=True)
            shutil.rmtree(staging_root)
    _print_summary(plans, statistics)


if __name__ == "__main__":
    main()
