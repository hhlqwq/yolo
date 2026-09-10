"""累计数据注册、训练配置快照和累计统计."""

from __future__ import annotations

import json
import os
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.io_utils import atomic_write_text
from ..core.stage_runner import StageOutcome
from .data import DETECT_CLASSES, PIDNET_CLASSES, SEGMENT_CLASSES, SPLITS


REGISTRY_VERSION = 2
YOLO_IMAGE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".dng",
    ".heic",
    ".heif",
    ".jp2",
    ".jpeg",
    ".jpeg2000",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def empty_registry() -> dict[str, Any]:
    """返回支持任意物理目录名称的空注册表."""
    return {
        "version": REGISTRY_VERSION,
        "yolo": {split: [] for split in SPLITS},
        "pidnet": {split: [] for split in SPLITS},
    }


def is_absolute_registered_path(value: str) -> bool:
    """判断字符串是否为 Linux、Windows 盘符或 UNC 绝对路径."""
    normalized = value.replace("\\", "/")
    return (
        normalized.startswith("/")
        or normalized.startswith("//")
        or (len(normalized) >= 3 and normalized[1:3] == ":/")
    )


def repair_registered_path(path: Path) -> Path:
    """仅在原路径不存在且候选唯一时修复常见 Unicode 括号差异."""
    if path.exists():
        return path
    text = str(path)
    variants = {
        text.replace("算法应用(主)", "算法应用（主）"),
        text.replace("算法应用（主）", "算法应用(主)"),
        unicodedata.normalize("NFC", text),
        unicodedata.normalize("NFKC", text),
    }
    existing = [Path(value) for value in variants if value != text and Path(value).exists()]
    unique = {str(candidate.resolve()): candidate for candidate in existing}
    return next(iter(unique.values())) if len(unique) == 1 else path


def normalize_registry(payload: dict[str, Any], context: RunContext | None = None) -> dict[str, Any]:
    """规范化 v2 注册表并按逻辑 train/val 去重,不限制物理目录名称."""
    if payload.get("version") != REGISTRY_VERSION:
        raise PipelineError(f"累计数据注册表版本不支持:{payload.get('version')!r}")
    yolo_section = payload.get("yolo")
    pidnet_section = payload.get("pidnet")
    if not isinstance(yolo_section, dict) or not isinstance(pidnet_section, dict):
        raise PipelineError("累计数据注册表缺少yolo或pidnet字典")
    normalized = empty_registry()
    for split in SPLITS:
        yolo_entries = yolo_section.get(split, [])
        pidnet_entries = pidnet_section.get(split, [])
        if not isinstance(yolo_entries, list) or not isinstance(pidnet_entries, list):
            raise PipelineError(f"累计数据注册表中的{split}必须是列表")
        seen_yolo: set[str] = set()
        for value in yolo_entries:
            original = Path(str(value)).expanduser()
            repaired = repair_registered_path(original)
            text = str(repaired)
            if not is_absolute_registered_path(text):
                raise PipelineError(f"YOLO注册路径必须是绝对路径:{value}")
            key = text.replace("\\", "/").casefold()
            if key not in seen_yolo:
                normalized["yolo"][split].append(text)
                seen_yolo.add(key)
            if context is not None and repaired != original:
                context.logger.warning("注册表路径自动修复:%s -> %s", original, repaired)

        seen_pidnet: set[str] = set()
        for entry in pidnet_entries:
            if not isinstance(entry, dict) or not entry.get("image") or not entry.get("mask"):
                raise PipelineError(f"PIDNet注册项缺少image或mask:{entry!r}")
            original_image = Path(str(entry["image"])).expanduser()
            original_mask = Path(str(entry["mask"])).expanduser()
            image_path = repair_registered_path(original_image)
            mask_path = repair_registered_path(original_mask)
            image_text, mask_text = str(image_path), str(mask_path)
            if not is_absolute_registered_path(image_text) or not is_absolute_registered_path(mask_text):
                raise PipelineError(f"PIDNet注册路径必须是绝对路径:{entry!r}")
            key = f"{image_text}\t{mask_text}".replace("\\", "/").casefold()
            if key not in seen_pidnet:
                normalized["pidnet"][split].append({"image": image_text, "mask": mask_text})
                seen_pidnet.add(key)
            if context is not None and image_path != original_image:
                context.logger.warning("注册表路径自动修复:%s -> %s", original_image, image_path)
            if context is not None and mask_path != original_mask:
                context.logger.warning("注册表路径自动修复:%s -> %s", original_mask, mask_path)
    return normalized


def convert_v3_registry(payload: dict[str, Any]) -> dict[str, Any]:
    """把曾经生成的 v3 数据集根目录结构无损转换回灵活的 v2 来源结构."""
    registry = empty_registry()
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise PipelineError("v3累计数据注册表缺少datasets列表")
    for item in datasets:
        if not isinstance(item, dict) or not item.get("dataset_dir"):
            raise PipelineError(f"v3累计数据注册项非法:{item!r}")
        dataset_dir = repair_registered_path(Path(str(item["dataset_dir"])).expanduser())
        for split in SPLITS:
            image_dir = dataset_dir / split / "images"
            registry["yolo"][split].append(str(image_dir))
            for image_path in sorted(image_dir.glob("*.jpg")):
                registry["pidnet"][split].append(
                    {
                        "image": str(image_path),
                        "mask": str(dataset_dir / split / "Seg" / f"{image_path.stem}.png"),
                    }
                )
    return registry


def load_registry(context: RunContext) -> dict[str, Any]:
    """只读加载累计注册表，保留任意训练目录名称和任务独立来源。"""
    path = context.config.registry_dir / "datasets.yaml"
    if not path.is_file():
        return empty_registry()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"累计数据注册表读取失败:{path},原因:{exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"累计数据注册表必须是字典:{path}")
    converted = payload.get("version") == 3
    if converted:
        payload = convert_v3_registry(payload)
    normalized = normalize_registry(payload, context)
    if converted:
        context.logger.warning("按v2来源结构只读兼容v3累计注册表:%s", path)
    return normalized


def backup_registry(path: Path) -> Path:
    """在注册表内容变化前创建不覆盖历史文件的时间戳备份."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"datasets_backup_{timestamp}.yaml")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"datasets_backup_{timestamp}_{suffix}.yaml")
        suffix += 1
    shutil.copy2(path, backup)
    return backup


def save_registry(context: RunContext, registry: dict[str, Any]) -> Path:
    """内容变化时先备份,再原子保存累计数据注册表."""
    path = context.config.registry_dir / "datasets.yaml"
    content = yaml.safe_dump(registry, allow_unicode=True, sort_keys=False)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return path
    if path.is_file():
        backup = backup_registry(path)
        context.logger.info("累计数据注册表已备份:%s", backup)
    atomic_write_text(path, content)
    return path


def validate_dataset_root(dataset_dir: Path) -> None:
    """验证一个累计数据集包含三个模型所需的全部文件."""
    for split in SPLITS:
        split_dir = dataset_dir / split
        for name in ("images", "labels_detect", "labels_segment", "Seg"):
            if not (split_dir / name).is_dir():
                raise PipelineError(f"累计数据集缺少目录:{split_dir / name}")
        image_stems = {path.stem for path in (split_dir / "images").glob("*.jpg")}
        for name, suffix in (("labels_detect", ".txt"), ("labels_segment", ".txt"), ("Seg", ".png")):
            stems = {path.stem for path in (split_dir / name).glob(f"*{suffix}")}
            if stems != image_stems:
                raise PipelineError(
                    f"累计数据集{split}/{name}与images未一一对应:images={len(image_stems)},"
                    f"{name}={len(stems)}"
                )


def image_component_index(path: Path) -> int | None:
    """返回路径中最后一个 images 分量的位置,不存在时返回 None."""
    parts = list(path.parts)
    return next(
        (index for index in range(len(parts) - 1, -1, -1) if parts[index].casefold() == "images"),
        None,
    )


def sibling_data_dir(image_dir: Path, directory_name: str) -> Path:
    """替换路径中最后一个 images 分量,得到对应标签或 mask 目录."""
    parts = list(image_dir.parts)
    image_index = image_component_index(image_dir)
    if image_index is None:
        raise PipelineError(f"YOLO注册图片目录中缺少images分量:{image_dir}")
    return Path(*parts[:image_index], directory_name, *parts[image_index + 1:])


def collect_yolo_images(source: Path) -> list[Path]:
    """按 Ultralytics 目录输入规则递归收集一个注册来源中的图片."""
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.casefold() in YOLO_IMAGE_SUFFIXES
    )


def yolo_label_path(image_path: Path, labels_dir: str) -> Path:
    """按照 img2label_paths 规则由图片路径得到一个任务的标签路径."""
    parts = list(image_path.parts)
    image_index = image_component_index(image_path)
    if image_index is None:
        return image_path.with_suffix(".txt")
    return Path(*parts[:image_index], labels_dir, *parts[image_index + 1:]).with_suffix(".txt")


def register_dataset(context: RunContext) -> StageOutcome:
    """将当前标准数据集幂等加入累计注册表并生成训练输入."""
    registry_path = context.config.registry_dir / "datasets.yaml"
    reuse_existing = (
        context.command == "train" and registry_path.is_file()
    ) or (context.command == "all" and not context.config.enable_data_update)
    if context.command == "all" and not context.config.enable_data_update and not registry_path.is_file():
        raise PipelineError(
            "PIPELINE_ENABLE_DATA_UPDATE=false 时必须存在累计数据注册表:"
            f"{registry_path}；请先设置 PIPELINE_ENABLE_DATA_UPDATE=true 完成一次数据处理和注册"
        )
    if reuse_existing:
        registry = load_registry(context)
        validate_registry(registry)
        generated = generate_training_inputs(context, registry)
        statistics = collect_registry_statistics(registry)
        context.logger.info(
            "DATA_REGISTRY | 只读复用 | yolo_sources=%d | train=%d | val=%d",
            sum(len(registry["yolo"][split]) for split in SPLITS),
            statistics["images"]["train"],
            statistics["images"]["val"],
        )
        log_training_dataset_summary(context, statistics)
        return StageOutcome(
            message="累计数据注册表只读复用完成",
            artifacts={
                "registry": str(registry_path),
                **{key: str(value) for key, value in generated.items()},
            },
            metrics=statistics,
            value={"registry": registry, "configs": generated, "statistics": statistics},
        )
    registry = load_registry(context)
    previous = json.dumps(registry, ensure_ascii=False, sort_keys=True)
    for dataset_dir in context.config.output_dirs:
        dataset_dir = dataset_dir.expanduser().resolve()
        validate_dataset_root(dataset_dir)
        for split in SPLITS:
            image_dir = dataset_dir / split / "images"
            registry["yolo"][split].append(str(image_dir))
            for image_path in sorted(image_dir.glob("*.jpg")):
                registry["pidnet"][split].append(
                    {
                        "image": str(image_path),
                        "mask": str(dataset_dir / split / "Seg" / f"{image_path.stem}.png"),
                    }
                )
    registry = normalize_registry(registry, context)
    status = "新增" if json.dumps(registry, ensure_ascii=False, sort_keys=True) != previous else "已存在"
    validate_registry(registry)
    registry_path = save_registry(context, registry)
    generated = generate_training_inputs(context, registry)
    statistics = collect_registry_statistics(registry)
    context.logger.info(
        "DATA_REGISTRY | %s | yolo_sources=%d | train=%d | val=%d",
        status,
        sum(len(registry["yolo"][split]) for split in SPLITS),
        statistics["images"]["train"],
        statistics["images"]["val"],
    )
    context.logger.info("DATA_STATISTICS | %s", json.dumps(statistics, ensure_ascii=False, sort_keys=True))
    log_training_dataset_summary(context, statistics)
    return StageOutcome(
        message=f"累计数据注册完成:{status}",
        artifacts={"registry": str(registry_path), **{key: str(value) for key, value in generated.items()}},
        metrics=statistics,
        value={"registry": registry, "configs": generated, "statistics": statistics},
    )


def validate_registry(registry: dict[str, Any]) -> None:
    """验证 YOLO 来源目录和 PIDNet 图片-mask 文件全部有效."""
    errors: list[str] = []
    for split in SPLITS:
        for value in registry["yolo"][split]:
            image_dir = Path(value).expanduser()
            if not image_dir.is_dir():
                errors.append(f"YOLO {split}图片目录不存在:{image_dir}")
                continue
            if not collect_yolo_images(image_dir):
                errors.append(f"YOLO {split}图片目录中没有支持的图片:{image_dir}")
                continue
            if image_component_index(image_dir) is not None:
                related_dirs = [sibling_data_dir(image_dir, name) for name in ("labels_detect", "labels_segment")]
                for label_dir in related_dirs:
                    if not label_dir.is_dir():
                        errors.append(f"YOLO {split}标签目录不存在:{label_dir}")
        for entry in registry["pidnet"][split]:
            image_path = Path(entry["image"]).expanduser()
            mask_path = Path(entry["mask"]).expanduser()
            if not image_path.is_file():
                errors.append(f"PIDNet {split}图片不存在:{image_path}")
            if not mask_path.is_file():
                errors.append(f"PIDNet {split} mask不存在:{mask_path}")
    if errors:
        raise PipelineError(f"累计数据注册表校验失败,共{len(errors)}项:\n" + "\n".join(errors[:100]))


def generate_training_inputs(context: RunContext, registry: dict[str, Any]) -> dict[str, Path]:
    """从唯一注册表生成两份YOLO配置、PIDNet列表和配置快照."""
    config_dir = context.work_dir / "training_configs"
    list_dir = context.work_dir / "pidnet_list"
    config_dir.mkdir(parents=True, exist_ok=True)
    list_dir.mkdir(parents=True, exist_ok=True)
    image_dirs = {split: list(registry["yolo"][split]) for split in SPLITS}
    outputs: dict[str, Path] = {}
    for task, labels_dir, names in (
        ("detect", "labels_detect", {0: "paper", 1: "liquid", 2: "metal"}),
        ("segment", "labels_segment", {0: "paper"}),
    ):
        path = config_dir / f"yolo_{task}.yaml"
        payload = {
            "path": str(context.work_dir),
            "train": image_dirs["train"],
            "val": image_dirs["val"],
            "labels_dir": labels_dir,
            "nc": len(names),
            "names": names,
        }
        atomic_write_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
        outputs[task] = path

    for split in SPLITS:
        list_path = list_dir / f"{split}.lst"
        lines = [f"{entry['image']}\t{entry['mask']}\n" for entry in registry["pidnet"][split]]
        atomic_write_text(list_path, "".join(lines))
        outputs[f"pidnet_{split}_list"] = list_path

    source_config = context.config.pidnet_config.expanduser().resolve()
    if not source_config.is_file():
        raise PipelineError(f"PIDNet配置不存在:{source_config}")
    try:
        pidnet_payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"PIDNet配置读取失败:{source_config},原因:{exc}") from exc
    if not isinstance(pidnet_payload, dict):
        raise PipelineError(f"PIDNet配置必须是字典:{source_config}")
    train_config = pidnet_payload.setdefault("TRAIN", {})
    test_config = pidnet_payload.setdefault("TEST", {})
    dataset_config = pidnet_payload.setdefault("DATASET", {})
    train_config["END_EPOCH"] = context.config.pidnet_epochs
    train_config["BATCH_SIZE_PER_GPU"] = context.config.pidnet_batch
    train_config["IMAGE_SIZE"] = context.config.model_image_size
    test_config["IMAGE_SIZE"] = context.config.model_image_size
    pidnet_payload["WORKERS"] = context.config.pidnet_workers
    dataset_config["ROOT"] = f"{context.work_dir}{os.sep}"
    dataset_config["TRAIN_SET"] = "pidnet_list/train.lst"
    dataset_config["TEST_SET"] = "pidnet_list/val.lst"
    pidnet_path = config_dir / "pidnet.yaml"
    atomic_write_text(pidnet_path, yaml.safe_dump(pidnet_payload, allow_unicode=True, sort_keys=False))
    outputs["pidnet"] = pidnet_path
    return outputs


def count_yolo_labels(label_files: list[Path], classes: dict[str, int]) -> dict[str, int]:
    """统计一组 YOLO 标签文件中的各类别实例数量."""
    counts = {name: 0 for name in classes}
    id_to_name = {class_id: name for name, class_id in classes.items()}
    for path in label_files:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                continue
            if class_id in id_to_name:
                counts[id_to_name[class_id]] += 1
    return counts


def collect_registry_statistics(registry: dict[str, Any]) -> dict[str, Any]:
    """按模型实际使用的累计注册表统计图片、实例和像素."""
    result: dict[str, Any] = {
        "images": {split: 0 for split in SPLITS},
        "detect": {split: {name: 0 for name in DETECT_CLASSES} for split in SPLITS},
        "segment": {split: {name: 0 for name in SEGMENT_CLASSES} for split in SPLITS},
        "pidnet_pixels": {split: {name: 0 for name in PIDNET_CLASSES} for split in SPLITS},
    }
    for split in SPLITS:
        for value in registry["yolo"][split]:
            image_dir = Path(value)
            image_paths = collect_yolo_images(image_dir)
            result["images"][split] += len(image_paths)
            for task, directory, classes in (
                ("detect", "labels_detect", DETECT_CLASSES),
                ("segment", "labels_segment", SEGMENT_CLASSES),
            ):
                label_files = [yolo_label_path(image_path, directory) for image_path in image_paths]
                counts = count_yolo_labels(label_files, classes)
                for name, count in counts.items():
                    result[task][split][name] += count
        for entry in registry["pidnet"][split]:
            mask_path = Path(entry["mask"])
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise PipelineError(f"无法读取累计PIDNet mask:{mask_path}")
            for name, class_id in PIDNET_CLASSES.items():
                result["pidnet_pixels"][split][name] += int(np.count_nonzero(mask == class_id))
    return result


def log_training_dataset_summary(context: RunContext, statistics: dict[str, Any]) -> None:
    """输出训练前累计 YOLO 检测数据的易读统计摘要。"""
    detect = statistics["detect"]
    labels = {
        split: sum(int(count) for count in detect[split].values())
        for split in SPLITS
    }
    classes = {
        name: {split: int(detect[split][name]) for split in SPLITS}
        for name in DETECT_CLASSES
    }
    context.logger.info(
        "TRAINING_DATASET_SUMMARY | train_images=%d | val_images=%d | "
        "train_labels=%d | val_labels=%d | classes=%s",
        statistics["images"]["train"],
        statistics["images"]["val"],
        labels["train"],
        labels["val"],
        json.dumps(classes, ensure_ascii=False, sort_keys=True),
    )


def save_registry_statistics(context: RunContext, statistics: dict[str, Any]) -> Path:
    """保存累计统计快照,避免报表阶段重新猜测统计口径."""
    path = context.work_dir / "reports" / "dataset_statistics.json"
    atomic_write_text(path, json.dumps(statistics, ensure_ascii=False, indent=2))
    return path
