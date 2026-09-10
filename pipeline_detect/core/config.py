"""纯检测流水线的环境变量配置。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pipeline.core.errors import PipelineError


def _text(name: str, default: str = "") -> str:
    """读取并去除环境变量首尾空白。"""
    return os.environ.get(name, default).strip()


def _integer(name: str, default: int, minimum: int = 0) -> int:
    """读取限定下界的整数环境变量。"""
    try:
        value = int(_text(name, str(default)))
    except ValueError as exc:
        raise PipelineError(f"环境变量{name}必须是整数") from exc
    if value < minimum:
        raise PipelineError(f"环境变量{name}必须大于等于{minimum}")
    return value


def _number(name: str, default: float, minimum: float = 0.0) -> float:
    """读取限定下界的浮点环境变量。"""
    try:
        value = float(_text(name, str(default)))
    except ValueError as exc:
        raise PipelineError(f"环境变量{name}必须是数字") from exc
    if value < minimum:
        raise PipelineError(f"环境变量{name}必须大于等于{minimum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    """读取严格的布尔环境变量。"""
    value = _text(name, str(default).lower()).casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise PipelineError(f"环境变量{name}必须是true或false")


def _path_list(name: str) -> tuple[Path, ...]:
    """读取非空 JSON 字符串路径数组。"""
    try:
        values = json.loads(_text(name))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"环境变量{name}必须是JSON路径数组") from exc
    if not isinstance(values, list) or not values or not all(isinstance(x, str) and x.strip() for x in values):
        raise PipelineError(f"环境变量{name}必须是非空JSON路径数组")
    return tuple(Path(value).expanduser() for value in values)


def _int_list(name: str, default: str) -> tuple[int, ...]:
    """读取 JSON 非负整数数组。"""
    try:
        values = json.loads(_text(name, default))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"环境变量{name}必须是JSON整数数组") from exc
    if not isinstance(values, list) or any(not isinstance(value, int) or value < 0 for value in values):
        raise PipelineError(f"环境变量{name}必须是非负整数JSON数组")
    if len(set(values)) != len(values):
        raise PipelineError(f"环境变量{name}不允许重复类别ID")
    return tuple(values)


def _number_list(name: str, default: str) -> tuple[float, ...]:
    """读取 JSON 概率数组，并校验范围与重复值。"""
    try:
        values = json.loads(_text(name, default))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"环境变量{name}必须是JSON数字数组") from exc
    if not isinstance(values, list) or not values:
        raise PipelineError(f"环境变量{name}必须是非空JSON数字数组")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise PipelineError(f"环境变量{name}必须是JSON数字数组")
    numbers = tuple(float(value) for value in values)
    if any(not 0.0 <= value <= 1.0 for value in numbers):
        raise PipelineError(f"环境变量{name}中的数值必须在0和1之间")
    if len(set(numbers)) != len(numbers):
        raise PipelineError(f"环境变量{name}不允许重复数值")
    return numbers


def _text_list(name: str, default: str = "[]") -> tuple[str, ...]:
    """读取 JSON 非空字符串数组，并转换为不区分大小写的匹配关键字。"""
    try:
        values = json.loads(_text(name, default))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"环境变量{name}必须是JSON字符串数组") from exc
    if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
        raise PipelineError(f"环境变量{name}必须是JSON非空字符串数组")
    keywords = tuple(value.strip().casefold() for value in values)
    if len(set(keywords)) != len(keywords):
        raise PipelineError(f"环境变量{name}不允许重复关键字")
    return keywords


@dataclass(frozen=True)
class DetectConfig:
    """纯检测流水线的一次不可变运行配置。"""

    repo_root: Path
    model_train_root: Path
    run_name: str
    input_dirs: tuple[Path, ...]
    output_dirs: tuple[Path, ...]
    excel_path: Path
    yolo_env: str
    gpu_device: int
    detect_epochs: int
    detect_batch: int
    image_height: int
    image_width: int
    yolo_workers: int
    yolo_amp: bool
    auto_batch: bool
    force_train: bool
    auto_finetune: bool
    enable_data_update: bool
    include_negative_samples: bool
    ignore_class_ids: tuple[int, ...]
    detect_fallback_weight: Path
    display_runtime_prefix: str
    display_nas_prefix: str
    val_ratio: float
    random_seed: int
    test_iou: float
    labeled_test_enabled: bool
    labeled_test_dir: Path
    test_output_dir: Path
    labeled_test_confidences: tuple[float, ...]
    labeled_test_exclude_keywords: tuple[str, ...]

    @classmethod
    def from_environment(cls) -> "DetectConfig":
        """从日期 Shell 脚本导出的环境变量构造配置。"""
        repo_root = Path(__file__).resolve().parents[2]
        root = Path(_text("PIPELINE_MODEL_TRAIN_ROOT", "~/model_training")).expanduser()
        run_name = _text("PIPELINE_RUN_NAME")
        config = cls(
            repo_root=repo_root,
            model_train_root=root,
            run_name=run_name,
            input_dirs=_path_list("PIPELINE_INPUT_DIR"),
            output_dirs=_path_list("PIPELINE_OUTPUT_DIR"),
            excel_path=Path(_text("PIPELINE_EXCEL_PATH", str(root / "模型训练版本汇总_精简版.xlsx"))).expanduser(),
            yolo_env=_text("PIPELINE_YOLO_ENV", "ult"),
            gpu_device=_integer("PIPELINE_GPU_DEVICE", 0),
            detect_epochs=_integer("PIPELINE_DETECT_EPOCHS", 200, 1),
            detect_batch=_integer("PIPELINE_DETECT_BATCH", 12, 1),
            image_height=_integer("PIPELINE_IMAGE_HEIGHT", 800, 1),
            image_width=_integer("PIPELINE_IMAGE_WIDTH", 1280, 1),
            yolo_workers=_integer("PIPELINE_YOLO_WORKERS", 16),
            yolo_amp=_boolean("PIPELINE_YOLO_AMP", False),
            auto_batch=_boolean("PIPELINE_AUTO_BATCH", False),
            force_train=_boolean("PIPELINE_FORCE_TRAIN", False),
            auto_finetune=_boolean("PIPELINE_AUTO_FINETUNE", True),
            enable_data_update=_boolean("PIPELINE_ENABLE_DATA_UPDATE", True),
            include_negative_samples=_boolean("PIPELINE_INCLUDE_NEGATIVE_SAMPLES", True),
            ignore_class_ids=_int_list("PIPELINE_IGNORE_CLASS_IDS", "[3]"),
            detect_fallback_weight=Path(_text("PIPELINE_DETECT_FALLBACK_WEIGHT")).expanduser(),
            display_runtime_prefix=_text("PIPELINE_DISPLAY_RUNTIME_PREFIX", str(Path("~/nas_smb").expanduser())).rstrip("/\\"),
            display_nas_prefix=_text("PIPELINE_DISPLAY_NAS_PREFIX", r"\\192.168.0.68").rstrip("/\\"),
            val_ratio=_number("PIPELINE_VAL_RATIO", 0.2),
            random_seed=_integer("PIPELINE_RANDOM_SEED", 42),
            test_iou=_number("PIPELINE_TEST_IOU", 0.5),
            labeled_test_enabled=_boolean("PIPELINE_LABELED_TEST_ENABLED", False),
            labeled_test_dir=Path(_text("PIPELINE_LABELED_TEST_DIR", ".")).expanduser(),
            test_output_dir=Path(_text("PIPELINE_TEST_OUTPUT_DIR", ".")).expanduser(),
            labeled_test_confidences=_number_list(
                "PIPELINE_LABELED_TEST_CONFIDENCES",
                "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]",
            ),
            labeled_test_exclude_keywords=_text_list("PIPELINE_LABELED_TEST_EXCLUDE_KEYWORDS"),
        )
        config.validate_values()
        return config

    @property
    def registry_dir(self) -> Path:
        """返回与原 Pipeline 共用的注册表目录。"""
        return self.model_train_root / "pipeline_registry"

    @property
    def work_dir(self) -> Path:
        """返回本次运行的专属目录。"""
        return self.model_train_root / self.run_name

    @property
    def detect_model_yaml(self) -> Path:
        """返回 P2 检测模型结构文件。"""
        return self.repo_root / "ultralytics/cfg/models/11/yolo11s_p2.yaml"

    @property
    def requested_image_size(self) -> list[int]:
        """返回外部输入尺寸 [height, width]。"""
        return [self.image_height, self.image_width]

    @property
    def model_image_size(self) -> list[int]:
        """返回 32 对齐的训练输入尺寸。"""
        return [((size + 31) // 32) * 32 for size in self.requested_image_size]

    @property
    def image_padding(self) -> list[int]:
        """返回 ONNX 居中补边 [left, right, top, bottom]。"""
        height, width = self.requested_image_size
        model_height, model_width = self.model_image_size
        left, top = (model_width - width) // 2, (model_height - height) // 2
        return [left, model_width - width - left, top, model_height - height - top]

    def validate_values(self) -> None:
        """验证跨平台安全和训练参数约束。"""
        if not self.run_name or self.run_name in {".", ".."} or "/" in self.run_name or "\\" in self.run_name:
            raise PipelineError("PIPELINE_RUN_NAME必须是单层非空目录名")
        if len(self.input_dirs) != len(self.output_dirs):
            raise PipelineError("PIPELINE_INPUT_DIR与PIPELINE_OUTPUT_DIR长度必须一致")
        if len(set(self.output_dirs)) != len(self.output_dirs):
            raise PipelineError("PIPELINE_OUTPUT_DIR不允许重复")
        if not 0.0 < self.val_ratio < 1.0:
            raise PipelineError("PIPELINE_VAL_RATIO必须在0和1之间")
        if not 0.0 <= self.test_iou <= 1.0:
            raise PipelineError("PIPELINE_TEST_IOU必须在0和1之间")
        if self.labeled_test_enabled:
            output = self.test_output_dir.resolve()
            if output.name != self.run_name:
                raise PipelineError("PIPELINE_TEST_OUTPUT_DIR末级目录必须等于PIPELINE_RUN_NAME")
            try:
                output.relative_to(self.model_train_root.resolve())
            except ValueError:
                pass
            else:
                raise PipelineError("PIPELINE_TEST_OUTPUT_DIR不能位于模型训练根目录内")

    def snapshot(self) -> dict[str, Any]:
        """返回可写入运行状态的完整配置快照。"""
        def serialize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [serialize(item) for item in value]
            return value

        payload = {key: serialize(value) for key, value in asdict(self).items()}
        payload.update(requested_image_size=self.requested_image_size, model_image_size=self.model_image_size, image_padding=self.image_padding)
        return payload
