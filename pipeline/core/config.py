"""流水线固定配置及日期脚本环境变量解析."""

from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import PipelineError


def _env_text(name: str, default: str = "") -> str:
    """读取并清理一个字符串环境变量."""
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """读取整数环境变量并检查下限."""
    value = _env_text(name, str(default))
    try:
        result = int(value)
    except ValueError as exc:
        raise PipelineError(f"环境变量 {name} 必须是整数,当前值:{value!r}") from exc
    if result < minimum:
        raise PipelineError(f"环境变量 {name} 必须大于等于 {minimum},当前值:{result}")
    return result


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    """读取浮点环境变量并检查下限."""
    value = _env_text(name, str(default))
    try:
        result = float(value)
    except ValueError as exc:
        raise PipelineError(f"环境变量 {name} 必须是数字,当前值:{value!r}") from exc
    if result < minimum:
        raise PipelineError(f"环境变量 {name} 必须大于等于 {minimum},当前值:{result}")
    return result


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量."""
    value = _env_text(name, "true" if default else "false").casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise PipelineError(f"环境变量 {name} 必须是 true/false,当前值:{value!r}")


def _env_path(name: str, default: str = "") -> Path:
    """读取路径环境变量,仅展开用户目录而不提前创建路径."""
    value = _env_text(name, default)
    return Path(value).expanduser() if value else Path()


def _env_path_list(name: str) -> tuple[Path, ...]:
    """读取 JSON 路径列表环境变量并拒绝空列表和非字符串元素。"""
    value = _env_text(name)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"环境变量 {name} 必须是 JSON 路径列表,当前值:{value!r}") from exc
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
        raise PipelineError(f"环境变量 {name} 必须是非空字符串路径列表")
    return tuple(Path(item).expanduser() for item in payload)


def _env_prelabel_path_list(name: str) -> tuple[Path, ...]:
    """读取预标注路径列表，并兼容历史单路径配置。"""
    value = _env_text(name)
    if not value:
        raise PipelineError(f"环境变量 {name} 必须是非空路径或 JSON 路径列表")
    if not value.startswith("["):
        return (Path(value).expanduser(),)
    return _env_path_list(name)


@dataclass(frozen=True)
class PipelineConfig:
    """一次日期版脚本完整解析后的不可变配置."""

    repo_root: Path
    pidnet_root: Path
    model_train_root: Path
    run_name: str
    input_dirs: tuple[Path, ...]
    output_dirs: tuple[Path, ...]
    prelabel_dirs: tuple[Path, ...]
    excel_path: Path
    yolo_env: str
    pidnet_env: str
    gpu_device: int
    detect_epochs: int
    segment_epochs: int
    pidnet_epochs: int
    detect_batch: int
    segment_batch: int
    pidnet_batch: int
    image_height: int
    image_width: int
    yolo_workers: int
    pidnet_workers: int
    yolo_amp: bool
    auto_batch: bool
    force_train: bool
    auto_finetune: bool
    enable_data_update: bool
    enable_prelabel: bool
    detect_fallback_weight: Path
    segment_fallback_weight: Path
    pidnet_fallback_weight: Path
    detect_model_yaml: Path
    segment_model_yaml: Path
    pidnet_config: Path
    display_runtime_prefix: str
    display_nas_prefix: str
    val_ratio: float
    random_seed: int
    jpg_quality: int
    prelabel_confidence: float
    prelabel_iou: float
    prelabel_batch: int
    prelabel_half: bool
    prelabel_retina_masks: bool
    prelabel_polygon_epsilon: float
    pidnet_prelabel_min_area: float
    prelabel_overwrite: bool
    test_images: int
    test_seed: int
    test_confidence: float
    test_iou: float

    @classmethod
    def from_environment(cls) -> "PipelineConfig":
        """从日期版 Shell 脚本导出的环境变量创建配置."""
        repo_root = Path(__file__).resolve().parents[2]
        pidnet_root = _env_path("PIPELINE_PIDNET_ROOT", "~/gitee/pidnet")
        model_train_root = _env_path(
            "PIPELINE_MODEL_TRAIN_ROOT",
            "~/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/"
            "应用场景/【舜宇】清洁机器人/DEMO开发/模型训练",
        )
        run_name = _env_text("PIPELINE_RUN_NAME")
        if not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
            raise PipelineError("PIPELINE_RUN_NAME 必须是单层非空目录名称")
        config = cls(
            repo_root=repo_root,
            pidnet_root=pidnet_root,
            model_train_root=model_train_root,
            run_name=run_name,
            input_dirs=_env_path_list("PIPELINE_INPUT_DIR"),
            output_dirs=_env_path_list("PIPELINE_OUTPUT_DIR"),
            prelabel_dirs=_env_prelabel_path_list("PIPELINE_PRELABEL_DIR"),
            excel_path=_env_path(
                "PIPELINE_EXCEL_PATH",
                str(model_train_root / "模型训练版本汇总_精简版.xlsx"),
            ),
            yolo_env=_env_text("PIPELINE_YOLO_ENV", "ult"),
            pidnet_env=_env_text("PIPELINE_PIDNET_ENV", "pid"),
            gpu_device=_env_int("PIPELINE_GPU_DEVICE", 0, minimum=0),
            detect_epochs=_env_int("PIPELINE_DETECT_EPOCHS", 200),
            segment_epochs=_env_int("PIPELINE_SEGMENT_EPOCHS", 200),
            pidnet_epochs=_env_int("PIPELINE_PIDNET_EPOCHS", 200),
            detect_batch=_env_int("PIPELINE_DETECT_BATCH", 12),
            segment_batch=_env_int("PIPELINE_SEGMENT_BATCH", 12),
            pidnet_batch=_env_int("PIPELINE_PIDNET_BATCH", 24),
            image_height=_env_int("PIPELINE_IMAGE_HEIGHT", 800),
            image_width=_env_int("PIPELINE_IMAGE_WIDTH", 1280),
            yolo_workers=_env_int("PIPELINE_YOLO_WORKERS", 16, minimum=0),
            pidnet_workers=_env_int("PIPELINE_PIDNET_WORKERS", 16, minimum=0),
            yolo_amp=_env_bool("PIPELINE_YOLO_AMP", False),
            auto_batch=_env_bool("PIPELINE_AUTO_BATCH", False),
            force_train=_env_bool("PIPELINE_FORCE_TRAIN", False),
            auto_finetune=_env_bool("PIPELINE_AUTO_FINETUNE", True),
            enable_data_update=_env_bool("PIPELINE_ENABLE_DATA_UPDATE", True),
            enable_prelabel=_env_bool("PIPELINE_ENABLE_PRELABEL", True),
            detect_fallback_weight=_env_path("PIPELINE_DETECT_FALLBACK_WEIGHT"),
            segment_fallback_weight=_env_path("PIPELINE_SEGMENT_FALLBACK_WEIGHT"),
            pidnet_fallback_weight=_env_path("PIPELINE_PIDNET_FALLBACK_WEIGHT"),
            detect_model_yaml=repo_root / "ultralytics/cfg/models/11/yolo11s_p2.yaml",
            segment_model_yaml=repo_root / "ultralytics/cfg/models/11/yolo11s-p2-seg.yaml",
            pidnet_config=_env_path(
                "PIPELINE_PIDNET_CONFIG",
                str(pidnet_root / "configs/liquid_metal_zicai260729.yaml"),
            ),
            display_runtime_prefix=_env_text(
                "PIPELINE_DISPLAY_RUNTIME_PREFIX",
                str(Path("~/nas_smb").expanduser()),
            ).rstrip("/\\"),
            display_nas_prefix=_env_text("PIPELINE_DISPLAY_NAS_PREFIX", r"\\192.168.0.68").rstrip("/\\"),
            val_ratio=_env_float("PIPELINE_VAL_RATIO", 0.2),
            random_seed=_env_int("PIPELINE_RANDOM_SEED", 42, minimum=0),
            jpg_quality=_env_int("PIPELINE_JPG_QUALITY", 95),
            prelabel_confidence=_env_float("PIPELINE_PRELABEL_CONFIDENCE", 0.25),
            prelabel_iou=_env_float("PIPELINE_PRELABEL_IOU", 0.7),
            prelabel_batch=_env_int("PIPELINE_PRELABEL_BATCH", 1),
            prelabel_half=_env_bool("PIPELINE_PRELABEL_HALF", False),
            prelabel_retina_masks=_env_bool("PIPELINE_PRELABEL_RETINA_MASKS", True),
            prelabel_polygon_epsilon=_env_float("PIPELINE_PRELABEL_POLYGON_EPSILON", 0.1),
            pidnet_prelabel_min_area=_env_float("PIPELINE_PIDNET_PRELABEL_MIN_AREA", 10.0),
            prelabel_overwrite=_env_bool("PIPELINE_PRELABEL_OVERWRITE", False),
            test_images=_env_int("PIPELINE_TEST_IMAGES", 500),
            test_seed=_env_int("PIPELINE_TEST_SEED", 42, minimum=0),
            test_confidence=_env_float("PIPELINE_TEST_CONFIDENCE", 0.25),
            test_iou=_env_float("PIPELINE_TEST_IOU", 0.5),
        )
        config.validate_values()
        return config

    @property
    def work_dir(self) -> Path:
        """返回本次训练运行目录."""
        return self.model_train_root / self.run_name

    @property
    def registry_dir(self) -> Path:
        """返回跨批次共享的数据与模型注册表目录."""
        return self.model_train_root / "pipeline_registry"

    @property
    def requested_image_size(self) -> list[int]:
        """返回外部模型输入尺寸，顺序为 [height, width]。"""
        return [self.image_height, self.image_width]

    @property
    def model_image_size(self) -> list[int]:
        """返回训练和骨干网络使用的 32 对齐尺寸。"""
        return [((value + 31) // 32) * 32 for value in self.requested_image_size]

    @property
    def image_padding(self) -> list[int]:
        """返回 ONNX 输入居中补边，顺序为 [left, right, top, bottom]。"""
        height, width = self.requested_image_size
        model_height, model_width = self.model_image_size
        left = (model_width - width) // 2
        top = (model_height - height) // 2
        return [left, model_width - width - left, top, model_height - height - top]

    def validate_values(self) -> None:
        """检查不依赖文件系统的配置约束."""
        if not 0.0 < self.val_ratio < 1.0:
            raise PipelineError(f"PIPELINE_VAL_RATIO 必须在 0 和 1 之间,当前值:{self.val_ratio}")
        if len(self.input_dirs) != len(self.output_dirs):
            raise PipelineError(
                "PIPELINE_INPUT_DIR 与 PIPELINE_OUTPUT_DIR 的路径数量必须一致,"
                f"当前输入={len(self.input_dirs)},输出={len(self.output_dirs)}"
            )
        if len(set(self.output_dirs)) != len(self.output_dirs):
            raise PipelineError("PIPELINE_OUTPUT_DIR 不允许包含重复输出目录")
        if not 1 <= self.jpg_quality <= 100:
            raise PipelineError(f"PIPELINE_JPG_QUALITY 必须在 1～100 之间,当前值:{self.jpg_quality}")
        if not 0.0 <= self.prelabel_confidence <= 1.0:
            raise PipelineError("PIPELINE_PRELABEL_CONFIDENCE 必须在 0～1 之间")
        if not 0.0 <= self.prelabel_iou <= 1.0:
            raise PipelineError("PIPELINE_PRELABEL_IOU 必须在 0～1 之间")
        if not 0.0 <= self.test_confidence <= 1.0:
            raise PipelineError("PIPELINE_TEST_CONFIDENCE 必须在 0～1 之间")
        if not 0.0 <= self.test_iou <= 1.0:
            raise PipelineError("PIPELINE_TEST_IOU 必须在 0～1 之间")

        legacy = [name for name in ("PIPELINE_YOLO_IMAGE_SIZE", "PIPELINE_PIDNET_IMAGE_SIZE") if name in os.environ]
        if legacy:
            raise PipelineError(
                f"已废弃尺寸变量 {', '.join(legacy)}，请统一使用 PIPELINE_IMAGE_HEIGHT 和 PIPELINE_IMAGE_WIDTH。"
            )

    def should_run_prelabel(self, command: str) -> bool:
        """判断当前命令是否明确要求执行预标注."""
        return command == "prelabel" or (command == "all" and self.enable_prelabel)

    def snapshot(self) -> dict[str, Any]:
        """返回可写入 JSON 的完整配置快照."""
        payload = asdict(self)
        payload.update(
            requested_image_size=self.requested_image_size,
            model_image_size=self.model_image_size,
            image_padding=self.image_padding,
        )
        def serialize(value: Any) -> Any:
            """把配置中的 Path、元组和列表转换为 JSON 可写类型。"""
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, (list, tuple)):
                return [serialize(item) for item in value]
            return value

        return {key: serialize(value) for key, value in payload.items()}
