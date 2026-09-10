"""纯检测 Pipeline 的命令入口。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "pipeline_detect"

from pipeline.core.context import RunContext
from pipeline.core.errors import ExcelPendingError, PipelineError
from pipeline.core.stage_runner import StageOutcome, run_stage
from pipeline.core.state import StageStatus
from pipeline.steps.exporting import export_yolo_detect
from pipeline.steps.training import train_yolo

from .core.config import DetectConfig
from .steps.data_registry import prepare_dataset, register_dataset
from .steps.labeled_testing import run_labeled_detect_test
from .steps.reporting import write_report

COMMANDS = ("prepare", "train", "export", "test", "report", "status", "all")


def parse_args() -> argparse.Namespace:
    """解析流水线子命令和强制重训选项。"""
    parser = argparse.ArgumentParser(description="清洁机器人纯检测训练流水线")
    parser.add_argument("command", choices=COMMANDS, nargs="?", default="all")
    parser.add_argument("--force-train", action="store_true", help="备份当前检测训练目录后重新训练")
    return parser.parse_args()


def run_preflight(config: DetectConfig, command: str) -> None:
    """在写入前完成命令所需目录、模型和参数预检。"""
    if not config.model_train_root.is_dir():
        raise PipelineError(f"模型训练根目录不存在:{config.model_train_root}")
    if command in {"prepare", "all"} and config.enable_data_update:
        for source in config.input_dirs:
            if not source.expanduser().is_dir():
                raise PipelineError(f"输入目录不存在:{source}")
    if command in {"train", "all"} and not config.detect_model_yaml.is_file():
        raise PipelineError(f"检测模型结构不存在:{config.detect_model_yaml}")
    if command in {"test", "all"} and not (config.repo_root / "tools" / "inference.py").is_file():
        raise PipelineError("缺少tools/inference.py")
    if command in {"test", "all"} and config.labeled_test_enabled:
        if not config.labeled_test_dir.is_dir():
            raise PipelineError(f"外部带标签测试目录不存在:{config.labeled_test_dir}")
    if command == "test" and not config.labeled_test_enabled:
        raise PipelineError("test要求PIPELINE_LABELED_TEST_ENABLED=true")


def _artifact(context: RunContext, stage: str, key: str) -> Path:
    """从成功阶段读取存在的文件产物。"""
    value = context.state.get_stage(stage).artifacts.get(key)
    if not value:
        raise PipelineError(f"阶段{stage}没有产物{key}，请先执行前置命令")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise PipelineError(f"阶段{stage}记录的产物不存在:{path}")
    return path


def prepare(context: RunContext) -> dict[str, Any]:
    """处理本批原始检测 TXT 数据并登记到累计来源。"""
    prepared = run_stage(context, "prepare_data", lambda: prepare_dataset(context))
    registered = run_stage(context, "register_data", lambda: register_dataset(context))
    return {"prepared": prepared.value, "registered": registered.value}


def ensure_registered(context: RunContext) -> Path:
    """训练前生成或只读复用本次检测训练 YAML。"""
    outcome = run_stage(context, "register_data", lambda: register_dataset(context))
    return Path(outcome.value["config"])


def show_status(context: RunContext) -> None:
    """打印可恢复阶段状态和关键产物。"""
    payload = {name: {"status": record.status, "attempts": record.attempts, "message": record.message, "artifacts": record.artifacts} for name, record in context.state.stages().items()}
    context.logger.info("PIPELINE_STATUS | %s", json.dumps(payload, ensure_ascii=False))


def run_labeled_test_if_enabled(context: RunContext, onnx_path: Path) -> StageOutcome:
    """按配置运行外部带标签测试，关闭时记录跳过状态。"""
    if not context.config.labeled_test_enabled:
        return StageOutcome(
            status=StageStatus.SKIPPED,
            message="PIPELINE_LABELED_TEST_ENABLED=false，跳过外部带标签测试",
        )
    return run_labeled_detect_test(context, onnx_path)


def execute(context: RunContext) -> None:
    """按照命令执行可独立重跑的纯检测阶段。"""
    command = context.command
    if command == "prepare":
        prepare(context)
    elif command == "train":
        run_stage(context, "train_detect", lambda: train_yolo(context, "detect", ensure_registered(context)))
    elif command == "export":
        run_stage(context, "export_detect", lambda: export_yolo_detect(context, _artifact(context, "train_detect", "best")))
    elif command == "test":
        onnx_path = _artifact(context, "export_detect", "onnx")
        run_stage(
            context,
            "labeled_test",
            lambda: run_labeled_detect_test(context, onnx_path),
        )
    elif command == "report":
        run_stage(context, "report", lambda: write_report(context))
    elif command == "status":
        show_status(context)
    else:
        if context.config.enable_data_update:
            data_yaml = prepare(context)["registered"]["config"]
        else:
            run_stage(context, "prepare_data", lambda: StageOutcome(status=StageStatus.SKIPPED, message="PIPELINE_ENABLE_DATA_UPDATE=false，复用累计检测数据"))
            data_yaml = ensure_registered(context)
        trained = run_stage(context, "train_detect", lambda: train_yolo(context, "detect", Path(data_yaml)))
        exported = run_stage(context, "export_detect", lambda: export_yolo_detect(context, trained.value))
        run_stage(
            context,
            "labeled_test",
            lambda: run_labeled_test_if_enabled(context, exported.value),
        )
        run_stage(context, "report", lambda: write_report(context))


def main() -> int:
    """创建上下文、执行预检并持久化最终运行状态。"""
    args = parse_args()
    context: RunContext | None = None
    try:
        config = DetectConfig.from_environment()
        if args.force_train:
            config = replace(config, force_train=True)
        run_preflight(config, args.command)
        context = RunContext.create(config, args.command)
        context.logger.info("CONFIG | %s", json.dumps(config.snapshot(), ensure_ascii=False, sort_keys=True))
        context.logger.info("PREFLIGHT_OK | command=%s", args.command)
        execute(context)
        context.finish("success", "所有请求阶段已完成")
        return 0
    except ExcelPendingError as exc:
        if context is not None:
            context.logger.error("EXCEL_PENDING | %s", exc)
            context.finish("partial", "模型产物安全，中心Excel待补写；请关闭Excel后执行report")
        else:
            print(f"ERROR:{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if context is not None:
            context.logger.warning("PIPELINE_INTERRUPTED | 用户中断，可用同一脚本恢复")
            context.finish("failed", "用户中断")
        return 130
    except Exception as exc:
        if context is not None:
            context.logger.exception("PIPELINE_FAILED | %s", exc)
            context.finish("failed", str(exc))
        else:
            print(f"ERROR:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
