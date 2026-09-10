#!/usr/bin/env python3
"""清洁机器人数据、三模型训练、导出、预标注和报表统一入口."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "pipeline"

from .core.config import PipelineConfig
from .core.context import RunContext
from .core.errors import ExcelPendingError, PipelineError
from .core.preflight import run_preflight
from .core.stage_runner import StageOutcome, run_stage
from .core.state import StageStatus
from .steps.data import prepare_dataset
from .steps.exporting import export_pidnet, export_yolo_detect, export_yolo_segment
from .steps.prelabel import generate_prelabels
from .steps.registry import register_dataset, save_registry_statistics
from .steps.reporting import write_report
from .steps.training import train_pidnet, train_yolo
from .steps.testing import run_model_tests


COMMANDS = ("prepare", "train", "export", "test", "prelabel", "report", "status", "all")


def parse_args() -> argparse.Namespace:
    """解析简洁子命令;日期策略由 Shell 环境变量提供."""
    parser = argparse.ArgumentParser(description="清洁机器人视觉训练流水线.")
    parser.add_argument("command", choices=COMMANDS, nargs="?", default="all")
    parser.add_argument("--force-train", action="store_true", help="本次强制重新训练三个模型.")
    return parser.parse_args()


def _value_from_stage(context: RunContext, stage_name: str, artifact: str) -> Path:
    """从状态中获取已完成阶段产物并检查文件存在."""
    value = context.state.get_stage(stage_name).artifacts.get(artifact)
    if not value:
        raise PipelineError(f"阶段{stage_name}没有记录产物{artifact},请先执行对应前置命令")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise PipelineError(f"阶段{stage_name}记录的产物不存在:{path}")
    return path.resolve()


def prepare(context: RunContext) -> dict[str, Any]:
    """处理当前批次数据并幂等加入累计数据注册表."""
    prepared = run_stage(context, "prepare_data", lambda: prepare_dataset(context))
    registered = run_stage(context, "register_data", lambda: register_dataset(context))
    save_registry_statistics(context, registered.metrics)
    return {"prepared": prepared.value, "registered": registered.value}


def ensure_registered(context: RunContext) -> dict[str, Any]:
    """训练前重新校验注册表并生成本次配置快照."""
    registered = run_stage(context, "register_data", lambda: register_dataset(context))
    save_registry_statistics(context, registered.metrics)
    return registered.value


def train(context: RunContext, registered: dict[str, Any] | None = None) -> dict[str, Path]:
    """依次训练/恢复两个 YOLO P2 模型和 PIDNet."""
    if registered is None:
        registered = ensure_registered(context)
    configs = registered["configs"]
    detect = run_stage(context, "train_detect", lambda: train_yolo(context, "detect", configs["detect"]))
    segment = run_stage(context, "train_segment", lambda: train_yolo(context, "segment", configs["segment"]))
    pidnet = run_stage(context, "train_pidnet", lambda: train_pidnet(context, configs["pidnet"]))
    return {"detect": detect.value, "segment": segment.value, "pidnet": pidnet.value}


def export(context: RunContext) -> dict[str, Path]:
    """导出三个任务的当前 best.pt 并校验 ONNX."""
    detect_weight = _value_from_stage(context, "train_detect", "best")
    segment_weight = _value_from_stage(context, "train_segment", "best")
    pidnet_weight = _value_from_stage(context, "train_pidnet", "best")
    pidnet_config = _value_from_stage(context, "register_data", "pidnet")
    detect = run_stage(context, "export_detect", lambda: export_yolo_detect(context, detect_weight))
    segment = run_stage(context, "export_segment", lambda: export_yolo_segment(context, segment_weight))
    pidnet = run_stage(context, "export_pidnet", lambda: export_pidnet(context, pidnet_weight, pidnet_config))
    return {"detect": detect.value, "segment": segment.value, "pidnet": pidnet.value}


def prelabel(context: RunContext) -> StageOutcome:
    """使用当前 YOLO 分割和 PIDNet best.pt 预标注缺少 JSON 的图片."""
    segment_weight = _value_from_stage(context, "train_segment", "best")
    pidnet_weight = _value_from_stage(context, "train_pidnet", "best")
    pidnet_config = _value_from_stage(context, "register_data", "pidnet")
    return run_stage(
        context,
        "prelabel",
        lambda: generate_prelabels(context, segment_weight, pidnet_weight, pidnet_config),
    )


def test(context: RunContext) -> StageOutcome:
    """使用三个已导出的 ONNX 对累计验证集执行抽样测试."""
    detect_onnx = _value_from_stage(context, "export_detect", "onnx")
    segment_onnx = _value_from_stage(context, "export_segment", "onnx")
    pidnet_onnx = _value_from_stage(context, "export_pidnet", "onnx")
    return run_stage(
        context,
        "test",
        lambda: run_model_tests(context, detect_onnx, segment_onnx, pidnet_onnx),
    )


def report(context: RunContext) -> StageOutcome:
    """幂等更新唯一中心 Excel;不触发训练或导出."""
    return run_stage(context, "report", lambda: write_report(context))


def show_status(context: RunContext) -> None:
    """打印当前运行的全部阶段状态和产物."""
    payload = {
        name: {
            "status": stage.status,
            "attempts": stage.attempts,
            "message": stage.message,
            "artifacts": stage.artifacts,
        }
        for name, stage in context.state.stages().items()
    }
    context.logger.info("PIPELINE_STATUS | %s", json.dumps(payload, ensure_ascii=False))


def execute(context: RunContext) -> None:
    """执行一个命令并保持各阶段可单独重跑."""
    command = context.command
    if command == "prepare":
        prepare(context)
    elif command == "train":
        train(context)
    elif command == "export":
        export(context)
    elif command == "test":
        test(context)
    elif command == "prelabel":
        prelabel(context)
    elif command == "report":
        report(context)
    elif command == "status":
        show_status(context)
    elif command == "all":
        if context.config.enable_data_update:
            prepared = prepare(context)
            configs = prepared["registered"]["configs"]
        else:
            run_stage(
                context,
                "prepare_data",
                lambda: StageOutcome(
                    status=StageStatus.SKIPPED,
                    message="PIPELINE_ENABLE_DATA_UPDATE=false,本次all复用已有训练数据",
                ),
            )
            configs = ensure_registered(context)["configs"]
        detect = run_stage(
            context,
            "train_detect",
            lambda: train_yolo(context, "detect", configs["detect"]),
        )
        run_stage(
            context,
            "export_detect",
            lambda: export_yolo_detect(context, detect.value),
        )
        segment = run_stage(
            context,
            "train_segment",
            lambda: train_yolo(context, "segment", configs["segment"]),
        )
        run_stage(
            context,
            "export_segment",
            lambda: export_yolo_segment(context, segment.value),
        )
        pidnet = run_stage(
            context,
            "train_pidnet",
            lambda: train_pidnet(context, configs["pidnet"]),
        )
        run_stage(
            context,
            "export_pidnet",
            lambda: export_pidnet(context, pidnet.value, configs["pidnet"]),
        )
        test(context)
        if context.config.should_run_prelabel(command):
            prelabel(context)
        else:
            run_stage(
                context,
                "prelabel",
                lambda: StageOutcome(
                    status=StageStatus.SKIPPED,
                    message="PIPELINE_ENABLE_PRELABEL=false,本次all跳过预标注",
                ),
            )
        report(context)


def main() -> int:
    """创建上下文、执行预检和命令并写入唯一最终状态."""
    args = parse_args()
    context: RunContext | None = None
    try:
        config = PipelineConfig.from_environment()
        if args.force_train:
            config = replace(config, force_train=True)
        context = RunContext.create(config, args.command)
        context.logger.info("CONFIG | %s", json.dumps(config.snapshot(), ensure_ascii=False, sort_keys=True))
        run_preflight(config, args.command)
        context.logger.info("PREFLIGHT_OK | command=%s", args.command)
        execute(context)
        context.finish("success", "所有请求阶段已完成")
        return 0
    except ExcelPendingError as exc:
        if context is not None:
            context.logger.error("EXCEL_PENDING | %s", exc)
            context.finish("partial", "模型产物安全,中心Excel待补写;请关闭Excel后执行report")
        else:
            print(f"ERROR:{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if context is not None:
            context.logger.warning("PIPELINE_INTERRUPTED | 收到用户中断,下次同名运行将检查断点")
            context.finish("failed", "用户中断,可使用同一日期脚本恢复")
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
