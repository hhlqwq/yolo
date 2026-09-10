"""流水线阶段执行器."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from .context import RunContext
from .state import StageStatus


T = TypeVar("T")


@dataclass
class StageOutcome:
    """阶段函数返回的状态、产物、指标和运行时值."""

    status: StageStatus = StageStatus.SUCCEEDED
    message: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    value: Any = None


def run_stage(context: RunContext, name: str, function: Callable[[], StageOutcome]) -> StageOutcome:
    """执行一个阶段并保证开始、完成和失败状态都被持久化."""
    context.state.start_stage(name)
    context.logger.info("STAGE_START | %s", name)
    started = time.monotonic()
    try:
        outcome = function()
    except Exception as exc:
        elapsed = time.monotonic() - started
        context.state.finish_stage(
            name,
            StageStatus.FAILED,
            elapsed,
            message=str(exc),
        )
        context.logger.exception("STAGE_FAILED | %s | %s", name, exc)
        raise
    elapsed = time.monotonic() - started
    context.state.finish_stage(
        name,
        outcome.status,
        elapsed,
        message=outcome.message,
        artifacts=outcome.artifacts,
        metrics=outcome.metrics,
    )
    context.logger.info(
        "STAGE_END | %s | status=%s | elapsed=%.1fs | %s",
        name,
        outcome.status,
        elapsed,
        outcome.message,
    )
    return outcome

