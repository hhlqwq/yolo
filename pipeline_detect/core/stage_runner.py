"""纯检测 Pipeline 的阶段执行器。"""

from __future__ import annotations

import time
from typing import Any, Callable

from .state import StageStatus, StateStore


def run_stage(
    state: StateStore,
    logger: Any,
    name: str,
    function: Callable[[], tuple[str, dict[str, Any], dict[str, Any]]],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """执行阶段并记录开始、结束、产物、指标和异常。"""
    state.start(name)
    logger.info("STAGE_START | %s", name)
    started = time.monotonic()
    try:
        message, artifacts, metrics = function()
    except Exception as exc:
        state.finish(name, StageStatus.FAILED, str(exc))
        logger.exception("STAGE_FAILED | %s | %s", name, exc)
        raise
    state.finish(name, StageStatus.SUCCEEDED, message, artifacts, metrics)
    logger.info("STAGE_END | %s | elapsed=%.1fs | %s", name, time.monotonic() - started, message)
    return message, artifacts, metrics
