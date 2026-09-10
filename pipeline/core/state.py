"""可恢复流水线状态机."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import PipelineError


class StageStatus(StrEnum):
    """流水线阶段状态."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class StageRecord:
    """一个阶段的只读状态视图."""

    name: str
    status: StageStatus
    attempts: int
    started_at: str | None
    finished_at: str | None
    elapsed_seconds: float | None
    accumulated_seconds: float
    message: str
    artifacts: dict[str, Any]
    metrics: dict[str, Any]


class StateStore:
    """使用原子 JSON 文件保存可恢复状态."""

    VERSION = 1

    def __init__(self, path: Path, run_name: str, config: dict[str, Any]):
        """初始化状态存储并加载已有状态."""
        self.path = path
        self.run_name = run_name
        self._data = self._load_or_create(config)

    def _load_or_create(self, config: dict[str, Any]) -> dict[str, Any]:
        """加载已有状态,不存在时创建空状态."""
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PipelineError(f"流水线状态文件损坏:{self.path},原因:{exc}") from exc
            if payload.get("version") != self.VERSION or payload.get("run_name") != self.run_name:
                raise PipelineError(f"流水线状态版本或 run-name 不匹配:{self.path}")
            payload["config"] = config
            return payload
        now = datetime.now().isoformat(timespec="seconds")
        return {
            "version": self.VERSION,
            "run_name": self.run_name,
            "created_at": now,
            "updated_at": now,
            "config": config,
            "invocations": [],
            "stages": {},
        }

    def save(self) -> None:
        """通过临时文件和原子替换保存当前状态."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def begin_invocation(self, command: str) -> int:
        """记录一次命令启动并返回调用序号."""
        invocation = {
            "command": command,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "status": "running",
            "message": "",
        }
        self._data["invocations"].append(invocation)
        self.save()
        return len(self._data["invocations"]) - 1

    def finish_invocation(self, index: int, status: str, message: str = "") -> None:
        """记录一次命令的最终状态."""
        invocation = self._data["invocations"][index]
        invocation["finished_at"] = datetime.now().isoformat(timespec="seconds")
        invocation["status"] = status
        invocation["message"] = message
        self.save()

    def start_stage(self, name: str) -> None:
        """将阶段标记为运行中并增加尝试次数."""
        previous = self._data["stages"].get(name, {})
        self._data["stages"][name] = {
            "status": StageStatus.RUNNING,
            "attempts": int(previous.get("attempts", 0)) + 1,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "elapsed_seconds": None,
            "accumulated_seconds": float(previous.get("accumulated_seconds", 0.0)),
            "message": "",
            "artifacts": previous.get("artifacts", {}),
            "metrics": previous.get("metrics", {}),
        }
        self.save()

    def finish_stage(
        self,
        name: str,
        status: StageStatus,
        elapsed_seconds: float,
        message: str = "",
        artifacts: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """保存阶段完成、跳过或失败状态."""
        record = self._data["stages"].setdefault(name, {"attempts": 1})
        record.update(
            {
                "status": status,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": round(float(elapsed_seconds), 3),
                "message": message,
            }
        )
        if status != StageStatus.SKIPPED:
            record["accumulated_seconds"] = round(
                float(record.get("accumulated_seconds", 0.0)) + float(elapsed_seconds),
                3,
            )
        if artifacts is not None:
            record["artifacts"] = artifacts
        if metrics is not None:
            record["metrics"] = metrics
        self.save()

    def get_stage(self, name: str) -> StageRecord:
        """返回阶段状态;尚未执行时返回 pending."""
        record = self._data["stages"].get(name, {})
        return StageRecord(
            name=name,
            status=StageStatus(record.get("status", StageStatus.PENDING)),
            attempts=int(record.get("attempts", 0)),
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            elapsed_seconds=record.get("elapsed_seconds"),
            accumulated_seconds=float(record.get("accumulated_seconds", 0.0)),
            message=str(record.get("message", "")),
            artifacts=dict(record.get("artifacts", {})),
            metrics=dict(record.get("metrics", {})),
        )

    def stages(self) -> dict[str, StageRecord]:
        """返回全部阶段的只读状态."""
        return {name: self.get_stage(name) for name in self._data["stages"]}

    def update_config(self, config: dict[str, Any]) -> None:
        """更新本次运行的最新配置快照."""
        self._data["config"] = config
        self.save()

    def snapshot(self) -> dict[str, Any]:
        """返回状态的深拷贝,供报告和状态命令只读使用."""
        return json.loads(json.dumps(self._data, ensure_ascii=False))
