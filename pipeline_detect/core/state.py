"""可恢复的纯检测流水线阶段状态机。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class StageStatus(StrEnum):
    """流水线阶段状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"


class StateStore:
    """以原子 JSON 保存阶段、耗时、产物和调用历史。"""

    def __init__(self, path: Path, run_name: str, config: dict[str, Any]) -> None:
        """加载已有状态或创建新状态。"""
        self.path = path
        self.run_name = run_name
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data.get("run_name") != run_name:
                raise RuntimeError(f"状态文件 run-name 不匹配: {path}")
            self.data["config"] = config
        else:
            self.data = {"version": 1, "run_name": run_name, "config": config, "invocations": [], "stages": {}}

    def save(self) -> None:
        """原子写入最新状态。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def start(self, name: str) -> None:
        """将阶段标记为运行中并保存。"""
        previous = self.data["stages"].get(name, {})
        self.data["stages"][name] = {"status": StageStatus.RUNNING, "attempts": int(previous.get("attempts", 0)) + 1, "started_at": datetime.now().isoformat(timespec="seconds"), "artifacts": previous.get("artifacts", {}), "metrics": previous.get("metrics", {})}
        self.save()

    def finish(self, name: str, status: StageStatus, message: str = "", artifacts: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None) -> None:
        """记录阶段最终状态、产物和指标。"""
        record = self.data["stages"].setdefault(name, {})
        record.update({"status": status, "finished_at": datetime.now().isoformat(timespec="seconds"), "message": message})
        if artifacts is not None: record["artifacts"] = artifacts
        if metrics is not None: record["metrics"] = metrics
        self.save()
