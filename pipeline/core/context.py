"""流水线运行上下文."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import PipelineError
from .logging_utils import DisplayPathMapper, create_pipeline_logger
from .state import StateStore


@dataclass
class RunContext:
    """集中提供配置、目录、日志器和状态存储."""

    config: PipelineConfig
    logger: logging.Logger
    mapper: DisplayPathMapper
    state: StateStore
    command: str
    invocation_index: int
    started_at: datetime
    lock_handle: Any

    @staticmethod
    def _acquire_pipeline_lock(path: Path):
        """获取训练根目录的系统级排他锁;进程退出后由系统自动释放."""
        handle = path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                if path.stat().st_size == 0:
                    handle.write("0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise PipelineError(f"同一训练根目录已有pipeline正在执行,锁文件:{path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"host={socket.gethostname()},pid={os.getpid()},started={datetime.now().isoformat()}\n")
        handle.flush()
        return handle

    def _release_run_lock(self) -> None:
        """释放运行排他锁并关闭文件句柄."""
        if self.lock_handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.lock_handle.seek(0)
                msvcrt.locking(self.lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_handle.close()
            self.lock_handle = None

    @classmethod
    def create(cls, config: PipelineConfig, command: str) -> "RunContext":
        """校验运行根目录并创建一次命令上下文."""
        if not config.model_train_root.is_dir():
            raise PipelineError(f"模型训练根目录不存在:{config.model_train_root}")
        work_dir = config.work_dir
        work_dir.mkdir(exist_ok=True)
        if work_dir.resolve().parent != config.model_train_root.resolve():
            raise PipelineError(f"运行目录超出模型训练根目录:{work_dir}")
        config.registry_dir.mkdir(exist_ok=True)
        lock_handle = cls._acquire_pipeline_lock(config.registry_dir / "pipeline.lock")
        mapper = DisplayPathMapper(config.display_runtime_prefix, config.display_nas_prefix)
        logger = create_pipeline_logger(work_dir / "pipeline.log", mapper)
        state = StateStore(work_dir / "pipeline_state.json", config.run_name, config.snapshot())
        state.update_config(config.snapshot())
        invocation_index = state.begin_invocation(command)
        started_at = datetime.now()
        logger.info("=" * 80)
        logger.info("PIPELINE_START | command=%s | run_name=%s", command, config.run_name)
        logger.info("运行目录:%s", work_dir)
        return cls(config, logger, mapper, state, command, invocation_index, started_at, lock_handle)

    @property
    def work_dir(self) -> Path:
        """返回本次运行目录."""
        return self.config.work_dir

    @property
    def log_path(self) -> Path:
        """返回本次运行的唯一日志文件路径."""
        return self.work_dir / "pipeline.log"

    def finish(self, status: str, message: str = "") -> None:
        """结束本次命令并写入最终机器可读标记."""
        try:
            self.state.finish_invocation(self.invocation_index, status, message)
            self.logger.info("PIPELINE_FINAL_STATUS | %s | command=%s | %s", status.upper(), self.command, message)
        finally:
            self._release_run_lock()
