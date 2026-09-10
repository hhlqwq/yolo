"""纯检测 Pipeline 的训练根目录排他锁。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def acquire(path: Path) -> Any:
    """获取跨进程文件锁，阻止同一训练根目录并发运行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            handle.write("0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"已有检测 Pipeline 正在运行，锁文件: {path}") from exc
    return handle


def release(handle: Any) -> None:
    """释放已获取的跨进程文件锁。"""
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
