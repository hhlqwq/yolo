"""不依赖训练库的通用文件写入工具."""

from __future__ import annotations

import errno
import os
import shutil
import time
from pathlib import Path
from types import TracebackType
from typing import Callable


RetryCallback = Callable[[Path, int, int, str], None]


def atomic_write_text(path: Path, content: str) -> None:
    """使用同目录临时文件原子写入 UTF-8 文本."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _contains_only_windows_metadata(root: Path) -> bool:
    """检查残留目录是否只包含 Windows 自动生成的 Thumbs.db."""
    found_metadata = False
    try:
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if path.name.casefold() != "thumbs.db":
                return False
            found_metadata = True
    except OSError:
        return False
    return found_metadata


def remove_tree_safely(
    path: Path,
    retries: int = 5,
    retry_delay_seconds: float = 1.0,
    retry_callback: RetryCallback | None = None,
) -> tuple[Path, ...]:
    """安全删除 NAS/SMB 目录，并保留被占用的 Thumbs.db."""
    retained_metadata: set[Path] = set()

    def handle_error(
        function: Callable[..., object],
        failed_path: str,
        error_info: tuple[type[BaseException], BaseException, TracebackType],
    ) -> None:
        """忽略已消失的目录项和被占用的 Thumbs.db."""
        del function
        error = error_info[1]
        failed = Path(failed_path)
        if isinstance(error, FileNotFoundError):
            return
        if (
            isinstance(error, OSError)
            and error.errno == errno.EBUSY
            and failed.name.casefold() == "thumbs.db"
        ):
            retained_metadata.add(failed)
            return
        if (
            isinstance(error, OSError)
            and error.errno == errno.ENOTEMPTY
            and _contains_only_windows_metadata(failed)
        ):
            return
        raise error

    for attempt in range(1, retries + 1):
        if not path.exists():
            return tuple(sorted(retained_metadata))
        try:
            shutil.rmtree(path, onerror=handle_error)
            if path.exists() and not _contains_only_windows_metadata(path):
                raise OSError(errno.ENOTEMPTY, "目录仍包含非 Thumbs.db 文件", str(path))
            return tuple(sorted(retained_metadata))
        except FileNotFoundError:
            return tuple(sorted(retained_metadata))
        except OSError as exc:
            if exc.errno not in {errno.EBUSY, errno.ENOTEMPTY} or attempt >= retries:
                raise
            reason = "resource_busy" if exc.errno == errno.EBUSY else "directory_not_empty"
            if retry_callback is not None:
                retry_callback(path, attempt, retries, reason)
            time.sleep(retry_delay_seconds)
    return tuple(sorted(retained_metadata))
