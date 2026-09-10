"""单文件日志、路径展示和子进程输出过滤."""

from __future__ import annotations

import copy
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import PipelineError


ANSI_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


@dataclass(frozen=True)
class DisplayPathMapper:
    """将服务器 NAS 映射路径转换为 Windows UNC 展示路径."""

    runtime_prefix: str
    display_prefix: str

    def map_path(self, value: str | Path) -> str:
        """转换一个完整路径;非 NAS 路径保持原样."""
        text = str(value)
        runtime = self.runtime_prefix.replace("\\", "/").rstrip("/")
        normalized = text.replace("\\", "/")
        if normalized == runtime or normalized.startswith(f"{runtime}/"):
            suffix = normalized[len(runtime):].replace("/", "\\")
            return f"{self.display_prefix}{suffix}"
        return text

    def map_text(self, value: str) -> str:
        """转换日志、JSON 和异常文本中的 NAS 路径片段."""
        runtime = self.runtime_prefix.replace("\\", "/").rstrip("/")
        replaced = value.replace(self.runtime_prefix, self.display_prefix).replace(runtime, self.display_prefix)
        # UNC 路径中允许括号等目录字符,仅以日志文本分隔符作为边界.
        pattern = re.compile(re.escape(self.display_prefix) + r'[^"\n,;，；\|]*')
        return pattern.sub(lambda match: match.group(0).replace("/", "\\"), replaced)


class DisplayFormatter(logging.Formatter):
    """写日志前统一转换 NAS 展示路径."""

    def __init__(self, mapper: DisplayPathMapper):
        """初始化带路径转换器的日志格式器."""
        super().__init__(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.mapper = mapper

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志副本,避免修改其他处理器共享的记录."""
        cloned = copy.copy(record)
        if isinstance(record.args, dict):
            mapped_args = {
                key: self.mapper.map_path(value) if isinstance(value, (str, Path)) else value
                for key, value in record.args.items()
            }
            rendered = str(record.msg) % mapped_args
        elif record.args:
            mapped_args = tuple(
                self.mapper.map_path(value) if isinstance(value, (str, Path)) else value
                for value in record.args
            )
            rendered = str(record.msg) % mapped_args
        else:
            rendered = str(record.msg)
        cloned.msg = self.mapper.map_text(rendered)
        cloned.args = ()
        return super().format(cloned)


def create_pipeline_logger(log_path: Path, mapper: DisplayPathMapper) -> logging.Logger:
    """创建同时输出控制台和唯一 pipeline.log 的日志器."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clean_robot_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = DisplayFormatter(mapper)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def normalize_output_line(raw_line: str) -> str:
    """清除 ANSI 控制字符并返回最后一个有效刷新行."""
    cleaned = ANSI_PATTERN.sub("", raw_line).replace("\r", "\n")
    parts = [part.strip() for part in cleaned.splitlines() if part.strip()]
    return parts[-1] if parts else ""


def is_progress_line(line: str) -> bool:
    """判断是否属于 tqdm 或 batch 级进度输出."""
    if not line:
        return True
    if "Iter:[" in line or "Iter: [" in line:
        return True
    return any(symbol in line for symbol in ("%|", "━━━━━━━━", "████", "it/s", "s/it"))


def is_key_training_line(line: str) -> bool:
    """判断训练输出是否应进入汇总日志."""
    markers = (
        "YOLO Epoch Summary",
        "PIDNet Config",
        "PIDNet Train Epoch",
        "PIDNet Val Epoch",
        "PIDNet Class",
        "PIDNet New best.pt",
        "PIDNet Training Finished",
        "最佳模型结果:",
        "Best validation result:",
        "ONNX export success",
        "ONNX export failure",
    )
    return any(marker in line for marker in markers)


def find_conda_executable() -> str | None:
    """查找 Conda 可执行文件,兼容已激活环境和常见用户安装目录."""
    configured = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if configured:
        return configured
    candidates = (
        Path.home() / "miniconda3/bin/conda",
        Path.home() / "anaconda3/bin/conda",
        Path.home() / "miniconda3/Scripts/conda.exe",
        Path.home() / "anaconda3/Scripts/conda.exe",
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def conda_python_command(environment_name: str, arguments: list[str]) -> list[str]:
    """构造当前或指定 Conda 环境中的 Python 命令."""
    if os.environ.get("CONDA_DEFAULT_ENV", "") == environment_name:
        return [sys.executable, *arguments]
    conda = find_conda_executable()
    if not conda:
        raise PipelineError(f"找不到 conda,无法切换到环境:{environment_name}")
    command = [conda]
    if os.name == "nt" and Path(conda).suffix.lower() in {".bat", ".cmd"}:
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", conda]
    return [*command, "run", "--no-capture-output", "-n", environment_name, "python", *arguments]


def run_subprocess(
    command: list[str],
    cwd: Path,
    title: str,
    logger: logging.Logger,
    key_line_filter: Callable[[str], bool] = is_key_training_line,
) -> None:
    """实时显示原始输出,仅将关键行及失败堆栈写入唯一日志."""
    logger.info("STAGE_COMMAND_START | %s", title)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["PIPELINE_SINGLE_LOG"] = "1"
    recent_lines: deque[str] = deque(maxlen=160)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = normalize_output_line(raw_line)
            if line:
                recent_lines.append(line)
            if line and not is_progress_line(line) and key_line_filter(line):
                logger.info("[%s] %s", title, line)
            else:
                print(raw_line, end="", flush=True)
    except KeyboardInterrupt:
        logger.warning("SUBPROCESS_INTERRUPTED | %s | 正在通知当前子进程安全停止", title)
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=15)
        raise
    return_code = process.wait()
    if return_code != 0:
        useful_tail = [line for line in recent_lines if not is_progress_line(line)][-30:]
        logger.error("SUBPROCESS_FAILURE_BEGIN | title=%s | exit_code=%d", title, return_code)
        for line in useful_tail:
            logger.error("[%s] %s", title, line)
        logger.error("SUBPROCESS_FAILURE_END | title=%s", title)
        raise PipelineError(f"{title}失败,退出码:{return_code}")
    logger.info("STAGE_COMMAND_SUCCESS | %s", title)
