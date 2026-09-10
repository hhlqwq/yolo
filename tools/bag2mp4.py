#!/usr/bin/env python3
# av==12.3.0
# opencv-python==5.0.0.93
# pyorbbecsdk2==2.1.1
"""Convert Orbbec SDK .bag color streams to MP4 videos."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBPlaybackStatus,
    OBSensorType,
    Pipeline,
    PlaybackDevice,
)


class BagPrecheckError(RuntimeError):
    """表示 BAG 转换前检查失败。"""


class BagStallError(RuntimeError):
    """表示 BAG 回放在指定时间内没有进展。"""


def render_progress(percent: float, width: int = 30) -> str:
    """返回可在终端单行刷新的进度条文本。"""
    bounded_percent = min(max(percent, 0.0), 100.0)
    completed = round(width * bounded_percent / 100.0)
    return (
        f"[{'#' * completed}{'-' * (width - completed)}] "
        f"{bounded_percent:5.1f}%"
    )


def precheck_bag(
    input_path: Path, playback_rate: float, timeout_seconds: float
) -> None:
    """在转换前验证 BAG 可打开、包含彩色流且可读到首帧。"""
    playback = None
    pipeline = None
    stopped = threading.Event()
    deadline = time.monotonic() + timeout_seconds

    try:
        playback = PlaybackDevice(str(input_path))
        playback.set_playback_rate(playback_rate)
        playback.set_playback_status_change_callback(
            lambda status: stopped.set()
            if status == OBPlaybackStatus.STOPPED
            else None
        )
        sensor_list = playback.get_sensor_list()
        sensor_types = [
            sensor_list[i].get_type() for i in range(len(sensor_list))
        ]
        if OBSensorType.COLOR_SENSOR not in sensor_types:
            raise BagPrecheckError("BAG 中没有 COLOR_SENSOR 彩色流")
        if playback.get_duration() <= 0:
            raise BagPrecheckError("BAG 时长无效，可能缺少索引或文件已损坏")

        pipeline = Pipeline(playback)
        config = Config()
        config.enable_stream(OBSensorType.COLOR_SENSOR)
        pipeline.start(config)
        while time.monotonic() < deadline:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            frames = pipeline.wait_for_frames(min(1000, remaining_ms))
            if frames is not None and frames.get_color_frame() is not None:
                return
            if stopped.is_set():
                raise BagPrecheckError("回放在读到首个彩色帧前结束")
        raise BagPrecheckError(
            f"{timeout_seconds:g} 秒内未读到首个彩色帧，可能文件头、索引或首段数据异常"
        )
    except BagPrecheckError:
        raise
    except Exception as exc:
        raise BagPrecheckError(f"无法打开或读取 BAG: {exc}") from exc
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass


def color_frame_to_bgr(frame) -> np.ndarray:
    """Decode an Orbbec color frame into an OpenCV BGR image."""
    width = frame.get_width()
    height = frame.get_height()
    frame_format = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)

    if frame_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif frame_format == OBFormat.RGB:
        image = cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    elif frame_format == OBFormat.BGR:
        image = data.reshape(height, width, 3)
    elif frame_format in (OBFormat.YUYV, OBFormat.YUY2):
        image = cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2
        )
    elif frame_format == OBFormat.UYVY:
        image = cv2.cvtColor(
            data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY
        )
    elif frame_format == OBFormat.NV12:
        image = cv2.cvtColor(
            data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12
        )
    elif frame_format == OBFormat.NV21:
        image = cv2.cvtColor(
            data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21
        )
    elif frame_format == OBFormat.I420:
        image = cv2.cvtColor(
            data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420
        )
    elif frame_format == OBFormat.BGRA:
        image = cv2.cvtColor(
            data.reshape(height, width, 4), cv2.COLOR_BGRA2BGR
        )
    elif frame_format == OBFormat.RGBA:
        image = cv2.cvtColor(
            data.reshape(height, width, 4), cv2.COLOR_RGBA2BGR
        )
    else:
        raise ValueError(f"不支持的彩色帧格式: {frame_format}")

    if image is None:
        raise ValueError(f"彩色帧解码失败,格式: {frame_format}")
    return image


class AvVideoWriter:
    """Write quality-based H.264/H.265 MP4 video through PyAV."""

    CODECS = {"h264": "libx264", "h265": "libx265"}

    def __init__(
        self,
        output_path: Path,
        fps: float,
        size: tuple[int, int],
        codec: str,
        crf: int,
        preset: str,
    ) -> None:
        self.container = av.open(str(output_path), mode="w")
        self.stream = self.container.add_stream(
            self.CODECS[codec],
            rate=Fraction(fps).limit_denominator(1001),
            options={"crf": str(crf), "preset": preset},
        )
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"

    def write(self, image: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def release(self) -> None:
        if self.container is None:
            return
        container = self.container
        self.container = None
        try:
            for packet in self.stream.encode():
                container.mux(packet)
        finally:
            container.close()


def convert_bag(
    input_path: Path,
    output_path: Path,
    fps_override: float | None,
    playback_rate: float,
    overwrite: bool,
    codec: str,
    crf: int,
    preset: str,
    stall_timeout: float,
) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已存在: {output_path}(使用 --overwrite 覆盖)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.stem + ".partial.mp4")
    if temporary_output.exists():
        temporary_output.unlink()

    playback = None
    pipeline = None
    writer = None
    stopped = threading.Event()
    frames_written = 0

    try:
        playback = PlaybackDevice(str(input_path))
        playback.set_playback_rate(playback_rate)
        playback.set_playback_status_change_callback(
            lambda status: stopped.set()
            if status == OBPlaybackStatus.STOPPED
            else None
        )

        sensor_list = playback.get_sensor_list()
        sensor_types = [sensor_list[i].get_type() for i in range(len(sensor_list))]
        if OBSensorType.COLOR_SENSOR not in sensor_types:
            raise RuntimeError("BAG 中没有 COLOR_SENSOR 彩色流")

        pipeline = Pipeline(playback)
        config = Config()
        config.enable_stream(OBSensorType.COLOR_SENSOR)
        try:
            config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION
            )
        except (AttributeError, RuntimeError):
            pass

        duration_ms = playback.get_duration()
        pipeline.start(config)
        empty_reads_after_stop = 0
        last_progress_at = time.monotonic()
        last_position_ms = 0
        last_progress_print_at = 0.0

        while True:
            frames = pipeline.wait_for_frames(1000)
            current_time = time.monotonic()
            position_ms = min(playback.get_position(), duration_ms)
            if position_ms > last_position_ms:
                last_position_ms = position_ms
                last_progress_at = current_time
            if current_time - last_progress_at >= stall_timeout:
                raise BagStallError(
                    f"连续 {stall_timeout:g} 秒未读到新帧且回放位置未前进 "
                    f"(停在 {position_ms / 1000:.1f} 秒)"
                )
            if frames is None:
                if stopped.is_set():
                    empty_reads_after_stop += 1
                    if empty_reads_after_stop >= 2:
                        break
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            if writer is None:
                profile = color_frame.get_stream_profile().as_video_stream_profile()
                source_fps = float(profile.get_fps())
                fps = fps_override if fps_override is not None else source_fps
                size = (color_frame.get_width(), color_frame.get_height())
                writer = AvVideoWriter(
                    temporary_output, fps, size, codec, crf, preset
                )
                print(
                    f"  视频参数: {size[0]}x{size[1]}, {fps:g} FPS, "
                    f"{color_frame.get_format()}, 编码器 {codec.upper()}, "
                    f"CRF {crf}, preset {preset}"
                )

            writer.write(color_frame_to_bgr(color_frame))
            frames_written += 1
            last_progress_at = current_time
            if frames_written == 1 or current_time - last_progress_print_at >= 0.2:
                percent = 100.0 * position_ms / duration_ms if duration_ms else 0.0
                print(
                    f"\r  {render_progress(percent)} 已写入 {frames_written} 帧, "
                    f"回放 {position_ms / 1000:.1f}/{duration_ms / 1000:.1f} 秒",
                    end="",
                    flush=True,
                )
                last_progress_print_at = current_time

        if writer is None or frames_written == 0:
            raise RuntimeError("没有从 BAG 中读到可写入的彩色帧")

        writer.release()
        writer = None
        if output_path.exists():
            output_path.unlink()
        temporary_output.replace(output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(
            f"\r  完成: {frames_written} 帧 -> {output_path} "
            f"({size_mb:.1f} MB)"
        )
    except Exception:
        if writer is not None:
            try:
                writer.release()
            finally:
                writer = None
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    finally:
        if writer is not None:
            writer.release()
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        playback = None


def collect_jobs(
    input_dir: Path, output_dir: Path
) -> list[tuple[Path, Path]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"找不到输入 BAG 目录: {input_dir}")

    bag_files = sorted(input_dir.glob("*.bag"))
    if not bag_files:
        raise FileNotFoundError(f"输入目录中没有 .bag 文件: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"输出路径不是目录: {output_dir}")
    return [(bag, output_dir / bag.with_suffix(".mp4").name) for bag in bag_files]


def format_file_size(size_bytes: int) -> str:
    """将字节数转换为便于控制台阅读的文件大小。"""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("文件大小单位转换失败")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 Orbbec Viewer/SDK 录制的 BAG 彩色流转换为 MP4"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="D:\\tmp\\bag",
        type=Path,
        help="包含 .bag 文件的输入目录",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="D:\\tmp\\bag_out",
        type=Path,
        help="保存 .mp4 文件的输出目录(不存在时自动创建)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="强制指定输出 FPS(默认采用 BAG 彩色流标称 FPS)",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=0.5,
        help="SDK 回放速率(默认 0.5,给编码器留出时间以免丢帧)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 MP4",
    )
    parser.add_argument(
        "--codec",
        choices=("h264", "h265"),
        default="h264",
        help="视频编码格式(默认: h264)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="恒定质量值,越大文件越小、画质越低(默认: 23)",
    )
    parser.add_argument(
        "--preset",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ),
        default="medium",
        help="压缩预设,越慢压缩率越高(默认: medium)",
    )
    parser.add_argument(
        "--precheck-timeout",
        type=float,
        default=30.0,
        help="转换前读取首帧的最长等待秒数(默认: 30)",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=120.0,
        help="转换中无新帧且回放位置不变的最长秒数(默认: 120)",
    )
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if args.playback_rate <= 0:
        parser.error("--playback-rate 必须大于 0")
    if not 0 <= args.crf <= 51:
        parser.error("--crf 必须在 0 到 51 之间")
    if args.precheck_timeout <= 0:
        parser.error("--precheck-timeout 必须大于 0")
    if args.stall_timeout <= 0:
        parser.error("--stall-timeout 必须大于 0")
    return args


def main() -> int:
    args = parse_args()
    try:
        jobs = collect_jobs(args.input_dir, args.output_dir)
        failed_jobs: list[tuple[Path, str]] = []
        skipped_jobs = 0
        for index, (input_path, output_path) in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] {input_path} -> {output_path}")
            if output_path.exists() and not args.overwrite:
                print(
                    "  跳过:已检测到同名 MP4，视为此前已解析完成。"
                    f"文件大小 {format_file_size(output_path.stat().st_size)}"
                )
                print(f"  已有输出: {output_path}")
                skipped_jobs += 1
                continue
            print(
                f"  预检:尝试在 {args.precheck_timeout:g} 秒内读取首个彩色帧..."
            )
            try:
                precheck_bag(
                    input_path, args.playback_rate, args.precheck_timeout
                )
                print("  预检通过,开始转换.")
                convert_bag(
                    input_path,
                    output_path,
                    args.fps,
                    args.playback_rate,
                    args.overwrite,
                    args.codec,
                    args.crf,
                    args.preset,
                    args.stall_timeout,
                )
            except Exception as exc:
                print(f"\n  跳过: {exc}", file=sys.stderr)
                failed_jobs.append((input_path, str(exc)))
        if failed_jobs:
            print("\n以下 BAG 未转换:", file=sys.stderr)
            for failed_path, reason in failed_jobs:
                print(f"- {failed_path}: {reason}", file=sys.stderr)
            return 1
        if skipped_jobs:
            print(f"\n完成:共跳过 {skipped_jobs} 个已有 MP4 文件.")
        return 0
    except KeyboardInterrupt:
        print("\n已取消.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n转换失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
