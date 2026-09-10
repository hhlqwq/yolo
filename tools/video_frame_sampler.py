#!/usr/bin/env python3
"""
用于数据集标注的视频抽帧工具.

使用示例:
  # 输入路径、输出目录;相似度阈值不填写时默认为 0.995
  python tools/video_frame_sampler.py /data/videos /data/frames

  # 第三个参数可临时指定相似度阈值
  python tools/video_frame_sampler.py /data/videos /data/frames 0.998

安装依赖:
  pip install opencv-python numpy tqdm

默认输出结构:
  输出根目录/
  ├── 视频1/
  │   ├── 视频1_000000.jpg
  │   └── 视频1_000030.jpg
  └── 视频2/
      └── 视频2_000000.jpg
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import platform
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - 在 main 中输出清晰的依赖错误
    cv2 = None
    np = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 为可选依赖
    tqdm = None


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".m4v",
    ".webm",
    ".mpeg",
    ".mpg",
}

# 固定配置:需要调整时直接修改这里,不再通过命令行传入.
FIXED_CONFIG = {
    "similarity_scope": "video",  # 相似度比较范围:"video" 表示每个视频单独去重,"global" 表示所有视频共同去重.
    "similarity_method": "hist",  # 相似度算法:"hist" 使用颜色直方图,"gray" 使用缩小后的灰度图.
    "recursive": True,  # 是否递归查找输入目录的子目录:True 递归查找,False 只查找当前目录.
    "flat": False,  # 输出目录结构:False 为一个视频一个子目录,True 为全部图片放在同一目录.
    "prefix": None,  # 图片文件名前缀:None 使用视频名称,也可以改为字符串,例如 "camera1".
    "name_template": "{prefix}_{frame:06d}",  # 图片命名模板;frame 是原视频帧序号,06d 表示用六位数字补零.
    "image_ext": "jpg",  # 输出图片格式:可设置为 "jpg"、"png" 或 "webp".
    "jpg_quality": 95,  # JPG 图片质量:范围 1～100,数值越高图片越清晰、文件越大.
    "png_compression": 3,  # PNG 压缩级别:范围 0～9,数值越高文件越小、保存速度越慢.
    "resize_long_side": None,  # 输出缩放:None 保持原分辨率;填写整数时将最长边缩放到该像素数.
    "start_sec": 0.0,  # 开始处理时间:单位为秒,0.0 表示从视频开头开始.
    "end_sec": None,  # 结束处理时间:单位为秒,None 表示一直处理到视频结尾.
    "max_frames": None,  # 每个视频最多保存的图片数量:None 表示不限制,填写整数表示达到数量后停止.
    "manifest": None,  # 抽帧记录 CSV:None 表示不生成;需要时设置为 Path("frames.csv").
    "overwrite": True,  # 同名图片已存在时是否覆盖:True 覆盖,False 跳过已有图片.
    "dry_run": False,  # 是否只测试而不保存图片:True 不写入图片,False 正常保存图片.
    "verbose": False,  # 是否显示详细调试日志:True 显示,False 只显示普通处理信息.
}

# 默认相似度阈值:范围 0～1,越高越容易保存图片;命令行未提供第三个参数时使用此值.
DEFAULT_SIMILARITY_THRESHOLD = 0.98

# 模糊检测尺寸:计算清晰度前将大图最长边缩小到此像素数,使不同分辨率的视频使用统一标准.
BLUR_CHECK_LONG_SIDE = 640
# 模糊过滤阈值:清晰度低于此值才丢弃;数值越高过滤越严格,当前 8.0 只过滤严重失焦图片.
BLUR_THRESHOLD = 20.0


@dataclass(frozen=True)
class FrameRecord:
    """一张已保存视频帧的元数据."""

    video_path: Path
    image_path: Path
    frame_index: int
    time_ms: float
    blur_score: Optional[float]
    max_similarity_to_saved: Optional[float]
    matched_image_path: Optional[Path]


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 帮助信息中的固定标题替换为中文."""

    def format_usage(self) -> str:
        """返回中文标题的用法文本."""
        return super().format_usage().replace("usage:", "用法:", 1)

    def format_help(self) -> str:
        """返回中文标题的帮助文本."""
        return (
            super()
            .format_help()
            .replace("usage:", "用法:", 1)
            .replace("positional arguments:", "位置参数:", 1)
            .replace("options:", "可选参数:", 1)
        )


def parse_args() -> argparse.Namespace:
    """只解析输入、输出和相似度阈值三个常用参数."""
    parser = ChineseArgumentParser(description="从单个视频或视频目录中抽取图片,用于数据集标注.", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出.")
    parser.add_argument(
        "input",
        type=Path,
        default="/data/users/hailong.he/datasets/opc_clean/zicai260729/mp4/",
        help="视频文件路径,或包含视频的目录路径.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        default="/data/users/hailong.he/datasets/opc_clean/zicai260729/img/",
        help="图片输出根目录,每个视频会建立一个独立子目录.",
    )
    parser.add_argument(
        "similarity_threshold",
        nargs="?",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"相似度阈值,范围 0～1,越高保留越多(默认 {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    parser.set_defaults(**FIXED_CONFIG)
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    """清理文件名中的非法或不安全字符."""
    name = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE)
    return name.strip("._") or "video"


WINDOWS_DRIVE_PATH_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")


def is_wsl() -> bool:
    """判断当前环境是否为 WSL."""
    if os.name == "nt":
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    return "microsoft" in platform.release().lower()


def normalize_cli_path(path: Path) -> Path:
    """
    统一处理 Windows、WSL 和 Linux 路径.

    Windows 下保持 D:/... 不变;WSL 下自动转换为 /mnt/d/....
    原生 Linux 仅在对应盘符挂载点存在时转换,避免把远程服务器路径误判为本地磁盘.
    """
    path_text = str(path)
    match = WINDOWS_DRIVE_PATH_RE.match(path_text)
    if os.name == "nt" or match is None:
        return path.expanduser()

    mount_root = Path(os.environ.get("VIDEO_FRAME_WINDOWS_MOUNT_ROOT", "/mnt")).expanduser()
    drive = match.group("drive").lower()
    drive_mount = mount_root / drive
    rest = match.group("rest").replace("\\", "/").lstrip("/")
    mapped_path = drive_mount / rest

    if is_wsl() or drive_mount.exists():
        logging.info("自动转换 Windows 路径:%s -> %s", path_text, mapped_path)
        return mapped_path
    return path.expanduser()


def normalize_argument_paths(args: argparse.Namespace) -> None:
    """一次性规范化所有命令行路径,避免只转换输入而遗漏输出路径."""
    args.input = normalize_cli_path(args.input)
    args.output_dir = normalize_cli_path(args.output_dir)
    if args.manifest is not None:
        args.manifest = normalize_cli_path(args.manifest)
    # 已存在的输入转为规范绝对路径;输出即使尚不存在也可以安全解析为绝对路径.
    if args.input.exists():
        args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()


def build_missing_input_hint(input_path: Path) -> str:
    """为无法访问的 Windows 风格输入路径生成中文提示."""
    path_text = str(input_path)
    if os.name != "nt" and WINDOWS_DRIVE_PATH_RE.match(path_text):
        drive = path_text[0].lower()
        rest = path_text[2:].replace("\\", "/").lstrip("/")
        return (
            f"\n检测到非 Windows 系统中的 Windows 风格路径:{path_text}"
            f"\n未找到可访问的磁盘挂载点:/mnt/{drive}"
            f"\nWSL 通常会将该路径映射为:/mnt/{drive}/{rest}"
            "\n如果当前是远程 Linux 服务器,请先挂载或上传文件;服务器无法直接访问本机磁盘."
        )
    return ""


def get_windows_short_path(path: Path) -> str | None:
    """获取 Windows 8.3 短路径,用于兼容不支持 Unicode 的 OpenCV 后端.

    Args:
        path: 已存在的绝对文件路径.

    Returns:
        Windows 短路径;非 Windows、系统未启用短路径或转换失败时返回 None.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_short_path.restype = ctypes.c_uint
        required_length = get_short_path(str(path), None, 0)
        if required_length == 0:
            return None
        buffer = ctypes.create_unicode_buffer(required_length)
        written_length = get_short_path(str(path), buffer, required_length)
        return buffer.value if written_length > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def open_video_capture(video_path: Path) -> tuple[Any | None, str | None]:
    """使用 Unicode 路径及 Windows 短路径候选打开视频.

    Args:
        video_path: 视频绝对路径.

    Returns:
        已打开的 OpenCV VideoCapture 和实际使用的路径;全部失败时返回两个 None.
    """
    candidates = [str(video_path)]
    if os.name == "nt":
        candidates.append(video_path.as_posix())
        short_path = get_windows_short_path(video_path)
        if short_path:
            candidates.append(short_path)

    attempted: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold() if os.name == "nt" else candidate
        if key in attempted:
            continue
        attempted.add(key)
        capture = cv2.VideoCapture(candidate)
        if capture.isOpened():
            if candidate != str(video_path):
                logging.info("OpenCV 使用兼容路径打开中文视频:%s", candidate)
            return capture, candidate
        capture.release()
    return None, None


def discover_videos(input_path: Path, recursive: bool) -> list[Path]:
    """发现单个视频或目录中的全部受支持视频."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"输入文件不是受支持的视频格式:{input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在:{input_path}{build_missing_input_hint(input_path)}")

    candidates = input_path.rglob("*") if recursive else input_path.glob("*")
    videos = sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        scan_mode = "(包含所有子目录)" if recursive else "(仅输入目录第一层)"
        supported_extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"未找到受支持的视频{scan_mode}:{input_path}\n"
            f"支持的扩展名:{supported_extensions}"
        )
    return videos


def build_unique_video_names(videos: list[Path]) -> dict[Path, str]:
    """生成唯一视频名称,防止同名视频输出到统一目录后互相覆盖."""
    base_names = [sanitize_name(video.stem) for video in videos]
    duplicate_names = {name for name, count in Counter(base_names).items() if count > 1}
    result = {}
    for video, base_name in zip(videos, base_names):
        if base_name in duplicate_names:
            path_hash = hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:8]
            base_name = f"{base_name}_{path_hash}"
        result[video] = base_name
    return result


def resize_long_side(frame: np.ndarray, long_side: Optional[int]) -> np.ndarray:
    """按指定最长边等比例缩放图片."""
    if not long_side:
        return frame
    height, width = frame.shape[:2]
    current_long_side = max(height, width)
    if current_long_side == long_side:
        return frame
    scale = long_side / current_long_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def calculate_blur_score(frame: np.ndarray) -> float:
    """计算图片清晰度分数,分数越低表示图片越模糊.

    为降低不同视频分辨率对阈值的影响,会先将过大的图片最长边缩小到 640 像素,再计算灰度图拉普拉斯方差.
    """
    height, width = frame.shape[:2]
    current_long_side = max(height, width)
    if current_long_side > BLUR_CHECK_LONG_SIDE:
        scale = BLUR_CHECK_LONG_SIDE / current_long_side
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_signature(frame: np.ndarray, method: str) -> np.ndarray:
    """计算用于相似度比较的图片特征."""
    if method == "hist":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    vector = small.astype(np.float32).flatten()
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else vector


def signature_similarity(sig_a: np.ndarray, sig_b: np.ndarray, method: str) -> float:
    """计算两组图片特征的相似度."""
    if method == "hist":
        score = cv2.compareHist(sig_a, sig_b, cv2.HISTCMP_CORREL)
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))

    score = float(np.dot(sig_a, sig_b))
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def build_output_path(
    output_root: Path,
    video_name: str,
    prefix: Optional[str],
    template: str,
    image_ext: str,
    frame_index: int,
    time_ms: float,
    saved_count: int,
    flat: bool,
) -> Path:
    """构造输出图片路径,默认在输出根目录下为每个视频创建独立子目录."""
    safe_prefix = f"{sanitize_name(prefix)}_{video_name}" if prefix else video_name
    target_dir = output_root if flat else output_root / video_name
    filename_stem = template.format(
        video=video_name,
        prefix=safe_prefix,
        frame=frame_index,
        time_ms=int(round(time_ms)),
        time_sec=f"{time_ms / 1000.0:.3f}",
        count=saved_count,
    )
    filename_stem = sanitize_name(filename_stem)
    return target_dir / f"{filename_stem}.{image_ext}"


def write_image(path: Path, frame: np.ndarray, image_ext: str, jpg_quality: int, png_compression: int) -> None:
    """按指定格式和质量写入图片,兼容 Windows 中文路径."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params: list[int] = []
    if image_ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpg_quality, 1, 100))]
    elif image_ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(np.clip(png_compression, 0, 9))]
    ok, encoded = cv2.imencode(f".{image_ext}", frame, params)
    if not ok:
        raise RuntimeError(f"图片编码失败:{path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise RuntimeError(f"图片写入失败:{path},原因:{exc}") from exc


def max_similarity_to_saved(
    sig: np.ndarray,
    saved_signatures: list[tuple[np.ndarray, Path]],
    method: str,
) -> tuple[Optional[float], Optional[Path]]:
    """查找当前帧与已保存图片之间的最大相似度."""
    if not saved_signatures:
        return None, None

    best_score = -1.0
    best_path: Optional[Path] = None
    for saved_sig, image_path in saved_signatures:
        score = signature_similarity(sig, saved_sig, method)
        if score > best_score:
            best_score = score
            best_path = image_path
    return best_score, best_path


def process_video(
    video_path: Path,
    video_name: str,
    args: argparse.Namespace,
    global_signatures: list[tuple[np.ndarray, Path]],
) -> list[FrameRecord]:
    """处理单个视频并返回已保存帧的记录."""
    cap, opened_path = open_video_capture(video_path)
    if cap is None:
        logging.warning(
            "无法打开视频:%s;文件可能损坏,或当前 OpenCV 视频后端不支持该中文路径且系统未提供可用短路径",
            video_path,
        )
        return []
    logging.debug("视频实际打开路径:%s", opened_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000.0)

    records: list[FrameRecord] = []
    video_signatures: list[tuple[np.ndarray, Path]] = []
    saved_count = 0
    blurry_skip_count = 0

    progress = None
    if tqdm is not None and total_frames > 0:
        progress = tqdm(total=total_frames, desc=video_path.name, unit="帧")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if fps > 0 and (not time_ms or time_ms < 0):
                time_ms = frame_index / fps * 1000.0

            if progress is not None:
                progress.n = min(frame_index + 1, total_frames)
                progress.refresh()

            if args.end_sec is not None and time_ms > args.end_sec * 1000.0:
                break

            blur_score = calculate_blur_score(frame)
            if blur_score < BLUR_THRESHOLD:
                blurry_skip_count += 1
                logging.debug(
                    "跳过严重模糊帧:%s,第 %d 帧,清晰度 %.2f < %.2f",
                    video_path.name,
                    frame_index,
                    blur_score,
                    BLUR_THRESHOLD,
                )
                continue

            similarity: Optional[float] = None
            matched_image_path: Optional[Path] = None
            sig = frame_signature(frame, args.similarity_method)
            compare_pool = global_signatures if args.similarity_scope == "global" else video_signatures
            similarity, matched_image_path = max_similarity_to_saved(sig, compare_pool, args.similarity_method)
            if similarity is not None and similarity >= args.similarity_threshold:
                continue

            output_path = build_output_path(
                args.output_dir,
                video_name,
                args.prefix,
                args.name_template,
                args.image_ext,
                frame_index,
                time_ms,
                saved_count,
                args.flat,
            )
            if output_path.exists() and not args.overwrite:
                logging.info("跳过已存在的图片:%s", output_path)
                continue

            output_frame = resize_long_side(frame, args.resize_long_side)
            if not args.dry_run:
                write_image(output_path, output_frame, args.image_ext, args.jpg_quality, args.png_compression)

            video_signatures.append((sig, output_path))
            global_signatures.append((sig, output_path))

            records.append(
                FrameRecord(
                    video_path=video_path,
                    image_path=output_path,
                    frame_index=frame_index,
                    time_ms=time_ms,
                    blur_score=blur_score,
                    max_similarity_to_saved=similarity,
                    matched_image_path=matched_image_path,
                )
            )
            saved_count += 1

            if args.max_frames is not None and saved_count >= args.max_frames:
                break
    finally:
        if progress is not None:
            progress.close()
        cap.release()

    logging.info("已从 %s 保存 %d 张图片,因模糊跳过 %d 帧", video_path, len(records), blurry_skip_count)
    return records


def write_manifest(records: Iterable[FrameRecord], manifest_path: Path) -> None:
    """将已保存帧的元数据写入 CSV 文件."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "video_path",
                "image_path",
                "frame_index",
                "time_ms",
                "blur_score",
                "max_similarity_to_saved",
                "matched_image_path",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "video_path": str(record.video_path),
                    "image_path": str(record.image_path),
                    "frame_index": record.frame_index,
                    "time_ms": f"{record.time_ms:.3f}",
                    "blur_score": "" if record.blur_score is None else f"{record.blur_score:.3f}",
                    "max_similarity_to_saved": (
                        "" if record.max_similarity_to_saved is None else f"{record.max_similarity_to_saved:.6f}"
                    ),
                    "matched_image_path": "" if record.matched_image_path is None else str(record.matched_image_path),
                }
            )


def main() -> int:
    """执行视频发现、抽帧、去重及清单写入流程."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    normalize_argument_paths(args)

    if cv2 is None or np is None:
        raise SystemExit("缺少依赖,请执行:pip install opencv-python numpy tqdm")

    if not 0.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("--similarity-threshold 必须在 0.0～1.0 之间")
    if args.start_sec < 0:
        raise ValueError("--start-sec 不能为负数")
    if args.end_sec is not None and args.end_sec <= args.start_sec:
        raise ValueError("--end-sec 必须大于 --start-sec")

    videos = discover_videos(args.input, args.recursive)
    video_names = build_unique_video_names(videos)
    logging.info("找到 %d 个视频,输出根目录:%s", len(videos), args.output_dir)

    all_records: list[FrameRecord] = []
    global_signatures: list[tuple[np.ndarray, Path]] = []
    for video_path in videos:
        all_records.extend(process_video(video_path, video_names[video_path], args, global_signatures))

    if args.manifest:
        if not args.dry_run:
            write_manifest(all_records, args.manifest)
        logging.info("抽帧清单:%s", args.manifest)

    logging.info("处理完成,共保存 %d 张图片", len(all_records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
