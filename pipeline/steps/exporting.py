"""三个模型的幂等 ONNX 导出与完整性校验."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.context import RunContext
from ..core.errors import PipelineError
from ..core.io_utils import atomic_write_text
from ..core.logging_utils import conda_python_command, run_subprocess
from ..core.stage_runner import StageOutcome
from .training import sha256_file


EXPORT_FORMAT_VERSION = 2


def _load_manifest(path: Path) -> dict[str, Any]:
    """读取导出清单;不存在或损坏时返回空字典."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _validate_onnx(
    context: RunContext,
    path: Path,
    expected_outputs: list[str] | None,
    expected_input: list[int] | None = None,
    require_pad: bool = False,
) -> None:
    """在对应环境中检查 ONNX 结构与输出名称."""
    code = f'''import onnx
path = {str(path)!r}
expected = {expected_outputs!r}
expected_input = {expected_input!r}
require_pad = {require_pad!r}
model = onnx.load(path)
onnx.checker.check_model(model)
actual = [item.name for item in model.graph.output]
if expected is not None and actual != expected:
    raise RuntimeError(f"ONNX outputs mismatch: expected={{expected}}, actual={{actual}}")
input_dims = [item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim]
if expected_input is not None and input_dims[-2:] != expected_input:
    raise RuntimeError(f"ONNX input mismatch: expected={{expected_input}}, actual={{input_dims[-2:]}}")
if require_pad and not any(node.op_type == "Pad" for node in model.graph.node):
    raise RuntimeError("ONNX 缺少非 32 倍数输入所需的 Pad 节点。")
print("PIPELINE_ONNX_VALIDATED", path, actual)
'''
    run_subprocess(
        conda_python_command(context.config.yolo_env, ["-c", code]),
        context.config.repo_root,
        f"ONNX完整性校验-{path.name}",
        context.logger,
        key_line_filter=lambda line: "PIPELINE_ONNX_VALIDATED" in line,
    )


def _can_skip_export(manifest_path: Path, weight: Path, output: Path, input_size: list[int]) -> bool:
    """判断现有 ONNX 是否来自同一个 best.pt."""
    payload = _load_manifest(manifest_path)
    return (
        output.is_file()
        and payload.get("format_version") == EXPORT_FORMAT_VERSION
        and payload.get("weight_sha256") == sha256_file(weight)
        and payload.get("input_size") == input_size
        and payload.get("output") == str(output.resolve())
        and payload.get("output_sha256") == sha256_file(output)
    )


def _save_manifest(
    manifest_path: Path, task: str, weight: Path, output: Path, input_size: list[int], padding: list[int]
) -> None:
    """保存一次成功导出的可追溯信息."""
    payload = {
        "format_version": EXPORT_FORMAT_VERSION,
        "task": task,
        "weight": str(weight.resolve()),
        "weight_sha256": sha256_file(weight),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "input_size": input_size,
        "padding": padding,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_text(manifest_path, json.dumps(payload, ensure_ascii=False, indent=2))


def export_yolo_detect(context: RunContext, weight: Path) -> StageOutcome:
    """按 tools/export.py 的 v3 原始 P2 输出格式导出检测模型."""
    weight = weight.expanduser().resolve()
    if not weight.is_file():
        raise PipelineError(f"YOLO检测权重不存在:{weight}")
    output_dir = context.work_dir / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "yolo11_p2_detect.onnx"
    manifest = output_dir / "yolo11_p2_detect.export.json"
    names = [
        "box_p2", "score_p2", "box_p3", "score_p3",
        "box_p4", "score_p4", "box_p5", "score_p5",
    ]
    if _can_skip_export(manifest, weight, output, context.config.requested_image_size):
        _validate_onnx(
            context, output, names, context.config.requested_image_size, any(context.config.image_padding)
        )
        return StageOutcome(message="ONNX已与当前best.pt一致", artifacts={"onnx": str(output)}, value=output)

    code = (
        "from tools.export import export_yolov11_detect_p2\n"
        f"path = export_yolov11_detect_p2({str(weight)!r}, {str(output)!r}, "
        f"imgsz={context.config.requested_image_size!r}, device='cpu', opset=11)\n"
        "print('PIPELINE_ONNX_EXPORTED', path)\n"
    )
    run_subprocess(
        conda_python_command(context.config.yolo_env, ["-c", code]),
        context.config.repo_root,
        "YOLO11 P2检测ONNX导出-v3",
        context.logger,
        key_line_filter=lambda line: "PIPELINE_ONNX_EXPORTED" in line,
    )
    _validate_onnx(context, output, names, context.config.requested_image_size, any(context.config.image_padding))
    _save_manifest(
        manifest, "detect", weight, output, context.config.requested_image_size, context.config.image_padding
    )
    return StageOutcome(
        message="YOLO检测ONNX导出完成",
        artifacts={"onnx": str(output), "manifest": str(manifest)},
        value=output,
    )


def export_yolo_segment(context: RunContext, weight: Path) -> StageOutcome:
    """调用仓库 v2_yolov11_seg_p2 导出实例分割模型."""
    weight = weight.expanduser().resolve()
    if not weight.is_file():
        raise PipelineError(f"YOLO分割权重不存在:{weight}")
    export_script = context.config.repo_root / "tools" / "export.py"
    if not export_script.is_file():
        raise PipelineError(f"YOLO导出脚本不存在:{export_script}")
    output_dir = context.work_dir / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "yolo11_p2_segment.onnx"
    manifest = output_dir / "yolo11_p2_segment.export.json"
    names = [
        name
        for level in range(2, 6)
        for name in (f"box_p{level}", f"score_p{level}", f"mask_coeff_p{level}")
    ] + ["proto"]
    if _can_skip_export(manifest, weight, output, context.config.requested_image_size):
        _validate_onnx(
            context, output, names, context.config.requested_image_size, any(context.config.image_padding)
        )
        return StageOutcome(message="ONNX已与当前best.pt一致", artifacts={"onnx": str(output)}, value=output)
    code = (
        "from tools.export import v2_yolov11_seg_p2\n"
        f"path = v2_yolov11_seg_p2({str(weight)!r}, {str(output)!r}, "
        f"imgsz={context.config.requested_image_size!r}, device='cpu', opset=11)\n"
        "print('PIPELINE_ONNX_EXPORTED', path)\n"
    )
    run_subprocess(
        conda_python_command(context.config.yolo_env, ["-c", code]),
        context.config.repo_root,
        "YOLO11 P2分割ONNX导出-v2",
        context.logger,
        key_line_filter=lambda line: "PIPELINE_ONNX_EXPORTED" in line,
    )
    _validate_onnx(context, output, names, context.config.requested_image_size, any(context.config.image_padding))
    _save_manifest(
        manifest, "segment", weight, output, context.config.requested_image_size, context.config.image_padding
    )
    return StageOutcome(
        message="YOLO分割ONNX导出完成",
        artifacts={"onnx": str(output), "manifest": str(manifest)},
        value=output,
    )


def export_pidnet(context: RunContext, weight: Path, config_path: Path) -> StageOutcome:
    """在 PIDNet 仓库和 pid 环境中导出语义分割 ONNX."""
    weight = weight.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    export_script = context.config.pidnet_root / "tools" / "export_onnx.py"
    for title, path in (("PIDNet权重", weight), ("PIDNet配置", config_path), ("PIDNet导出脚本", export_script)):
        if not path.is_file():
            raise PipelineError(f"{title}不存在:{path}")
    output_dir = context.work_dir / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pidnet_semantic_segment.onnx"
    manifest = output_dir / "pidnet_semantic_segment.export.json"
    if _can_skip_export(manifest, weight, output, context.config.requested_image_size):
        _validate_onnx(
            context, output, ["output"], context.config.requested_image_size, any(context.config.image_padding)
        )
        return StageOutcome(message="ONNX已与当前best.pt一致", artifacts={"onnx": str(output)}, value=output)
    source = weight.with_suffix(".onnx")
    run_subprocess(
        conda_python_command(
            context.config.pidnet_env,
            [
                str(export_script), "--cfg", str(config_path), "--weight", str(weight), "--no-verify",
                "--imgsz", *map(str, context.config.requested_image_size),
            ],
        ),
        context.config.pidnet_root,
        "PIDNet语义分割ONNX导出",
        context.logger,
        key_line_filter=lambda line: "ONNX" in line,
    )
    if not source.is_file():
        raise PipelineError(f"PIDNet导出完成但未生成ONNX:{source}")
    temporary = output.with_suffix(".onnx.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(output)
    _validate_onnx(
        context, output, ["output"], context.config.requested_image_size, any(context.config.image_padding)
    )
    _save_manifest(
        manifest, "pidnet", weight, output, context.config.requested_image_size, context.config.image_padding
    )
    return StageOutcome(
        message="PIDNet ONNX导出完成",
        artifacts={"onnx": str(output), "manifest": str(manifest)},
        value=output,
    )
