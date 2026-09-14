import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F

from ultralytics import YOLO


def _normalize_imgsz(imgsz: int | Sequence[int]) -> tuple[int, int]:
    """将输入尺寸统一为 (height, width)。"""
    if isinstance(imgsz, int):
        return imgsz, imgsz
    if len(imgsz) != 2 or min(imgsz) <= 0:
        raise ValueError(f"imgsz 必须是正整数或 [height, width]，当前值为 {imgsz!r}。")
    return int(imgsz[0]), int(imgsz[1])


def _stride_padding(height: int, width: int, stride: int = 32) -> tuple[int, int, int, int]:
    """计算居中补齐到 stride 倍数所需的 (left, right, top, bottom)。"""
    padded_height = ((height + stride - 1) // stride) * stride
    padded_width = ((width + stride - 1) // stride) * stride
    left = (padded_width - width) // 2
    top = (padded_height - height) // 2
    return left, padded_width - width - left, top, padded_height - height - top


class _InputPadWrapper(torch.nn.Module):
    """在 ONNX 图首部执行固定补边，然后调用 YOLO 网络。"""

    def __init__(self, model: torch.nn.Module, padding: tuple[int, int, int, int]):
        """保存网络和补边参数。"""
        super().__init__()
        self.model = model
        self.padding = padding

    def forward(self, images: torch.Tensor):
        """将输入补齐到 stride 对齐尺寸并执行前向。"""
        if any(self.padding):
            images = F.pad(images, self.padding, value=114.0 / 255.0)
        return self.model(images)


def v1_yolov11_seg():
    pt_path = "runs/segment/paper_zicai260721/v1_260722/weights/best.pt"
    model = YOLO(pt_path)
    det_model = model.model  # DetectionModel
    det_model.eval()
    head = det_model.model[-1]  # Segment

    # 替换 forward:每层独立输出,不 concat、不 DFL、不 sigmoid、不 reshape
    def raw_forward(x):
        out = []
        for i in range(head.nl):
            out.append(head.cv2[i](x[i]))  # [B, 4*reg_max, H, W]
            out.append(head.cv3[i](x[i]))  # [B, nc, H, W]
            out.append(head.cv4[i](x[i]))  # [B, nm, H, W]
        out.append(head.proto(x[0]))
        return tuple(out)

    head.forward = raw_forward

    dummy = torch.randn(1, 3, 640, 640)

    # 验证输出形状
    with torch.no_grad():
        outputs = det_model(dummy)
    names = [
        "box_p3", "score_p3", "mask_coeff_p3",
        "box_p4", "score_p4", "mask_coeff_p4",
        "box_p5", "score_p5", "mask_coeff_p5",
        "proto",
    ]
    for n, o in zip(names, outputs):
        print(f"{n}: {list(o.shape)}")

    # 导出 ONNX(dynamo 会把权重存到外部 .onnx.data,之后合并为单文件)
    out_dir = os.path.dirname(pt_path)
    tmp_path = os.path.join(out_dir, "_tmp_raw.onnx")
    onnx_path = os.path.join(out_dir, "best_raw.onnx")

    torch.onnx.export(
        det_model,
        dummy,
        tmp_path,
        opset_version=11,
        input_names=["images"],
        output_names=names,
        dynamo=False,  # 使用旧版 TorchScript 导出器,避免强制 opset 18
    )

    # 合并外部数据为单个 ONNX 文件
    onnx_model = onnx.load(tmp_path)
    onnx_model.ir_version = 6  # 地平线 X5 工具链只支持 IR version ≤ 9 (opset 11 对应 ir_version 6)
    onnx.save(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_model)

    # 清理临时文件
    os.remove(tmp_path)
    tmp_data = tmp_path + ".data"
    if os.path.exists(tmp_data):
        os.remove(tmp_data)

    print(f"\n✓ ONNX exported to: {onnx_path}")
    print("✓ ONNX model validation passed!")
    print("\nONNX output shapes:")
    for out in onnx_model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")


def v2_yolov11_seg_p2(
    pt_path: str = "runs/segment/p2_paper_zicai260721/v1_260721/weights/best.pt",
    onnx_path: str | None = None,
    imgsz: int | Sequence[int] = 640,
    device: str = "cpu",
    opset: int = 11,
) -> str:
    """Export a YOLO11 P2 segmentation model as board-friendly raw ONNX outputs.

    DFL decoding, sigmoid, mask assembly, and NMS are intentionally left to the board application. The output order is
    box/score/mask coefficient for P2 through P5, followed by the mask prototype.
    """
    model = YOLO(pt_path)
    det_model = model.model.to(device).eval()
    head = det_model.model[-1]
    strides = [int(x) for x in head.stride.tolist()]
    if head.__class__.__name__ != "Segment" or strides != [4, 8, 16, 32]:
        raise ValueError(f"Expected a four-level P2 Segment head with strides [4, 8, 16, 32], got {strides}")

    levels = [stride.bit_length() - 1 for stride in strides]
    names = [
        name
        for level in levels
        for name in (f"box_p{level}", f"score_p{level}", f"mask_coeff_p{level}")
    ] + ["proto"]
    original_forward = head.forward

    def raw_forward(x):
        outputs = []
        for i in range(head.nl):
            outputs.extend((head.cv2[i](x[i]), head.cv3[i](x[i]), head.cv4[i](x[i])))
        outputs.append(head.proto(x[0]))
        return tuple(outputs)

    height, width = _normalize_imgsz(imgsz)
    padding = _stride_padding(height, width, max(strides))
    export_model = _InputPadWrapper(det_model, padding)
    dummy = torch.zeros(1, 3, height, width, device=device)
    output_path = Path(onnx_path) if onnx_path else Path(pt_path).with_name(f"{Path(pt_path).stem}_raw_p2.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"_{output_path.stem}.tmp.onnx")

    try:
        try:
            head.forward = raw_forward
            with torch.no_grad():
                outputs = export_model(dummy)
            if len(outputs) != len(names):
                raise RuntimeError(f"Expected {len(names)} raw outputs, got {len(outputs)}")
            for name, output in zip(names, outputs):
                print(f"{name}: {list(output.shape)}")

            torch.onnx.export(
                export_model,
                dummy,
                str(tmp_path),
                opset_version=opset,
                input_names=["images"],
                output_names=names,
                dynamo=False,
            )
        finally:
            head.forward = original_forward

        onnx_model = onnx.load(str(tmp_path))
        if opset == 11:
            onnx_model.ir_version = 6  # Required by the target board toolchain.
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(output_path))
        onnx.checker.check_model(str(output_path))
    finally:
        for temporary in (tmp_path, Path(f"{tmp_path}.data")):
            if temporary.exists():
                temporary.unlink()

    print(f"\nONNX exported to: {output_path}")
    print("ONNX model validation passed.")
    return str(output_path)


def export_detect_p2(
    pt_path: str,
    onnx_path: str | None = None,
    imgsz: int | Sequence[int] = 640,
    device: str = "cpu",
    opset: int = 11,
) -> str:
    """导出 P2 检测模型，并在 ONNX 输入端自动补齐非 32 倍数尺寸。"""
    model = YOLO(pt_path)
    det_model = model.model.to(device).eval()
    head = det_model.model[-1]
    strides = [int(value) for value in head.stride.tolist()]
    if head.__class__.__name__ != "Detect" or strides != [4, 8, 16, 32]:
        raise ValueError(f"期望 P2 Detect 和步长 [4, 8, 16, 32]，实际为 {head.__class__.__name__} {strides}。")

    levels = [stride.bit_length() - 1 for stride in strides]
    names = [name for level in levels for name in (f"box_p{level}", f"score_p{level}")]
    original_forward = head.forward

    def raw_forward(features):
        """返回各特征层未经解码的检测输出。"""
        outputs = []
        for index in range(head.nl):
            outputs.extend((head.cv2[index](features[index]), head.cv3[index](features[index])))
        return tuple(outputs)

    height, width = _normalize_imgsz(imgsz)
    padding = _stride_padding(height, width, max(strides))
    dummy = torch.zeros(1, 3, height, width, device=device)
    output_path = Path(onnx_path) if onnx_path else Path(pt_path).with_name(f"{Path(pt_path).stem}_raw_p2_detect.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"_{output_path.stem}.tmp.onnx")
    try:
        try:
            head.forward = raw_forward
            export_model = _InputPadWrapper(det_model, padding)
            with torch.no_grad():
                outputs = export_model(dummy)
            for name, output in zip(names, outputs):
                print(f"{name}: {list(output.shape)}")
            torch.onnx.export(
                export_model,
                dummy,
                str(temporary),
                opset_version=opset,
                input_names=["images"],
                output_names=names,
                dynamo=False,
            )
        finally:
            head.forward = original_forward
        onnx_model = onnx.load(str(temporary))
        if opset == 11:
            onnx_model.ir_version = 6
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(output_path))
        onnx.checker.check_model(str(output_path))
    finally:
        for path in (temporary, Path(f"{temporary}.data")):
            if path.exists():
                path.unlink()
    print(f"ONNX exported to: {output_path}; input={(height, width)}; padding={padding}")
    return str(output_path)


def v3_yolov11_detect_p2():
    """导出带P2检测头的YOLO11检测模型,去除DFL等后处理算子"""
    pt_path = "runs/detect/yolov11s_p2_detect_3cls/v5/weights/best.pt"
    model = YOLO(pt_path)
    det_model = model.model  # DetectionModel
    det_model.eval()
    head = det_model.model[-1]  # Detect (P2, P3, P4, P5)

    # 替换 forward:每层独立输出 box 和 score,不 concat、不 DFL、不 sigmoid、不 reshape
    def raw_forward(x):
        out = []
        for i in range(head.nl):
            out.append(head.cv2[i](x[i]))  # [B, 4*reg_max, H, W]
            out.append(head.cv3[i](x[i]))  # [B, nc, H, W]
        return tuple(out)

    head.forward = raw_forward

    dummy = torch.randn(1, 3, 640, 640)

    # 验证输出形状
    with torch.no_grad():
        outputs = det_model(dummy)
    # 4个检测层:P2/4, P3/8, P4/16, P5/32
    names = [
        "box_p2", "score_p2",
        "box_p3", "score_p3",
        "box_p4", "score_p4",
        "box_p5", "score_p5",
    ]
    for n, o in zip(names, outputs):
        print(f"{n}: {list(o.shape)}")

    # 导出 ONNX
    out_dir = os.path.dirname(pt_path)
    tmp_path = os.path.join(out_dir, "_tmp_raw_p2_detect.onnx")
    onnx_path = os.path.join(out_dir, "best_raw_p2_detect.onnx")

    torch.onnx.export(
        det_model,
        dummy,
        tmp_path,
        opset_version=11,
        input_names=["images"],
        output_names=names,
        dynamo=False,
    )

    # 合并外部数据为单个 ONNX 文件
    onnx_model = onnx.load(tmp_path)
    onnx_model.ir_version = 6  # 地平线 X5 工具链只支持 IR version ≤ 9 (opset 11 对应 ir_version 6)
    onnx.save(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_model)

    # 清理临时文件
    os.remove(tmp_path)
    tmp_data = tmp_path + ".data"
    if os.path.exists(tmp_data):
        os.remove(tmp_data)

    print(f"\n✓ ONNX exported to: {onnx_path}")
    print("✓ ONNX model validation passed!")
    print("\nONNX output shapes:")
    for out in onnx_model.graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出带原始 P2 输出的 YOLO ONNX。")
    parser.add_argument("--task", choices=("detect", "segment"), default="segment")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--output")
    parser.add_argument("--imgsz", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=[640, 640])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()
    function = export_detect_p2 if args.task == "detect" else v2_yolov11_seg_p2
    function(args.weight, args.output, args.imgsz, args.device, args.opset)
