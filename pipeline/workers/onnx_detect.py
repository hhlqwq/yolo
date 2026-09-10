#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测头 ONNX 模型推理内部实现

输入一个图片目录,自动检测同级目录下是否有 YOLO 格式的 labels/ 目录:
  - 有 labels/ → 生成竖向三视图 (原图/GT/预测) + 评估指标
  - 无 labels/ → 仅生成预测结果图 (无 GT、无指标)

用法:
    python tools/inference.py --onnx-model /path/to/detect.onnx
    python tools/inference.py --onnx-model /path/to/detect.onnx --img-dir /path/to/images
    python tools/inference.py --onnx-model /path/to/detect.onnx --img-dir /path/to/images --output-dir runs/my_out
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from pipeline.core.io_utils import remove_tree_safely


# ==============================================================================
# 配置
# ==============================================================================

@dataclass
class Config:
    """集中管理所有配置参数."""
    # 模型 & 数据
    onnx_model_path: str = "runs/detect/yolov11s_p2_detect_3cls/v5/weights/best_raw_p2_detect.onnx"
    dataset_yaml_path: str = "ultralytics/cfg/datasets/clean_v2_seg_zicai0721.yaml"
    output_dir: str = ""  # 空字符串表示自动生成,在 main() 中处理
    img_dir: Optional[str] = None
    manifest: Optional[str] = None

    # 推理参数
    imgsz: list[int] = field(default_factory=lambda: [640, 640])
    conf_threshold: float = 0.2
    iou_threshold: float = 0.7
    max_det: int = 300
    mask_threshold: float = 0.5
    mask_alpha: float = 0.45

    # 检测头参数
    nc: int = 1
    nm: int = 32
    reg_max: int = 16
    strides: list = field(default_factory=lambda: [4, 8, 16, 32])

    # ONNX 输出名称
    output_names: list = field(default_factory=lambda: [
        "box_p2", "score_p2", "box_p3", "score_p3",
        "box_p4", "score_p4", "box_p5", "score_p5",
    ])

    # 类别颜色 (BGR)
    class_colors: list = field(default_factory=lambda: [
        (0, 180, 0),    # 0: paper  绿色
        (0, 100, 255),  # 1: liquid 橙色
        (255, 80, 80),  # 2: metal  蓝色调
    ])

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        cfg = cls()
        if args.onnx_model:
            cfg.onnx_model_path = args.onnx_model
        if args.yaml:
            cfg.dataset_yaml_path = args.yaml
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.img_dir:
            cfg.img_dir = args.img_dir
        if args.manifest:
            cfg.manifest = args.manifest
        if args.imgsz:
            cfg.imgsz = args.imgsz
        if args.conf is not None:
            cfg.conf_threshold = args.conf
        if args.iou is not None:
            cfg.iou_threshold = args.iou
        if args.max_det:
            cfg.max_det = args.max_det
        return cfg


# ==============================================================================
# 数据加载
# ==============================================================================

def collect_image_label_pairs(img_dirs: list[str]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """从多个图片目录收集 (图片路径, label路径) 对,自动匹配同级 labels/ 目录.
    Returns:
        pairs: [(图片路径, label路径), ...]
        dir_counts: {目录路径: 该目录贡献的图片数}
    """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG")
    pairs = []
    seen = set()
    dir_counts = {}
    for img_dir in img_dirs:
        img_dir = Path(img_dir)
        if not img_dir.is_dir():
            print(f"[Warn] 目录不存在,跳过: {img_dir}")
            continue
        cnt = 0
        labels_dir = img_dir.parent / "labels"
        for ext in extensions:
            for img_path in sorted(img_dir.glob(ext)):
                base_name = img_path.stem
                if base_name in seen:
                    continue
                seen.add(base_name)
                label_path = labels_dir / f"{base_name}.txt"
                pairs.append((str(img_path), str(label_path) if label_path.exists() else ""))
                cnt += 1
        dir_counts[str(img_dir)] = cnt
    return pairs, dir_counts


def load_manifest_pairs(manifest_path: str) -> tuple[list[tuple[str, str, str, str]], dict[str, int]]:
    """从 pipeline 测试清单加载检测图片和 GT 标签."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    pairs = [
        (
            str(item["image"]),
            str(item["detect_label"]),
            str(item.get("id", Path(item["image"]).stem)),
            str(item.get("group", "")),
        )
        for item in samples
    ]
    group_counts: dict[str, int] = {}
    for _, _, _, group in pairs:
        key = group or "pipeline_manifest"
        group_counts[key] = group_counts.get(key, 0) + 1
    return pairs, group_counts


def clear_pipeline_visualizations(output_dir: str) -> None:
    """清理 pipeline 检测测试上次生成的三联图."""
    visualization_dir = Path(output_dir) / "images"
    if not visualization_dir.is_dir():
        return
    retained_metadata = remove_tree_safely(visualization_dir)
    if retained_metadata:
        print(
            "[Warning] 可视化目录保留被占用的 Thumbs.db: "
            f"files={len(retained_metadata)}, path={visualization_dir}"
        )


def resolve_img_dirs(cfg: Config) -> list[str]:
    """解析最终使用的图片目录列表: --img-dir > yaml val 路径."""
    if cfg.img_dir:
        return [cfg.img_dir]

    # 从 yaml 读取 val 路径
    if not os.path.exists(cfg.dataset_yaml_path):
        print(f"[Error] 未指定 --img-dir 且 yaml 不存在: {cfg.dataset_yaml_path}")
        sys.exit(1)

    with open(cfg.dataset_yaml_path, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    val_paths = ds.get("val", [])
    if not isinstance(val_paths, list):
        val_paths = [val_paths]
    if val_paths:
        return val_paths
    print("[Error] yaml 中未找到 val 路径,请使用 --img-dir 指定")
    sys.exit(1)


def find_labels_dir(img_dir: str) -> Optional[str]:
    """在图片目录的同级目录下查找 labels/ 目录."""
    parent = Path(img_dir).parent
    labels_dir = parent / "labels"
    if labels_dir.is_dir():
        return str(labels_dir)
    return None


def _class_id_from_name(name: str, class_names: dict[int, str]) -> int | None:
    """将 LabelMe 类别名称映射为模型类别编号。"""
    normalized = name.strip().casefold()
    if normalized.isdigit():
        class_id = int(normalized)
        return class_id if class_id in class_names else None
    return next(
        (
            class_id
            for class_id, class_name in class_names.items()
            if str(class_name).strip().casefold() == normalized
        ),
        None,
    )


def _load_labelme_gt(
    label_path: str,
    img_w: int,
    img_h: int,
    class_names: dict[int, str],
) -> dict:
    """读取 LabelMe JSON，并将矩形或多边形转换为检测框。"""
    data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    boxes, classes, masks = [], [], []
    for shape in data.get("shapes", []):
        class_id = _class_id_from_name(str(shape.get("label", "")), class_names)
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if class_id is None or points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, img_w - 1)
        points[:, 1] = np.clip(points[:, 1], 0, img_h - 1)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        boxes.append([minimum[0], minimum[1], maximum[0], maximum[1]])
        classes.append(class_id)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if points.shape[0] >= 3:
            cv2.fillPoly(mask, [points.round().astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    if not boxes:
        return _empty_gt()
    return {
        "boxes": np.asarray(boxes, dtype=np.float32),
        "classes": np.asarray(classes, dtype=np.int64),
        "masks": np.stack(masks),
    }


def load_gt_label(
    label_path: str,
    img_w: int,
    img_h: int,
    class_names: dict[int, str] | None = None,
) -> dict:
    """加载 YOLO TXT 或 LabelMe JSON 标注，并转换为像素坐标。"""
    boxes, classes, masks = [], [], []
    if not os.path.exists(label_path):
        return _empty_gt()
    if Path(label_path).suffix.casefold() == ".json":
        return _load_labelme_gt(label_path, img_w, img_h, class_names or {})
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            if len(parts) == 5:
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
                masks.append(np.zeros((img_h, img_w), dtype=bool))
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                polygon = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
                polygon *= np.array([img_w, img_h], dtype=np.float32)
                boxes.append([polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()])
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [polygon.round().astype(np.int32)], 1)
                masks.append(mask.astype(bool))
            else:
                continue
            classes.append(cls_id)
    if boxes:
        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "classes": np.array(classes, dtype=np.int64),
            "masks": np.stack(masks),
        }
    return _empty_gt()


def _empty_gt() -> dict:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
        "masks": np.zeros((0, 0, 0), dtype=bool),
    }


# ==============================================================================
# 预处理
# ==============================================================================

def preprocess(img: np.ndarray, imgsz: list[int] | tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple, tuple]:
    """letterbox + BGR→RGB + normalize + BCHW."""
    h0, w0 = img.shape[:2]
    target_h, target_w = imgsz
    r = min(target_h / h0, target_w / w0)
    new_w, new_h = round(w0 * r), round(h0 * r)
    dw, dh = target_w - new_w, target_h - new_h
    pad_left, pad_top = dw // 2, dh // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, pad_top, dh - pad_top, pad_left, dw - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    padded = padded[..., ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    padded = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    return padded[None], (r, r), (pad_left, pad_top)


# ==============================================================================
# 后处理
# ==============================================================================

def dfl_numpy(box: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """DFL 解码."""
    B, _, spatial_dim = box.shape
    box = box.reshape(B, 4, reg_max, spatial_dim)
    box = np.exp(box - box.max(axis=2, keepdims=True))
    box /= box.sum(axis=2, keepdims=True)
    weights = np.arange(reg_max, dtype=np.float32).reshape(1, 1, -1, 1)
    return (box * weights).sum(axis=2)


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_det: int,
) -> np.ndarray:
    """Class-aware NMS."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    boxes_sorted = boxes[order]
    classes_sorted = classes[order]
    keep = []

    while len(order) > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        x1 = np.maximum(boxes_sorted[0, 0], boxes_sorted[1:, 0])
        y1 = np.maximum(boxes_sorted[0, 1], boxes_sorted[1:, 1])
        x2 = np.minimum(boxes_sorted[0, 2], boxes_sorted[1:, 2])
        y2 = np.minimum(boxes_sorted[0, 3], boxes_sorted[1:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        a1 = (boxes_sorted[0, 2] - boxes_sorted[0, 0]) * (boxes_sorted[0, 3] - boxes_sorted[0, 1])
        a2 = (boxes_sorted[1:, 2] - boxes_sorted[1:, 0]) * (boxes_sorted[1:, 3] - boxes_sorted[1:, 1])
        iou = inter / np.maximum(a1 + a2 - inter, 1e-16)

        mask = (iou < iou_threshold) | (classes_sorted[1:] != classes_sorted[0])
        order = order[1:][mask]
        boxes_sorted = boxes_sorted[1:][mask]
        classes_sorted = classes_sorted[1:][mask]

    return np.array(keep, dtype=np.int64)


def postprocess(
    outputs: dict[str, np.ndarray],
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.8,
    max_det: int = 300,
    nc: int = 3,
    nm: int = 32,
    reg_max: int = 16,
    strides: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """后处理: ONNX raw 输出 → NMS 后的 (boxes, scores, classes, mask coefficients)."""
    if strides is None:
        strides = [4, 8, 16, 32]

    all_boxes, all_scores, all_classes = [], [], []

    for stride in strides:
        p_idx = stride.bit_length() - 1
        box_raw = outputs[f"box_p{p_idx}"]
        score_raw = outputs[f"score_p{p_idx}"]
        B, _, H, W = box_raw.shape
        num_anchors = H * W

        decoded = dfl_numpy(box_raw.reshape(B, 4 * reg_max, num_anchors), reg_max)

        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        anchor_grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5
        anchor_grid = anchor_grid[np.newaxis, ...]

        lt, rb = np.split(decoded, 2, axis=1)
        x1y1 = anchor_grid - lt.transpose(0, 2, 1)
        x2y2 = anchor_grid + rb.transpose(0, 2, 1)
        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=-1) * stride

        scores = 1 / (1 + np.exp(-np.clip(score_raw, -50, 50)))
        scores = scores.reshape(B, nc, num_anchors).transpose(0, 2, 1)

        boxes_xyxy = boxes_xyxy[0]
        scores = scores[0]

        max_scores = scores.max(axis=1)
        keep = max_scores > conf_threshold
        if keep.any():
            all_boxes.append(boxes_xyxy[keep])
            all_scores.append(max_scores[keep])
            all_classes.append(scores[keep].argmax(axis=1))

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, nm), dtype=np.float32),
        )

    boxes_all = np.concatenate(all_boxes, axis=0)
    scores_all = np.concatenate(all_scores, axis=0)
    classes_all = np.concatenate(all_classes, axis=0)
    keep_idx = nms_numpy(boxes_all, scores_all, classes_all, iou_threshold, max_det)
    return (
        boxes_all[keep_idx], scores_all[keep_idx], classes_all[keep_idx],
        np.zeros((len(keep_idx), nm), dtype=np.float32),
    )


def _empty_pred(mask_h: int = 0, mask_w: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0, mask_h, mask_w), dtype=bool),
    )


def scale_boxes(
    boxes: np.ndarray, ratio: tuple, pad: tuple, model_pad: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """letterbox 坐标 → 原图坐标."""
    rw, rh = ratio
    pad_left, pad_top = pad
    model_left, model_top = model_pad
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= model_left
    boxes[:, [1, 3]] -= model_top
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, [0, 2]] /= rw
    boxes[:, [1, 3]] /= rh
    return boxes


def process_masks(
    proto: np.ndarray,
    coefficients: np.ndarray,
    boxes: np.ndarray,
    input_shape: tuple[int, int],
    original_shape: tuple[int, int],
    ratio: tuple[float, float],
    pad: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Combine mask coefficients and prototypes, crop them to boxes, and undo letterbox padding."""
    if not len(coefficients):
        return np.zeros((0, *original_shape), dtype=bool)

    proto = proto[0] if proto.ndim == 4 else proto
    nm, mask_h, mask_w = proto.shape
    masks = coefficients @ proto.reshape(nm, -1)
    masks = 1.0 / (1.0 + np.exp(-np.clip(masks, -50, 50)))
    masks = masks.reshape(-1, mask_h, mask_w)

    scaled_boxes = boxes * np.array(
        [mask_w / input_shape[1], mask_h / input_shape[0], mask_w / input_shape[1], mask_h / input_shape[0]],
        dtype=np.float32,
    )
    rows = np.arange(mask_h, dtype=np.float32)[None, :, None]
    cols = np.arange(mask_w, dtype=np.float32)[None, None, :]
    x1, y1, x2, y2 = np.split(scaled_boxes[:, :, None], 4, axis=1)
    masks *= (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)

    pad_left, pad_top = pad
    resized_w = round(original_shape[1] * ratio[0])
    resized_h = round(original_shape[0] * ratio[1])
    output = []
    for mask in masks:
        mask = cv2.resize(mask, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        output.append(mask > threshold)
    return np.stack(output)


# ==============================================================================
# 可视化
# ==============================================================================

def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[dict] = None,
    colors: Optional[list] = None,
    masks: Optional[np.ndarray] = None,
    mask_alpha: float = 0.45,
    prefix: str = "",
    line_thickness: int = 1,
    font_scale: float = 0.4,
    font_thickness: int = 1,
) -> np.ndarray:
    """在图上绘制检测框.scores=None 时视为 GT 框."""
    if colors is None:
        colors = [(0, 180, 0), (0, 100, 255), (255, 80, 80)]
    if class_names is None:
        class_names = {}

    vis = img.copy()
    if masks is not None:
        overlay = vis.copy()
        for mask, cls_id in zip(masks, classes):
            overlay[mask] = colors[int(cls_id) % len(colors)]
        vis = cv2.addWeighted(overlay, mask_alpha, vis, 1 - mask_alpha, 0)
    for i, box in enumerate(boxes):
        cls_id = int(classes[i])
        color = colors[cls_id % len(colors)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)

        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        label = f"{prefix}{cls_name} {scores[i]:.2f}" if scores is not None else f"{prefix}{cls_name}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_y = y1 - th - 2 if y1 - th - 2 > 0 else y1 + th + 2
        cv2.rectangle(vis, (x1, label_y - 2), (x1 + tw + 2, label_y + th), color, -1)
        cv2.putText(
            vis, label, (x1 + 1, label_y + th - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA,
        )
    return vis


def make_vertical_three_view(
    original: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """为原图、GT 和预测图添加标题并竖向拼接。"""
    panels = []
    for title, image in (
        ("Original", original),
        ("Ground Truth", ground_truth),
        ("Prediction", prediction),
    ):
        header = np.full((32, image.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(
            header,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        panels.append(np.vstack((header, image)))
    return np.vstack(panels)


# ==============================================================================
# 评估指标 (仅在有 GT 标注时使用)
# ==============================================================================

def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """[N,4] x [M,4] → [N,M] IoU 矩阵."""
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    a2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / np.maximum(a1[:, None] + a2[None, :] - inter, 1e-16)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """从 PR 点计算 AP."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def _compute_tp_fp_for_iou(
    cls_preds: list[dict],
    cls_gts_list: list[dict],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """在指定 IoU 阈值下计算 TP/FP 数组."""
    num_pred = len(cls_preds)
    for gt in cls_gts_list:
        gt["matched"] = False

    gts_by_img = {}
    for gt_idx, gt in enumerate(cls_gts_list):
        gts_by_img.setdefault(gt["img_idx"], []).append((gt_idx, gt))

    tp = np.zeros(num_pred, dtype=bool)
    match_ious_out = np.zeros(num_pred, dtype=np.float32)

    for p_idx, pred in enumerate(cls_preds):
        img_idx = pred["img_idx"]
        if img_idx not in gts_by_img:
            continue
        img_gts = gts_by_img[img_idx]
        best_iou, best_gt_idx = 0.0, -1
        gt_boxes = np.array([gt["box"] for _, gt in img_gts])
        ious = box_iou(pred["box"][None, :], gt_boxes)[0]
        for g_idx, (orig_gt_idx, gt) in enumerate(img_gts):
            if not gt["matched"] and ious[g_idx] > best_iou:
                best_iou, best_gt_idx = ious[g_idx], orig_gt_idx
        if best_iou >= iou_threshold:
            tp[p_idx] = True
            match_ious_out[p_idx] = best_iou
            cls_gts_list[best_gt_idx]["matched"] = True

    return tp, match_ious_out, gts_by_img


def evaluate_predictions(
    all_preds: list[dict],
    all_gts: list[dict],
    class_names: dict[int, str],
    nc: int,
) -> dict:
    """评估预测结果,返回每类和总体的 P/R/AP50/mAP50-95."""
    all_class_preds = {c: [] for c in range(nc)}
    all_class_gts = {c: [] for c in range(nc)}

    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        for i in range(len(preds["scores"])):
            cls = int(preds["classes"][i])
            if cls not in all_class_preds:
                continue
            all_class_preds[cls].append({
                "img_idx": img_idx, "score": float(preds["scores"][i]), "box": preds["boxes"][i].copy(),
            })
        for j in range(len(gts["classes"])):
            cls = int(gts["classes"][j])
            if cls not in all_class_gts:
                continue
            all_class_gts[cls].append({
                "img_idx": img_idx, "box": gts["boxes"][j].copy(), "matched": False,
            })

    iou_thresholds = np.arange(0.50, 1.0, 0.05)
    eval_results_per_class = {}

    for cls in range(nc):
        cls_preds = all_class_preds[cls]
        cls_gts_list = all_class_gts[cls]
        cls_name = class_names.get(cls, str(cls))
        num_gt, num_pred = len(cls_gts_list), len(cls_preds)

        if num_gt == 0:
            eval_results_per_class[cls] = dict(
                name=cls_name, num_gt=0, num_pred=num_pred,
                tp=0, fp=num_pred, fn=0, precision=0.0, recall=0.0,
                ap50=0.0, map50_95=0.0,
            )
            continue

        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        ap_per_threshold = []
        for iou_t in iou_thresholds:
            gts_copy = [dict(gt) for gt in cls_gts_list]
            tp, _, _ = _compute_tp_fp_for_iou(cls_preds, gts_copy, float(iou_t))
            fp = ~tp
            tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
            recalls = tp_c / num_gt
            precisions = tp_c / np.maximum(tp_c + fp_c, 1e-16)
            ap_per_threshold.append(compute_ap(recalls, precisions))

        gts_final = [dict(gt) for gt in cls_gts_list]
        tp_final, _, _ = _compute_tp_fp_for_iou(cls_preds, gts_final, 0.5)
        true_positive = int(tp_final.sum())
        false_positive = num_pred - true_positive
        false_negative = num_gt - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)

        eval_results_per_class[cls] = dict(
            name=cls_name, num_gt=num_gt, num_pred=num_pred,
            tp=true_positive, fp=false_positive, fn=false_negative,
            precision=precision, recall=recall,
            ap50=ap_per_threshold[0], map50_95=float(np.mean(ap_per_threshold)),
        )

    total_gt = sum(r["num_gt"] for r in eval_results_per_class.values())
    total_pred = sum(r["num_pred"] for r in eval_results_per_class.values())
    true_positive = sum(r["tp"] for r in eval_results_per_class.values())
    false_positive = sum(r["fp"] for r in eval_results_per_class.values())
    false_negative = sum(r["fn"] for r in eval_results_per_class.values())
    valid = [item for item in eval_results_per_class.values() if item["num_gt"] > 0]
    overall = dict(
        total_gt=total_gt, total_pred=total_pred,
        tp=true_positive, fp=false_positive, fn=false_negative,
        precision=true_positive / max(1, true_positive + false_positive),
        recall=true_positive / max(1, true_positive + false_negative),
        ap50=np.mean([item["ap50"] for item in valid]).item() if valid else 0.0,
        map50_95=np.mean([item["map50_95"] for item in valid]).item() if valid else 0.0,
    )
    return {"per_class": eval_results_per_class, "overall": overall}


def format_eval_results(results: dict) -> str:
    """格式化评估结果表格."""
    lines = ["", "=" * 72, "  评估结果汇总 (Detection)", "=" * 72]
    header = f"{'类别':<8} {'GT':>6} {'Pred':>6} {'P':>8} {'R':>8} {'AP50':>8} {'mAP50-95':>10}"
    lines.extend([header, "-" * 72])
    for cls in sorted(results["per_class"].keys()):
        r = results["per_class"][cls]
        lines.append(
            f"{r['name']:<8} {r['num_gt']:>6} {r['num_pred']:>6} "
            f"{r['precision']:>8.3f} {r['recall']:>8.3f} "
            f"{r['ap50']:>8.4f} {r['map50_95']:>10.4f}"
        )
    lines.append("-" * 72)
    o = results["overall"]
    lines.append(
        f"{'Overall':<8} {o['total_gt']:>6} {o['total_pred']:>6} "
        f"{o['precision']:>8.3f} {o['recall']:>8.3f} "
        f"{o['ap50']:>8.4f} {o['map50_95']:>10.4f}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


# ==============================================================================
# ONNX 推理核心
# ==============================================================================

def load_onnx_session(model_path: str) -> ort.InferenceSession:
    """加载 ONNX 模型,自动选择最佳 provider."""
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available] or available
    print(f"[ONNX] 可用 providers: {available}")
    print(f"[ONNX] 使用 providers: {providers}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
    output_names = [output.name for output in session.get_outputs()]
    levels = sorted(
        int(match.group(1)) for name in output_names if (match := re.fullmatch(r"box_p(\d+)", name))
    )
    required = {name for level in levels for name in (f"box_p{level}", f"score_p{level}")}
    if levels not in ([2, 3, 4, 5], [3, 4, 5]) or not required.issubset(output_names):
        raise ValueError(f"仅支持 P2-P5 或 P3-P5 原始检测输出，实际输出为: {output_names}")
    print("[ONNX] 模型加载成功")
    return session


def configure_model(session: ort.InferenceSession, cfg: Config) -> None:
    """根据 ONNX 输出节点配置 P2/P3 层级和类别数。"""
    outputs = {output.name: output for output in session.get_outputs()}
    levels = sorted(int(name.removeprefix("box_p")) for name in outputs if name.startswith("box_p"))
    cfg.strides = [2**level for level in levels]
    cfg.output_names = [name for level in levels for name in (f"box_p{level}", f"score_p{level}")]
    score_shape = outputs[f"score_p{levels[0]}"].shape
    if isinstance(score_shape[1], int):
        cfg.nc = score_shape[1]
    print(f"[ONNX] 检测层级: {levels}; strides={cfg.strides}; nc={cfg.nc}")


def run_inference(
    session: ort.InferenceSession,
    img: np.ndarray,
    orig_h: int,
    orig_w: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单图 ONNX 推理 → 原图坐标下的 (boxes, scores, classes, masks)."""
    prep_img, ratio, pad = preprocess(img, cfg.imgsz)
    input_name = session.get_inputs()[0].name
    raw_outputs = session.run(None, {input_name: prep_img})
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, raw_outputs))

    boxes, scores, classes, coefficients = postprocess(
        outputs,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        max_det=cfg.max_det,
        nc=cfg.nc,
        nm=cfg.nm,
        reg_max=cfg.reg_max,
        strides=cfg.strides,
    )

    if len(boxes) == 0:
        return _empty_pred(orig_h, orig_w)

    target_h, target_w = cfg.imgsz
    model_pad = (((target_w + 31) // 32 * 32 - target_w) // 2, ((target_h + 31) // 32 * 32 - target_h) // 2)
    scaled_boxes = scale_boxes(boxes, ratio, pad, model_pad)
    scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]].clip(0, orig_w)
    scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]].clip(0, orig_h)
    valid = (scaled_boxes[:, 2] > scaled_boxes[:, 0]) & (scaled_boxes[:, 3] > scaled_boxes[:, 1])
    return (
        scaled_boxes[valid], scores[valid], classes[valid], np.zeros((int(valid.sum()), orig_h, orig_w), dtype=bool)
    )


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    # --- 解析参数 ---
    parser = argparse.ArgumentParser(
        description="检测 ONNX 推理脚本，自动兼容 P2-P5 和 P3-P5 原始输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
    python tools/inference.py --onnx-model /path/to/detect.onnx
    python tools/inference.py --onnx-model /path/to/detect.onnx --img-dir data/datasets/xxx/images
    """,
    )
    parser.add_argument("--img-dir", type=str, default=None, help="输入图片目录路径")
    parser.add_argument("--manifest", type=str, default=None, help="pipeline 生成的测试清单 JSON")
    parser.add_argument("--onnx-model", type=str, default=None, help="ONNX 模型路径 (覆盖默认)")
    parser.add_argument("--yaml", type=str, default=None, help="数据集 yaml 路径 (覆盖默认)")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录 (覆盖默认)")
    parser.add_argument(
        "--imgsz", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None, help="模型输入尺寸 [height width]"
    )
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值")
    parser.add_argument("--max-det", type=int, default=None, help="每张图最大实例数")
    args = parser.parse_args()
    cfg = Config.from_args(args)

    # --- 确定图片目录 ---
    img_dirs = [] if cfg.manifest else resolve_img_dirs(cfg)
    print(f"[Data] 图片目录: {img_dirs}")

    if cfg.manifest:
        image_label_pairs, dir_image_counts = load_manifest_pairs(cfg.manifest)
    else:
        image_label_pairs, dir_image_counts = collect_image_label_pairs(img_dirs)
    print(f"[Data] 共找到 {len(image_label_pairs)} 张图片")
    if img_dirs and len(img_dirs) > 1:
        for d, cnt in dir_image_counts.items():
            print(f"  - {d}: {cnt} 张")
    if not image_label_pairs:
        print("[Error] 未找到图片,退出")
        sys.exit(1)

    # --- 判断是否有 GT 标签 ---
    # yaml 指定时强制有 GT;--img-dir 手动指定时才检测
    if cfg.manifest or cfg.img_dir is None:
        # yaml 路径:一定存在标签
        has_gt = True
        print("[Data] 模式: yaml 指定 → 有 GT → 生成三视图 + 评估指标")
    else:
        # --img-dir 手动指定:检测 labels 目录
        labels_dir = find_labels_dir(cfg.img_dir)
        has_gt = labels_dir is not None
        if has_gt:
            print(f"[Data] 检测到 labels 目录: {labels_dir}")
            print(f"[Data] 模式: 有 GT → 生成三视图 + 评估指标")
        else:
            print(f"[Data] 未找到 labels 目录 (期望路径: {Path(cfg.img_dir).parent / 'labels'})")
            print(f"[Data] 模式: 无 GT → 仅保存预测结果图")

    # --- 加载 yaml 配置 (类别名、颜色等) ---
    class_names = {}
    nc = cfg.nc
    if os.path.exists(cfg.dataset_yaml_path):
        with open(cfg.dataset_yaml_path, "r", encoding="utf-8") as f:
            ds_cfg = yaml.safe_load(f)
        nc = ds_cfg.get("nc", cfg.nc)
        names_raw = ds_cfg.get("names", {})
        if isinstance(names_raw, dict):
            class_names = {int(k): v for k, v in names_raw.items()}
    cfg.nc = nc
    print(f"[Config] 类别数: {nc}, 类别名: {class_names}")
    print(f"[Config] ONNX 模型: {cfg.onnx_model_path}")

    # --- 输出目录 (自动生成或使用指定) ---
    if not cfg.output_dir:
        # 在 runs/inference/ 下创建新目录,名称基于数据集名 + 序号
        base = "runs/inference"
        # 数据集名:优先用 yaml 文件名,否则用第一个图片目录的父目录名
        if cfg.img_dir:
            ds_name = Path(cfg.img_dir).parent.name
        else:
            ds_name = Path(cfg.dataset_yaml_path).stem
        prefix = ds_name if ds_name else "infer"
        counter = 0
        while True:
            suffix = f"{counter:02d}" if counter > 0 else ""
            candidate = os.path.join(base, f"{prefix}{suffix}")
            if not os.path.exists(candidate):
                cfg.output_dir = candidate
                break
            counter += 1
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.manifest:
        clear_pipeline_visualizations(cfg.output_dir)
    print(f"[Output] 输出目录: {cfg.output_dir}")

    # --- 加载 ONNX ---
    session = load_onnx_session(cfg.onnx_model_path)
    configure_model(session, cfg)
    input_shape = session.get_inputs()[0].shape
    onnx_size = [int(input_shape[2]), int(input_shape[3])]
    if cfg.imgsz != onnx_size:
        print(f"[ONNX] 使用模型固定输入尺寸 {onnx_size}，覆盖参数 imgsz={cfg.imgsz}。")
        cfg.imgsz = onnx_size

    # --- 逐图推理 ---
    total_time, processed, total_pred = 0.0, 0, 0
    all_preds, all_gts_list = [], []
    grouped_predictions: dict[str, list[dict]] = {}
    grouped_ground_truth: dict[str, list[dict]] = {}

    for idx, pair in enumerate(image_label_pairs):
        img_path, label_path = pair[:2]
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [Skip] 无法读取: {img_path}")
            continue
        orig_h, orig_w = img.shape[:2]
        img_name = os.path.basename(img_path)
        base_name = str(pair[2]) if len(pair) > 2 else os.path.splitext(img_name)[0]
        group = str(pair[3]) if len(pair) > 3 else ""

        # GT (如果有)
        gt = _empty_gt()
        if has_gt and label_path:
            gt = load_gt_label(label_path, orig_w, orig_h, class_names)

        # 推理
        t0 = time.time()
        boxes, scores, classes, masks = run_inference(session, img, orig_h, orig_w, cfg)
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed
        processed += 1
        total_pred += len(boxes)

        # 可视化预测
        vis_pred = draw_boxes(img, boxes, classes, scores=scores,
                              class_names=class_names, colors=cfg.class_colors,
                              masks=masks, mask_alpha=cfg.mask_alpha)

        if has_gt:
            # 有 GT: 保存三视图 + 收集评估数据
            all_preds.append({"boxes": boxes, "scores": scores, "classes": classes})
            all_gts_list.append(gt)
            if group:
                grouped_predictions.setdefault(group, []).append(all_preds[-1])
                grouped_ground_truth.setdefault(group, []).append(gt)

            vis_gt = draw_boxes(img, gt["boxes"], gt["classes"], scores=None,
                                class_names=class_names, colors=cfg.class_colors, prefix="GT:",
                                masks=gt["masks"], mask_alpha=cfg.mask_alpha)
            concat = make_vertical_three_view(img, vis_gt, vis_pred)
            visualization_dir = os.path.join(cfg.output_dir, "images") if cfg.manifest else cfg.output_dir
            destination = Path(visualization_dir) / group / f"{base_name}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), concat)
        else:
            # 无 GT: 仅保存预测结果图
            cv2.imwrite(os.path.join(cfg.output_dir, base_name + ".png"), vis_pred)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_label_pairs):
            avg_ms = total_time / processed if processed > 0 else 0
            gt_num = len(gt["boxes"]) if has_gt else "N/A"
            print(f"  [{idx + 1}/{len(image_label_pairs)}] {img_name}  "
                  f"GT={gt_num} Pred={len(boxes)}  {elapsed:.0f}ms  avg={avg_ms:.0f}ms")

    avg_ms = total_time / processed if processed > 0 else 0
    print(f"\n[Summary] 处理图片数: {processed}")
    print(f"[Summary] 总检测实例数: {total_pred}")
    print(f"[Summary] 平均推理时间: {avg_ms:.1f} ms")
    print(f"[Summary] 输出目录: {cfg.output_dir}")
    print(f"[Summary] 推理参数: imgsz={cfg.imgsz}  conf={cfg.conf_threshold}  iou={cfg.iou_threshold}  max_det={cfg.max_det}")
    if dir_image_counts and len(dir_image_counts) > 1:
        print("[Summary] 各目录图片数量:")
        for d, cnt in dir_image_counts.items():
            print(f"  - {d}: {cnt} 张")

    # --- 评估指标 (仅在有 GT 时) ---
    if has_gt:
        print("\n正在计算评估指标...")
        eval_results = evaluate_predictions(all_preds, all_gts_list, class_names, nc)
        result_str = format_eval_results(eval_results)
        print(result_str)

        metrics_path = os.path.join(cfg.output_dir, "metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"测试图片总数: {len(image_label_pairs)}\n")
            f.write(f"测试图片目录: {len(dir_image_counts)} 个\n")
            for d, cnt in dir_image_counts.items():
                f.write(f"  {d}: {cnt} 张\n")
            f.write(f"推理参数: imgsz={cfg.imgsz}  conf_threshold={cfg.conf_threshold}  iou_threshold={cfg.iou_threshold}  max_det={cfg.max_det}\n")
            f.write("\n")
            f.write(result_str)
            f.write("\n\nPer-Class Details:\n")
            f.write("-" * 56 + "\n")
            for cls in sorted(eval_results["per_class"].keys()):
                r = eval_results["per_class"][cls]
                f.write(f"\n  Class '{r['name']}' (id={cls}):\n")
                f.write(f"    GT 数量:     {r['num_gt']}\n")
                f.write(f"    预测数量:    {r['num_pred']}\n")
                f.write(f"    TP/FP/FN:    {r['tp']}/{r['fp']}/{r['fn']}\n")
                f.write(f"    Precision:   {r['precision']:.3f}\n")
                f.write(f"    Recall:      {r['recall']:.3f}\n")
                f.write(f"    AP@0.50:     {r['ap50']:.6f}\n")
                f.write(f"    mAP@0.50:0.95: {r['map50_95']:.6f}\n")
            o = eval_results["overall"]
            f.write(f"\n  Overall (micro average):\n")
            f.write(f"    TP/FP/FN:    {o['tp']}/{o['fp']}/{o['fn']}\n")
            f.write(f"    Precision:   {o['precision']:.3f}\n")
            f.write(f"    Recall:      {o['recall']:.3f}\n")
            f.write(f"    mAP@0.50:    {o['ap50']:.6f}\n")
            f.write(f"    mAP@0.50:0.95: {o['map50_95']:.6f}\n")
        print(f"[Metrics] 评估结果已保存: {metrics_path}")
        metrics_json_path = os.path.join(cfg.output_dir, "metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"[Metrics] JSON评估结果已保存: {metrics_json_path}")
        if grouped_predictions:
            group_results = {
                group: evaluate_predictions(
                    predictions,
                    grouped_ground_truth[group],
                    class_names,
                    nc,
                )
                for group, predictions in sorted(grouped_predictions.items())
            }
            scene_metrics_path = os.path.join(cfg.output_dir, "scene_metrics.json")
            with open(scene_metrics_path, "w", encoding="utf-8") as f:
                json.dump(group_results, f, ensure_ascii=False, indent=2)
            print(f"[Metrics] 逐场景评估结果已保存: {scene_metrics_path}")


if __name__ == "__main__":
    main()
