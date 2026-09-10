#!/usr/bin/env python3
"""
板端ONNX模型推理与评估脚本

功能:
  1. 单图对比模式 (--single):用ONNX和PyTorch模型分别推理同一张图,对比结果正确性
  2. 批量评估模式 (--eval):对val集所有图片用ONNX推理,计算R/P/mAP/mask mAP等指标

用法:
    # 单图对比(先验证推理正确性)
    python tools/inference.py --onnx-model /path/to/segment.onnx --single

    # 全量评估
    python tools/inference.py --onnx-model /path/to/segment.onnx --eval
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
ort.preload_dlls()

import yaml

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ONNX_MODEL_PATH = "runs/segment/11s-seg/clean_seg_v1/weights/best_raw.onnx"
PT_MODEL_PATH = "runs/segment/11s-seg/clean_seg_v1/weights/best.pt"
DATASET_YAML_PATH = "ultralytics/cfg/datasets/clean_v1_seg_testBUU.yaml"
OUTPUT_DIR = "runs/test/infer"
IMGSZ = [640, 640]
CONF_THRESHOLD = 0.25   # YOLO默认置信度阈值
IOU_THRESHOLD = 0.7
MAX_DET = 300

# YOLOv11 seg head 参数
NC = 1          # pipeline 的 YOLO paper 类别数
NM = 32         # mask系数数量
REG_MAX = 16    # DFL通道数
STRIDES = [4, 8, 16, 32]

# ONNX 输出名列表
OUTPUT_NAMES = [
    "box_p2", "score_p2", "mask_coeff_p2",
    "box_p3", "score_p3", "mask_coeff_p3",
    "box_p4", "score_p4", "mask_coeff_p4",
    "box_p5", "score_p5", "mask_coeff_p5",
    "proto",
]


def configure_model(session: ort.InferenceSession) -> None:
    """根据 ONNX 输出节点配置 P2/P3 分割头、类别数和 mask 系数数。"""
    global NC, NM, STRIDES, OUTPUT_NAMES
    outputs = {output.name: output for output in session.get_outputs()}
    levels = sorted(
        int(match.group(1)) for name in outputs if (match := re.fullmatch(r"box_p(\d+)", name))
    )
    expected = {
        name
        for level in levels
        for name in (f"box_p{level}", f"score_p{level}", f"mask_coeff_p{level}")
    } | {"proto"}
    if levels not in ([2, 3, 4, 5], [3, 4, 5]) or not expected.issubset(outputs):
        raise ValueError(f"仅支持 P2-P5 或 P3-P5 原始分割输出，实际输出为: {sorted(outputs)}")
    STRIDES = [2**level for level in levels]
    OUTPUT_NAMES = [
        name
        for level in levels
        for name in (f"box_p{level}", f"score_p{level}", f"mask_coeff_p{level}")
    ] + ["proto"]
    score_shape = outputs[f"score_p{levels[0]}"].shape
    coefficient_shape = outputs[f"mask_coeff_p{levels[0]}"].shape
    if isinstance(score_shape[1], int):
        NC = score_shape[1]
    if isinstance(coefficient_shape[1], int):
        NM = coefficient_shape[1]
    print(f"[ONNX] 分割层级: {levels}; strides={STRIDES}; nc={NC}; nm={NM}")


def setup_logger(output_dir: str, name: str = "inference") -> logging.Logger:
    """配置日志:控制台INFO+,文件DEBUG+(保留所有详细信息)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{name}_{timestamp}.txt")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"日志文件: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_val_image_paths(dataset_yaml_path: str) -> list[str]:
    """从dataset yaml中读取val图片路径列表."""
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    val_paths = cfg.get("val", [])
    if not isinstance(val_paths, list):
        val_paths = [val_paths]
    image_paths = []
    for val_dir in val_paths:
        val_dir = Path(val_dir)
        if val_dir.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                image_paths.extend(str(p) for p in sorted(val_dir.glob(ext)))
        elif val_dir.is_file():
            with open(val_dir, "r", encoding="utf-8") as f:
                image_paths.extend(line.strip() for line in f if line.strip())
    return image_paths


def load_gt_label(label_path: str, img_w: int, img_h: int) -> dict:
    """加载YOLO分割格式的GT标注,返回boxes/classes/segments."""
    boxes, classes, segments = [], [], []
    if not os.path.exists(label_path):
        return {'boxes': np.zeros((0, 4), dtype=np.float32),
                'classes': np.zeros((0,), dtype=np.int64), 'segments': []}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5 or len(parts) % 2 != 1:
                continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(cls_id)
            segments.append(coords)
    if boxes:
        return {'boxes': np.array(boxes, dtype=np.float32),
                'classes': np.array(classes, dtype=np.int64), 'segments': segments}
    return {'boxes': np.zeros((0, 4), dtype=np.float32),
            'classes': np.zeros((0,), dtype=np.int64), 'segments': []}


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------

def preprocess(
    img: np.ndarray, imgsz: list[int] | tuple[int, int] = (640, 640)
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """letterbox + 归一化 + BGR2RGB + BCHW."""
    h0, w0 = img.shape[:2]
    target_h, target_w = imgsz
    r = min(target_h / h0, target_w / w0)
    new_w, new_h = round(w0 * r), round(h0 * r)
    dw, dh = target_w - new_w, target_h - new_h
    pad_left, pad_top = dw // 2, dh // 2
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, dh - pad_top, pad_left, dw - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_padded = img_padded[..., ::-1].transpose((2, 0, 1))
    img_padded = np.ascontiguousarray(img_padded, dtype=np.float32) / 255.0
    img_padded = img_padded[None]
    return img_padded, (r, r), (pad_left, pad_top)


# ---------------------------------------------------------------------------
# ONNX 后处理
# ---------------------------------------------------------------------------

def dfl_numpy(box: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """DFL解码 - numpy实现."""
    B = box.shape[0]
    spatial_dim = np.prod(box.shape[2:])
    box_reshaped = box.reshape(B, 4, reg_max, spatial_dim)
    box_softmax = np.exp(box_reshaped - box_reshaped.max(axis=2, keepdims=True))
    box_softmax = box_softmax / box_softmax.sum(axis=2, keepdims=True)
    weights = np.arange(reg_max, dtype=np.float32).reshape(1, 1, -1, 1)
    decoded = (box_softmax * weights).sum(axis=2)
    return decoded


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7, max_det: int = 300) -> np.ndarray:
    """NMS numpy版."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        x1 = np.maximum(boxes[0, 0], boxes[1:, 0])
        y1 = np.maximum(boxes[0, 1], boxes[1:, 1])
        x2 = np.minimum(boxes[0, 2], boxes[1:, 2])
        y2 = np.minimum(boxes[0, 3], boxes[1:, 3])
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])
        area2 = (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3] - boxes[1:, 1])
        iou = inter_area / np.maximum(area1 + area2 - inter_area, 1e-16)
        mask = iou < iou_threshold
        order = order[1:][mask]
        boxes = boxes[1:][mask]
    return np.array(keep, dtype=np.int64)


def postprocess_raw(
    outputs: dict[str, np.ndarray],
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    max_det: int = 300,
    nc: int = 2,
    nm: int = 32,
    reg_max: int = 16,
    strides: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """后处理raw ONNX输出."""
    if strides is None:
        strides = [8, 16, 32]
    proto = outputs["proto"]
    all_boxes, all_scores, all_classes, all_mask_coeffs = [], [], [], []

    first_level = 2 if len(strides) == 4 else 3
    for i, stride in enumerate(strides):
        level = first_level + i
        box_raw = outputs[f"box_p{level}"]              # [1, 4*reg_max, H, W]
        score_raw = outputs[f"score_p{level}"]          # [1, nc, H, W]
        mask_coeff_raw = outputs[f"mask_coeff_p{level}"]  # [1, nm, H, W]
        B, _, H, W = box_raw.shape
        num_anchors = H * W

        # DFL decode: [1, 4*reg_max, H*W] -> [1, 4, H*W]
        box_decoded = dfl_numpy(box_raw.reshape(B, 4 * reg_max, num_anchors), reg_max)

        # anchor grid [1, H*W, 2] - center points in feature map (0.5, 1.5, ...)
        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        anchor_grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
        anchor_grid += 0.5  # center of each grid cell
        anchor_grid = anchor_grid[np.newaxis, ...]

        # dist2bbox: (anchor ± dist) * stride, where dist is in feature-map scale
        lt, rb = np.split(box_decoded, 2, axis=1)  # each [1, 2, H*W]
        x1y1 = anchor_grid - lt.transpose(0, 2, 1)   # [1, H*W, 2]
        x2y2 = anchor_grid + rb.transpose(0, 2, 1)   # [1, H*W, 2]
        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=-1) * stride  # [1, H*W, 4]

        # scores: sigmoid
        scores_sigmoid = 1 / (1 + np.exp(-np.clip(score_raw, -50, 50)))  # [1, nc, H, W]
        scores_flat = scores_sigmoid.reshape(B, nc, num_anchors).transpose(0, 2, 1)  # [1, H*W, nc]

        # mask coefficients: [1, nm, H*W] -> [1, H*W, nm]
        mask_coeff_flat = mask_coeff_raw.reshape(B, nm, num_anchors).transpose(0, 2, 1)

        # 去掉batch维度
        boxes_xyxy = boxes_xyxy[0]        # [H*W, 4]
        scores_flat = scores_flat[0]      # [H*W, nc]
        mask_coeff_flat = mask_coeff_flat[0]  # [H*W, nm]

        # 筛选置信度
        max_scores = scores_flat.max(axis=1)
        keep = max_scores > conf_threshold
        if keep.any():
            all_boxes.append(boxes_xyxy[keep])
            all_scores.append(max_scores[keep])
            all_classes.append(scores_flat[keep].argmax(axis=1))
            all_mask_coeffs.append(mask_coeff_flat[keep])

    if not all_boxes:
        return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), np.zeros((0, NM), dtype=np.float32), proto)

    boxes_all = np.concatenate(all_boxes, axis=0)
    scores_all = np.concatenate(all_scores, axis=0)
    classes_all = np.concatenate(all_classes, axis=0)
    mask_coeffs_all = np.concatenate(all_mask_coeffs, axis=0)

    # NMS
    keep_idx = nms_numpy(boxes_all, scores_all, iou_threshold, max_det)
    return (boxes_all[keep_idx], scores_all[keep_idx], classes_all[keep_idx],
            mask_coeffs_all[keep_idx], proto)


# ---------------------------------------------------------------------------
# Mask生成
# ---------------------------------------------------------------------------

def process_mask(
    proto: np.ndarray,
    mask_coeffs: np.ndarray,
    boxes: np.ndarray,
    input_shape: tuple[int, int],
    orig_shape: tuple[int, int],
    ratio: tuple[float, float],
    pad: tuple[int, int],
    external_shape: tuple[int, int],
) -> np.ndarray:
    """在 letterbox 坐标生成实例 mask 并准确还原到原图."""
    nm = proto.shape[1]
    proto_h, proto_w = proto.shape[2], proto.shape[3]
    proto_flat = proto[0].reshape(nm, -1)
    masks = (mask_coeffs @ proto_flat).reshape(-1, proto_h, proto_w)
    masks = 1 / (1 + np.exp(-np.clip(masks, -50, 50)))
    input_h, input_w = input_shape
    scaled_boxes = boxes * np.array(
        [proto_w / input_w, proto_h / input_h, proto_w / input_w, proto_h / input_h],
        dtype=np.float32,
    )
    rows = np.arange(proto_h, dtype=np.float32)[None, :, None]
    cols = np.arange(proto_w, dtype=np.float32)[None, None, :]
    x1, y1, x2, y2 = np.split(scaled_boxes[:, :, None], 4, axis=1)
    masks *= (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)
    orig_h, orig_w = orig_shape
    external_h, external_w = external_shape
    model_top = (input_h - external_h) // 2
    model_left = (input_w - external_w) // 2
    resized_h, resized_w = round(orig_h * ratio[1]), round(orig_w * ratio[0])
    pad_left, pad_top = pad
    restored = []
    for mask in masks:
        mask = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        mask = mask[model_top:model_top + external_h, model_left:model_left + external_w]
        mask = mask[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]
        restored.append(cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5)
    return np.stack(restored) if restored else np.zeros((0, orig_h, orig_w), dtype=bool)


def scale_boxes_to_orig(
    boxes: np.ndarray, ratio: tuple[float, float], pad: tuple[int, int], model_pad: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """letterbox坐标系 -> 原始图像坐标系."""
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


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def draw_segmentation(img, boxes, scores, classes, masks, class_names,
                      colors=None, alpha=0.4):
    """在图像上绘制分割mask、bbox和标签."""
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    vis = img.copy()
    for box, score, cls_id, mask in zip(boxes, scores, classes, masks):
        color = colors[int(cls_id) % len(colors)]
        cm = np.zeros_like(vis, dtype=np.uint8)
        cm[mask] = color
        vis = cv2.addWeighted(vis, 1.0, cm, alpha, 0)
        mu8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(int(cls_id), str(cls_id))} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """[N,4] x [M,4] -> [N,M] IoU矩阵."""
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-16)


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """两个bool mask的IoU."""
    return float(np.logical_and(m1, m2).sum()) / max(np.logical_or(m1, m2).sum(), 1)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """从PR点计算AP (AUC with monotonic decreasing precision)."""
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在指定IoU阈值下计算TP/FP数组."""
    num_pred = len(cls_preds)
    # 重置GT匹配状态
    for gt in cls_gts_list:
        gt['matched'] = False

    # 按图片分组GT
    gts_by_img: dict[int, list] = {}
    for gt_idx, gt in enumerate(cls_gts_list):
        img_idx = gt['img_idx']
        gts_by_img.setdefault(img_idx, []).append((gt_idx, gt))

    tp = np.zeros(num_pred, dtype=bool)
    match_ious_out = np.zeros(num_pred, dtype=np.float32)

    for p_idx, pred in enumerate(cls_preds):
        img_idx = pred['img_idx']
        if img_idx not in gts_by_img:
            continue  # fp
        img_gts = gts_by_img[img_idx]
        best_iou = 0.0
        best_gt_idx = -1
        pred_box = pred['box']
        gt_boxes = np.array([gt['box'] for _, gt in img_gts])
        ious = box_iou(pred_box[None, :], gt_boxes)[0]
        for g_idx, (orig_gt_idx, gt) in enumerate(img_gts):
            if not gt['matched'] and ious[g_idx] > best_iou:
                best_iou, best_gt_idx = ious[g_idx], orig_gt_idx
        if best_iou >= iou_threshold:
            tp[p_idx] = True
            match_ious_out[p_idx] = best_iou
            cls_gts_list[best_gt_idx]['matched'] = True

    return tp, match_ious_out, gts_by_img


def evaluate_predictions(
    all_preds: list[dict],
    all_gts: list[dict],
    class_names: dict[int, str],
    nc: int,
    logger: logging.Logger | None = None,
) -> dict:
    """全面评估预测结果,返回每类和总体指标(含mAP50, mAP50-95, maskAP50, maskAP50-95)."""
    # 按类别收集预测和GT
    all_class_preds = {c: [] for c in range(nc)}
    all_class_gts = {c: [] for c in range(nc)}

    for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
        for i in range(len(preds['scores'])):
            cls = int(preds['classes'][i])
            all_class_preds[cls].append({
                'img_idx': img_idx, 'score': float(preds['scores'][i]),
                'box': preds['boxes'][i].copy(),
                'mask': preds['masks'][i].copy() if i < len(preds['masks']) else None,
            })
        for j in range(len(gts['classes'])):
            cls = int(gts['classes'][j])
            all_class_gts[cls].append({
                'img_idx': img_idx, 'box': gts['boxes'][j].copy(),
                'segment': gts['segments'][j] if j < len(gts['segments']) else None,
                'matched': False,
            })

    iou_thresholds = np.arange(0.50, 1.0, 0.05)  # [0.50, 0.55, ..., 0.95]

    eval_results_per_class = {}
    for cls in range(nc):
        cls_preds = all_class_preds[cls]
        cls_gts_list = all_class_gts[cls]
        cls_name = class_names.get(cls, str(cls))
        num_gt = len(cls_gts_list)
        num_pred = len(cls_preds)

        if num_gt == 0:
            eval_results_per_class[cls] = {
                'name': cls_name, 'num_gt': 0, 'num_pred': num_pred,
                'precision': 0.0, 'recall': 0.0,
                'ap50': 0.0, 'map50_95': 0.0,
                'mask_ap50': 0.0, 'mask_map50_95': 0.0,
            }
            continue

        cls_preds.sort(key=lambda x: x['score'], reverse=True)

        # 对每个IoU阈值计算AP
        ap_per_threshold = []
        mask_ap_per_threshold = []

        for iou_t in iou_thresholds:
            # 需要深拷贝GT的matched状态,因为每次匹配独立
            gts_copy = [dict(gt) for gt in cls_gts_list]

            tp, match_ious, gts_by_img_local = _compute_tp_fp_for_iou(
                cls_preds, gts_copy, float(iou_t),
            )
            fp = ~tp

            tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
            recalls = tp_cumsum / num_gt
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-16)
            ap = compute_ap(recalls, precisions)
            ap_per_threshold.append(ap)

            # Mask AP 在该 IoU 阈值下
            matched_mask_ious = []
            for p_idx, pred in enumerate(cls_preds):
                if tp[p_idx] and pred['mask'] is not None:
                    img_idx = pred['img_idx']
                    if img_idx in gts_by_img_local:
                        gt_boxes_img = np.array([gt['box'] for _, gt in gts_by_img_local[img_idx]])
                        ious_img = box_iou(pred['box'][None, :], gt_boxes_img)[0]
                        best_gt_in_img = int(np.argmax(ious_img))
                        gt_seg = gts_by_img_local[img_idx][best_gt_in_img][1]['segment']
                        if gt_seg is not None and len(gt_seg) > 0:
                            gt_mask = np.zeros(pred['mask'].shape, dtype=np.uint8)
                            cv2.fillPoly(gt_mask, [gt_seg.astype(np.int32).reshape(-1, 1, 2)], 1)
                            matched_mask_ious.append(mask_iou(pred['mask'], gt_mask.astype(bool)))
            mask_ap = float(np.mean(matched_mask_ious)) if matched_mask_ious else 0.0
            mask_ap_per_threshold.append(mask_ap)

        ap50 = ap_per_threshold[0]  # IoU=0.50
        map50_95 = float(np.mean(ap_per_threshold))
        mask_ap50 = mask_ap_per_threshold[0]
        mask_map50_95 = float(np.mean(mask_ap_per_threshold))

        # 用 AP50 的匹配结果计算 P/R(在 F1 最大点)
        gts_final = [dict(gt) for gt in cls_gts_list]
        tp_final, _, gts_by_img_final = _compute_tp_fp_for_iou(cls_preds, gts_final, 0.5)
        fp_final = ~tp_final
        tp_c, fp_c = np.cumsum(tp_final), np.cumsum(fp_final)
        r_final = tp_c / num_gt
        p_final = tp_c / np.maximum(tp_c + fp_c, 1e-16)
        f1 = 2 * p_final * r_final / np.maximum(p_final + r_final, 1e-16)
        best_idx = int(np.argmax(f1)) if len(f1) > 0 else 0
        final_p, final_r = float(p_final[best_idx]), float(r_final[best_idx])

        eval_results_per_class[cls] = {
            'name': cls_name, 'num_gt': num_gt, 'num_pred': num_pred,
            'precision': final_p, 'recall': final_r,
            'ap50': ap50, 'map50_95': map50_95,
            'mask_ap50': mask_ap50, 'mask_map50_95': mask_map50_95,
        }

    total_gt = sum(r['num_gt'] for r in eval_results_per_class.values())
    total_pred = sum(r['num_pred'] for r in eval_results_per_class.values())
    valid = [r for r in eval_results_per_class.values() if r['num_gt'] > 0]
    overall = {
        'total_gt': total_gt, 'total_pred': total_pred,
        'precision': np.mean([r['precision'] for r in valid]).item() if valid else 0.0,
        'recall': np.mean([r['recall'] for r in valid]).item() if valid else 0.0,
        'ap50': np.mean([r['ap50'] for r in valid]).item() if valid else 0.0,
        'map50_95': np.mean([r['map50_95'] for r in valid]).item() if valid else 0.0,
        'mask_ap50': np.mean([r['mask_ap50'] for r in valid]).item() if valid else 0.0,
        'mask_map50_95': np.mean([r['mask_map50_95'] for r in valid]).item() if valid else 0.0,
    }
    return {'per_class': eval_results_per_class, 'overall': overall}


def format_eval_results(results: dict) -> str:
    """格式化评估结果表格."""
    lines = ["=" * 80, "评估结果汇总", "=" * 80]
    header = f"{'类别':<8} {'GT':>5} {'Pred':>6} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10} {'Mask mAP50':>12} {'Mask mAP50-95':>15}"
    lines.extend([header, "-" * 80])
    for cls in sorted(results['per_class'].keys()):
        r = results['per_class'][cls]
        lines.append(f"{r['name']:<8} {r['num_gt']:>5} {r['num_pred']:>6} "
                     f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
                     f"{r['ap50']:>8.4f} {r['map50_95']:>10.4f} "
                     f"{r['mask_ap50']:>12.4f} {r['mask_map50_95']:>15.4f}")
    lines.append("-" * 80)
    o = results['overall']
    lines.append(f"{'Overall':<8} {o['total_gt']:>5} {o['total_pred']:>6} "
                 f"{o['precision']:>8.4f} {o['recall']:>8.4f} "
                 f"{o['ap50']:>8.4f} {o['map50_95']:>10.4f} "
                 f"{o['mask_ap50']:>12.4f} {o['mask_map50_95']:>15.4f}")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ONNX / PyTorch 推理
# ---------------------------------------------------------------------------

def onnx_inference(session: ort.InferenceSession, img: np.ndarray,
                   orig_h: int, orig_w: int):
    """ONNX推理单图,返回 boxes, scores, classes, masks."""
    prep_img, ratio, pad = preprocess(img, IMGSZ)
    external_h, external_w = IMGSZ
    internal_shape = (((external_h + 31) // 32) * 32, ((external_w + 31) // 32) * 32)
    model_pad = ((internal_shape[1] - external_w) // 2, (internal_shape[0] - external_h) // 2)
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: prep_img})
    outputs_dict = dict(zip((item.name for item in session.get_outputs()), onnx_outputs))
    det_boxes, det_scores, det_classes, det_mask_coeffs, proto = postprocess_raw(
        outputs_dict, conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD, max_det=MAX_DET,
        nc=NC, nm=NM, reg_max=REG_MAX, strides=STRIDES)
    if len(det_boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), np.zeros((0, orig_h, orig_w), dtype=bool))
    masks = process_mask(
        proto, det_mask_coeffs, det_boxes, internal_shape, (orig_h, orig_w), ratio, pad, (external_h, external_w),
    )
    det_boxes = scale_boxes_to_orig(det_boxes, ratio, pad, model_pad)
    det_boxes[:, [0, 2]] = det_boxes[:, [0, 2]].clip(0, orig_w)
    det_boxes[:, [1, 3]] = det_boxes[:, [1, 3]].clip(0, orig_h)
    valid = (det_boxes[:, 2] > det_boxes[:, 0]) & (det_boxes[:, 3] > det_boxes[:, 1])
    return det_boxes[valid], det_scores[valid], det_classes[valid], masks[valid]


def pytorch_inference(pt_model, img: np.ndarray, orig_h: int, orig_w: int):
    """PyTorch推理单图."""
    results = pt_model(img, imgsz=IMGSZ, conf=0.25, iou=IOU_THRESHOLD, max_det=MAX_DET, verbose=False)
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return (np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64), np.zeros((0, orig_h, orig_w), dtype=bool))
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int64)
    if result.masks is not None:
        masks_data = result.masks.data.cpu().numpy().astype(np.float32)
        masks = np.stack([cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5 for m in masks_data])
    else:
        masks = np.zeros((len(boxes), orig_h, orig_w), dtype=bool)
    return boxes, scores, classes, masks


# ---------------------------------------------------------------------------
# 单图对比
# ---------------------------------------------------------------------------

def single_image_compare(session, pt_model, img_path, label_dir, class_names, output_dir, logger):
    """ONNX vs PyTorch单图对比+GT可视化."""
    logger.info(f"单图对比: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        logger.error(f"无法读取图片: {img_path}")
        return
    orig_h, orig_w = img.shape[:2]
    img_name = os.path.basename(img_path)
    logger.info(f"图片尺寸: {orig_w}x{orig_h}")

    # GT
    label_name = os.path.splitext(img_name)[0] + ".txt"
    gt = load_gt_label(os.path.join(label_dir, label_name), orig_w, orig_h)
    logger.debug(f"GT实例数: {len(gt['classes'])}")
    for i in range(len(gt['classes'])):
        logger.debug(f"  GT[{i}]: {class_names.get(int(gt['classes'][i]))} box={gt['boxes'][i].tolist()}")

    # ONNX
    logger.info("ONNX推理...")
    ob, os_, oc, om = onnx_inference(session, img, orig_h, orig_w)
    logger.info(f"  ONNX: {len(ob)} 个实例")
    for i in range(len(ob)):
        logger.debug(f"  ONNX[{i}]: {class_names.get(int(oc[i]))} conf={os_[i]:.4f} box={ob[i].tolist()}")

    # PyTorch
    logger.info("PyTorch推理...")
    pb, ps, pc, pm = pytorch_inference(pt_model, img, orig_h, orig_w)
    logger.info(f"  PyTorch: {len(pb)} 个实例")
    for i in range(len(pb)):
        logger.debug(f"  PT[{i}]: {class_names.get(int(pc[i]))} conf={ps[i]:.4f} box={pb[i].tolist()}")

    # 对比
    logger.info("\n" + "=" * 60)
    logger.info(f"对比: GT={len(gt['classes'])} ONNX={len(ob)} PT={len(pb)}")
    if len(ob) > 0 and len(pb) > 0:
        iou_mat = box_iou(ob, pb)
        logger.info(f"  ONNX->PT 平均最大IoU: {iou_mat.max(axis=1).mean():.4f}")
        logger.info(f"  PT->ONNX 平均最大IoU: {iou_mat.max(axis=0).mean():.4f}")
        matched = sum(1 for oi in range(len(ob)) for pi in range(len(pb))
                      if oc[oi] == pc[pi] and iou_mat[oi, pi] > 0.5)
        logger.info(f"  IoU>0.5同类别匹配对数: {matched}")
    logger.info("=" * 60 + "\n")

    # 保存可视化
    os.makedirs(output_dir, exist_ok=True)
    vis_onnx = draw_segmentation(img, ob, os_, oc, om, class_names) if len(ob) > 0 else img.copy()
    vis_pt = draw_segmentation(img, pb, ps, pc, pm, class_names) if len(pb) > 0 else img.copy()
    cv2.imwrite(os.path.join(output_dir, f"onnx_{img_name}"), vis_onnx)
    cv2.imwrite(os.path.join(output_dir, f"pt_{img_name}"), vis_pt)
    logger.info(f"可视化: onnx_{img_name}, pt_{img_name}")

    # GT可视化
    vis_gt = img.copy()
    gt_colors = [(0, 0, 255), (255, 255, 0)]
    for box, cls_id, seg in zip(gt['boxes'], gt['classes'], gt['segments']):
        color = gt_colors[int(cls_id) % 2]
        cv2.rectangle(vis_gt, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), color, 2)
        cv2.putText(vis_gt, f"GT:{class_names.get(int(cls_id), cls_id)}",
                    (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if seg is not None and len(seg) > 0:
            cv2.polylines(vis_gt, [seg.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
    cv2.imwrite(os.path.join(output_dir, f"gt_{img_name}"), vis_gt)

    # 并排对比
    comp = np.hstack([vis_gt, vis_onnx, vis_pt])
    cv2.imwrite(os.path.join(output_dir, f"comparison_{img_name}"), comp)
    logger.info(f"对比图: comparison_{img_name}")
    logger.info("单图对比完成！")


# ---------------------------------------------------------------------------
# 批量评估
# ---------------------------------------------------------------------------

def run_evaluation(session, image_paths, label_dir, class_names, output_dir, logger):
    """ONNX批量推理 + 评估."""
    logger.info(f"开始评估,共 {len(image_paths)} 张图片")
    all_preds, all_gts, total_inst = [], [], 0
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            logger.info(f"  进度: [{idx + 1}/{len(image_paths)}]")
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"跳过: {img_path}")
            continue
        orig_h, orig_w = img.shape[:2]
        img_name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        gt = load_gt_label(label_path, orig_w, orig_h)
        boxes, scores, classes, masks = onnx_inference(session, img, orig_h, orig_w)
        total_inst += len(boxes)
        logger.debug(f"[{idx+1}] {img_name}: {len(gt['classes'])} GT, {len(boxes)} pred")
        # 保存每张图的可视化结果
        if len(boxes) > 0:
            vis_img = draw_segmentation(img, boxes, scores, classes, masks, class_names)
        else:
            vis_img = img.copy()
        cv2.imwrite(os.path.join(vis_dir, img_name), vis_img)
        all_preds.append({'boxes': boxes, 'scores': scores, 'classes': classes, 'masks': masks})
        all_gts.append(gt)

    logger.info(f"推理完成,总检测: {total_inst} 实例")

    logger.info("正在计算评估指标...")
    eval_results = evaluate_predictions(all_preds, all_gts, class_names, NC, logger=logger)
    result_str = format_eval_results(eval_results)
    logger.info("\n" + result_str)

    # 保存评估结果
    eval_dir = os.path.join(output_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = os.path.join(eval_dir, f"eval_metrics_{ts}.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(result_str)
        f.write("\n\n详细PR曲线数据:\n" + "-" * 60 + "\n")
        for cls in sorted(eval_results['per_class'].keys()):
            r = eval_results['per_class'][cls]
            f.write(f"\n类别 '{r['name']}' (ID={cls}):\n")
            f.write(f"  GT数:        {r['num_gt']}\n")
            f.write(f"  预测数:      {r['num_pred']}\n")
            f.write(f"  Precision:   {r['precision']:.6f}\n")
            f.write(f"  Recall:      {r['recall']:.6f}\n")
            f.write(f"  Box AP50:    {r['ap50']:.6f}\n")
            f.write(f"  Mask AP50:   {r['mask_ap50']:.6f}\n")
    logger.info(f"评估结果已保存: {eval_path}")
    return eval_results


def pidnet_inference(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    """执行 PIDNet ONNX letterbox 推理并还原到原图尺寸."""
    input_meta = session.get_inputs()[0]
    target_h, target_w = (int(input_meta.shape[2]), int(input_meta.shape[3]))
    orig_h, orig_w = image.shape[:2]
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = round(orig_h * scale), round(orig_w * scale)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized, top, target_h - new_h - top, left, target_w - new_w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    tensor = padded[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((tensor - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    logits = session.run(None, {input_meta.name: tensor})[0][0]
    internal_h, internal_w = ((target_h + 31) // 32 * 32, (target_w + 31) // 32 * 32)
    logits = cv2.resize(logits.transpose(1, 2, 0), (internal_w, internal_h), interpolation=cv2.INTER_LINEAR)
    model_top, model_left = (internal_h - target_h) // 2, (internal_w - target_w) // 2
    logits = logits[model_top:model_top + target_h, model_left:model_left + target_w]
    mask = np.argmax(logits, axis=2).astype(np.uint8)
    mask = mask[top:top + new_h, left:left + new_w]
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def semantic_metrics(confusion: np.ndarray, class_names: dict[int, str]) -> dict:
    """由语义分割混淆矩阵计算逐类和整体指标."""
    per_class = {}
    for class_id, name in class_names.items():
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[class_id] = {
            "name": name, "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "iou": iou, "dice": dice,
        }
    valid = list(per_class.values())
    total = float(confusion.sum())
    return {
        "per_class": per_class,
        "overall": {
            "precision": float(np.mean([item["precision"] for item in valid])) if valid else 0.0,
            "recall": float(np.mean([item["recall"] for item in valid])) if valid else 0.0,
            "iou": float(np.mean([item["iou"] for item in valid])) if valid else 0.0,
            "dice": float(np.mean([item["dice"] for item in valid])) if valid else 0.0,
            "pixel_accuracy": float(np.trace(confusion) / total) if total else 0.0,
        },
        "confusion_matrix": confusion.tolist(),
    }


def draw_semantic(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """绘制 background、paper、liquid、metal 融合语义分割图."""
    colors = np.array([[0, 0, 0], [0, 180, 0], [0, 100, 255], [255, 80, 80]], dtype=np.uint8)
    overlay = image.copy()
    foreground = (mask > 0) & (mask < len(colors))
    overlay[foreground] = colors[mask[foreground]]
    result = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    result[~foreground] = image[~foreground]
    return result


def update_confusion(confusion: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    """累加同尺寸语义 mask 的混淆矩阵."""
    classes = confusion.shape[0]
    valid = (gt >= 0) & (gt < classes) & (pred >= 0) & (pred < classes)
    values = classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    confusion += np.bincount(values, minlength=classes * classes).reshape(classes, classes)


def run_pipeline_evaluation(args: argparse.Namespace) -> dict:
    """使用 pipeline 清单执行 YOLO+PIDNet 融合分割评测."""
    global IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, NC, STRIDES, OUTPUT_NAMES
    IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, NC = args.imgsz, args.conf, args.iou, 1
    output_dir = Path(args.output)
    vis_dir = output_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for path in vis_dir.iterdir():
        if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg"}:
            path.unlink()
    logger = setup_logger(str(output_dir), "pipeline_segment_test")
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    providers = [
        provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ] or ort.get_available_providers()
    yolo_session = ort.InferenceSession(args.onnx_model, providers=providers)
    configure_model(yolo_session)
    pidnet_session = ort.InferenceSession(args.pidnet_onnx, providers=providers)
    yolo_input = yolo_session.get_inputs()[0].shape
    IMGSZ = [int(yolo_input[2]), int(yolo_input[3])]
    actual_outputs = [item.name for item in yolo_session.get_outputs()]
    if set(actual_outputs) != set(OUTPUT_NAMES):
        raise ValueError(f"YOLO分割ONNX输出不匹配:expected={OUTPUT_NAMES},actual={actual_outputs}")

    yolo_preds, yolo_gts = [], []
    pidnet_confusion = np.zeros((3, 3), dtype=np.int64)
    combined_confusion = np.zeros((4, 4), dtype=np.int64)
    for index, item in enumerate(samples, start=1):
        image = cv2.imread(str(item["image"]))
        pidnet_gt = cv2.imread(str(item["pidnet_mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or pidnet_gt is None:
            raise FileNotFoundError(f"测试图片或PIDNet GT无法读取:{item}")
        height, width = image.shape[:2]
        yolo_gt = load_gt_label(str(item["segment_label"]), width, height)
        boxes, scores, classes, masks = onnx_inference(yolo_session, image, height, width)
        pidnet_pred = pidnet_inference(pidnet_session, image)
        yolo_preds.append({"boxes": boxes, "scores": scores, "classes": classes, "masks": masks})
        yolo_gts.append(yolo_gt)
        update_confusion(pidnet_confusion, pidnet_gt, pidnet_pred)

        combined_gt = np.zeros((height, width), dtype=np.uint8)
        combined_pred = np.zeros((height, width), dtype=np.uint8)
        combined_gt[pidnet_gt == 1], combined_gt[pidnet_gt == 2] = 2, 3
        combined_gt[pidnet_gt == 255] = 255
        combined_pred[pidnet_pred == 1], combined_pred[pidnet_pred == 2] = 2, 3
        for segment in yolo_gt["segments"]:
            cv2.fillPoly(combined_gt, [segment.round().astype(np.int32)], 1)
        for mask in masks:
            combined_pred[mask] = 1
        update_confusion(combined_confusion, combined_gt, combined_pred)

        comparison = np.hstack([image, draw_semantic(image, combined_gt), draw_semantic(image, combined_pred)])
        cv2.imwrite(str(vis_dir / f"{item.get('id', Path(item['image']).stem)}.png"), comparison)
        if index % 50 == 0 or index == len(samples):
            logger.info(f"PIPELINE_TEST_PROGRESS | segment | {index}/{len(samples)}")

    yolo_metrics = evaluate_predictions(yolo_preds, yolo_gts, {0: "paper"}, 1, logger=logger)
    results = {
        "images": len(samples),
        "yolo_segment": yolo_metrics,
        "pidnet": semantic_metrics(pidnet_confusion, {1: "liquid", 2: "metal"}),
        "combined": semantic_metrics(combined_confusion, {1: "paper", 2: "liquid", 3: "metal"}),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    text = format_eval_results(yolo_metrics)
    text += "\n\nPIDNet:\n" + json.dumps(results["pidnet"], ensure_ascii=False, indent=2)
    text += "\n\nCombined:\n" + json.dumps(results["combined"], ensure_ascii=False, indent=2)
    (output_dir / "metrics.txt").write_text(text, encoding="utf-8")
    logger.info(f"PIPELINE_TEST_METRICS | segment | {json.dumps(results, ensure_ascii=False)}")
    return results


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="分割 ONNX 推理与评估，兼容 P2-P5 和 P3-P5 原始输出")
    parser.add_argument("--single", action="store_true", help="单图对比模式")
    parser.add_argument("--eval", action="store_true", help="批量评估模式")
    parser.add_argument("--img", type=str, default=None, help="单图模式指定图片")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=str, default=None, help="pipeline 测试清单 JSON")
    parser.add_argument("--onnx-model", type=str, default=None, help="YOLO Segment ONNX")
    parser.add_argument("--pidnet-onnx", type=str, default=None, help="PIDNet ONNX")
    parser.add_argument(
        "--imgsz", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=[640, 640], help="YOLO 输入尺寸"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    args = parser.parse_args()

    if args.manifest:
        if not args.onnx_model or not args.pidnet_onnx:
            parser.error("--manifest 模式需要 --onnx-model 和 --pidnet-onnx")
        run_pipeline_evaluation(args)
        return

    if not args.single and not args.eval:
        args.eval = True  # 默认批量评估

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    with open(DATASET_YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names = {int(k): v for k, v in cfg.get("names", {}).items()} if isinstance(cfg.get("names"), dict) else {}

    logger = setup_logger(output_dir)
    logger.info(f"类别: {class_names}")
    model_path = args.onnx_model or ONNX_MODEL_PATH
    logger.info(f"ONNX模型: {model_path}")

    available = ort.get_available_providers()
    logger.debug(f"可用providers: {available}")
    # 优先使用 GPU(CUDA),其次 CPU
    providers = []
    for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if p in available:
            providers.append(p)
    if not providers:
        providers = available
    logger.info(f"使用providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
    configure_model(session)
    input_shape = session.get_inputs()[0].shape
    IMGSZ[:] = [int(input_shape[2]), int(input_shape[3])]
    logger.info("ONNX模型加载成功")

    # 单图对比
    if args.single:
        image_paths = load_val_image_paths(DATASET_YAML_PATH)
        if not image_paths:
            logger.error("未找到验证集图片")
            return
        img_path = args.img if args.img else image_paths[0]
        label_dir = str(Path(img_path).parent.parent / "labels")

        logger.info("加载PyTorch模型...")
        from ultralytics import YOLO
        pt_model = YOLO(PT_MODEL_PATH)
        logger.info("PyTorch模型加载成功")

        single_image_compare(session, pt_model, img_path, label_dir, class_names, output_dir, logger)

    # 批量评估
    if args.eval:
        image_paths = load_val_image_paths(DATASET_YAML_PATH)
        if not image_paths:
            logger.error("未找到验证集图片")
            return
        label_dir = str(Path(image_paths[0]).parent.parent / "labels")
        run_evaluation(session, image_paths, label_dir, class_names, output_dir, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
