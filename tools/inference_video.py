#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2检测头ONNX模型视频推理脚本

对视频逐帧推理,绘制预测框,输出带检测结果的视频.

用法:
    python tools/inference_video.py
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

# ==============================================================================
# 配置
# ==============================================================================
ONNX_MODEL_PATH = "runs/detect/yolov11s_p2_detect_3cls/v4/weights/best_raw_p2_detect.onnx"
DATASET_YAML_PATH = "ultralytics/cfg/datasets/clean_v1_detect.yaml"
OUTPUT_DIR = "runs/inference/p2_detect_video"
IMGSZ = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.8
MAX_DET = 300

# 检测头参数
NC = 3
REG_MAX = 16
STRIDES = [4, 8, 16, 32]  # P2, P3, P4, P5

OUTPUT_NAMES = [
    "box_p2", "score_p2",
    "box_p3", "score_p3",
    "box_p4", "score_p4",
    "box_p5", "score_p5",
]

# 类别颜色 (BGR)
CLASS_COLORS = [
    (0, 180, 0),    # 0: paper  (绿色)
    (0, 100, 255),  # 1: liquid (橙色)
    (255, 80, 80),  # 2: metal  (蓝色调)
]


# ==============================================================================
# 预处理
# ==============================================================================

def preprocess(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, tuple, tuple]:
    """letterbox + BGR→RGB + normalize + BCHW."""
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    new_w, new_h = round(w0 * r), round(h0 * r)
    dw, dh = imgsz - new_w, imgsz - new_h
    pad_left, pad_top = dw // 2, dh // 2

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, dh - pad_top, pad_left, dw - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    img_padded = img_padded[..., ::-1].transpose(2, 0, 1)
    img_padded = np.ascontiguousarray(img_padded, dtype=np.float32) / 255.0
    return img_padded[None], (r, r), (pad_left, pad_top)


# ==============================================================================
# 后处理
# ==============================================================================

def dfl_numpy(box: np.ndarray, reg_max: int = 16) -> np.ndarray:
    """DFL解码 (numpy)."""
    B, _, spatial_dim = box.shape
    box_reshaped = box.reshape(B, 4, reg_max, spatial_dim)
    box_softmax = np.exp(box_reshaped - box_reshaped.max(axis=2, keepdims=True))
    box_softmax /= box_softmax.sum(axis=2, keepdims=True)
    weights = np.arange(reg_max, dtype=np.float32).reshape(1, 1, -1, 1)
    return (box_softmax * weights).sum(axis=2)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int) -> np.ndarray:
    """NMS."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    order = scores.argsort()[::-1]
    boxes_sorted = boxes[order]
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
        area1 = (boxes_sorted[0, 2] - boxes_sorted[0, 0]) * (boxes_sorted[0, 3] - boxes_sorted[0, 1])
        area2 = (boxes_sorted[1:, 2] - boxes_sorted[1:, 0]) * (boxes_sorted[1:, 3] - boxes_sorted[1:, 1])
        iou = inter / np.maximum(area1 + area2 - inter, 1e-16)
        mask = iou < iou_threshold
        order = order[1:][mask]
        boxes_sorted = boxes_sorted[1:][mask]
    return np.array(keep, dtype=np.int64)


def postprocess(
    outputs: dict[str, np.ndarray],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_det: int = 300,
    nc: int = 3,
    reg_max: int = 16,
    strides: list = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """后处理 ONNX raw 输出 → NMS 后的 boxes/scores/classes."""
    if strides is None:
        strides = [4, 8, 16, 32]
    all_boxes, all_scores, all_classes = [], [], []

    for i, stride in enumerate(strides):
        p_idx = i + 2
        box_raw = outputs[f"box_p{p_idx}"]
        score_raw = outputs[f"score_p{p_idx}"]
        B, _, H, W = box_raw.shape
        num_anchors = H * W

        box_decoded = dfl_numpy(box_raw.reshape(B, 4 * reg_max, num_anchors), reg_max)

        yv, xv = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        anchor_grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
        anchor_grid += 0.5
        anchor_grid = anchor_grid[np.newaxis, ...]

        lt, rb = np.split(box_decoded, 2, axis=1)
        x1y1 = anchor_grid - lt.transpose(0, 2, 1)
        x2y2 = anchor_grid + rb.transpose(0, 2, 1)
        boxes_xyxy = np.concatenate([x1y1, x2y2], axis=-1) * stride

        scores_sigmoid = 1 / (1 + np.exp(-np.clip(score_raw, -50, 50)))
        scores_flat = scores_sigmoid.reshape(B, nc, num_anchors).transpose(0, 2, 1)

        boxes_xyxy = boxes_xyxy[0]
        scores_flat = scores_flat[0]

        max_scores = scores_flat.max(axis=1)
        keep = max_scores > conf_threshold
        if keep.any():
            all_boxes.append(boxes_xyxy[keep])
            all_scores.append(max_scores[keep])
            all_classes.append(scores_flat[keep].argmax(axis=1))

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes_all = np.concatenate(all_boxes, axis=0)
    scores_all = np.concatenate(all_scores, axis=0)
    classes_all = np.concatenate(all_classes, axis=0)

    keep_idx = nms_numpy(boxes_all, scores_all, iou_threshold, max_det)
    return boxes_all[keep_idx], scores_all[keep_idx], classes_all[keep_idx]


def scale_boxes(boxes: np.ndarray, ratio: tuple, pad: tuple) -> np.ndarray:
    """letterbox坐标 → 原图坐标."""
    rw, rh = ratio
    pad_left, pad_top = pad
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, [0, 2]] /= rw
    boxes[:, [1, 3]] /= rh
    return boxes


# ==============================================================================
# 可视化
# ==============================================================================

def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray = None,
    class_names: dict = None,
    colors: list = None,
    line_thickness: int = 1,
    font_scale: float = 0.4,
    font_thickness: int = 1,
) -> np.ndarray:
    """在图上绘制检测框."""
    if colors is None:
        colors = [(0, 180, 0), (0, 100, 255), (255, 80, 80)]
    if class_names is None:
        class_names = {}
    vis = img.copy()
    for i, box in enumerate(boxes):
        cls_id = int(classes[i])
        color = colors[cls_id % len(colors)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)
        cls_name = class_names.get(cls_id, f"cls{cls_id}")
        if scores is not None:
            label = f"{cls_name} {scores[i]:.2f}"
        else:
            label = cls_name
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness,
        )
        label_y = y1 - th - 2 if y1 - th - 2 > 0 else y1 + th + 2
        cv2.rectangle(vis, (x1, label_y - 2), (x1 + tw + 2, label_y + th), color, -1)
        cv2.putText(
            vis, label, (x1 + 1, label_y + th - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness,
            cv2.LINE_AA,
        )
    return vis


# ==============================================================================
# ONNX 推理
# ==============================================================================

def onnx_inference(
    session: ort.InferenceSession,
    img: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单帧 ONNX 推理."""
    prep_img, ratio, pad = preprocess(img, IMGSZ)
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: prep_img})
    outputs_dict = dict(zip(OUTPUT_NAMES, onnx_outputs))

    boxes, scores, classes = postprocess(
        outputs_dict,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        max_det=MAX_DET,
        nc=NC,
        reg_max=REG_MAX,
        strides=STRIDES,
    )

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    boxes = scale_boxes(boxes, ratio, pad)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
    return boxes, scores, classes


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    # 视频文件配置
    # 从 val 图片目录推断视频路径,也可直接指定
    video_dir = "data/video/"
    video_name = "metal_260724.mp4"  # 修改为实际视频文件名
    video_path = os.path.join(video_dir, video_name)

    # 如果指定视频不存在,尝试从 val 数据集 path 反向查找
    if not os.path.exists(video_path):
        # 自动查找第一个 mp4 文件
        for root, dirs, files in os.walk(video_dir):
            for f in sorted(files):
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    video_path = os.path.join(root, f)
                    break
            if os.path.exists(video_path):
                break

    if not os.path.exists(video_path):
        print(f"[Error] 视频文件不存在: {video_path}")
        print(f"[Hint] 请修改 video_path 变量指向实际视频文件")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载配置
    with open(DATASET_YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names = (
        {int(k): v for k, v in cfg.get("names", {}).items()}
        if isinstance(cfg.get("names"), dict)
        else {}
    )
    nc = cfg.get("nc", 3)
    print(f"[Config] 类别数: {nc}, 类别名: {class_names}")
    print(f"[Config] ONNX模型: {ONNX_MODEL_PATH}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] {video_path}")
    print(f"[Video] 分辨率: {orig_w}x{orig_h}, FPS: {fps:.1f}, 总帧数: {frame_count}")

    # 输出视频
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(OUTPUT_DIR, f"{base_name}_pred.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_w, orig_h))
    if not out_writer.isOpened():
        print(f"[Error] 无法创建输出视频: {out_video_path}")
        sys.exit(1)

    # 加载 ONNX
    available_providers = ort.get_available_providers()
    print(f"[ONNX] 可用 providers: {available_providers}")
    providers = []
    for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if p in available_providers:
            providers.append(p)
    if not providers:
        providers = available_providers
    print(f"[ONNX] 使用 providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_options, providers=providers)
    print("[ONNX] 模型加载成功")

    # 逐帧处理
    total_time = 0.0
    frame_idx = 0
    total_pred = 0

    print("\n开始处理视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        t0 = time.time()
        boxes, scores, classes = onnx_inference(session, frame, orig_h, orig_w)
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed
        total_pred += len(boxes)

        # 绘制预测
        vis_pred = draw_boxes(
            frame, boxes, classes, scores=scores, class_names=class_names,
            colors=CLASS_COLORS, line_thickness=1, font_scale=0.4, font_thickness=1,
        )

        out_writer.write(vis_pred)

        if frame_idx % 100 == 0 or frame_idx == frame_count:
            avg_ms = total_time / frame_idx if frame_idx > 0 else 0
            print(
                f"  [{frame_idx}/{frame_count}] "
                f"{len(boxes)} detections  "
                f"{elapsed:.0f}ms  avg={avg_ms:.0f}ms"
            )

    cap.release()
    out_writer.release()

    avg_ms = total_time / frame_idx if frame_idx > 0 else 0
    print(f"\n[Summary] 处理帧数: {frame_idx}")
    print(f"[Summary] 总检测实例数: {total_pred}")
    print(f"[Summary] 平均推理时间: {avg_ms:.1f} ms")
    print(f"[Summary] 输出视频: {out_video_path}")


if __name__ == "__main__":
    main()