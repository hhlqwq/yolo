"""
验证集测试代码(分割任务)
功能:
1. 对验证集进行推理
2. 根据iou和conf阈值对比分割mask的结果
3. 保存漏检和误检图片(GT mask和预测mask使用明显不同的方式)
4. 打印每个类别和整体融合验证集的mAP、召回率、准确率等信息
5. 包含分割mask的IoU评估
"""

from ultralytics import YOLO
import cv2
import yaml
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import argparse
import shutil
import random


def polygon_to_mask(polygon_points, img_shape):
    """将多边形点转换为mask"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(polygon_points) < 6:  # 至少需要3个点
        return mask
    pts = np.array(polygon_points).reshape(-1, 2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_mask_iou(mask1, mask2):
    """计算两个mask的IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_mask_iou_matrix(pred_masks, gt_masks):
    """
    向量化计算预测mask和GT mask之间的IoU矩阵
    利用numpy广播避免Python循环,速度提升显著
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    # 将mask列表堆叠为3D数组: (N, H, W)
    pred_stack = np.stack(pred_masks, axis=0).astype(np.bool_)   # (n_pred, H, W)
    gt_stack = np.stack(gt_masks, axis=0).astype(np.bool_)       # (n_gt, H, W)

    # 利用广播计算交集: (n_pred, 1, H, W) & (1, n_gt, H, W) -> (n_pred, n_gt)
    intersection = np.logical_and(
        pred_stack[:, np.newaxis, :, :],   # (n_pred, 1, H, W)
        gt_stack[np.newaxis, :, :, :]      # (1, n_gt, H, W)
    ).sum(axis=(2, 3))                      # (n_pred, n_gt)

    # 并集 = A + B - 交集
    pred_area = pred_stack.sum(axis=(1, 2))[:, np.newaxis]  # (n_pred, 1)
    gt_area = gt_stack.sum(axis=(1, 2))[np.newaxis, :]      # (1, n_gt)
    union = pred_area + gt_area - intersection

    iou_matrix = np.where(union > 0, intersection / union, 0.0).astype(np.float32)
    return iou_matrix


def load_config(cfg_path):
    """加载配置文件,返回验证集路径列表和类别信息"""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    val_paths = cfg.get('val', [])
    if isinstance(val_paths, str):
        val_paths = [val_paths]
    
    names = cfg.get('names', {})
    nc = cfg.get('nc', len(names))
    
    return val_paths, names, nc


def find_labels_path(images_path):
    """根据images路径查找对应的labels路径"""
    images_path = Path(images_path)
    labels_path = str(images_path).replace('/images/', '/labels/').replace('/images', '/labels')
    labels_path = Path(labels_path)
    if labels_path.exists():
        return labels_path
    return None


def load_gt_labels(label_path, img_shape):
    """
    加载GT标签(mask)
    YOLO分割格式: cls x_center y_center width height x1 y1 x2 y2 ... xn yn
    返回: cls_list, mask_list (mask为像素坐标的多边形点)
    """
    cls_list = []
    mask_list = []
    
    if not label_path.exists():
        return cls_list, mask_list
    
    h, w = img_shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            cls = int(parts[0])
            
            # 只处理有多边形分割数据的标签,跳过纯bbox标签
            if len(parts) <= 5:
                continue
            
            cls_list.append(cls)
            mask_points_norm = [float(x) for x in parts[5:]]
            # 转换为像素坐标
            mask_points = []
            for i in range(0, len(mask_points_norm), 2):
                if i + 1 < len(mask_points_norm):
                    px = int(mask_points_norm[i] * w)
                    py = int(mask_points_norm[i + 1] * h)
                    mask_points.extend([px, py])
            mask_list.append(mask_points)
    
    return cls_list, mask_list


def match_predictions_mask(pred_cls, gt_cls, iou_matrix, iou_threshold):
    """
    基于mask IoU匹配预测和GT,返回TP/FP/FN索引
    使用贪心匹配,按IoU从大到小
    """
    n_pred = len(pred_cls)
    n_gt = len(gt_cls)
    
    tp_pred_idx = set()
    tp_gt_idx = set()
    fp_pred_idx = set()
    fn_gt_idx = set()
    
    if n_pred == 0:
        fn_gt_idx = set(range(n_gt))
        return tp_pred_idx, fp_pred_idx, fn_gt_idx, tp_gt_idx
    
    if n_gt == 0:
        fp_pred_idx = set(range(n_pred))
        return tp_pred_idx, fp_pred_idx, fn_gt_idx, tp_gt_idx
    
    # 获取所有IoU值并排序
    iou_pairs = []
    for i in range(n_pred):
        for j in range(n_gt):
            if pred_cls[i] == gt_cls[j] and iou_matrix[i, j] >= iou_threshold:
                iou_pairs.append((iou_matrix[i, j], i, j))
    
    iou_pairs.sort(reverse=True)
    
    # 贪心匹配
    matched_pred = set()
    matched_gt = set()
    
    for iou_val, pred_idx, gt_idx in iou_pairs:
        if pred_idx not in matched_pred and gt_idx not in matched_gt:
            tp_pred_idx.add(pred_idx)
            tp_gt_idx.add(gt_idx)
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    # 未匹配的预测为FP
    fp_pred_idx = set(range(n_pred)) - matched_pred
    # 未匹配的GT为FN
    fn_gt_idx = set(range(n_gt)) - matched_gt
    
    return tp_pred_idx, fp_pred_idx, fn_gt_idx, tp_gt_idx


def _add_title_bar(panel: np.ndarray, title: str, bar_height: int = 36) -> np.ndarray:
    """在图片顶部添加标题栏"""
    h, w = panel.shape[:2]
    bar = np.full((bar_height, w, 3), 40, dtype=np.uint8)
    cv2.putText(bar, title, (10, bar_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def draw_prediction_panel(img: np.ndarray, pred_masks_xy, pred_cls, pred_conf,
                          tp_pred_idx, fp_pred_idx, names,
                          conf_threshold: float = 0.25,
                          pred_ious: dict = None,
                          gt_cls=None, gt_masks_points=None,
                          fn_gt_idx=None) -> np.ndarray:
    """
    绘制预测结果面板: 展示模型的全部表现
    - TP=蓝色半透明 (正确检测)
    - FP=红色半透明 (误检)
    - FN=黄色虚线轮廓 (漏检:GT有但模型没检测到)
    """
    panel = img.copy()
    h, w = img.shape[:2]
    overlay = panel.copy()
    if pred_ious is None:
        pred_ious = {}
    if fn_gt_idx is None:
        fn_gt_idx = set()

    # 先绘制漏检区域 (FN) — 黄色半透明,用虚线轮廓标出
    if gt_cls is not None and gt_masks_points is not None:
        for i, (cls, mask_pts) in enumerate(zip(gt_cls, gt_masks_points)):
            if i not in fn_gt_idx:
                continue
            if len(mask_pts) < 6:
                continue
            gt_mask = polygon_to_mask(mask_pts, img.shape)
            if gt_mask.sum() == 0:
                continue
            # 黄色半透明填充
            overlay[gt_mask == 1] = [0, 200, 255]  # 黄色 (BGR)
            # 虚线轮廓
            pts = np.array(mask_pts).reshape(-1, 2).astype(np.int32)
            for k in range(0, len(pts), 2):
                pt1 = tuple(pts[k])
                pt2 = tuple(pts[(k + 1) % len(pts)])
                cv2.line(panel, pt1, pt2, (0, 180, 255), 2, cv2.LINE_AA)
            # 标签
            ys, xs = np.where(gt_mask == 1)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"{cls} FN(missed)"
                cv2.putText(panel, label, (cx - 50, cy + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(panel, label, (cx - 50, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1, cv2.LINE_AA)

    # 绘制预测结果 (TP / FP)
    if pred_masks_xy is not None:
        for i, (cls, conf) in enumerate(zip(pred_cls, pred_conf)):
            if conf < conf_threshold:
                continue
            if i >= len(pred_masks_xy):
                continue

            poly = pred_masks_xy[i]
            if len(poly) < 6:
                continue

            binary = polygon_to_mask(poly, img.shape)
            if binary.sum() == 0:
                continue

            if i in fp_pred_idx:
                color = [0, 0, 255]   # 红色 — 误检
                tag = "FP"
            elif i in tp_pred_idx:
                color = [255, 100, 0]  # 蓝色 — 正确
                tag = "TP"
            else:
                continue

            overlay[binary == 1] = color
            ys, xs = np.where(binary == 1)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"{cls} {conf:.2f} {tag}"
                cv2.putText(panel, label, (cx - 40, cy + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(panel, label, (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.45, panel, 0.55, 0, panel)
    return _add_title_bar(panel, "Prediction (TP=Blue, FP=Red, FN=Yellow)")


def draw_gt_panel(img: np.ndarray, gt_masks_points, gt_cls,
                  tp_gt_idx, names) -> np.ndarray:
    """
    绘制真实标签面板: 只展示已正确匹配的GT(全绿)
    """
    panel = img.copy()
    overlay = panel.copy()

    for i, (cls, mask_pts) in enumerate(zip(gt_cls, gt_masks_points)):
        if i not in tp_gt_idx:
            continue
        if len(mask_pts) < 6:
            continue
        gt_mask = polygon_to_mask(mask_pts, img.shape)
        if gt_mask.sum() == 0:
            continue

        color = [0, 200, 0]    # 深绿 — 已匹配
        overlay[gt_mask == 1] = color
        ys, xs = np.where(gt_mask == 1)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"{cls}"
            cv2.putText(panel, label, (cx - 40, cy + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, label, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.45, panel, 0.55, 0, panel)
    return _add_title_bar(panel, "Ground Truth (Matched Only)")


def make_comparison_image(img: np.ndarray, pred_panel: np.ndarray,
                          gt_panel: np.ndarray,
                          iou_thresh: float = 0.0,
                          conf_thresh: float = 0.0) -> np.ndarray:
    """
    横向拼接: 原图 | 预测图 | GT图
    pred_panel / gt_panel 已包含标题栏,这里给原图也加上再统一高度.
    """
    title = f"Original"
    if iou_thresh > 0 or conf_thresh > 0:
        title += f"  (IoU>{iou_thresh}, Conf>{conf_thresh})"
    img_panel = _add_title_bar(img, title)
    h_target = max(img_panel.shape[0], pred_panel.shape[0], gt_panel.shape[0])

    def _resize_to_height(src: np.ndarray) -> np.ndarray:
        if src.shape[0] == h_target:
            return src
        scale = h_target / src.shape[0]
        return cv2.resize(src, (int(src.shape[1] * scale), h_target),
                          interpolation=cv2.INTER_LINEAR)

    panels = [_resize_to_height(img_panel),
              _resize_to_height(pred_panel),
              _resize_to_height(gt_panel)]

    sep = np.full((h_target, 3, 3), 180, dtype=np.uint8)

    parts = []
    for p in panels:
        parts.append(sep)
        parts.append(p)
    parts.append(sep)
    return np.hstack(parts)


def calculate_metrics(tp, fp, fn):
    """计算Precision, Recall, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def main(model_path, cfg_path, iou_threshold=0.5, conf_threshold=0.25,
         output_dir="runs/test/results", limit=0, save_images=True):
    """主函数"""
    print("=" * 70)
    print("验证集测试(分割任务 - 仅Mask评估)")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    print(f"配置文件: {cfg_path}")
    print(f"IoU阈值: {iou_threshold}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"输出目录: {output_dir}")
    print(f"保存图片: {save_images}")
    print("=" * 70)
    
    # 1. 加载配置
    val_paths, names, nc = load_config(cfg_path)
    print(f"\n验证集路径数量: {len(val_paths)}")
    for i, path in enumerate(val_paths):
        print(f"  [{i+1}] {path}")
    print(f"类别数量: {nc}")
    print(f"类别名称: {names}")
    
    # 2. 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 3. 创建输出目录(先删除旧数据)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"已删除旧输出目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
    
    # 4. 收集所有验证集图片
    all_images = []
    for val_path in val_paths:
        val_path = Path(val_path)
        if val_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                all_images.extend(val_path.glob(ext))
                all_images.extend(val_path.glob(ext.upper()))
    
    print(f"\n总共找到 {len(all_images)} 张验证图片")
    
    # 限制测试数量(随机采样)
    if limit > 0 and len(all_images) > limit:
        random.seed(42)
        all_images = random.sample(all_images, limit)
        print(f"已随机采样 {limit} 张图片")
    
    # 5. 统计变量
    total_tp = 0
    total_fp = 0
    total_fn = 0
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # Mask IoU统计
    total_mask_iou = 0.0
    mask_iou_count = 0
    class_mask_iou = defaultdict(list)
    
    # 6. 预处理:收集有效图片路径和对应的GT标签
    print("\n预处理GT标签...")
    valid_items = []  # (img_path, gt_cls, gt_masks_points)
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        labels_path = find_labels_path(img_path.parent)
        if labels_path is None:
            continue
        label_file = labels_path / (img_path.stem + ".txt")
        gt_cls, gt_masks_points = load_gt_labels(label_file, img.shape)
        if len(gt_cls) > 0:
            valid_items.append((img_path, gt_cls, gt_masks_points))
        del img  # 释放内存

    print(f"有效图片: {len(valid_items)} 张")

    # 7. 逐张推理(显存友好)
    print("\n开始推理...")
    valid_paths = [str(item[0]) for item in valid_items]

    for idx, img_path in enumerate(valid_paths):
        _, gt_cls, gt_masks_points = valid_items[idx]

        if (idx + 1) % 100 == 0:
            print(f"  处理进度: {idx + 1}/{len(valid_items)}")
        
        # 逐张推理
        result = model.predict(img_path, conf=conf_threshold,
                               iou=iou_threshold, verbose=False)[0]

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        n_gt = len(gt_cls)

        # 获取预测结果
        pred_cls = []
        pred_conf = []
        pred_masks_xy = None

        if result.boxes is not None:
            pred_cls = result.boxes.cls.cpu().numpy().tolist() if result.boxes.cls is not None else []
            pred_conf = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else []
        if result.masks is not None and result.masks.xy is not None:
            pred_masks_xy = [pts.flatten().tolist() for pts in result.masks.xy]

        pred_cls = [int(c) for c in pred_cls]
        n_pred = len(pred_cls)
        
        # 将GT mask转换为二值mask
        gt_masks_binary = []
        for mask_pts in gt_masks_points:
            if len(mask_pts) >= 6:
                gt_mask = polygon_to_mask(mask_pts, img.shape)
                gt_masks_binary.append(gt_mask)
            else:
                gt_masks_binary.append(np.zeros((h, w), dtype=np.uint8))
        
        # 将预测mask转换为二值mask — 使用多边形坐标(已映射回原图)
        pred_masks_binary = []
        if pred_masks_xy is not None:
            for poly in pred_masks_xy:
                if len(poly) >= 6:
                    pred_mask = polygon_to_mask(poly, img.shape)
                    pred_masks_binary.append(pred_mask)
                else:
                    pred_masks_binary.append(np.zeros((h, w), dtype=np.uint8))
        
        if n_pred > 0 and n_gt > 0 and len(pred_masks_binary) > 0 and len(gt_masks_binary) > 0:
            # 计算mask IoU矩阵
            iou_matrix = compute_mask_iou_matrix(pred_masks_binary, gt_masks_binary)
            
            # 匹配
            tp_pred_idx, fp_pred_idx, fn_gt_idx, tp_gt_idx = match_predictions_mask(
                pred_cls, gt_cls, iou_matrix, iou_threshold
            )
            
            # 计算匹配对的mask IoU,并记录每个TP预测的IoU
            pred_ious = {}  # pred_idx -> matched IoU
            for pred_i in tp_pred_idx:
                best_iou = 0
                best_gt_i = -1
                for gt_i in range(n_gt):
                    if iou_matrix[pred_i, gt_i] > best_iou and pred_cls[pred_i] == gt_cls[gt_i]:
                        best_iou = iou_matrix[pred_i, gt_i]
                        best_gt_i = gt_i
                
                if best_gt_i >= 0:
                    mask_iou_val = iou_matrix[pred_i, best_gt_i]
                    pred_ious[pred_i] = mask_iou_val
                    total_mask_iou += mask_iou_val
                    mask_iou_count += 1
                    class_mask_iou[pred_cls[pred_i]].append(mask_iou_val)
        
        else:
            tp_pred_idx = set()
            fp_pred_idx = set(range(n_pred)) if n_pred > 0 else set()
            fn_gt_idx = set(range(n_gt)) if n_gt > 0 else set()
            tp_gt_idx = set()
            pred_ious = {}
        
        # 更新统计
        total_tp += len(tp_pred_idx)
        total_fp += len(fp_pred_idx)
        total_fn += len(fn_gt_idx)
        
        # 更新类别统计
        for i in tp_pred_idx:
            class_stats[pred_cls[i]]['tp'] += 1
        for i in fp_pred_idx:
            class_stats[pred_cls[i]]['fp'] += 1
        for i in fn_gt_idx:
            class_stats[gt_cls[i]]['fn'] += 1
        
        # 只有存在漏检或误检时才保存三图对比
        if save_images and (len(fn_gt_idx) > 0 or len(fp_pred_idx) > 0):
            pred_panel = draw_prediction_panel(
                img, pred_masks_xy, pred_cls, pred_conf,
                tp_pred_idx, fp_pred_idx, names, conf_threshold,
                pred_ious,
                gt_cls, gt_masks_points, fn_gt_idx
            )
            gt_panel = draw_gt_panel(
                img, gt_masks_points, gt_cls,
                tp_gt_idx, names
            )
            comparison = make_comparison_image(img, pred_panel, gt_panel,
                                                iou_threshold, conf_threshold)

            # 文件名标注错误类型
            tags = []
            if len(fn_gt_idx) > 0:
                tags.append("FN")
            if len(fp_pred_idx) > 0:
                tags.append("FP")
            tag_str = "_".join(tags)
            save_path = comparison_dir / f"{Path(img_path).stem}_{tag_str}.jpg"
            cv2.imwrite(str(save_path), comparison)
    
    # 8. 计算并打印指标
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    # 整体指标
    precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_fn)
    avg_mask_iou = total_mask_iou / mask_iou_count if mask_iou_count > 0 else 0
    
    print(f"\n【整体融合验证集指标】")
    print(f"  总TP (正确检测): {total_tp}")
    print(f"  总FP (误检): {total_fp}")
    print(f"  总FN (漏检): {total_fn}")
    print(f"  Precision (精确率): {precision:.4f}")
    print(f"  Recall (召回率): {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  平均Mask IoU: {avg_mask_iou:.4f} (共{mask_iou_count}个匹配对)")
    
    # 每个类别的指标
    print(f"\n【每个类别指标】")
    print("-" * 90)
    print(f"{'类别':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MaskIoU':<12}")
    print("-" * 90)
    
    for cls_id in range(nc):
        stats = class_stats[cls_id]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        p, r, f1_val = calculate_metrics(tp, fp, fn)
        cls_name = names.get(cls_id, str(cls_id))
        
        # 计算该类别的平均mask IoU
        cls_mask_iou_list = class_mask_iou.get(cls_id, [])
        cls_avg_mask_iou = sum(cls_mask_iou_list) / len(cls_mask_iou_list) if cls_mask_iou_list else 0
        
        print(f"{cls_name:<15} {tp:<8} {fp:<8} {fn:<8} {p:<12.4f} {r:<12.4f} {f1_val:<12.4f} {cls_avg_mask_iou:<12.4f}")
    
    print("-" * 90)
    
    # 保存指标到文件
    metrics_file = output_dir / "metrics_summary.txt"
    with open(metrics_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("验证集测试结果汇总(分割任务 - 仅Mask评估)\n")
        f.write("=" * 70 + "\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"配置文件: {cfg_path}\n")
        f.write(f"IoU阈值: {iou_threshold}\n")
        f.write(f"置信度阈值: {conf_threshold}\n")
        f.write(f"验证集图片数量: {len(valid_items)}\n")
        f.write("\n")
        
        f.write("【整体融合验证集指标】\n")
        f.write(f"  总TP (正确检测): {total_tp}\n")
        f.write(f"  总FP (误检): {total_fp}\n")
        f.write(f"  总FN (漏检): {total_fn}\n")
        f.write(f"  Precision (精确率): {precision:.4f}\n")
        f.write(f"  Recall (召回率): {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  平均Mask IoU: {avg_mask_iou:.4f} (共{mask_iou_count}个匹配对)\n")
        f.write("\n")
        
        f.write("【每个类别指标】\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'类别':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MaskIoU':<12}\n")
        f.write("-" * 90 + "\n")
        
        for cls_id in range(nc):
            stats = class_stats[cls_id]
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            p, r, f1_val = calculate_metrics(tp, fp, fn)
            cls_name = names.get(cls_id, str(cls_id))
            cls_mask_iou_list = class_mask_iou.get(cls_id, [])
            cls_avg_mask_iou = sum(cls_mask_iou_list) / len(cls_mask_iou_list) if cls_mask_iou_list else 0
            f.write(f"{cls_name:<15} {tp:<8} {fp:<8} {fn:<8} {p:<12.4f} {r:<12.4f} {f1_val:<12.4f} {cls_avg_mask_iou:<12.4f}\n")
        
        f.write("-" * 90 + "\n")
    
    print(f"\n指标已保存到: {metrics_file}")
    if save_images:
        print(f"三图对比保存在: {comparison_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证集测试(分割任务 - 仅Mask评估)")
    parser.add_argument("--model", type=str, default="runs/segment/p2_paper_zicai260720/v1_260720-3/weights/best.pt",
                        help="模型路径")
    parser.add_argument("--cfg", type=str, default="ultralytics/cfg/datasets/clean_v2_seg_zicai.yaml",
                        help="配置文件路径")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU阈值")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="置信度阈值")
    parser.add_argument("--output", type=str, default="runs/test/results_zicai260720",
                        help="输出目录")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制测试图片数量,0表示全部测试")
    parser.add_argument("--no_save", action="store_true", default=False,
                        help="不保存图片,仅输出评估结果")
    
    args = parser.parse_args()
    
    main(args.model, args.cfg, args.iou, args.conf, args.output, args.limit,
         save_images=not args.no_save)
