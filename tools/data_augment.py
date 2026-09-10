import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import shutil
import subprocess
import json
import random


def _apply_brightness_contrast(img):
    """随机亮度 + 对比度调整"""
    alpha = np.random.uniform(0.5, 1.5)  # 对比度
    beta = np.random.randint(-50, 50)    # 亮度
    result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return result

def _apply_color_jitter(img):
    """HSV 色彩抖动"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] += np.random.randint(-10, 10)       # 色调
    hsv[:, :, 1] *= np.random.uniform(0.5, 1.5)       # 饱和度
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] += np.random.randint(-50, 50)        # 明度
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def _apply_color_transform(img):
    """颜色变换:随机选择亮度对比度 或 色彩抖动"""
    if np.random.random() < 0.5:
        return _apply_brightness_contrast(img)
    else:
        return _apply_color_jitter(img)

def _apply_gaussian_noise(img):
    """高斯噪声"""
    std = np.random.randint(5, 26)
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def _apply_salt_pepper_noise(img):
    """椒盐噪声"""
    amount = np.random.uniform(0.001, 0.01)
    result = img.copy()
    # 盐噪声(白点)
    num_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    result[coords[0], coords[1]] = 255
    # 椒噪声(黑点)
    num_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    result[coords[0], coords[1]] = 0
    return result

def _apply_noise(img):
    """噪声注入:随机选择高斯噪声或椒盐噪声"""
    if np.random.random() < 0.5:
        return _apply_gaussian_noise(img)
    else:
        return _apply_salt_pepper_noise(img)

def _apply_gaussian_blur(img):
    """高斯模糊"""
    ksize = np.random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def _apply_motion_blur(img):
    """运动模糊"""
    ksize = np.random.randint(5, 16)
    if ksize % 2 == 0:
        ksize += 1  # 确保奇数
    # 随机角度
    angle = np.random.uniform(0, 360)
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for i in range(ksize):
        x = int(center + (i - center) * dx)
        y = int(center + (i - center) * dy)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

def _apply_sharpen(img):
    """锐化"""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)

def _apply_blur_sharpen(img):
    """模糊或锐化:随机选择"""
    r = np.random.random()
    if r < 0.4:
        return _apply_gaussian_blur(img)
    elif r < 0.8:
        return _apply_motion_blur(img)
    else:
        return _apply_sharpen(img)

# ==================== 几何变换(会修改坐标) ====================

def _transform_points(points, M):
    """对多边形点集应用仿射变换矩阵"""
    new_points = []
    for pt in points:
        px, py = pt[0], pt[1]
        new_pt = np.dot(M, [px, py, 1.0])
        new_points.append([new_pt[0], new_pt[1]])
    return new_points

def _clip_points(points_list, w, h):
    """将所有多边形坐标裁剪到 [0, w] × [0, h] 范围内,防止坐标越界"""
    return [
        [[max(0.0, min(w - 1e-9, pt[0])), max(0.0, min(h - 1e-9, pt[1]))] for pt in pts]
        for pts in points_list
    ]

def _rotate_image_and_points(img, points_list, h, w, angle=None):
    """旋转图像和坐标点"""
    if angle is None:
        angle = np.random.uniform(-30, 30)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    new_points_list = [_transform_points(pts, M) for pts in points_list]
    # 裁剪旋转后可能超出边界的坐标
    new_points_list = [[[max(0.0, min(w - 1e-9, pt[0])), 
                         max(0.0, min(h - 1e-9, pt[1]))] for pt in pts] 
                       for pts in new_points_list]
    return rotated, new_points_list, h, w

def _flip_image_and_points(img, points_list, h, w, flip_type=None):
    """翻转图像和坐标点"""
    if flip_type is None:
        flip_type = np.random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
    flipped = cv2.flip(img, flip_type)
    new_points_list = []
    for pts in points_list:
        new_pts = []
        for pt in pts:
            px, py = pt[0], pt[1]
            if flip_type == 1 or flip_type == -1:  # 水平翻转
                px = w - px
            if flip_type == 0 or flip_type == -1:  # 垂直翻转
                py = h - py
            new_pts.append([px, py])
        new_points_list.append(new_pts)
    return flipped, new_points_list, h, w

def _scale_image_and_points(img, points_list, h, w, scale=None, new_w=None, new_h=None):
    """缩放图像和坐标点"""
    if scale is None:
        scale = np.random.uniform(0.8, 1.2)
    if new_w is None:
        new_w = int(round(w * scale))
    if new_h is None:
        new_h = int(round(h * scale))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    new_points_list = []
    for pts in points_list:
        new_pts = [[pt[0] * scale, pt[1] * scale] for pt in pts]
        new_points_list.append(new_pts)
    # 裁剪缩放后可能超出边界的坐标
    new_points_list = [[[max(0.0, min(new_w - 1e-9, pt[0])), 
                         max(0.0, min(new_h - 1e-9, pt[1]))] for pt in pts] 
                       for pts in new_points_list]
    return scaled, new_points_list, new_w, new_h

def _crop_image_and_points(img, points_list, h, w):
    """随机裁剪并更新坐标"""
    crop_w = int(w * np.random.uniform(0.6, 0.9))
    crop_h = int(h * np.random.uniform(0.6, 0.9))
    x1 = np.random.randint(0, max(1, w - crop_w))
    y1 = np.random.randint(0, max(1, h - crop_h))
    cropped = img[y1:y1 + crop_h, x1:x1 + crop_w]
    # 调整坐标:将原始坐标减去裁剪起始点,然后裁剪到新图像边界内
    new_points_list = []
    for pts in points_list:
        new_pts = [[max(0.0, min(crop_w - 1e-9, pt[0] - x1)), 
                    max(0.0, min(crop_h - 1e-9, pt[1] - y1))] 
                   for pt in pts]
        new_points_list.append(new_pts)
    return cropped, new_points_list, crop_w, crop_h

def _translate_image_and_points(img, points_list, h, w):
    """随机平移"""
    dx = int(w * np.random.uniform(-0.1, 0.1))
    dy = int(h * np.random.uniform(-0.1, 0.1))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    new_points_list = [_transform_points(pts, M) for pts in points_list]
    # 裁剪平移后可能超出边界的坐标
    new_points_list = [[[max(0.0, min(w - 1e-9, pt[0])), 
                         max(0.0, min(h - 1e-9, pt[1]))] for pt in pts] 
                       for pts in new_points_list]
    return translated, new_points_list, h, w

def _is_within_bounds(points_list, w, h):
    """检查所有坐标是否都在 [0, w] × [0, h] 范围内"""
    for pts in points_list:
        for pt in pts:
            if pt[0] < 0 or pt[0] > w or pt[1] < 0 or pt[1] > h:
                return False
    return True

def _has_valid_annotation(points_list):
    """检查是否至少有一个有效标注(非空多边形)"""
    return any(len(pts) >= 3 for pts in points_list)

def _is_valid_result(points_list, w, h):
    """综合检查:坐标不越界 且 至少有一个有效标注"""
    return _is_within_bounds(points_list, w, h) and _has_valid_annotation(points_list)

def _apply_geometric(img, points_list, h, w):
    """随机应用多种几何变换(每种独立判断是否触发),如果变换导致目标越界或无效则跳过该变换"""
    # 水平翻转(50%概率)
    if np.random.random() < 0.5:
        new_img, new_pts, new_h, new_w = _flip_image_and_points(img, points_list, h, w)
        if _is_valid_result(new_pts, new_w, new_h):
            img, points_list, h, w = new_img, new_pts, new_h, new_w

    # 旋转(50%概率)
    if np.random.random() < 0.5:
        new_img, new_pts, new_h, new_w = _rotate_image_and_points(img, points_list, h, w)
        if _is_valid_result(new_pts, new_w, new_h):
            img, points_list, h, w = new_img, new_pts, new_h, new_w

    # 缩放(50%概率)
    if np.random.random() < 0.5:
        new_img, new_pts, new_w_sc, new_h_sc = _scale_image_and_points(img, points_list, h, w)
        if _is_valid_result(new_pts, new_w_sc, new_h_sc):
            img, points_list, h, w = new_img, new_pts, new_h_sc, new_w_sc

    # 裁剪(50%概率)
    if np.random.random() < 0.5:
        new_img, new_pts, new_w_cr, new_h_cr = _crop_image_and_points(img, points_list, h, w)
        if _is_valid_result(new_pts, new_w_cr, new_h_cr):
            img, points_list, h, w = new_img, new_pts, new_h_cr, new_w_cr

    # 平移(50%概率)
    if np.random.random() < 0.5:
        new_img, new_pts, new_h, new_w = _translate_image_and_points(img, points_list, h, w)
        if _is_valid_result(new_pts, new_w, new_h):
            img, points_list, h, w = new_img, new_pts, new_h, new_w

    return img, points_list, h, w


def img_transformation(root_dir, save_dir, aug_num=4, max_samples=10):
    """
    离线数据增强:对图像应用随机的像素级变换和几何变换,并同步更新 JSON 标注中的多边形坐标.

    目录结构:
        root_dir/img/   — 原始图片
        root_dir/json/  — 原始 JSON 标注
        save_dir/img/   — 增强后的图片
        save_dir/json/  — 增强后的 JSON 标注

    每张图片生成 4 个经随机增强的版本,包含:
        - 颜色变换(亮度、对比度、色彩抖动)
        - 噪声注入(高斯噪声、椒盐噪声)
        - 模糊/锐化(高斯模糊、运动模糊、锐化)
        - 几何变换(旋转、翻转、缩放、裁剪、平移)
    """

    img_dir = os.path.join(root_dir, "img")
    json_dir = os.path.join(root_dir, "json")
    out_img_dir = os.path.join(save_dir, "img")
    out_json_dir = os.path.join(save_dir, "json")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_json_dir, exist_ok=True)

    img_names = sorted(os.listdir(img_dir))
    if max_samples is not None:
        img_names = img_names[:max_samples]

    for img_name in tqdm(img_names, desc="augmentation"):
        img_path = os.path.join(img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]
        json_path = os.path.join(json_dir, base_name + ".json")

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告:无法读取图片 {img_path},跳过")
            continue

        # 读取 JSON 标注
        if not os.path.exists(json_path):
            print(f"警告:找不到标注文件 {json_path},跳过")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # 提取所有多边形的点集列表
        points_list = [shape.get("points", []) for shape in json_data.get("shapes", [])]

        for aug_idx in range(aug_num):
            aug_img = img.copy()
            aug_points_list = [pts.copy() for pts in points_list]
            aug_w, aug_h = json_data.get("imageWidth", img.shape[1]), json_data.get("imageHeight", img.shape[0])

            # 1. 随机像素级增强(独立判断每种是否触发)
            if np.random.random() < 0.5:
                aug_img = _apply_color_transform(aug_img)
            if np.random.random() < 0.5:
                aug_img = _apply_noise(aug_img)
            if np.random.random() < 0.25:
                aug_img = _apply_blur_sharpen(aug_img)

            # 2. 几何变换(100% 触发,多种变换可叠加)
            aug_img, aug_points_list, aug_w, aug_h = _apply_geometric(aug_img, aug_points_list, aug_h, aug_w)
            # 最终强制裁剪,确保坐标不会越界
            aug_points_list = _clip_points(aug_points_list, aug_w, aug_h)

            # 构建新的 JSON 数据
            new_json = json.loads(json.dumps(json_data))  # 深拷贝
            new_json["imageWidth"] = aug_w
            new_json["imageHeight"] = aug_h
            new_json["imagePath"] = f"{base_name}_aug{aug_idx}{ext}"
            for i, shape in enumerate(new_json.get("shapes", [])):
                if i < len(aug_points_list):
                    shape["points"] = aug_points_list[i]

            # 保存增强后的图片和 JSON
            out_img_name = f"{base_name}_aug{aug_idx}{ext}"
            out_json_name = f"{base_name}_aug{aug_idx}.json"
            cv2.imwrite(os.path.join(out_img_dir, out_img_name), aug_img)
            with open(os.path.join(out_json_dir, out_json_name), "w", encoding="utf-8") as f:
                json.dump(new_json, f, indent=2, ensure_ascii=False)

    print(f"增强完成,共生成 {len(os.listdir(out_json_dir))} 个标注文件")


def img_copy_paste(root_dir, save_dir):
    # todo
    pass


if __name__ == "__main__":
    # img_transformation()
    pass