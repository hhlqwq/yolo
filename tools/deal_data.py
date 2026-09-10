import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import shutil
import subprocess
import json
import random


def rename_image_label_pairs(image_dir, label_dir, new_name):
    """将两个目录中同名图片和标签成对重命名。

    Args:
        image_dir: 图片目录，只处理目录第一层的普通文件。
        label_dir: 标签目录，只处理目录第一层的普通文件。
        new_name: 新文件名前缀，例如 ``sample``。

    Returns:
        dict: 包含配对、图片独有和标签独有文件数量的统计结果。

    Raises:
        FileNotFoundError: 输入目录不存在时抛出。
        ValueError: 前缀非法或同一目录存在同名不同扩展名文件时抛出。
        FileExistsError: 新文件名会覆盖未参与本次改名的文件时抛出。
    """
    image_dir = os.path.abspath(image_dir)
    label_dir = os.path.abspath(label_dir)
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError("图片目录或标签目录不存在。")
    if not new_name or new_name != os.path.basename(new_name):
        raise ValueError("new_name 必须是非空文件名前缀，不能包含路径。")

    def collect_files(directory):
        """按不带扩展名的文件名收集目录第一层普通文件。"""
        files = {}
        for file_name in sorted(os.listdir(directory)):
            path = os.path.join(directory, file_name)
            if not os.path.isfile(path):
                continue
            stem, _ = os.path.splitext(file_name)
            if stem in files:
                raise ValueError(f"目录存在同名不同扩展名文件: {directory}/{stem}")
            files[stem] = path
        return files

    images = collect_files(image_dir)
    labels = collect_files(label_dir)
    common_names = sorted(set(images) & set(labels))
    image_only = len(set(images) - set(labels))
    label_only = len(set(labels) - set(images))
    if not common_names:
        print("没有找到同名图片和标签文件。")
        return {"pairs": 0, "image_only": image_only, "label_only": label_only}

    rename_plan = []
    source_paths = set(images.values()) | set(labels.values())
    for index, old_name in enumerate(common_names):
        new_stem = f"{new_name}_{index:06d}"
        image_ext = os.path.splitext(images[old_name])[1]
        label_ext = os.path.splitext(labels[old_name])[1]
        rename_plan.extend([
            (images[old_name], os.path.join(image_dir, new_stem + image_ext)),
            (labels[old_name], os.path.join(label_dir, new_stem + label_ext)),
        ])

    for _, target in rename_plan:
        if os.path.exists(target) and target not in source_paths:
            raise FileExistsError(f"目标文件已存在，已停止以避免覆盖: {target}")

    temporary_plan = []
    for index, (source, target) in enumerate(rename_plan):
        temporary = f"{source}.__rename_tmp_{index:06d}"
        if os.path.exists(temporary):
            raise FileExistsError(f"临时文件已存在，请先处理: {temporary}")
        os.rename(source, temporary)
        temporary_plan.append((temporary, target))
    for temporary, target in temporary_plan:
        os.rename(temporary, target)

    summary = {
        "pairs": len(common_names),
        "image_only": image_only,
        "label_only": label_only,
    }
    print(
        f"重命名完成: 配对 {summary['pairs']} 对，图片未配对 {image_only} 个，"
        f"标签未配对 {label_only} 个。"
    )
    return summary


def remap_yolo_class_ids(label_dir, class_mapping):
    """直接替换一个 YOLO 标签目录中的类别 ID。

    Args:
        label_dir: 包含 YOLO ``.txt`` 标签的目录，只处理目录第一层文件。
        class_mapping: 原类别 ID 到新类别 ID 的映射，例如 ``{0: 1, 1: 0}``。

    Returns:
        dict: 包含处理文件数和替换标签行数的统计结果。

    Raises:
        FileNotFoundError: 标签目录不存在时抛出。
        ValueError: 映射或标签首列不是非负整数时抛出。
    """
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    if not isinstance(class_mapping, dict) or not class_mapping:
        raise ValueError("class_mapping 必须是非空字典，例如 {0: 1, 1: 0}。")
    if any(
        not isinstance(source, int) or not isinstance(target, int) or source < 0 or target < 0
        for source, target in class_mapping.items()
    ):
        raise ValueError("class_mapping 的键和值必须是非负整数。")

    changed_files = 0
    changed_lines = 0
    for file_name in sorted(os.listdir(label_dir)):
        if not file_name.lower().endswith(".txt"):
            continue
        path = os.path.join(label_dir, file_name)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        output_lines = []
        file_changed = False
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if not fields:
                output_lines.append("")
                continue
            try:
                class_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(f"标签类别ID不是整数: {path}:{line_number}") from exc
            if class_id in class_mapping:
                fields[0] = str(class_mapping[class_id])
                file_changed = True
                changed_lines += 1
            output_lines.append(" ".join(fields))
        if file_changed:
            with open(path, "w", encoding="utf-8") as file:
                file.write("\n".join(output_lines) + "\n")
            changed_files += 1

    summary = {"files": changed_files, "lines": changed_lines}
    print(f"类别置换完成: 修改 {changed_files} 个文件，替换 {changed_lines} 行标签。")
    return summary

def scale_and_center(input_path, output_dir=None, scales=(3/4, 1/2, 1/4, 1/8), bg_color=(114, 114, 114)):
    """
    将图像内容缩放到指定比例,并居中放在与原图相同尺寸的画布上.

    :param input_path:  原始图片路径
    :param output_dir:  输出目录(None 表示原图目录)
    :param scales:      缩放比例元组
    :param bg_color:    背景色,BGR 元组,例如 (255,255,255) 白色,(0,0,0) 黑色
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        print(f"错误:无法读取图片 {input_path}")
        return

    # 获取原图尺寸和通道数
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3
    print(f"原图尺寸: {w} x {h}, 彩色: {is_color}")

    # 2. 准备输出目录
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]   # 如 .jpg

    # 3. 对每个比例进行处理
    for scale in scales:
        # 计算缩小后的尺寸
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        # 防止尺寸为 0(极小图)
        if new_w < 1 or new_h < 1:
            print(f"警告:比例 {scale} 导致尺寸为 0,跳过")
            continue

        # 缩放图像(缩小使用 INTER_AREA 效果最佳)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建与原图尺寸相同的画布(指定背景色)
        if is_color:
            canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
        else:
            # 灰度图,背景色取第一个通道值(若 bg_color 是彩色则转灰度)
            bg_val = bg_color[0] if isinstance(bg_color, tuple) else bg_color
            canvas = np.full((h, w), bg_val, dtype=np.uint8)

        # 计算居中偏移量
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2

        # 将缩放后的图像贴到画布上
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # 构造输出文件名,例如 "image_0.75.jpg"
        scale_str = str(scale).replace('/', '-')
        output_name = f"{base_name}_{scale_str}{ext}"
        output_path = os.path.join(output_dir, output_name)

        # 保存
        cv2.imwrite(output_path, canvas)
        print(f"已保存: {output_path}  (画布尺寸: {w}x{h}, 内容尺寸: {new_w}x{new_h})")


def letterbox_and_save(input_path, output_path=None, target_size=640, color=(114, 114, 114)):
    """
    将输入图片转换为 target_size x target_size 的正方形(letterbox 处理),并保存.

    :param input_path:  原始图片路径
    :param output_path: 输出路径(若为 None,则在原图目录生成以 _letterbox 结尾的文件)
    :param target_size: 目标尺寸(边长),默认 640
    :param color:       填充颜色(BGR),默认灰色 (114,114,114)
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        print(f"❌ 错误:无法读取图片 {input_path}")
        return None

    h, w = img.shape[:2]
    print(f"原图尺寸: {w} x {h}")

    # 2. 计算缩放比例(长边缩放到 target_size)
    scale = target_size / max(h, w)          # 等比例缩放因子
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    print(f"缩放后尺寸(不加填充): {new_w} x {new_h}")

    # 3. 缩放图片(使用线性插值,更平滑)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 4. 创建目标大小的画布(灰色背景)
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    # 5. 计算居中偏移并粘贴
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # 6. 确定最终输出路径(智能处理)
    if output_path is None:
        # 默认:在原图目录生成以 _letterbox 结尾的文件
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_letterbox{ext}"
    else:
        # 如果 output_path 是一个已存在的目录,或末尾带有路径分隔符,则作为目录处理
        if os.path.isdir(output_path) or output_path.endswith(('/', '\\')):
            os.makedirs(output_path, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(input_path))
            output_path = os.path.join(output_path, f"{base}_letterbox{ext}")
        else:
            # 否则视为完整文件路径,确保其所在目录存在
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            # 如果文件名没有扩展名,默认添加 .jpg
            if not os.path.splitext(output_path)[1]:
                output_path += ".jpg"

    # 7. 保存(可调整 JPEG 质量)
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✅ 已保存预处理图像: {output_path} (尺寸 {target_size}x{target_size})")
    print(f"   → 内容实际尺寸: {new_w} x {new_h},周围填充了灰边 (padding)")
    return output_path


def copy_jpg_by_json(json_dir, jpg_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for file_name in tqdm(os.listdir(json_dir), desc="copy"):
        if not file_name.endswith(".json"):
            continue

        name = os.path.splitext(file_name)[0]

        src_jpg = os.path.join(jpg_dir, name + ".jpg")

        if os.path.exists(src_jpg):
            shutil.copy(src_jpg, save_dir)
        else:
            print(f"not found: {src_jpg}")
    print(len(os.listdir(save_dir)))    # subprocess.run(f"ls -l {save_dir} | wc -l", shell=True)
    

def json2yolo(json_dir, save_dir, class_mapping=None):
    """
    将 LabelMe/Label Studio 风格的 JSON 标注文件转换为 YOLO 分割标注格式.
    生成的 YOLO 标注文件格式(每行一个多边形): class_id x1 y1 x2 y2 ... xn yn
    坐标已归一化到 [0, 1] 范围.
    :param json_dir:      包含 JSON 标注文件的目录
    :param save_dir:      输出 YOLO 标注文件的目录
    :param class_mapping: 类别名称到 ID 的映射字典,例如 {"metal": 0, "paper": 1}
                          若为 None,则自动从所有 JSON 文件中收集类别并分配 ID(按字母排序)
    """
    
    os.makedirs(save_dir, exist_ok=True)

    # 如果未提供类别映射,则自动从数据中收集所有类别
    if class_mapping is None:
        labels_set = set()
        for file_name in os.listdir(json_dir):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label", "")
                if label:
                    labels_set.add(label)
        # 按字母排序,确保类别ID稳定
        class_mapping = {name: idx for idx, name in enumerate(sorted(labels_set))}
        print(f"自动检测到 {len(class_mapping)} 个类别: {class_mapping}")

    for file_name in tqdm(os.listdir(json_dir), desc="json2yolo"):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth", 1)
        img_h = data.get("imageHeight", 1)

        # 输出文件名:与 JSON 同名,扩展名为 .txt
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(save_dir, base_name + ".txt")

        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            if label not in class_mapping:
                continue  # 跳过未知类别

            class_id = class_mapping[label]
            points = shape.get("points", [])

            if not points:
                continue

            # 归一化坐标:x / img_w, y / img_h
            normalized_coords = []
            for pt in points:
                x, y = pt[0], pt[1]
                nx = x / img_w
                ny = y / img_h
                normalized_coords.extend([nx, ny])

            # 构建 YOLO 分割行:class_id x1 y1 x2 y2 ...
            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized_coords)
            lines.append(line)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

    print(f"转换完成,共生成 {len(os.listdir(save_dir))} 个标注文件")


def divide_train_data(src_dir, dst_dir, ratio=0.8):
    # 划分训练集、验证集
    os.makedirs(dst_dir, exist_ok=True)
    files = os.listdir(src_dir)
    selected = random.sample(files, int(len(files) * ratio))
    for f in tqdm(selected, desc="divide..."):
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    print(len(os.listdir(dst_dir)))


def according_img_move_json(img_dir, json_src_dir, json_dst_dir):
    os.makedirs(json_dst_dir, exist_ok=True)
    for f in tqdm(os.listdir(img_dir), desc="according_img_move_json"):
        name = os.path.splitext(f)[0]
        json_src = os.path.join(json_src_dir, name + ".json")
        if os.path.exists(json_src):
            shutil.move(json_src, json_dst_dir)
        else:
            print(f"not found: {json_src}")
    print(len(os.listdir(json_dst_dir))) 


def according_img_copy_json(img_dir, json_src_dir, json_dst_dir):
    os.makedirs(json_dst_dir, exist_ok=True)
    for f in tqdm(os.listdir(img_dir), desc="according_img_move_json"):
        name = os.path.splitext(f)[0]
        json_src = os.path.join(json_src_dir, name + ".json")
        if os.path.exists(json_src):
            shutil.copy(json_src, json_dst_dir)
        else:
            print(f"not found: {json_src}")
    print(len(os.listdir(json_dst_dir))) 


def according_json_copy_img(json_src_dir, img_dir, img_dst_dir):
    os.makedirs(img_dst_dir, exist_ok=True)
    for f in tqdm(os.listdir(json_src_dir), desc="according_img_move_json"):
        name = os.path.splitext(f)[0]
        img_src = os.path.join(img_dir, name + ".jpg")
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst_dir)
        else:
            print(f"not found: {img_src}")
    print(len(os.listdir(img_dst_dir))) 

def deal_ignore(img_dir, json_dir, save_dir, ignore_label="ignore"):
    """
    根据 LabelMe JSON 中的 ignore 标签,在原图上将 ignore 区域涂黑.
    将 label 为 ignore_label 的多边形区域填充为黑色 (0, 0, 0).

    :param img_dir:      原始图片目录
    :param json_dir:     LabelMe JSON 标注文件目录
    :param save_dir:     输出处理后图片的保存目录
    :param ignore_label: ignore 标签名称,默认 "ignore"
    """
    os.makedirs(save_dir, exist_ok=True)

    # 先扫描所有 JSON 文件,收集全部类别
    all_labels = set()
    for file_name in os.listdir(json_dir):
        if not file_name.endswith(".json"):
            continue
        json_path = os.path.join(json_dir, file_name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            if label:
                all_labels.add(label)
    print(f"全部类别 ({len(all_labels)} 个): {sorted(all_labels)}")

    skip_count = 0
    for file_name in tqdm(os.listdir(json_dir), desc="deal_ignore"):
        if not file_name.endswith(".json"):
            continue

        base_name = os.path.splitext(file_name)[0]

        # 查找对应图片(尝试常见扩展名)
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            skip_count += 1
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            skip_count += 1
            continue

        h, w = img.shape[:2]

        # 读取 JSON 标注
        json_path = os.path.join(json_dir, file_name)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 遍历 shapes,找到 ignore 标注,直接在原图上涂黑
        ignore_count = 0
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            if label != ignore_label:
                continue

            points = shape.get("points", [])
            if not points:
                continue

            shape_type = shape.get("shape_type", "polygon")
            pts = np.array(points, dtype=np.int32)

            if shape_type == "rectangle" and len(points) == 2:
                # 矩形:两个点(左上+右下),涂黑矩形区域
                cv2.rectangle(img, tuple(pts[0]), tuple(pts[1]), (0, 0, 0), -1)
                ignore_count += 1
            elif len(points) >= 3:
                # 多边形:用 fillPoly 填充
                cv2.fillPoly(img, [pts], (0, 0, 0))
                ignore_count += 1

        # 保存处理后的图片(保持与原图相同的扩展名)
        out_ext = os.path.splitext(img_path)[1]
        out_path = os.path.join(save_dir, base_name + out_ext)
        cv2.imwrite(out_path, img)

        if ignore_count > 0:
            print(f"  ignore: {base_name}{out_ext} ({ignore_count} 个区域)")

    print(f"deal_ignore 完成,共处理 {len(os.listdir(save_dir))} 个图片文件")


def generate_detection_box_labels(
    root_dir="/data/users/hailong.he/datasets/opc_clean/",
    class_mapping=None
):
    """
    将 root_dir 下各子目录中 train/val 的 json 多边形分割标注
    转换为 YOLO 检测框格式标签,保存到对应 labels 目录下.

    目录结构:
        root_dir/
        ├── subdir1/
        │   ├── train/
        │   │   ├── json/      ← 原始 LabelMe JSON 标注(多边形)
        │   │   └── labels/    ← 输出 YOLO 检测框 txt
        │   └── val/
        │       ├── json/
        │       └── labels/
        └── subdir2/
            ├── train/
            │   ├── json/
            │   └── labels/
            └── val/
                ├── json/
                └── labels/

    :param root_dir:      数据集根目录
    :param class_mapping: 类别名称到 ID 的映射字典,默认 {"paper":0, "liquid":1, "metal":2}
    """
    if class_mapping is None:
        class_mapping = {"paper": 0, "liquid": 1, "metal": 2}

    # 遍历 root_dir 下所有子目录
    sub_dirs = sorted(os.listdir(root_dir))
    print(f"子目录: {sub_dirs}")

    total_generated = 0
    for sub_dir in sub_dirs:
        sub_path = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue

        # 遍历 train 和 val
        for split in ["train", "val"]:
            split_path = os.path.join(sub_path, split)
            if not os.path.isdir(split_path):
                continue

            json_dir = os.path.join(split_path, "json")
            labels_dir = os.path.join(split_path, "labels")

            if not os.path.isdir(json_dir):
                print(f"  跳过: {json_dir} 不存在")
                continue

            os.makedirs(labels_dir, exist_ok=True)

            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
            if not json_files:
                print(f"  {split_path}: 无 JSON 文件")
                continue

            print(f"处理: {sub_dir}/{split} ({len(json_files)} 个文件)")

            converted = 0
            skipped = 0
            for file_name in tqdm(json_files, desc=f"  {sub_dir}/{split}"):
                json_path = os.path.join(json_dir, file_name)
                base_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(labels_dir, base_name + ".txt")

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                img_w = data.get("imageWidth", 1)
                img_h = data.get("imageHeight", 1)

                lines = []
                for shape in data.get("shapes", []):
                    label = shape.get("label", "")
                    if label not in class_mapping:
                        continue

                    points = shape.get("points", [])
                    if not points:
                        continue

                    # 从多边形坐标计算最小外接矩形 (bounding box)
                    xs = [pt[0] for pt in points]
                    ys = [pt[1] for pt in points]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    # 转为 YOLO 检测框格式:cx, cy, w, h(归一化到 [0,1])
                    w_box = x_max - x_min
                    h_box = y_max - y_min
                    cx = (x_min + x_max) / 2.0
                    cy = (y_min + y_max) / 2.0

                    cx_n = cx / img_w
                    cy_n = cy / img_h
                    w_n = w_box / img_w
                    h_n = h_box / img_h

                    class_id = class_mapping[label]
                    lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

                # 无论是否有标注,都生成 txt 文件(无标注则为空文件,作为负样本)
                with open(output_path, "w", encoding="utf-8") as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")
                converted += 1
                if not lines:
                    skipped += 1

            print(f"  → 转换 {converted} 个, 其中空标签(负样本) {skipped} 个")
            total_generated += converted

    print(f"\n完成！共生成 {total_generated} 个检测框标签文件.")

def seg2detect(work_dir, class_mapping=None):
    """
    将一个目录下的 LabelMe JSON 分割标注转换为 YOLO 检测框格式标签.

    目录结构要求:
        work_dir/
        ├── json/      ← LabelMe JSON 多边形分割标注
        └── labels/    ← 输出 YOLO 检测框 txt (自动创建)

    每个 JSON 中的多边形 → 最小外接矩形 → YOLO 检测框: cls cx cy w h (归一化)

    :param work_dir:      工作目录路径 (包含 json/ 子目录的上层目录)
    :param class_mapping: 类别名到 ID 的映射,默认 {"paper":0, "liquid":1, "metal":2}
    """
    if class_mapping is None:
        class_mapping = {"paper": 0, "liquid": 1, "metal": 2}

    json_dir = os.path.join(work_dir, "json")
    if not os.path.isdir(json_dir):
        print(f"[Error] json 目录不存在: {json_dir}")
        return

    labels_dir = os.path.join(work_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    print(f"[seg2detect] 工作目录: {work_dir}")
    print(f"[seg2detect] json 文件数: {len(json_files)}")
    print(f"[seg2detect] 类别映射: {class_mapping}")

    converted, skipped = 0, 0
    for file_name in tqdm(json_files, desc="seg2detect"):
        json_path = os.path.join(json_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(labels_dir, base_name + ".txt")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth", 1)
        img_h = data.get("imageHeight", 1)

        lines = []
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            if label not in class_mapping:
                continue

            points = shape.get("points", [])
            if not points:
                continue

            # 多边形外接矩形 → YOLO 检测框
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            w_box = x_max - x_min
            h_box = y_max - y_min
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0

            cx_n = cx / img_w
            cy_n = cy / img_h
            w_n = w_box / img_w
            h_n = h_box / img_h

            class_id = class_mapping[label]
            lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        with open(output_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        converted += 1
        if not lines:
            skipped += 1

    print(f"[seg2detect] 完成！转换 {converted} 个, 空标签(负样本) {skipped} 个")
    print(f"[seg2detect] labels 输出目录: {labels_dir}")

def extract_frames(video_path, output_dir, frame_interval=None):
    """
    从视频中抽取帧并保存为图片.

    Args:
        video_path: 输入视频文件路径
        output_dir:  输出图片目录
        frame_interval: 每隔多少帧抽取一帧
    """
    # 检查视频文件是否存在
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 创建输出目录
    basepath = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join(output_dir, basepath)
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    # 获取视频基本参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    extract_num = int(total_frames / frame_interval)

    print(f"视频信息: {total_frames} 帧, {fps:.2f} fps, 时长 {duration:.2f} 秒, 分辨率 {width}x{height}, 预计抽 {extract_num} 张")

    # 确定抽帧策略
    if frame_interval is None:
        # 默认每 30 帧抽一帧
        frame_interval = 30
        print("未指定间隔,默认每 30 帧抽取一帧")
    print(f"间隔: {frame_interval} 帧抽取一帧")

    # 开始抽帧
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or saved_count > extract_num:
            break

        if frame_count % frame_interval == 0:
            # 构造文件名,用帧序号填充8位数字
            filename = f"{basepath}_{saved_count:08d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            print(f"已保存: {filename}")

        frame_count += 1

    cap.release()
    print(f"抽帧完成！共抽取 {saved_count} 张图片,保存在: {output_dir}")


def coco_to_labelme(coco_json_path, output_dir):
    """将 CVAT 导出的 COCO 标注转换为按图片保存的 LabelMe JSON 文件。

    每张 COCO 图片生成一个同名 JSON。多边形分割标注转换为 LabelMe
    ``polygon``；没有有效多边形的标注回退为 COCO ``bbox`` 对应的
    LabelMe ``rectangle``。不支持的 RLE 分割标注会被跳过。

    Args:
        coco_json_path: CVAT 导出的 COCO 标注 JSON 文件路径。
        output_dir: LabelMe JSON 输出目录。

    Returns:
        dict: 转换统计，包含图片数、生成形状数、矩形数和跳过标注数。

    Raises:
        FileNotFoundError: COCO 标注文件不存在时抛出。
        ValueError: COCO JSON 缺少必需字段或字段类型错误时抛出。
    """
    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(f"COCO 标注文件不存在: {coco_json_path}")

    with open(coco_json_path, "r", encoding="utf-8-sig") as file:
        coco_data = json.load(file)

    images = coco_data.get("images")
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON 中 images 和 annotations 必须为列表。")

    category_names = {
        category.get("id"): category.get("name", str(category.get("id")))
        for category in categories
        if "id" in category
    }
    annotations_by_image = {}
    for annotation in annotations:
        image_id = annotation.get("image_id")
        annotations_by_image.setdefault(image_id, []).append(annotation)

    os.makedirs(output_dir, exist_ok=True)
    polygon_count = 0
    rectangle_count = 0
    skipped_count = 0

    for image in tqdm(images, desc="coco_to_labelme"):
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not file_name or not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            skipped_count += len(annotations_by_image.get(image_id, []))
            print(f"跳过缺少文件名或尺寸的图片记录: {image}")
            continue

        shapes = []
        for annotation in annotations_by_image.get(image_id, []):
            label = category_names.get(annotation.get("category_id"))
            if label is None:
                skipped_count += 1
                print(f"跳过未知类别的标注: annotation_id={annotation.get('id')}")
                continue

            segmentation = annotation.get("segmentation")
            group_id = annotation.get("id")
            has_polygon = False
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                        continue
                    points = [[polygon[index], polygon[index + 1]] for index in range(0, len(polygon), 2)]
                    shapes.append({
                        "label": label,
                        "points": points,
                        "group_id": group_id,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    })
                    polygon_count += 1
                    has_polygon = True

            if has_polygon:
                continue

            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                skipped_count += 1
                continue
            x, y, box_width, box_height = bbox
            if box_width <= 0 or box_height <= 0:
                skipped_count += 1
                continue
            shapes.append({
                "label": label,
                "points": [[x, y], [x + box_width, y + box_height]],
                "group_id": group_id,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            })
            rectangle_count += 1

        image_name = os.path.basename(file_name)
        output_name = os.path.splitext(image_name)[0] + ".json"
        output_path = os.path.join(output_dir, output_name)
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": int(height),
            "imageWidth": int(width),
        }
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(labelme_data, file, ensure_ascii=False, indent=2)

    summary = {
        "images": len(images),
        "polygons": polygon_count,
        "rectangles": rectangle_count,
        "skipped_annotations": skipped_count,
    }
    print(
        "COCO 转 LabelMe 完成: "
        f"图片 {summary['images']} 张, 多边形 {summary['polygons']} 个, "
        f"矩形 {summary['rectangles']} 个, 跳过标注 {summary['skipped_annotations']} 个。"
    )
    return summary


def rename_with_index(dir_path, prefix=""):
    files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    for i, f in enumerate(tqdm(files), start=1):
        name, ext = os.path.splitext(f)
        new_name = f"{prefix}{i:08d}{ext}"
        os.rename(os.path.join(dir_path, f), os.path.join(dir_path, new_name))

if __name__ == "__main__":
   rename_with_index(
       "/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/mixup_indoor_260825/IndoorSurfaceDataset/test/carpet",
       "IndoorSurfaceCarpet_test_"
   )
   