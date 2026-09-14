# Ultralytics YOLO11 P2 检测与实例分割

纯检测采集数据可使用独立的 `pipeline_detect/` 流程：它保留数组输入/输出、原子数据 staging、`images/labels`、`img/yolo` 或图片/标签直存的数据处理、累计注册、阶段状态、训练恢复、P2 ONNX 校验、固定抽样测试和 Excel 三表更新，但不执行 PIDNet、实例分割或预标注。该流程复用原 `PIPELINE_MODEL_TRAIN_ROOT`、实验输出目录与 `pipeline_registry/datasets.yaml` 的 YOLO 数据来源；历史来源按 train/val 单独校验。数据准备完成和训练开始前都会输出图片、标签实例及类别数量汇总；Excel 新记录会紧邻真实数据行追加，并记录批次时间、累计耗时和标签统计。旧状态缺少类别统计时，`report` 会自动从共享注册表重建。`PIPELINE_INCLUDE_NEGATIVE_SAMPLES` 默认保留空 TXT 负样本，设为 `false` 时跳过；`PIPELINE_IGNORE_CLASS_IDS` 默认将类别 3 区域涂黑并从训练标签移除。使用说明见 `pipeline_detect/README.md`。

## 矩形训练与任意尺寸 ONNX

detect/segment 的训练和验证支持 `imgsz=[height, width]`，例如本项目使用 `imgsz=[800, 1280]`。两个方向都会按最大 stride 自动对齐；Mosaic、仿射增强、AutoBatch 和验证均保留矩形宽高比。训练进度条会以 `高度x宽度` 显示矩形输入尺寸，例如 `800x1280`。

板端原始输出模型可使用固定的任意外部尺寸导出。若高或宽不能被 32 整除，导出器会在 ONNX 图首部加入居中 Pad，推理脚本会自动扣除该补边并还原坐标和 mask：

```bash
python tools/export.py --task detect --weight /path/to/best.pt --output /path/to/detect.onnx --imgsz 800 1280
python tools/export.py --task segment --weight /path/to/best.pt --output /path/to/segment.onnx --imgsz 800 1280
python tools/inference.py --onnx-model /path/to/detect.onnx --imgsz 800 1280
python tools/inference.py --onnx-model /path/to/segment.onnx --imgsz 800 1280
```

命令行尺寸顺序始终是 `HEIGHT WIDTH`。`640x360` 宽高输入应写为 `--imgsz 360 640`，ONNX 内部会自动补为 `384x640`。

复用现有标准数据集、不重新处理数据，并执行 1280x800 训练、ONNX 导出、500 张验证集测试和报告：

```bash
cd pipeline/scripts
./0812_all_0812_1280x800.sh
```

该脚本使用批次名 `V013_20260812_1280x800`，默认顺序为 `train → export → test → report`。也可以把 `train`、`export`、`test`、`report` 或 `status` 作为第一个参数单独执行。

新增数据批次脚本可通过 `PIPELINE_ENABLE_DATA_UPDATE` 决定是否在训练前执行 `prepare` 并注册新增数据；设为 `false` 时只读取已有 `pipeline_registry/datasets.yaml`，不扫描或校验脚本中的输入、输出数据目录。该开关不影响预标注。`PIPELINE_ENABLE_PRELABEL=true` 时，脚本会在训练、导出、测试和预标注完成后写入报告；设为 `false` 可安全跳过预标注。`PIPELINE_PRELABEL_DIR` 支持历史单路径或 JSON 路径列表；每项是一个直接存放图片的目录，预标注不会递归扫描子目录。批次总览的总耗时累计已执行流水线阶段，失败后的人工检查与重启间隔不计入。

`pipeline_detect/scripts/0827_detect.sh` 用于处理并注册 20260825 新增检测数据，从 V018 `best.pt` 微调训练 V019，随后导出 ONNX、执行外部带标签测试并写入报告。脚本默认执行 `all`，也可传入 `prepare`、`train`、`export`、`test`、`report` 或 `status`；全新 V019 运行目录不能直接执行 `test`，因为该命令要求当前运行状态已经登记 `export_detect.onnx`。

本仓库基于 Ultralytics，主要用于清洁机器人场景的 YOLO11 目标检测与实例分割训练。当前版本增加了 P2 小目标特征层、同一份图片使用不同任务标签目录，以及面向板端部署的原始输出 ONNX 导出。

本文只说明 YOLO 的环境安装、数据准备、训练参数、训练验证、ONNX 导出和 `tools/` 常用工具。

## 1. 环境安装

推荐使用仓库约定的 `ult` Conda 环境：

```bash
conda create -n ult python=3.12 -y
conda activate ult

cd /data/users/USER/gitee/ultralytics
pip install -e .
pip install onnx onnxruntime
```

PyTorch 应与服务器 NVIDIA 驱动和 CUDA 版本匹配。如果当前环境已经能正常使用 GPU，不要随意重新安装 PyTorch。

检查环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "from ultralytics import YOLO; print('Ultralytics import OK')"
```

服务器通过 SSH 或 tmux 运行且没有图形界面时，建议设置：

```bash
export MPLBACKEND=Agg
```

这样可以避免 Matplotlib 尝试加载 Qt 图形后端。

## 2. 数据准备

### 2.1 按日期增量更新已处理子数据集

当已修改训练数据根目录内任一子数据集的 LabelMe 标注时，在训练数据根目录执行。唯一参数为更新起始时间，脚本遍历所有子数据集内直接含有 `json/` 的 split，只重新生成该时间及之后修改的 JSON 对应样本的 `images`、`labels_detect`、`labels_segment` 和 `Seg`；`img_src`、`json` 保持不变。每个更新 JSON 都必须能在同一 split 的 `img_src/` 中找到同名图片，否则打印失败标签并继续处理其他样本。

```bash
cd /data/datasets/training_data
python /data/users/USER/gitee/ultralytics/tools/update_data.py "2026-08-14 08:30:00"
```

可将 `update_data.py` 复制到训练数据根目录后直接执行 `python update_data.py "2026-08-14 08:30:00"`。脚本是独立脚本，不依赖仓库 `pipeline` 包；其类别映射和输出格式与 Pipeline 保持一致。没有 `val/` 时会自动跳过；如存在 `train_neg` 等其他含 `json/` 的 split 也会处理，结束时打印更新、失败和跳过统计。

### 2.2 合并零散 YOLO 检测框

`tools/merge_detect_objects.py` 递归处理 YOLO 检测标签，只合并同一类别的检测框；不同类别即使相交也不会合并。判断标准为检测框相交，或检测框最短距离不超过 20 像素。每次生成最小外接框后会再次判断，直到没有检测框可以继续合并。输入 `labels_detect` 后默认读取同级 `img/`（兼容 `images/`）以获取图片尺寸，并输出同级 `labels_detect_merge/`：

```bash
python tools/merge_detect_objects.py /data/yolodetect/labels_detect
```

可通过 `--distance 30` 调整阈值，或通过 `--image-dir`、`--output-dir` 指定非标准路径。输出目录已存在时，显式传入 `--overwrite` 可删除旧输出后重新生成；原始检测标签不会修改。

### 2.3 推荐目录结构

同一份图片可以同时用于检测和实例分割，不需要复制两份 `images`：

```text
dataset/
├── train/
│   ├── images/
│   ├── labels_detect/
│   └── labels_segment/
└── val/
    ├── images/
    ├── labels_detect/
    └── labels_segment/
```

- `images`：JPG、PNG 等训练图片。
- `labels_detect`：YOLO 检测框标签。
- `labels_segment`：YOLO 实例分割多边形标签。
- 图片与标签通过不带扩展名的文件名一一对应。
- 原始数据处理只复制同名图片与 LabelMe JSON 的交集；没有同名配对的图片或 JSON 会跳过并写入日志统计。
- 没有目标的图片可以使用同名空 TXT，作为负样本参与训练。

### 2.2 标签格式

检测标签每行格式：

```text
class_id center_x center_y width height
```

实例分割标签每行格式：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

所有坐标均归一化到 `0～1`。

当前清洁机器人数据通常使用以下类别：

- 检测：`paper=0`、`liquid=1`、`metal=2`。
- 实例分割：只分割 `paper=0`。

检测和分割可以使用不同类别数量，但必须分别使用对应的标签目录和数据集 YAML。

### 2.3 检测数据集 YAML

例如 `ultralytics/cfg/datasets/clean_detect.yaml`：

```yaml
path: /data/datasets/clean_robot
train: train/images
val: val/images

labels_dir: labels_detect

nc: 3
names:
  0: paper
  1: liquid
  2: metal
```

### 2.4 实例分割数据集 YAML

例如 `ultralytics/cfg/datasets/clean_segment.yaml`：

```yaml
path: /data/datasets/clean_robot
train: train/images
val: val/images

labels_dir: labels_segment

nc: 1
names:
  0: paper
```

### 2.5 组合多个数据目录

`path` 必须是一个字符串，不能写成列表。多个训练或验证来源应写在 `train`、`val` 中：

```yaml
path: /data/datasets

train:
  - clean_v1/train/images
  - clean_v2/train/images
  - negative_samples/images

val:
  - clean_v1/val/images
  - clean_v2/val/images

labels_dir: labels_detect

nc: 3
names:
  0: paper
  1: liquid
  2: metal
```

也可以直接填写多个绝对图片目录。无论采用哪种方式，`path` 本身都不能是列表，否则会出现 `Path` 接收到 `list` 的错误。

修改图片或标签后，如果仍然读到旧统计结果，可以删除数据目录旁的 `*.cache` 文件，让 Ultralytics 重新扫描数据。

## 3. P2 模型配置

本仓库使用以下模型结构：

| 任务 | 模型 YAML | 类别示例 |
|---|---|---|
| 目标检测 | `ultralytics/cfg/models/11/yolo11s_p2.yaml` | paper、liquid、metal |
| 实例分割 | `ultralytics/cfg/models/11/yolo11s-p2-seg.yaml` | paper |

两个模型都使用 P2、P3、P4、P5 四层输出，对应 stride：

```text
[4, 8, 16, 32]
```

P2 特征层适合较小目标，但会增加显存、计算量和输出张量数量。

## 4. 训练参数配置

日常训练主要修改以下参数：

| 参数 | 常用设置 | 说明 |
|---|---:|---|
| `data` | 检测或分割 YAML | 决定图片、标签目录、类别数量和类别名称。 |
| `epochs` | 50～300 | 总训练轮数；测试流程可以先用 5～10。 |
| `batch` | 检测 12、分割 10 | 24 GB 显卡常用值；显存不足时优先降低。 |
| `imgsz` | 640 | 输入尺寸；增大会提高小目标分辨率，也会明显增加显存。 |
| `device` | 0 | 使用的 GPU 编号；CPU 使用 `cpu`。 |
| `workers` | 16 | 数据加载进程数；CPU 或网络存储较慢时适当降低。 |
| `amp` | False | 自动混合精度；确认模型和硬件稳定后可改为 True。 |
| `cos_lr` | True | 使用余弦学习率调度。 |
| `warmup_epochs` | 3 | 训练开始阶段的预热轮数。 |
| `close_mosaic` | 10 | 最后 10 个 Epoch 关闭 Mosaic，稳定最终收敛。 |
| `optimizer` | auto | 默认自动选择；没有明确实验依据时不必修改。 |
| `project` | 按任务设置 | 实验输出根目录。 |
| `name` | 按版本设置 | 当前实验名称，例如 `v008_20260807`。 |
| `exist_ok` | False | False 时避免直接复用同名目录，降低覆盖历史结果的风险。 |
| `resume` | False/True | 微调使用 False；恢复中断训练使用 True。 |

### 4.1 显存不足时怎么调

建议按以下顺序调整：

1. 降低 `batch`，例如检测从 12 改为 8，分割从 10 改为 6。
2. 仍然不足时再降低 `imgsz`，例如从 640 改为 576 或 512。
3. 确认 GPU 上没有其他训练或推理进程占用显存。
4. 模型验证和推理时使用较小 batch。

`batch=-1` 可以让 Ultralytics 自动估算 batch，但正式版本训练建议记录并固定实际 batch，便于不同版本公平对比。

### 4.2 微调与断点恢复

从上一版本 `best.pt` 开始新一轮微调：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11s_p2.yaml")
model.load("runs/detect/previous/weights/best.pt")
model.train(data="ultralytics/cfg/datasets/clean_detect.yaml", epochs=200, resume=False)
```

恢复被中断的同一次训练：

```python
from ultralytics import YOLO

model = YOLO("runs/detect/current/weights/last.pt")
model.train(resume=True)
```

注意：

- `best.pt + resume=False` 是新实验微调，从 Epoch 1 开始。
- `last.pt + resume=True` 是恢复优化器、学习率和 Epoch 状态。
- 不要把“加载 best.pt 微调”和“resume=True”混在一起。
- 恢复训练时应保持模型结构、数据配置和主要训练参数一致。

## 5. 目标检测训练

推荐在 `tools/train.py` 中为每个正式版本增加一个清晰的训练函数。检测示例：

```python
from ultralytics import YOLO


def train_detect() -> None:
    """训练带 P2 检测头的 YOLO11s 模型."""
    model = YOLO("ultralytics/cfg/models/11/yolo11s_p2.yaml")
    model.load("runs/detect/previous/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_detect.yaml",
        epochs=200,
        batch=12,
        imgsz=640,
        device=0,
        workers=16,
        amp=False,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        project="runs/detect/yolo11s_p2",
        name="v008_20260807",
        exist_ok=False,
        resume=False,
    )


if __name__ == "__main__":
    train_detect()
```

运行：

```bash
conda activate ult
cd /data/users/USER/gitee/ultralytics
python tools/train.py
```

## 6. 实例分割训练

分割示例：

```python
from ultralytics import YOLO


def train_segment() -> None:
    """训练带 P2 检测头的 YOLO11s 实例分割模型."""
    model = YOLO("ultralytics/cfg/models/11/yolo11s-p2-seg.yaml")
    model.load("runs/segment/previous/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_segment.yaml",
        epochs=200,
        batch=10,
        imgsz=640,
        device=0,
        workers=16,
        amp=False,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        project="runs/segment/yolo11s_p2",
        name="v008_20260807",
        exist_ok=False,
        resume=False,
    )


if __name__ == "__main__":
    train_segment()
```

可以写入 `tools/train.py`，也可以参考 `tools/train_seg.py` 中已有的分割训练示例。

## 7. 训练结果

一次标准训练目录通常包含：

```text
runs/<task>/<project>/<name>/
├── args.yaml
├── results.csv
├── results.png
├── best_metrics.json
└── weights/
    ├── best.pt
    └── last.pt
```

- `best.pt`：验证集 Fitness 最好的权重，通常用于测试、导出和下一版本微调。
- `last.pt`：最后一个 Epoch 的完整 checkpoint，主要用于断点恢复。
- `results.csv`：每个 Epoch 的损失和验证指标。
- `best_metrics.json`：最佳 Epoch、总体指标和逐类别指标。

正式训练结束后至少检查：

- 最佳 Epoch 是否过早，判断训练是否充分。
- 检测任务的 P、R、mAP50、mAP50-95。
- 分割任务的 Box 和 Mask P、R、mAP50、mAP50-95。
- 每个类别的独立指标，避免总体平均值掩盖弱类别。
- 训练集和验证集图片、标签数量是否符合预期。

## 8. 验证、测试和预测

### 8.1 Python API 验证

```python
from ultralytics import YOLO

model = YOLO("runs/segment/yolo11s_p2/v008_20260807/weights/best.pt")
metrics = model.val(
    data="ultralytics/cfg/datasets/clean_segment.yaml",
    imgsz=640,
    batch=1,
    device=0,
    plots=False,
)
print(metrics.results_dict)
print(metrics.summary())
```

### 8.2 工具脚本

```bash
python tools/val.py
python tools/test.py --help
python tools/predict.py
```

- `tools/val.py`：使用 Ultralytics 验证接口验证 `.pt` 模型。
- `tools/test.py`：实例分割验证集测试、Mask IoU 匹配、指标统计和可视化对比。
- `tools/predict.py`：使用 `.pt` 模型进行快速预测。

这些脚本中部分路径是固定示例，运行前需要修改为本次的模型和数据路径。

## 9. ONNX 导出

### 9.1 P2 实例分割导出

`tools/export.py` 中的 `v2_yolov11_seg_p2` 用于导出板端友好的 P2 分割原始输出：

```bash
python -c "from tools.export import v2_yolov11_seg_p2; v2_yolov11_seg_p2('runs/segment/xxx/weights/best.pt', 'runs/segment/xxx/weights/best_raw_p2.onnx')"
```

输出顺序为：

```text
P2: box、score、mask_coeff
P3: box、score、mask_coeff
P4: box、score、mask_coeff
P5: box、score、mask_coeff
proto
```

### 9.2 P2 检测导出

`tools/export.py` 中的 `v3_yolov11_detect_p2` 用于导出 P2 检测原始输出。先修改函数内的 `pt_path`，再把文件末尾调用改为：

```python
if __name__ == "__main__":
    v3_yolov11_detect_p2()
```

然后运行：

```bash
python tools/export.py
```

检测输出顺序为：

```text
P2: box、score
P3: box、score
P4: box、score
P5: box、score
```

导出的模型保留卷积原始输出，不在 ONNX 内执行 DFL 解码、Sigmoid、网格还原、NMS 和实例 Mask 合成。这些步骤需要由板端后处理实现。

默认使用 Opset 11，并将 ONNX IR version 调整为 6，以兼容目标板工具链。导出结束后脚本会调用 ONNX Checker 校验模型。

## 10. ONNX 推理

检测 ONNX 推理（自动识别 P2/P3）：

```bash
python tools/inference.py \
  --onnx-model runs/detect/xxx/weights/best_raw_p2_detect.onnx \
  --yaml ultralytics/cfg/datasets/clean_detect.yaml \
  --output-dir runs/inference/detect_v008 \
  --imgsz 800 1280 \
  --conf 0.25 \
  --iou 0.7
```

直接推理无标签图片目录：

```bash
python tools/inference.py \
  --onnx-model runs/detect/xxx/weights/best_raw_p2_detect.onnx \
  --img-dir /data/test_images \
  --output-dir runs/inference/images
```

实例分割使用同一个入口，任务由 ONNX 输出节点自动识别：

```bash
python tools/inference.py --onnx-model runs/segment/xxx/weights/best_raw.onnx --help
python tools/inference_video.py --help
```

- `tools/inference.py`：统一的检测/实例分割 ONNX 入口，兼容 P2-P5 和 P3-P5 原始输出；被 Pipeline 调用时会自动解析仓库内的 `pipeline` 模块。
- `tools/inference_video.py`：视频逐帧 ONNX 推理与结果视频保存。

## 11. `tools/` 常用工具

下表只列与 YOLO 训练、数据准备、验证和部署直接相关的工具：

| 文件 | 用途 |
|---|---|
| `train.py` | P2 检测和分割训练版本入口。 |
| `train_seg.py` | 历史实例分割训练示例。 |
| `val.py` | `.pt` 模型快速验证。 |
| `test.py` | 实例分割 Mask 指标测试和预测/GT 对比图。 |
| `predict.py` | `.pt` 模型快速预测。 |
| `export.py` | P2 检测、P2 实例分割板端 ONNX 导出。 |
| `inference.py` | 检测/分割统一 ONNX 推理、评估和可视化，自动识别 P2/P3。 |
| `inference_video.py` | ONNX 视频推理。 |
| `deal_data.py` | LabelMe 数据整理、COCO 转 LabelMe、ignore 处理、数据划分、YOLO 标签转换和成对重命名。 |
| `data_augment.py` | 图像与多边形标签同步数据增强。 |

### 11.0 图片与标签成对重命名

`rename_image_label_pairs` 只处理图片目录与标签目录第一层中不带扩展名同名的文件，并保留各自扩展名。按原文件名排序后，图片和标签会同步改为 `new_name_000000`、`new_name_000001` 等形式；没有同名配对的文件不会修改。

```bash
python -c "from tools.deal_data import rename_image_label_pairs; rename_image_label_pairs('/data/images', '/data/labels', 'sample')"
```

### 11.1 YOLO 类别 ID 置换

`remap_yolo_class_ids` 直接修改标签目录第一层的 `.txt` 文件，仅替换每行首列的类别 ID。映射同步生效，因此可安全交换两个类别：

```bash
python -c "from tools.deal_data import remap_yolo_class_ids; remap_yolo_class_ids('/data/labels', {0: 1, 1: 0})"
```

### 11.2 COCO 转 LabelMe

CVAT 导出 COCO 格式标注后，可调用 `coco_to_labelme` 为每张图片生成同名的 LabelMe JSON。COCO 多边形分割会转换为 LabelMe `polygon`；只有 `bbox` 的标注会转换为 `rectangle`。函数不复制图片，`imagePath` 保留 COCO 中图片文件名。

```bash
python -c "from tools.deal_data import coco_to_labelme; coco_to_labelme('/data/cvat/annotations.json', '/data/cvat/labelme_json')"
```
| `video_frame_sampler.py` | 按相似度从视频抽帧，每个视频保存到独立目录，并过滤明显模糊帧。 |
| `synthetic_yolo_generator.py` | 按需更新实体 OBJ，并将 paper、metal 实体目标贴到背景后输出 YOLO 检测数据。 |
| `bag2mp4.py` | 将相机 BAG 数据转换为 MP4。 |

### 11.2 BAG 转 MP4

`bag2mp4.py` 在每个文件转换前自动检查 BAG 能否打开、是否含有彩色流，以及能否读取首个彩色帧；转换时显示持续刷新的回放进度条。默认预检等待 30 秒，转换期间连续 120 秒没有新帧且回放位置不前进时，会说明可能的文件或回放异常、删除不完整的临时 MP4 并跳过该文件，继续处理目录中的其他 BAG。所有新增保护功能均有默认值，通常无需添加参数：

```bash
python tools/bag2mp4.py "/data/input_bag" "/data/output_mp4"
```

脚本每次都会先检查输出目录中是否已有同名 `.mp4`。默认发现已有文件即明确提示其大小与路径并跳过，适合重复执行同一命令；只有传入 `--overwrite` 才会重新转换该 BAG。

只有在设备或磁盘较慢时才需要延长等待时间，例如：

```bash
python tools/bag2mp4.py "/data/input_bag" "/data/output_mp4" \
  --precheck-timeout 60 --stall-timeout 300
```

预检只能发现文件头、索引、彩色流或首段数据的明显问题；文件中后段的损坏会在转换期间由无进度超时检测并跳过。

### 11.3 视频抽帧

中文或包含空格的路径必须使用引号：

```bash
python tools/video_frame_sampler.py \
  "/data/包含空格和中文/输入视频.mp4" \
  "/data/输出图片" \
  0.98
```

三个参数依次为输入视频或目录、输出目录、相似度阈值。阈值越高，通常保留的图片越多。

### 11.4 YOLO 检测目标提取与背景合成

所有路径和参数都集中在脚本顶部的 `CONFIG`，无需传入命令行参数。源数据统一使用 `images/labels_detect` 目录结构；脚本只扫描源数据集的 `images`，不会把同一数据集中的 `Seg`、`labels_segment` 或其他图片目录误当成检测图片。源数据配置项既可以指向包含 `images` 的数据集根目录，也可以直接指向 `images` 目录。

首次运行或需要更新 OBJ 素材库时，将 `CONFIG["update_objects"]` 设为 `True`，填写多个 `source_datasets` 和统一的 `object_library`。脚本会先在同级 `.building` 临时目录中重建素材库；成功后把已有素材库改名为带时间戳的 `.backup_YYYYMMDD_HHMMSS` 目录，再启用新素材库并自动执行 MixUp。默认仅提取并合成纸张和金属；它们从 YOLO 检测框中使用 GrabCut 分离并保存透明 PNG、二值 mask 和来源元数据。液体不参与实例贴图，训练只使用真实采集并人工标注的液体数据：

```bash
python tools/synthetic_yolo_generator.py
```

OBJ 素材库中的 `objects/<类别ID>/*.png` 为纸张和金属的透明目标，`masks/<类别ID>/*.png` 为内部抠图 mask，`metadata.jsonl` 记录原始数据集、图片、标签和检测框。建议先人工抽查纸张和金属透明 PNG 的边缘质量；不满意的目标可以直接删除对应的 `objects/*.png`，MixUp 阶段会将其视为人工剔除并跳过，不要求同步修改 `metadata.jsonl` 或删除 mask。

后续不需要更新 OBJ 时，将 `CONFIG["update_objects"]` 保持为 `False`。脚本不会扫描原始训练集或执行 GrabCut，而是直接读取现有 `object_library`，然后根据 `background_datasets`、`output` 和 `num_images` 执行 MixUp：

```bash
python tools/synthetic_yolo_generator.py
```

背景配置项支持两种结构：目录下直接存放图片，或者标准的 `images` 子目录。目录下有直接图片时优先使用这些图片，不再强制查找 `images`；如果同目录的 `labels_detect` 中存在同名 YOLO TXT，则保留原检测框，否则按纯背景处理。最终输出为 `images/*.jpg + labels_detect/*.txt`，标签始终是 `class x_center y_center width height` 检测格式；新增目标框根据实际前景范围重新计算，不输出 LabelMe 或分割标签。输出文件沿用背景文件名并增加 MixUp 次数，例如 `IndoorSurfaceWood_test_000001_mixup_001.jpg` 和同名 `.txt`；同一背景再次使用时序号递增。脚本会拒绝覆盖非空 MixUp 输出目录。

透明液体不能从普通检测框中可靠 GrabCut；程序化液体容易产生与真实液体无关的固定边缘和高光伪特征，因此默认 `CONFIG["class_ids"]` 为 `[0, 2]`、`CONFIG["compose"]["liquid_class_ids"]` 为空。请用真实采集的液体图、不同光照/反光条件和困难负样本补充 liquid 类数据，而不是开启液体实例贴图。

合成阶段按轮次打乱并使用全部背景素材，使每张背景的使用次数尽量接近；默认每张图生成 4–8 个纸张或金属目标。若已放入目标的实际包围框面积达到画面的 20%，该图会停止继续贴入目标，避免大目标场景被强行塞满。可通过 `min_objects`、`max_objects` 和 `large_object_area_ratio` 调整这一策略。某个计划类别失败时只能更换同类别 OBJ 重试，不能用其他类别顶替。`summary.json` 中的 `pasted_label_counts` 仅统计本次新增目标并用于判断软均衡，`final_label_counts` 则包含背景原标签和新增标签；`available_background_counts`、`output_background_counts` 和 `background_usage_range` 用于核对背景分布。启动日志会显示实时类别均衡进度及单图目标数量策略。

脚本启动后会立即显示 `update_objects` 状态和关键路径。更新 OBJ 时，每个数据集都会显示开始与完成信息，进度条实时显示接受、失败和类别过滤数量；MixUp 进度条实时显示新增目标、重试、放置失败和大目标数量，同时提供处理速度与预计剩余时间。

## 12. 常见问题

### 12.1 数据集 `path` 报 list 类型错误

错误写法：

```yaml
path:
  - /data/dataset_a
  - /data/dataset_b
```

正确做法是让 `path` 保持字符串，把多个来源写入 `train`、`val`。

### 12.2 找不到标签

依次检查：

1. YAML 的 `labels_dir` 是否为 `labels_detect` 或 `labels_segment`。
2. 图片与 TXT 是否同名。
3. 检测和分割 YAML 是否使用了正确的标签目录。
4. 是否残留旧的 `.cache` 文件。

### 12.3 训练中断后如何继续

使用当前实验的 `weights/last.pt`：

```python
from ultralytics import YOLO

YOLO("runs/xxx/weights/last.pt").train(resume=True)
```

不要使用 `best.pt` 冒充断点恢复。`best.pt` 更适合新实验微调、验证和导出。

### 12.4 CUDA 显存不足

先通过 `nvidia-smi` 确认其他进程占用，再降低 `batch`。P2 分割模型通常比 P2 检测模型占用更多显存。

### 12.5 Headless 环境出现 Qt 后端错误

运行前设置：

```bash
export MPLBACKEND=Agg
```

验证时不需要曲线图片可以使用 `plots=False`。

### 12.6 ONNX 与 `.pt` 结果不一致

重点检查：

- LetterBox 的缩放比例和 padding 是否一致。
- P2～P5 输出顺序是否一致。
- DFL 解码、Sigmoid、网格坐标和 stride 是否正确。
- NMS 的置信度、IoU 和最大实例数是否一致。
- 分割 Mask 系数、Proto 和原图裁剪还原是否正确。

建议先使用 `tools/inference.py` 在 PC 上验证，再接入板端推理代码。

## 13. 官方代码同步

本仓库以 Ultralytics 官方仓库为上游，`upstream` 指向：

```text
https://github.com/ultralytics/ultralytics.git
```

2026-09-14 已将核心代码从本地 `v8.4.65` 基线同步至官方 `main` 的 `c33f13f5a8cfe7b01cc621279fd9caefff137374`，对应代码版本 `8.4.150`。本次同步保留 `pipeline/`、`pipeline_detect/`、`tools/`、本地数据集配置、矩形输入适配和训练日志增强；按项目约定未同步官方 `docs/` 文档目录。

后续更新前先确认工作区干净，再抓取并合并官方分支：

```bash
git -c http.sslBackend=openssl fetch upstream main
git merge --no-ff upstream/main
```

合并后应重点复核 `ultralytics/data/`、`ultralytics/engine/` 和 `ultralytics/utils/` 中的本地适配，不要直接用官方文件覆盖这些定制。
