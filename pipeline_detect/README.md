# 纯检测 Pipeline

`pipeline_detect` 是原 `pipeline/` 的纯 YOLO 检测入口：保留运行锁、配置快照、阶段状态、日志、训练恢复、AutoBatch、模型历史、P2 ONNX 校验、指定目录测试以及中心 Excel 三表更新；不执行实例分割、PIDNet、预标注或累计验证集抽样测试。

日常运行只需在服务器仓库根目录执行：

```bash
./pipeline_detect/scripts/0820_detect.sh all
```

脚本中的 `PIPELINE_INPUT_DIR` 和 `PIPELINE_OUTPUT_DIR` 均为 JSON 数组，按下标一一对应。输入目录兼容以下检测标注布局：

```text
scene/images/*.jpg + scene/labels/*.txt
scene/img/*.jpg    + scene/yolo/*.txt
scene/*.jpg        + scene/*.txt
scene/images/*     + scene/labels/<场景>.json  # 场景级 COCO 检测 JSON
```

COCO 输入按 `categories[].name` 映射类别，不直接使用可能从 1 开始或不连续的 `category_id`：`paper=0`、`liquid=1`、`metal=2`。`ignore` 标注与训练检测框从解析阶段分离，只用于在复制后的训练图片上涂黑对应 `bbox`，不会写入中间或最终 YOLO TXT。JSON `images[].file_name` 可以包含历史导出路径，流水线会使用其文件名在同场景 `images` 目录中唯一匹配；未登记在 COCO `images` 数组中的磁盘图片会统计并跳过。

每个输出目录以 staging 方式原子生成以下结构；转换前会完整校验 YOLO TXT 的类别、坐标、宽高以及图片/标签一一对应关系：

```text
dataset_detect/
├── train/images/
├── train/labels_detect/
├── val/images/
└── val/labels_detect/
```

每个输入输出映射完成后，日志会输出 `DATASET_OUTPUT`，包含 `prepared/reused` 状态、输出目录、训练/验证图片数量和逐类别检测实例数；全部映射完成后会输出 `DATASET_OUTPUT_SUMMARY`。

同名空 TXT 默认作为负样本保留。设置 `PIPELINE_INCLUDE_NEGATIVE_SAMPLES=false` 后会在处理阶段跳过这些样本，并输出 `DATA_NEGATIVE_FILTER` 统计；为避免静默混入负样本，该模式不会复用包含空 TXT 的既有输出目录。

`PIPELINE_IGNORE_CLASS_IDS` 默认值为 `[3]`。这些类别的 YOLO 框会在输出图片中涂黑，对应标签行会从 `labels_detect` 移除，因此不会作为训练类别；处理统计见 `DATA_IGNORE_PROCESS`。

训练开始前，`register_data` 会输出 `TRAINING_DATASET_SUMMARY`，包含累计训练/验证图片数、标签实例总数和 paper、liquid、metal 各类别的 train/val 数量。

累计来源使用 `PIPELINE_MODEL_TRAIN_ROOT/pipeline_registry/datasets.yaml` 的 `yolo.train` 和 `yolo.val`。更新数据时只会增补这两个列表，保留同一注册表中的 PIDNet 等其他区段；写入前自动备份并原子替换。历史来源按 train/val 单独校验，因此兼容不共享同一个数据集根目录的累计目录。设置 `PIPELINE_ENABLE_DATA_UPDATE=false` 时不会处理或修改数据，只读校验已有检测来源。

支持 `prepare`、`train`、`export`、`test`、`report`、`status` 和 `all`。`test` 只复用已有 ONNX 测试 `PIPELINE_LABELED_TEST_DIR` 指定目录；`all` 在训练和导出后也只执行该目录测试。任何命令都不读取累计验证集做抽样测试。其中 `report` 只补写报告，不重新训练；Excel 被占用时会保存 `reports/excel_pending.json`，关闭 Excel 后重新执行 `report` 即可。

## 外部带标签测试

设置 `PIPELINE_LABELED_TEST_ENABLED=true` 后，`test` 和 `all` 会对 `PIPELINE_LABELED_TEST_DIR` 下所有标准测试场景执行完整推理。测试场景必须为 `场景目录/images/* + 场景目录/labels/*`，标签支持 YOLO `.txt` 和 LabelMe `.json`；缺少同名标签的图片以及路径中包含 `[deprecated]` 的目录会自动跳过。

场景路径中任意一级目录名以 `[deprecated]` 开头时，该场景不会参与推理或任何精度统计。`PIPELINE_LABELED_TEST_EXCLUDE_KEYWORDS` 可设为 JSON 字符串数组；任一级目录名包含其中任一关键字时也会跳过，匹配不区分大小写。例如 `["1x1cm", "1-2cm", "2x2cm"]` 会排除对应的小尺寸场景。每次执行 `test` 都会先清空当前训练版本的 `results/`，再写入本次图片、场景指标和汇总报告，因此不会保留旧版的 `detect/`、`labeled_test/`、旧置信度或旧场景结果。NAS/SMB 目录因缓存导致待删除目录项提前消失时视为已经清理；返回“目录非空”时会自动重试。Windows 资源管理器占用的 `Thumbs.db` 可以保留，但其他旧结果文件仍须全部删除。连续重试仍失败则应检查是否有其他进程正在写入同一 `PIPELINE_TEST_OUTPUT_DIR`。

`PIPELINE_LABELED_TEST_CONFIDENCES` 是非空 JSON 数字数组，例如：

```bash
export PIPELINE_LABELED_TEST_ENABLED=true
export PIPELINE_LABELED_TEST_DIR="/path/to/02_annotation"
export PIPELINE_TEST_OUTPUT_DIR="/path/to/6_Result/$PIPELINE_RUN_NAME"
export PIPELINE_LABELED_TEST_CONFIDENCES='[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]'
export PIPELINE_LABELED_TEST_EXCLUDE_KEYWORDS='["1x1cm", "1-2cm", "2x2cm"]'
```

每档置信度只加载一次 ONNX，清单中的全部场景连续推理，再分别计算场景指标。P/R 使用已经过置信度过滤的全部预测，在 IoU=0.5 下固定匹配计算；因此汇总 TP、FP、FN 可以直接相加，不会再额外选择最佳 F1 阈值。输出结构如下：

```text
6_Result/<PIPELINE_RUN_NAME>/
├── overall_accuracy_summary.csv  # 每个置信度一行总体数据
├── class_accuracy_report.csv     # 各置信度的类别 P/R 集中报告
├── test_summary.md               # 总体和类别 P/R 摘要
└── confidence_10pct/
    ├── images/<场景>/*.png         # 原图、GT、预测竖向三视图
    ├── scene_accuracy_report_10pct.csv  # 每个场景目录一行，不展开类别
    └── metrics.txt                 # 本档文本指标摘要
```

推理程序用于进程间传递数据的临时 JSON 会在读取后立即删除，成功完成的完整结果目录中不保留 JSON 文件。

完整结果按 `PIPELINE_TEST_OUTPUT_DIR` 写入 `6_Result/<PIPELINE_RUN_NAME>/`。同时将关键报告覆盖镜像到模型版本目录的 `result/`，仅保留根汇总 CSV/Markdown，以及各置信度的 `accuracy_report.csv` 和 `metrics.txt`，不重复复制三视图：

```text
模型训练/<PIPELINE_RUN_NAME>/result/
├── test_summary.csv
└── test_summary.md
```

模型训练目录的 `result/` 不创建置信度子目录，也不保存逐场景或逐类别明细。所有 CSV 中的 Precision、Recall 固定保留三位小数；Markdown 中的百分比固定保留三位小数。

`test_summary.md` 会根据本次真实数据自动总结综合最平衡置信度、Precision/Recall 最弱类别、FP/FN 较多场景，以及置信度变化带来的 P/R 取舍；不会写入固定的模板结论。

测试集根目录中的日期或采集批次父目录不会复制到结果中，所有输出直接按 `TESTs...` 场景目录归档。如果不同父目录下存在同名场景，第一个保留原名，后续稳定添加 `__2`、`__3` 等编号后缀，继续完成推理和统计，不会相互覆盖或中止测试。

如只需要一个实际部署阈值，可将数组改为单个值，例如 `[0.25]`，从而减少完整测试时间。设置 `PIPELINE_LABELED_TEST_ENABLED=false` 时，`all` 会跳过测试；单独执行 `test` 会明确报错，提示配置指定测试目录。

已有 V018 训练和 ONNX 产物时，可直接运行以下命令。`0824_detect.sh` 固定执行 `test`，只复用模型测试 `PIPELINE_LABELED_TEST_DIR` 指定目录，不会运行验证集抽样或重新训练：

```bash
bash pipeline_detect/scripts/0824_detect.sh
```

新增 20260825 数据并训练 V019 时，运行 `0827_detect.sh`。该脚本默认执行 `all`：处理并注册 20260825 数据、从 V018 `best.pt` 微调训练、导出 ONNX、执行外部带标签测试并写入报告；也可以显式传入单个子命令。全新的 V019 运行目录不能首先执行 `test`，因为 `test` 只读取当前 `pipeline_state.json` 中已经由 `export_detect` 阶段登记的 ONNX：

```bash
bash pipeline_detect/scripts/0827_detect.sh
bash pipeline_detect/scripts/0827_detect.sh status
```

Excel 报表按真实的“批次 ID”数据行紧邻追加，忽略仅有样式的空白行；写入日志会以 `EXCEL_ROW_UPSERT` 标明三张表的实际行号。批次总览会记录首次开始时间、最后结束时间和累计阶段耗时；模型汇总和类别明细会填写检测标签总数及逐类别验证标签数。

旧批次若在早期版本运行、状态文件没有保存逐类别检测统计，执行 `report` 时会自动扫描共享注册表重建统计，并输出 `REPORT_STATISTICS_REBUILT`；不会重新训练、导出或测试。

## Train2 两类独立验证实验

`tools/split_train2_detect.py` 和 `tools/train2_detect_verify.py` 用于在不接入 `pipeline_detect` 注册表的情况下，独立验证 `liquid/debris` 两类检测任务。默认数据根目录为：

```text
/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/3_Train2
```

划分脚本逐批次读取原始 `images/`，标签目录兼容 `labels_detect/` 和 `label_detect/`。图片缺少标签、标签缺少图片、同名冲突或标签格式非法时跳过对应样本，并将原因写入根目录的 `split_manifest.json`；不会因为单个问题样本停止整个划分。只有数据根目录不存在、所有批次有效样本均不足或已有非空 `train/val` 等无法安全继续的情况才会停止。标签校验、图片复制和输出校验均实时显示进度，每个批次完成后输出有效及跳过数量。

脚本对每个批次分别使用固定随机种子划分训练集和验证集，默认验证集比例为 `0.2`。原始图片和原始标签均保留，划分结果复制到每个批次自己的目录。复制后的文件名增加批次前缀，避免不同批次中的同名帧在 ONNX 汇总测试时相互覆盖：

```text
260721_260812/
├── images/
├── labels_detect/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

旧标签在复制时执行以下转换，原始 TXT 不修改：

```text
旧 0=paper  -> 删除该框，图片继续保留为背景样本。
旧 1=liquid -> 新 0=liquid。
旧 2=metal  -> 新 1=debris。
```

执行划分：

```bash
conda run --no-capture-output -n ult python tools/split_train2_detect.py
```

也可以覆盖默认参数：

```bash
conda run --no-capture-output -n ult python tools/split_train2_detect.py \
    --source-root "/path/to/3_Train2" \
    --val-ratio 0.2 \
    --seed 42 \
    --copy-workers 8
```

图片默认使用 8 个线程并发复制到 NAS，以避免单线程逐张复制长时间没有反馈。NAS 负载较高时可以将 `--copy-workers` 调低到 `4`；带宽充足时可以尝试 `12`，不建议无上限提高并发数。

划分成功后会在数据根目录生成 `train2_detect.yaml`。独立实验脚本读取该配置，可选 YOLO11s-P2 或 YOLO26s-P2，依次执行训练、`best.pt` 验证、板端格式 P2 ONNX 导出和 ONNX 验证集测试，不读取或写入 `pipeline_registry`、`model_history.json` 和 Pipeline Excel。

默认使用 `yolo11s_p2`，初始权重为 V021 三类模型。加载到两类模型时只迁移形状匹配的参数，两类检测头由当前训练重新学习。默认输出目录为 `3_Train2/experiments/liquid_debris_verify_20260902`，已有非空输出目录时会停止，避免覆盖已有实验：

```bash
conda run --no-capture-output -n ult python tools/train2_detect_verify.py
```

YOLO26s-P2 使用 `/data/users/hailong.he/github/yolo/models/yolo26s.pt` 作为初始权重，默认输出到 `3_Train2/experiments/liquid_debris_yolo26s_p2_20260914`：

```bash
conda run --no-capture-output -n ult python tools/train2_detect_verify.py --model yolo26s_p2
```

Train2 使用独立的 `yolo26s_p2.yaml`，使用 YOLO26s 主干和颈部结构，检测头按照原 YOLO11s-P2 板端流程关闭 `end2end`、使用 `reg_max=16`，导出 P2-P5 四层原始 box/score 输出，保持现有板端 DFL 解码和 NMS 流程不变。初始 `yolo26s.pt` 中形状不匹配的 P2 分支和检测头参数不迁移，由当前两类训练重新学习。

正式运行前可以显式覆盖权重和输出目录：

```bash
conda run --no-capture-output -n ult python tools/train2_detect_verify.py \
    --initial-weight "/path/to/best.pt" \
    --output-dir "/path/to/liquid_debris_experiment" \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 640 \
    --device 0
```

ONNX 测试使用本次划分的验证集，因此结果属于验证集评估，不等同于独立外部测试集结果。测试默认使用 `conf=0.3`、`iou=0.5`，结果保存在实验目录的 `onnx_test/`。
