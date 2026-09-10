# 清洁机器人三模型训练流水线

本目录是重新整理后的独立流水线。旧的 `tools/data_pipeline.py` 和 `bash/` 脚本保留作为回退，新流程不导入也不修改它们。

## 源码目录

```text
pipeline/
├── pipeline.py             # 唯一 Python 入口.
├── README.md
├── requirements.txt
├── scripts/                # 每批次可复制并修改的 all 脚本.
├── core/                   # 配置、状态、日志、预检等基础设施.
├── steps/                  # 数据、训练、导出、预标注和报表步骤.
└── workers/                # 在独立 Conda 环境执行的辅助入口.
```

日常仍使用 `./scripts/日期_all.sh` 或 `python -m pipeline.pipeline`，目录整理不改变命令行和运行行为。

## 核心约定

- 日期版 Shell 脚本只保存本批次的绝对路径和训练策略，公共 Python 只维护一份。
- 用户主要直接执行 `./0806_all.sh`。新批次复制为 `0807_all.sh` 后只修改配置区。
- 支持 `prepare`、`train`、`export`、`test`、`prelabel`、`report`、`status` 和 `all` 独立运行。
- 同一个 `PIPELINE_RUN_NAME` 重跑时，完整阶段校验后跳过；YOLO 从 `last.pt`、PIDNet 从 `checkpoint.pth.tar` 恢复。
- 累计注册表以逻辑 `train/val` 分组为准，支持 `train_neg/images` 和直接图片目录等历史结构；历史样本不会被改名或丢弃。
- 新 run-name 或强制重训没有断点时，自动加载最近一次成功的三个 `best.pt`；历史为空才使用日期脚本的兜底权重。
- 数据处理只复制原始 JPG/JSON，不移动、不删除、不修改原始数据。
- 原始目录递归扫描图片和 JSON，只处理同名文件的交集；未配对图片、未配对 JSON（例如 `instances_default.json`）只记录跳过统计，不会阻断处理。
- 三个模型依次执行“训练完成立即导出 ONNX”：YOLO P2 检测、YOLO P2 实例分割、PIDNet 语义分割。
- 预标注只使用 YOLO paper 分割和 PIDNet liquid/metal 分割，检测模型不参与。
- 预标注 JSON 直接放在图片旁边，默认跳过已有同名 JSON；`PIPELINE_PRELABEL_DIR` 支持历史单路径或非空 JSON 路径列表，每项必须是直接存放图片的目录，不递归扫描子目录。
- 日期脚本设置 `PIPELINE_ENABLE_PRELABEL=false` 时，`all` 跳过预标注且不检查预标注目录；直接执行 `prelabel` 命令仍会正常运行。
- 每个运行目录只有一个持续追加的 `pipeline.log`；原生 tqdm 只显示在控制台，不写入日志。
- 同一训练根目录使用系统排他锁，防止两个 tmux 会话同时训练或同时写注册表、权重、状态和 Excel。
- 所有批次只维护一个中心 Excel，不在运行目录复制工作簿。

## 多组输入输出

`PIPELINE_INPUT_DIR` 和 `PIPELINE_OUTPUT_DIR` 使用 JSON 路径列表，两个列表按索引一对一处理，长度必须一致。每个输入目录独立输出到对应目录，不会在数据处理阶段合并：

```bash
export PIPELINE_INPUT_DIR='["/data/raw_a", "/data/raw_b"]'
export PIPELINE_OUTPUT_DIR='["/data/processed_a", "/data/processed_b"]'
```

后续注册和训练会读取这些独立标准数据集，并将训练来源汇总到生成的 YOLO/PIDNet 配置中。

每组输入输出处理完成后，日志会输出 `DATASET_OUTPUT`，包含处理状态、输出目录、训练/验证图片数量和检测实例数量；全部目录完成后会输出 `DATASET_OUTPUT_SUMMARY`。

在训练开始前，`register_data` 会输出 `TRAINING_DATASET_SUMMARY`，其中包含累计训练/验证图片数、标签实例总数和 paper、liquid、metal 各类别的 train/val 数量。

## 运行产物

```text
PIPELINE_MODEL_TRAIN_ROOT/
├── 模型训练版本汇总_精简版.xlsx        # 唯一中心表格.
├── pipeline_registry/
│   ├── datasets.yaml                    # 累计数据唯一来源.
│   └── model_history.json                # 三模型历史 best.pt.
└── PIPELINE_RUN_NAME/
    ├── pipeline.log                      # 唯一日志,重跑继续追加.
    ├── pipeline_state.json               # 阶段状态、指标和产物.
    ├── training_configs/                 # YOLO/PIDNet 配置快照.
    ├── pidnet_list/                      # PIDNet 累计数据列表.
    ├── runs/                             # 三模型权重及原生结果.
    ├── onnx/                             # 三个已校验 ONNX.
    ├── results/                          # 抽样测试清单、三联图和精度指标.
    └── reports/
        ├── dataset_statistics.json
        ├── training_report.json
        └── excel_pending.json             # 仅 Excel 待补写时存在.
```

标准数据集只保存数据：

```text
dataset/
├── train/
│   ├── img_src/          # 未做 ignore 处理的原图副本.
│   ├── images/           # ignore 区域处理后的训练图.
│   ├── json/
│   ├── labels_detect/    # paper=0, liquid=1, metal=2.
│   ├── labels_segment/   # 仅 paper=0.
│   └── Seg/              # background=0, liquid=1, metal=2.
└── val/
    └── ...
```

## 日常运行

首次同步到 Linux 后确认执行权限：

```bash
cd /data/users/hailong.he/gitee/ultralytics/pipeline/scripts
chmod +x 0806_all.sh
tmux new -s train_0806
./0806_all.sh
```

中心 Excel 写入依赖 `openpyxl`。首次部署新流程时在 `ult` 环境安装一次：

```bash
conda activate ult
python -m pip install -r pipeline/requirements.txt
```

流水线不会在正式运行时联网安装依赖；依赖缺失会在耗时训练开始前明确报错。

默认就是 `all`。同一份日期脚本可单独运行：

```bash
./0806_all.sh status
./0806_all.sh prepare
./0806_all.sh train
./0806_all.sh export
./0806_all.sh test
./0806_all.sh prelabel
./0806_all.sh report
```

如果训练和 ONNX 导出已经完成，仅最终报表阶段失败，修复问题后执行 `./0806_all.sh report` 即可补写报告和中心 Excel，不会重新训练或导出模型。报表 JSON 会把 Excel 使用的原生时间值保存为 ISO 时间字符串。

中心 Excel 中的 Fitness、P、R、mAP、IoU、Dice 和像素准确率统一显示 3 位小数，并以 `0.xxx` 显示，不使用百分号；单元格仍保留原始数值精度。每次执行 `report` 都会同时规范 Excel 中已有的历史指标记录，`pipeline.log` 和 `training_report.json` 也保留原始指标精度。

- `prepare`：处理本批次数据并加入累计注册表，不训练。
- `train`：不处理原始数据，训练或恢复三个模型。
- `export`：不训练，只把当前三个 `best.pt` 导出 ONNX。
- `test`：不训练，随机抽取验证集有目标图片并调用现有 ONNX 推理脚本测试。
- `prelabel`：不训练，只为缺少 JSON 的图片预标注。
- `report`：不训练、不导出，只重试写入中心 Excel。
- `status`：查看阶段状态、尝试次数和产物。
- `all`：预检后依次执行处理、训练、导出、测试、报表和预标注。

日期脚本可使用 `PIPELINE_ENABLE_DATA_UPDATE` 控制本次是否处理并注册新增训练数据：`true` 时先执行 `prepare`，`false` 时仅读取已有 `pipeline_registry/datasets.yaml`，不会扫描或校验 `PIPELINE_INPUT_DIR`、`PIPELINE_OUTPUT_DIR`。`false` 需要已有注册表；首次运行请设为 `true`。该开关不影响预标注；`PIPELINE_ENABLE_PRELABEL=true` 时，日期脚本会在训练、导出、测试和预标注完成后写入报告。`PIPELINE_PRELABEL_DIR` 支持单路径或 JSON 路径列表，例如 `['/data/prelabel_a/images', '/data/prelabel_b/images']`。

`all` 在耗时训练前一次性检查原始数据、输出父目录、两个仓库、三个模型配置/初始权重、全部预标注目录和 Excel 表头。

累计注册表只保存服务器真实绝对路径，不再维护 `PATH_PREFIX_REMAP`。读取旧注册表时，如果仅存在 `算法应用(主)` 与 `算法应用（主）` 这类 Unicode 括号差异且只有一个真实目录，流水线会自动修复并原子更新注册表；无法唯一判断的路径会停止并明确指出，不会创建一个新的错误目录。

如果原始目录只从 `picture` 改名为 `img`，已有标准数据集的清单路径会不同。流水线会比较全部文件名、LabelMe shapes、JPG 大小和 SHA256；确认每张原图及标注都完全一致后自动重新绑定新绝对路径并跳过处理。只要任一图片或标签不同就停止，不会把另一批数据误认为已处理。

## 重跑与强制重训

正常重跑同一个日期脚本时：

- 完整数据严格校验后跳过。
- YOLO 未完成且存在 `last.pt` 时用 `resume=True`。
- PIDNet 未完成且存在 `checkpoint.pth.tar` 时用 `TRAIN.RESUME=True`。
- 已完成模型跳过训练；ONNX 仍校验是否对应当前 `best.pt`。
- 已有同名预标注 JSON 的图片跳过。
- Excel 按业务主键更新原行，不产生重复批次。

正式重训时在日期脚本中设置：

```bash
export PIPELINE_FORCE_TRAIN=true
```

当前三个训练目录会改名备份，再从备份或最近历史的 `best.pt` 做 finetune，并从 Epoch 1 开始。完成后应改回 `false`。新策略应复制日期脚本并更换 `PIPELINE_RUN_NAME`；新 run-name 自动加载上次成功的三个 `best.pt`，不会恢复上次 optimizer/epoch。

## 三模型自动 Batch

日期脚本中的统一开关同时控制 YOLO P2 检测、YOLO P2 实例分割和 PIDNet：

```bash
export PIPELINE_AUTO_BATCH=true
```

开启后，每个需要从 Epoch 1 开始的新训练模型都会在启动前读取指定 GPU 的实时总显存、空闲显存和已用显存，统一以总显存 80% 作为目标上限。YOLO 使用仓库现有 AutoBatch 按 80% 探测；PIDNet 执行完整前向、反向和优化器步骤测量峰值总显存。手工配置的 Batch 峰值不超过 80% 时直接使用，超过后才自动寻找 80% 以内的最大 Batch。三个手工 Batch 参数作为最大值，自动结果不会超过它们：

```bash
export PIPELINE_DETECT_BATCH=16
export PIPELINE_SEGMENT_BATCH=12
export PIPELINE_PIDNET_BATCH=32
```

选择结果、目标显存比例及 PIDNet 实测峰值保存在本次运行的 `training_configs/auto_batch.json`，并写入 `pipeline.log`、模型阶段状态和中心 Excel。同一个 run-name 断点恢复时复用首次选择结果，不会因为重跑时显存占用变化而改变 Batch。已有模型直接跳过训练时也不会重复探测；新策略从新的 run-name 开始生效。

关闭开关后，三个模型固定使用日期脚本配置的 Batch：

```bash
export PIPELINE_AUTO_BATCH=false
```

## PIDNet 独立仓库

日期脚本统一配置 PIDNet：

```bash
export PIPELINE_PIDNET_ROOT="/data/users/hailong.he/gitee/pidnet"
export PIPELINE_PIDNET_CONFIG="$PIPELINE_PIDNET_ROOT/configs/liquid_metal_zicai260729.yaml"
export PIPELINE_PIDNET_EPOCHS=200
export PIPELINE_PIDNET_BATCH=24
export PIPELINE_IMAGE_HEIGHT=800
export PIPELINE_IMAGE_WIDTH=1280
export PIPELINE_PIDNET_WORKERS=16
```

流水线生成本次 PIDNet 配置快照，通过 `conda run -n pid` 完成训练、恢复、指标验证、ONNX 和预标注。原 PIDNet 配置不覆盖。YOLO 使用 `ult` 环境。

PIDNet 配置快照固定为 `training_configs/pidnet.yaml`，训练输出固定为 `runs/pidnet/train/`。同名运行首次使用新代码时，会把唯一的旧版 `runs/pidnet/liquid_metal/<配置名>/` 自动迁移到新目录；新旧目录同时存在或旧目录不唯一时停止，避免覆盖权重。

## 验证集抽样测试

1280x800 复用数据训练使用 `pipeline/scripts/0812_all_0812_1280x800.sh`。该脚本不运行 `prepare` 和 `prelabel`，默认依次执行 `train`、`export`、`test`、`report`；其中 `train` 只校验已有标准数据集并为新批次生成训练配置。

日期脚本默认配置：

```bash
export PIPELINE_TEST_IMAGES=500
export PIPELINE_TEST_SEED=42
export PIPELINE_TEST_CONFIDENCE=0.25
export PIPELINE_TEST_IOU=0.5
```

单独执行当前批次测试：

```bash
cd /data/users/hailong.he/gitee/ultralytics/pipeline/scripts
./0807_all_0731.sh test
```

测试从累计验证集中筛选检测 GT 非空且检测、YOLO 分割、PIDNet GT 都完整的图片，按固定种子随机选择最多 500 张。检测和融合分割统一复用 `tools/inference.py`；脚本根据 ONNX 输出节点自动识别任务及 P2/P3 检测头。两次评测使用同一份 `results/test_samples.json`，每张图保存“原图｜GT｜预测”三联图。

```text
results/
├── test_samples.json
├── detect/
│   ├── images/
│   ├── metrics.json
│   └── metrics.txt
└── segment/
    ├── images/
    ├── metrics.json
    └── metrics.txt
```

检测指标包括逐类和整体 P、R、AP50、mAP50-95。分割指标分别保存 YOLO paper 的实例分割指标、PIDNet liquid/metal 的语义指标，以及 paper/liquid/metal 融合结果的 Precision、Recall、IoU、Dice、Pixel Accuracy 和混淆矩阵。路径和指标摘要同时写入 `pipeline_state.json` 与 `pipeline.log`。

## ONNX

训练、验证、YOLO 与 PIDNet 共用一组尺寸参数，顺序固定为高、宽。训练尺寸会自动向上对齐到 32；ONNX 仍保留配置的外部输入尺寸，并在模型首部自动加入居中 Pad。例如外部输入 `360x640` 时，骨干网络实际接收 `384x640`。当前日期脚本使用 `800x1280`，本身已被 32 整除，因此不会产生额外 Pad。

```text
onnx/yolo11_p2_detect.onnx          # tools/export.py v3 相同的 P2 原始输出格式.
onnx/yolo11_p2_segment.onnx         # 直接调用 v2_yolov11_seg_p2().
onnx/pidnet_semantic_segment.onnx   # 调用 PIDNet tools/export_onnx.py.
```

每个模型另有 `.export.json`，记录输入 `best.pt` 和输出 ONNX 的 SHA256。只有权重一致且 ONNX 结构校验通过才跳过导出。

## 日志和固定 Excel

`pipeline.log` 面向训练人员，保存完整配置、预检、累计数据数、阶段状态、训练模式、初始权重、每 Epoch 摘要、最佳 Epoch、总体/逐类别指标、ONNX、预标注统计和错误堆栈。batch 级 tqdm 不进入日志。

NAS 路径在日志和 Excel 中显示为全反斜杠的 UNC 格式，例如 `\\192.168.0.68\Docs_Internal\...`；状态、注册表和训练配置仍使用 Linux 真实绝对路径。

中心 Excel 固定使用现有三个 Sheet 和表头，不再改变结构：

- `批次总览`：每个 run-name 一行。
- `模型汇总`：每批次三个模型各一行。
- `类别明细`：每批次、模型、验证集和类别各一行；PIDNet 只记录 `liquid`、`metal`，不记录背景。

`总耗时`和`训练时长`使用 Excel 累计小时格式 `[h]:mm:ss`。小时不会在 24 小时后归零，例如训练 10 天显示为 `240:00:00`，并且仍可参与排序、求和和平均值计算。`批次总览`的`总耗时`累计本批次所有已执行阶段的实际耗时，包含数据处理、训练、导出、测试、预标注和报表；失败后的人工检查与重启间隔不计入。开始、结束时间仅用于审计，不参与总耗时计算。

PIDNet 的`全类P`、`全类R`、`mIoU(FG)`、`Mean Acc`和`Macro Dice`只按 `liquid`、`metal`计算。由于`mIoU(All)`和全局`Pixel Acc`的定义包含背景，PIDNet 对应单元格留空，避免与前景指标混用。流水线写入报表时会自动迁移历史时长文本，并清理历史 PIDNet 背景行和含背景汇总值。

三个 Sheet 均使用第 1 行作为标题、第 2 行作为固定表头，并从第 3 行开始保存训练记录。请勿删除或移动第 2 行表头。

流水线只更新值并沿用现有样式/公式，不新增 Sheet、不新增列。Excel 正被打开、NAS 暂不可写或保存失败时，模型和 ONNX 不受影响，`all` 仍会继续完成预标注，最终状态为 `PARTIAL`，并生成 `reports/excel_pending.json`。恢复后执行 `./0806_all.sh report`；同一业务主键会更新原记录，成功后待补写文件自动删除。

日志最终标记：

```text
PIPELINE_FINAL_STATUS | SUCCESS | ...
PIPELINE_FINAL_STATUS | PARTIAL | ...
PIPELINE_FINAL_STATUS | FAILED | ...
```
