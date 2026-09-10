#!/usr/bin/env bash
set -Eeuo pipefail

# 处理并注册 20260825 新增检测数据,用于训练 V019.
export PIPELINE_INPUT_DIR='[
    "/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/0_Collection/Cleaning_robot/04_labeled/20260901/"
]'
export PIPELINE_OUTPUT_DIR='[
    "/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260901"
]'
export PIPELINE_MODEL_TRAIN_ROOT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练"
export PIPELINE_RUN_NAME="V022_20260902_add0902"
export PIPELINE_EXCEL_PATH="$PIPELINE_MODEL_TRAIN_ROOT/模型训练版本汇总.xlsx"
export PIPELINE_DETECT_EPOCHS=100

# true 时由 Ultralytics 自动探测 Batch;false 时使用 DETECT_BATCH.
export PIPELINE_AUTO_BATCH=true
export PIPELINE_DETECT_BATCH=16
export PIPELINE_IMAGE_HEIGHT=640
export PIPELINE_IMAGE_WIDTH=640
export PIPELINE_YOLO_WORKERS=16
export PIPELINE_YOLO_AMP=false

# 外部带标签测试使用的 NMS IoU 阈值.
export PIPELINE_TEST_IOU=0.5

# 对外部带标签测试集执行完整测试.
export PIPELINE_LABELED_TEST_ENABLED=true
export PIPELINE_LABELED_TEST_DIR="/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/5_Test/02_annotation"
export PIPELINE_TEST_OUTPUT_DIR="/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/6_Result/$PIPELINE_RUN_NAME"
export PIPELINE_LABELED_TEST_CONFIDENCES='[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'
# 按场景路径目录名排除小尺寸样本，且不纳入任何测试精度统计。
export PIPELINE_LABELED_TEST_EXCLUDE_KEYWORDS='["1x1cm", "1-2cm", "2x2cm"]'

export PIPELINE_GPU_DEVICE=0
export PIPELINE_VAL_RATIO=0.2
export PIPELINE_RANDOM_SEED=42

# 新运行目录从上一版本 best.pt 微调;普通重跑保持 false 以支持恢复或跳过已完成训练.
export PIPELINE_FORCE_TRAIN=false
export PIPELINE_AUTO_FINETUNE=true
export PIPELINE_DETECT_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V018_20260824_clean/runs/yolo_detect_p2/train/weights/best.pt"

# false 时跳过原始数据预处理,只读复用累计注册表中的清洗后数据.
export PIPELINE_ENABLE_DATA_UPDATE=true
# 该开关仅用于预处理阶段;空标签图片不纳入本批训练数据.
export PIPELINE_INCLUDE_NEGATIVE_SAMPLES=false

# 日志与 Excel 路径显示规则.
export PIPELINE_DISPLAY_RUNTIME_PREFIX="/data/users/hailong.he/nas_smb"
export PIPELINE_DISPLAY_NAS_PREFIX='\\192.168.0.68'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
# 默认执行 prepare、train、export、test 和 report;也可显式传入单个子命令.
exec conda run --no-capture-output -n ult python -m pipeline_detect.pipeline "${1:-all}"
