#!/usr/bin/env bash
set -Eeuo pipefail
# 原始采集目录与标准训练数据输出目录，两个 JSON 数组按下标一一对应。
export PIPELINE_INPUT_DIR='[
    "/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/0_Collection/Cleaning_robot/04_labeled/20260819/"
]'
export PIPELINE_OUTPUT_DIR='[
    "/data/users/hailong.he/nas_smb/Datasets/internal/P000_SHUNYU_2026/2_Train/20260819"
]'
export PIPELINE_MODEL_TRAIN_ROOT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练"
export PIPELINE_RUN_NAME="V016_20260819_merge_detect"
export PIPELINE_EXCEL_PATH="$PIPELINE_MODEL_TRAIN_ROOT/模型训练版本汇总.xlsx"
export PIPELINE_DETECT_EPOCHS=100
# true 时由 Ultralytics 自动探测 Batch；false 时使用 DETECT_BATCH。
export PIPELINE_AUTO_BATCH=true
export PIPELINE_DETECT_BATCH=16
export PIPELINE_IMAGE_HEIGHT=640
export PIPELINE_IMAGE_WIDTH=640
export PIPELINE_YOLO_WORKERS=16
export PIPELINE_YOLO_AMP=false
# 外部带标签测试使用的 NMS IoU 阈值。
export PIPELINE_TEST_IOU=0.5
export PIPELINE_GPU_DEVICE=0
export PIPELINE_VAL_RATIO=0.2
export PIPELINE_RANDOM_SEED=42
export PIPELINE_FORCE_TRAIN=false
# 新批次优先从最近成功的检测 best.pt 微调；历史为空时使用下方兜底权重。
export PIPELINE_AUTO_FINETUNE=true
export PIPELINE_DETECT_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V015_20260814_add0813/runs/yolo_detect_p2/train/weights/best.pt"
# true 时处理并注册本批 img/yolo 数据；false 时只复用累计注册表。
export PIPELINE_ENABLE_DATA_UPDATE=true
# true 时保留同名空 TXT 的负样本；false 时处理阶段跳过它们。
export PIPELINE_INCLUDE_NEGATIVE_SAMPLES=true
# 输入标签中的这些类别仅用于涂黑图片区域，不进入检测训练标签。
export PIPELINE_IGNORE_CLASS_IDS='[3]'
# 日志与 Excel 路径显示规则。
export PIPELINE_DISPLAY_RUNTIME_PREFIX="/data/users/hailong.he/nas_smb"
export PIPELINE_DISPLAY_NAS_PREFIX='\\192.168.0.68'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO_ROOT"
exec conda run --no-capture-output -n ult python -m pipeline_detect.pipeline "${1:-all}"
