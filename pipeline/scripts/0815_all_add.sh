#!/usr/bin/env bash
set -Eeuo pipefail

# ============================== 本批次路径 ==============================
# PIDNet 仓库绝对路径。
export PIPELINE_PIDNET_ROOT="/data/users/hailong.he/gitee/pidnet"
# 全部训练批次、注册表和中心 Excel 的根目录。
export PIPELINE_MODEL_TRAIN_ROOT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练"
# 本批次唯一名称；同名重跑会复用已完成阶段或从断点恢复。
export PIPELINE_RUN_NAME="V015_20260814_add0813"
# 新增 LabelMe 原始数据根目录，流程会递归寻找同名图片与 JSON。
export PIPELINE_INPUT_DIR='["/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/数据采集/Cleaning_robot/标注/02_annotation/20260813_B001"]'
# 原始数据处理后的标准训练数据输出目录。
export PIPELINE_OUTPUT_DIR='["/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/训练数据/zicai260813"]'
# 预标注图片目录列表；每项仅扫描当前目录，不递归扫描子目录。
export PIPELINE_PRELABEL_DIR='[
  "/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/数据采集/Cleaning_robot/00_workspace/dervied/02_controlled_scene/20260813_B001/s105_liquid_cookoil_100ml_q1_wheeltrack_on-white-top.mp4/RGB",
  "/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/数据采集/Cleaning_robot/00_workspace/dervied/02_controlled_scene/20260813_B001/s102_liquid_cookoil_50ml_q2_leftpool_off-warm-r45/RGB",
  "/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/数据采集/Cleaning_robot/00_workspace/dervied/02_controlled_scene/20260813_B001/s102_liquid_cookoil_50ml_q1_leftpool_off-warm-r45/RGB"
]'
# 唯一中心 Excel；运行结果会按批次名新增或更新记录。
export PIPELINE_EXCEL_PATH="$PIPELINE_MODEL_TRAIN_ROOT/模型训练版本汇总_精简版.xlsx"

# ============================== 三模型训练策略 ==============================
# 三个模型各自的训练轮数。
export PIPELINE_DETECT_EPOCHS=100
export PIPELINE_SEGMENT_EPOCHS=100
export PIPELINE_PIDNET_EPOCHS=100
# true 时按 GPU 空闲显存自动选择 Batch，且不超过下方三个 Batch 上限；同一批次重跑复用首次选择结果。
export PIPELINE_AUTO_BATCH=true
# 自动 Batch 的上限；关闭自动模式后即为固定 Batch。
export PIPELINE_DETECT_BATCH=16
export PIPELINE_SEGMENT_BATCH=16
export PIPELINE_PIDNET_BATCH=32
# YOLO 和 PIDNet 共用的输入尺寸，顺序为高、宽。
export PIPELINE_IMAGE_HEIGHT=640
export PIPELINE_IMAGE_WIDTH=640
# 两个训练环境的数据加载进程数。
export PIPELINE_YOLO_WORKERS=16
export PIPELINE_PIDNET_WORKERS=16
# 验证集抽样测试数量、随机种子和 ONNX 推理阈值。
export PIPELINE_TEST_IMAGES=500
export PIPELINE_TEST_SEED=42
export PIPELINE_TEST_CONFIDENCE=0.25
export PIPELINE_TEST_IOU=0.5
# 使用的逻辑 GPU 编号。
export PIPELINE_GPU_DEVICE=0
# true 时备份当前输出并从历史 best.pt 重新训练；普通重跑保持 false，以便断点恢复或跳过已完成模型。
export PIPELINE_FORCE_TRAIN=false
# 新批次优先加载最近一次成功训练的 best.pt；历史为空时才使用下方兜底权重。
export PIPELINE_AUTO_FINETUNE=true

# 首次训练且没有可用历史模型时使用的三个兜底权重。
export PIPELINE_DETECT_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V012_20260808_add0731/runs/yolo_detect_p2/train/weights/best.pt"
export PIPELINE_SEGMENT_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V012_20260808_add0731/runs/yolo_segment_p2/train/weights/best.pt"
export PIPELINE_PIDNET_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V012_20260808_add0731/runs/pidnet/liquid_metal/liquid_metal_zicai260729/best.pt"
export PIPELINE_PIDNET_CONFIG="$PIPELINE_PIDNET_ROOT/configs/liquid_metal_zicai260729.yaml"

# ============================== 预标注与日志展示 ==============================
# true 时 all 的最后执行预标注；false 时完全跳过预标注目录检查和推理。
export PIPELINE_ENABLE_PRELABEL=true
# true 时处理本批原始数据并更新累计注册表；false 时复用已有注册表，适合数据已处理成功后的重跑。
export PIPELINE_ENABLE_DATA_UPDATE=true
# 预标注单次推理图片数；保持 1 可降低 YOLO 与 PIDNet 依次推理时的显存峰值。
export PIPELINE_PRELABEL_BATCH=1
# false 时跳过已有同名 JSON，保护人工修改过的标签；true 时覆盖并重新生成 JSON。
export PIPELINE_PRELABEL_OVERWRITE=false
# 日志和 Excel 将服务器路径显示为 Windows 可访问的 UNC 路径。
export PIPELINE_DISPLAY_RUNTIME_PREFIX="/data/users/hailong.he/nas_smb"
export PIPELINE_DISPLAY_NAS_PREFIX='\\192.168.0.68'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

COMMAND="${1:-all}"
shift || true
exec conda run --no-capture-output -n ult python -m pipeline.pipeline "$COMMAND" "$@"
