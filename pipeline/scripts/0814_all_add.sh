#!/usr/bin/env bash
set -Eeuo pipefail

# ============================== 本批次路径 ==============================
# PIDNet 仓库绝对路径.仓库位置变化时只修改这一行.
export PIPELINE_PIDNET_ROOT="/data/users/hailong.he/gitee/pidnet"
# 全部训练版本的根目录.中心 Excel 和跨批次注册表也保存在这里.
export PIPELINE_MODEL_TRAIN_ROOT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练"
# 本次运行名称.每套新训练策略使用新名称;同名重跑会恢复或跳过已完成阶段.
export PIPELINE_RUN_NAME="V014_20260814_add0812"
# 本批次原始 LabelMe 数据目录.数据处理只复制,不会移动或修改这里的文件.
export PIPELINE_INPUT_DIR='["/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/训练数据/zicai260812_pre"]'
# 本批次标准数据集目录.首次运行生成,同名重跑严格校验后跳过处理.
export PIPELINE_OUTPUT_DIR='["/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/训练数据/zicai260812"]'
# 需要生成预标注的图片目录.已有同名 JSON 默认不会覆盖.
export PIPELINE_PRELABEL_DIR="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/数据采集/Cleaning_robot/标注/20260813_会议室_双目_ch"
# 唯一维护的汇总表.表结构固定,每个 run-name 幂等新增或更新记录.
export PIPELINE_EXCEL_PATH="$PIPELINE_MODEL_TRAIN_ROOT/模型训练版本汇总_精简版.xlsx"

# ============================== 三模型训练策略 ==============================
# 三个模型的 Epoch.修改本日期脚本不会影响其他训练批次.
export PIPELINE_DETECT_EPOCHS=100
export PIPELINE_SEGMENT_EPOCHS=100
export PIPELINE_PIDNET_EPOCHS=100
# 自动 Batch 开关.true=每个新训练模型启动前探测显存;false=固定使用下面三个数值.
export PIPELINE_AUTO_BATCH=true
# 自动模式下是三个模型允许使用的最大 Batch;关闭自动模式时是固定 Batch.
export PIPELINE_DETECT_BATCH=16
export PIPELINE_SEGMENT_BATCH=16
export PIPELINE_PIDNET_BATCH=32
# YOLO 和 PIDNet 输入尺寸.
export PIPELINE_IMAGE_HEIGHT=640
export PIPELINE_IMAGE_WIDTH=640
# YOLO 和 PIDNet 数据加载进程数.
export PIPELINE_YOLO_WORKERS=16
export PIPELINE_PIDNET_WORKERS=16
# 验证集测试随机抽样数量、种子及 ONNX 推理阈值.
export PIPELINE_TEST_IMAGES=500
export PIPELINE_TEST_SEED=42
export PIPELINE_TEST_CONFIDENCE=0.25
export PIPELINE_TEST_IOU=0.5
# 使用的逻辑 GPU 编号.
export PIPELINE_GPU_DEVICE=0
# 强制重训开关.false=正常恢复/跳过;true=备份当前输出并从最近 best.pt 重新训练.
export PIPELINE_FORCE_TRAIN=false
# 自动加载上一次成功训练的 best.pt.true 时优先级高于下面三个兜底权重.
export PIPELINE_AUTO_FINETUNE=true

# 首次运行且历史注册表为空时使用的兜底权重.
export PIPELINE_DETECT_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V012_20260808_add0731/runs/yolo_detect_p2/train/weights/best.pt"
export PIPELINE_SEGMENT_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V012_20260808_add0731/runs/yolo_segment_p2/train/weights/best.pt"
export PIPELINE_PIDNET_FALLBACK_WEIGHT="/data/users/hailong.he/nas_smb/Docs_Internal/知识库(钉钉同构)/算法工具链/算法应用（主）/应用场景/【舜宇】清洁机器人/DEMO开发/模型训练/V012_20260808_add0731/runs/pidnet/liquid_metal/liquid_metal_zicai260729/best.pt"
export PIPELINE_PIDNET_CONFIG="$PIPELINE_PIDNET_ROOT/configs/liquid_metal_zicai260729.yaml"

# ============================== 预标注与日志展示 ==============================
# 是否在本次流程末尾执行预标注.
export PIPELINE_ENABLE_PRELABEL=true
# 本次是否处理并注册新增训练数据.true=先执行 prepare;false=复用已有训练数据.
export PIPELINE_ENABLE_DATA_UPDATE=true
# 预标注每次只推理一张图,避免两个 P2/语义分割模型造成显存峰值.
export PIPELINE_PRELABEL_BATCH=1
# false 表示已有同名 JSON 一律跳过,保护人工微调过的标签.
export PIPELINE_PRELABEL_OVERWRITE=false
# 日志和 Excel 中把服务器映射路径显示为客户可访问的 UNC 路径.
export PIPELINE_DISPLAY_RUNTIME_PREFIX="/data/users/hailong.he/nas_smb"
export PIPELINE_DISPLAY_NAS_PREFIX='\\192.168.0.68'

# 默认直接执行 pipeline all。也可传入单阶段命令.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

COMMAND="${1:-all}"
shift || true
exec conda run --no-capture-output -n ult python -m pipeline.pipeline "$COMMAND" "$@"
