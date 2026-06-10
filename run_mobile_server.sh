#!/bin/bash
# VINGS-Mono 移动端实时服务器启动脚本
# 用法: bash run_mobile_server.sh
# 此脚本在 GPU 服务器上运行，启动 WebSocket 服务器 + Tracking + Mapping 三进程

set -e

cd /root/autodl-tmp/VINGS-Mono

# 1. 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vings_vio

# 2. 设置 PYTHONPATH
export PYTHONPATH=$PWD/scripts:$PWD/submodules:$PWD/submodules/metric_modules/metric3d:$PWD:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 3. 创建日志目录
mkdir -p run_logs output

# 4. 启动多进程 pipeline
#   - 进程1: WebSocket 服务器 (server/server.py) — 接收手机图像+IMU，回传渲染结果
#   - 进程2: Tracking  (DBAFusion 前端) — VIO/VO 追踪
#   - 进程3: Mapping   (GaussianModel 后端) — 3DGS 建图
CONFIG="configs/mobile/mobile_vo.yaml"
TS=$(date +%Y%m%d_%H%M%S)
LOG="run_logs/mobile_${TS}.txt"

echo "============================================"
echo "VINGS-Mono Mobile Server"
echo "Config:  $CONFIG"
echo "Log:     $LOG"
echo "WebSocket: ws://0.0.0.0:5000"
echo "============================================"

python scripts/run_multiprocess_mobile.py "$CONFIG" 2>&1 | tee "$LOG"
