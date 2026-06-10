#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/VINGS-Mono
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vings_vio

log_path="run_logs/kitti_0028_try_20260605_2340.txt"
nohup python scripts/run.py configs/kitti/sync/kitti_2011_09_30_drive_0028_try.yaml > "${log_path}" 2>&1 < /dev/null &
echo "PID=$!"
echo "LOG=${log_path}"
