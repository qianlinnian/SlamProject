#!/usr/bin/env bash
cd /root/autodl-tmp/VINGS-Mono
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vings_vio
python scripts/run.py configs/hierarchical/smallcity_vis.yaml > run_logs/vis_smallcity_20260607.txt 2>&1
python scripts/run.py configs/kitti360/unsync/kitti360_2013_05_28_drive_0000_0000_3999_lite_vis.yaml > run_logs/vis_kitti360_0000_lite_20260607.txt 2>&1
