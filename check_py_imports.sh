#!/bin/bash
# 验证 Python 环境关键依赖
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vings_vio

cd /root/autodl-tmp/VINGS-Mono
export PYTHONPATH=$PWD/scripts:$PWD/submodules:$PWD/submodules/metric_modules/metric3d:$PWD:$PYTHONPATH

echo "=== Python ==="
python --version

echo "=== PyTorch + CUDA ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

echo "=== Server imports ==="
python -c "import websockets; print('websockets OK')"
python -c "import cv2; print('cv2 OK')"

echo "=== Tracker imports ==="
python -c "from frontend.dbaf import DBAFusion; print('DBAFusion OK')"
python -c "from gaussian.gaussian_model import GaussianModel; print('GaussianModel OK')"

echo "=== Mobile dataset imports ==="
python -c "from datasets.phone_server import get_dataset; print('phone_server OK')"
python -c "from datasets.mobile import get_dataset; print('mobile OK')"

echo "=== Server import ==="
python -c "from server.server import WebsocketServer; print('WebsocketServer OK')"

echo "=== GTSAM ==="
python -c "import gtsam; print(f'GTSAM {gtsam.__version__}')"

echo "=== ALL PASS ==="
