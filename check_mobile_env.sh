#!/bin/bash
# 远端环境检查脚本（无副作用，只读）
# 检查移动端实时链路所需的所有依赖

echo "============================================"
echo "VINGS-Mono 移动端环境检查"
echo "============================================"

# 1. 项目根目录
PROJ="/root/autodl-tmp/VINGS-Mono"
echo -n "[1] 项目路径: "
if [ -d "$PROJ" ]; then
    echo "OK — $PROJ"
else
    echo "MISSING — $PROJ"
fi

# 2. 配置文件
echo -n "[2] mobile_vo.yaml: "
if [ -f "$PROJ/configs/mobile/mobile_vo.yaml" ]; then
    echo "OK"
else
    echo "MISSING"
fi

# 3. 关键脚本
for f in \
    "scripts/run_multiprocess_mobile.py" \
    "scripts/server/server.py" \
    "scripts/datasets/phone_server.py" \
    "scripts/datasets/phone.py" \
    "scripts/datasets/mobile.py" \
    "scripts/datasets/mobile_offline.py" \
    "scripts/run_mobile.py"
do
    echo -n "[3] $f: "
    if [ -f "$PROJ/$f" ]; then
        echo "OK"
    else
        echo "MISSING"
    fi
done

# 4. conda 环境
echo -n "[4] conda env 'vings_vio': "
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null
if conda env list 2>/dev/null | grep -q "vings_vio"; then
    echo "OK"
else
    echo "MISSING"
fi

# 5. checkpoint
echo -n "[5] droid.pth: "
if [ -f "$PROJ/ckpts/droid.pth" ]; then
    echo "OK"
else
    echo "MISSING"
fi

echo -n "[6] GPU info: "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "NO GPU"

# 6. 端口占用
echo -n "[7] Port 5000: "
if ss -tlnp 2>/dev/null | grep -q ":5000"; then
    echo "IN USE — $(ss -tlnp | grep :5000)"
else
    echo "FREE"
fi

# 7. 磁盘
echo -n "[8] Disk: "
df -h /root/autodl-tmp/ 2>/dev/null | tail -1

echo ""
echo "============================================"
echo "检查完成"
echo "============================================"
