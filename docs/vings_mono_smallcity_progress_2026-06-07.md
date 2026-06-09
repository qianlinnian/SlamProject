# VINGS-Mono SmallCity 实验记录（阶段版）

日期：2026-06-07  
项目路径：`/root/autodl-tmp/VINGS-Mono`  
当前目标：跑通 VINGS-Mono README Step 3 中的 SmallCity demo，并记录环境配置、问题定位、解决方法和当前实验结果。

## 1. 阶段结论

- `SmallCity` 模块已经可以在 `vings_vio` conda 环境中完整跑通。
- 当前使用命令：

```bash
python scripts/run.py configs/hierarchical/smallcity.yaml
```

- 最新完整输出目录：

```text
output/06-07-01-05-hierarchical-smallcit-/
```

- 最终 2DGS 点云/高斯结果已经生成：

```text
output/06-07-01-05-hierarchical-smallcit-/ply/idx=876_2dgs.ply
```

- 当前产物规模：
  - 输入帧数：`877`
  - `droid_c2w` 位姿文件：`310`
  - `rgbdnua` 可视化/中间结果：`310`
  - `map` 可视化图：`473`
  - `bev` 可视化图：`473`
  - 最终 `idx=876_2dgs.ply`：约 `1.7GB`

这说明当前阶段不是只解决了启动问题，而是已经完成了 SmallCity 的端到端运行。

## 2. 实验环境

### 2.1 基础环境

- 系统路径：`/root/autodl-tmp/VINGS-Mono`
- conda 环境：`vings_vio`
- Python：`3.9.19`
- PyTorch：`2.0.1+cu118`
- GPU：NVIDIA GeForce RTX 4090
- CUDA 可见性验证结果：

```text
torch 2.0.1+cu118
cuda_available True
imports ok
```

### 2.2 关键 Python/CUDA 包验证

已验证以下模块可以导入：

```python
import torch
import droid_backends
import lietorch
import diff_surfel_rasterization
import gtsam
```

这些模块分别对应 PyTorch、DBAF/DROID CUDA 扩展、LieTorch、Gaussian rasterization 和 GTSAM。它们能正常导入是 Step 3 能继续运行的前提。

## 3. 数据和权重准备

### 3.1 SmallCity 数据

数据路径：

```text
/root/autodl-tmp/VINGS-Mono/dataset/small_city/
```

当前目录结构包含：

```text
color/
color_nosky/
fake_depth/
intrinsic.txt
name.txt
pose/
```

`/root/autodl-tmp/VINGS-Mono/scripts/datasets/hierarchical.py` 会从 `color/*.png` 读取 SmallCity 图像序列。

### 3.2 Checkpoint

权重目录：

```text
/root/autodl-tmp/VINGS-Mono/ckpts/
```

当前已就位：

```text
droid.pth                         16M
metric_depth_vit_small_800k.pth   144M
```

作用：

- `droid.pth`：DBAF/DROID 前端跟踪权重。
- `metric_depth_vit_small_800k.pth`：Metric3D 单目深度估计权重。

## 4. SmallCity 配置

配置文件：

```text
configs/hierarchical/smallcity.yaml
```

当前关键配置：

```yaml
use_metric: True
mode: 'vo'

dataset:
  root: /root/autodl-tmp/VINGS-Mono/dataset/small_city/

output:
  save_dir: /root/autodl-tmp/VINGS-Mono/output/

frontend:
  weight: /root/autodl-tmp/VINGS-Mono/ckpts/droid.pth

use_vis: True
```

说明：

- `mode: vo`：当前 SmallCity 以视觉里程计模式运行。
- `use_metric: True`：如果数据包中没有深度，会调用 Metric3D 预测深度。
- `use_vis: True`：运行过程中会额外生成 `map/` 和 `bev/` 可视化图，耗时和输出体积会增加。

## 5. 遇到的困难和解决方法

### 5.1 Python 环境没有对齐

现象：

- 直接执行 `bash set_env.sh` 时，shell 里的 Python 不是项目要求的 `3.9.19`。
- 项目依赖和 CUDA 扩展需要固定在正确 conda 环境内安装。

解决：

```bash
conda create -n vings_vio python=3.9.19
conda activate vings_vio
```

后续所有安装和运行都在 `vings_vio` 中进行：

```bash
conda run -n vings_vio bash set_env.sh
conda run -n vings_vio python scripts/run.py configs/hierarchical/smallcity.yaml
```

### 5.2 `set_env.sh` 原脚本不可重复执行

现象：

- 原脚本直接创建 conda 环境，不检查当前环境。
- 使用裸 `pip`，容易装到错误 Python 环境。
- 构建 DBAF 时使用 `sudo python setup.py install`，会脱离当前 conda 环境。

解决：

- 将 `set_env.sh` 改成：
  - 检查当前 conda 环境是否为 `vings_vio`。
  - 检查 Python 是否为 `3.9`。
  - 使用 `python -m pip`，保证安装到当前环境。
  - 不再使用 `sudo`。
  - 自动同步和初始化子模块。
  - 自动创建 `ckpts/` 和 `output/`。

关键逻辑：

```bash
ENV_NAME="${VINGS_CONDA_ENV:-vings_vio}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  echo "Please activate ${ENV_NAME}"
  exit 1
fi

python -m pip install --no-build-isolation -r requirements.txt

(
  cd "$REPO_ROOT/submodules/dbaf"
  python setup.py install
)
```

### 5.3 CUDA 扩展构建失败

现象：

- 本地 CUDA 扩展构建时，PEP517 build isolation 隔离环境里找不到 `torch`。
- `setuptools` 过新时，PyTorch 2.0.1 构建链路会遇到 `pkg_resources` 相关问题。

解决：

```bash
python -m pip install --no-build-isolation -r requirements.txt
python -m pip install setuptools==69.5.1
```

### 5.4 `gtsam==4.3a0` 不可用

现象：

- 当前 pip index 中找不到 `gtsam==4.3a0`。
- 依赖安装因此中断。

解决：

- 将 `requirements.txt` 中的 GTSAM 版本调整为：

```text
gtsam==4.2
```

补充说明：

- 代码中已有 `scripts/frontend/gtsam_compat.py`，用于兼容 public GTSAM 4.2 的接口。
- 当前导入验证通过。

### 5.5 子模块状态不一致

现象：

- `metric3d` 嵌套子模块的 gitdir 缺失，报错：

```text
fatal: not a git repository
```

- `eigen` / `gtsam` 子模块拉取时遇到 HTTP/2 或 TLS 中断。
- `diff-surfel-rasterization` 工作区只剩 `.git`，缺少 `setup.py`，导致 pip 无法安装。

解决：

- 修复 `metric3d` 子模块 gitdir，并 checkout 到父仓库要求的 commit：

```text
34afafe58d9543f13c01b65222255dab53333838
```

- 降低 git 子模块拉取并发并使用 HTTP/1.1：

```bash
git config http.version HTTP/1.1
git config submodule.fetchJobs 1
```

- 对 `eigen` 和 `gtsam` 使用 shallow update。
- 对 `diff-surfel-rasterization` 执行 `checkout -f HEAD` 恢复工作区文件。

### 5.6 Metric3D 目录结构和包装代码不匹配

现象：

运行 SmallCity 时最初报错：

```text
ModuleNotFoundError: No module named 'metric_modules.metric3d.mono.mono_utils'
```

原因：

- 当前 `metric3d` 子模块实际目录是：

```text
metric3d/mono/utils/
```

- 但 `submodules/metric_modules/metric.py` 里仍按旧路径导入：

```python
metric3d.mono.mono_utils
```

解决：

```python
from .metric3d.mono.utils.running import load_ckpt
from .metric3d.mono.utils.do_test import transform_test_data_scalecano, get_prediction
from .metric3d.mono.utils.mldb import load_data_info, reset_ckpt_path
from .metric3d.mono.utils.transform import gray_to_colormap
```

### 5.7 作者机器硬编码路径

现象：

`/root/autodl-tmp/VINGS-Mono/scripts/metric/metric_model.py` 原本写死了作者本机路径：

```python
sys.path.append('/data/wuke/workspace/VINGS-Mono/submodules/')
```

在当前机器 `/root/autodl-tmp/VINGS-Mono` 上运行时，可能出现：

```text
ModuleNotFoundError: No module named 'metric_modules'
```

解决：

- 改为根据当前文件位置动态定位仓库根目录：

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULES_DIR = REPO_ROOT / 'submodules'
if str(SUBMODULES_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMODULES_DIR))
```

- 同时将 Metric3D checkpoint 路径改成仓库内绝对定位：

```python
ckpt_path = REPO_ROOT / 'ckpts' / 'metric_depth_vit_small_800k.pth'
```

这样无论从哪个当前工作目录启动，只要仓库结构不变，都能找到 `submodules/` 和 `ckpts/`。

## 6. 当前完整实验步骤

### Step 1：进入项目目录

```bash
cd /root/autodl-tmp/VINGS-Mono
```

### Step 2：激活 conda 环境

```bash
conda activate vings_vio
```

或使用：

```bash
conda run -n vings_vio <command>
```

### Step 3：同步子模块

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

### Step 4：安装依赖并编译 CUDA 扩展

```bash
bash set_env.sh
```

### Step 5：确认 checkpoint

```bash
ls -lh ckpts
```

期望看到：

```text
droid.pth
metric_depth_vit_small_800k.pth
```

### Step 6：检查配置路径

```bash
sed -n '1,120p' configs/hierarchical/smallcity.yaml
```

重点检查：

```yaml
dataset:
  root: /root/autodl-tmp/VINGS-Mono/dataset/small_city/

output:
  save_dir: /root/autodl-tmp/VINGS-Mono/output/

frontend:
  weight: /root/autodl-tmp/VINGS-Mono/ckpts/droid.pth
```

### Step 7：运行 SmallCity

```bash
python scripts/run.py configs/hierarchical/smallcity.yaml
```

### Step 8：检查输出

```bash
ls -lh output/06-07-01-05-hierarchical-smallcit-/ply
```

当前结果：

```text
idx=876_2dgs.ply
intrinsic.yaml
```

## 7. 运行过程中的诊断方法

### 7.1 判断是不是 GPU 问题

命令：

```bash
nvidia-smi
```

观察点：

- 是否有 `python` 进程占用 GPU。
- 显存是否变化。
- GPU Util 是否大于 0。

实际观察：

- SmallCity 运行时 Python 进程占用过约 `4GB - 9GB` 显存。
- GPU Util 有明显计算占用。

因此当前阶段的主要问题不是 CUDA 不可见，而是环境、扩展、子模块和路径对齐问题。

### 7.2 判断是不是程序卡死

现象：

- `conda run` 有时不会实时显示 tqdm 输出。
- 长时间没有 stdout/stderr 不一定代表卡死。

有效判断方式：

```bash
ls -lh output/<run-dir>/droid_c2w
ls -lh output/<run-dir>/rgbdnua
ls -lh output/<run-dir>/map
ls -lh output/<run-dir>/bev
```

如果这些目录持续增长，说明程序仍在推进。

### 7.3 判断是否完整跑完

关键标志：

```text
output/<run-dir>/ply/idx=876_2dgs.ply
```

因为 `/root/autodl-tmp/VINGS-Mono/scripts/run.py` 中的保存逻辑是在最后一帧附近保存 PLY：

```python
if (idx == len(self.dataset) - 1) and self.mapper._xyz.shape[0] > 0:
    save_ply(self.mapper, idx, save_mode='2dgs')
```

所以 PLY 生成可以作为 SmallCity 完整跑完的重要标志。

## 8. 当前产物清单

最新输出目录：

```text
output/06-07-01-05-hierarchical-smallcit-/
```

目录内容：

```text
bev/
config.yaml
droid_c2w/
keyframelist.txt
map/
ply/
rgbdnua/
```

重要产物：

```text
ply/idx=876_2dgs.ply        # 最终 2DGS PLY，约 1.7GB
ply/intrinsic.yaml          # 相机内参
keyframelist.txt            # 关键帧列表
droid_c2w/*.txt             # 关键帧位姿
rgbdnua/*.png               # RGB/depth/normal/uncertainty/alpha 等中间可视化
map/*.png                   # map 可视化
bev/*.png                   # BEV 可视化
```

可用于汇报截图的示例：

```text
output/06-07-01-05-hierarchical-smallcit-/map/FrameId=00481.png
output/06-07-01-05-hierarchical-smallcit-/bev/FrameId=00481.png
```

## 9. 当前改动记录

当前与原始仓库相比，和本阶段相关的主要改动包括：

```text
configs/hierarchical/smallcity.yaml
requirements.txt
scripts/metric/metric_model.py
set_env.sh
submodules/metric_modules/metric.py
```

其中：

- `smallcity.yaml`：修改本地数据、输出和权重路径，并开启 `use_vis`。
- `requirements.txt`：加入/调整 `gtsam==4.2`。
- `metric_model.py`：去掉作者机器硬编码路径，动态定位仓库路径和 checkpoint。
- `set_env.sh`：改成可重复执行、环境安全、无 sudo 的安装脚本。
- `metric.py`：修复 Metric3D `mono_utils` 到 `utils` 的路径兼容问题。

注意：

- 工作区里还有一些早已存在或其他实验相关的改动，例如 `configs/ours/`、`dataset/`、`dataset_zips/`、`scripts/datasets/ours_vio.py` 等。
- 这些不是 SmallCity 当前阶段的核心修复点，汇报时可以暂时不展开。

## 10. 后续计划

下一阶段建议补充：

1. 打开并检查 `idx=876_2dgs.ply` 的可视化效果。
2. 从 `map/` 和 `bev/` 中挑选几张代表性截图，用于展示建图过程。
3. 记录完整运行耗时、GPU 峰值显存和平均显存。
4. 做轨迹质量检查，例如查看 `droid_c2w` 是否连续、是否有明显漂移。
5. 如果有 GT pose，可补充轨迹误差或定性对比。
6. 将修复内容整理为最小 patch，区分“环境必要修复”和“实验配置改动”。
7. 后续如果切换 Hotel / KITTI / 自定义数据集，复用本次路径和环境诊断流程。

## 11. 后续转 PPT 的拆页建议

这份 Markdown 后续可以拆成以下 PPT 页面：

1. 标题页：VINGS-Mono SmallCity 实验记录。
2. 阶段结论：SmallCity 已完整跑通，列出最终产物。
3. 实验目标：复现 Step 3 SmallCity。
4. 环境配置：conda、Python、PyTorch、CUDA、GPU。
5. 数据与权重：SmallCity 数据结构和两个 checkpoint。
6. 运行步骤：从环境到运行命令。
7. 困难 1：Python 环境和 CUDA 扩展。
8. 困难 2：子模块状态不一致。
9. 困难 3：Metric3D 路径和作者硬编码路径。
10. 诊断方法：GPU、输出目录、PLY 生成。
11. 当前实验结果：877 帧、310 关键帧、473 张可视化、1.7GB PLY。
12. 结果截图：map 和 BEV 示例。
13. 后续计划：质量评估、截图整理、轨迹/GT 对比。
