#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${VINGS_CONDA_ENV:-vings_vio}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  cat <<EOF
Please create and activate the expected conda environment first:

  conda create -n ${ENV_NAME} python=3.9.19
  conda activate ${ENV_NAME}
  bash set_env.sh
EOF
  exit 1
fi

python - <<'PY_CHECK'
import sys

if sys.version_info[:2] != (3, 9):
    raise SystemExit(
        f"Python 3.9 is required for VINGS-Mono; got {sys.version.split()[0]}"
    )
PY_CHECK

git submodule sync --recursive
git submodule update --init --recursive

mkdir -p "$REPO_ROOT/ckpts" "$REPO_ROOT/output"

python -m pip install --upgrade pip wheel
python -m pip install setuptools==69.5.1
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
python -m pip install --no-build-isolation -r requirements.txt

# Build the VIO GTSAM fork into the active conda environment. The public
# PyPI gtsam wheel does not provide all methods used by this repository.
cmake -S "$REPO_ROOT/submodules/gtsam" -B "$REPO_ROOT/submodules/gtsam/build_vio" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DGTSAM_BUILD_PYTHON=ON \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_WITH_TBB=OFF
cmake --build "$REPO_ROOT/submodules/gtsam/build_vio" --target install -j"${VINGS_BUILD_JOBS:-$(nproc)}"

# Build CUDA extensions into the active conda environment, without sudo.
(
  cd "$REPO_ROOT/submodules/dbaf"
  python setup.py install
)
