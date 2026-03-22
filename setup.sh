#!/usr/bin/env bash
# ============================================================
# OpenClaw-RL 一键安装脚本
# 前提：uv 虚拟环境（激活状态），CUDA 12.2+ 驱动
# 用法：bash setup.sh
# ============================================================
set -euo pipefail

# uv 缓存和 venv 可能在不同文件系统上（计算节点常见），hardlink 会失败
# 直接用 copy 模式，避免每个包都打一次 warning
export UV_LINK_MODE=copy

# CUDA 12.2 适配
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# PyTorch C++ extensions 需要 GCC 9+，但 CUDA 12.2 nvcc 只支持 GCC ≤ 12
# 优先选 gcc-toolset-12，兼顾两边要求
if ! gcc -dumpversion 2>/dev/null | awk -F. '{exit ($1 >= 9 && $1 <= 12 ? 1 : 0)}'; then
  for _ts in 12 11; do
    _enable="/opt/rh/gcc-toolset-${_ts}/enable"
    if [ -f "$_enable" ]; then
      echo "  ⚠ 系统 GCC 不在 9-12 范围，启用 gcc-toolset-${_ts}"
      source "$_enable"
      break
    fi
  done
fi

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================="
echo " OpenClaw-RL 环境安装"
echo " 仓库路径: $REPO_DIR"
echo "=============================="

# --------------------------------------------------
# 0. 检查前置条件
# --------------------------------------------------
echo ""
echo "[0/6] 检查前置条件..."

python --version 2>/dev/null && echo "  ✓ Python 可用" \
  || { echo "  ✗ 未找到 python，请确认虚拟环境已激活"; exit 1; }

command -v uv >/dev/null 2>&1 && echo "  ✓ uv $(uv --version 2>/dev/null)" \
  || echo "  ⚠ 未找到 uv，将使用 pip"

if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
else
  PIP="python -m pip"
fi

# 确保关闭 system-site-packages（避免系统包和 venv 包冲突，尤其 torch）
VENV_DIR="${VIRTUAL_ENV:-$(python -c "import sys; print(sys.prefix)" 2>/dev/null)}"
PYVENV_CFG="$VENV_DIR/pyvenv.cfg"
if [ -f "$PYVENV_CFG" ] && grep -q "include-system-site-packages = true" "$PYVENV_CFG"; then
  echo "  ⚠ 检测到 system-site-packages=true，关闭以避免 torch 循环 import 冲突"
  sed -i 's/include-system-site-packages = true/include-system-site-packages = false/' "$PYVENV_CFG"
fi

# --------------------------------------------------
# 1. 安装 PyTorch — 如系统已有则跳过
#    注意: fallback 版本是 2.4.1+cu121，适用于没有预装 torch 的环境
#    如果预装了 torch 2.9.1+cu128，后续 CUDA 编译步骤会自动适配
# --------------------------------------------------
echo ""
echo "[1/6] 检查 PyTorch..."

if python -c "import torch; print(f'  ✓ PyTorch {torch.__version__} (CUDA {torch.version.cuda}) 已可用，跳过')" 2>/dev/null; then
  :
else
  echo "  → 系统也没有 PyTorch，正在安装 torch==2.4.1+cu121..."
  $PIP install \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    torchaudio==2.4.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  echo "  ✓ PyTorch 安装完成"
fi

# --------------------------------------------------
# 2. 安装 requirements.txt 中的普通依赖
#    排除：torch/nvidia 系列、git+ 依赖、系统包、重型 CUDA 编译包
# --------------------------------------------------
echo ""
echo "[2/6] 安装 Python 依赖 (requirements.txt)..."

# 需排除的包（系统包装不了 / 已装 / 后面单独装 / 需 CUDA 编译）
EXCLUDE_PATTERN='(^torch==|^torchvision==|^torchaudio==|^torchao==|^nvidia-|^git\+|.*@ git\+|^dbus-python|^PyGObject|^devscripts|^transformer_engine|^transformer_engine_cu12|^transformer_engine_torch|^flash-attn|^flash_attn|^flashinfer)'

grep -v -E "$EXCLUDE_PATTERN" "$REPO_DIR/requirements.txt" \
  > /tmp/openclaw-rl-filtered-requirements.txt

echo "  → 安装常规依赖 (已排除 torch/nvidia/系统包/CUDA编译包)..."
$PIP install -r /tmp/openclaw-rl-filtered-requirements.txt
rm /tmp/openclaw-rl-filtered-requirements.txt
echo "  ✓ 基础依赖安装完成"

# --------------------------------------------------
# 3. 安装 git+ 源码依赖
# --------------------------------------------------
echo ""
echo "[3/6] 安装 git+ 源码依赖..."

# sglang (从 sgl-project fork)
$PIP install "sglang @ git+https://github.com/sgl-project/sglang.git@dce8b0606c06d3a191a24c7b8cbe8e238ab316c9#subdirectory=python"
echo "  ✓ sglang"

# megatron_core
$PIP install "megatron_core @ git+https://github.com/NVIDIA/Megatron-LM.git@3714d81d418c9f1bca4594fc35f9e8289f652862"
echo "  ✓ megatron_core"

# mbridge
$PIP install "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c"
echo "  ✓ mbridge"

# megatron-bridge
$PIP install "megatron-bridge @ git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a"
echo "  ✓ megatron-bridge"

# torch_memory_saver
$PIP install "torch_memory_saver @ git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b"
echo "  ✓ torch_memory_saver"

# --------------------------------------------------
# 4. 本地 editable install (slime + int4_qat kernel)
# --------------------------------------------------
echo ""
echo "[4/6] 安装本地包 (editable)..."

# 本地 slime（用本地开发版）
$PIP install -e "$REPO_DIR/slime/"
echo "  ✓ slime (local editable)"

# int4_qat CUDA kernel
$PIP install -e "$REPO_DIR/slime/slime/backends/megatron_utils/kernels/int4_qat" --no-build-isolation
echo "  ✓ int4_qat kernel"

# --------------------------------------------------
# 5. 源码编译依赖 (apex + flash-attn + transformer_engine)
# --------------------------------------------------
echo ""
echo "[5/6] 编译安装 CUDA 依赖 (耗时较长)..."

# transformer_engine (如系统有则跳过)
if python -c "import transformer_engine" 2>/dev/null; then
  echo "  ✓ transformer_engine 已安装，跳过"
else
  echo "  → 安装 transformer_engine==2.10.0 (CUDA 编译，可能需要 20-60 分钟)..."
  # transformer_engine / _cu12 是纯 wheel，正常装
  $PIP install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0
  # transformer_engine_torch 需要源码编译，必须 --no-build-isolation 使用当前 venv 的 torch
  $PIP install transformer_engine_torch==2.10.0 --no-build-isolation
  echo "  ✓ transformer_engine"
fi

# apex
if python -c "import apex" 2>/dev/null; then
  echo "  ✓ apex 已安装，跳过"
else
  APEX_TMP=$(mktemp -d)
  echo "  → 克隆 apex 到 $APEX_TMP ..."
  git clone --depth 1 https://github.com/NVIDIA/apex.git "$APEX_TMP/apex"
  cd "$APEX_TMP/apex"
  # apex 会严格检查 torch CUDA 版本 == nvcc 版本，12.8 vs 12.2 minor mismatch 是安全的
  # 参考 https://github.com/NVIDIA/apex/pull/323#discussion_r287021798
  # raise RuntimeError(...) 跨多行，sed 无法匹配，用 python 替换
  python -c "
import re, pathlib
p = pathlib.Path('setup.py')
src = p.read_text()
src = re.sub(r'def check_cuda_torch_binary_vs_bare_metal.*?(?=\ndef )', 'def check_cuda_torch_binary_vs_bare_metal(cuda_dir):\n    import warnings; warnings.warn(\"Skipping CUDA version check\")\n\n', src, flags=re.DOTALL)
p.write_text(src)
"
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 $PIP install -v --no-build-isolation .
  cd "$REPO_DIR"
  rm -rf "$APEX_TMP"
  echo "  ✓ apex 编译安装完成"
fi

# flash-attn
if python -c "import flash_attn" 2>/dev/null; then
  echo "  ✓ flash-attn 已安装，跳过"
else
  echo "  → 编译安装 flash-attn==2.7.4.post1 (MAX_JOBS=8)..."
  MAX_JOBS=8 $PIP install --no-build-isolation -v flash-attn==2.7.4.post1
  echo "  ✓ flash-attn"
fi

# flashinfer — 索引需匹配 torch 的 CUDA 版本
TORCH_CUDA_VER=$(python -c "import torch; v=torch.version.cuda; print(f'cu{v.split(\".\")[0]}{v.split(\".\")[1]}')" 2>/dev/null || echo "cu128")
if python -c "import flashinfer_jit_cache" 2>/dev/null; then
  echo "  ✓ flashinfer-jit-cache 已安装，跳过"
else
  $PIP install "flashinfer-jit-cache==0.5.3" --extra-index-url "https://flashinfer.ai/whl/${TORCH_CUDA_VER}"
  echo "  ✓ flashinfer-jit-cache"
fi

# --------------------------------------------------
# 6. 验证安装
# --------------------------------------------------
echo ""
echo "[6/6] 验证关键依赖..."

python -c "
pkgs = [
    ('torch',               'torch'),
    ('transformers',        'transformers'),
    ('accelerate',          'accelerate'),
    ('ray',                 'ray'),
    ('sglang',              'sglang'),
    ('slime',               'slime'),
    ('megatron.core',       'megatron.core'),
    ('peft',                'peft'),
    ('flash_attn',          'flash_attn'),
    ('apex',                'apex'),
    ('transformer_engine',  'transformer_engine'),
    ('wandb',               'wandb'),
    ('fastapi',             'fastapi'),
]
ok, fail = 0, 0
for name, mod in pkgs:
    try:
        __import__(mod)
        print(f'  ✓ {name}')
        ok += 1
    except ImportError:
        print(f'  ✗ {name} — 未安装')
        fail += 1
print(f'\n  结果: {ok} 成功, {fail} 失败')
"

echo ""
echo "=============================="
echo " 安装完成！"
echo ""
echo " 快速开始（Combined 方法，推荐）:"
echo "   cd $REPO_DIR/slime"
echo "   bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh"
echo "=============================="
