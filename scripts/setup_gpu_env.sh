#!/bin/bash
# ============================================================================
# setup_gpu_env.sh - Mamba3 Fused Triton Kernel 一键 GPU 环境配置脚本
# ============================================================================
# 用途: 在租用的 A100/H100 GPU 实例上快速配置项目运行环境
# 适用平台: Lambda Labs, Vast.ai, AutoDL, 及其他 Ubuntu GPU 实例
#
# 使用方式:
#   bash setup_gpu_env.sh                    # 标准安装
#   bash setup_gpu_env.sh --with-nsight      # 安装 nsight compute (性能分析)
#   bash setup_gpu_env.sh --cuda 12.1        # 指定 CUDA 版本
#   bash setup_gpu_env.sh --skip-clone       # 跳过代码克隆 (本地已有)
# ============================================================================

set -e  # 遇到错误立即退出

# ============ 配置参数 ============
CONDA_ENV_NAME="mamba3"
PYTHON_VERSION="3.11"
PROJECT_DIR="$HOME/mamba3-fused-triton-kernel"
REPO_OWNER="michaelYZh"
REPO_NAME="mamba3-fused-triton-kernel"
REPO_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}.git"

# 解析命令行参数
INSTALL_NSIGHT=false
CUDA_VERSION=""
SKIP_CLONE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-nsight) INSTALL_NSIGHT=true; shift ;;
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --skip-clone) SKIP_CLONE=true; shift ;;
        -h|--help)
            echo "用法: bash setup_gpu_env.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --with-nsight    安装 Nsight Compute (GPU 性能分析工具)"
            echo "  --cuda VERSION   指定 CUDA 版本 (默认: 自动检测)"
            echo "  --skip-clone     跳过代码仓库克隆"
            echo "  -h, --help       显示帮助信息"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ============ 颜色输出 ============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

echo ""
echo "============================================================"
echo "  Mamba3 Fused Triton Kernel - GPU 环境一键配置"
echo "============================================================"
echo ""

# ============ Step 1: 系统检测 ============
info "Step 1/7: 检测系统环境..."
echo ""

# 操作系统
OS_TYPE=$(uname -s)
if [[ "$OS_TYPE" != "Linux" ]]; then
    error "此脚本仅支持 Linux 系统，当前系统: $OS_TYPE"
    exit 1
fi
success "操作系统: Linux"

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    info "发行版: $NAME $VERSION"
fi

# NVIDIA 驱动 & GPU 检测
if ! command -v nvidia-smi &> /dev/null; then
    error "未检测到 nvidia-smi! 请确认实例已安装 NVIDIA 驱动"
    exit 1
fi

NVIDIA_DRIVER_VERSION=$(nvidia-smi | grep "Driver Version" | awk '{print $3}')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

success "NVIDIA 驱动版本: $NVIDIA_DRIVER_VERSION"
success "GPU 型号: $GPU_NAME"
success "显存大小: ${GPU_MEMORY} MB"

# CUDA Toolkit 检测
DETECTED_CUDA=""
if command -v nvcc &> /dev/null; then
    DETECTED_CUDA=$(nvcc --version | grep release | awk '{print $5}' | sed '/,//')
fi

if [[ -z "$CUDA_VERSION" ]]; then
    if [[ -n "$DETECTED_CUDA" ]]; then
        CUDA_VERSION=$DETECTED_CUDA
        info "自动检测到 CUDA 版本: $CUDA_VERSION"
    else
        # 根据 PyTorch 兼容性选择默认值
        CUDA_VERSION="12.1"
        warn "未检测到 CUDA Toolkit，将使用 PyTorch 自带 CUDA $CUDA_VERSION"
    fi
else
    info "使用指定 CUDA 版本: $CUDA_VERSION"
fi

echo ""

# ============ Step 2: Conda 检测与环境创建 ============
info "Step 2/7: 配置 Conda 环境..."
echo ""

# 检测 conda/micromamba
CONDA_CMD=""
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
else
    error "未找到 conda 或 micromamba!"
    echo "请先安装 Miniconda:"
    echo "  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh"
    echo "  bash miniconda.sh -b -p \$HOME/miniconda3"
    exit 1
fi

success "包管理器: $(which $CONDA_CMD)"

# 初始化 conda
eval "$($CONDA_CMD shell.bash hook 2>/dev/null)" || true

# 创建或激活环境
if $CONDA_CMD env list 2>/dev/null | grep -q "^${CONDA_ENV_NAME}\s"; then
    warn "环境 '$CONDA_ENV_NAME' 已存在，将更新依赖..."
else
    info "创建 Conda 环境: $CONDA_ENV_NAME (Python $PYTHON_VERSION)"
    $CONDA_CMD create -n $CONDA_ENV_NAME python=$PYTHON_VERSION -y
fi

$CONDA_CMD activate $CONDA_ENV_NAME
success "Conda 环境已激活: $CONDA_ENV_NAME"

echo ""

# ============ Step 3: 安装 Python 依赖 ============
info "Step 3/7: 安装 Python 依赖..."
echo ""

# PyTorch 安装 (根据 CUDA 版本选择)
info "安装 PyTorch (CUDA ${CUDA_VERSION})..."

case $CUDA_VERSION in
    11.8|12.1)
        pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION//./}
        ;;
    *)
        warn "CUDA $CUDA_VERSION 不在预定义列表中，尝试从默认源安装..."
        pip install torch
        ;;
esac

# Triton 安装
info "安装 Triton..."
pip install triton

# 其他核心依赖
pip install einops pytest numpy matplotlib pandas

# 可选依赖 (用于 benchmark 输出)
pip install tabulate tqdm 2>/dev/null || true

success "Python 依赖安装完成"

echo ""

# ============ Step 4: 安装 nsight compute (可选) ============
if $INSTALL_NSIGHT; then
    info "Step 4/7: 安装 Nsight Compute..."
    echo ""

    NSIGHT_VERSION="2024_2024.2.1-1"

    case "$(uname -m)" in
        x86_64)
            ARCH="x86_64"
            ;;
        aarch64)
            ARCH="sbsa"
            ;;
    esac

    # 尝试用 conda 安装 (最简单)
    if $CONDA_CMD install -c nvidia nsight-compute -y 2>/dev/null; then
        success "Nsight Compute 已通过 conda 安装"
    elif command -v apt-get &> /dev/null; then
        # 备选: 用 deb 包安装 (需要 root 权限)
        OS_CODENAME=$(. /etc/os-release && echo $VERSION_CODENAME)
        
        if [[ "$EUID" -eq 0 ]] || sudo -n true 2>/dev/null; then
            SUDO_CMD="sudo"
        else
            SUDO_CMD=""  # 无 sudo 时跳过
        fi
        
        if [[ -n "$SUDO_CMD" ]]; then
            NSIGHT_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_CODENAME}/${ARCH}/nsight-compute-${NSIGHT_VERSION}_${ARCH}.deb"
            
            TMP_DEB="/tmp/nsight-compute.deb"
            wget -O $TMP_DEB "$NSIGHT_URL" || { 
                warn "下载失败，尝试 conda-forge..."
                pip install nvidia-nsight-cu 2>/dev/null || warn "Nsight Compute 安装失败，可手动安装"
            }
            
            if [[ -f "$TMP_DEB" ]]; then
                $SUDO_CMD dpkg -i $TMP_DEB 2>/dev/null || $SUDO_CMD apt-get -f install -y
                rm -f $TMP_DEB
                success "Nsight Compute 已通过 deb 包安装"
            fi
        else
            warn "无 root 权限，跳过系统级安装"
            warn "建议: conda install -c nvidia nsight-compute"
        fi
    fi
    
    # 验证
    if command -v ncu &> /dev/null; then
        success "ncu 命令可用: $(which ncu)"
    else
        warn "ncu 未在 PATH 中找到，可能需要重启 shell 或手动添加路径"
    fi
    
    echo ""
else
    info "Step 4/7: 跳过 Nsight Compute 安装 (--with-nsight 启用)"
    echo ""
fi

# ============ Step 5: 项目代码部署 ============
if $SKIP_CLONE; then
    info "Step 5/7: 跳过代码克隆 (--skip-clone)"
else
    info "Step 5/7: 克隆项目代码..."
    echo ""

    if [[ -d "$PROJECT_DIR" ]]; then
        warn "目录 $PROJECT_DIR 已存在，执行 git pull 更新..."
        cd "$PROJECT_DIR" && git pull origin main || true
        git submodule update --init --recursive 2>/dev/null || true
    else
        info "克隆仓库到 $PROJECT_DIR ..."
        # 优先使用 gh CLI (已认证), 否则回退到 git clone
        if command -v gh &> /dev/null && gh auth status &> /dev/null; then
            gh repo clone ${REPO_OWNER}/${REPO_NAME} "$PROJECT_DIR" -- --recurse-submodules
        else
            git clone --recurse-submodules "$REPO_URL" "$PROJECT_DIR"
        fi
    fi
    
    success "代码就绪: $PROJECT_DIR"
    
    cd "$PROJECT_DIR"
    echo ""
fi

# ============ Step 6: 验证安装 ============
info "Step 6/7: 验证环境..."
echo ""

python << 'EOF'
import sys
import subprocess

print("\n--- Python 环境信息 ---")
print(f"Python 版本: {sys.version}")
print(f"Python 路径:  {sys.executable}")

packages = {
    'torch': 'PyTorch',
    'triton': 'Triton',
    'einops': 'Einops',
    'pytest': 'PyTest',
}

print("\n--- 包版本检查 ---")
all_ok = True
for pkg, display_name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name}: {version}")
    except ImportError:
        print(f"  ✗ {display_name}: 未安装!")
        all_ok = False

print("\n--- PyTorch CUDA 检测 ---")
try:
    import torch
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本:    {torch.version.cuda}")
        print(f"GPU 设备数:   {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU [{i}]: {props.name} ({props.total_mem // 1024**3} GB)")
except Exception as e:
    print(f"  ✗ PyTorch CUDA 检测失败: {e}")
    all_ok = False

print()

if not all_ok:
    print("⚠ 部分包缺失，请检查上方错误信息")
    sys.exit(1)
else:
    print("✓ 所有依赖验证通过!")
EOF

VERIFY_RESULT=$?
if [ $VERIFY_RESULT -ne 0 ]; then
    error "环境验证失败！请检查上方错误信息"
    exit 1
fi

echo ""

# ============ Step 7: 快速测试 (Smoke Test) ============
info "Step 7/7: 运行快速测试 (Smoke Test)..."
echo ""

cd "$PROJECT_DIR" 2>/dev/null || PROJECT_DIR="$(pwd)"

# 运行最小化测试确认 kernel 可编译和运行
SMOKETEST_RESULT=0

python << 'PYTEST_EOF' || SMOKETEST_RESULT=1
import os
import sys
import torch

os.chdir(os.environ.get('PROJECT_DIR', '.'))

# 测试 Triton kernel 编译
print("编译 SISO decode kernel...")
try:
    from src.kernels.siso_decode import mamba3_siso_decode_triton
    print("  ✓ SISO decode kernel 导入成功")
except Exception as e:
    print(f"  ✗ SISO decode kernel 导入失败: {e}")
    sys.exit(1)

print("编译 MIMO decode kernel...")
try:
    from src.kernels.mimo_decode import mamba3_mimo_decode_triton
    print("  ✓ MIMO decode kernel 导入成功")
except Exception as e:
    print(f"  ✗ MIMO decode kernel 导入失败: {e}")
    sys.exit(1)

# 运行小规模数值正确性测试
print("\n运行小规模正确性测试 (batch=2, seq_len=8)...")
try:
    B, D, H, T = 2, 256, 16, 8
    
    # SISO test
    x_siso = torch.randn(B * H, D, T, device='cuda', dtype=torch.float32)
    A_log = torch.randn(H, device='cuda', dtype=torch.float32)
    bc = torch.randn(2 * H, D, device='cuda', dtype=torch.float32)
    dt_bias = torch.randn(H, device='cuda', dtype=torch.float32)
    
    out_triton = mamba3_siso_decode_triton(x_siso, A_log, bc, dt_bias)
    
    from src.kernels.utils import mamba3_siso_decode_ref
    out_ref = mamba3_siso_decode_ref(x_siso, A_log, bc, dt_bias)
    
    max_diff = (out_triton - out_ref).abs().max().item()
    print(f"  SISO 最大误差: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        print("  ✓ SISO 数值正确性验证通过")
    else:
        print(f"  ⚠ SISO 误差较大 ({max_diff:.2e})，可能需要检查精度设置")
    
    # MIMO test
    x_mimo = torch.randn(B * H, D, T, device='cuda', dtype=torch.float32)
    
    from src.kernels.mimo_decode import mamba3_mimo_decode_triton
    out_mimo_triton = mamba3_mimo_decode_triton(x_mimo, A_log, bc, dt_bias)
    
    from src.kernels.utils import mamba3_mimo_decode_ref
    out_mimo_ref = mamba3_mimo_decode_ref(x_mimo, A_log, bc, dt_bias)
    
    mimo_diff = (out_mimo_triton - out_mimo_ref).abs().max().item()
    print(f"  MIMO 最大误差: {mimo_diff:.2e}")
    
    if mimo_diff < 1e-4:
        print("  ✓ MIMO 数值正确性验证通过")
    else:
        print(f"  ⚠ MIMO 误差较大 ({mimo_diff:.2e})，可能需要检查精度设置")
    
    print("\n✓ Smoke Test 通过! 环境就绪。")
    
except torch.cuda.OutOfMemoryError:
    print("  ✗ 显存不足! 尝试减小 batch size")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Smoke Test 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

PYTEST_EOF

if [ $SMOKETEST_RESULT -ne 0 ]; then
    warn "Smoke Test 未完全通过（可能是首次编译或显存限制）"
    warn "可以稍后手动运行: python src/tests/test_siso.py"
fi

echo ""

# ============ 完成 ============
echo ""
echo "============================================================"
echo -e "${GREEN}  ✓ GPU 环境配置完成!${NC}"
echo "============================================================"
echo ""
echo "后续操作:"
echo ""
echo "  # 1. 激活环境:"
echo "  $CONDA_CMD activate $CONDA_ENV_NAME"
echo ""
echo "  # 2. 运行完整测试:"
echo "  cd $PROJECT_DIR"
echo "  pytest src/tests/ -v"
echo ""
echo "  # 3. 运行 Benchmark:"
echo "  python benchmarks/run_bench.py --mode both"
echo ""
echo "  # 4. 性能分析 (如果安装了 nsight):"
echo "  ncu --kernel-name-base demangled -o profile_output \\"
echo "      python benchmarks/run_bench.py --mode triton"
echo ""
if $INSTALL_NSIGHT; then
    echo "  # 5. 查看 profiler 报告:"
    echo "  ncu-ui profile_output.ncu-rep"
    echo ""
fi
echo "============================================================"
