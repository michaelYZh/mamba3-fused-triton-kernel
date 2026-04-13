# GPU 环境配置指南 & 会话交接文档

> **最后更新**: 2026-04-13
> **目标**: 确保任何新会话都能快速恢复 GPU 开发环境

---

## 1. GPU 实例选型记录

### 1.1 租用平台: Vast.ai

| 配置项 | 选择 | 原因 |
|--------|------|------|
| **GPU 型号** | A100-PCIE-40GB | 单卡开发 SXM4 无优势，省 ~50% 费用 |
| **价格** | $0.51/hr | 性价比最优 |
| **Template** | PyTorch (Vast) | 预装 PyTorch + CUDA，最干净 |
| **可靠性 (Reliability)** | ≥ 0.9 | 低可靠性实例频繁断连 |
| **Compute Capability** | ≥ 8.0 (Ampere+) | Triton 要求 Ampere 架构 |
| **显存要求** | ≥ 40GB | 当前项目够用，大 batch 再升级 |

### 1.2 为什么不选其他选项

- ❌ A100-SXM4-80GB @ $1.072/hr — 单卡无 NVLink 优势，贵一倍
- ❌ NVIDIA CUDA Template — 太基础，需手动装 Python/PyTorch
- ❌ vLLM/Ollama/AIQ — LLM 推理专用，环境太杂
- ❌ RTX 3090/4090 — 显存 24GB 偏紧，非数据中心卡不稳定
- ❌ V100 — Turing 架构 (SM_70)，Triton 支持有问题

### 1.3 升级时机 (何时换 80GB)

- [ ] 做 **大 batch benchmark** (batch > 256) 显存 OOM
- [ ] 跑 **nsight compute profiling** + 大模型同时吃紧
- [ ] 测试 **更大 d_model (1024/2048)** 的模型
- [ ] 最终性能调优时想要更充裕的空间

---

## 2. 一键配置流程

### 2.1 SSH 连接后执行 (复制即用)

```bash
# ===== Step 1: 克隆代码 =====
gh repo clone michaelYZh/mamba3-fused-triton-kernel
cd mamba3-fused-triton-kernel

# ===== Step 2: 初始化 submodules =====
gh repo clone michaelYZh/mamba3-fused-triton-kernel -- --recurse-submodules
# 或者如果已克隆但 submodule 缺失:
git submodule update --init --recursive

# ===== Step 3: 运行一键配置脚本 =====
bash scripts/setup_gpu_env.sh

# ===== Step 4: 验证环境 =====
conda activate mamba3
pytest src/tests/ -v

# ===== Step 5: 跑 benchmark =====
python benchmarks/run_bench.py --mode both
```

### 2.2 脚本会自动完成以下工作

| Step | 内容 | 预计耗时 |
|------|------|---------|
| 1. 系统检测 | OS / NVIDIA驱动 / GPU型号 / CUDA版本 | < 10s |
| 2. Conda 环境 | 创建/更新 `mamba3` 环境 (Python 3.11) | ~2 min |
| 3. 安装依赖 | PyTorch + Triton + einops + pytest 等 | ~3-5 min |
| 4. Nsight Compute | 可选 (`--with-nsight`) | ~2 min |
| 5. 代码部署 | clone + submodule 初始化 | ~30s |
| 6. 环境验证 | 检测所有包 + CUDA 可用性 | < 10s |
| 7. Smoke Test | 编译 kernel + 小规模正确性测试 | ~1-2 min |

**总计**: 约 **7-10 分钟**

---

## 3. 会话交接检查清单

> 新会话开始时，按此清单确认环境状态。

### 3.1 快速状态检查

```bash
# ===== 1. GPU 状态 =====
nvidia-smi
# 期望: A100-PCIE-40GB, 显存大部分空闲

# ===== 2. 项目目录 =====
ls ~/mamba3-fused-triton-kernel/
# 期望: benchmarks/ docs/ notes/ references/ scripts/ src/

# ===== 3. Git 状态 =====
cd ~/mamba3-fused-triton-kernel && gh repo status
# 期望: clean 或有未提交的改动

# ===== 4. Conda 环境 =====
conda env list | grep mamba3
# 期望: mamba3 * (已激活) 或 mamba3 (存在)

# ===== 5. 关键包验证 =====
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import triton; print(f'Triton {triton.__version__}')"
# 期望: 版本号 + GPU 名称 = A100-PCIe...
```

### 3.2 如果环境有问题

| 问题 | 解决方案 |
|------|---------|
| conda 环境不存在 | `bash scripts/setup_gpu_env.sh` 重新运行 |
| Triton 导入失败 | `pip install triton --upgrade --force-reinstall` |
| CUDA 不可用 | 检查 nvidia-smi → 重装 PyTorch 匹配的 CUDA 版本 |
| gh 未认证 | `gh auth login` |
| submodule 缺失 | `git submodule update --init --recursive` |

---

## 4. 日常同步工作流

### 4.1 本地 ↔ GitHub ↔ GPU 三端同步

```
本地 Mac              GitHub                  GPU (vast.ai)
   │                    │                        │
   │ gh repo sync ───→  │  ←─── gh repo clone   │
   │                    │                        │
   │ ←── gh repo clone  │  ───── gh repo sync ─→ │
   │                    │                        │
```

### 4.2 常用操作速查 (使用 gh CLI)

```bash
# ========== 从本地推送改动到 GitHub ==========
git add -A && git commit -m "描述改动" && git push origin main

# ========== 在 GPU 上拉取最新代码 ==========
cd ~/mamba3-fused-triton-kernel && git pull origin main

# ========== 在 GPU 上提交 benchmark 结果 ==========
git add -A && git commit -m "benchmark: xxx 结果" && git push

# ========== 本地拉取 GPU 的改动 ==========
git pull origin main

# ========== 创建 PR ==========
gh pr create --title "描述" --body "详细说明"

# ========== 查看 repo 状态 ==========
gh repo view michaelYZh/mamba3-fused-triton-kernel

# ========== 查看 CI 状态 ==========
gh run list
```

### 4.3 冲突处理

如果两端都有修改导致冲突：

```bash
git stash                    # 暂存当前改动
git pull origin main         # 拉取远程
git stash pop                # 恢复暂存
# 手动解决冲突后
git add -A && git commit && git push
```

---

## 5. 项目待办事项

### 5.1 已完成

- [x] 项目脚手架和规划文档
- [x] 环境搭建 + Mamba3 算法深入理解
- [x] **SISO fused kernel** (核心 + trapezoidal + RoPE)
- [x] **MIMO fused kernel** (R=4 rank-1 updates)
- [x] 正确性测试 (Triton vs PyTorch reference, atol < 1e-2)
- [x] GitHub 仓库初始化并推送
- [x] GPU 环境配置脚本 (`setup_gpu_env.sh`)
- [x] Submodule 引用 (`references/mamba3-pytorch`)

### 5.2 待完成 — 性能优化

- [ ] Block size tuning (D tiling for D=64/128/256)
- [ ] Dtype optimization (fp16/bf16 中间变量)
- [ ] Register pressure analysis via `triton.testing`
- [ ] Cache hint optimization
- [ ] `torch.compile` 兼容性测试

### 5.3 待完成 — 完整 Benchmark + 文档

- [ ] 全量 benchmark suite (多种 batch/seq_len/SISO vs MIMO)
- [ ] 性能分析报告
- [ ] README 更新 (安装/使用/结果)
- [ ] End-to-end demo

### 5.4 成功标准

| 指标 | MVP 目标 | 优秀 |
|------|---------|------|
| **正确性** | vs PyTorch atol < 1e-2 | atol < 1e-3 |
| **单步延迟 (batch=1)** | 比 Python 快 1.5x | 比 Python 快 2x+ |
| **端到端 decode (seq=1K)** | 比 Python 快 1.3x | 比 Python 快 1.5x+ |

---

## 6. 关键文件索引

| 文件 | 用途 | 说明 |
|------|------|------|
| `src/kernels/siso_decode.py` | SISO fused kernel | 核心代码 |
| `src/kernels/mimo_decode.py` | MIMO fused kernel | 核心代码 |
| `src/kernels/utils.py` | 共享工具函数 | RoPE, trapezoidal, PyTorch 参考 |
| `src/models/mamba3.py` | Mamba3 模型定义 | 含训练 forward + 单步 decode |
| `src/models/inference.py` | 推理接口 | Triton/PyTorch 后端切换 |
| `benchmarks/run_bench.py` | Benchmark 主脚本 | 延迟对比测试 |
| `scripts/setup_gpu_env.sh` | GPU 一键配置 | **新实例必运行** |
| `docs/plan.md` | MVP 完整计划 | 算法细节 |
| `docs/gpu-setup-guide.md` | 本文档 | 会话交接 |

---

## 7. 故障排除

### 7.1 常见问题

**Q: Triton 编译报错 `ptxas error`**
```bash
# 可能是 block size 不兼容，尝试减小 BLOCK_D
# 或检查 CUDA 版本是否匹配
nvcc --version  # 需要 11.8+
```

**Q: OOM (Out of Memory)**
```bash
# 减小 batch size 或 seq_len
python benchmarks/run_bench.py --mode siso --batch-sizes 1,8,32

# 如果还是 OOM，先清理显存
python -c "import torch; torch.cuda.empty_cache()"
```

**Q: GPU 实例被回收或断连**
```bash
# vast.ai 实例可能因各种原因断连
# 重新租用后直接跑:
cd ~/mamba3-fused-triton-kernel || {
  gh repo clone michaelYZh/mamba3-fused-triton-kernel
  cd mamba3-fused-triton-kernel
}
bash scripts/setup_gpu_env.sh --skip-clone  # 跳过克隆，只配环境
```

**Q: `gh` 未认证**
```bash
gh auth login
# 按提示选择 GitHub.com → HTTPS → 浏览器认证
```

**Q: Git push 报错 `permission denied`**
```bash
# 确认 gh 认证状态
gh auth status

# 如果未认证或 token 过期
gh auth login

# 检查远程 URL
git remote -v
# 应该显示: https://github.com/michaelYZh/mamba3-fused-triton-kernel.git
```

---

## 8. 费用参考

| 项目 | 价格 | 备注 |
|------|------|------|
| A100-PCIE-40GB | $0.51/hr | 当前选择 |
| 预估开发总时长 | 20-30 hrs | ~$10-15 |
| 如升级 80GB | +$0.56/hr | 按需升级 |

---

*本文档确保所有新会话都能无缝接续 GPU 开发工作。*
