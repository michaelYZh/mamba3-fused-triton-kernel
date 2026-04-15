# Experiment Log

Development progress and experiment results.

---

## MVP Phase (2026-04-05 ~ 2026-04-13)

All MVP milestones completed. See:
- `docs/plan.md` — Full execution plan with completion status
- `docs/gpu-setup-guide.md` — Project TODO and session handover
- `README.md` — Milestones and benchmark results

### Key Results

- **SISO + MIMO fused Triton kernels**: Correctness validated (atol < 1e-3)
- **CUDA Graph integration**: 5-10x speedup over PyTorch baseline
- **MIMO kernel**: Supports R=1,2,4,8 rank-1 updates
- **End-to-end decode (seq=1K, batch=4)**: CUDA Graph achieves **8.00x** speedup

---

## Improvement TODO (2026-04-14)

### P0 — 必须做，评审必定问 ✅ COMPLETED

1. **P0-1: 添加 `torch.compile` baseline** ✅ — 已添加 `use_compile` 模式到 `Mamba3Decoder`
2. **P0-2: 延迟分解 profiling** ✅ — 新建 `benchmarks/profile_step.py`，证明 launch overhead 占主导
3. **P0-3: CUDA Graph 消融实验** ✅ — 4-way 对比已完成，结果见下方

**关键发现（A100, d_model=256, bs=1）：**
| Backend | SISO Latency | MIMO Latency | Speedup vs PyTorch |
|---------|-------------|--------------|-------------------|
| PyTorch eager | 1.34 ms | 1.24 ms | 1.0x |
| torch.compile | 0.35 ms | 0.50 ms | 2.5-3.8x |
| Triton fused | 0.90 ms | 0.93 ms | 1.3-1.5x |
| **Triton + CUDA Graph** | **0.11 ms** | **0.19 ms** | **6.5-12.5x** |

**Profiling 结论（P0-2）：**
- PyTorch eager: CPU 22.7ms / GPU 4.5ms → **CPU/GPU = 5.0x** (80% overhead)
- Triton fused: CPU 14.5ms / GPU 3.3ms → **CPU/GPU = 4.5x** (still launch-bound)
- CUDA Graph: CPU 1.4ms / GPU 1.2ms → **CPU/GPU = 1.2x** (overhead eliminated)

### P1 — 显著提升分析深度

4. **P1-4: SISO vs MIMO 算术强度分析** — 计算 FLOPs/byte 比值 vs SISO，展示 MIMO 理论上更高的 arithmetic intensity，用 benchmark 验证
5. **P1-5: Multi-step 精度累积测试** — 1000 步 decode 后 Triton vs PyTorch ref 的误差
6. **P1-6: P99 延迟报告** ✅ — 已完成，见 benchmark 结果中的 percentiles 字段

**P99 延迟数据（SISO, bs=1）：**
| Backend | P50 | P95 | P99 |
|---------|-----|-----|-----|
| PyTorch | 1.12 ms | 1.30 ms | 1.32 ms |
| Triton | 0.93 ms | 0.95 ms | 0.96 ms |
| CUDA Graph | 0.20 ms | 0.21 ms | 0.21 ms |
| torch.compile | 0.33 ms | 0.50 ms | 0.52 ms |

**观察**：CUDA Graph 的 P99 非常稳定（方差小），torch.compile 的尾延迟较高（P99/P50 = 1.6x）

### P2 — 锦上添花

7. **P2-7: fp16 输入的正确性测试** — 即使 kernel 内部用 fp32 计算，输入/输出是 fp16 的场景
8. **P2-8: 不同 d_state 下的 benchmark** — 展示 SISO 对 d_state 更敏感（state=B×H×P×D），MIMO 的 state 只有 B×H×D
9. **P2-9: 大模型 (d_model=1024) benchmark** — 展示在大模型下加速比的变化

---

## 实验结果文件 (2026-04-14)

| 文件 | 配置 | 说明 |
|------|------|------|
| `siso_r4_20260414_114829.json` | d_model=256, d_state=64 | SISO 4-way ablation + P99 |
| `mimo_r4_20260414_114925.json` | d_model=256, d_state=64 | MIMO 4-way ablation + P99 |
| `mimo_r4_20260414_115407.json` | d_model=512, d_state=128 | MIMO 大模型配置 |
| `trace_pytorch_eager.json` | bs=4, d_model=256 | Chrome trace for PyTorch |
| `trace_triton_fused.json` | bs=4, d_model=256 | Chrome trace for Triton |

---

## Full-Step Fusion Phase (2026-04-14 ~ 2026-04-15)

### 核心成果

**Full-step fused kernel**: 将整个 decode step (split→rearrange→softplus→sigmoid→RMSNorm→RoPE→SSM→silu) 融合进单个 Triton kernel launch。

**8-way Ablation 结果 (A100, d_model=256, d_state=64)**：

#### SISO
| BS | PyTorch | Fused+Graph | Full Fused | Full+Graph | Best Speedup |
|----|---------|-------------|------------|------------|--------------|
| 1 | 1.15 ms | 0.19 ms (6.0x) | 0.20 ms (5.7x) | **0.08 ms** | **14.1x** |
| 8 | 1.37 ms | 0.23 ms (6.1x) | 0.21 ms (6.7x) | **0.10 ms** | **13.8x** |
| 32 | 1.23 ms | 0.24 ms (5.0x) | 0.21 ms (5.9x) | **0.12 ms** | **10.4x** |
| 128 | 1.79 ms | 0.36 ms (5.0x) | 0.28 ms (6.3x) | **0.29 ms** | **6.2x** |

#### MIMO (R=4)
| BS | PyTorch | Fused+Graph | Full Fused | Full+Graph | Best Speedup |
|----|---------|-------------|------------|------------|--------------|
| 1 | 1.28 ms | 0.11 ms (12.0x) | 0.19 ms (6.7x) | **0.07 ms** | **19.2x** |
| 8 | 1.31 ms | 0.23 ms (5.8x) | 0.19 ms (6.8x) | **0.14 ms** | **9.3x** |
| 32 | 1.42 ms | 0.26 ms (5.6x) | 0.21 ms (6.8x) | **0.18 ms** | **7.7x** |
| 128 | 1.54 ms | 0.34 ms (4.5x) | 0.51 ms (3.0x) | **0.52 ms** | **4.5x** |

### 关键发现

1. **Full fusion alone > partial fusion + Graph**: Full fused (无 Graph) 5.7-6.8x > Fused+Graph 5.0-6.1x (SISO)，证明融合范围比 CUDA Graph 更重要
2. **MIMO BS=1 达到 19.2x**: 全融合 + CUDA Graph 的极致组合
3. **BS=128 时 full fused 反而比 fused+Graph 慢**: MIMO full fused 0.51ms > fused+Graph 0.34ms，说明大 batch 下 kernel 变成 compute-bound，需要优化 block size
4. **torch.compile + CUDA Graph 全部失败**: RecompileLimitExceeded 错误

### 修复的 Bug

1. **Triton JIT `break`/`return`**: 不支持循环中的 break → 用 `tl.constexpr` + compile-time range unroll
2. **动态索引 `angles[i]`**: Triton 不支持变量索引 block tensor → inline RoPE + 逐个 load
3. **Batch stride bug**: `stride_zx_d` → `stride_zx_b`，导致 bs>1 完全错误
4. **A 计算算符优先级**: `-(softplus.clamp(...))` vs `(-softplus).clamp(...)` → 匹配 reference 的求值顺序

---

## P0+P1 Phase (2026-04-15)

### P0: Autoregressive Benchmark ✅

添加 8-way AR 解码 benchmark 到 `run_bench_extended.py`。测量 256-token 连续解码的 per-token 延迟。

**关键发现**：AR 加速比与单步加速比一致，证明 kernel launch overhead 是主导因素。

| Mode | BS=1 AR Speedup (Full+Graph) | 单步 Speedup |
|------|------------------------------|-------------|
| SISO d_model=256 | 14.8x | ~14x |
| MIMO d_model=256 | 12.9x | ~13x |

**Bug 修复**：`torch.compile` 的 CUDAGraphTrees 与 explicit CUDA Graph 冲突 → AR benchmark 跳过 compile 后端。

### P1: Multi-step Accuracy Drift ✅

新建 `test_accuracy_drift.py`，1000 步 decode 比较 Triton vs PyTorch 的累积误差。

**结果**：所有后端在 1000 步后累积误差 < 1.3e-4，数值稳定性良好。

---

## P2: d_model=512 8-way Ablation (2026-04-15)

**配置**: d_model=512, d_state=128, headdim=32, A100-PCIE-40GB

### SISO (d_model=512, d_state=128)

| BS | PyTorch | compile | Triton | Fused | Fused+Graph | **Full Fused** | **Full+Graph** | Best Speedup |
|----|---------|---------|--------|-------|-------------|----------------|----------------|--------------|
| 1 | 1.29 ms | 0.44 ms (2.9x) | 0.95 ms | 0.92 ms | 0.24 ms | 0.21 ms (6.1x) | **0.15 ms** | **8.4x** |
| 8 | 1.24 ms | 0.49 ms (2.5x) | 0.95 ms | 0.93 ms | 0.26 ms | 0.21 ms (5.9x) | **0.18 ms** | **7.0x** |
| 32 | 1.71 ms | 0.88 ms (1.9x) | 0.98 ms | 0.95 ms | 0.22 ms | 0.25 ms (6.9x) | **0.25 ms** | **7.7x** |

> BS=128 OOM (SISO state = B×H×P×D = 128×16×32×128 = 8M floats × 3 states)

### MIMO (d_model=512, d_state=128, R=4)

| BS | PyTorch | compile | Triton | Fused | Fused+Graph | **Full Fused** | **Full+Graph** | Best Speedup |
|----|---------|---------|--------|-------|-------------|----------------|----------------|--------------|
| 1 | 1.27 ms | 0.54 ms (2.3x) | 0.95 ms | 0.92 ms | **0.32 ms** (4.0x) | 0.35 ms (3.6x) | 0.35 ms (3.6x) | **4.0x** |
| 8 | 1.47 ms | 0.67 ms (2.2x) | 0.96 ms | 0.94 ms | **0.25 ms** (5.8x) | 0.35 ms (4.2x) | 0.36 ms (4.1x) | **5.8x** |
| 32 | 1.45 ms | 0.60 ms (2.4x) | 0.99 ms | 0.96 ms | **0.18 ms** (8.1x) | 0.53 ms (2.8x) | 0.53 ms (2.8x) | **8.1x** |
| 128 | 1.53 ms | 0.74 ms (2.1x) | 0.98 ms | 0.96 ms | **0.34 ms** (4.4x) | 1.84 ms (0.8x) ⚠️ | 1.84 ms (0.8x) ⚠️ | **4.4x** |

### Autoregressive Speedup (d_model=512, 256 tokens)

#### SISO

| BS | PyTorch | Triton | Fused | Fused+Graph | Full Fused | Full+Graph |
|----|---------|--------|-------|-------------|------------|------------|
| 1 | 1.10 ms/tok | 1.14x | 1.19x | 4.94x | 5.30x | **9.93x** |
| 8 | 1.12 ms/tok | 1.18x | 1.21x | 5.97x | 5.47x | **11.67x** |
| 32 | 1.33 ms/tok | 1.37x | 1.40x | 5.66x | 5.40x | **5.32x** |

#### MIMO

| BS | PyTorch | Triton | Fused | Fused+Graph | Full Fused | Full+Graph |
|----|---------|--------|-------|-------------|------------|------------|
| 1 | 1.17 ms/tok | 1.20x | 1.29x | **9.32x** | 6.06x | 7.85x |
| 8 | 1.33 ms/tok | 1.37x | 1.45x | **5.94x** | 5.15x | 6.82x |
| 32 | 1.37 ms/tok | 1.42x | 1.47x | **6.10x** | 2.39x | 2.60x |
| 128 | 1.39 ms/tok | 1.43x | 1.49x | **4.03x** | 0.76x ⚠️ | 0.75x ⚠️ |

### 关键发现

1. **Full fused kernel 在大模型 (d_model=512) 大 batch (BS≥32 MIMO) 下性能退化**：
   - MIMO BS=128: full fused 仅 0.83x (比 eager 还慢！)
   - 原因：d_model=512 → d_inner=1024 → nheads=32, d_state=128, R=4 → kernel 内需要加载 4×(B_raw+C_raw) 各 128 元素 + 4×(B_bias+C_bias) + 4×(B_exp+C_exp) + mimo_x/mimo_o → **寄存器溢出 (register spillover)**

2. **Fused+Graph 在大模型下反而更优**：
   - MIMO BS=1: fused+Graph 4.0x > full+Graph 3.6x
   - MIMO BS=32: fused+Graph 8.1x >> full+Graph 2.8x
   - Fused kernel 只融合 SSM+silu gate，寄存器占用小，不会溢出

3. **SISO full fused 仍然高效**：SISO 没有寄存器溢出问题，因为不需要同时持有 R=4 组 B/C/RoPE

4. **最佳配置取决于模型大小**：
   - d_model=256: Full+Graph 最优 (14-19x)
   - d_model=512 MIMO: Fused+Graph 最优 (4-8x)
   - d_model=512 SISO: Full+Graph 仍然最优 (7-12x)

---

## 下一步建议

1. **MIMO full fused kernel 寄存器优化**: d_model≥512 时 register spillover → 考虑 split-k 或减少 in-kernel 中间变量
2. **fp16/bf16 中间变量**: 减少寄存器压力，可能提升大 batch 性能
3. **自适应 backend 选择**: 根据模型大小和 batch size 自动选择 fused vs full-fused
