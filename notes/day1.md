# Experiment Log

Daily experiment notes are recorded here as development progresses.

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

## 下一步建议

1. **MIMO BS=128 kernel 优化**: 调整 BLOCK_D 或 split K 以提高 compute-bound 场景性能
2. **P1-5 精度累积**: 跑 1000-step decode，对比 Triton vs PyTorch 的 hidden state drift
3. **fp16/bf16 中间变量**: 减少寄存器压力，可能提升大 batch 性能
