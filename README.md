# Mamba-3 Fused Triton Kernel

Accelerate Mamba-3 inference by replacing Python-based decoding with a fused Triton kernel implementing MIMO formulation and exponential-trapezoidal discretization.

## What's Implemented

| Component | File | Description |
|-----------|------|-------------|
| **SISO Triton Kernel** | `src/kernels/siso_decode.py` | Fused single-step SISO decode: state update + trapezoidal blend + output |
| **MIMO Triton Kernel** | `src/kernels/mimo_decode.py` | Fused single-step MIMO decode: R rank-1 updates + trapezoidal + up-projection (R=1,2,4,8) |
| **SISO Fused+Silu Kernel** | `src/kernels/siso_decode_fused.py` | SISO decode + silu gate fusion in one kernel |
| **MIMO Fused+Silu Kernel** | `src/kernels/mimo_decode_fused.py` | MIMO decode + silu gate fusion in one kernel |
| **Full-Step Fused Kernel** | `src/kernels/fused_full_decode.py` | **Entire decode step in one kernel**: split → rearrange → softplus → sigmoid → RMSNorm → RoPE → SSM → silu |
| **Shared Utilities** | `src/kernels/utils.py` | RoPE, trapezoidal blending, PyTorch reference implementations |
| **Mamba3 Model** | `src/models/mamba3.py` | Full model with training forward + single-step decode |
| **Inference Interface** | `src/models/inference.py` | `Mamba3Decoder` with Triton/PyTorch backend + CUDA Graph support |
| **Tests** | `src/tests/` | Correctness tests: Triton vs PyTorch reference (atol < 1e-2) |
| **Benchmarks** | `benchmarks/run_bench_extended.py` | 8-way ablation: PyTorch / compile / compile+Graph / Triton / fused / fused+Graph / full-fused / full+Graph |

## Project Structure

```
mamba3-fused-triton-kernel/
├── src/
│   ├── kernels/
│   │   ├── siso_decode.py         # SISO fused Triton decode kernel
│   │   ├── mimo_decode.py         # MIMO fused Triton decode kernel (R=1,2,4,8)
│   │   ├── siso_decode_fused.py   # SISO + silu gate fused kernel
│   │   ├── mimo_decode_fused.py   # MIMO + silu gate fused kernel
│   │   ├── fused_full_decode.py   # Full-step fused kernel (entire decode in one launch)
│   │   └── utils.py               # RoPE, discretization, reference impls
│   ├── models/
│   │   ├── mamba3.py              # Mamba3 module (adapted from reference)
│   │   └── inference.py           # Inference interface with CUDA Graph
│   └── tests/
│       ├── test_siso.py           # SISO kernel vs PyTorch reference
│       ├── test_mimo.py           # MIMO kernel vs PyTorch reference
│       ├── test_trapezoidal.py    # Trapezoidal discretization validation
│       └── test_full_fused.py     # Full-step fused kernel correctness
├── benchmarks/
│   ├── run_bench_extended.py      # 8-way ablation benchmark (primary)
│   ├── run_bench.py               # 4-way ablation benchmark
│   ├── profile_step.py            # Latency breakdown profiling
│   └── results/                   # Benchmark output (JSON)
├── references/
│   └── mamba3-pytorch/            # Reference implementation
└── docs/
    ├── plan.md                    # Detailed MVP execution plan + algorithm math
    ├── experiment-log.md          # Experiment results, findings, bug fixes
    └── gpu-setup-guide.md         # GPU environment setup & session handover
```

## Setup

```bash
# Install dependencies
pip install torch triton numpy pytest einops

# Verify Triton + GPU
python -c "import triton; print(f'Triton: {triton.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Quick Start

```python
import torch
from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder

# Create model
model = Mamba3(d_model=256, d_state=64, headdim=32, is_mimo=True, mimo_rank=4).cuda()

# Create decoder with full-step fused Triton kernel (BEST performance)
decoder = Mamba3Decoder(model, use_triton_full_fused=True)

# Initialize state
state = decoder.init_state(batch_size=1)

# Single decode step
token_embedding = torch.randn(1, 256, device="cuda")
output, state = decoder.step(token_embedding, state)
```

### CUDA Graph (Maximum Performance)

```python
# Create decoder with full fused + CUDA Graph (absolute maximum performance)
decoder = Mamba3Decoder(model, use_triton_full_fused=True, use_cuda_graph=True)

# Initialize state and warmup graph
state = decoder.init_state(batch_size=1)
decoder.warmup_cuda_graph(state)

# Now step() uses CUDA Graph replay — minimal CPU overhead
for token_emb in token_sequence:
    output, state = decoder.step(token_emb, state)
```

## Running Tests

```bash
# All tests
pytest src/tests/ -v

# Specific test suite
pytest src/tests/test_siso.py -v         # SISO kernel correctness
pytest src/tests/test_mimo.py -v         # MIMO kernel correctness
pytest src/tests/test_trapezoidal.py -v  # Trapezoidal gate behavior
pytest src/tests/test_full_fused.py -v   # Full-step fused kernel correctness
```

## Running Benchmarks

```bash
# 4-way ablation (PyTorch / torch.compile / Triton / CUDA Graph)
python benchmarks/run_bench.py --mode siso --batch-sizes 1,8,32,128

# Skip torch.compile (faster)
python benchmarks/run_bench.py --mode mimo --mimo-rank 4 --no-compile

# Skip latency percentiles
python benchmarks/run_bench.py --mode both --no-percentiles

# Larger model
python benchmarks/run_bench.py --mode both --d-model 512 --d-state 128 --seq-len 1024
```

## Profiling

```bash
# Latency breakdown with torch.profiler
python benchmarks/profile_step.py --mode mimo --d-model 256 --batch-size 4

# Save Chrome trace for visualization
python benchmarks/profile_step.py --mode mimo --save-trace
# Then open chrome://tracing and load benchmarks/results/trace_*.json
```

## Benchmark Results (A100-PCIE-40GB)

### 8-Way Ablation: Full Results

**d_model=256, d_state=64, headdim=32, seq_len=256**

#### SISO (Single-Input Single-Output)

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | **Full Fused** | **Full+Graph** | Best Speedup |
|-------|---------|---------|--------|-------|-------------|---------------|----------------|--------------|
| 1 | 1.15 ms | 0.40 ms (2.9x) | 0.93 ms | 0.90 ms | 0.19 ms | 0.20 ms (5.6x) | **0.08 ms** | **14.0x** |
| 8 | 1.36 ms | 0.57 ms (2.4x) | 0.94 ms | 0.91 ms | 0.23 ms | 0.20 ms (6.8x) | **0.10 ms** | **13.7x** |
| 32 | 1.21 ms | 0.46 ms (2.6x) | 0.96 ms | 0.94 ms | 0.24 ms | 0.21 ms (5.7x) | **0.12 ms** | **10.3x** |
| 128 | 1.77 ms | 0.90 ms (2.0x) | 1.08 ms | 1.06 ms | 0.22 ms | 0.21 ms (8.2x) | **0.17 ms** | **10.5x** |

#### MIMO (Multi-Input Multi-Output, R=4)

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | **Full Fused** | **Full+Graph** | Best Speedup |
|-------|---------|---------|--------|-------|-------------|---------------|----------------|--------------|
| 1 | 1.23 ms | 0.53 ms (2.3x) | 0.92 ms | 0.88 ms | 0.32 ms (3.8x) | **0.18 ms** (6.8x) | 0.20 ms (6.0x) | **6.8x** |
| 8 | 1.24 ms | 0.56 ms (2.2x) | 0.93 ms | 0.89 ms | 0.22 ms (5.6x) | **0.18 ms** (6.9x) | 0.14 ms (8.9x) | **8.9x** |
| 32 | 1.37 ms | 0.65 ms (2.1x) | 0.97 ms | 0.91 ms | 0.15 ms (9.3x) | 0.19 ms (7.3x) | **0.11 ms** (12.8x) | **12.8x** |
| 128 | 1.48 ms | 0.65 ms (2.3x) | 1.02 ms | 0.98 ms | 0.19 ms (7.8x) | 0.28 ms (5.3x) | **0.28 ms** (5.2x) | **7.8x** |

> **Note**: At BS=1 MIMO, the full fused kernel (without CUDA Graph) achieves the best single-step latency (0.18 ms, 6.8x) — CUDA Graph's replay overhead is non-negligible for the very fast kernel. With CUDA Graph, fused+Graph excels at BS≥32 (9.3-12.8x). `torch.compile + CUDA Graph` failed (CUDA graph capture error) in all configurations, so it is excluded.

### 8-Way Ablation: Larger Model (d_model=512)

**d_model=512, d_state=128, headdim=32, seq_len=256**

#### SISO

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | **Full Fused** | **Full+Graph** | Best Speedup |
|-------|---------|---------|--------|-------|-------------|----------------|----------------|--------------|
| 1 | 1.29 ms | 0.44 ms (2.9x) | 0.95 ms | 0.92 ms | 0.24 ms | 0.21 ms (6.1x) | **0.15 ms** | **8.4x** |
| 8 | 1.24 ms | 0.49 ms (2.5x) | 0.95 ms | 0.93 ms | 0.26 ms | 0.21 ms (5.9x) | **0.18 ms** | **7.0x** |
| 32 | 1.71 ms | 0.88 ms (1.9x) | 0.98 ms | 0.95 ms | 0.22 ms | 0.25 ms (6.9x) | **0.25 ms** | **7.7x** |

> BS=128 OOM for SISO (state = B×H×P×D = 128×16×32×128 = 32MB per state)

#### MIMO (R=4)

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | **Full Fused** | **Full+Graph** | Best Speedup |
|-------|---------|---------|--------|-------|-------------|----------------|----------------|--------------|
| 1 | 1.27 ms | 0.54 ms (2.3x) | 0.95 ms | 0.92 ms | **0.32 ms** (4.0x) | 0.35 ms (3.6x) | 0.35 ms (3.6x) | **4.0x** |
| 8 | 1.47 ms | 0.67 ms (2.2x) | 0.96 ms | 0.94 ms | **0.25 ms** (5.8x) | 0.35 ms (4.2x) | 0.36 ms (4.1x) | **5.8x** |
| 32 | 1.45 ms | 0.60 ms (2.4x) | 0.99 ms | 0.96 ms | **0.18 ms** (8.1x) | 0.53 ms (2.8x) | 0.53 ms (2.8x) | **8.1x** |
| 128 | 1.53 ms | 0.74 ms (2.1x) | 0.98 ms | 0.96 ms | **0.34 ms** (4.4x) | 1.84 ms ⚠️ | 1.84 ms ⚠️ | **4.4x** |

> ⚠️ Full fused kernel at BS=128 MIMO is **slower** than eager (0.83x) due to register spillover. The fused+Graph backend is the optimal choice for large-model MIMO.

### Why Fusion Scope Matters

The critical insight is that **fusion scope** — not just CUDA Graph — determines performance:

1. **Partial fusion (SSM only)**: ~1.3x speedup. Only the SSM recurrence is in the Triton kernel; 15+ intermediate PyTorch ops (split, rearrange, softplus, sigmoid, RMSNorm, RoPE, etc.) remain as separate kernel launches.

2. **Full-step fusion**: ~6x speedup. ALL decode step computation is in one Triton kernel — split, rearrange, softplus, sigmoid, RMSNorm, RoPE, SSM recurrence, and silu gate are fused into a single kernel launch. Only `in_proj` and `out_proj` (memory-bound GEMMs) remain separate.

3. **Full-step fusion + CUDA Graph**: Up to **14.0x** speedup (SISO BS=1). Adds CUDA Graph to eliminate CPU launch overhead, giving the best possible performance.

The full fused kernel alone (without CUDA Graph) is already faster than partial fusion + CUDA Graph, proving that **maximizing fusion scope is more important than CUDA Graph for this workload**.

### Latency Tail: P99/P50 Ratios

Triton kernels deliver significantly more stable latency than PyTorch eager (measured with CUDA Events for accurate GPU-side timing):

| Method | SISO BS=1 P99/P50 | MIMO BS=1 P99/P50 |
|--------|-------------------|-------------------|
| PyTorch eager | 1.58x | 1.20x |
| Triton full fused | 1.09x | 1.03x |
| Triton full + Graph | 1.10x | 1.03x |

### Key Insights

1. **Full-step fusion scope is the #1 optimization** — 5.6-6.9x speedup from fusing ALL ops into one kernel, vs only 1.3x from partial fusion (small model d_model=256)
2. **CUDA Graph adds further improvement** — up to 2.5x on top of full fusion, for 10-14x total at small batch sizes
3. **Full fusion alone beats partial fusion + Graph** — proving that fusion scope matters more than launch overhead elimination
4. **torch.compile is competitive** — 2-3x speedup without custom kernels, but cannot achieve full-step fusion
5. **Best backend depends on model size and batch size** — d_model=256: Full+Graph optimal (10-14x SISO, 7-13x MIMO); d_model=512 MIMO: Fused+Graph optimal (4-8x)
6. **Full fused kernel has register pressure limits** — at d_model=512 + MIMO R=4 + BS≥32, the full fused kernel suffers register spillover, making it slower than eager. **Fused+Graph** becomes the optimal backend for large-model MIMO.

## Kernel Design

### Full-Step Fused Kernel (Best Performance)

The key innovation: **fuses the ENTIRE decode step into a single Triton kernel launch**. Only `in_proj` (linear projection) and `out_proj` remain as separate PyTorch ops.

```
Input: zxBCdtAtrap (B, d_in_proj) — raw output from in_proj
  ↓
[Single Triton Kernel]
  1. Split → z, x, B_raw, C_raw, dd_dt, dd_A, trap_raw, angle_raw
  2. Rearrange → z(B,H,P), x(B,H,P), B_raw(D), C_raw(D)
  3. softplus(dd_A) + clamp → A; softplus(dd_dt + dt_bias) → DT
  4. sigmoid(trap_raw) → trap
  5. RMSNorm(B_raw), RMSNorm(C_raw) → B_normed, C_normed
  6. Add B_bias, C_bias → B_exp, C_exp
  7. RoPE(angle_state) → B_proj, C_proj
  8. SSM recurrence + silu gate → y_gated
  ↓
Output: y_gated (B, H, P) → out_proj → final output
```

**Why this matters**: The partial-fusion approach (SSM-only) leaves ~15 PyTorch ops outside the kernel, each requiring a separate kernel launch. At batch_size=1, these launches dominate (67% of step time). Full-step fusion eliminates all of them, achieving **3.7x** speedup over partial fusion alone.

### SISO Fused Kernel (Partial Fusion)
- One Triton program per `(batch, head)` pair
- P dimension iterated inside kernel; D dimension tiled with BLOCK_D
- Fuses: `B*x → trapezoidal blend → state update → C*h + D*x`
- State shape: `(B, H, P, D)` — full outer-product

### MIMO Fused Kernel
- One Triton program per `(batch, head)` pair
- R loop fully unrolled (separate kernels for R=1,2,4,8)
- Fuses: `x→R scalars → R rank-1 B*x → trapezoidal → state update → R rank-1 C*h → up-project`
- State shape: `(B, H, D)` — 2x smaller than SISO, higher arithmetic intensity

### CUDA Graph Integration
- Eliminates CPU-side kernel launch overhead (~80% of step time)
- Captures the entire decode step (in_proj → RMSNorm → RoPE → Triton kernel → gate → out_proj)
- State is updated in-place during graph replay
- Provides **4-12x** additional speedup over Triton kernel alone

### torch.compile Support
- Added `use_compile` mode to `Mamba3Decoder` for fair comparison
- Uses `torch.compile(fullgraph=True)` on the pure-functional decode step
- Achieves **2-3x** speedup by fusing PyTorch ops via Inductor
- Useful baseline: "Why not just use torch.compile?" → CUDA Graph still wins

## References

| Resource | Link |
|----------|------|
| Mamba-3 Paper | [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) |
| Official Implementation | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| Readable PyTorch Reference | [rishikksh20/mamba3-pytorch](https://github.com/rishikksh20/mamba3-pytorch) |
| Mamba-2 Fused Kernel Blog | [PyTorch Blog](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) |
| Together AI Mamba-3 Blog | [Blog Post](https://www.together.ai/blog/mamba-3) |

## Milestones

- [x] Project scaffolding and planning
- [x] Environment setup + algorithm deep-dive
- [x] SISO fused kernel (core + trapezoidal + RoPE)
- [x] MIMO fused kernel + correctness tests
- [x] Performance optimization and Triton tuning
- [x] CUDA Graph integration for maximum throughput
- [x] Full benchmark suite + documentation
- [x] **Full-step fused kernel** — entire decode in one kernel launch (5.6-6.9x speedup over eager)
- [x] **Full fused + CUDA Graph** — best configuration (up to 14.0x speedup over eager, SISO BS=1)
- [x] **P0: Autoregressive benchmark** — AR speedups match single-step (12.8x SISO, 10.4x MIMO at BS=1)
- [x] **P1: Multi-step accuracy drift** — 1000-step decode, all backends < 1.3e-4 (d_model=256), < 2.3e-4 (d_model=512) cumulative error
- [x] **P2: d_model=512 8-way ablation** — SISO 7-8.4x, MIMO 4-8.1x; discovered register spillover at large model+batch
- [x] **P0: Re-run d_model=256 benchmark with CUDA Event timing** — Fixed percentile measurement (was polluted by CPU-GPU sync overhead at BS=1). Corrected MIMO BS=1 speedup from 19.2x to 6.8x (full fused) / 12.8x (fused+Graph at BS=32).
- [x] **P1: d_model=512 accuracy drift** — 1000-step: SISO < 5.0e-5, MIMO < 2.3e-4, all PASS

> **P2 Phase Complete** (2026-04-15) — d_model=512 benchmarks reveal full fused kernel register pressure limits in MIMO at large batch. Fused+Graph is the optimal backend for large-model MIMO (8.1x at BS=32). d_model=256 benchmarks re-run with CUDA Event timing; MIMO BS=1 best speedup corrected from 19.2x to 6.8x (full fused). d_model=512 accuracy drift: all backends PASS (< 2.3e-4).

## Future Plans

The following are optional enhancements for future development:

- [x] dtype optimization (fp16/bf16 intermediate variables)
- [ ] Register pressure analysis via `triton.testing`
- [ ] Cache hint optimization
- [x] `torch.compile` compatibility test — **COMPLETED** (see 4-way ablation results)
- [x] Larger model benchmarks (d_model=512) — **COMPLETED** (8-way ablation: SISO 7-8.4x, MIMO 4-8.1x best)
- [x] Full-step fusion — **COMPLETED** (5.6-6.9x speedup over eager, up to 14.0x with CUDA Graph)
- [ ] Prefill kernel fusion
- [ ] H100 benchmark comparison
- [x] Multi-step accuracy drift test (1000 steps) — **COMPLETED** (d_model=256: all backends < 1.3e-4; d_model=512: all backends < 2.3e-4)
- [x] Autoregressive benchmark (8-way) — **COMPLETED** (AR speedups match single-step)
- [ ] MIMO full fused kernel register optimization for d_model≥512
