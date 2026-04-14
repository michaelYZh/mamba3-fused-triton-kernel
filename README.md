# Mamba-3 Fused Triton Kernel

Accelerate Mamba-3 inference by replacing Python-based decoding with a fused Triton kernel implementing MIMO formulation and exponential-trapezoidal discretization.

## What's Implemented

| Component | File | Description |
|-----------|------|-------------|
| **SISO Triton Kernel** | `src/kernels/siso_decode.py` | Fused single-step SISO decode: state update + trapezoidal blend + output |
| **MIMO Triton Kernel** | `src/kernels/mimo_decode.py` | Fused single-step MIMO decode: R rank-1 updates + trapezoidal + up-projection (R=1,2,4,8) |
| **Shared Utilities** | `src/kernels/utils.py` | RoPE, trapezoidal blending, PyTorch reference implementations |
| **Mamba3 Model** | `src/models/mamba3.py` | Full model with training forward + single-step decode |
| **Inference Interface** | `src/models/inference.py` | `Mamba3Decoder` with Triton/PyTorch backend + CUDA Graph support |
| **Tests** | `src/tests/` | Correctness tests: Triton vs PyTorch reference (atol < 1e-2) |
| **Benchmarks** | `benchmarks/run_bench.py` | Latency comparison: PyTorch vs Triton vs CUDA Graph |

## Project Structure

```
mamba3-fused-triton-kernel/
├── src/
│   ├── kernels/
│   │   ├── siso_decode.py    # SISO fused Triton decode kernel
│   │   ├── mimo_decode.py    # MIMO fused Triton decode kernel (R=1,2,4,8)
│   │   └── utils.py          # RoPE, discretization, reference impls
│   ├── models/
│   │   ├── mamba3.py         # Mamba3 module (adapted from reference)
│   │   └── inference.py      # Inference interface with CUDA Graph
│   └── tests/
│       ├── test_siso.py      # SISO kernel vs PyTorch reference
│       ├── test_mimo.py      # MIMO kernel vs PyTorch reference
│       └── test_trapezoidal.py # Trapezoidal discretization validation
├── benchmarks/
│   ├── run_bench.py          # Latency benchmark script
│   └── results/              # Benchmark output (JSON)
├── references/
│   └── mamba3-pytorch/       # Reference implementation
└── docs/
    └── plan.md               # Detailed MVP execution plan
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

# Create decoder with Triton backend
decoder = Mamba3Decoder(model, use_triton=True)

# Initialize state
state = decoder.init_state(batch_size=1)

# Single decode step
token_embedding = torch.randn(1, 256, device="cuda")
output, state = decoder.step(token_embedding, state)
```

### CUDA Graph (Maximum Performance)

```python
# Create decoder with CUDA Graph support
decoder = Mamba3Decoder(model, use_triton=True, use_cuda_graph=True)

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
pytest src/tests/test_siso.py -v     # SISO kernel correctness
pytest src/tests/test_mimo.py -v     # MIMO kernel correctness
pytest src/tests/test_trapezoidal.py -v  # Trapezoidal gate behavior
```

## Running Benchmarks

```bash
# SISO benchmark
python benchmarks/run_bench.py --mode siso --batch-sizes 1,8,32,128

# MIMO benchmark
python benchmarks/run_bench.py --mode mimo --mimo-rank 4 --batch-sizes 1,8,32,128

# Both modes
python benchmarks/run_bench.py --mode both --seq-len 256

# Larger model
python benchmarks/run_bench.py --mode both --d-model 512 --d-state 128 --seq-len 1024
```

## Benchmark Results (A100-PCIE-40GB)

### Small Model (d_model=256, d_state=64, headdim=32, seq_len=256)

| Mode | Batch | PyTorch | Triton | Speedup | CUDA Graph | Speedup |
|------|-------|---------|--------|---------|-----------|---------|
| SISO | 1 | 1.11 ms | 0.89 ms | 1.25x | **0.19 ms** | **5.86x** |
| SISO | 8 | 1.33 ms | 0.91 ms | 1.46x | **0.18 ms** | **7.52x** |
| SISO | 128 | 1.70 ms | 0.98 ms | 1.74x | **0.22 ms** | **7.76x** |
| MIMO | 1 | 1.13 ms | 0.95 ms | 1.18x | **0.11 ms** | **10.52x** |
| MIMO | 8 | 1.25 ms | 0.91 ms | 1.37x | **0.14 ms** | **9.15x** |
| MIMO | 32 | 1.44 ms | 0.93 ms | 1.54x | **0.16 ms** | **8.77x** |

### Larger Model (d_model=512, d_state=128, headdim=64, seq_len=1024)

| Mode | Batch | PyTorch | Triton | Speedup | CUDA Graph | Speedup |
|------|-------|---------|--------|---------|-----------|---------|
| MIMO | 1 | 1.21 ms | 0.93 ms | 1.29x | **0.13 ms** | **9.25x** |
| MIMO | 4 | 1.28 ms | 1.02 ms | 1.25x | **0.13 ms** | **9.89x** |
| MIMO | 16 | 1.39 ms | 0.97 ms | 1.44x | **0.15 ms** | **9.10x** |
| MIMO | 64 | 1.47 ms | 1.06 ms | 1.39x | **0.20 ms** | **7.38x** |

### Autoregressive Decoding (1024 tokens)

| Mode | Batch | PyTorch ms/tok | Triton ms/tok | Speedup | Graph ms/tok | Speedup |
|------|-------|----------------|---------------|---------|-------------|---------|
| MIMO | 1 | 1.17 | 0.95 | 1.23x | **0.16** | **7.53x** |
| MIMO | 4 | 1.27 | 0.97 | 1.31x | **0.16** | **8.00x** |
| MIMO | 16 | 1.33 | 0.98 | 1.36x | **0.17** | **7.91x** |
| MIMO | 64 | 1.40 | 0.99 | 1.42x | **0.20** | **7.05x** |

## Kernel Design

### SISO Fused Kernel
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
- Provides **5-10x** additional speedup over Triton kernel alone

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
