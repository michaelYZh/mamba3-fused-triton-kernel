# Mamba-3 Fused Triton Kernel

Accelerate Mamba-3 inference by replacing Python-based decoding with a fused Triton kernel implementing MIMO formulation and exponential-trapezoidal discretization.

## What's Implemented

| Component | File | Description |
|-----------|------|-------------|
| **SISO Triton Kernel** | `src/kernels/siso_decode.py` | Fused single-step SISO decode: state update + trapezoidal blend + output |
| **MIMO Triton Kernel** | `src/kernels/mimo_decode.py` | Fused single-step MIMO decode: R rank-1 updates + trapezoidal + up-projection |
| **Shared Utilities** | `src/kernels/utils.py` | RoPE, trapezoidal blending, PyTorch reference implementations |
| **Mamba3 Model** | `src/models/mamba3.py` | Full model with training forward + single-step decode |
| **Inference Interface** | `src/models/inference.py` | `Mamba3Decoder` with Triton/PyTorch backend switch |
| **Tests** | `src/tests/` | Correctness tests: Triton vs PyTorch reference (atol < 1e-2) |
| **Benchmarks** | `benchmarks/run_bench.py` | Latency comparison: PyTorch vs Triton across batch sizes |

## Project Structure

```
mamba3-fused-triton-kernel/
├── src/
│   ├── kernels/
│   │   ├── siso_decode.py    # SISO fused Triton decode kernel
│   │   ├── mimo_decode.py    # MIMO fused Triton decode kernel
│   │   └── utils.py          # RoPE, discretization, reference impls
│   ├── models/
│   │   ├── mamba3.py         # Mamba3 module (adapted from reference)
│   │   └── inference.py      # Inference interface with state management
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
# Create virtual environment
python -m venv venv
source venv/bin/activate

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
```

## Kernel Design

### SISO Fused Kernel
- One Triton program per `(batch, head)` pair
- Tiles over `P` (headdim) and `D` (d_state) dimensions
- Fuses: `B*x → trapezoidal blend → state update → C*h + D*x`
- State shape: `(B, H, P, D)` — full outer-product

### MIMO Fused Kernel
- One Triton program per `(batch, head)` pair
- R loop fully unrolled (R=4, tiny loop)
- Fuses: `x→R scalars → R rank-1 B*x → trapezoidal → state update → R rank-1 C*h → up-project`
- State shape: `(B, H, D)` — 2x smaller than SISO, higher arithmetic intensity

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
- [x] Day 1: Environment setup + algorithm deep-dive
- [x] Day 2: SISO fused kernel (core + trapezoidal + RoPE)
- [x] Day 3: MIMO fused kernel + correctness tests
- [ ] Day 4: Performance optimization and Triton tuning
- [ ] Day 5: Full benchmark suite + documentation
