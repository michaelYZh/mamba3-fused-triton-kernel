# Mamba-3 Fused Triton Kernel

Accelerate Mamba-3 inference by replacing Python-based decoding with a fused Triton kernel implementing MIMO formulation and exponential-trapezoidal discretization.

## What's Implemented

| Component | File | Description |
|-----------|------|-------------|
| **SISO Triton Kernel** | `src/kernels/siso_decode.py` | Fused single-step SISO decode: state update + trapezoidal blend + output |
| **MIMO Triton Kernel** | `src/kernels/mimo_decode.py` | Fused single-step MIMO decode: R rank-1 updates + trapezoidal + up-projection (R=1,2,4,8) |
| **SISO Fused+Silu Kernel** | `src/kernels/siso_decode_fused.py` | SISO decode + silu gate fusion in one kernel |
| **MIMO Fused+Silu Kernel** | `src/kernels/mimo_decode_fused.py` | MIMO decode + silu gate fusion in one kernel |
| **Full-Step Fused Kernel** | `src/kernels/fused_full_decode.py` | **All decode ops between `in_proj` and `out_proj` in one kernel**: split → rearrange → softplus → sigmoid → RMSNorm → RoPE → SSM → silu |
| **Shared Utilities** | `src/kernels/utils.py` | RoPE, trapezoidal blending, PyTorch reference implementations |
| **Mamba3 Model** | `src/models/mamba3.py` | `Mamba3` layer with training forward + single-step decode, plus a multi-layer `MambaLMHeadModel` wrapper (forward only). Adapted from [rishikksh20/mamba3-pytorch](https://github.com/rishikksh20/mamba3-pytorch). `forward` and `step` compute the decay rate `A = (-softplus(dd_A)).clamp(max=-A_floor)`, as upstream `forward()` does. Before the fix, `A` was the constant `+A_floor` (see [Decay sign fix](#decay-sign-fix)) |
| **Inference Interface** | `src/models/inference.py` | `Mamba3Decoder` with Triton/PyTorch backend + CUDA Graph support |
| **Tests** | `src/tests/` | Correctness tests against the PyTorch reference. The SISO and MIMO kernel tests assert max abs error < 1e-2 against the reference. `test_full_fused.py` prints a verdict but does not assert (see [Running Tests](#running-tests)). `test_decay_sign.py` asserts that `A ≤ -A_floor` and, on CUDA, that the full-step kernel's state decays (see [Decay sign fix](#decay-sign-fix)) |
| **Benchmarks** | `benchmarks/run_bench_extended.py` | 7-way ablation: PyTorch / compile / Triton / fused / fused+Graph / full-fused / full+Graph. The script also lists compile+Graph but skips it (see [How These Numbers Are Measured](#how-these-numbers-are-measured)) |

## Project Structure

```
mamba3-fused-triton-kernel/
├── src/
│   ├── kernels/
│   │   ├── siso_decode.py         # SISO fused Triton decode kernel
│   │   ├── mimo_decode.py         # MIMO fused Triton decode kernel (R=1,2,4,8)
│   │   ├── siso_decode_fused.py   # SISO + silu gate fused kernel
│   │   ├── mimo_decode_fused.py   # MIMO + silu gate fused kernel
│   │   ├── fused_full_decode.py   # Full-step fused kernel (all non-GEMM decode ops in one launch)
│   │   └── utils.py               # RoPE, discretization, reference impls
│   ├── models/
│   │   ├── mamba3.py              # Mamba3 module (adapted from rishikksh20/mamba3-pytorch)
│   │   └── inference.py           # Inference interface with CUDA Graph
│   └── tests/
│       ├── test_siso.py           # SISO kernel vs PyTorch reference
│       ├── test_mimo.py           # MIMO kernel vs PyTorch reference
│       ├── test_trapezoidal.py    # Trapezoidal discretization validation
│       ├── test_decay_sign.py     # Sign of the decay rate A, state decay of the full-step kernel
│       └── test_full_fused.py     # Full-step fused kernel correctness
├── benchmarks/
│   ├── run_bench_extended.py      # 7-way ablation benchmark (primary)
│   ├── run_bench.py               # 4-way ablation benchmark
│   ├── test_accuracy_drift.py     # Multi-step accuracy drift test
│   ├── profile_step.py            # Latency breakdown profiling
│   ├── run_bench_fusion_comparison.py  # Empty file (0 lines), not used
│   └── results/                   # Benchmark output (JSON)
├── references/
│   ├── README.md                  # Lists papers (local paths), repos and blog posts
│   └── mamba3-pytorch/            # Reference implementation (git submodule: rishikksh20/mamba3-pytorch)
├── scripts/
│   └── setup_gpu_env.sh           # Sets up the conda env on a rented A100/H100 Ubuntu instance
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

# Create decoder with the full-step fused Triton kernel
decoder = Mamba3Decoder(model, use_triton_full_fused=True)

# Initialize state
state = decoder.init_state(batch_size=1)

# Single decode step
token_embedding = torch.randn(1, 256, device="cuda")
output, state = decoder.step(token_embedding, state)
```

### CUDA Graph

```python
# Create decoder with full fused + CUDA Graph (fastest backend in most d_model=256 configs by
# single-step mean, by P50 and in the 256-token loop; see Benchmark Results)
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
pytest src/tests/test_decay_sign.py -v -s  # Sign of A (runs on CPU), full-step kernel state decay (CUDA rows, -s prints the norm ratio)
pytest src/tests/test_full_fused.py -v   # Full-step fused kernel (SISO only under pytest, prints a verdict, does not assert)

# Full-step fused kernel with a pass/fail exit code
python -m src.tests.test_full_fused
```

`test_full_fused.py` compares the full-step kernel with the PyTorch decoder. It reports PASS when the max abs error is < 1e-3 at the first step and < 1e-2 after 10 more steps. The test function returns this verdict and does not assert, so pytest does not fail it on a numerical mismatch. pytest runs the function with its default arguments only, which is the SISO case. Run the module directly to test SISO and MIMO and to get a non-zero exit code on failure.

## Running Benchmarks

```bash
# Run from the repo root. The scripts import `src.*`, so the repo root must be on PYTHONPATH.

# 4-way ablation (PyTorch / torch.compile / Triton / CUDA Graph)
PYTHONPATH=. python benchmarks/run_bench.py --mode siso --batch-sizes 1,8,32,128

# Skip torch.compile (faster)
PYTHONPATH=. python benchmarks/run_bench.py --mode mimo --mimo-rank 4 --no-compile

# Skip latency percentiles
PYTHONPATH=. python benchmarks/run_bench.py --mode both --no-percentiles

# Larger model
PYTHONPATH=. python benchmarks/run_bench.py --mode both --d-model 512 --d-state 128 --seq-len 1024

# 7-way ablation (the script behind the tables below)
PYTHONPATH=. python benchmarks/run_bench_extended.py --mode both
PYTHONPATH=. python benchmarks/run_bench_extended.py --mode siso --d-model 512 --d-state 128 --batch-sizes 1,8,32
PYTHONPATH=. python benchmarks/run_bench_extended.py --mode mimo --d-model 512 --d-state 128

# Multi-step accuracy drift
PYTHONPATH=. python benchmarks/test_accuracy_drift.py --mode both --num-steps 1000
PYTHONPATH=. python benchmarks/test_accuracy_drift.py --mode both --num-steps 1000 --d-model 512 --d-state 128
```

## Profiling

```bash
# Latency breakdown with torch.profiler
PYTHONPATH=. python benchmarks/profile_step.py --mode mimo --d-model 256 --batch-size 4

# Save Chrome trace for visualization
PYTHONPATH=. python benchmarks/profile_step.py --mode mimo --save-trace
# Then open chrome://tracing and load benchmarks/results/trace_*.json
```

## Benchmark Results (A100-PCIE-40GB)

### How These Numbers Are Measured

Every measured value below comes from a file in `benchmarks/results/`. Iteration counts and tolerances come from the benchmark code. Each table names its source file and the run's `timestamp` field. Each table is one run of one configuration, so the tables have no confidence intervals. All timings are for one `Mamba3` layer (`in_proj` → decode step → `out_proj`) in fp32, not for a multi-layer model. All result files were produced before the [decay sign fix](#decay-sign-fix). The timings are not expected to change, but the accuracy-drift results must be re-run on a GPU.

- **Backends.** *PyTorch* (eager) is the PyTorch decode step. *compile* is `torch.compile` of that step. *Triton* ("Triton basic" in the latency tables) runs the SSM recurrence in a Triton kernel and keeps the other decode ops in PyTorch. *Fused* runs the SSM recurrence and the silu gate in one Triton kernel, called the SSM+gate kernel below. *Full Fused* runs every decode op between `in_proj` and `out_proj` in one Triton kernel. *+Graph* captures the same step once in a CUDA Graph and replays it.
- **7-way ablation tables** (`benchmark_single_step` in `benchmarks/run_bench_extended.py`): wall-clock `time.perf_counter` around 50 back-to-back calls, with `torch.cuda.synchronize()` before and after the loop. The value is the mean (loop time / 50). The loop has no synchronization between calls, so the CPU work of one call can overlap the GPU work of the previous call. The loop runs after 5 warmup calls (20 for torch.compile). CUDA Graph backends run 10 un-captured warmup steps and then capture the graph, with no replay before the timed loop. Speedup = PyTorch eager mean / backend mean.
- **Latency tail tables** (`benchmark_latency_percentiles`): a separate pass. Each sample is one call between two CUDA Events recorded on the current stream, followed by `torch.cuda.synchronize()`, which is outside the timed window. There are 200 samples per cell, taken after 10 warmup calls plus 1 untimed call. For CUDA Graph backends, the 10 warmup calls are un-captured steps before the capture, and the untimed call is the first replay. P50 and P99 come from `numpy.percentile` with linear interpolation, so P99 lies between the 3rd and 2nd largest of the samples.
- **Autoregressive tables** (`benchmark_autoregressive`, `benchmark_autoregressive_with_graph`): `time.perf_counter` around 256 sequential `step()` calls, with synchronization before and after; the value is the mean of 3 repetitions. Every step gets the same random input embedding and the output is not fed back, so the loop measures dependent state updates, not sampling (fused+Graph does not advance the RoPE angle, see [CUDA Graph Integration](#cuda-graph-integration)). For CUDA Graph backends, each repetition captures the graph again (3 warmup steps) before the timed loop. torch.compile is not in this benchmark: the script skips it.
- **torch.compile** runs `torch.compile(fullgraph=True)` in the default mode, without CUDA Graphs. **torch.compile + CUDA Graph was not measured.** `run_bench_extended.py` skips it and stores NaN: `compile_cuda_graph_ms` is NaN in all 18 extended result files. The script's comment says that torch.compile's CUDAGraphTrees conflicts with explicit CUDA Graph capture. An earlier version of the script (commit `5c016ba`) tried it inside `try/except RuntimeError` and also stored NaN, without saving the error. So the tables have 7 measured backends, and the CUDA Graph backends have no like-for-like torch.compile baseline.
- **The two timing methods disagree.** For the same run and cell, the P50/mean ratio runs from 0.15 to 1.83. Here P50 comes from the latency tail pass and the mean from the 7-way ablation table, not the `mean_ms` that the percentile pass also stores. The larger of the two is more than 1.25x the smaller in 28 of 105 cells. Do not compare a mean from one table with a P50 from another.
- **Run-to-run variance.** Reruns of one config can differ by more than the gap between backends. d_model=256 MIMO BS=1 full+Graph measured 0.067 ms (19.2x, `035152`), 0.078 ms (14.9x, `051042`), 0.120 ms (10.4x, `053357`) and 0.203 ms (6.0x, `053417`, the run in the tables). Some cells are stable: d_model=256 SISO BS=1 full+Graph measured 14.0–14.2x in all 5 runs. Others are not: d_model=256 SISO BS=128 full+Graph measured 6.0–10.5x across the same 5 runs, and d_model=256 MIMO BS=32 full+Graph 6.6–12.8x across 4 runs. Over the 56 d_model=256 cells with 4 or 5 runs, the slowest mean is within 1.2x of the fastest in 42 cells. The cause of the spread was not diagnosed.

### 7-Way Ablation: Full Results

**d_model=256, d_state=64, headdim=32, seq_len=256**

Speedup in parentheses = PyTorch eager mean / backend mean. Bold cell = lowest mean latency in the row.

#### SISO (Single-Input Single-Output)

Source: `benchmarks/results/siso_r4_extended_20260415_053314.json` (timestamp `2026-04-15T05:32:42`), field `single_step`.

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | Full Fused | Full+Graph | Best Speedup |
|-------|---------|---------|--------|-------|-------------|---------------|----------------|--------------|
| 1 | 1.15 ms | 0.40 ms (2.9x) | 0.93 ms | 0.90 ms | 0.19 ms | 0.20 ms (5.6x) | **0.08 ms** | **14.0x** |
| 8 | 1.36 ms | 0.57 ms (2.4x) | 0.94 ms | 0.91 ms | 0.23 ms | 0.20 ms (6.8x) | **0.10 ms** | **13.7x** |
| 32 | 1.21 ms | 0.46 ms (2.6x) | 0.96 ms | 0.94 ms | 0.24 ms | 0.21 ms (5.7x) | **0.12 ms** | **10.3x** |
| 128 | 1.77 ms | 0.90 ms (2.0x) | 1.08 ms | 1.06 ms | 0.22 ms | 0.21 ms (8.2x) | **0.17 ms** | **10.5x** |

#### MIMO (Multi-Input Multi-Output, R=4)

Source: `benchmarks/results/mimo_r4_extended_20260415_053417.json` (timestamp `2026-04-15T05:33:44`), field `single_step`.

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | Full Fused | Full+Graph | Best Speedup |
|-------|---------|---------|--------|-------|-------------|---------------|----------------|--------------|
| 1 | 1.23 ms | 0.53 ms (2.3x) | 0.92 ms | 0.88 ms | 0.32 ms (3.8x) | **0.18 ms** (6.8x) | 0.20 ms (6.0x) | **6.8x** |
| 8 | 1.24 ms | 0.56 ms (2.2x) | 0.93 ms | 0.89 ms | 0.22 ms (5.6x) | 0.18 ms (6.9x) | **0.14 ms** (8.9x) | **8.9x** |
| 32 | 1.37 ms | 0.65 ms (2.1x) | 0.97 ms | 0.91 ms | 0.15 ms (9.3x) | 0.19 ms (7.3x) | **0.11 ms** (12.8x) | **12.8x** |
| 128 | 1.48 ms | 0.65 ms (2.3x) | 1.02 ms | 0.98 ms | **0.19 ms** (7.8x) | 0.28 ms (5.3x) | 0.28 ms (5.2x) | **7.8x** |

> **Note**: In this run at BS=1, full fused without CUDA Graph had the lowest mean (0.18 ms) and full+Graph 0.20 ms. This ordering did not reproduce in other runs of the same config: full+Graph 0.12 ms against full fused 0.19 ms in `053357`, and full+Graph 0.08 ms against full fused 0.18 ms in `051042`. So the BS=1 Best Speedup of 6.8x holds for this run only. Full+Graph is fastest at BS=8 and 32 (8.9x, 12.8x) and fused+Graph at BS=128 (7.8x), all by mean. `torch.compile + CUDA Graph` was not measured (see above).

### 7-Way Ablation: Larger Model (d_model=512)

**d_model=512, d_state=128, headdim=32, seq_len=256**

#### SISO

Source: `benchmarks/results/siso_r4_extended_20260415_061338.json` (timestamp `2026-04-15T06:13:11`), field `single_step`.

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | Full Fused | Full+Graph | Best Speedup |
|-------|---------|---------|--------|-------|-------------|----------------|----------------|--------------|
| 1 | 1.28 ms | 0.43 ms (3.0x) | 0.96 ms | 0.90 ms | 0.33 ms (3.9x) | 0.22 ms (5.9x) | **0.21 ms** | **6.0x** |
| 8 | 1.21 ms | 0.68 ms (1.8x) | 0.95 ms | 0.92 ms | **0.14 ms** (8.6x) | 0.21 ms (5.8x) | 0.24 ms (4.9x) | **8.6x** |
| 32 | 1.73 ms | 0.84 ms (2.1x) | 0.99 ms | 0.97 ms | **0.22 ms** (7.8x) | 0.25 ms (6.9x) | 0.25 ms (6.9x) | **7.8x** |

> BS=128 was not run for d_model=512 SISO: the d_model=512 SISO result files cover BS=1, 8 and 32 only. The experiment log reports an out-of-memory error at BS=128, but no result file records it.

#### MIMO (R=4)

Source: `benchmarks/results/mimo_r4_extended_20260415_061428.json` (timestamp `2026-04-15T06:13:50`), field `single_step`.

| Batch | PyTorch | compile | Triton | Fused | Fused+Graph | Full Fused | Full+Graph | Best Speedup |
|-------|---------|---------|--------|-------|-------------|----------------|----------------|--------------|
| 1 | 1.26 ms | 0.54 ms (2.3x) | 0.95 ms | 0.92 ms | **0.23 ms** (5.4x) | 0.27 ms (4.7x) | 0.27 ms (4.6x) | **5.4x** |
| 8 | 1.49 ms | 0.69 ms (2.2x) | 0.95 ms | 0.93 ms | **0.25 ms** (5.9x) | 0.35 ms (4.2x) | 0.36 ms (4.2x) | **5.9x** |
| 32 | 1.50 ms | 0.62 ms (2.4x) | 0.99 ms | 0.97 ms | **0.18 ms** (8.3x) | 0.53 ms (2.8x) | 0.53 ms (2.8x) | **8.3x** |
| 128 | 1.58 ms | 0.78 ms (2.0x) | 0.98 ms | 0.95 ms | **0.34 ms** (4.6x) | 1.84 ms (0.86x) ⚠️ | 1.84 ms (0.86x) ⚠️ | **4.6x** |

> ⚠️ At BS=128 the full fused kernel (1.84 ms) is slower than eager (0.86x). At BS=32 it is still 2.8x faster than eager but slower than fused+Graph (0.53 vs 0.18 ms). The suspected cause of this slowdown is register pressure in the full-step MIMO kernel. It has not been measured (see Future Plans). Fused+Graph stays at 0.34 ms (4.6x) at BS=128. By single-step mean, fused+Graph is the fastest backend for d_model=512 MIMO at every batch size.

### Why Fusion Scope Matters

Both fusion scope and CUDA Graph matter, and which one matters more depends on the config. Single-step means, d_model=256 unless noted:

1. **Partial fusion (the SSM+gate kernel, Fused column)**: 1.3–1.7x over eager. Only the SSM recurrence and the silu gate are in the Triton kernel. The other decode ops (softplus, sigmoid, RMSNorm, RoPE, bias adds, etc.) remain separate PyTorch kernel launches.

2. **Full-step fusion**: 5.3–8.2x over eager, and 3.5–5.0x faster than the SSM+gate kernel. Every decode op between the two GEMMs is in one Triton kernel — split, rearrange, softplus, sigmoid, RMSNorm, RoPE, SSM recurrence, and silu gate are fused into a single kernel launch. Only `in_proj` and `out_proj` (GEMMs) remain separate.

3. **Full-step fusion + CUDA Graph**: Up to **14.0x** speedup (SISO BS=1). At d_model=256, full+Graph gives 10.3–14.0x for SISO and 5.2–12.8x for MIMO. CUDA Graph replays the captured step instead of launching each kernel from the CPU.

Full-step fusion alone does not replace CUDA Graph: across all 15 configs (both model sizes), the full fused kernel without CUDA Graph beats fused+Graph in 6 of 15 configs by mean and in 3 of the 13 configs without a <sup>†</sup> cell by P50 (<sup>†</sup> is defined under [Latency Tail](#latency-tail-p50-and-p99-cuda-event-timing)). At d_model=512 MIMO, the SSM+gate kernel + CUDA Graph is the fastest backend by mean and by P50 at every batch size. In the 256-token loop it is the fastest only at BS=32 and 128, and full+Graph is the fastest at BS=1 and 8.

### Latency Tail: P50 and P99 (CUDA Event Timing)

These tables come from the separate CUDA Event pass (see [How These Numbers Are Measured](#how-these-numbers-are-measured)). The `torch.cuda.synchronize()` after each sample is outside the timed window. Commit `4ead87d` switched this pass to CUDA Events from `perf_counter` with a `synchronize()` around each sample. With the old timer, one run (`mimo_r4_extended_20260415_052234.json`, d_model=512 MIMO) put 4 of 7 BS=1 P50 values at 2.65–2.75 ms. Run `052204` has the same config and timer, and its BS=1 P50 values are 0.24–1.26 ms. None of the other 9 extended result files from before commit `4ead87d` has a BS=1 P50 above 1.35 ms. So the data does not show that the old timer itself inflated BS=1 percentiles.

<sup>†</sup> marks a cell that is bimodal or did not reproduce: P99 is more than 3x P50, or P50 differs by more than 2x from the median P50 of the other CUDA Event runs of the same config (`053357` for d_model=256 MIMO, `061253` and `061327` for d_model=512 SISO, `061307` for d_model=512 MIMO; d_model=256 SISO has one CUDA Event run). 5 of the 105 percentile cells are marked. Their cause was not diagnosed. They are kept as measured and are not used for comparisons: a count that compares P50 or P99 values leaves out each config in which a compared cell is marked. Bold = lowest P50 in the column among unmarked cells.

#### d_model=256 SISO — P50 / P99 (ms)

Source: `benchmarks/results/siso_r4_extended_20260415_053314.json` (timestamp `2026-04-15T05:32:42`), field `single_step.<BS>.percentiles`.

| Backend | BS=1 | BS=8 | BS=32 | BS=128 |
|---------|------|------|-------|--------|
| PyTorch eager | 1.151 / 1.819 | 1.487 / 1.647 | 1.365 / 1.909 | 1.947 / 2.891 |
| torch.compile | 0.327 / 0.391 | 0.394 / 0.447 | 0.404 / 0.581 | 0.417 / 0.684 |
| Triton basic | 0.950 / 0.973 | 0.980 / 1.006 | 1.001 / 1.054 | 1.019 / 1.064 |
| Triton fused | 0.922 / 0.931 | 0.950 / 0.994 | 0.969 / 1.025 | 0.980 / 1.034 |
| Triton fused+Graph | 0.200 / 0.206 | 0.236 / 0.244 | 0.255 / 0.270 | 0.317 / 0.330 |
| Triton full fused | 0.228 / 0.249 | 0.231 / 0.246 | 0.245 / 0.295 | 0.309 / 0.351 |
| **Triton full+Graph** | **0.091 / 0.100** | **0.095 / 0.099** | **0.117 / 0.131** | **0.184 / 0.196** |

> torch.compile at BS=128: P50 0.417 ms against a mean of 0.900 ms in the 7-way table. The 7-way means for this cell are 0.86–0.90 ms in all 5 runs, so this gap is a difference between the timing methods, not noise in one run. The low P50 also reproduces under the old timer: the 4 earlier runs, which used `perf_counter` with a `synchronize()` around each sample, give P50 0.409–0.413 ms for this cell.

#### d_model=256 MIMO — P50 / P99 (ms)

Source: `benchmarks/results/mimo_r4_extended_20260415_053417.json` (timestamp `2026-04-15T05:33:44`), field `single_step.<BS>.percentiles`.

| Backend | BS=1 | BS=8 | BS=32 | BS=128 |
|---------|------|------|-------|--------|
| PyTorch eager | 1.236 / 1.488 | 1.360 / 1.800 | 1.539 / 1.700 | 1.622 / 2.097 |
| torch.compile | 0.472 / 0.552 | 0.526 / 0.582 | 0.541 / 0.588 | 0.546 / 0.654 |
| Triton basic | 0.924 / 1.429 | 0.959 / 0.980 | 0.970 / 1.006 | 0.971 / 0.986 |
| Triton fused | 0.881 / 1.002 | 0.929 / 0.942 | 0.938 / 0.951 | 0.945 / 0.963 |
| Triton fused+Graph | 0.121 / 0.128 | 0.231 / 0.239 | 0.226 / 0.239 | **0.269 / 0.278** |
| Triton full fused | 0.065 / 0.502<sup>†</sup> | 0.257 / 0.270 | 0.285 / 0.300 | 0.510 / 0.523 |
| Triton full+Graph | **0.070 / 0.072** | **0.150 / 0.157** | **0.177 / 0.182** | 0.295 / 0.303 |

> <sup>†</sup> At BS=1 the full fused P50 (0.065 ms) is about a third of its mean (0.180 ms) in the same run, and its P99 (0.502 ms) is 7.8x its P50. The full fused BS=1 P50 in the sibling run: 0.245 ms (`053357`). The Triton basic P99 (1.429 ms) is 1.55x its P50, under the 3x marking threshold. P99 lies between the 3rd and 2nd largest samples, so a few slow samples move it.

#### d_model=512 SISO — P50 / P99 (ms)

Source: `benchmarks/results/siso_r4_extended_20260415_061338.json` (timestamp `2026-04-15T06:13:11`), field `single_step.<BS>.percentiles`.

| Backend | BS=1 | BS=8 | BS=32 |
|---------|------|------|-------|
| PyTorch eager | 1.090 / 1.786 | 0.291 / 1.615<sup>†</sup> | 1.936 / 2.097 |
| torch.compile | 0.067 / 0.457<sup>†</sup> | 0.397 / 0.453 | 0.414 / 0.460 |
| Triton basic | 0.635 / 0.657 | 0.976 / 1.397 | 1.012 / 1.033 |
| Triton fused | 0.927 / 1.029 | 0.948 / 1.040 | 0.983 / 1.020 |
| Triton fused+Graph | 0.142 / 0.148 | 0.165 / 0.170 | **0.244 / 0.322** |
| Triton full fused | 0.239 / 0.349 | 0.238 / 0.264 | 0.392 / 0.407 |
| Triton full+Graph | **0.103 / 0.111** | **0.114 / 0.122** | 0.266 / 0.274 |

> <sup>†</sup> torch.compile at BS=1 has P50 0.067 ms but P95 0.456 ms and P99 0.457 ms; its mean in the 7-way table is 0.431 ms. torch.compile BS=1 P50 in the other CUDA Event runs: 0.332 ms (`061253`) and 0.075 ms (`061327`). PyTorch eager at BS=8 has P50 0.291 ms against a mean of 1.206 ms and P95 1.504 ms. Eager BS=8 P50 in the other runs: 1.359 ms (`061253`) and 1.327 ms (`061327`). Both cells mix a fast group and a slow group of samples. The cause was not diagnosed, so do not read these two P50 values as real speed. Triton basic at BS=1 is unmarked but unusual: its P50 (0.635 ms) is about two thirds of its 7-way mean (0.960 ms) and of its P50 in the other runs (0.968 ms in `061253`, 0.934 ms in `061327`). Its P99 is 0.657 ms, so nearly all samples of that percentile pass were fast, not only a few. It is inside both marking limits, so it is not marked, but do not read it as Triton basic beating Triton fused at BS=1.

#### d_model=512 MIMO — P50 / P99 (ms)

Source: `benchmarks/results/mimo_r4_extended_20260415_061428.json` (timestamp `2026-04-15T06:13:50`), field `single_step.<BS>.percentiles`.

| Backend | BS=1 | BS=8 | BS=32 | BS=128 |
|---------|------|------|-------|--------|
| PyTorch eager | 1.310 / 1.563 | 1.557 / 6.566<sup>†</sup> | 1.617 / 2.898 | 1.674 / 3.208 |
| torch.compile | 0.499 / 0.551 | 0.535 / 0.588 | 0.545 / 0.588 | 0.562 / 1.226 |
| Triton basic | 0.976 / 1.079 | 0.981 / 1.017 | 1.001 / 1.025 | 1.000 / 1.013 |
| Triton fused | 0.944 / 1.012 | 0.950 / 0.976 | 0.966 / 1.004 | 0.969 / 0.999 |
| Triton fused+Graph | **0.241 / 0.248** | **0.261 / 0.268** | **0.280 / 0.286** | **0.361 / 0.369** |
| Triton full fused | 0.376 / 0.387<sup>†</sup> | 0.470 / 0.480 | 0.665 / 0.941 | 1.947 / 1.965 |
| Triton full+Graph | 0.284 / 0.290 | 0.310 / 0.314 | 0.543 / 0.549 | 1.859 / 1.865 |

> ⚠️ At d_model=512 MIMO, **fused+Graph** has the lowest P50 at every batch size, not full+Graph. <sup>†</sup> PyTorch eager at BS=8 has P99 6.566 ms, 4.2x its P50. The full fused BS=1 P50 (0.376 ms) is 2.5x the sibling run's 0.147 ms (`061307`).

### Key Insights

1. **Without CUDA Graph, full-step fusion is the largest gain at d_model=256** — the full fused kernel alone gives 5.3–8.2x over eager, against 1.3–1.7x for the SSM+gate kernel (single-step mean). With CUDA Graph, the order depends on the config. At d_model=256 SISO BS=1 and MIMO BS=32 and 128, adding CUDA Graph to the SSM+gate kernel gains more than widening the fusion, so fused+Graph has a lower mean than full fused there.
2. **CUDA Graph matters most for the SSM+gate kernel** — CUDA Graph on top of the SSM+gate kernel gives 2.7–6.5x. CUDA Graph on top of the full fused kernel gives 0.85–2.49x (below 1 means slower), with the largest gain at d_model=256 SISO (all 15 configs, single-step mean).
3. **Neither fusion scope nor CUDA Graph wins everywhere** — the full fused kernel without CUDA Graph beats fused+Graph in 6 of 15 configs by mean and in 3 of the 13 configs without a <sup>†</sup> cell by P50.
4. **Comparison with torch.compile** (default mode, no CUDA Graph) — torch.compile alone gives 1.8–3.0x over eager. By mean, the fastest Triton backend beats torch.compile in 15 of 15 configs (2.0–6.0x). By P50 and by P99, it beats torch.compile in each of the 14 configs whose torch.compile cell is unmarked (d_model=512 SISO BS=1 has a <sup>†</sup> torch.compile cell). The fastest Triton backend by mean uses CUDA Graph in 14 of 15 configs, so these comparisons are not like-for-like. The full fused kernel without CUDA Graph is the like-for-like case: it beats torch.compile in 14 of 15 configs by mean and in 10 of the 12 configs without a <sup>†</sup> cell by P50. It loses at d_model=512 MIMO BS=128 under both methods and at d_model=512 MIMO BS=32 by P50. At d_model=256 it is 1.9–4.2x faster by mean and 1.1–2.0x by P50 (unmarked cells). torch.compile + CUDA Graph was not measured, so the CUDA Graph backends have no like-for-like compile baseline.
5. **Best backend depends on model size and batch size** (single-step mean) — d_model=256 SISO: best backend gives 10.3–14.0x (full+Graph at every batch size). d_model=256 MIMO: best backend gives 6.8–12.8x (full fused at BS=1, full+Graph at BS=8 and 32, fused+Graph at BS=128). d_model=512 SISO: best backend gives 6.0–8.6x (full+Graph at BS=1, fused+Graph at BS=8 and 32). d_model=512 MIMO: best backend gives 4.6–8.3x (fused+Graph at every batch size). The d_model=256 MIMO BS=1 label comes from this run only (see the note under the d_model=256 MIMO table). The 256-token loop ranks some configs differently. For example, full+Graph is the fastest there at d_model=512 MIMO BS=1 and 8.
6. **The full fused MIMO kernel trails fused+Graph at every batch size at d_model=512 and is slower than eager at BS=128** — by single-step mean and in the 256-token loop. By P50 the same holds at BS=8, 32 and 128, and the BS=1 full fused P50 cell is marked. Register pressure is the suspected cause. It has not been profiled.

### Autoregressive Decoding Speedup (256 tokens)

Speedup = eager time / backend time for the 256-token loop (see [How These Numbers Are Measured](#how-these-numbers-are-measured)). Bold = best in the row. These speedups can differ a lot from the single-step table at d_model=256: MIMO BS=1 full+Graph is 10.39x in the 256-token loop but 6.05x in the single-step table, and SISO BS=32 full+Graph is 17.37x against 10.31x.

#### d_model=256 SISO

Source: `benchmarks/results/siso_r4_extended_20260415_053314.json` (timestamp `2026-04-15T05:32:42`), field `autoregressive`.

| BS | PyTorch (ms/tok) | Triton | Fused | Fused+Graph | Full Fused | Full+Graph |
|----|------------------|--------|-------|-------------|------------|------------|
| 1 | 1.023 | 1.12x | 1.13x | 5.44x | 5.14x | **12.77x** |
| 8 | 1.171 | 1.19x | 1.25x | 5.22x | 5.81x | **11.97x** |
| 32 | 1.132 | 1.17x | 1.20x | 6.10x | 5.42x | **17.37x** |
| 128 | 1.349 | 1.40x | 1.43x | 5.94x | 6.45x | **8.05x** |

#### d_model=256 MIMO

Source: `benchmarks/results/mimo_r4_extended_20260415_053417.json` (timestamp `2026-04-15T05:33:44`), field `autoregressive`.

| BS | PyTorch (ms/tok) | Triton | Fused | Fused+Graph | Full Fused | Full+Graph |
|----|------------------|--------|-------|-------------|------------|------------|
| 1 | 1.109 | 1.26x | 1.30x | 6.54x | 6.29x | **10.39x** |
| 8 | 1.231 | 1.31x | 1.38x | 5.57x | 7.00x | **15.18x** |
| 32 | 1.292 | 1.38x | 1.43x | 5.76x | 6.87x | **12.81x** |
| 128 | 1.379 | 1.46x | 1.51x | **6.45x** | 4.97x | 4.91x |

#### d_model=512 SISO

Source: `benchmarks/results/siso_r4_extended_20260415_061338.json` (timestamp `2026-04-15T06:13:11`), field `autoregressive`.

| BS | PyTorch (ms/tok) | Triton | Fused | Fused+Graph | Full Fused | Full+Graph |
|----|------------------|--------|-------|-------------|------------|------------|
| 1 | 1.087 | 1.17x | 1.20x | 4.71x | 5.15x | **9.12x** |
| 8 | 1.316 | 1.37x | 1.42x | 7.14x | 6.42x | **13.61x** |
| 32 | 1.340 | 1.39x | 1.44x | **5.95x** | 5.41x | 5.35x |

> BS=128 was not run for d_model=512 SISO (see the 7-way ablation note).

#### d_model=512 MIMO

Source: `benchmarks/results/mimo_r4_extended_20260415_061428.json` (timestamp `2026-04-15T06:13:50`), field `autoregressive`.

| BS | PyTorch (ms/tok) | Triton | Fused | Fused+Graph | Full Fused | Full+Graph |
|----|------------------|--------|-------|-------------|------------|------------|
| 1 | 1.171 | 1.26x | 1.30x | 5.13x | 4.97x | **7.90x** |
| 8 | 1.313 | 1.37x | 1.42x | 5.62x | 4.29x | **6.75x** |
| 32 | 1.391 | 1.44x | 1.49x | **6.79x** | 2.65x | 2.65x |
| 128 | 1.402 | 1.45x | 1.50x | **4.09x** | 0.76x ⚠️ | 0.76x ⚠️ |

> ⚠️ Full+Graph is best at BS=1 and 8 (7.90x, 6.75x), fused+Graph at BS=32 and 128 (6.79x, 4.09x), in this 256-token loop. By single-step mean, fused+Graph is best at every batch size. The full fused kernel is slower than eager at BS=128 (0.76x).

### Accuracy Drift (1000-step, atol < 1e-2)

`benchmarks/test_accuracy_drift.py` runs each backend for 1000 decode steps (batch 4, seed 42, fp32 model and state) and compares its output with PyTorch eager at checkpoints 1, 10, 50, 100, 200, 500, 1000. All backends get the same random input at each step. The tables give the step-1000 values: abs_max = max |out − ref| and rel_max = max (|out − ref| / (|ref| + 1e-6)). A backend passes if abs_max < 1e-2 when the run has at least 500 steps (1e-3 for shorter runs). The CUDA Graph backends are not in this test, so it does not catch the fused+Graph angle-state defect (see [CUDA Graph Integration](#cuda-graph-integration)).

> ⚠️ These drift results were measured before the [decay sign fix](#decay-sign-fix), when the state did not decay. They must be re-run on a GPU. Until then, they describe the code before the fix.

#### d_model=256

Sources: `benchmarks/results/accuracy_drift_siso_r4_20260415_044816.json` (timestamp `2026-04-15T04:48:16`), `benchmarks/results/accuracy_drift_mimo_r4_20260415_044827.json` (timestamp `2026-04-15T04:48:27`), field `summary`.

| Backend | SISO abs_max | SISO rel_max | MIMO abs_max | MIMO rel_max |
|---------|-------------|-------------|-------------|-------------|
| torch.compile | 1.53e-5 ✅ | 1.51e-4 | 5.72e-5 ✅ | 4.14e-4 |
| Triton basic | 1.72e-5 ✅ | 2.77e-4 | 6.10e-5 ✅ | 2.53e-4 |
| Triton fused | 1.91e-5 ✅ | 2.14e-4 | 5.72e-5 ✅ | 3.07e-4 |
| Triton full fused | 2.22e-5 ✅ | 8.85e-4 | 1.28e-4 ✅ | 1.54e-4 |

#### d_model=512

Sources: `benchmarks/results/accuracy_drift_siso_r4_20260415_053438.json` (timestamp `2026-04-15T05:34:38`), `benchmarks/results/accuracy_drift_mimo_r4_20260415_053449.json` (timestamp `2026-04-15T05:34:49`), field `summary`.

| Backend | SISO abs_max | SISO rel_max | MIMO abs_max | MIMO rel_max |
|---------|-------------|-------------|-------------|-------------|
| torch.compile | 2.00e-5 ✅ | 5.19e-4 | 1.07e-4 ✅ | 1.99e-4 |
| Triton basic | 2.67e-5 ✅ | 9.13e-4 | 1.22e-4 ✅ | 5.64e-4 |
| Triton fused | 2.29e-5 ✅ | 7.75e-4 | 1.22e-4 ✅ | 6.64e-4 |
| Triton full fused | 4.96e-5 ✅ | 4.68e-3 | 2.29e-4 ✅ | 1.57e-3 |

> All four tested backends pass. The largest abs_max is 2.29e-4 (d_model=512 MIMO, full fused), and the largest error at any checkpoint is 2.29e-4. The full fused kernel has the largest abs_max in each abs_max column. The test does not show why.

## Kernel Design

### Full-Step Fused Kernel

The main design: **fuses every decode-step op between `in_proj` and `out_proj` into a single Triton kernel launch**. Only `in_proj` (linear projection) and `out_proj` remain as separate PyTorch ops.

```
Input: zxBCdtAtrap (B, d_in_proj) — raw output from in_proj
  ↓
[Single Triton Kernel]
  1. Split → z, x, B_raw, C_raw, dd_dt, dd_A, trap_raw, angle_raw
  2. Rearrange → z(B,H,P), x(B,H,P), B_raw(D), C_raw(D)
  3. (-softplus(dd_A)).clamp(max=-A_floor) → A (A ≤ -A_floor < 0, see Decay sign fix);
     softplus(dd_dt + dt_bias) → DT
  4. sigmoid(trap_raw) → trap
  5. RMSNorm(B_raw), RMSNorm(C_raw) → B_normed, C_normed
  6. Add B_bias, C_bias → B_exp, C_exp
  7. RoPE(angle_state) → B_proj, C_proj
  8. SSM recurrence + silu gate → y_gated
  ↓
Output: y_gated (B, H, P) → out_proj → final output
```

**Why this matters**: The partial-fusion backends (Triton, Fused) leave the other decode ops outside the kernel, each with its own kernel launch. The profiler traces in `benchmarks/results/` (MIMO R=4, d_model=512, headdim=64, batch 4, 5 steps; headdim=64 is the `profile_step.py` default, while the benchmarks use headdim=32): eager launches 70 kernels per step and the Triton basic path 51. GPU kernels are active for only 14% of the kernel span in both traces, so the step is launch-bound. The profiler records call stacks, which adds CPU time of its own. Full-step fusion removes most of these launches (from the code; the repo has no profiler trace of the full fused path): at d_model=256 the full fused kernel is 3.5–5.0x faster than the SSM+gate kernel by single-step mean.

### Decay sign fix

- **Bug.** Through commit `4a98872`, `Mamba3._compute_params` and the three `Mamba3Decoder` steps that compute `A` used `A = -F.softplus(dd_A.float()).clamp(max=-A_floor)`. Python applies `.clamp` before the unary minus, so this is `-(softplus(dd_A).clamp(max=-A_floor))`. softplus is never negative, so the clamp always returned `-A_floor`, and `A` was the constant `+A_floor`. `A` did not depend on `dd_A`, and `DT` is positive, so `exp(A·DT)` was above one: each step scaled the state up instead of decaying it. The full-step kernel copied the expression on purpose (`A_val = -tl.where(sp_A > -A_floor, -A_floor, sp_A)`). The upstream `mamba3.py` at the pinned submodule commit `1aa707d` has the expression only in `step()`. Its `forward()` negates first and then clamps, which gives a negative `A`.
- **History.** The expression was first flagged on 2026-04-15, in a kernel comment from commit `5c016ba`. The comment called it "technically a bug in the reference code (operator precedence)", but the kernel kept it to match the PyTorch code. On 2026-11-10, commit `4a98872` recorded the constant decay in this README as known behavior.
- **Fix** (2026-11-11). Every site now computes the `A` of upstream `forward()`, which negates before it clamps. The PyTorch code computes `A = (-F.softplus(dd_A.float())).clamp(max=-A_floor)` in `Mamba3._compute_params`, which `forward` and `step` share ([`src/models/mamba3.py` L196](src/models/mamba3.py#L196)), and in the torch.compile, Triton basic and fused decode steps ([`src/models/inference.py` L225](src/models/inference.py#L225), [L331](src/models/inference.py#L331), [L422](src/models/inference.py#L422)). The full-step kernel computes `A_val = -tl.where(sp_A < A_floor, A_floor, sp_A)`, which is `-max(softplus(dd_A), A_floor)` and keeps a NaN as NaN, as `torch.clamp` does ([`src/kernels/fused_full_decode.py` L104](src/kernels/fused_full_decode.py#L104) for SISO, [L362](src/kernels/fused_full_decode.py#L362) for MIMO). So `A ≤ -A_floor`, `A` depends on `dd_A`, and `exp(A·DT)` is below one: the state decays.
- **All result files predate the fix.** Every JSON file in `benchmarks/results/` (timings, accuracy drift and profiler traces) was produced before this fix. No measured value in this README was changed. No branch or loop bound depends on `A`: the kernels use it only in arithmetic, for example `ADT_val = A_val * DT_val` and `tl.exp(ADT_val)` ([L107–L109](src/kernels/fused_full_decode.py#L107)). So the timings are not expected to change, but they were not re-measured. The drift tables compare each backend with PyTorch eager on a state that did not decay. These drift values must be re-run on a GPU.
- **Tests.** All backends compute `A` the same way, before and after the fix, so `test_full_fused.py` and `test_accuracy_drift.py` still compare like with like. `test_siso.py` and `test_mimo.py` call the Triton basic SSM kernels directly with a random negative `A_log` ([`src/tests/test_siso.py` L37](src/tests/test_siso.py#L37), [`src/tests/test_mimo.py` L38](src/tests/test_mimo.py#L38)). No test calls the SSM+gate kernels or the full-step kernel directly. [`src/tests/test_decay_sign.py`](src/tests/test_decay_sign.py), added with the fix, checks the sign of the model's `A`. For SISO and MIMO, it asserts that `Mamba3._compute_params` gives `A ≤ -A_floor` and the `A` of upstream `forward()`. On CUDA, it also runs random steps and then zero-input steps through the full-step kernel. It asserts that the kernel's state stays close to the eager decoder's state and that the state norm goes down. On the code before the fix, the `_compute_params` rows fail.

### SISO Fused Kernel (Partial Fusion)
- One Triton program per `(batch, head)` pair
- P dimension iterated inside kernel; D dimension tiled with BLOCK_D
- Fuses: `B*x → trapezoidal blend → state update → C*h + D*x`
- State shape: `(B, H, P, D)` — full outer-product

### MIMO Fused Kernel
- One Triton program per `(batch, head)` pair
- R loop fully unrolled (separate kernels for R=1,2,4,8)
- Fuses: `x→R scalars → R rank-1 B*x → trapezoidal → state update → R rank-1 C*h → up-project`
- State shape: `(B, H, D)` — no P axis, so 32x smaller than SISO at headdim=32

### CUDA Graph Integration
- Removes per-kernel CPU launch work: `step()` copies the input into a static buffer and replays the captured graph
- Captures one call of the chosen backend's step function (`warmup_cuda_graph` in `src/models/inference.py`): `in_proj` → that backend's decode ops → `out_proj`. With Triton basic the silu gate is a separate PyTorch op, with fused+Graph it is inside the SSM+gate kernel, and with full+Graph one kernel runs every op between the two GEMMs
- For full+Graph, the kernel updates all three state tensors (SSM state, previous B·x, RoPE angle) in place, so each replay advances the state
- For fused+Graph, the SSM kernel updates the SSM state and previous B·x in place, but `_step_triton_fused` builds the RoPE angle as a new tensor (`state["angle_state"] = state["angle_state"] + delta_angle`). The captured graph keeps reading the memory of the pre-capture angle tensor, so replays do not accumulate the RoPE angle. This is a correctness defect in fused+Graph. The graph replays the same kernels either way, so the defect is not expected to change the timings, but that was not measured. No test covers the CUDA Graph backends
- CUDA Graph on top of the SSM+gate kernel gives 2.7–6.5x (all 15 configs, single-step mean)

### torch.compile Support
- Added `use_compile` mode to `Mamba3Decoder` for fair comparison
- Uses `torch.compile(fullgraph=True)` on the pure-functional decode step, in the default mode (no CUDA Graph)
- torch.compile gives 1.8–3.0x over eager via Inductor (single-step mean)
- "Why not just use torch.compile?": see [Key Insights](#key-insights). `Mamba3Decoder` also has a `use_compile_cuda_graph` mode, but it was not benchmarked

## References

| Resource | Link |
|----------|------|
| Mamba-3 Paper | [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) |
| Official Implementation | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| Readable PyTorch Reference | [rishikksh20/mamba3-pytorch](https://github.com/rishikksh20/mamba3-pytorch) — `src/models/mamba3.py` is adapted from this implementation |
| Mamba-2 Fused Kernel Blog | [PyTorch Blog](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) |
| Together AI Mamba-3 Blog | [Blog Post](https://www.together.ai/blog/mamba-3) |

## Milestones

- [x] Project scaffolding and planning
- [x] Environment setup + algorithm deep-dive
- [x] SISO fused kernel (core + trapezoidal + RoPE)
- [x] MIMO fused kernel + correctness tests
- [x] Performance optimization and Triton tuning
- [x] CUDA Graph integration
- [x] Full benchmark suite + documentation
- [x] **Full-step fused kernel** — all non-GEMM decode ops in one kernel launch (5.3–8.2x over eager by single-step mean at d_model=256)
- [x] **Full fused + CUDA Graph** — 14.0x over eager by single-step mean at the best config (d_model=256 SISO BS=1)
- [x] **P0: Autoregressive benchmark** — 256-token decode loop. BS=1 full+Graph: 12.77x SISO and 10.39x MIMO at d_model=256. AR speedups differ from single-step ones (see the AR section)
- [x] **P1: Multi-step accuracy drift** — 1000-step decode, all four tested backends (no CUDA Graph backends) < 1.3e-4 (d_model=256) and < 2.3e-4 (d_model=512) max abs error at step 1000 (measured before the decay sign fix)
- [x] **P2: d_model=512 7-way ablation** — best backend 6.0–8.6x (SISO) and 4.6–8.3x (MIMO) by single-step mean. The full fused MIMO kernel slows down at large batch (suspected register pressure)
- [x] **P0: Re-run d_model=256 benchmark with CUDA Event timing** — Switched the percentile pass to CUDA Events after one old-timer run (`052234`, d_model=512 MIMO) put 4 of 7 BS=1 P50 values at 2.65–2.75 ms. Other old-timer runs, including `052204` with the same config, did not show this (see Latency Tail). The earlier 19.2x figure (full+Graph, run `035152`) did not reproduce in later runs (see Run-to-run variance)
- [x] **P1: d_model=512 accuracy drift** — 1000-step: SISO < 5.0e-5, MIMO < 2.3e-4, all four tested backends PASS (measured before the decay sign fix)
- [ ] Re-run the accuracy drift test on a GPU after the [decay sign fix](#decay-sign-fix)

> **P2 Phase Complete** (2026-04-15) — d_model=512 benchmarks show the full fused MIMO kernel slowing down at large batch. Fused+Graph is the fastest backend for d_model=512 MIMO at every batch size by single-step mean (8.3x at BS=32). In the 256-token loop, full+Graph is the fastest there at BS=1 and 8. d_model=256 percentiles were re-run with CUDA Event timing. d_model=512 accuracy drift: all four tested backends PASS (< 2.3e-4), before the decay sign fix.

## Future Plans

The following are optional enhancements for future development:

- [ ] dtype optimization (fp16/bf16 intermediate variables) — not implemented yet: `src/` has no fp16/bf16 code
- [ ] Register pressure analysis via `triton.testing`
- [ ] Cache hint optimization
- [x] `torch.compile` compatibility test — **COMPLETED** (default mode, see the 7-way ablation; compile + CUDA Graph not benchmarked)
- [x] Larger model benchmarks (d_model=512) — **COMPLETED** (7-way ablation: best backend 6.0–8.6x (SISO) and 4.6–8.3x (MIMO) by single-step mean)
- [x] Full-step fusion — **COMPLETED** (5.3–8.2x over eager at d_model=256, and 14.0x with CUDA Graph at SISO BS=1, by single-step mean)
- [ ] Prefill kernel fusion
- [ ] H100 benchmark comparison
- [x] Multi-step accuracy drift test (1000 steps) — **COMPLETED** before the decay sign fix (all four tested backends < 1.3e-4 (d_model=256) and < 2.3e-4 (d_model=512)). Re-run pending
- [x] Autoregressive benchmark (6-way) — **COMPLETED** (AR and single-step speedups differ; see the AR section)
- [ ] MIMO full fused kernel register optimization for d_model≥512
