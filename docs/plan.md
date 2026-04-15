# Mamba-3 Fused Triton Kernel — 1-Week MVP Plan

> **Goal**: Replace Python-based Mamba-3 decoding with a fused Triton kernel implementing MIMO formulation and exponential-trapezoidal discretization. Achieve measurable speedup on A100/H100.

---

## 1. Core Math You Need to Understand

### 1.1 Continuous-Time SSM

$$\dot{h}(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

### 1.2 Exponential-Trapezoidal Discretization (Mamba-3 §3.1)

The biggest algorithmic change from Mamba-2. Instead of simple Euler/ZOH, Mamba-3 uses trapezoidal integration with exponential kernel:

```
h_t = exp(A·dt_t) · h_{t-1} + dt_t · trap_t · (B_t·x_t + B_{t-1}·x_{t-1}) / 2
```

Key components:
- **`dt_t`**: Per-token step size (selective mechanism)
- **`trap_t`**: Learned sigmoid gate interpolating between pure Euler (trap→0) and full trapezoidal (trap→1)
- **`bx_prev = B_{t-1}·x_{t-1}`**: **Trapezoidal memory** — extra state that must be carried across decoding steps
- **`exp(A·dt_t)`**: Diagonal matrix exponential decay (A is structured diagonal)

### 1.3 MIMO Formulation (Mamba-3 §3.3)

**Problem**: Mamba-2 is SISO — state shape `(P×D)`, one rank-1 update per token. Decoding is memory-bound with extremely low arithmetic intensity.

**Mamba-3 MIMO solution**:
- Expand B and C to R parallel streams (`mimo_rank = R`, default R=4)
- State shrinks from `(B, H, P, D)` to `(B, H, D)` — **2x smaller state**
- R rank-1 updates per token instead of 1 — higher arithmetic intensity
- Memory-bound decoding absorbs extra FLOPs via idle Tensor Cores

```
# R parallel streams sharing D-dim state
for r in range(R):
    x_r = mimo_x_proj(x)        # (B, H, R) — R scalar inputs
    B_r = mimo_B_proj[..., r]    # (H, D) — R D-dim vectors
    h = h + x_r[..., r] * B_r    # accumulate R times
y = sum(C_r * h for r in range(R))  # R weighted outputs
```

### 1.4 Complex-Valued State via RoPE (Mamba-3 §3.2)

- Apply Rotary Position Embedding to B and C projections
- Learned "angle" projections accumulate via dt
- Essentially applies complex rotation to SSM parameters
- Standard RoPE on dimension pairs `(dim[2i], dim[2i+1])`

### 1.5 Full Decoding Step Computation Flow

```
1. Input proj:    x = x_proj(token_embed)                    # (B, H, D)
2. Params:        dt = softplus(dt_proj(x))                  # (B, H)
                  A = A_log                                    # (H, D)
                  B, C = B_proj(x), C_proj(x)                  # (B, H, D)
3. RoPE:          B, C = apply_rope(B, C, angle_state)
4. Trapezoidal:   bx = B * x                                   # (B, H, D)
                  trap = sigmoid(trap_proj(x))                  # (B, H)
                  bx_trap = (bx + bx_prev) * 0.5 * dt * trap
5. MIMO expand:   x_r = mimo_x_proj(x)                        # (B, H, R)
                  B_mimo, C_mimo = mimo_proj(B), mimo_proj(C)
6. State update:  dA = exp(A * dt)
                  h = dA * h + sum_r(x_r[..., r] * B_mimo[..., r, :] * dt) + bx_trap
7. Output:        y = D * x + sum_r(C_mimo[..., r, :] * h)
8. State save:    angle_state, ssm_state=h, bx_prev=bx
```

---

## 2. Reference Implementations

| Resource | Description | Link |
|----------|-------------|------|
| **Mamba-3 Paper** | Full math, §3 algorithm + §5 inference optimization | [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) |
| **Official Code** | Together AI implementation (Triton + CuTe DSL + TileLang) | [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| **Readable PyTorch** | Pure PyTorch reference for understanding the algorithm | [rishikksh20/mamba3-pytorch](https://github.com/rishikksh20/mamba3-pytorch) |
| **Mamba-2 Fused Kernel** | PyTorch blog — kernel fusion design patterns | [PyTorch Blog](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/) |
| **Together AI Blog** | Official inference optimization strategy & benchmarks | [Together AI Blog](https://www.together.ai/blog/mamba-3) |

---

## 3. MVP Technical Approach

### 3.1 Scope

**In scope**:
- SISO + MIMO fused Triton decoding kernels
- Exponential-trapezoidal discretization
- RoPE angle accumulation
- `torch.compile` compatibility
- Correctness validation (vs PyTorch reference)
- Latency benchmarks

**Out of scope** (for now):
- Prefill kernel fusion
- Continuous batching / speculative decoding
- Quantization (INT8/INT4)
- CuTe DSL / TileLang

### 3.2 Fusion Strategy

**Current Python decoding problems**:
1. Multiple `torch.compile` may not fully fuse independent ops
2. Intermediate tensors (bx, dA, trap) frequently written/read from global memory
3. Extremely low GPU utilization at small batch sizes

**Fused kernel merges into one launch**:
```
[Input proj] → [Param compute(dt,A,B,C)] → [RoPE] → [Trapezoidal] → [MIMO unfold] → [State update] → [Output]
```

### 3.3 Triton Kernel Design Sketch

```python
import triton
import triton.language as tl

@triton.jit
def mamba3_decoding_kernel(
    # I/O pointers
    x_ptr,        # (B, H, D) — current input
    h_ptr,        # (B, H, D) — SSM state (in-out)
    bx_prev_ptr,  # (B, H, D) — previous step bx (in-out)
    angle_ptr,    # (B, H, D//2) — RoPE angle accumulator (in-out)
    y_ptr,        # (B, H, D) — output

    # Parameter pointers (read-only)
    A_log_ptr,    # (H, D)
    D_ptr,        # (H, D) — skip connection
    dt_bias_ptr,  # (H,)
    B_ptr, C_ptr, # Pre-projected (B, H, D)
    trap_logit_ptr,  # (H, D) — trapezoidal gate

    # MIMO pointers
    mimo_x_proj_ptr,  # (B, H, R)
    mimo_B_ptr,       # (B, H, R, D)
    mimo_C_ptr,       # (B, H, R, D)

    # Dimensions
    B, H, D, R,
    stride_b, stride_h, stride_d,

    # Block size
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H

    # Offsets
    bh_offset = b * stride_b + h * stride_h
    d_offsets = tl.arange(0, BLOCK_D)

    # 1. Load states
    h_state = tl.load(h_ptr + bh_offset + d_offsets, mask=d_offsets < D)
    bx_prev = tl.load(bx_prev_ptr + bh_offset + d_offsets, mask=d_offsets < D)
    x = tl.load(x_ptr + bh_offset + d_offsets, mask=d_offsets < D)

    # 2. Discretization
    A = tl.load(A_log_ptr + h * stride_d + d_offsets, mask=d_offsets < D)
    dt = ...  # loaded or pre-computed
    dA = tl.exp(A * dt)

    # 3. Trapezoidal discretization
    bx = B * x
    trap = tl.sigmoid(trap_logit)
    bx_trap = (bx + bx_prev) * 0.5 * dt * trap

    # 4. MIMO state update
    h_state = dA * h_state + bx_trap * dt
    for r in range(R):
        x_r = tl.load(mimo_x_proj_ptr + bh_offset + r, ...)
        B_r = tl.load(mimo_B_ptr + bh_offset + r * D + d_offsets, ...)
        h_state = h_state + x_r * B_r * dt

    # 5. Output
    y = D * x
    for r in range(R):
        C_r = tl.load(mimo_C_ptr + bh_offset + r * D + d_offsets, ...)
        y = y + C_r * h_state

    # 6. Write back
    tl.store(h_ptr + bh_offset + d_offsets, h_state, mask=d_offsets < D)
    tl.store(bx_prev_ptr + bh_offset + d_offsets, bx, mask=d_offsets < D)
    tl.store(y_ptr + bh_offset + d_offsets, y, mask=d_offsets < D)
```

### 3.4 Key Optimization Techniques (from Mamba-2 experience)

| Optimization | Source | Expected Gain |
|-------------|--------|--------------|
| **Single kernel launch** | Mamba-2 fusion | Eliminate 5+ launch overheads |
| **Intermediate results in registers** | Kernel fusion | ~60% less global memory R/W |
| **fp16 intermediate compute** | Mamba-2 optimization | ~16% extra speedup |
| **Per-(batch, head) program** | Natural parallelism | Parallelism = B×H |
| **MIMO loop unrolling** | Mamba-3 | R=4 loop is tiny, unroll fully |
| **Compile-time mask** | `@triton.heuristics` | Skip mask when D % BLOCK_D == 0 |

---

## 4. Weekly Execution Plan

### Day 1 (Mon): Environment Setup + Algorithm Deep-Dive

**Morning**:
- [ ] Clone [rishikksh20/mamba3-pytorch](https://github.com/rishikksh20/mamba3-pytorch) and [state-spaces/mamba](https://github.com/state-spaces/mamba)
- [ ] Run PyTorch reference SISO decoding (single-step inference)
- [ ] Verify GPU environment (A100/H100), install Triton

**Afternoon**:
- [ ] Walk through `mamba3_siso_scan` and `mamba3_mimo_scan` line by line
- [ ] Map out the **complete computation graph** (tensor shapes + dependencies)
- [ ] Profile Python decoding with `torch.profiler` — quantify the bottleneck

**Deliverable**: Computation graph doc + profiler results

### Day 2 (Tue): SISO Fused Kernel — Core Implementation

**Morning**:
- [ ] Implement minimal SISO decoding kernel (no trapezoidal, no RoPE)
  - Input: `x, h, A_log, B, C, D, dt`
  - Output: `y, h_new`
  - Validate: compare with `h = dA * h + B * x * dt; y = C * h + D * x`

**Afternoon**:
- [ ] Add exponential-trapezoidal discretization
  - New inputs: `bx_prev, trap_logit`
- [ ] Add RoPE angle accumulation
  - New in/out: `angle_state`
- [ ] Write correctness tests (atol=1e-2, per Mamba-2 standard)

**Deliverable**: Correctness-validated SISO fused kernel

### Day 3 (Wed): MIMO Fused Kernel

**Morning**:
- [ ] Understand exact MIMO projection and state update formulas
- [ ] Extend kernel to support `mimo_rank=R` parameter
- [ ] Trapezoidal discretization under MIMO path (bx_trap × MIMO interaction)

**Afternoon**:
- [ ] MIMO kernel correctness validation
- [ ] Initial benchmark: SISO vs MIMO single-step latency (batch=1, 8, 32, 128)
- [ ] Analyze MIMO arithmetic intensity improvement

**Deliverable**: Correctness-validated MIMO fused kernel + initial benchmark

### Day 4 (Thu): Performance Tuning

**Morning**:
- [ ] Block size tuning (D tiling for D=64/128/256)
- [ ] Dtype optimization: which intermediates can safely be fp16/bf16
- [ ] Register pressure analysis via `triton.testing`

**Afternoon**:
- [ ] Cache hint optimization (streaming vs eviction priority)
- [ ] `torch.compile` compatibility test
- [ ] Optional: TMA (Hopper GPU) for state loading

**Deliverable**: Optimized kernel + performance analysis

### Day 5 (Fri): Full Benchmark + Documentation + Demo

**Morning**:
- [ ] Complete benchmark suite:
  - Python reference vs Fused Kernel
  - Varying batch sizes (1, 8, 32, 128)
  - Total decode time at different sequence lengths
  - SISO vs MIMO (R=4)
  - A100 vs H100 (if available)

**Afternoon**:
- [ ] Finalize README with install, usage, results
- [ ] End-to-end demo (token in → token out)
- [ ] Cleanup, tests, CI

**Deliverable**: Complete project + benchmark report + demo

### Day 6-7 (Weekend Buffer) ✅

- [x] Edge cases (different D values, head counts)
- [ ] Optional: partial prefill kernel fusion *(future)*
- [ ] Prepare presentation *(future)*

---

## 5. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Official code already has Triton kernel | High | Lower MVP novelty | Focus on **incremental optimization** or **different fusion strategy** |
| Triton trapezoidal precision issues | Medium | Test failure | Use fp32 state + fp16 compute (Mamba-2 strategy) |
| bx_prev cross-step dependency blocks fusion | Medium | Suboptimal perf | Pass bx_prev as kernel input, not across-step |
| MIMO loop unroll causes register spillover | Low | Compile fail / low occupancy | Reduce block size or split into 2 kernels |
| Paper formulas differ from actual code | Medium | Wrong direction | Follow official code over paper |

---

## 6. Success Criteria

| Metric | MVP Target | Excellent |
|--------|-----------|-----------|
| **Correctness** | vs PyTorch reference atol < 1e-2 | atol < 1e-3 |
| **Single-step latency (batch=1)** | 1.5x faster than Python | 2x+ faster |
| **End-to-end decode (seq=1K)** | 1.3x faster than Python | 1.5x+ faster |
| **Code quality** | Passes correctness tests + benchmark | Integrable into inference framework |

---

*Generated: 2026-04-05*
*Tech stack: Triton + PyTorch + Python*
