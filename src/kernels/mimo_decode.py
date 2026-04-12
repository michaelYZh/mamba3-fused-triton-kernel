"""
Mamba-3 MIMO fused Triton decoding kernel.

Fuses the entire single-step MIMO decode into one kernel launch:
  [Load state] → [Project x to R scalars] → [Accumulate R rank-1 B*x] →
  [Trapezoidal blend] → [State update] → [R rank-1 C*h output] → [Up-project] → [Write-back]

Key difference from SISO:
  - State is (B, H, D) instead of (B, H, P, D) — P is projected away
  - R rank-1 updates instead of one full outer-product update
  - Higher arithmetic intensity: more FLOPs for same memory traffic

Each program handles one (batch, head) pair, tiling over D dimension.
The R loop is fully unrolled (R is small, typically 4).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mimo_decode_kernel(
    # I/O pointers
    x_ptr,           # (B, H, P) — current input values
    h_ptr,           # (B, H, D) — SSM state (in-out)
    bx_prev_ptr,     # (B, H, D) — previous B*x (in-out)
    y_ptr,           # (B, H, P) — output

    # Parameter pointers (read-only)
    B_proj_ptr,      # (B, R, H, D)
    C_proj_ptr,      # (B, R, H, D)
    ADT_ptr,         # (B, H)
    DT_ptr,          # (B, H)
    trap_ptr,        # (B, H)
    D_skip_ptr,      # (H,)
    mimo_x_ptr,      # (H, R, P)
    mimo_o_ptr,      # (H, R, P)

    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,

    # Strides for x: (B, H, P)
    stride_x_b, stride_x_h, stride_x_p,
    # Strides for h: (B, H, D)
    stride_h_b, stride_h_h, stride_h_d,
    # Strides for bx_prev: same as h
    stride_bx_b, stride_bx_h, stride_bx_d,
    # Strides for y: same as x
    stride_y_b, stride_y_h, stride_y_p,
    # Strides for B_proj: (B, R, H, D)
    stride_B_b, stride_B_r, stride_B_h, stride_B_d,
    # Strides for C_proj: same as B_proj
    stride_C_b, stride_C_r, stride_C_h, stride_C_d,
    # Strides for ADT, DT, trap: (B, H)
    stride_adt_b, stride_adt_h,
    # Strides for D_skip: (H,)
    stride_D_h,
    # Strides for mimo_x: (H, R, P)
    stride_mx_h, stride_mx_r, stride_mx_p,
    # Strides for mimo_o: (H, R, P)
    stride_mo_h, stride_mo_r, stride_mo_p,

    # Block sizes
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """MIMO decode kernel: one program per (batch, head) pair."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # ── Load scalar parameters ──────────────────────────────────────────
    adt_val = tl.load(ADT_ptr + b * stride_adt_b + h * stride_adt_h)
    dt_val  = tl.load(DT_ptr + b * stride_adt_b + h * stride_adt_h)
    trap_val = tl.load(trap_ptr + b * stride_adt_b + h * stride_adt_h)
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)

    decay = tl.exp(adt_val)  # scalar

    # ── D-tile offsets ──────────────────────────────────────────────────
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # ── Load current state h[b, h, d] and bx_prev[b, h, d] ─────────────
    h_base = b * stride_h_b + h * stride_h_h
    h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
    bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)

    # ── Step 1: Project x from P to R scalars: x_r = einsum(x, mimo_x) ─
    # x_r[r] = sum_p x[b,h,p] * mimo_x[h,r,p]
    x_r = tl.zeros([R], dtype=tl.float32)
    for p_idx in range(P):
        x_p = tl.load(x_ptr + b * stride_x_b + h * stride_x_h + p_idx * stride_x_p)
        for r_idx in range(R):
            mx = tl.load(mimo_x_ptr + h * stride_mx_h + r_idx * stride_mx_r + p_idx * stride_mx_p)
            x_r[r_idx] += x_p * mx

    # ── Step 2: Accumulate R rank-1 Bx contributions ────────────────────
    # Bx_curr[d] = sum_r x_r[r] * B_proj[b,r,h,d]
    Bx_curr = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r_idx in range(R):
        x_r_val = x_r[r_idx]
        B_r = tl.load(
            B_proj_ptr + b * stride_B_b + r_idx * stride_B_r + h * stride_B_h + d_offsets * stride_B_d,
            mask=d_mask, other=0.0,
        )
        Bx_curr += x_r_val * B_r

    # ── Step 3: Trapezoidal blend ───────────────────────────────────────
    Bx_blended = (1.0 - trap_val) * Bx_curr + trap_val * 0.5 * (Bx_curr + bx_prev_vals)

    # ── Step 4: State update ────────────────────────────────────────────
    h_new = decay * h_vals + dt_val * Bx_blended

    # ── Step 5: Per-rank output scalars + skip + up-project ─────────────
    # y_r[r] = sum_d C_proj[b,r,h,d] * h_new[d]   (scalar per rank)
    # skip_r[r] = D[h] * x_r[r]
    # y[p] = sum_r (y_r[r] + skip_r[r]) * mimo_o[h,r,p]

    # Compute y_r + skip for each rank
    y_r_plus_skip = tl.zeros([R], dtype=tl.float32)
    for r_idx in range(R):
        C_r = tl.load(
            C_proj_ptr + b * stride_C_b + r_idx * stride_C_r + h * stride_C_h + d_offsets * stride_C_d,
            mask=d_mask, other=0.0,
        )
        y_r_scalar = tl.sum(C_r * h_new, axis=0)  # scalar
        y_r_plus_skip[r_idx] = y_r_scalar + d_skip_val * x_r[r_idx]

    # Up-project: y[p] = sum_r y_r_plus_skip[r] * mimo_o[h,r,p]
    for p_idx in range(P):
        y_p = 0.0
        for r_idx in range(R):
            mo = tl.load(mimo_o_ptr + h * stride_mo_h + r_idx * stride_mo_r + p_idx * stride_mo_p)
            y_p += y_r_plus_skip[r_idx] * mo
        tl.store(y_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_p)

    # ── Write back h and bx_prev ───────────────────────────────────────
    tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
    tl.store(bx_prev_ptr + h_base + d_offsets * stride_h_d, Bx_curr, mask=d_mask)


def mamba3_mimo_decode_triton(
    x: torch.Tensor,       # (B, H, P)
    B_proj: torch.Tensor,  # (B, R, H, D)
    C_proj: torch.Tensor,  # (B, R, H, D)
    ADT: torch.Tensor,     # (B, H)
    DT: torch.Tensor,      # (B, H)
    trap: torch.Tensor,    # (B, H)
    D_skip: torch.Tensor,  # (H,)
    mimo_x: torch.Tensor,  # (H, R, P)
    mimo_o: torch.Tensor,  # (H, R, P)
    h: torch.Tensor,       # (B, H, D)
    bx_prev: torch.Tensor, # (B, H, D)
) -> tuple:
    """Launch the MIMO fused decode kernel.

    Returns:
        y:         (B, H, P)
        h:         (B, H, D) — updated in-place
        bx_prev:   (B, H, D) — updated in-place
    """
    B_dim = x.shape[0]
    H_dim = x.shape[1]
    P_dim = x.shape[2]
    R_dim = B_proj.shape[1]
    D_dim = B_proj.shape[-1]

    assert h.shape == (B_dim, H_dim, D_dim)
    assert bx_prev.shape == (B_dim, H_dim, D_dim)
    assert mimo_x.shape == (H_dim, R_dim, P_dim)
    assert mimo_o.shape == (H_dim, R_dim, P_dim)
    assert x.is_cuda and h.is_cuda

    y = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D_dim)
    BLOCK_P = 1  # We iterate over P inside the kernel

    grid = (B_dim * H_dim,)

    _mimo_decode_kernel[grid](
        x, h, bx_prev, y,
        B_proj, C_proj, ADT, DT, trap, D_skip, mimo_x, mimo_o,
        B_dim, H_dim, P_dim, D_dim, R_dim,
        x.stride(0), x.stride(1), x.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        bx_prev.stride(0), bx_prev.stride(1), bx_prev.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        B_proj.stride(0), B_proj.stride(1), B_proj.stride(2), B_proj.stride(3),
        C_proj.stride(0), C_proj.stride(1), C_proj.stride(2), C_proj.stride(3),
        ADT.stride(0), ADT.stride(1),
        D_skip.stride(0),
        mimo_x.stride(0), mimo_x.stride(1), mimo_x.stride(2),
        mimo_o.stride(0), mimo_o.stride(1), mimo_o.stride(2),
        BLOCK_D=BLOCK_D,
        BLOCK_P=BLOCK_P,
    )

    return y, h, bx_prev
