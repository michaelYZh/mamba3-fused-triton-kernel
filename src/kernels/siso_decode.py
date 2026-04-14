"""
Mamba-3 SISO fused Triton decoding kernel.

Fuses the entire single-step SISO decode into one kernel launch:
  [Load state] → [Compute dA=exp(A*dt)] → [Compute Bx_curr = B*x] →
  [Trapezoidal blend] → [State update] → [Output y = C*h + D*x] → [Write-back]

Each program handles one (batch, head) pair, iterating over P and D dimensions.
P is iterated inside the kernel as a loop; D is tiled with BLOCK_D.

State shapes (SISO):
  ssm_state:  (B, H, P, D)  — full outer-product state
  bx_prev:    (B, H, P, D)  — previous B*x for trapezoidal integration
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _siso_decode_kernel(
    # I/O pointers
    x_ptr,           # (B, H, P) — current input values
    h_ptr,           # (B, H, P, D) — SSM state (in-out)
    bx_prev_ptr,     # (B, H, P, D) — previous B*x (in-out)
    y_ptr,           # (B, H, P) — output

    # Parameter pointers (read-only)
    B_proj_ptr,      # (B, H, D)
    C_proj_ptr,      # (B, H, D)
    ADT_ptr,         # (B, H) — A*dt (negative, so exp(ADT) < 1)
    DT_ptr,          # (B, H) — dt (positive)
    trap_ptr,        # (B, H) — sigmoid gate in [0,1]
    D_skip_ptr,      # (H,) — skip connection weight

    # Dimensions
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # num heads
    P: tl.constexpr,  # headdim
    D: tl.constexpr,  # d_state

    # Strides for x: (B, H, P)
    stride_x_b, stride_x_h, stride_x_p,
    # Strides for h: (B, H, P, D)
    stride_h_b, stride_h_h, stride_h_p, stride_h_d,
    # Strides for bx_prev: same as h
    stride_bx_b, stride_bx_h, stride_bx_p, stride_bx_d,
    # Strides for y: same as x
    stride_y_b, stride_y_h, stride_y_p,
    # Strides for B_proj: (B, H, D)
    stride_B_b, stride_B_h, stride_B_d,
    # Strides for C_proj: same as B_proj
    stride_C_b, stride_C_h, stride_C_d,
    # Strides for ADT, DT, trap: (B, H)
    stride_adt_b, stride_adt_h,
    # Stride for D_skip: (H,)
    stride_D_h,

    # Block sizes
    BLOCK_D: tl.constexpr,
):
    """SISO decode kernel: one program per (batch, head) pair."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    # ── Load scalar parameters for this (b, h) ──────────────────────────
    adt_val = tl.load(ADT_ptr + b * stride_adt_b + h * stride_adt_h)
    dt_val  = tl.load(DT_ptr + b * stride_adt_b + h * stride_adt_h)
    trap_val = tl.load(trap_ptr + b * stride_adt_b + h * stride_adt_h)
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)

    decay = tl.exp(adt_val)  # exp(A*dt), scalar

    # ── D-tile offsets ──────────────────────────────────────────────────
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load B_proj[b, h, d] and C_proj[b, h, d] — (BLOCK_D,) once, reuse
    B_vals = tl.load(
        B_proj_ptr + b * stride_B_b + h * stride_B_h + d_offsets * stride_B_d,
        mask=d_mask, other=0.0,
    )
    C_vals = tl.load(
        C_proj_ptr + b * stride_C_b + h * stride_C_h + d_offsets * stride_C_d,
        mask=d_mask, other=0.0,
    )

    # Pre-compute constants
    one_minus_trap = 1.0 - trap_val
    half_trap = trap_val * 0.5

    # ── Iterate over P dimension ────────────────────────────────────────
    for p_idx in range(P):
        # Load x[b, h, p] — scalar
        x_p = tl.load(x_ptr + b * stride_x_b + h * stride_x_h + p_idx * stride_x_p)

        # Load h[b, h, p, d] and bx_prev[b, h, p, d] — (BLOCK_D,)
        h_base = b * stride_h_b + h * stride_h_h + p_idx * stride_h_p
        h_vals = tl.load(
            h_ptr + h_base + d_offsets * stride_h_d,
            mask=d_mask, other=0.0,
        )
        bx_prev_vals = tl.load(
            bx_prev_ptr + h_base + d_offsets * stride_h_d,
            mask=d_mask, other=0.0,
        )

        # Bx_curr = x[p] * B[d] — (BLOCK_D,)
        bx_curr = x_p * B_vals

        # Trapezoidal blend: (1-trap)*curr + trap*0.5*(curr+prev)
        bx_blended = one_minus_trap * bx_curr + half_trap * (bx_curr + bx_prev_vals)

        # State update: h = decay * h + dt * bx_blended
        h_new = decay * h_vals + dt_val * bx_blended

        # Output contribution: y = C[d] * h[d] summed over d + D*x
        y_p = tl.sum(C_vals * h_new, axis=0) + d_skip_val * x_p

        # Write y[b, h, p]
        tl.store(y_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_p)

        # Write back h and bx_prev
        tl.store(
            h_ptr + h_base + d_offsets * stride_h_d,
            h_new, mask=d_mask,
        )
        tl.store(
            bx_prev_ptr + h_base + d_offsets * stride_h_d,
            bx_curr, mask=d_mask,
        )


def mamba3_siso_decode_triton(
    x: torch.Tensor,       # (B, H, P)
    B_proj: torch.Tensor,  # (B, H, D)
    C_proj: torch.Tensor,  # (B, H, D)
    ADT: torch.Tensor,     # (B, H)
    DT: torch.Tensor,      # (B, H)
    trap: torch.Tensor,    # (B, H)
    D_skip: torch.Tensor,  # (H,)
    h: torch.Tensor,       # (B, H, P, D)
    bx_prev: torch.Tensor, # (B, H, P, D)
) -> tuple:
    """Launch the SISO fused decode kernel.

    Returns:
        y:         (B, H, P)
        h:         (B, H, P, D) — updated in-place
        bx_prev:   (B, H, P, D) — updated in-place
    """
    B_dim, H_dim, P_dim = x.shape
    D_dim = B_proj.shape[-1]

    assert h.shape == (B_dim, H_dim, P_dim, D_dim)
    assert bx_prev.shape == (B_dim, H_dim, P_dim, D_dim)
    assert x.is_cuda and h.is_cuda

    y = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D_dim)

    # Choose num_warps based on problem size
    num_warps = 4 if D_dim <= 64 else 8

    # Grid: one program per (batch, head)
    grid = (B_dim * H_dim,)

    _siso_decode_kernel[grid](
        x, h, bx_prev, y,
        B_proj, C_proj, ADT, DT, trap, D_skip,
        B_dim, H_dim, P_dim, D_dim,
        x.stride(0), x.stride(1), x.stride(2),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        bx_prev.stride(0), bx_prev.stride(1), bx_prev.stride(2), bx_prev.stride(3),
        y.stride(0), y.stride(1), y.stride(2),
        B_proj.stride(0), B_proj.stride(1), B_proj.stride(2),
        C_proj.stride(0), C_proj.stride(1), C_proj.stride(2),
        ADT.stride(0), ADT.stride(1),
        D_skip.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return y, h, bx_prev
