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

IMPORTANT: Triton does not support dynamic array indexing inside @triton.jit
(e.g., x_r[r_idx]). We provide separate kernel implementations for each R value.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# R=1 kernel
# ============================================================================

@triton.jit
def _mimo_decode_kernel_R1(
    x_ptr, h_ptr, bx_prev_ptr, y_ptr,
    B_proj_ptr, C_proj_ptr, ADT_ptr, DT_ptr, trap_ptr, D_skip_ptr,
    mimo_x_ptr, mimo_o_ptr,
    B: tl.constexpr, H: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_p,
    stride_h_b, stride_h_h, stride_h_d,
    stride_bx_b, stride_bx_h, stride_bx_d,
    stride_y_b, stride_y_h, stride_y_p,
    stride_B_b, stride_B_r, stride_B_h, stride_B_d,
    stride_C_b, stride_C_r, stride_C_h, stride_C_d,
    stride_adt_b, stride_adt_h,
    stride_D_h,
    stride_mx_h, stride_mx_r, stride_mx_p,
    stride_mo_h, stride_mo_r, stride_mo_p,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    adt_val = tl.load(ADT_ptr + b * stride_adt_b + h * stride_adt_h)
    dt_val = tl.load(DT_ptr + b * stride_adt_b + h * stride_adt_h)
    trap_val = tl.load(trap_ptr + b * stride_adt_b + h * stride_adt_h)
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)
    decay = tl.exp(adt_val)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    h_base = b * stride_h_b + h * stride_h_h
    h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
    bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_bx_d, mask=d_mask, other=0.0)

    # Step 1: Project x to R=1 scalar
    x_r_0 = 0.0
    for p_idx in range(P):
        x_p = tl.load(x_ptr + b * stride_x_b + h * stride_x_h + p_idx * stride_x_p)
        mx_0 = tl.load(mimo_x_ptr + h * stride_mx_h + 0 * stride_mx_r + p_idx * stride_mx_p)
        x_r_0 += x_p * mx_0

    # Step 2: Accumulate R=1 rank-1 Bx
    B_0 = tl.load(B_proj_ptr + b * stride_B_b + 0 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    Bx_curr = x_r_0 * B_0

    # Step 3: Trapezoidal blend
    Bx_blended = (1.0 - trap_val) * Bx_curr + trap_val * 0.5 * (Bx_curr + bx_prev_vals)

    # Step 4: State update
    h_new = decay * h_vals + dt_val * Bx_blended

    # Step 5: Per-rank output + skip
    C_0 = tl.load(C_proj_ptr + b * stride_C_b + 0 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    y_r_0 = tl.sum(C_0 * h_new, axis=0) + d_skip_val * x_r_0

    # Up-project
    for p_idx in range(P):
        mo_0 = tl.load(mimo_o_ptr + h * stride_mo_h + 0 * stride_mo_r + p_idx * stride_mo_p)
        y_p = y_r_0 * mo_0
        tl.store(y_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_p)

    tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
    tl.store(bx_prev_ptr + h_base + d_offsets * stride_bx_d, Bx_curr, mask=d_mask)


# ============================================================================
# R=2 kernel
# ============================================================================

@triton.jit
def _mimo_decode_kernel_R2(
    x_ptr, h_ptr, bx_prev_ptr, y_ptr,
    B_proj_ptr, C_proj_ptr, ADT_ptr, DT_ptr, trap_ptr, D_skip_ptr,
    mimo_x_ptr, mimo_o_ptr,
    B: tl.constexpr, H: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_p,
    stride_h_b, stride_h_h, stride_h_d,
    stride_bx_b, stride_bx_h, stride_bx_d,
    stride_y_b, stride_y_h, stride_y_p,
    stride_B_b, stride_B_r, stride_B_h, stride_B_d,
    stride_C_b, stride_C_r, stride_C_h, stride_C_d,
    stride_adt_b, stride_adt_h,
    stride_D_h,
    stride_mx_h, stride_mx_r, stride_mx_p,
    stride_mo_h, stride_mo_r, stride_mo_p,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    adt_val = tl.load(ADT_ptr + b * stride_adt_b + h * stride_adt_h)
    dt_val = tl.load(DT_ptr + b * stride_adt_b + h * stride_adt_h)
    trap_val = tl.load(trap_ptr + b * stride_adt_b + h * stride_adt_h)
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)
    decay = tl.exp(adt_val)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    h_base = b * stride_h_b + h * stride_h_h
    h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
    bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_bx_d, mask=d_mask, other=0.0)

    # Step 1: Project x to R=2 scalars
    x_r_0 = 0.0
    x_r_1 = 0.0
    for p_idx in range(P):
        x_p = tl.load(x_ptr + b * stride_x_b + h * stride_x_h + p_idx * stride_x_p)
        mx_0 = tl.load(mimo_x_ptr + h * stride_mx_h + 0 * stride_mx_r + p_idx * stride_mx_p)
        mx_1 = tl.load(mimo_x_ptr + h * stride_mx_h + 1 * stride_mx_r + p_idx * stride_mx_p)
        x_r_0 += x_p * mx_0
        x_r_1 += x_p * mx_1

    # Step 2: Accumulate R=2 rank-1 Bx
    B_0 = tl.load(B_proj_ptr + b * stride_B_b + 0 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_1 = tl.load(B_proj_ptr + b * stride_B_b + 1 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    Bx_curr = x_r_0 * B_0 + x_r_1 * B_1

    # Step 3: Trapezoidal blend
    Bx_blended = (1.0 - trap_val) * Bx_curr + trap_val * 0.5 * (Bx_curr + bx_prev_vals)

    # Step 4: State update
    h_new = decay * h_vals + dt_val * Bx_blended

    # Step 5: Per-rank output + skip
    C_0 = tl.load(C_proj_ptr + b * stride_C_b + 0 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_1 = tl.load(C_proj_ptr + b * stride_C_b + 1 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    y_r_0 = tl.sum(C_0 * h_new, axis=0) + d_skip_val * x_r_0
    y_r_1 = tl.sum(C_1 * h_new, axis=0) + d_skip_val * x_r_1

    # Up-project
    for p_idx in range(P):
        mo_0 = tl.load(mimo_o_ptr + h * stride_mo_h + 0 * stride_mo_r + p_idx * stride_mo_p)
        mo_1 = tl.load(mimo_o_ptr + h * stride_mo_h + 1 * stride_mo_r + p_idx * stride_mo_p)
        y_p = y_r_0 * mo_0 + y_r_1 * mo_1
        tl.store(y_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_p)

    tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
    tl.store(bx_prev_ptr + h_base + d_offsets * stride_bx_d, Bx_curr, mask=d_mask)


# ============================================================================
# R=4 kernel (most common default)
# ============================================================================

@triton.jit
def _mimo_decode_kernel_R4(
    x_ptr, h_ptr, bx_prev_ptr, y_ptr,
    B_proj_ptr, C_proj_ptr, ADT_ptr, DT_ptr, trap_ptr, D_skip_ptr,
    mimo_x_ptr, mimo_o_ptr,
    B: tl.constexpr, H: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_p,
    stride_h_b, stride_h_h, stride_h_d,
    stride_bx_b, stride_bx_h, stride_bx_d,
    stride_y_b, stride_y_h, stride_y_p,
    stride_B_b, stride_B_r, stride_B_h, stride_B_d,
    stride_C_b, stride_C_r, stride_C_h, stride_C_d,
    stride_adt_b, stride_adt_h,
    stride_D_h,
    stride_mx_h, stride_mx_r, stride_mx_p,
    stride_mo_h, stride_mo_r, stride_mo_p,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    adt_val = tl.load(ADT_ptr + b * stride_adt_b + h * stride_adt_h)
    dt_val = tl.load(DT_ptr + b * stride_adt_b + h * stride_adt_h)
    trap_val = tl.load(trap_ptr + b * stride_adt_b + h * stride_adt_h)
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)
    decay = tl.exp(adt_val)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    h_base = b * stride_h_b + h * stride_h_h
    h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
    bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_bx_d, mask=d_mask, other=0.0)

    # Step 1: Project x to R=4 scalars
    x_r_0 = 0.0
    x_r_1 = 0.0
    x_r_2 = 0.0
    x_r_3 = 0.0
    for p_idx in range(P):
        x_p = tl.load(x_ptr + b * stride_x_b + h * stride_x_h + p_idx * stride_x_p)
        mx_0 = tl.load(mimo_x_ptr + h * stride_mx_h + 0 * stride_mx_r + p_idx * stride_mx_p)
        mx_1 = tl.load(mimo_x_ptr + h * stride_mx_h + 1 * stride_mx_r + p_idx * stride_mx_p)
        mx_2 = tl.load(mimo_x_ptr + h * stride_mx_h + 2 * stride_mx_r + p_idx * stride_mx_p)
        mx_3 = tl.load(mimo_x_ptr + h * stride_mx_h + 3 * stride_mx_r + p_idx * stride_mx_p)
        x_r_0 += x_p * mx_0
        x_r_1 += x_p * mx_1
        x_r_2 += x_p * mx_2
        x_r_3 += x_p * mx_3

    # Step 2: Accumulate R=4 rank-1 Bx
    B_0 = tl.load(B_proj_ptr + b * stride_B_b + 0 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_1 = tl.load(B_proj_ptr + b * stride_B_b + 1 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_2 = tl.load(B_proj_ptr + b * stride_B_b + 2 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_3 = tl.load(B_proj_ptr + b * stride_B_b + 3 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    Bx_curr = x_r_0 * B_0 + x_r_1 * B_1 + x_r_2 * B_2 + x_r_3 * B_3

    # Step 3: Trapezoidal blend
    Bx_blended = (1.0 - trap_val) * Bx_curr + trap_val * 0.5 * (Bx_curr + bx_prev_vals)

    # Step 4: State update
    h_new = decay * h_vals + dt_val * Bx_blended

    # Step 5: Per-rank output + skip
    C_0 = tl.load(C_proj_ptr + b * stride_C_b + 0 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_1 = tl.load(C_proj_ptr + b * stride_C_b + 1 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_2 = tl.load(C_proj_ptr + b * stride_C_b + 2 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_3 = tl.load(C_proj_ptr + b * stride_C_b + 3 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    y_r_0 = tl.sum(C_0 * h_new, axis=0) + d_skip_val * x_r_0
    y_r_1 = tl.sum(C_1 * h_new, axis=0) + d_skip_val * x_r_1
    y_r_2 = tl.sum(C_2 * h_new, axis=0) + d_skip_val * x_r_2
    y_r_3 = tl.sum(C_3 * h_new, axis=0) + d_skip_val * x_r_3

    # Up-project
    for p_idx in range(P):
        mo_0 = tl.load(mimo_o_ptr + h * stride_mo_h + 0 * stride_mo_r + p_idx * stride_mo_p)
        mo_1 = tl.load(mimo_o_ptr + h * stride_mo_h + 1 * stride_mo_r + p_idx * stride_mo_p)
        mo_2 = tl.load(mimo_o_ptr + h * stride_mo_h + 2 * stride_mo_r + p_idx * stride_mo_p)
        mo_3 = tl.load(mimo_o_ptr + h * stride_mo_h + 3 * stride_mo_r + p_idx * stride_mo_p)
        y_p = y_r_0 * mo_0 + y_r_1 * mo_1 + y_r_2 * mo_2 + y_r_3 * mo_3
        tl.store(y_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_p)

    tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
    tl.store(bx_prev_ptr + h_base + d_offsets * stride_bx_d, Bx_curr, mask=d_mask)


# ============================================================================
# R=8 kernel
# ============================================================================

@triton.jit
def _mimo_decode_kernel_R8(
    x_ptr, h_ptr, bx_prev_ptr, y_ptr,
    B_proj_ptr, C_proj_ptr, ADT_ptr, DT_ptr, trap_ptr, D_skip_ptr,
    mimo_x_ptr, mimo_o_ptr,
    B: tl.constexpr, H: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
    stride_x_b, stride_x_h, stride_x_p,
    stride_h_b, stride_h_h, stride_h_d,
    stride_bx_b, stride_bx_h, stride_bx_d,
    stride_y_b, stride_y_h, stride_y_p,
    stride_B_b, stride_B_r, stride_B_h, stride_B_d,
    stride_C_b, stride_C_r, stride_C_h, stride_C_d,
    stride_adt_b, stride_adt_h,
    stride_D_h,
    stride_mx_h, stride_mx_r, stride_mx_p,
    stride_mo_h, stride_mo_r, stride_mo_p,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    adt_val = tl.load(ADT_ptr + b * stride_adt_b + h * stride_adt_h)
    dt_val = tl.load(DT_ptr + b * stride_adt_b + h * stride_adt_h)
    trap_val = tl.load(trap_ptr + b * stride_adt_b + h * stride_adt_h)
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)
    decay = tl.exp(adt_val)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    h_base = b * stride_h_b + h * stride_h_h
    h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
    bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_bx_d, mask=d_mask, other=0.0)

    # Step 1: Project x to R=8 scalars
    x_r_0 = 0.0
    x_r_1 = 0.0
    x_r_2 = 0.0
    x_r_3 = 0.0
    x_r_4 = 0.0
    x_r_5 = 0.0
    x_r_6 = 0.0
    x_r_7 = 0.0
    for p_idx in range(P):
        x_p = tl.load(x_ptr + b * stride_x_b + h * stride_x_h + p_idx * stride_x_p)
        mx_0 = tl.load(mimo_x_ptr + h * stride_mx_h + 0 * stride_mx_r + p_idx * stride_mx_p)
        mx_1 = tl.load(mimo_x_ptr + h * stride_mx_h + 1 * stride_mx_r + p_idx * stride_mx_p)
        mx_2 = tl.load(mimo_x_ptr + h * stride_mx_h + 2 * stride_mx_r + p_idx * stride_mx_p)
        mx_3 = tl.load(mimo_x_ptr + h * stride_mx_h + 3 * stride_mx_r + p_idx * stride_mx_p)
        mx_4 = tl.load(mimo_x_ptr + h * stride_mx_h + 4 * stride_mx_r + p_idx * stride_mx_p)
        mx_5 = tl.load(mimo_x_ptr + h * stride_mx_h + 5 * stride_mx_r + p_idx * stride_mx_p)
        mx_6 = tl.load(mimo_x_ptr + h * stride_mx_h + 6 * stride_mx_r + p_idx * stride_mx_p)
        mx_7 = tl.load(mimo_x_ptr + h * stride_mx_h + 7 * stride_mx_r + p_idx * stride_mx_p)
        x_r_0 += x_p * mx_0
        x_r_1 += x_p * mx_1
        x_r_2 += x_p * mx_2
        x_r_3 += x_p * mx_3
        x_r_4 += x_p * mx_4
        x_r_5 += x_p * mx_5
        x_r_6 += x_p * mx_6
        x_r_7 += x_p * mx_7

    # Step 2: Accumulate R=8 rank-1 Bx
    B_0 = tl.load(B_proj_ptr + b * stride_B_b + 0 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_1 = tl.load(B_proj_ptr + b * stride_B_b + 1 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_2 = tl.load(B_proj_ptr + b * stride_B_b + 2 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_3 = tl.load(B_proj_ptr + b * stride_B_b + 3 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_4 = tl.load(B_proj_ptr + b * stride_B_b + 4 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_5 = tl.load(B_proj_ptr + b * stride_B_b + 5 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_6 = tl.load(B_proj_ptr + b * stride_B_b + 6 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    B_7 = tl.load(B_proj_ptr + b * stride_B_b + 7 * stride_B_r + h * stride_B_h + d_offsets * stride_B_d, mask=d_mask, other=0.0)
    Bx_curr = x_r_0 * B_0 + x_r_1 * B_1 + x_r_2 * B_2 + x_r_3 * B_3 + x_r_4 * B_4 + x_r_5 * B_5 + x_r_6 * B_6 + x_r_7 * B_7

    # Step 3: Trapezoidal blend
    Bx_blended = (1.0 - trap_val) * Bx_curr + trap_val * 0.5 * (Bx_curr + bx_prev_vals)

    # Step 4: State update
    h_new = decay * h_vals + dt_val * Bx_blended

    # Step 5: Per-rank output + skip
    C_0 = tl.load(C_proj_ptr + b * stride_C_b + 0 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_1 = tl.load(C_proj_ptr + b * stride_C_b + 1 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_2 = tl.load(C_proj_ptr + b * stride_C_b + 2 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_3 = tl.load(C_proj_ptr + b * stride_C_b + 3 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_4 = tl.load(C_proj_ptr + b * stride_C_b + 4 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_5 = tl.load(C_proj_ptr + b * stride_C_b + 5 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_6 = tl.load(C_proj_ptr + b * stride_C_b + 6 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    C_7 = tl.load(C_proj_ptr + b * stride_C_b + 7 * stride_C_r + h * stride_C_h + d_offsets * stride_C_d, mask=d_mask, other=0.0)
    y_r_0 = tl.sum(C_0 * h_new, axis=0) + d_skip_val * x_r_0
    y_r_1 = tl.sum(C_1 * h_new, axis=0) + d_skip_val * x_r_1
    y_r_2 = tl.sum(C_2 * h_new, axis=0) + d_skip_val * x_r_2
    y_r_3 = tl.sum(C_3 * h_new, axis=0) + d_skip_val * x_r_3
    y_r_4 = tl.sum(C_4 * h_new, axis=0) + d_skip_val * x_r_4
    y_r_5 = tl.sum(C_5 * h_new, axis=0) + d_skip_val * x_r_5
    y_r_6 = tl.sum(C_6 * h_new, axis=0) + d_skip_val * x_r_6
    y_r_7 = tl.sum(C_7 * h_new, axis=0) + d_skip_val * x_r_7

    # Up-project
    for p_idx in range(P):
        mo_0 = tl.load(mimo_o_ptr + h * stride_mo_h + 0 * stride_mo_r + p_idx * stride_mo_p)
        mo_1 = tl.load(mimo_o_ptr + h * stride_mo_h + 1 * stride_mo_r + p_idx * stride_mo_p)
        mo_2 = tl.load(mimo_o_ptr + h * stride_mo_h + 2 * stride_mo_r + p_idx * stride_mo_p)
        mo_3 = tl.load(mimo_o_ptr + h * stride_mo_h + 3 * stride_mo_r + p_idx * stride_mo_p)
        mo_4 = tl.load(mimo_o_ptr + h * stride_mo_h + 4 * stride_mo_r + p_idx * stride_mo_p)
        mo_5 = tl.load(mimo_o_ptr + h * stride_mo_h + 5 * stride_mo_r + p_idx * stride_mo_p)
        mo_6 = tl.load(mimo_o_ptr + h * stride_mo_h + 6 * stride_mo_r + p_idx * stride_mo_p)
        mo_7 = tl.load(mimo_o_ptr + h * stride_mo_h + 7 * stride_mo_r + p_idx * stride_mo_p)
        y_p = y_r_0 * mo_0 + y_r_1 * mo_1 + y_r_2 * mo_2 + y_r_3 * mo_3 + y_r_4 * mo_4 + y_r_5 * mo_5 + y_r_6 * mo_6 + y_r_7 * mo_7
        tl.store(y_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_p)

    tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
    tl.store(bx_prev_ptr + h_base + d_offsets * stride_bx_d, Bx_curr, mask=d_mask)


# ============================================================================
# Kernel dispatch
# ============================================================================

_KERNEL_MAP = {
    1: _mimo_decode_kernel_R1,
    2: _mimo_decode_kernel_R2,
    4: _mimo_decode_kernel_R4,
    8: _mimo_decode_kernel_R8,
}


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

    if R_dim not in _KERNEL_MAP:
        raise ValueError(
            f"Unsupported mimo_rank={R_dim}. "
            f"Supported values: {list(_KERNEL_MAP.keys())}"
        )

    kernel = _KERNEL_MAP[R_dim]

    y = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D_dim)

    grid = (B_dim * H_dim,)

    kernel[grid](
        x, h, bx_prev, y,
        B_proj, C_proj, ADT, DT, trap, D_skip, mimo_x, mimo_o,
        B_dim, H_dim, P_dim, D_dim,
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
    )

    return y, h, bx_prev
