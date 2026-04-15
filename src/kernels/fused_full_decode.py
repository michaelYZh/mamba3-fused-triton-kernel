"""
Full-step fused Triton decode kernels for Mamba-3.

Fuses the ENTIRE decode step into a single kernel launch:
  split → rearrange → softplus(A) → softplus(dt) → sigmoid(trap) →
  RMSNorm(B,C) → expand+bias → RoPE → SSM recurrence → silu gate

Input:  zxBCdtAtrap (B, d_in_proj) — raw output from in_proj
Output: y_gated (B, H, P) — gated output, ready for out_proj

This eliminates ALL intermediate kernel launches between in_proj and out_proj,
achieving maximal fusion benefit. The only separate operations remaining are
the in_proj linear and out_proj linear, which are memory-bound GEMMs that
cannot be fused without custom GEMM kernels.
"""

import math
import torch
import triton
import triton.language as tl


# ────────────────────────────────────────────────────────────────────────────
# Triton helpers
# ────────────────────────────────────────────────────────────────────────────

@triton.jit
def _softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


# ────────────────────────────────────────────────────────────────────────────
# SISO full-fusion kernel
# ────────────────────────────────────────────────────────────────────────────

@triton.jit
def _siso_full_fused_kernel(
    # ── Primary I/O ─────────────────────────────────────────────────────────
    zxBCdtAtrap_ptr,    # (B, d_in_proj) — raw in_proj output
    y_gated_ptr,        # (B, H, P) — gated output

    # ── State (in-out) ──────────────────────────────────────────────────────
    h_ptr,              # (B, H, P, D) — SSM state
    bx_prev_ptr,        # (B, H, P, D) — previous B*x
    angle_state_ptr,    # (B, H, num_rope_angles) — RoPE angle accumulator

    # ── Model parameters (read-only) ────────────────────────────────────────
    dt_bias_ptr,        # (H,)
    D_skip_ptr,         # (H,)
    B_norm_weight_ptr,  # (D,) — RMSNorm weight for B
    C_norm_weight_ptr,  # (D,) — RMSNorm weight for C
    B_bias_ptr,         # (H, D) — B bias (after rearrange from H,R=1,D)
    C_bias_ptr,         # (H, D) — C bias (after rearrange from H,R=1,D)

    # ── Dimensions (constexpr) ──────────────────────────────────────────────
    B_batch: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    d_in_proj: tl.constexpr,
    d_inner: tl.constexpr,
    d_BC: tl.constexpr,
    num_rope_angles: tl.constexpr,
    split_tensor_size: tl.constexpr,
    A_floor: tl.constexpr,

    # ── Strides ─────────────────────────────────────────────────────────────
    stride_zx_b, stride_zx_d,
    stride_y_b, stride_y_h, stride_y_p,
    stride_h_b, stride_h_h, stride_h_p, stride_h_d,
    stride_bx_b, stride_bx_h, stride_bx_p, stride_bx_d,
    stride_as_b, stride_as_h, stride_as_a,
    stride_dt_h,
    stride_D_h,
    stride_bnw_d,
    stride_cnw_d,
    stride_bb_h, stride_bb_d,
    stride_cb_h, stride_cb_d,

    BLOCK_D: tl.constexpr,
):
    """SISO full-step fused kernel: one program per (batch, head) pair."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    zx_base = b * stride_zx_b

    # ── Load scalar params from split positions ────────────────────────────
    offset_dd_dt = 2 * d_inner + 2 * d_BC + h
    offset_dd_A = 2 * d_inner + 2 * d_BC + H + h
    offset_trap_raw = 2 * d_inner + 2 * d_BC + 2 * H + h

    dd_dt_val = tl.load(zxBCdtAtrap_ptr + zx_base + offset_dd_dt * stride_zx_d)
    dd_A_val = tl.load(zxBCdtAtrap_ptr + zx_base + offset_dd_A * stride_zx_d)
    trap_raw_val = tl.load(zxBCdtAtrap_ptr + zx_base + offset_trap_raw * stride_zx_d)

    # ── Compute A, DT, ADT, trap ──────────────────────────────────────────
    # Match reference: A = -(softplus(dd_A).clamp(max=-A_floor))
    # Note: softplus is always positive, so clamp(max=-A_floor) always triggers,
    # resulting in A = -(-A_floor) = A_floor (a small positive value).
    # This is technically a bug in the reference code (operator precedence),
    # but we match it for numerical consistency.
    sp_A = _softplus(dd_A_val)
    A_val = -tl.where(sp_A > -A_floor, -A_floor, sp_A)
    dt_bias_val = tl.load(dt_bias_ptr + h * stride_dt_h)
    DT_val = _softplus(dd_dt_val + dt_bias_val)
    ADT_val = A_val * DT_val
    trap_val = 1.0 / (1.0 + tl.exp(-trap_raw_val))  # sigmoid
    decay = tl.exp(ADT_val)

    # ── Update angle_state ─────────────────────────────────────────────────
    # Load angle_raw from the split input and update angle_state in-place
    as_base = b * stride_as_b + h * stride_as_h
    angle_raw_base = 2 * d_inner + 2 * d_BC + 3 * H

    # Process each angle individually (num_rope_angles is tl.constexpr, so loop is unrolled)
    for a_idx in range(32):  # max 32 rope angles
        if a_idx >= num_rope_angles:
            pass  # no-op: Triton can't break/return from for loops
        else:
            angle_raw_val = tl.load(zxBCdtAtrap_ptr + zx_base + (angle_raw_base + a_idx) * stride_zx_d)
            angle_old = tl.load(angle_state_ptr + as_base + a_idx * stride_as_a)
            angle_new = angle_old + angle_raw_val * DT_val
            tl.store(angle_state_ptr + as_base + a_idx * stride_as_a, angle_new)

    # ── Load B_raw and C_raw, apply RMSNorm + bias ────────────────────────
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    B_raw_base = 2 * d_inner
    C_raw_base = 2 * d_inner + d_BC

    B_raw_vals = tl.load(
        zxBCdtAtrap_ptr + zx_base + (B_raw_base + d_offsets) * stride_zx_d,
        mask=d_mask, other=0.0,
    )
    C_raw_vals = tl.load(
        zxBCdtAtrap_ptr + zx_base + (C_raw_base + d_offsets) * stride_zx_d,
        mask=d_mask, other=0.0,
    )

    # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    B_rms = tl.sqrt(tl.sum(B_raw_vals * B_raw_vals, axis=0) / D + 1e-5)
    C_rms = tl.sqrt(tl.sum(C_raw_vals * C_raw_vals, axis=0) / D + 1e-5)

    B_norm_weight = tl.load(B_norm_weight_ptr + d_offsets * stride_bnw_d, mask=d_mask, other=1.0)
    C_norm_weight = tl.load(C_norm_weight_ptr + d_offsets * stride_cnw_d, mask=d_mask, other=1.0)

    B_exp = B_raw_vals / B_rms * B_norm_weight
    C_exp = C_raw_vals / C_rms * C_norm_weight

    # Add bias
    B_bias_vals = tl.load(B_bias_ptr + h * stride_bb_h + d_offsets * stride_bb_d, mask=d_mask, other=0.0)
    C_bias_vals = tl.load(C_bias_ptr + h * stride_cb_h + d_offsets * stride_cb_d, mask=d_mask, other=0.0)

    B_exp = B_exp + B_bias_vals
    C_exp = C_exp + C_bias_vals

    # ── Apply RoPE to B and C ─────────────────────────────────────────────
    # RoPE rotates pairs (2i, 2i+1) by angle_state[i]
    # We iterate over pairs, loading each angle individually from angle_state
    for a_idx in range(32):  # max 32 rope angles
        if a_idx >= num_rope_angles:
            pass
        else:
            d0 = 2 * a_idx
            d1 = 2 * a_idx + 1

            # Load the angle from the updated angle_state
            angle_val = tl.load(angle_state_ptr + as_base + a_idx * stride_as_a)
            cos_a = tl.cos(angle_val)
            sin_a = tl.sin(angle_val)

            # Extract pair values using comparison masks
            mask_d0 = d_offsets == d0
            mask_d1 = d_offsets == d1

            b_d0 = tl.sum(tl.where(mask_d0, B_exp, 0.0), axis=0)
            b_d1 = tl.sum(tl.where(mask_d1, B_exp, 0.0), axis=0)
            c_d0 = tl.sum(tl.where(mask_d0, C_exp, 0.0), axis=0)
            c_d1 = tl.sum(tl.where(mask_d1, C_exp, 0.0), axis=0)

            # Apply rotation
            B_exp = tl.where(mask_d0, b_d0 * cos_a - b_d1 * sin_a, B_exp)
            B_exp = tl.where(mask_d1, b_d0 * sin_a + b_d1 * cos_a, B_exp)
            C_exp = tl.where(mask_d0, c_d0 * cos_a - c_d1 * sin_a, C_exp)
            C_exp = tl.where(mask_d1, c_d0 * sin_a + c_d1 * cos_a, C_exp)

    # ── SSM recurrence with silu gate ─────────────────────────────────────
    one_minus_trap = 1.0 - trap_val
    half_trap = trap_val * 0.5
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)

    for p_idx in range(P):
        # Load z[b, h, p] and x[b, h, p] from the split input
        z_offset = h * P + p_idx
        x_offset = d_inner + h * P + p_idx
        z_p = tl.load(zxBCdtAtrap_ptr + zx_base + z_offset * stride_zx_d)
        x_p = tl.load(zxBCdtAtrap_ptr + zx_base + x_offset * stride_zx_d)

        # Load state
        h_base = b * stride_h_b + h * stride_h_h + p_idx * stride_h_p
        h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
        bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)

        # SSM recurrence
        bx_curr = x_p * B_exp
        bx_blended = one_minus_trap * bx_curr + half_trap * (bx_curr + bx_prev_vals)
        h_new = decay * h_vals + DT_val * bx_blended

        # Output: y = C*h + D*x
        y_p = tl.sum(C_exp * h_new, axis=0) + d_skip_val * x_p

        # Silu gate: silu(z) = z * sigmoid(z)
        sigmoid_z = 1.0 / (1.0 + tl.exp(-z_p))
        y_gated = y_p * z_p * sigmoid_z

        # Write output
        tl.store(y_gated_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_gated)

        # Write back state
        tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
        tl.store(bx_prev_ptr + h_base + d_offsets * stride_h_d, bx_curr, mask=d_mask)


def mamba3_siso_full_fused_triton(
    zxBCdtAtrap: torch.Tensor,
    dt_bias: torch.Tensor,
    D_skip: torch.Tensor,
    B_norm_weight: torch.Tensor,
    C_norm_weight: torch.Tensor,
    B_bias: torch.Tensor,
    C_bias: torch.Tensor,
    h: torch.Tensor,
    bx_prev: torch.Tensor,
    angle_state: torch.Tensor,
    d_inner: int,
    d_state: int,
    ngroups: int,
    mimo_rank: int,
    nheads: int,
    headdim: int,
    split_tensor_size: int,
    num_rope_angles: int,
    A_floor: float,
) -> tuple:
    """Launch the SISO full-step fused decode kernel."""
    B_batch = zxBCdtAtrap.shape[0]
    d_in_proj = zxBCdtAtrap.shape[-1]
    H = nheads
    P = headdim
    D = d_state
    d_BC = d_state * ngroups * mimo_rank

    assert h.shape == (B_batch, H, P, D)
    assert bx_prev.shape == (B_batch, H, P, D)
    assert angle_state.shape == (B_batch, H, num_rope_angles)
    assert zxBCdtAtrap.is_cuda and h.is_cuda

    y_gated = torch.empty(B_batch, H, P, device=zxBCdtAtrap.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if D <= 64 else 8
    grid = (B_batch * H,)

    _siso_full_fused_kernel[grid](
        zxBCdtAtrap, y_gated,
        h, bx_prev, angle_state,
        dt_bias, D_skip, B_norm_weight, C_norm_weight, B_bias, C_bias,
        B_batch, H, P, D, d_in_proj, d_inner, d_BC,
        num_rope_angles, split_tensor_size, A_floor,
        zxBCdtAtrap.stride(0), zxBCdtAtrap.stride(1),
        y_gated.stride(0), y_gated.stride(1), y_gated.stride(2),
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        bx_prev.stride(0), bx_prev.stride(1), bx_prev.stride(2), bx_prev.stride(3),
        angle_state.stride(0), angle_state.stride(1), angle_state.stride(2),
        dt_bias.stride(0),
        D_skip.stride(0),
        B_norm_weight.stride(0),
        C_norm_weight.stride(0),
        B_bias.stride(0), B_bias.stride(1),
        C_bias.stride(0), C_bias.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return y_gated, h, bx_prev, angle_state


# ────────────────────────────────────────────────────────────────────────────
# MIMO full-fusion kernel (R=4)
# ────────────────────────────────────────────────────────────────────────────

@triton.jit
def _mimo_full_fused_kernel_R4(
    # ── Primary I/O ─────────────────────────────────────────────────────────
    zxBCdtAtrap_ptr,    # (B, d_in_proj) — raw in_proj output
    y_gated_ptr,        # (B, H, P) — gated output

    # ── State (in-out) ──────────────────────────────────────────────────────
    h_ptr,              # (B, H, D) — MIMO SSM state
    bx_prev_ptr,        # (B, H, D) — previous B*x
    angle_state_ptr,    # (B, H, num_rope_angles) — RoPE angle accumulator

    # ── Model parameters (read-only) ────────────────────────────────────────
    dt_bias_ptr,        # (H,)
    D_skip_ptr,         # (H,)
    B_norm_weight_ptr,  # (D,) — RMSNorm weight for B
    C_norm_weight_ptr,  # (D,) — RMSNorm weight for C
    B_bias_ptr,         # (H, R, D) — B bias
    C_bias_ptr,         # (H, R, D) — C bias
    mimo_x_ptr,         # (H, R, P) — MIMO x projection
    mimo_o_ptr,         # (H, R, P) — MIMO output projection

    # ── Dimensions (constexpr) ──────────────────────────────────────────────
    B_batch: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    d_in_proj: tl.constexpr,
    d_inner: tl.constexpr,
    d_BC: tl.constexpr,
    num_rope_angles: tl.constexpr,
    split_tensor_size: tl.constexpr,
    A_floor: tl.constexpr,

    # ── Strides ─────────────────────────────────────────────────────────────
    stride_zx_b, stride_zx_d,
    stride_y_b, stride_y_h, stride_y_p,
    stride_h_b, stride_h_h, stride_h_d,
    stride_bx_b, stride_bx_h, stride_bx_d,
    stride_as_b, stride_as_h, stride_as_a,
    stride_dt_h,
    stride_D_h,
    stride_bnw_d,
    stride_cnw_d,
    stride_bb_h, stride_bb_r, stride_bb_d,
    stride_cb_h, stride_cb_r, stride_cb_d,
    stride_mx_h, stride_mx_r, stride_mx_p,
    stride_mo_h, stride_mo_r, stride_mo_p,

    BLOCK_D: tl.constexpr,
):
    """MIMO full-step fused kernel (R=4): one program per (batch, head) pair."""
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    zx_base = b * stride_zx_b

    # ── Load scalar params ──────────────────────────────────────────────────
    offset_dd_dt = 2 * d_inner + 2 * d_BC + h
    offset_dd_A = 2 * d_inner + 2 * d_BC + H + h
    offset_trap_raw = 2 * d_inner + 2 * d_BC + 2 * H + h

    dd_dt_val = tl.load(zxBCdtAtrap_ptr + zx_base + offset_dd_dt * stride_zx_d)
    dd_A_val = tl.load(zxBCdtAtrap_ptr + zx_base + offset_dd_A * stride_zx_d)
    trap_raw_val = tl.load(zxBCdtAtrap_ptr + zx_base + offset_trap_raw * stride_zx_d)

    sp_A = _softplus(dd_A_val)
    A_val = -tl.where(sp_A > -A_floor, -A_floor, sp_A)
    dt_bias_val = tl.load(dt_bias_ptr + h * stride_dt_h)
    DT_val = _softplus(dd_dt_val + dt_bias_val)
    ADT_val = A_val * DT_val
    trap_val = 1.0 / (1.0 + tl.exp(-trap_raw_val))
    decay = tl.exp(ADT_val)

    # ── Update angle_state ──────────────────────────────────────────────────
    as_base = b * stride_as_b + h * stride_as_h
    angle_raw_base = 2 * d_inner + 2 * d_BC + 3 * H

    for a_idx in range(32):
        if a_idx >= num_rope_angles:
            pass
        else:
            angle_raw_val = tl.load(zxBCdtAtrap_ptr + zx_base + (angle_raw_base + a_idx) * stride_zx_d)
            angle_old = tl.load(angle_state_ptr + as_base + a_idx * stride_as_a)
            angle_new = angle_old + angle_raw_val * DT_val
            tl.store(angle_state_ptr + as_base + a_idx * stride_as_a, angle_new)

    # ── Load B_raw and C_raw for all 4 ranks, RMSNorm + bias + RoPE ────────
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    B_raw_base = 2 * d_inner
    C_raw_base = 2 * d_inner + d_BC

    # d_BC = 4 * D for MIMO R=4 with ngroups=1
    B_raw_0 = tl.load(zxBCdtAtrap_ptr + zx_base + (B_raw_base + 0 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)
    B_raw_1 = tl.load(zxBCdtAtrap_ptr + zx_base + (B_raw_base + 1 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)
    B_raw_2 = tl.load(zxBCdtAtrap_ptr + zx_base + (B_raw_base + 2 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)
    B_raw_3 = tl.load(zxBCdtAtrap_ptr + zx_base + (B_raw_base + 3 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)

    C_raw_0 = tl.load(zxBCdtAtrap_ptr + zx_base + (C_raw_base + 0 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)
    C_raw_1 = tl.load(zxBCdtAtrap_ptr + zx_base + (C_raw_base + 1 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)
    C_raw_2 = tl.load(zxBCdtAtrap_ptr + zx_base + (C_raw_base + 2 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)
    C_raw_3 = tl.load(zxBCdtAtrap_ptr + zx_base + (C_raw_base + 3 * D + d_offsets) * stride_zx_d, mask=d_mask, other=0.0)

    # RMSNorm for each rank
    B_norm_weight = tl.load(B_norm_weight_ptr + d_offsets * stride_bnw_d, mask=d_mask, other=1.0)
    C_norm_weight = tl.load(C_norm_weight_ptr + d_offsets * stride_cnw_d, mask=d_mask, other=1.0)

    B_rms_0 = tl.sqrt(tl.sum(B_raw_0 * B_raw_0, axis=0) / D + 1e-5)
    B_rms_1 = tl.sqrt(tl.sum(B_raw_1 * B_raw_1, axis=0) / D + 1e-5)
    B_rms_2 = tl.sqrt(tl.sum(B_raw_2 * B_raw_2, axis=0) / D + 1e-5)
    B_rms_3 = tl.sqrt(tl.sum(B_raw_3 * B_raw_3, axis=0) / D + 1e-5)

    C_rms_0 = tl.sqrt(tl.sum(C_raw_0 * C_raw_0, axis=0) / D + 1e-5)
    C_rms_1 = tl.sqrt(tl.sum(C_raw_1 * C_raw_1, axis=0) / D + 1e-5)
    C_rms_2 = tl.sqrt(tl.sum(C_raw_2 * C_raw_2, axis=0) / D + 1e-5)
    C_rms_3 = tl.sqrt(tl.sum(C_raw_3 * C_raw_3, axis=0) / D + 1e-5)

    B_normed_0 = B_raw_0 / B_rms_0 * B_norm_weight
    B_normed_1 = B_raw_1 / B_rms_1 * B_norm_weight
    B_normed_2 = B_raw_2 / B_rms_2 * B_norm_weight
    B_normed_3 = B_raw_3 / B_rms_3 * B_norm_weight

    C_normed_0 = C_raw_0 / C_rms_0 * C_norm_weight
    C_normed_1 = C_raw_1 / C_rms_1 * C_norm_weight
    C_normed_2 = C_raw_2 / C_rms_2 * C_norm_weight
    C_normed_3 = C_raw_3 / C_rms_3 * C_norm_weight

    # Add bias for each rank
    B_bias_0 = tl.load(B_bias_ptr + h * stride_bb_h + 0 * stride_bb_r + d_offsets * stride_bb_d, mask=d_mask, other=0.0)
    B_bias_1 = tl.load(B_bias_ptr + h * stride_bb_h + 1 * stride_bb_r + d_offsets * stride_bb_d, mask=d_mask, other=0.0)
    B_bias_2 = tl.load(B_bias_ptr + h * stride_bb_h + 2 * stride_bb_r + d_offsets * stride_bb_d, mask=d_mask, other=0.0)
    B_bias_3 = tl.load(B_bias_ptr + h * stride_bb_h + 3 * stride_bb_r + d_offsets * stride_bb_d, mask=d_mask, other=0.0)

    C_bias_0 = tl.load(C_bias_ptr + h * stride_cb_h + 0 * stride_cb_r + d_offsets * stride_cb_d, mask=d_mask, other=0.0)
    C_bias_1 = tl.load(C_bias_ptr + h * stride_cb_h + 1 * stride_cb_r + d_offsets * stride_cb_d, mask=d_mask, other=0.0)
    C_bias_2 = tl.load(C_bias_ptr + h * stride_cb_h + 2 * stride_cb_r + d_offsets * stride_cb_d, mask=d_mask, other=0.0)
    C_bias_3 = tl.load(C_bias_ptr + h * stride_cb_h + 3 * stride_cb_r + d_offsets * stride_cb_d, mask=d_mask, other=0.0)

    B_exp_0 = B_normed_0 + B_bias_0
    B_exp_1 = B_normed_1 + B_bias_1
    B_exp_2 = B_normed_2 + B_bias_2
    B_exp_3 = B_normed_3 + B_bias_3

    C_exp_0 = C_normed_0 + C_bias_0
    C_exp_1 = C_normed_1 + C_bias_1
    C_exp_2 = C_normed_2 + C_bias_2
    C_exp_3 = C_normed_3 + C_bias_3

    # ── Apply RoPE to each rank ─────────────────────────────────────────────
    for a_idx in range(32):
        if a_idx >= num_rope_angles:
            pass
        else:
            d0 = 2 * a_idx
            d1 = 2 * a_idx + 1
            angle_val = tl.load(angle_state_ptr + as_base + a_idx * stride_as_a)
            cos_a = tl.cos(angle_val)
            sin_a = tl.sin(angle_val)

            mask_d0 = d_offsets == d0
            mask_d1 = d_offsets == d1

            # Rotate B for all 4 ranks
            for B_vec_ref in range(4):
                pass  # Can't iterate over registers, must unroll manually

            # B_exp_0
            b0_d0 = tl.sum(tl.where(mask_d0, B_exp_0, 0.0), axis=0)
            b0_d1 = tl.sum(tl.where(mask_d1, B_exp_0, 0.0), axis=0)
            B_exp_0 = tl.where(mask_d0, b0_d0 * cos_a - b0_d1 * sin_a, B_exp_0)
            B_exp_0 = tl.where(mask_d1, b0_d0 * sin_a + b0_d1 * cos_a, B_exp_0)

            c0_d0 = tl.sum(tl.where(mask_d0, C_exp_0, 0.0), axis=0)
            c0_d1 = tl.sum(tl.where(mask_d1, C_exp_0, 0.0), axis=0)
            C_exp_0 = tl.where(mask_d0, c0_d0 * cos_a - c0_d1 * sin_a, C_exp_0)
            C_exp_0 = tl.where(mask_d1, c0_d0 * sin_a + c0_d1 * cos_a, C_exp_0)

            # B_exp_1
            b1_d0 = tl.sum(tl.where(mask_d0, B_exp_1, 0.0), axis=0)
            b1_d1 = tl.sum(tl.where(mask_d1, B_exp_1, 0.0), axis=0)
            B_exp_1 = tl.where(mask_d0, b1_d0 * cos_a - b1_d1 * sin_a, B_exp_1)
            B_exp_1 = tl.where(mask_d1, b1_d0 * sin_a + b1_d1 * cos_a, B_exp_1)

            c1_d0 = tl.sum(tl.where(mask_d0, C_exp_1, 0.0), axis=0)
            c1_d1 = tl.sum(tl.where(mask_d1, C_exp_1, 0.0), axis=0)
            C_exp_1 = tl.where(mask_d0, c1_d0 * cos_a - c1_d1 * sin_a, C_exp_1)
            C_exp_1 = tl.where(mask_d1, c1_d0 * sin_a + c1_d1 * cos_a, C_exp_1)

            # B_exp_2
            b2_d0 = tl.sum(tl.where(mask_d0, B_exp_2, 0.0), axis=0)
            b2_d1 = tl.sum(tl.where(mask_d1, B_exp_2, 0.0), axis=0)
            B_exp_2 = tl.where(mask_d0, b2_d0 * cos_a - b2_d1 * sin_a, B_exp_2)
            B_exp_2 = tl.where(mask_d1, b2_d0 * sin_a + b2_d1 * cos_a, B_exp_2)

            c2_d0 = tl.sum(tl.where(mask_d0, C_exp_2, 0.0), axis=0)
            c2_d1 = tl.sum(tl.where(mask_d1, C_exp_2, 0.0), axis=0)
            C_exp_2 = tl.where(mask_d0, c2_d0 * cos_a - c2_d1 * sin_a, C_exp_2)
            C_exp_2 = tl.where(mask_d1, c2_d0 * sin_a + c2_d1 * cos_a, C_exp_2)

            # B_exp_3
            b3_d0 = tl.sum(tl.where(mask_d0, B_exp_3, 0.0), axis=0)
            b3_d1 = tl.sum(tl.where(mask_d1, B_exp_3, 0.0), axis=0)
            B_exp_3 = tl.where(mask_d0, b3_d0 * cos_a - b3_d1 * sin_a, B_exp_3)
            B_exp_3 = tl.where(mask_d1, b3_d0 * sin_a + b3_d1 * cos_a, B_exp_3)

            c3_d0 = tl.sum(tl.where(mask_d0, C_exp_3, 0.0), axis=0)
            c3_d1 = tl.sum(tl.where(mask_d1, C_exp_3, 0.0), axis=0)
            C_exp_3 = tl.where(mask_d0, c3_d0 * cos_a - c3_d1 * sin_a, C_exp_3)
            C_exp_3 = tl.where(mask_d1, c3_d0 * sin_a + c3_d1 * cos_a, C_exp_3)

    # ── MIMO SSM recurrence with silu gate ─────────────────────────────────
    h_base = b * stride_h_b + h * stride_h_h
    h_vals = tl.load(h_ptr + h_base + d_offsets * stride_h_d, mask=d_mask, other=0.0)
    bx_prev_vals = tl.load(bx_prev_ptr + h_base + d_offsets * stride_bx_d, mask=d_mask, other=0.0)

    # Project x to R=4 scalars
    x_r_0 = 0.0
    x_r_1 = 0.0
    x_r_2 = 0.0
    x_r_3 = 0.0
    for p_idx in range(P):
        x_offset = d_inner + h * P + p_idx
        x_p = tl.load(zxBCdtAtrap_ptr + zx_base + x_offset * stride_zx_d)
        mx_0 = tl.load(mimo_x_ptr + h * stride_mx_h + 0 * stride_mx_r + p_idx * stride_mx_p)
        mx_1 = tl.load(mimo_x_ptr + h * stride_mx_h + 1 * stride_mx_r + p_idx * stride_mx_p)
        mx_2 = tl.load(mimo_x_ptr + h * stride_mx_h + 2 * stride_mx_r + p_idx * stride_mx_p)
        mx_3 = tl.load(mimo_x_ptr + h * stride_mx_h + 3 * stride_mx_r + p_idx * stride_mx_p)
        x_r_0 += x_p * mx_0
        x_r_1 += x_p * mx_1
        x_r_2 += x_p * mx_2
        x_r_3 += x_p * mx_3

    # Accumulate R=4 rank-1 Bx
    Bx_curr = x_r_0 * B_exp_0 + x_r_1 * B_exp_1 + x_r_2 * B_exp_2 + x_r_3 * B_exp_3

    # Trapezoidal blend
    one_minus_trap = 1.0 - trap_val
    half_trap = trap_val * 0.5
    Bx_blended = one_minus_trap * Bx_curr + half_trap * (Bx_curr + bx_prev_vals)

    # State update
    h_new = decay * h_vals + DT_val * Bx_blended

    # Per-rank output + skip
    d_skip_val = tl.load(D_skip_ptr + h * stride_D_h)
    y_r_0 = tl.sum(C_exp_0 * h_new, axis=0) + d_skip_val * x_r_0
    y_r_1 = tl.sum(C_exp_1 * h_new, axis=0) + d_skip_val * x_r_1
    y_r_2 = tl.sum(C_exp_2 * h_new, axis=0) + d_skip_val * x_r_2
    y_r_3 = tl.sum(C_exp_3 * h_new, axis=0) + d_skip_val * x_r_3

    # Up-project and apply silu gate
    for p_idx in range(P):
        mo_0 = tl.load(mimo_o_ptr + h * stride_mo_h + 0 * stride_mo_r + p_idx * stride_mo_p)
        mo_1 = tl.load(mimo_o_ptr + h * stride_mo_h + 1 * stride_mo_r + p_idx * stride_mo_p)
        mo_2 = tl.load(mimo_o_ptr + h * stride_mo_h + 2 * stride_mo_r + p_idx * stride_mo_p)
        mo_3 = tl.load(mimo_o_ptr + h * stride_mo_h + 3 * stride_mo_r + p_idx * stride_mo_p)
        y_p = y_r_0 * mo_0 + y_r_1 * mo_1 + y_r_2 * mo_2 + y_r_3 * mo_3

        # Load z and apply silu gate
        z_offset = h * P + p_idx
        z_p = tl.load(zxBCdtAtrap_ptr + zx_base + z_offset * stride_zx_d)
        sigmoid_z = 1.0 / (1.0 + tl.exp(-z_p))
        y_gated = y_p * z_p * sigmoid_z

        tl.store(y_gated_ptr + b * stride_y_b + h * stride_y_h + p_idx * stride_y_p, y_gated)

    # Write back state
    tl.store(h_ptr + h_base + d_offsets * stride_h_d, h_new, mask=d_mask)
    tl.store(bx_prev_ptr + h_base + d_offsets * stride_bx_d, Bx_curr, mask=d_mask)


def mamba3_mimo_full_fused_triton(
    zxBCdtAtrap: torch.Tensor,
    dt_bias: torch.Tensor,
    D_skip: torch.Tensor,
    B_norm_weight: torch.Tensor,
    C_norm_weight: torch.Tensor,
    B_bias: torch.Tensor,
    C_bias: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    h: torch.Tensor,
    bx_prev: torch.Tensor,
    angle_state: torch.Tensor,
    d_inner: int,
    d_state: int,
    ngroups: int,
    mimo_rank: int,
    nheads: int,
    headdim: int,
    split_tensor_size: int,
    num_rope_angles: int,
    A_floor: float,
) -> tuple:
    """Launch the MIMO full-step fused decode kernel (R=4)."""
    B_batch = zxBCdtAtrap.shape[0]
    d_in_proj = zxBCdtAtrap.shape[-1]
    H = nheads
    P = headdim
    D = d_state
    d_BC = d_state * ngroups * mimo_rank

    assert mimo_rank == 4, "Full fused kernel only supports R=4 for now"
    assert h.shape == (B_batch, H, D)
    assert bx_prev.shape == (B_batch, H, D)
    assert angle_state.shape == (B_batch, H, num_rope_angles)
    assert zxBCdtAtrap.is_cuda and h.is_cuda

    y_gated = torch.empty(B_batch, H, P, device=zxBCdtAtrap.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if D <= 64 else 8
    grid = (B_batch * H,)

    _mimo_full_fused_kernel_R4[grid](
        zxBCdtAtrap, y_gated,
        h, bx_prev, angle_state,
        dt_bias, D_skip, B_norm_weight, C_norm_weight, B_bias, C_bias,
        mimo_x, mimo_o,
        B_batch, H, P, D, d_in_proj, d_inner, d_BC,
        num_rope_angles, split_tensor_size, A_floor,
        zxBCdtAtrap.stride(0), zxBCdtAtrap.stride(1),
        y_gated.stride(0), y_gated.stride(1), y_gated.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        bx_prev.stride(0), bx_prev.stride(1), bx_prev.stride(2),
        angle_state.stride(0), angle_state.stride(1), angle_state.stride(2),
        dt_bias.stride(0),
        D_skip.stride(0),
        B_norm_weight.stride(0),
        C_norm_weight.stride(0),
        B_bias.stride(0), B_bias.stride(1), B_bias.stride(2),
        C_bias.stride(0), C_bias.stride(1), C_bias.stride(2),
        mimo_x.stride(0), mimo_x.stride(1), mimo_x.stride(2),
        mimo_o.stride(0), mimo_o.stride(1), mimo_o.stride(2),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return y_gated, h, bx_prev, angle_state
