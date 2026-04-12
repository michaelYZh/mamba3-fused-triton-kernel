"""
Shared utilities for Mamba-3 Triton kernels.

Provides:
  - RoPE (Rotary Position Embedding) helpers used by both SISO and MIMO kernels
  - Exponential-trapezoidal discretization helpers
  - Common Triton helper functions (stride computation, etc.)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# PyTorch-level RoPE utilities (for reference & testing)
# ---------------------------------------------------------------------------

def build_rope_freqs(num_angles: int, device: torch.device) -> torch.Tensor:
    """Build standard RoPE inverse-frequency vector.

    Each pair of dimensions (2i, 2i+1) rotates at frequency 1/10000^(2i/d).
    Returns:
        freqs: (num_angles,)
    """
    i = torch.arange(num_angles, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (i / num_angles))
    return freqs


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions of x by the given angles.

    x:       (..., 2 * num_angles) — tensor to rotate
    angles:  (..., num_angles)     — rotation angles in radians
    """
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    out = torch.stack([x_rotated_1, x_rotated_2], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------------------
# PyTorch-level trapezoidal blending (for reference & testing)
# ---------------------------------------------------------------------------

def trapezoidal_blend(Bx_curr: torch.Tensor, Bx_prev: torch.Tensor,
                      trap: torch.Tensor) -> torch.Tensor:
    """Blend current and previous Bx using trapezoidal gate.

    trap=0: pure Euler (current only)
    trap=1: full trapezoidal average of current and previous

    Args:
        Bx_curr: current B*x, shape (..., D)
        Bx_prev: previous B*x, shape (..., D)
        trap:    sigmoid gate, shape broadcastable to Bx_curr

    Returns:
        Blended Bx, same shape as Bx_curr
    """
    return (1.0 - trap) * Bx_curr + trap * 0.5 * (Bx_curr + Bx_prev)


# ---------------------------------------------------------------------------
# PyTorch SISO single-step decode (reference implementation)
# ---------------------------------------------------------------------------

def mamba3_siso_decode_ref(
    x: torch.Tensor,       # (B, H, P)
    B_proj: torch.Tensor,  # (B, H, D)
    C_proj: torch.Tensor,  # (B, H, D)
    ADT: torch.Tensor,     # (B, H) — A*dt (negative)
    DT: torch.Tensor,      # (B, H) — dt (positive)
    trap: torch.Tensor,    # (B, H) — sigmoid gate
    D_skip: torch.Tensor,  # (H,)
    h: torch.Tensor,       # (B, H, P, D) — SSM state
    bx_prev: torch.Tensor, # (B, H, P, D) — previous B*x
) -> tuple:
    """Single-step SISO decode reference (PyTorch).

    Returns:
        y:         (B, H, P)
        h_new:     (B, H, P, D)
        bx_new:    (B, H, P, D)
    """
    decay = torch.exp(ADT).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
    dt_e = DT.unsqueeze(-1).unsqueeze(-1)               # (B, H, 1, 1)
    tr_e = trap.unsqueeze(-1).unsqueeze(-1)              # (B, H, 1, 1)

    Bx_curr = torch.einsum("bhp,bhd->bhpd", x.float(), B_proj.float())
    Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + bx_prev.float())

    h_new = decay * h.float() + dt_e * Bx_blended
    y = torch.einsum("bhd,bhpd->bhp", C_proj.float(), h_new)
    y = y + D_skip.unsqueeze(0).unsqueeze(-1) * x.float()

    return y, h_new, Bx_curr


# ---------------------------------------------------------------------------
# PyTorch MIMO single-step decode (reference implementation)
# ---------------------------------------------------------------------------

def mamba3_mimo_decode_ref(
    x: torch.Tensor,       # (B, H, P)
    B_proj: torch.Tensor,  # (B, R, H, D)
    C_proj: torch.Tensor,  # (B, R, H, D)
    ADT: torch.Tensor,     # (B, H)
    DT: torch.Tensor,      # (B, H)
    trap: torch.Tensor,    # (B, H)
    D_skip: torch.Tensor,  # (H,)
    mimo_x: torch.Tensor,  # (H, R, P)
    mimo_o: torch.Tensor,  # (H, R, P)
    h: torch.Tensor,       # (B, H, D) — MIMO state
    bx_prev: torch.Tensor, # (B, H, D) — previous Bx
) -> tuple:
    """Single-step MIMO decode reference (PyTorch).

    Returns:
        y:         (B, H, P)
        h_new:     (B, H, D)
        bx_new:    (B, H, D)
    """
    # Down-project x from P to R scalars per head
    x_r = torch.einsum("bhp,hrp->bhr", x.float(), mimo_x.float())  # (B, H, R)

    # Accumulate R rank-1 contributions
    Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_proj.float())  # (B, H, D)

    # Trapezoidal blend
    tr_e = trap.unsqueeze(-1)  # (B, H, 1)
    Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + bx_prev.float())

    # State update
    h_new = ADT.unsqueeze(-1).exp() * h.float() + DT.unsqueeze(-1) * Bx_blended  # (B, H, D)

    # Per-rank output scalars
    y_r_scalar = torch.einsum("brhd,bhd->brh", C_proj.float(), h_new)  # (B, R, H)
    skip = D_skip.unsqueeze(0).unsqueeze(0) * x_r.permute(0, 2, 1)    # (B, R, H)
    y_pre = y_r_scalar + skip                                           # (B, R, H)
    y = torch.einsum("brh,hrp->bhp", y_pre, mimo_o.float())            # (B, H, P)

    return y, h_new, Bx_curr


# ---------------------------------------------------------------------------
# Triton kernel helper: compute strides from shape
# ---------------------------------------------------------------------------

def _get_strides(shape: tuple, contiguous: bool = True) -> list:
    """Compute strides for a tensor of given shape (row-major / C-contiguous)."""
    if contiguous:
        strides = [1]
        for s in reversed(shape[1:]):
            strides.append(strides[-1] * s)
        return list(reversed(strides))
    else:
        raise ValueError("Only contiguous strides are supported")
