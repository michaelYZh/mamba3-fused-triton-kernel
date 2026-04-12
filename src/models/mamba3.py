"""
Mamba-3 model definition (adapted from reference implementation).

This module provides the Mamba3 layer with full parameter initialization,
forward (training) pass, and single-step decode (inference) pass.
The reference implementation is adapted from:
    https://github.com/rishikksh20/mamba3-pytorch

Key Mamba-3 innovations:
  1. Exponential-trapezoidal discretization (vs Euler/ZOH in Mamba-2)
  2. Complex-valued state via RoPE on B/C projections
  3. MIMO formulation (R parallel rank-1 updates vs single SISO update)
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.kernels.utils import apply_rope


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


class Mamba3(nn.Module):
    """Mamba-3 sequence mixing layer.

    Drop-in replacement for a Transformer attention layer.
    Input/output shape: (batch, seq_len, d_model).

    Parameters
    ----------
    d_model : int
        Token embedding dimension
    d_state : int
        SSM hidden state dimension per head (D in paper)
    expand : int
        Inner dimension multiplier; d_inner = expand * d_model
    headdim : int
        Dimension per SSM head; nheads = d_inner / headdim
    ngroups : int
        Number of groups for B/C projections
    rope_fraction : float
        Fraction of d_state that uses rotation (0.5 or 1.0)
    dt_min / dt_max : float
        Range for initial dt values
    is_mimo : bool
        If True, use MIMO formulation
    mimo_rank : int
        Number of parallel MIMO streams (R in paper)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.A_floor = A_floor
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.num_bc_heads = ngroups

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim

        # RoPE / angle dimensions
        assert rope_fraction in [0.5, 1.0]
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Input projection
        d_in_proj = (
            2 * self.d_inner
            + 2 * d_state * ngroups * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt bias
        _dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True

        # B and C biases
        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.B_bias._no_weight_decay = True
        self.C_bias._no_weight_decay = True

        # RMS norms for B and C
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        # MIMO projection matrices
        if self.is_mimo:
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs) / self.mimo_rank
            )
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs)
            )
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs) / self.mimo_rank
            )

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def _compute_params(self, zxBCdtAtrap, batch_size, L=None):
        """Split projected input into named components and compute derived params.

        Works for both training (L present) and inference (L=None) modes.
        """
        (z, x, B_raw, C_raw,
         dd_dt, dd_A, trap_raw, angle_raw) = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        if L is not None:
            z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
            x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
            B_raw = rearrange(B_raw, "b l (r g n) -> b l r g n",
                              r=self.mimo_rank, g=self.num_bc_heads)
            C_raw = rearrange(C_raw, "b l (r g n) -> b l r g n",
                              r=self.mimo_rank, g=self.num_bc_heads)
        else:
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            B_raw = rearrange(B_raw, "b (r g n) -> b r g n",
                              r=self.mimo_rank, g=self.num_bc_heads)
            C_raw = rearrange(C_raw, "b (r g n) -> b r g n",
                              r=self.mimo_rank, g=self.num_bc_heads)

        A = -F.softplus(dd_A.float()).clamp(max=-self.A_floor)
        DT = F.softplus(dd_dt.float() + self.dt_bias)
        ADT = A * DT
        trap = torch.sigmoid(trap_raw.float())

        # RMS norm + expand + bias
        B_normed = self.B_norm(B_raw.float())
        C_normed = self.C_norm(C_raw.float())

        if L is not None:
            B_exp = B_normed.expand(-1, -1, -1, self.nheads, -1)
            C_exp = C_normed.expand(-1, -1, -1, self.nheads, -1)
        else:
            B_exp = B_normed.expand(-1, -1, self.nheads, -1)
            C_exp = C_normed.expand(-1, -1, self.nheads, -1)

        B_bias_t = rearrange(self.B_bias, "h r d -> r h d")
        C_bias_t = rearrange(self.C_bias, "h r d -> r h d")
        B_exp = B_exp + B_bias_t
        C_exp = C_exp + C_bias_t

        return z, x, B_exp, C_exp, ADT, DT, trap, angle_raw

    def _apply_rope_to_bc(self, B_exp, C_exp, angle_raw, DT, batch_size, L=None):
        """Apply RoPE rotation to B and C projections."""
        if L is not None:
            angle_increments = (
                angle_raw.float().unsqueeze(2) * DT.float().unsqueeze(-1)
            )
            cumulative_angles = torch.cumsum(angle_increments, dim=1)
            angles_for_rot = cumulative_angles.unsqueeze(2).expand(
                batch_size, L, self.mimo_rank, self.nheads, self.num_rope_angles
            )
        else:
            # Single step: angle_state is handled externally in step()
            # This path is not used in step() — see step() for angle accumulation
            raise NotImplementedError("Use step() for single-step decode with RoPE")

        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)

        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)

        return B_proj, C_proj

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Training forward pass: (B, L, d_model) → (B, L, d_model)"""
        batch, L, _ = u.shape

        zxBCdtAtrap = self.in_proj(u)
        z, x, B_exp, C_exp, ADT, DT, trap, angle_raw = self._compute_params(
            zxBCdtAtrap, batch, L=L
        )

        B_proj, C_proj = self._apply_rope_to_bc(
            B_exp, C_exp, angle_raw, DT, batch, L=L
        )

        if self.is_mimo:
            y = self._mimo_scan(x, B_proj, C_proj, ADT, DT, trap)
            y = y * F.silu(z.float())
        else:
            y = self._siso_scan(x, B_proj[:, :, 0], C_proj[:, :, 0], ADT, DT, trap)
            y = y * F.silu(z.float())

        y = rearrange(y, "b l h p -> b l (h p)")
        out = self.out_proj(y.to(u.dtype))
        return out

    def _siso_scan(self, x, B_proj, C_proj, ADT, DT, trap):
        """Sequential SISO scan for training."""
        from src.kernels.utils import mamba3_siso_decode_ref

        B_batch, L, H, P = x.shape
        D_state = B_proj.shape[-1]
        h = torch.zeros(B_batch, H, P, D_state, device=x.device, dtype=torch.float32)
        bx_prev = torch.zeros_like(h)
        ys = []

        for t in range(L):
            y, h, bx_prev = mamba3_siso_decode_ref(
                x[:, t], B_proj[:, t], C_proj[:, t],
                ADT[:, t], DT[:, t], trap[:, t],
                self.D, h, bx_prev,
            )
            ys.append(y)

        return torch.stack(ys, dim=1)

    def _mimo_scan(self, x, B_proj, C_proj, ADT, DT, trap):
        """Sequential MIMO scan for training."""
        from src.kernels.utils import mamba3_mimo_decode_ref

        B_batch, L, H, P = x.shape
        D_state = B_proj.shape[-1]
        h = torch.zeros(B_batch, H, D_state, device=x.device, dtype=torch.float32)
        bx_prev = torch.zeros_like(h)
        ys = []

        for t in range(L):
            y, h, bx_prev = mamba3_mimo_decode_ref(
                x[:, t], B_proj[:, t], C_proj[:, t],
                ADT[:, t], DT[:, t], trap[:, t],
                self.D, self.mimo_x, self.mimo_o, h, bx_prev,
            )
            ys.append(y)

        return torch.stack(ys, dim=1)

    def step(self, u, angle_state, ssm_state, Bx_prev_state):
        """Single-step autoregressive decode.

        Args:
            u:              (batch, d_model) — single token embedding
            angle_state:    (batch, H, num_rope_angles)
            ssm_state:      (batch, H, P, D) or (batch, H, D)
            Bx_prev_state:  same shape as ssm_state

        Returns:
            out, angle_state, ssm_state, Bx_prev_state
        """
        batch = u.shape[0]

        zxBCdtAtrap = self.in_proj(u)
        z, x, B_exp, C_exp, ADT, DT, trap, angle_raw = self._compute_params(
            zxBCdtAtrap, batch, L=None
        )

        # RoPE: accumulate angle state
        delta_angle = angle_raw.float().unsqueeze(1) * DT.float().unsqueeze(-1)
        angle_state = angle_state + delta_angle

        angles_for_rot = angle_state.unsqueeze(1).expand(-1, self.mimo_rank, -1, -1)
        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)
        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)

        if self.is_mimo:
            from src.kernels.utils import mamba3_mimo_decode_ref
            y, ssm_state, Bx_prev_state = mamba3_mimo_decode_ref(
                x, B_proj, C_proj, ADT, DT, trap,
                self.D, self.mimo_x, self.mimo_o,
                ssm_state, Bx_prev_state,
            )
        else:
            from src.kernels.utils import mamba3_siso_decode_ref
            y, ssm_state, Bx_prev_state = mamba3_siso_decode_ref(
                x, B_proj[:, 0], C_proj[:, 0], ADT, DT, trap,
                self.D, ssm_state, Bx_prev_state,
            )

        y = y * F.silu(z.float())
        y = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(y.to(u.dtype))

        return out, angle_state, ssm_state, Bx_prev_state

    def allocate_inference_cache(self, batch_size: int, device=None, dtype=None):
        """Allocate zero-initialized states for autoregressive inference."""
        device = device or self.in_proj.weight.device

        angle_state = torch.zeros(
            batch_size, self.nheads, self.num_rope_angles,
            device=device, dtype=torch.float32,
        )
        if self.is_mimo:
            ssm_state = torch.zeros(
                batch_size, self.nheads, self.d_state,
                device=device, dtype=torch.float32,
            )
            Bx_prev_state = torch.zeros(
                batch_size, self.nheads, self.d_state,
                device=device, dtype=torch.float32,
            )
        else:
            ssm_state = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            )
            Bx_prev_state = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            )
        return angle_state, ssm_state, Bx_prev_state

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_inner={self.d_inner}, nheads={self.nheads}, "
            f"headdim={self.headdim}, is_mimo={self.is_mimo}, "
            f"mimo_rank={self.mimo_rank}, num_rope_angles={self.num_rope_angles}"
        )


# ---------------------------------------------------------------------------
# Full stacked model
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class MambaBlock(nn.Module):
    """Single Mamba-3 residual block: Norm → Mamba3 → residual add."""

    def __init__(self, d_model: int, ssm_cfg: dict, device=None, dtype=None):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba3(d_model=d_model, **ssm_cfg, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MLP(nn.Module):
    """SwiGLU-style feed-forward layer."""

    def __init__(self, d_model: int, d_intermediate: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=False, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * val)


class MambaLMHeadModel(nn.Module):
    """Full language model built from stacked Mamba-3 blocks."""

    def __init__(self, config: MambaConfig, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config

        vocab_size = config.vocab_size
        r = vocab_size % config.pad_vocab_size_multiple
        if r != 0:
            vocab_size += config.pad_vocab_size_multiple - r
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model, **factory_kwargs)
        self.layers = nn.ModuleList([
            MambaBlock(config.d_model, config.ssm_cfg, **factory_kwargs)
            for _ in range(config.n_layer)
        ])

        if config.d_intermediate > 0:
            self.mlp_norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(config.n_layer)])
            self.mlp_layers = nn.ModuleList([
                MLP(config.d_model, config.d_intermediate, **factory_kwargs)
                for _ in range(config.n_layer)
            ])
        else:
            self.mlp_norms = None
            self.mlp_layers = None

        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(vocab_size, config.d_model, bias=False, **factory_kwargs)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for i, block in enumerate(self.layers):
            x = block(x)
            if self.mlp_layers is not None:
                x = x + self.mlp_layers[i](self.mlp_norms[i](x))
        x = self.norm_f(x)
        return self.lm_head(x)
