"""
Mamba-3 inference interface with Triton kernel integration.

Provides a high-level Mamba3Decoder that:
  - Manages inference state (angle, ssm_state, bx_prev) across decoding steps
  - Supports both PyTorch reference and Triton fused kernel backends
  - Optional CUDA Graph support for minimal launch overhead
  - Handles the full projection pipeline (in_proj → split → norm → RoPE → decode → out_proj)
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models.mamba3 import Mamba3, RMSNorm


class Mamba3Decoder:
    """High-level Mamba-3 decoder for autoregressive inference.

    Wraps a trained Mamba3 module and provides a simple step() interface
    that manages state automatically. Supports both Triton and PyTorch backends,
    with optional CUDA Graph acceleration.

    Usage:
        model = Mamba3(d_model=768, ...)
        decoder = Mamba3Decoder(model, use_triton=True)
        state = decoder.init_state(batch_size=1)
        for token_emb in token_sequence:
            output, state = decoder.step(token_emb, state)

    CUDA Graph usage (for fixed-shape autoregressive decoding):
        decoder = Mamba3Decoder(model, use_triton=True, use_cuda_graph=True)
        state = decoder.init_state(batch_size=1)
        decoder.warmup_cuda_graph(state)  # warmup + capture
        for token_emb in token_sequence:
            output, state = decoder.step(token_emb, state)
    """

    def __init__(self, model: Mamba3, use_triton: bool = True, use_cuda_graph: bool = False):
        self.model = model
        self.use_triton = use_triton and torch.cuda.is_available()
        self.use_cuda_graph = use_cuda_graph and self.use_triton
        self._cuda_graph = None
        self._graph_input = None
        self._graph_output = None
        self._graph_state = None

        if use_triton and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to PyTorch backend")
        if use_cuda_graph and not self.use_triton:
            print("Warning: CUDA Graph requires Triton backend, disabling")

    def init_state(self, batch_size: int, device=None, dtype=None):
        """Initialize inference state.

        Returns:
            dict with keys: angle_state, ssm_state, bx_prev
        """
        angle_state, ssm_state, bx_prev = self.model.allocate_inference_cache(
            batch_size, device=device, dtype=dtype
        )
        return {
            "angle_state": angle_state,
            "ssm_state": ssm_state,
            "bx_prev": bx_prev,
        }

    def warmup_cuda_graph(self, state: dict, n_warmup: int = 10):
        """Warm up and capture CUDA Graph for the step function.

        Must be called before the first step() when use_cuda_graph=True.
        The state dict is captured in the graph and will be updated in-place.

        Args:
            state: initial state dict from init_state()
            n_warmup: number of warmup iterations before capturing
        """
        if not self.use_cuda_graph:
            return

        batch_size = state["ssm_state"].shape[0]
        device = state["ssm_state"].device

        # Create static input
        self._graph_input = torch.zeros(batch_size, self.model.d_model, device=device)

        # Warmup runs
        for _ in range(n_warmup):
            self._step_triton(self._graph_input, state)
        torch.cuda.synchronize()

        # Capture CUDA Graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        self._graph_state = state

        with torch.cuda.graph(self._cuda_graph):
            self._graph_output, self._graph_state = self._step_triton(
                self._graph_input, self._graph_state
            )

    def step(self, u: torch.Tensor, state: dict) -> tuple:
        """Run a single decoding step.

        Args:
            u:     (batch, d_model) — token embedding
            state: dict from init_state() or previous step()

        Returns:
            output: (batch, d_model)
            state:  updated state dict
        """
        if self.use_cuda_graph and self._cuda_graph is not None:
            # Copy input to static buffer and replay graph
            self._graph_input.copy_(u)
            self._cuda_graph.replay()
            return self._graph_output, self._graph_state
        elif self.use_triton:
            return self._step_triton(u, state)
        else:
            return self._step_pytorch(u, state)

    def _step_pytorch(self, u: torch.Tensor, state: dict) -> tuple:
        """PyTorch reference decode step (uses model.step())."""
        out, angle_state, ssm_state, bx_prev = self.model.step(
            u,
            state["angle_state"],
            state["ssm_state"],
            state["bx_prev"],
        )
        state["angle_state"] = angle_state
        state["ssm_state"] = ssm_state
        state["bx_prev"] = bx_prev
        return out, state

    def _step_triton(self, u: torch.Tensor, state: dict) -> tuple:
        """Triton fused kernel decode step.

        Performs the projection/normalization/RoPE in PyTorch, then
        dispatches to the Triton kernel for the core SSM recurrence.
        """
        model = self.model
        batch = u.shape[0]

        # ── Input projection ────────────────────────────────────────────
        zxBCdtAtrap = model.in_proj(u)

        (z, x, B_raw, C_raw,
         dd_dt, dd_A, trap_raw, angle_raw) = torch.split(
            zxBCdtAtrap,
            [
                model.d_inner,
                model.d_inner,
                model.d_state * model.num_bc_heads * model.mimo_rank,
                model.d_state * model.num_bc_heads * model.mimo_rank,
                model.nheads,
                model.nheads,
                model.nheads,
                model.num_rope_angles,
            ],
            dim=-1,
        )

        z = rearrange(z, "b (h p) -> b h p", p=model.headdim)
        x = rearrange(x, "b (h p) -> b h p", p=model.headdim)
        B_raw = rearrange(B_raw, "b (r g n) -> b r g n",
                          r=model.mimo_rank, g=model.num_bc_heads)
        C_raw = rearrange(C_raw, "b (r g n) -> b r g n",
                          r=model.mimo_rank, g=model.num_bc_heads)

        # ── Compute parameters ──────────────────────────────────────────
        A = -F.softplus(dd_A.float()).clamp(max=-model.A_floor)
        DT = F.softplus(dd_dt.float() + model.dt_bias)
        ADT = A * DT
        trap = torch.sigmoid(trap_raw.float())

        # ── RMS norm + expand + bias ────────────────────────────────────
        B_normed = model.B_norm(B_raw.float())
        C_normed = model.C_norm(C_raw.float())
        B_exp = B_normed.expand(-1, -1, model.nheads, -1)
        C_exp = C_normed.expand(-1, -1, model.nheads, -1)
        B_bias_t = rearrange(model.B_bias, "h r d -> r h d")
        C_bias_t = rearrange(model.C_bias, "h r d -> r h d")
        B_exp = B_exp + B_bias_t
        C_exp = C_exp + C_bias_t

        # ── RoPE angle accumulation ─────────────────────────────────────
        delta_angle = angle_raw.float().unsqueeze(1) * DT.float().unsqueeze(-1)
        state["angle_state"] = state["angle_state"] + delta_angle

        angles_for_rot = state["angle_state"].unsqueeze(1).expand(
            -1, model.mimo_rank, -1, -1
        )

        from src.kernels.utils import apply_rope
        B_rot = apply_rope(B_exp[..., :model.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :model.split_tensor_size], angles_for_rot)
        B_proj = torch.cat([B_rot, B_exp[..., model.split_tensor_size:]], dim=-1)
        C_proj = torch.cat([C_rot, C_exp[..., model.split_tensor_size:]], dim=-1)

        # ── Dispatch to Triton kernel ───────────────────────────────────
        if model.is_mimo:
            from src.kernels.mimo_decode import mamba3_mimo_decode_triton
            y, ssm_state_new, bx_prev_new = mamba3_mimo_decode_triton(
                x, B_proj, C_proj, ADT, DT, trap,
                model.D, model.mimo_x, model.mimo_o,
                state["ssm_state"], state["bx_prev"],
            )
        else:
            from src.kernels.siso_decode import mamba3_siso_decode_triton
            y, ssm_state_new, bx_prev_new = mamba3_siso_decode_triton(
                x, B_proj[:, 0], C_proj[:, 0], ADT, DT, trap,
                model.D, state["ssm_state"], state["bx_prev"],
            )

        # ── Gate + output projection ────────────────────────────────────
        y = y * F.silu(z.float())
        y = rearrange(y, "b h p -> b (h p)")
        out = model.out_proj(y.to(u.dtype))

        state["ssm_state"] = ssm_state_new
        state["bx_prev"] = bx_prev_new

        return out, state

    def decode_sequence(self, token_embeddings: torch.Tensor, state: dict = None) -> tuple:
        """Decode a full sequence autoregressively.

        Args:
            token_embeddings: (batch, seq_len, d_model)
            state: initial state (if None, will be initialized)

        Returns:
            outputs: (batch, seq_len, d_model)
            state: final state dict
        """
        batch, seq_len, d_model = token_embeddings.shape
        if state is None:
            state = self.init_state(batch, device=token_embeddings.device)

        outputs = []
        for t in range(seq_len):
            out, state = self.step(token_embeddings[:, t], state)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return outputs, state
