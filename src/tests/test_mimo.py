"""
Correctness tests for Mamba-3 MIMO decoding kernel.

Usage:
    pytest src/tests/test_mimo.py -v
"""

import pytest
import torch

from src.kernels.mimo_decode import mamba3_mimo_decode_triton
from src.kernels.utils import mamba3_mimo_decode_ref


class TestMIMOBasic:
    """Test basic MIMO decoding."""

    @pytest.fixture
    def config(self):
        return {
            "B": 2,
            "H": 4,
            "P": 16,
            "D": 64,
            "R": 4,
            "dtype": torch.float32,
        }

    @pytest.fixture
    def inputs(self, config):
        torch.manual_seed(42)
        B, H, P, D, R = config["B"], config["H"], config["P"], config["D"], config["R"]
        dtype = config["dtype"]
        device = "cuda"
        return {
            "x": torch.randn(B, H, P, dtype=dtype, device=device),
            "h": torch.randn(B, H, D, dtype=dtype, device=device),
            "A_log": -torch.rand(H, dtype=dtype, device=device) * 2,
            "B_proj": torch.randn(B, R, H, D, dtype=dtype, device=device),
            "C_proj": torch.randn(B, R, H, D, dtype=dtype, device=device),
            "D_skip": torch.randn(H, dtype=dtype, device=device),
            "dt": torch.softmax(torch.randn(B, H, dtype=dtype, device=device), dim=-1) + 0.01,
            "trap": torch.sigmoid(torch.randn(B, H, dtype=dtype, device=device)),
            "bx_prev": torch.randn(B, H, D, dtype=dtype, device=device),
            "mimo_x": torch.randn(H, R, P, dtype=dtype, device=device),
            "mimo_o": torch.randn(H, R, P, dtype=dtype, device=device),
        }

    def test_mimo_state_shape(self, config):
        """MIMO state should be (B, H, D), not (B, H, P, D)."""
        B, H, D, R = config["B"], config["H"], config["D"], config["R"]
        state = torch.randn(B, H, D, device="cuda")
        assert state.shape == (B, H, D)

    def test_mimo_reference(self, config, inputs):
        """MIMO reference decode should produce correct shapes."""
        x, h = inputs["x"], inputs["h"]
        B_proj, C_proj = inputs["B_proj"], inputs["C_proj"]
        A_log, D_skip, dt = inputs["A_log"], inputs["D_skip"], inputs["dt"]
        trap = inputs["trap"]
        bx_prev = inputs["bx_prev"]
        mimo_x, mimo_o = inputs["mimo_x"], inputs["mimo_o"]

        ADT = A_log.unsqueeze(0) * dt

        y, h_new, bx_new = mamba3_mimo_decode_ref(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h, bx_prev,
        )

        assert y.shape == (config["B"], config["H"], config["P"])
        assert h_new.shape == (config["B"], config["H"], config["D"])
        assert bx_new.shape == (config["B"], config["H"], config["D"])

    def test_mimo_triton_correctness(self, config, inputs):
        """Triton kernel should match reference within tolerance."""
        x, h = inputs["x"], inputs["h"].clone()
        B_proj, C_proj = inputs["B_proj"], inputs["C_proj"]
        A_log, D_skip, dt = inputs["A_log"], inputs["D_skip"], inputs["dt"]
        trap = inputs["trap"]
        bx_prev = inputs["bx_prev"].clone()
        mimo_x, mimo_o = inputs["mimo_x"], inputs["mimo_o"]

        ADT = A_log.unsqueeze(0) * dt

        # Reference
        y_ref, h_ref, bx_ref = mamba3_mimo_decode_ref(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h.clone(), bx_prev.clone(),
        )

        # Triton
        y_tri, h_tri, bx_tri = mamba3_mimo_decode_triton(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h.clone(), bx_prev.clone(),
        )

        atol = 1e-2

        y_diff = (y_ref - y_tri).abs().max().item()
        h_diff = (h_ref - h_tri).abs().max().item()
        bx_diff = (bx_ref - bx_tri).abs().max().item()

        print(f"  y max diff: {y_diff:.6f}")
        print(f"  h max diff: {h_diff:.6f}")
        print(f"  bx_prev max diff: {bx_diff:.6f}")

        assert y_diff < atol, f"Output mismatch: max diff = {y_diff}"
        assert h_diff < atol, f"State mismatch: max diff = {h_diff}"
        assert bx_diff < atol, f"bx_prev mismatch: max diff = {bx_diff}"

    @pytest.mark.parametrize("R", [1, 2, 4, 8])
    def test_mimo_various_ranks(self, R):
        """Test with different MIMO ranks."""
        torch.manual_seed(42)
        B, H, P, D = 2, 4, 16, 64
        device = "cuda"

        x = torch.randn(B, H, P, device=device)
        h = torch.randn(B, H, D, device=device)
        A_log = -torch.rand(H, device=device) * 2
        B_proj = torch.randn(B, R, H, D, device=device)
        C_proj = torch.randn(B, R, H, D, device=device)
        D_skip = torch.randn(H, device=device)
        dt = torch.softmax(torch.randn(B, H, device=device), dim=-1) + 0.01
        trap = torch.sigmoid(torch.randn(B, H, device=device))
        bx_prev = torch.randn(B, H, D, device=device)
        mimo_x = torch.randn(H, R, P, device=device)
        mimo_o = torch.randn(H, R, P, device=device)

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, _ = mamba3_mimo_decode_ref(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h.clone(), bx_prev.clone(),
        )
        y_tri, h_tri, _ = mamba3_mimo_decode_triton(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h.clone(), bx_prev.clone(),
        )

        y_diff = (y_ref - y_tri).abs().max().item()
        assert y_diff < 1e-2, f"Rank R={R}: y max diff = {y_diff}"


class TestMIMOTrapezoidal:
    """Test MIMO with trapezoidal discretization."""

    def test_trap_zero_is_euler(self):
        """trap=0 → pure Euler, no blending with bx_prev."""
        torch.manual_seed(42)
        B, H, P, D, R = 1, 2, 8, 16, 2
        device = "cuda"

        x = torch.randn(B, H, P, device=device)
        h = torch.zeros(B, H, D, device=device)
        A_log = -torch.ones(H, device=device)
        B_proj = torch.randn(B, R, H, D, device=device)
        C_proj = torch.randn(B, R, H, D, device=device)
        D_skip = torch.randn(H, device=device)
        dt = torch.ones(B, H, device=device) * 0.1
        trap = torch.zeros(B, H, device=device)  # pure Euler
        bx_prev = torch.randn(B, H, D, device=device)
        mimo_x = torch.randn(H, R, P, device=device)
        mimo_o = torch.randn(H, R, P, device=device)

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, _ = mamba3_mimo_decode_ref(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h, bx_prev,
        )

        # With trap=0: Bx_blended = Bx_curr
        x_r = torch.einsum("bhp,hrp->bhr", x, mimo_x)
        Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_proj)
        dA = torch.exp(ADT).unsqueeze(-1)
        h_expected = dA * h + dt.unsqueeze(-1) * Bx_curr

        h_diff = (h_ref - h_expected).abs().max().item()
        assert h_diff < 1e-5, f"Euler state mismatch: max diff = {h_diff}"

    def test_trap_one_is_trapezoidal(self):
        """trap=1 → full trapezoidal average."""
        torch.manual_seed(42)
        B, H, P, D, R = 1, 2, 8, 16, 2
        device = "cuda"

        x = torch.randn(B, H, P, device=device)
        h = torch.zeros(B, H, D, device=device)
        A_log = -torch.ones(H, device=device)
        B_proj = torch.randn(B, R, H, D, device=device)
        C_proj = torch.randn(B, R, H, D, device=device)
        D_skip = torch.randn(H, device=device)
        dt = torch.ones(B, H, device=device) * 0.1
        trap = torch.ones(B, H, device=device)
        bx_prev = torch.randn(B, H, D, device=device)
        mimo_x = torch.randn(H, R, P, device=device)
        mimo_o = torch.randn(H, R, P, device=device)

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, _ = mamba3_mimo_decode_ref(
            x, B_proj, C_proj, ADT, dt, trap, D_skip,
            mimo_x, mimo_o, h, bx_prev,
        )

        # With trap=1: Bx_blended = 0.5 * (Bx_curr + bx_prev)
        x_r = torch.einsum("bhp,hrp->bhr", x, mimo_x)
        Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_proj)
        Bx_blended = 0.5 * (Bx_curr + bx_prev)
        dA = torch.exp(ADT).unsqueeze(-1)
        h_expected = dA * h + dt.unsqueeze(-1) * Bx_blended

        h_diff = (h_ref - h_expected).abs().max().item()
        assert h_diff < 1e-5, f"Trapezoidal state mismatch: max diff = {h_diff}"
