"""
Correctness tests for Mamba-3 SISO decoding kernel.

Usage:
    pytest src/tests/test_siso.py -v
"""

import pytest
import torch

from src.kernels.siso_decode import mamba3_siso_decode_triton
from src.kernels.utils import mamba3_siso_decode_ref


class TestSISOBasic:
    """Test basic SISO decoding without trapezoidal or RoPE."""

    @pytest.fixture
    def config(self):
        return {
            "B": 2,
            "H": 4,
            "P": 16,
            "D": 64,
            "dtype": torch.float32,
        }

    @pytest.fixture
    def inputs(self, config):
        torch.manual_seed(42)
        B, H, P, D = config["B"], config["H"], config["P"], config["D"]
        dtype = config["dtype"]
        device = "cuda"
        return {
            "x": torch.randn(B, H, P, dtype=dtype, device=device),
            "h": torch.randn(B, H, P, D, dtype=dtype, device=device),
            "A_log": -torch.rand(H, dtype=dtype, device=device) * 2,  # negative
            "B_param": torch.randn(B, H, D, dtype=dtype, device=device),
            "C_param": torch.randn(B, H, D, dtype=dtype, device=device),
            "D_skip": torch.randn(H, dtype=dtype, device=device),
            "dt": torch.softmax(torch.randn(B, H, dtype=dtype, device=device), dim=-1) + 0.01,
            "trap": torch.sigmoid(torch.randn(B, H, dtype=dtype, device=device)),
            "bx_prev": torch.randn(B, H, P, D, dtype=dtype, device=device),
        }

    def test_siso_reference(self, config, inputs):
        """Basic SISO reference: h = dA*h + B*x*dt, y = C*h + D*x"""
        x, h = inputs["x"], inputs["h"]
        A_log, B_p, C_p, D_skip, dt = (
            inputs["A_log"], inputs["B_param"], inputs["C_param"],
            inputs["D_skip"], inputs["dt"]
        )
        trap = inputs["trap"]
        bx_prev = inputs["bx_prev"]

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, bx_ref = mamba3_siso_decode_ref(
            x, B_p, C_p, ADT, dt, trap, D_skip, h, bx_prev
        )

        assert y_ref.shape == (config["B"], config["H"], config["P"])
        assert h_ref.shape == (config["B"], config["H"], config["P"], config["D"])

    def test_siso_triton_correctness(self, config, inputs):
        """Triton kernel should match reference within tolerance."""
        x, h = inputs["x"], inputs["h"].clone()
        A_log, B_p, C_p, D_skip, dt = (
            inputs["A_log"], inputs["B_param"], inputs["C_param"],
            inputs["D_skip"], inputs["dt"]
        )
        trap = inputs["trap"]
        bx_prev = inputs["bx_prev"].clone()

        ADT = A_log.unsqueeze(0) * dt

        # Reference
        y_ref, h_ref, bx_ref = mamba3_siso_decode_ref(
            x, B_p, C_p, ADT, dt, trap, D_skip, h.clone(), bx_prev.clone()
        )

        # Triton
        y_tri, h_tri, bx_tri = mamba3_siso_decode_triton(
            x, B_p, C_p, ADT, dt, trap, D_skip, h.clone(), bx_prev.clone()
        )

        atol_loose = 1e-2
        atol_medium = 1e-3

        y_diff = (y_ref - y_tri).abs().max().item()
        h_diff = (h_ref - h_tri).abs().max().item()
        bx_diff = (bx_ref - bx_tri).abs().max().item()

        print(f"  y max diff: {y_diff:.6f}")
        print(f"  h max diff: {h_diff:.6f}")
        print(f"  bx_prev max diff: {bx_diff:.6f}")

        assert y_diff < atol_loose, f"Output mismatch (loose): max diff = {y_diff}"
        assert h_diff < atol_loose, f"State mismatch (loose): max diff = {h_diff}"
        assert bx_diff < atol_loose, f"bx_prev mismatch (loose): max diff = {bx_diff}"

    @pytest.mark.parametrize("P,D", [(16, 64), (32, 128), (64, 64)])
    def test_siso_various_sizes(self, P, D):
        """Test with different headdim/d_state combinations."""
        torch.manual_seed(42)
        B, H = 2, 4
        device = "cuda"

        x = torch.randn(B, H, P, device=device)
        h = torch.randn(B, H, P, D, device=device)
        A_log = -torch.rand(H, device=device) * 2
        B_p = torch.randn(B, H, D, device=device)
        C_p = torch.randn(B, H, D, device=device)
        D_skip = torch.randn(H, device=device)
        dt = torch.softmax(torch.randn(B, H, device=device), dim=-1) + 0.01
        trap = torch.sigmoid(torch.randn(B, H, device=device))
        bx_prev = torch.randn(B, H, P, D, device=device)

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, _ = mamba3_siso_decode_ref(
            x, B_p, C_p, ADT, dt, trap, D_skip, h.clone(), bx_prev.clone()
        )
        y_tri, h_tri, _ = mamba3_siso_decode_triton(
            x, B_p, C_p, ADT, dt, trap, D_skip, h.clone(), bx_prev.clone()
        )

        y_diff = (y_ref - y_tri).abs().max().item()
        assert y_diff < 1e-2, f"Size ({P},{D}): y max diff = {y_diff}"


class TestSISOTrapezoidal:
    """Test SISO decoding with trapezoidal gate behavior."""

    def test_trap_zero_is_euler(self):
        """trap=0 should give pure Euler (no trapezoidal blending)."""
        torch.manual_seed(42)
        B, H, P, D = 1, 2, 8, 16
        device = "cuda"

        x = torch.randn(B, H, P, device=device)
        h = torch.zeros(B, H, P, D, device=device)
        A_log = -torch.ones(H, device=device)
        B_p = torch.randn(B, H, D, device=device)
        C_p = torch.randn(B, H, D, device=device)
        D_skip = torch.randn(H, device=device)
        dt = torch.ones(B, H, device=device) * 0.1
        trap = torch.zeros(B, H, device=device)  # pure Euler
        bx_prev = torch.randn(B, H, P, D, device=device)

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, _ = mamba3_siso_decode_ref(
            x, B_p, C_p, ADT, dt, trap, D_skip, h, bx_prev
        )

        # With trap=0: Bx_blended = Bx_curr (Euler)
        # h = dA * h + dt * Bx_curr
        dA = torch.exp(ADT).unsqueeze(-1).unsqueeze(-1)
        Bx_curr = torch.einsum("bhp,bhd->bhpd", x, B_p)
        h_expected = dA * h + dt.unsqueeze(-1).unsqueeze(-1) * Bx_curr

        h_diff = (h_ref - h_expected).abs().max().item()
        assert h_diff < 1e-5, f"Euler state mismatch: max diff = {h_diff}"

    def test_trap_one_is_trapezoidal(self):
        """trap=1 should give full trapezoidal average."""
        torch.manual_seed(42)
        B, H, P, D = 1, 2, 8, 16
        device = "cuda"

        x = torch.randn(B, H, P, device=device)
        h = torch.zeros(B, H, P, D, device=device)
        A_log = -torch.ones(H, device=device)
        B_p = torch.randn(B, H, D, device=device)
        C_p = torch.randn(B, H, D, device=device)
        D_skip = torch.randn(H, device=device)
        dt = torch.ones(B, H, device=device) * 0.1
        trap = torch.ones(B, H, device=device)  # full trapezoidal
        bx_prev = torch.randn(B, H, P, D, device=device)

        ADT = A_log.unsqueeze(0) * dt

        y_ref, h_ref, _ = mamba3_siso_decode_ref(
            x, B_p, C_p, ADT, dt, trap, D_skip, h, bx_prev
        )

        # With trap=1: Bx_blended = 0.5 * (Bx_curr + bx_prev)
        dA = torch.exp(ADT).unsqueeze(-1).unsqueeze(-1)
        Bx_curr = torch.einsum("bhp,bhd->bhpd", x, B_p)
        Bx_blended = 0.5 * (Bx_curr + bx_prev)
        h_expected = dA * h + dt.unsqueeze(-1).unsqueeze(-1) * Bx_blended

        h_diff = (h_ref - h_expected).abs().max().item()
        assert h_diff < 1e-5, f"Trapezoidal state mismatch: max diff = {h_diff}"
