"""
Sign of the decay rate A (see "Decay sign fix" in the README).

A = (-softplus(dd_A)).clamp(max=-A_floor), so A <= -A_floor < 0 and the
state decays: exp(A * DT) < 1. Through commit 4a98872 the code computed
-(softplus(dd_A).clamp(max=-A_floor)), which is the constant +A_floor.

Usage:
    pytest src/tests/test_decay_sign.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F

# src.kernels.utils, which src.models.mamba3 imports, imports triton at module level.
pytest.importorskip("triton")

from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder


def _dd_A_slice(model):
    # zxBCdtAtrap = [z, x, B, C, dd_dt, dd_A, trap, angle], as Mamba3._compute_params splits it
    start = (2 * model.d_inner
             + 2 * model.d_state * model.num_bc_heads * model.mimo_rank
             + model.nheads)
    return slice(start, start + model.nheads)


@pytest.mark.parametrize("is_mimo", [False, True], ids=["siso", "mimo"])
def test_compute_params_A_is_negative_and_follows_dd_A(is_mimo):
    torch.manual_seed(0)
    model = Mamba3(d_model=64, d_state=16, headdim=16, is_mimo=is_mimo, mimo_rank=4)
    batch = 4
    zx = torch.randn(batch, model.in_proj.out_features) * 3.0
    zx[0, _dd_A_slice(model)] = -20.0  # softplus(-20) < A_floor: the clamp is active on row 0
    dd_A = zx[:, _dd_A_slice(model)]

    with torch.no_grad():
        _, _, _, _, ADT, DT, _, _ = model._compute_params(zx, batch)

    assert (ADT / DT <= -model.A_floor * (1 - 1e-6)).all()
    # Upstream rishikksh20/mamba3-pytorch @ 1aa707d, Mamba3.forward (L508-L509)
    upstream_A = (-F.softplus(dd_A.float())).clamp(max=-model.A_floor)
    torch.testing.assert_close(ADT, upstream_A * DT, rtol=1e-6, atol=0.0)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("the full-step Triton kernel needs CUDA")
    return "cuda"


@pytest.mark.parametrize("is_mimo", [False, True], ids=["siso", "mimo"])
def test_full_step_kernel_state_decays_like_eager(is_mimo, device):
    """50 random steps, then 1000 zero-input steps through the full-step kernel."""
    torch.manual_seed(0)
    # Build on the CPU, so that the weights and inputs do not depend on the device RNG.
    model = Mamba3(d_model=256, d_state=64, headdim=32, is_mimo=is_mimo, mimo_rank=4).to(device)
    eager = Mamba3Decoder(model, use_triton=False)
    full = Mamba3Decoder(model, use_triton_full_fused=True)
    batch, n_random, n_zero = 2, 50, 1000
    inputs = torch.randn(n_random, batch, model.d_model).to(device)
    zero = torch.zeros(batch, model.d_model, device=device)
    s_eager, s_full = eager.init_state(batch), full.init_state(batch)

    with torch.no_grad():
        for u in inputs:
            _, s_eager = eager.step(u, s_eager)
            _, s_full = full.step(u, s_full)
        torch.testing.assert_close(s_full["ssm_state"], s_eager["ssm_state"], rtol=1e-4, atol=1e-4)
        norm_before = s_full["ssm_state"].norm().item()

        for _ in range(n_zero):
            _, s_eager = eager.step(zero, s_eager)
            _, s_full = full.step(zero, s_full)

    norm_after = s_full["ssm_state"].norm().item()
    ratio = norm_after / norm_before
    diff = (s_full["ssm_state"] - s_eager["ssm_state"]).abs().max().item()
    print(f"\n{'MIMO' if is_mimo else 'SISO'}: ||ssm_state|| {norm_before:.4g} -> {norm_after:.4g} "
          f"after {n_zero} zero-input steps (ratio {ratio:.3g}), max |full - eager| = {diff:.2e}")
    torch.testing.assert_close(s_full["ssm_state"], s_eager["ssm_state"], rtol=1e-4, atol=1e-4)
    assert ratio < 1.0
