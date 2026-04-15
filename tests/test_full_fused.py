"""
Correctness test for the full-step fused Triton decode kernel.

Compares output of the full-fused kernel against the PyTorch reference
implementation to verify numerical equivalence.
"""

import torch
import sys

from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder


def test_full_fused_correctness(mode="siso", batch_size=2, d_model=256, d_state=64,
                                 headdim=32, mimo_rank=4, device="cuda"):
    """Test that full-fused kernel produces same results as PyTorch reference."""
    is_mimo = mode == "mimo"

    model = Mamba3(
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        device=device,
    ).to(device)

    # Reference decoder (PyTorch eager)
    decoder_ref = Mamba3Decoder(model, use_triton=False)
    state_ref = decoder_ref.init_state(batch_size, device=device)

    # Full-fused decoder
    decoder_fused = Mamba3Decoder(model, use_triton_full_fused=True)
    state_fused = decoder_fused.init_state(batch_size, device=device)

    # Same initial state
    state_fused["angle_state"].copy_(state_ref["angle_state"])
    state_fused["ssm_state"].copy_(state_ref["ssm_state"])
    state_fused["bx_prev"].copy_(state_ref["bx_prev"])

    u = torch.randn(batch_size, d_model, device=device)

    # Run reference
    with torch.no_grad():
        out_ref, state_ref = decoder_ref.step(u, state_ref)

    # Run fused
    with torch.no_grad():
        out_fused, state_fused = decoder_fused.step(u, state_fused)

    # Compare outputs
    out_diff = (out_ref.float() - out_fused.float()).abs()
    max_diff = out_diff.max().item()
    rel_diff = (out_diff / (out_ref.float().abs() + 1e-6)).max().item()

    print(f"\n{'='*60}")
    print(f"Full-Fused Kernel Correctness Test: {mode.upper()}")
    print(f"{'='*60}")
    print(f"  Batch size: {batch_size}")
    print(f"  d_model: {d_model}, d_state: {d_state}, headdim: {headdim}")
    if is_mimo:
        print(f"  MIMO rank: {mimo_rank}")
    print(f"\n  Output comparison:")
    print(f"    Max absolute diff:  {max_diff:.2e}")
    print(f"    Max relative diff:  {rel_diff:.2e}")
    print(f"    Mean absolute diff: {out_diff.mean().item():.2e}")

    # Compare states
    angle_diff = (state_ref["angle_state"].float() - state_fused["angle_state"].float()).abs().max().item()
    ssm_diff = (state_ref["ssm_state"].float() - state_fused["ssm_state"].float()).abs().max().item()
    bx_diff = (state_ref["bx_prev"].float() - state_fused["bx_prev"].float()).abs().max().item()
    print(f"\n  State comparison:")
    print(f"    angle_state max diff: {angle_diff:.2e}")
    print(f"    ssm_state max diff:   {ssm_diff:.2e}")
    print(f"    bx_prev max diff:     {bx_diff:.2e}")

    # Run multiple steps to check state accumulation
    print(f"\n  Multi-step test (10 steps):")
    for step in range(10):
        u = torch.randn(batch_size, d_model, device=device)
        with torch.no_grad():
            out_ref, state_ref = decoder_ref.step(u, state_ref)
            out_fused, state_fused = decoder_fused.step(u, state_fused)

        step_diff = (out_ref.float() - out_fused.float()).abs().max().item()
        if step % 3 == 0 or step == 9:
            print(f"    Step {step}: max diff = {step_diff:.2e}")

    # Final diff after 10 steps
    final_diff = (out_ref.float() - out_fused.float()).abs().max().item()
    final_angle_diff = (state_ref["angle_state"].float() - state_fused["angle_state"].float()).abs().max().item()
    final_ssm_diff = (state_ref["ssm_state"].float() - state_fused["ssm_state"].float()).abs().max().item()

    print(f"\n  Final state after 10 steps:")
    print(f"    Output max diff:     {final_diff:.2e}")
    print(f"    angle_state max diff: {final_angle_diff:.2e}")
    print(f"    ssm_state max diff:   {final_ssm_diff:.2e}")

    # Verdict
    tol = 1e-3  # tolerance for float32 operations
    passed = max_diff < tol and final_diff < tol * 10  # allow slightly more error after 10 steps
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n  Verdict: {status}")
    print(f"{'='*60}")

    return passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        sys.exit(0)

    torch.manual_seed(42)

    # Test SISO
    siso_passed = test_full_fused_correctness(mode="siso")

    # Test MIMO
    mimo_passed = test_full_fused_correctness(mode="mimo")

    print(f"\n{'='*60}")
    print(f"Overall: SISO={'PASSED' if siso_passed else 'FAILED'}, MIMO={'PASSED' if mimo_passed else 'FAILED'}")
    print(f"{'='*60}")

    sys.exit(0 if (siso_passed and mimo_passed) else 1)
