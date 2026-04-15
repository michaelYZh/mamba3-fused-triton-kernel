"""
Multi-step accuracy drift test for Mamba-3 Triton kernels.

Runs 1000+ decode steps and compares the hidden state drift between
Triton kernels and PyTorch reference. This verifies that the Triton
kernels maintain numerical stability over long sequences.

Usage:
    python benchmarks/test_accuracy_drift.py --mode siso --num-steps 1000
    python benchmarks/test_accuracy_drift.py --mode mimo --num-steps 1000
    python benchmarks/test_accuracy_drift.py --mode both --num-steps 2000
"""

import argparse
import json
import sys
from datetime import datetime

import torch

from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder


def run_accuracy_drift_test(
    mode: str = "siso",
    batch_size: int = 4,
    num_steps: int = 1000,
    d_model: int = 256,
    d_state: int = 64,
    headdim: int = 32,
    mimo_rank: int = 4,
    device: str = "cuda",
    seed: int = 42,
):
    """Run multi-step accuracy drift test comparing all backends to PyTorch reference."""
    is_mimo = mode == "mimo"

    torch.manual_seed(seed)

    model = Mamba3(
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        device=device,
    ).to(device)

    print(f"{'='*80}")
    print(f"Multi-step Accuracy Drift Test: {mode.upper()}")
    print(f"{'='*80}")
    print(f"  Config: d_model={d_model}, d_state={d_state}, headdim={headdim}")
    if is_mimo:
        print(f"  MIMO rank: {mimo_rank}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num steps: {num_steps}")
    print(f"  Seed: {seed}")
    print()

    # ── Create all decoders ────────────────────────────────────────────
    backend_configs = [
        ("PyTorch eager",       dict(use_triton=False)),
        ("torch.compile",       dict(use_triton=False, use_compile=True)),
        ("Triton basic",        dict(use_triton=True)),
        ("Triton fused",        dict(use_triton_fused=True)),
        ("Triton full fused",   dict(use_triton_full_fused=True)),
    ]

    decoders = {}
    states = {}
    for name, kwargs in backend_configs:
        decoders[name] = Mamba3Decoder(model, **kwargs)

    # Initialize all states with the same seed
    torch.manual_seed(seed)
    for name in decoders:
        states[name] = decoders[name].init_state(batch_size, device=device)

    # ── Generate fixed input sequence ──────────────────────────────────
    torch.manual_seed(seed + 1)
    inputs = [torch.randn(batch_size, d_model, device=device) for _ in range(num_steps)]

    # ── Run all backends ───────────────────────────────────────────────
    # Track error at checkpoints
    checkpoints = [1, 10, 50, 100, 200, 500, 1000, 2000]
    checkpoints = [c for c in checkpoints if c <= num_steps]
    if num_steps not in checkpoints:
        checkpoints.append(num_steps)

    # Store drift data
    drift_data = {name: {"output_abs_err": [], "output_rel_err": [],
                          "ssm_state_abs_err": [], "angle_state_abs_err": []}
                  for name in decoders if name != "PyTorch eager"}

    ref_name = "PyTorch eager"
    ref_outputs = []  # store reference outputs at checkpoints

    print(f"Running {num_steps} decode steps...")

    with torch.no_grad():
        for step in range(num_steps):
            u = inputs[step]

            # Step all backends
            for name in decoders:
                _, states[name] = decoders[name].step(u, states[name])

            # Check error at checkpoints
            if (step + 1) in checkpoints:
                ref_state = states[ref_name]
                print(f"\n  Step {step + 1}:")
                for name in decoders:
                    if name == ref_name:
                        continue
                    s = states[name]
                    # Output error (use last step's output from state comparison)
                    ssm_err = (ref_state["ssm_state"].float() - s["ssm_state"].float()).abs()
                    ssm_max = ssm_err.max().item()
                    ssm_mean = ssm_err.mean().item()
                    angle_err = (ref_state["angle_state"].float() - s["angle_state"].float()).abs()
                    angle_max = angle_err.max().item()
                    bx_err = (ref_state["bx_prev"].float() - s["bx_prev"].float()).abs()
                    bx_max = bx_err.max().item()

                    drift_data[name]["ssm_state_abs_err"].append(ssm_max)
                    drift_data[name]["angle_state_abs_err"].append(angle_max)

                    print(f"    {name:25s}: ssm_max={ssm_max:.2e}  ssm_mean={ssm_mean:.2e}  "
                          f"angle_max={angle_max:.2e}  bx_max={bx_max:.2e}")

    # ── Run a separate output-comparison pass ──────────────────────────
    # We need to compare outputs at each checkpoint, so re-run with output tracking
    print(f"\n{'='*80}")
    print(f"Output comparison pass (re-running with output tracking)...")
    print(f"{'='*80}")

    # Reset states
    torch.manual_seed(seed)
    for name in decoders:
        states[name] = decoders[name].init_state(batch_size, device=device)

    output_drift = {name: [] for name in decoders if name != ref_name}

    with torch.no_grad():
        for step in range(num_steps):
            u = inputs[step]

            # Step all backends and collect outputs
            outputs = {}
            for name in decoders:
                out, states[name] = decoders[name].step(u, states[name])
                outputs[name] = out

            # Compare at checkpoints
            if (step + 1) in checkpoints:
                ref_out = outputs[ref_name]
                for name in decoders:
                    if name == ref_name:
                        continue
                    diff = (ref_out.float() - outputs[name].float()).abs()
                    abs_max = diff.max().item()
                    abs_mean = diff.mean().item()
                    rel_max = (diff / (ref_out.float().abs() + 1e-6)).max().item()

                    output_drift[name].append({
                        "step": step + 1,
                        "abs_max": abs_max,
                        "abs_mean": abs_mean,
                        "rel_max": rel_max,
                    })

                    print(f"  Step {step + 1:5d} | {name:25s}: "
                          f"abs_max={abs_max:.2e}  abs_mean={abs_mean:.2e}  rel_max={rel_max:.2e}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")

    final_output_data = {name: output_drift[name][-1] for name in output_drift}

    all_pass = True
    for name, data in final_output_data.items():
        # Tolerance: abs_max should be < 1e-3 for float32 with moderate accumulation
        # After 1000 steps, we allow up to ~1e-2 due to accumulation
        tol = 1e-2 if num_steps >= 500 else 1e-3
        passed = data["abs_max"] < tol
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s}: abs_max={data['abs_max']:.2e}  rel_max={data['rel_max']:.2e}  [{status}]")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": mode,
            "mimo_rank": mimo_rank if is_mimo else 1,
            "batch_size": batch_size,
            "num_steps": num_steps,
            "d_model": d_model,
            "d_state": d_state,
            "headdim": headdim,
            "seed": seed,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "output_drift": output_drift,
        "state_drift": drift_data,
        "checkpoints": checkpoints,
        "summary": final_output_data,
    }

    output_path = f"benchmarks/results/accuracy_drift_{mode}_r{mimo_rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Mamba-3 Multi-step Accuracy Drift Test")
    parser.add_argument("--mode", choices=["siso", "mimo", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Number of decode steps to run")
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        sys.exit(0)

    modes = ["siso", "mimo"] if args.mode == "both" else [args.mode]
    all_pass = True

    for mode in modes:
        passed = run_accuracy_drift_test(
            mode=mode,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            d_model=args.d_model,
            d_state=args.d_state,
            headdim=args.headdim,
            mimo_rank=args.mimo_rank,
            device=args.device,
            seed=args.seed,
        )
        all_pass = all_pass and passed

    print(f"\n{'='*80}")
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*80}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
