"""
Latency breakdown profiling for Mamba-3 decoding step.

Uses torch.profiler to show where time is spent in a single decode step:
  - Kernel launch overhead (CPU-side)
  - Actual GPU compute time
  - Per-operator breakdown

Usage:
    python benchmarks/profile_step.py
    python benchmarks/profile_step.py --mode mimo --d-model 512
"""

import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder


def profile_decoder_step(
    mode: str = "mimo",
    d_model: int = 512,
    d_state: int = 128,
    headdim: int = 64,
    mimo_rank: int = 4,
    batch_size: int = 4,
    device: str = "cuda",
    n_warmup: int = 10,
    n_profile: int = 5,
    save_trace: bool = False,
):
    """Profile a single decode step and print latency breakdown."""
    is_mimo = mode == "mimo"

    model = Mamba3(
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        device=device,
    ).to(device)

    u = torch.randn(batch_size, d_model, device=device)

    # ── Profile PyTorch eager ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Profiling: PyTorch Eager | Mode={mode} | d_model={d_model} | bs={batch_size}")
    print(f"{'='*70}")

    decoder_pt = Mamba3Decoder(model, use_triton=False)
    state_pt = decoder_pt.init_state(batch_size, device=device)

    # Warmup
    for _ in range(n_warmup):
        decoder_pt.step(u, state_pt)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof_pt:
        for _ in range(n_profile):
            decoder_pt.step(u, state_pt)

    sort_by_cuda = "device_time_total" if hasattr(list(prof_pt.key_averages())[0], "device_time_total") else "cuda_time_total"

    print("\n--- PyTorch Eager: Top-20 ops by GPU time ---")
    print(prof_pt.key_averages().table(sort_by=sort_by_cuda, row_limit=20))

    print("\n--- PyTorch Eager: Top-20 ops by CPU time ---")
    print(prof_pt.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Compute CPU vs GPU time ratio
    cpu_total = sum(e.cpu_time_total for e in prof_pt.key_averages())
    gpu_total = sum(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) for e in prof_pt.key_averages())
    print(f"\n--- CPU vs GPU breakdown ---")
    print(f"  Total CPU time:  {cpu_total/1000:.3f} ms")
    print(f"  Total GPU time:  {gpu_total/1000:.3f} ms")
    print(f"  CPU/GPU ratio:   {cpu_total/gpu_total:.2f}x")
    print(f"  Launch overhead: ~{(1 - gpu_total/cpu_total)*100:.1f}% of wall-clock time")

    if save_trace:
        prof_pt.export_chrome_trace("benchmarks/results/trace_pytorch_eager.json")

    # ── Profile Triton fused ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Profiling: Triton Fused | Mode={mode} | d_model={d_model} | bs={batch_size}")
    print(f"{'='*70}")

    decoder_tri = Mamba3Decoder(model, use_triton=True)
    state_tri = decoder_tri.init_state(batch_size, device=device)

    # Warmup
    for _ in range(n_warmup):
        decoder_tri.step(u, state_tri)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof_tri:
        for _ in range(n_profile):
            decoder_tri.step(u, state_tri)

    print("\n--- Triton Fused: Top-20 ops by GPU time ---")
    print(prof_tri.key_averages().table(sort_by=sort_by_cuda, row_limit=20))

    print("\n--- Triton Fused: Top-20 ops by CPU time ---")
    print(prof_tri.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    cpu_total_tri = sum(e.cpu_time_total for e in prof_tri.key_averages())
    gpu_total_tri = sum(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) for e in prof_tri.key_averages())
    print(f"\n--- CPU vs GPU breakdown ---")
    print(f"  Total CPU time:  {cpu_total_tri/1000:.3f} ms")
    print(f"  Total GPU time:  {gpu_total_tri/1000:.3f} ms")
    print(f"  CPU/GPU ratio:   {cpu_total_tri/gpu_total_tri:.2f}x")

    if save_trace:
        prof_tri.export_chrome_trace("benchmarks/results/trace_triton_fused.json")

    # ── Profile CUDA Graph ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Profiling: Triton + CUDA Graph | Mode={mode} | d_model={d_model} | bs={batch_size}")
    print(f"{'='*70}")

    decoder_graph = Mamba3Decoder(model, use_triton=True, use_cuda_graph=True)
    state_graph = decoder_graph.init_state(batch_size, device=device)
    decoder_graph.warmup_cuda_graph(state_graph, n_warmup=n_warmup)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof_graph:
        for _ in range(n_profile):
            decoder_graph.step(u, state_graph)

    print("\n--- CUDA Graph: Top-10 ops by GPU time ---")
    print(prof_graph.key_averages().table(sort_by=sort_by_cuda, row_limit=10))

    cpu_total_graph = sum(e.cpu_time_total for e in prof_graph.key_averages())
    gpu_total_graph = sum(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) for e in prof_graph.key_averages())
    print(f"\n--- CPU vs GPU breakdown ---")
    print(f"  Total CPU time:  {cpu_total_graph/1000:.3f} ms")
    print(f"  Total GPU time:  {gpu_total_graph/1000:.3f} ms")
    print(f"  CPU/GPU ratio:   {cpu_total_graph/gpu_total_graph:.2f}x")

    # ── Summary comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY: Step latency breakdown comparison")
    print(f"{'='*70}")
    print(f"{'Backend':<20s} {'CPU (ms)':<12s} {'GPU (ms)':<12s} {'CPU/GPU':<10s}")
    print(f"{'-'*54}")
    print(f"{'PyTorch eager':<20s} {cpu_total/1000:<12.3f} {gpu_total/1000:<12.3f} {cpu_total/gpu_total:<10.2f}")
    print(f"{'Triton fused':<20s} {cpu_total_tri/1000:<12.3f} {gpu_total_tri/1000:<12.3f} {cpu_total_tri/gpu_total_tri:<10.2f}")
    print(f"{'Triton+CUDA Graph':<20s} {cpu_total_graph/1000:<12.3f} {gpu_total_graph/1000:<12.3f} {cpu_total_graph/gpu_total_graph:<10.2f}")

    print(f"\nKey insight: Launch overhead (CPU time) dominates decode latency.")
    print(f"CUDA Graph eliminates launch overhead by replaying a captured graph.")
    print(f"")
    print(f"Ablation breakdown:")
    print(f"  PyTorch eager → Triton fused:  reduces GPU compute via kernel fusion")
    print(f"  Triton fused → +CUDA Graph:    eliminates CPU launch overhead")
    print(f"  PyTorch eager → torch.compile: fuses ops via Inductor, reduces launches")


def main():
    parser = argparse.ArgumentParser(description="Mamba-3 Step Latency Profiling")
    parser.add_argument("--mode", choices=["siso", "mimo"], default="mimo")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-trace", action="store_true",
                        help="Save Chrome trace files for viewing in chrome://tracing")
    args = parser.parse_args()

    profile_decoder_step(
        mode=args.mode,
        d_model=args.d_model,
        d_state=args.d_state,
        headdim=args.headdim,
        mimo_rank=args.mimo_rank,
        batch_size=args.batch_size,
        save_trace=args.save_trace,
    )


if __name__ == "__main__":
    main()
