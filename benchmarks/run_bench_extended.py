"""
Extended benchmark script for Mamba-3 decoding kernels.

Includes 8-way ablation:
  1. PyTorch eager
  2. torch.compile
  3. torch.compile + CUDA Graph
  4. Triton (basic)
  5. Triton fused (SSM + silu gate fused)
  6. Triton fused + CUDA Graph
  7. Triton full fused (split→rearrange→softplus→sigmoid→RMSNorm→RoPE→SSM→silu in one kernel)
  8. Triton full fused + CUDA Graph

Usage:
    python benchmarks/run_bench_extended.py --mode siso --batch-sizes 1,8,32,128
    python benchmarks/run_bench_extended.py --mode mimo --mimo-rank 4
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch

from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder


def warmup(fn, args, n_warmup=10):
    """Run function n_warmup times to warm up GPU."""
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()


def benchmark_single_step(fn, args, n_iters=100):
    """Benchmark a single decoding step (average latency)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / n_iters * 1000
    return avg_ms


def benchmark_latency_percentiles(fn, args, n_iters=200):
    """Benchmark single decoding step and return P50/P95/P99 latency.

    Uses CUDA Events for GPU-side timing to avoid CPU-GPU synchronization
    overhead polluting measurements for low-latency kernels (e.g. BS=1).
    Falls back to CPU timing if CUDA is not available.
    """
    use_cuda_events = torch.cuda.is_available()
    times = []

    if use_cuda_events:
        # Warmup once before measurement
        fn(*args)
        torch.cuda.synchronize()

        for _ in range(n_iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn(*args)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            times.append(elapsed_ms)
    else:
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(*args)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "p99_p50_ratio": float(np.percentile(times, 99) / np.percentile(times, 50)),
    }


def benchmark_autoregressive(decoder, model, batch_size, seq_len, n_iters=5):
    """Benchmark full autoregressive decoding for seq_len tokens."""
    times = []
    for _ in range(n_iters):
        state = decoder.init_state(batch_size, device="cuda")
        u = torch.randn(batch_size, model.d_model, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in range(seq_len):
            _, state = decoder.step(u, state)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return {
        "total_ms": sum(times) / len(times),
        "per_token_ms": sum(times) / len(times) / seq_len,
        "tokens_per_sec": seq_len / (sum(times) / len(times) / 1000),
    }


def benchmark_autoregressive_with_graph(decoder, model, batch_size, seq_len, n_iters=5):
    """Benchmark autoregressive decoding with CUDA Graph.

    For CUDA Graph backends, we warmup the graph once then replay it each step.
    """
    times = []
    for _ in range(n_iters):
        state = decoder.init_state(batch_size, device="cuda")
        decoder.warmup_cuda_graph(state, n_warmup=3)
        u = torch.randn(batch_size, model.d_model, device="cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for t in range(seq_len):
            decoder.step(u, state)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return {
        "total_ms": sum(times) / len(times),
        "per_token_ms": sum(times) / len(times) / seq_len,
        "tokens_per_sec": seq_len / (sum(times) / len(times) / 1000),
    }


def run_benchmark_suite(
    mode: str = "siso",
    batch_sizes: list = None,
    seq_len: int = 1024,
    mimo_rank: int = 4,
    d_model: int = 256,
    d_state: int = 64,
    headdim: int = 32,
    device: str = "cuda",
):
    """Run the full benchmark suite with 8-way ablation."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128]

    is_mimo = mode == "mimo"

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": mode,
            "mimo_rank": mimo_rank if is_mimo else 1,
            "seq_len": seq_len,
            "batch_sizes": batch_sizes,
            "d_model": d_model,
            "d_state": d_state,
            "headdim": headdim,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
        "single_step": {},
        "autoregressive": {},
    }

    print(f"{'='*80}")
    print(f"Mamba-3 Decoding Benchmark (8-way Ablation)")
    print(f"Mode: {mode} | MIMO rank: {mimo_rank if is_mimo else 1} | Seq len: {seq_len}")
    print(f"GPU: {results['config']['gpu_name']}")
    print(f"{'='*80}")
    print("\nConfigurations:")
    print("  1. PyTorch eager (baseline)")
    print("  2. torch.compile (Inductor optimization)")
    print("  3. torch.compile + CUDA Graph (optimal baseline)")
    print("  4. Triton basic (SSM recurrence only)")
    print("  5. Triton fused (SSM + silu gate fused)")
    print("  6. Triton fused + CUDA Graph")
    print("  7. Triton full fused (entire step in one kernel)")
    print("  8. Triton full fused + CUDA Graph (our best)")
    print()

    for bs in batch_sizes:
        print(f"\n{'='*40}")
        print(f"Batch size: {bs}")
        print(f"{'='*40}")

        model = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            device=device,
        ).to(device)

        # Create all decoders
        decoder_pytorch = Mamba3Decoder(model, use_triton=False)
        # NOTE: torch.compile with fullgraph=True can't handle dynamic batch
        # sizes (RecompileLimitExceeded), so create fresh per batch size.
        # Also, torch.compile's CUDAGraphTrees conflicts with explicit CUDA
        # Graph capture, so skip compile+graph entirely.
        decoder_triton = Mamba3Decoder(model, use_triton=True)
        decoder_triton_fused = Mamba3Decoder(model, use_triton_fused=True)
        decoder_triton_fused_graph = Mamba3Decoder(model, use_triton_fused=True, use_cuda_graph=True)
        decoder_triton_full_fused = Mamba3Decoder(model, use_triton_full_fused=True)
        decoder_triton_full_fused_graph = Mamba3Decoder(model, use_triton_full_fused=True, use_cuda_graph=True)

        u = torch.randn(bs, d_model, device=device)

        # ── 1. PyTorch eager ───────────────────────────────────────────
        state_pt = decoder_pytorch.init_state(bs, device=device)
        warmup(decoder_pytorch.step, (u, state_pt), n_warmup=5)
        ms_pt = benchmark_single_step(decoder_pytorch.step, (u, state_pt), n_iters=50)

        # ── 2. torch.compile ───────────────────────────────────────────
        # Create fresh compile decoder per batch size to avoid RecompileLimitExceeded
        ms_compile = float('nan')
        speedup_compile = float('nan')
        decoder_compile = None
        try:
            decoder_compile = Mamba3Decoder(model, use_triton=False, use_compile=True)
            state_compile = decoder_compile.init_state(bs, device=device)
            warmup(decoder_compile.step, (u, state_compile), n_warmup=20)
            ms_compile = benchmark_single_step(decoder_compile.step, (u, state_compile), n_iters=50)
            speedup_compile = ms_pt / ms_compile
        except Exception:
            pass  # RecompileLimitExceeded or other compile errors

        # ── 3. torch.compile + CUDA Graph ──────────────────────────────
        # Always skip: torch.compile's CUDAGraphTrees internal memory
        # management conflicts with explicit CUDA Graph capture.
        ms_compile_graph = float('nan')
        speedup_compile_graph = float('nan')

        # ── 4. Triton basic ────────────────────────────────────────────
        state_tri = decoder_triton.init_state(bs, device=device)
        warmup(decoder_triton.step, (u, state_tri), n_warmup=5)
        ms_tri = benchmark_single_step(decoder_triton.step, (u, state_tri), n_iters=50)

        # ── 5. Triton fused ────────────────────────────────────────────
        state_tri_fused = decoder_triton_fused.init_state(bs, device=device)
        warmup(decoder_triton_fused.step, (u, state_tri_fused), n_warmup=5)
        ms_tri_fused = benchmark_single_step(decoder_triton_fused.step, (u, state_tri_fused), n_iters=50)

        # ── 6. Triton fused + CUDA Graph ───────────────────────────────
        state_tri_fused_graph = decoder_triton_fused_graph.init_state(bs, device=device)
        decoder_triton_fused_graph.warmup_cuda_graph(state_tri_fused_graph, n_warmup=10)
        ms_tri_fused_graph = benchmark_single_step(decoder_triton_fused_graph.step, (u, state_tri_fused_graph), n_iters=50)

        # ── 7. Triton full fused ──────────────────────────────────────
        state_tri_full_fused = decoder_triton_full_fused.init_state(bs, device=device)
        warmup(decoder_triton_full_fused.step, (u, state_tri_full_fused), n_warmup=5)
        ms_tri_full_fused = benchmark_single_step(decoder_triton_full_fused.step, (u, state_tri_full_fused), n_iters=50)

        # ── 8. Triton full fused + CUDA Graph ─────────────────────────
        state_tri_full_fused_graph = decoder_triton_full_fused_graph.init_state(bs, device=device)
        decoder_triton_full_fused_graph.warmup_cuda_graph(state_tri_full_fused_graph, n_warmup=10)
        ms_tri_full_fused_graph = benchmark_single_step(decoder_triton_full_fused_graph.step, (u, state_tri_full_fused_graph), n_iters=50)

        # Calculate speedups
        speedup_tri = ms_pt / ms_tri
        speedup_tri_fused = ms_pt / ms_tri_fused
        speedup_tri_fused_graph = ms_pt / ms_tri_fused_graph
        speedup_tri_full_fused = ms_pt / ms_tri_full_fused
        speedup_tri_full_fused_graph = ms_pt / ms_tri_full_fused_graph

        # compile speedup already calculated above (or NaN)
        # compile+graph always NaN (CUDAGraphTrees conflict)

        bs_results = {
            "pytorch_ms": ms_pt,
            "compile_ms": ms_compile,
            "compile_cuda_graph_ms": ms_compile_graph,
            "triton_ms": ms_tri,
            "triton_fused_ms": ms_tri_fused,
            "triton_fused_cuda_graph_ms": ms_tri_fused_graph,
            "triton_full_fused_ms": ms_tri_full_fused,
            "triton_full_fused_cuda_graph_ms": ms_tri_full_fused_graph,
            "speedup_compile": speedup_compile,
            "speedup_compile_cuda_graph": speedup_compile_graph,
            "speedup_triton": speedup_tri,
            "speedup_triton_fused": speedup_tri_fused,
            "speedup_triton_fused_cuda_graph": speedup_tri_fused_graph,
            "speedup_triton_full_fused": speedup_tri_full_fused,
            "speedup_triton_full_fused_cuda_graph": speedup_tri_full_fused_graph,
        }
        results["single_step"][str(bs)] = bs_results

        print(f"\n  Single-step latency:")
        print(f"    PyTorch eager:               {ms_pt:.4f} ms")
        compile_str = f"{ms_compile:.4f} ms ({speedup_compile:.2f}x)" if not np.isnan(ms_compile) else "N/A (RecompileLimitExceeded)"
        print(f"    torch.compile:               {compile_str}")
        print(f"    torch.compile + CUDA Graph:  N/A (CUDAGraphTrees conflict)")
        print(f"    Triton basic:                {ms_tri:.4f} ms ({speedup_tri:.2f}x)")
        print(f"    Triton fused:                {ms_tri_fused:.4f} ms ({speedup_tri_fused:.2f}x)")
        print(f"    Triton fused + CUDA Graph:   {ms_tri_fused_graph:.4f} ms ({speedup_tri_fused_graph:.2f}x)")
        print(f"    Triton full fused:           {ms_tri_full_fused:.4f} ms ({speedup_tri_full_fused:.2f}x)")
        print(f"    Triton full fused + Graph:   {ms_tri_full_fused_graph:.4f} ms ({speedup_tri_full_fused_graph:.2f}x)")

        # ── Latency percentiles ────────────────────────────────────────
        print(f"\n  Latency percentiles (P50/P95/P99, n=200):")
        pct_results = {}
        
        # PyTorch eager
        state = decoder_pytorch.init_state(bs, device=device)
        warmup(decoder_pytorch.step, (u, state), n_warmup=10)
        pstats = benchmark_latency_percentiles(decoder_pytorch.step, (u, state), n_iters=200)
        pct_results["pytorch"] = pstats
        print(f"    PyTorch:            P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")

        # torch.compile
        if decoder_compile is not None:
            state = decoder_compile.init_state(bs, device=device)
            warmup(decoder_compile.step, (u, state), n_warmup=10)
            pstats = benchmark_latency_percentiles(decoder_compile.step, (u, state), n_iters=200)
            pct_results["compile"] = pstats
            print(f"    torch.compile:      P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")
        else:
            pct_results["compile"] = None
            print(f"    torch.compile:      SKIPPED (compile failed)")

        # torch.compile + CUDA Graph
        pct_results["compile_cuda_graph"] = None
        print(f"    compile+CUDA Graph: SKIPPED (CUDAGraphTrees conflict)")

        # Triton basic
        state = decoder_triton.init_state(bs, device=device)
        warmup(decoder_triton.step, (u, state), n_warmup=10)
        pstats = benchmark_latency_percentiles(decoder_triton.step, (u, state), n_iters=200)
        pct_results["triton"] = pstats
        print(f"    Triton basic:       P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")

        # Triton fused
        state = decoder_triton_fused.init_state(bs, device=device)
        warmup(decoder_triton_fused.step, (u, state), n_warmup=10)
        pstats = benchmark_latency_percentiles(decoder_triton_fused.step, (u, state), n_iters=200)
        pct_results["triton_fused"] = pstats
        print(f"    Triton fused:       P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")

        # Triton fused + CUDA Graph
        state = decoder_triton_fused_graph.init_state(bs, device=device)
        decoder_triton_fused_graph.warmup_cuda_graph(state, n_warmup=10)
        pstats = benchmark_latency_percentiles(decoder_triton_fused_graph.step, (u, state), n_iters=200)
        pct_results["triton_fused_cuda_graph"] = pstats
        print(f"    Triton fused+Graph: P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")

        # Triton full fused
        state = decoder_triton_full_fused.init_state(bs, device=device)
        warmup(decoder_triton_full_fused.step, (u, state), n_warmup=10)
        pstats = benchmark_latency_percentiles(decoder_triton_full_fused.step, (u, state), n_iters=200)
        pct_results["triton_full_fused"] = pstats
        print(f"    Triton full fused:  P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")

        # Triton full fused + CUDA Graph
        state = decoder_triton_full_fused_graph.init_state(bs, device=device)
        decoder_triton_full_fused_graph.warmup_cuda_graph(state, n_warmup=10)
        pstats = benchmark_latency_percentiles(decoder_triton_full_fused_graph.step, (u, state), n_iters=200)
        pct_results["triton_full_fused_cuda_graph"] = pstats
        print(f"    Triton full+Graph:  P50={pstats['p50_ms']:.4f} | P95={pstats['p95_ms']:.4f} | P99={pstats['p99_ms']:.4f} | P99/P50={pstats['p99_p50_ratio']:.2f}x")

        results["single_step"][str(bs)]["percentiles"] = pct_results

        # ── Ensure CUDA state is clean before AR benchmark ─────────────
        # Reset CUDA allocator state after percentile benchmarks
        # (torch.compile's CUDAGraphTrees can leave stale state)
        torch.cuda.synchronize()

        # ── Autoregressive benchmark (8-way ablation) ─────────────────
        # NOTE: Skip torch.compile in AR benchmark because torch.compile's
        # internal CUDAGraphTrees memory management conflicts with explicit
        # CUDA Graph capture, leaving the CUDA allocator in a bad state.
        print(f"\n  Autoregressive decoding ({seq_len} tokens):")
        ar_results = {}

        # 1. PyTorch eager
        ar_pt = benchmark_autoregressive(decoder_pytorch, model, bs, seq_len, n_iters=3)
        ar_results["pytorch"] = ar_pt
        print(f"    PyTorch eager:               {ar_pt['per_token_ms']:.4f} ms/tok  ({ar_pt['tokens_per_sec']:.1f} tok/s)")

        # 2. torch.compile (skip - CUDAGraphTrees conflicts with CUDA Graph)
        # Instead, run compile AR in a separate subprocess or skip it
        ar_results["compile"] = None
        ar_results["compile_cuda_graph"] = None
        print(f"    torch.compile:               SKIPPED (CUDAGraphTrees conflict)")
        print(f"    torch.compile + CUDA Graph:  SKIPPED")

        # 4. Triton basic
        ar_tri = benchmark_autoregressive(decoder_triton, model, bs, seq_len, n_iters=3)
        ar_results["triton"] = ar_tri
        print(f"    Triton basic:                {ar_tri['per_token_ms']:.4f} ms/tok  ({ar_tri['tokens_per_sec']:.1f} tok/s)")

        # 5. Triton fused
        ar_tri_fused = benchmark_autoregressive(decoder_triton_fused, model, bs, seq_len, n_iters=3)
        ar_results["triton_fused"] = ar_tri_fused
        print(f"    Triton fused:                {ar_tri_fused['per_token_ms']:.4f} ms/tok  ({ar_tri_fused['tokens_per_sec']:.1f} tok/s)")

        # 6. Triton fused + CUDA Graph (fresh decoder)
        decoder_tri_fused_graph_ar = Mamba3Decoder(model, use_triton_fused=True, use_cuda_graph=True)
        ar_tri_fused_graph = benchmark_autoregressive_with_graph(decoder_tri_fused_graph_ar, model, bs, seq_len, n_iters=3)
        ar_results["triton_fused_cuda_graph"] = ar_tri_fused_graph
        print(f"    Triton fused + CUDA Graph:   {ar_tri_fused_graph['per_token_ms']:.4f} ms/tok  ({ar_tri_fused_graph['tokens_per_sec']:.1f} tok/s)")

        # 7. Triton full fused
        ar_tri_full_fused = benchmark_autoregressive(decoder_triton_full_fused, model, bs, seq_len, n_iters=3)
        ar_results["triton_full_fused"] = ar_tri_full_fused
        print(f"    Triton full fused:           {ar_tri_full_fused['per_token_ms']:.4f} ms/tok  ({ar_tri_full_fused['tokens_per_sec']:.1f} tok/s)")

        # 8. Triton full fused + CUDA Graph (fresh decoder)
        decoder_tri_full_fused_graph_ar = Mamba3Decoder(model, use_triton_full_fused=True, use_cuda_graph=True)
        ar_tri_full_fused_graph = benchmark_autoregressive_with_graph(decoder_tri_full_fused_graph_ar, model, bs, seq_len, n_iters=3)
        ar_results["triton_full_fused_cuda_graph"] = ar_tri_full_fused_graph
        print(f"    Triton full fused + Graph:   {ar_tri_full_fused_graph['per_token_ms']:.4f} ms/tok  ({ar_tri_full_fused_graph['tokens_per_sec']:.1f} tok/s)")

        # Calculate autoregressive speedups vs PyTorch
        ar_speedups = {
            "vs_pytorch": {
                "triton": ar_pt["total_ms"] / ar_tri["total_ms"],
                "triton_fused": ar_pt["total_ms"] / ar_tri_fused["total_ms"],
                "triton_fused_cuda_graph": ar_pt["total_ms"] / ar_tri_fused_graph["total_ms"],
                "triton_full_fused": ar_pt["total_ms"] / ar_tri_full_fused["total_ms"],
                "triton_full_fused_cuda_graph": ar_pt["total_ms"] / ar_tri_full_fused_graph["total_ms"],
            }
        }
        if ar_results.get("compile") is not None:
            ar_speedups["vs_pytorch"]["compile"] = ar_pt["total_ms"] / ar_results["compile"]["total_ms"]
        if ar_results.get("compile_cuda_graph") is not None:
            ar_speedups["vs_pytorch"]["compile_cuda_graph"] = ar_pt["total_ms"] / ar_results["compile_cuda_graph"]["total_ms"]

            # Also vs compile+graph (optimal PyTorch baseline)
            ar_speedups["vs_compile_graph"] = {
                "triton": ar_results["compile_cuda_graph"]["total_ms"] / ar_tri["total_ms"],
                "triton_fused": ar_results["compile_cuda_graph"]["total_ms"] / ar_tri_fused["total_ms"],
                "triton_fused_cuda_graph": ar_results["compile_cuda_graph"]["total_ms"] / ar_tri_fused_graph["total_ms"],
                "triton_full_fused": ar_results["compile_cuda_graph"]["total_ms"] / ar_tri_full_fused["total_ms"],
                "triton_full_fused_cuda_graph": ar_results["compile_cuda_graph"]["total_ms"] / ar_tri_full_fused_graph["total_ms"],
            }

        ar_results["speedups"] = ar_speedups
        results["autoregressive"][str(bs)] = ar_results

        # Print speedup summary
        print(f"\n  Autoregressive speedups vs PyTorch:")
        for name, speedup in ar_speedups["vs_pytorch"].items():
            print(f"    {name}: {speedup:.2f}x")

    # Save results
    output_path = f"benchmarks/results/{mode}_r{mimo_rank}_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mamba-3 Decoding Extended Benchmark")
    parser.add_argument("--mode", choices=["siso", "mimo", "both"], default="both")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    modes = ["siso", "mimo"] if args.mode == "both" else [args.mode]

    for mode in modes:
        run_benchmark_suite(
            mode=mode,
            batch_sizes=batch_sizes,
            seq_len=args.seq_len,
            mimo_rank=args.mimo_rank,
            d_model=args.d_model,
            d_state=args.d_state,
            headdim=args.headdim,
            device=args.device,
        )


if __name__ == "__main__":
    main()
