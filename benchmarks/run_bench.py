"""
Benchmark script for Mamba-3 decoding kernels.

4-way ablation: PyTorch eager / torch.compile / Triton / Triton+CUDA Graph.
Also reports P50/P95/P99 latency percentiles.

Usage:
    python benchmarks/run_bench.py --mode siso --batch-sizes 1,8,32,128 --seq-len 1024
    python benchmarks/run_bench.py --mode mimo --mimo-rank 4 --batch-sizes 1,8,32,128
    python benchmarks/run_bench.py --mode mimo --no-compile  # skip torch.compile
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
    """Benchmark single decoding step and return P50/P95/P99 latency."""
    times = []
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
    }


def benchmark_autoregressive(decoder, model, batch_size, seq_len, n_iters=10):
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


def run_benchmark_suite(
    mode: str = "siso",
    batch_sizes: list = None,
    seq_len: int = 1024,
    mimo_rank: int = 4,
    d_model: int = 256,
    d_state: int = 64,
    headdim: int = 32,
    device: str = "cuda",
    with_compile: bool = True,
    with_percentiles: bool = True,
):
    """Run the full benchmark suite with 4-way ablation."""
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

    print(f"{'='*60}")
    print(f"Mamba-3 Decoding Benchmark (4-way Ablation)")
    print(f"Mode: {mode} | MIMO rank: {mimo_rank if is_mimo else 1} | Seq len: {seq_len}")
    print(f"GPU: {results['config']['gpu_name']}")
    print(f"{'='*60}")

    for bs in batch_sizes:
        print(f"\nBatch size: {bs}")

        model = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            device=device,
        ).to(device)

        decoder_pytorch = Mamba3Decoder(model, use_triton=False)
        decoder_triton = Mamba3Decoder(model, use_triton=True)
        decoder_graph = Mamba3Decoder(model, use_triton=True, use_cuda_graph=True)

        decoder_compile = None
        if with_compile:
            decoder_compile = Mamba3Decoder(model, use_triton=False, use_compile=True)

        state_pt = decoder_pytorch.init_state(bs, device=device)
        state_tri = decoder_triton.init_state(bs, device=device)
        state_graph = decoder_graph.init_state(bs, device=device)

        u = torch.randn(bs, d_model, device=device)

        # ── PyTorch eager ──────────────────────────────────────────────
        warmup(decoder_pytorch.step, (u, state_pt), n_warmup=5)
        ms_pt = benchmark_single_step(decoder_pytorch.step, (u, state_pt), n_iters=50)

        # ── torch.compile ──────────────────────────────────────────────
        ms_compile = None
        if decoder_compile is not None:
            state_compile = decoder_compile.init_state(bs, device=device)
            warmup(decoder_compile.step, (u, state_compile), n_warmup=20)
            ms_compile = benchmark_single_step(decoder_compile.step, (u, state_compile), n_iters=50)

        # ── Triton fused ───────────────────────────────────────────────
        warmup(decoder_triton.step, (u, state_tri), n_warmup=5)
        ms_tri = benchmark_single_step(decoder_triton.step, (u, state_tri), n_iters=50)

        # ── CUDA Graph ─────────────────────────────────────────────────
        decoder_graph.warmup_cuda_graph(state_graph, n_warmup=10)
        ms_graph = benchmark_single_step(decoder_graph.step, (u, state_graph), n_iters=50)

        speedup_tri = ms_pt / ms_tri if ms_tri > 0 else float('inf')
        speedup_graph = ms_pt / ms_graph if ms_graph > 0 else float('inf')
        speedup_compile = ms_pt / ms_compile if ms_compile and ms_compile > 0 else None

        bs_results = {
            "pytorch_ms": ms_pt,
            "triton_ms": ms_tri,
            "cuda_graph_ms": ms_graph,
            "speedup_triton": speedup_tri,
            "speedup_cuda_graph": speedup_graph,
        }
        if ms_compile is not None:
            bs_results["compile_ms"] = ms_compile
            bs_results["speedup_compile"] = speedup_compile
        results["single_step"][str(bs)] = bs_results

        compile_str = f" | Compile: {ms_compile:.4f} ms ({speedup_compile:.2f}x)" if ms_compile else ""
        print(f"  PyTorch: {ms_pt:.4f} ms | Triton: {ms_tri:.4f} ms ({speedup_tri:.2f}x) | "
              f"CUDA Graph: {ms_graph:.4f} ms ({speedup_graph:.2f}x){compile_str}")

        # ── Latency percentiles ────────────────────────────────────────
        if with_percentiles:
            print(f"  Latency percentiles (n=200):")
            pct_results = {}
            for name, dec, needs_graph in [
                ("PyTorch", decoder_pytorch, False),
                ("Triton", decoder_triton, False),
                ("CUDA Graph", decoder_graph, True),
            ]:
                st = dec.init_state(bs, device=device)
                if needs_graph:
                    dec.warmup_cuda_graph(st, n_warmup=10)
                else:
                    warmup(dec.step, (u, st), n_warmup=10)
                pstats = benchmark_latency_percentiles(dec.step, (u, st), n_iters=200)
                pct_results[name] = pstats
                print(f"    {name:12s}: P50={pstats['p50_ms']:.4f} ms | "
                      f"P95={pstats['p95_ms']:.4f} ms | P99={pstats['p99_ms']:.4f} ms")

            if decoder_compile is not None:
                state_c = decoder_compile.init_state(bs, device=device)
                warmup(decoder_compile.step, (u, state_c), n_warmup=10)
                pstats = benchmark_latency_percentiles(decoder_compile.step, (u, state_c), n_iters=200)
                pct_results["Compile"] = pstats
                print(f"    {'Compile':12s}: P50={pstats['p50_ms']:.4f} ms | "
                      f"P95={pstats['p95_ms']:.4f} ms | P99={pstats['p99_ms']:.4f} ms")

            results["single_step"][str(bs)]["percentiles"] = pct_results

        # ── Autoregressive benchmark ───────────────────────────────────
        try:
            ar_pt = benchmark_autoregressive(
                decoder_pytorch, model, bs, seq_len=seq_len, n_iters=3,
            )
            ar_tri = benchmark_autoregressive(
                decoder_triton, model, bs, seq_len=seq_len, n_iters=3,
            )
            # CUDA Graph AR
            ar_graph_times = []
            for _ in range(3):
                sg = decoder_graph.init_state(bs, device=device)
                decoder_graph.warmup_cuda_graph(sg, n_warmup=3)
                ug = torch.randn(bs, d_model, device=device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for t in range(seq_len):
                    decoder_graph.step(ug, sg)
                torch.cuda.synchronize()
                ar_graph_times.append((time.perf_counter() - start) * 1000)
            ar_graph = {
                "total_ms": sum(ar_graph_times) / len(ar_graph_times),
                "per_token_ms": sum(ar_graph_times) / len(ar_graph_times) / seq_len,
                "tokens_per_sec": seq_len / (sum(ar_graph_times) / len(ar_graph_times) / 1000),
            }

            ar_speedup_tri = ar_pt["total_ms"] / ar_tri["total_ms"] if ar_tri["total_ms"] > 0 else float('inf')
            ar_speedup_graph = ar_pt["total_ms"] / ar_graph["total_ms"] if ar_graph["total_ms"] > 0 else float('inf')

            ar_results = {
                "pytorch": ar_pt,
                "triton": ar_tri,
                "cuda_graph": ar_graph,
                "speedup_triton": ar_speedup_tri,
                "speedup_cuda_graph": ar_speedup_graph,
            }

            # torch.compile AR benchmark
            if decoder_compile is not None:
                ar_compile = benchmark_autoregressive(
                    decoder_compile, model, bs, seq_len=seq_len, n_iters=3,
                )
                ar_speedup_compile = ar_pt["total_ms"] / ar_compile["total_ms"] if ar_compile["total_ms"] > 0 else float('inf')
                ar_results["compile"] = ar_compile
                ar_results["speedup_compile"] = ar_speedup_compile
                print(f"  AR ({seq_len} tok): PT {ar_pt['per_token_ms']:.4f} | "
                      f"Compile {ar_compile['per_token_ms']:.4f} ({ar_speedup_compile:.2f}x) | "
                      f"Tri {ar_tri['per_token_ms']:.4f} ({ar_speedup_tri:.2f}x) | "
                      f"Graph {ar_graph['per_token_ms']:.4f} ({ar_speedup_graph:.2f}x)")
            else:
                print(f"  AR ({seq_len} tok): PT {ar_pt['per_token_ms']:.4f} | "
                      f"Tri {ar_tri['per_token_ms']:.4f} ({ar_speedup_tri:.2f}x) | "
                      f"Graph {ar_graph['per_token_ms']:.4f} ({ar_speedup_graph:.2f}x)")

            results["autoregressive"][str(bs)] = ar_results
        except Exception as e:
            print(f"  AR benchmark failed: {e}")

    # Save results
    output_path = f"benchmarks/results/{mode}_r{mimo_rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mamba-3 Decoding Kernel Benchmark")
    parser.add_argument("--mode", choices=["siso", "mimo", "both"], default="both")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32,128",
                        help="Comma-separated batch sizes")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile benchmark")
    parser.add_argument("--no-percentiles", action="store_true",
                        help="Skip latency percentile measurement")
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
            with_compile=not args.no_compile,
            with_percentiles=not args.no_percentiles,
        )


if __name__ == "__main__":
    main()
