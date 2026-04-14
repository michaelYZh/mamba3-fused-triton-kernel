"""
Benchmark script for Mamba-3 decoding kernels.

Compares PyTorch reference vs Triton fused kernel performance.

Usage:
    python benchmarks/run_bench.py --mode siso --batch-sizes 1,8,32,128 --seq-len 1024
    python benchmarks/run_bench.py --mode mimo --mimo-rank 4 --batch-sizes 1,8,32,128
"""

import argparse
import json
import time
from datetime import datetime

import torch

from src.models.mamba3 import Mamba3
from src.models.inference import Mamba3Decoder


def warmup(fn, args, n_warmup=10):
    """Run function n_warmup times to warm up GPU."""
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()


def benchmark_single_step(fn, args, n_iters=100):
    """Benchmark a single decoding step."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / n_iters * 1000
    return avg_ms


def benchmark_autoregressive(decoder, model, batch_size, seq_len, n_iters=10):
    """Benchmark full autoregressive decoding for seq_len tokens."""
    times = []
    for _ in range(n_iters):
        # Reset state
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
):
    """Run the full benchmark suite."""
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
    print(f"Mamba-3 Decoding Benchmark")
    print(f"Mode: {mode} | MIMO rank: {mimo_rank if is_mimo else 1} | Seq len: {seq_len}")
    print(f"GPU: {results['config']['gpu_name']}")
    print(f"{'='*60}")

    for bs in batch_sizes:
        print(f"\nBatch size: {bs}")

        # Create model and decoder
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

        state_pt = decoder_pytorch.init_state(bs, device=device)
        state_tri = decoder_triton.init_state(bs, device=device)
        state_graph = decoder_graph.init_state(bs, device=device)

        u = torch.randn(bs, d_model, device=device)

        # PyTorch reference
        warmup(decoder_pytorch.step, (u, state_pt), n_warmup=5)
        ms_pt = benchmark_single_step(decoder_pytorch.step, (u, state_pt), n_iters=50)

        # Triton fused
        warmup(decoder_triton.step, (u, state_tri), n_warmup=5)
        ms_tri = benchmark_single_step(decoder_triton.step, (u, state_tri), n_iters=50)

        # CUDA Graph
        decoder_graph.warmup_cuda_graph(state_graph, n_warmup=10)
        ms_graph = benchmark_single_step(decoder_graph.step, (u, state_graph), n_iters=50)

        speedup_tri = ms_pt / ms_tri if ms_tri > 0 else float('inf')
        speedup_graph = ms_pt / ms_graph if ms_graph > 0 else float('inf')

        results["single_step"][str(bs)] = {
            "pytorch_ms": ms_pt,
            "triton_ms": ms_tri,
            "cuda_graph_ms": ms_graph,
            "speedup_triton": speedup_tri,
            "speedup_cuda_graph": speedup_graph,
        }
        print(f"  PyTorch: {ms_pt:.4f} ms | Triton: {ms_tri:.4f} ms ({speedup_tri:.2f}x) | "
              f"CUDA Graph: {ms_graph:.4f} ms ({speedup_graph:.2f}x)")

        # Autoregressive benchmark (smaller n_iters due to longer runtime)
        try:
            ar_pt = benchmark_autoregressive(
                decoder_pytorch, model, bs,
                seq_len=seq_len, n_iters=3,
            )
            ar_tri = benchmark_autoregressive(
                decoder_triton, model, bs,
                seq_len=seq_len, n_iters=3,
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

            results["autoregressive"][str(bs)] = {
                "pytorch": ar_pt,
                "triton": ar_tri,
                "cuda_graph": ar_graph,
                "speedup_triton": ar_speedup_tri,
                "speedup_cuda_graph": ar_speedup_graph,
            }
            print(f"  AR ({seq_len} tok): PT {ar_pt['per_token_ms']:.4f} ms/tok | "
                  f"Tri {ar_tri['per_token_ms']:.4f} ms/tok ({ar_speedup_tri:.2f}x) | "
                  f"Graph {ar_graph['per_token_ms']:.4f} ms/tok ({ar_speedup_graph:.2f}x)")
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
