#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NeuroDistance V3 Benchmark Suite

Tests the 22 V3 production-scale strategies:
- QT: Quantization family (5 strategies)
- DK: Disk family (4 strategies)
- PT: Partition family (4 strategies)
- SY: System family (5 strategies)
- BZ: Binarization family (4 strategies)

Run from faiss root directory:
    PYTHONPATH=build/faiss/python python benchs/bench_neurodistance_v3.py
"""

import sys
import time
import numpy as np
import argparse

# Import SWIG module directly since loader may not export new classes
sys.path.insert(0, 'build/faiss/python')
import swigfaiss as faiss


def generate_data(n, d, seed=42):
    """Generate random normalized vectors."""
    np.random.seed(seed)
    x = np.random.randn(n, d).astype('float32')
    # Normalize
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    return x


def compute_recall(gt, results, k):
    """Compute recall@k."""
    n = gt.shape[0]
    recall = 0
    for i in range(n):
        gt_set = set(gt[i, :k])
        result_set = set(results[i, :k])
        recall += len(gt_set.intersection(result_set)) / k
    return recall / n


def benchmark_strategy(name, index, xb, xq, gt, k=10):
    """Benchmark a single strategy."""
    results = {
        'name': name,
        'recall_1': 0.0,
        'recall_10': 0.0,
        'latency_ms': 0.0,
        'throughput_qps': 0.0,
        'memory_mb': 0.0,
        'compression': 1.0,
    }

    try:
        n_train = min(10000, len(xb))

        # Train
        t0 = time.time()
        index.train(n_train, faiss.swig_ptr(xb[:n_train]))
        train_time = time.time() - t0

        # Add
        t0 = time.time()
        index.add(len(xb), faiss.swig_ptr(xb))
        add_time = time.time() - t0

        # Search (warm-up)
        D = np.zeros((len(xq), k), dtype='float32')
        I = np.zeros((len(xq), k), dtype='int64')
        index.search(len(xq), faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I))

        # Search (timed)
        n_runs = 3
        total_time = 0
        for _ in range(n_runs):
            t0 = time.time()
            index.search(len(xq), faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I))
            total_time += time.time() - t0

        avg_time = total_time / n_runs

        # Compute recall
        recall_1 = compute_recall(gt, I, 1)
        recall_10 = compute_recall(gt, I, k)

        # Estimate memory (approximate)
        memory_mb = (len(xb) * xb.shape[1] * 4) / (1024 * 1024)  # Base float32

        # Get compression ratio if available
        compression = 1.0
        if hasattr(index, 'get_compression_ratio'):
            compression = index.get_compression_ratio()

        results['recall_1'] = recall_1
        results['recall_10'] = recall_10
        results['latency_ms'] = (avg_time / len(xq)) * 1000
        results['throughput_qps'] = len(xq) / avg_time
        results['memory_mb'] = memory_mb / compression
        results['compression'] = compression
        results['train_time'] = train_time
        results['add_time'] = add_time
        results['status'] = 'OK'

    except Exception as e:
        results['status'] = f'ERROR: {str(e)}'

    return results


def run_benchmarks(n_base=10000, n_query=100, d=128, k=10):
    """Run all V3 benchmarks."""
    print(f"NeuroDistance V3 Benchmark")
    print(f"=" * 60)
    print(f"Config: n_base={n_base}, n_query={n_query}, d={d}, k={k}")
    print()

    # Generate data
    print("Generating data...")
    xb = generate_data(n_base, d, seed=42)
    xq = generate_data(n_query, d, seed=123)

    # Compute ground truth with brute force
    print("Computing ground truth...")
    flat = faiss.IndexFlatL2(d)
    flat.add(n_base, faiss.swig_ptr(xb))
    gt_D = np.zeros((n_query, k), dtype='float32')
    gt_I = np.zeros((n_query, k), dtype='int64')
    flat.search(n_query, faiss.swig_ptr(xq), k, faiss.swig_ptr(gt_D), faiss.swig_ptr(gt_I))

    # Phase 1 strategies (T01-T03)
    strategies = []

    # QT-01: Scalar Quantization
    if hasattr(faiss, 'IndexNeuroScalarQuantization'):
        print("\n--- QT-01: IndexNeuroScalarQuantization ---")
        idx = faiss.IndexNeuroScalarQuantization(d, 8)  # 8-bit
        idx.rerank_k = 100
        result = benchmark_strategy("QT-01 ScalarQuant (8-bit)", idx, xb, xq, gt_I, k)
        strategies.append(result)
        print(f"  Recall@1: {result['recall_1']:.3f}, Recall@10: {result['recall_10']:.3f}")
        print(f"  Latency: {result['latency_ms']:.2f}ms, QPS: {result['throughput_qps']:.1f}")
        print(f"  Compression: {result['compression']:.1f}x, Status: {result.get('status', 'OK')}")

        # Also test 4-bit
        idx4 = faiss.IndexNeuroScalarQuantization(d, 4)  # 4-bit
        idx4.rerank_k = 100
        result4 = benchmark_strategy("QT-01 ScalarQuant (4-bit)", idx4, xb, xq, gt_I, k)
        strategies.append(result4)
        print(f"\n  4-bit variant:")
        print(f"  Recall@1: {result4['recall_1']:.3f}, Recall@10: {result4['recall_10']:.3f}")
        print(f"  Compression: {result4['compression']:.1f}x, Status: {result4.get('status', 'OK')}")
    else:
        print("\n--- QT-01: IndexNeuroScalarQuantization NOT AVAILABLE ---")

    # BZ-01: Zoned Binarization
    if hasattr(faiss, 'IndexNeuroZonedBinarization'):
        print("\n--- BZ-01: IndexNeuroZonedBinarization ---")
        idx = faiss.IndexNeuroZonedBinarization(d, 500)  # Higher rerank for better recall
        result = benchmark_strategy("BZ-01 ZonedBinary (10/20/70, r=500)", idx, xb, xq, gt_I, k)
        strategies.append(result)
        print(f"  Recall@1: {result['recall_1']:.3f}, Recall@10: {result['recall_10']:.3f}")
        print(f"  Latency: {result['latency_ms']:.2f}ms, QPS: {result['throughput_qps']:.1f}")
        print(f"  Compression: {result['compression']:.1f}x, Status: {result.get('status', 'OK')}")

        # Custom zone config with more float precision
        idx2 = faiss.IndexNeuroZonedBinarization(d, 0.20, 0.30, 500)  # 20/30/50 zones
        result2 = benchmark_strategy("BZ-01 ZonedBinary (20/30/50, r=500)", idx2, xb, xq, gt_I, k)
        strategies.append(result2)
        print(f"\n  20/30/50 variant:")
        print(f"  Recall@1: {result2['recall_1']:.3f}, Recall@10: {result2['recall_10']:.3f}")
        print(f"  Compression: {result2['compression']:.1f}x, Status: {result2.get('status', 'OK')}")
    else:
        print("\n--- BZ-01: IndexNeuroZonedBinarization NOT AVAILABLE ---")

    # SY-02: SIMD Distance
    if hasattr(faiss, 'IndexNeuroSIMDDistance'):
        print("\n--- SY-02: IndexNeuroSIMDDistance ---")
        inner = faiss.IndexFlatL2(d)
        idx = faiss.IndexNeuroSIMDDistance(inner, 32)
        result = benchmark_strategy("SY-02 SIMDDistance", idx, xb, xq, gt_I, k)
        strategies.append(result)
        print(f"  Recall@1: {result['recall_1']:.3f}, Recall@10: {result['recall_10']:.3f}")
        print(f"  Latency: {result['latency_ms']:.2f}ms, QPS: {result['throughput_qps']:.1f}")
        print(f"  Status: {result.get('status', 'OK')}")
    else:
        print("\n--- SY-02: IndexNeuroSIMDDistance NOT AVAILABLE ---")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<35} {'R@1':>6} {'R@10':>6} {'ms':>8} {'QPS':>8} {'Comp':>6}")
    print("-" * 60)
    for s in strategies:
        status = s.get('status', 'OK')
        if status == 'OK':
            print(f"{s['name']:<35} {s['recall_1']:>6.3f} {s['recall_10']:>6.3f} "
                  f"{s['latency_ms']:>8.2f} {s['throughput_qps']:>8.1f} {s['compression']:>6.1f}x")
        else:
            print(f"{s['name']:<35} {status}")

    return strategies


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuroDistance V3 Benchmark')
    parser.add_argument('--n-base', type=int, default=10000, help='Number of base vectors')
    parser.add_argument('--n-query', type=int, default=100, help='Number of query vectors')
    parser.add_argument('--dim', type=int, default=128, help='Vector dimension')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')

    args = parser.parse_args()

    run_benchmarks(
        n_base=args.n_base,
        n_query=args.n_query,
        d=args.dim,
        k=args.k
    )
