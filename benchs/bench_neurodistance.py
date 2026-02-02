# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark for NeuroDistance progressive elimination strategies.

Compares IndexFlatL2 (brute force baseline) against:
  - ED-01: Fixed elimination (IndexNeuroElimination with NEURO_FIXED)
  - ED-02: Adaptive dispersion (IndexNeuroElimination with NEURO_ADAPTIVE_DISPERSION)

Usage:
    python bench_neurodistance.py
    python bench_neurodistance.py --d 64 --nb 100000 --nq 200 --k 10
    python bench_neurodistance.py --min-candidates 500 --cutoff 0.8
"""

import argparse
import time

import numpy as np

import faiss


def generate_clustered_data(d, nb, nq, n_clusters=50, seed=42):
    """Generate clustered synthetic data with varying dimension importance."""
    rng = np.random.RandomState(seed)

    # Cluster centers with increasing scale per dimension
    scales = 1.0 + 2.0 * np.arange(d) / d
    centers = rng.uniform(0, 10, (n_clusters, d)) * scales[np.newaxis, :]

    # Database vectors: clustered with noise
    assignments = rng.randint(0, n_clusters, nb)
    xb = centers[assignments] + rng.normal(0, 0.1, (nb, d))

    # Query vectors: also clustered
    q_assignments = rng.randint(0, n_clusters, nq)
    xq = centers[q_assignments] + rng.normal(0, 0.1, (nq, d))

    return xb.astype("float32"), xq.astype("float32")


def compute_recall(labels_test, labels_gt, k):
    """Compute recall@k."""
    nq = labels_gt.shape[0]
    hits = 0
    for q in range(nq):
        gt_set = set(labels_gt[q, :k].tolist())
        for label in labels_test[q, :k]:
            if label in gt_set:
                hits += 1
    return hits / (nq * k)


def benchmark_index(index, xq, k, n_warmup=2, n_runs=5):
    """Benchmark search with warmup and multiple runs."""
    # Warmup
    for _ in range(n_warmup):
        index.search(xq[:1], k)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    return D, I, avg_ms, std_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NeuroDistance elimination strategies"
    )
    parser.add_argument("--d", type=int, default=32, help="Dimension")
    parser.add_argument("--nb", type=int, default=10000, help="Database size")
    parser.add_argument("--nq", type=int, default=100, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument(
        "--n-clusters", type=int, default=50, help="Number of clusters"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cutoff", type=float, default=0.8,
        help="Cutoff percentile for ED-01"
    )
    parser.add_argument(
        "--min-candidates", type=int, default=0,
        help="Minimum candidates (0=auto)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of timed runs"
    )
    args = parser.parse_args()

    print(f"NeuroDistance Benchmark")
    print(f"=" * 70)
    print(f"d={args.d}, nb={args.nb}, nq={args.nq}, k={args.k}")
    print(f"clusters={args.n_clusters}, seed={args.seed}")
    print(f"cutoff={args.cutoff}, min_candidates={args.min_candidates}")
    print()

    # Generate data
    print("Generating clustered data...", end=" ", flush=True)
    xb, xq = generate_clustered_data(
        args.d, args.nb, args.nq, args.n_clusters, args.seed
    )
    print("done.")

    # Ground truth from brute force
    print("Computing ground truth (IndexFlatL2)...", end=" ", flush=True)
    index_flat = faiss.IndexFlatL2(args.d)
    index_flat.add(xb)
    D_gt, I_gt, time_flat, std_flat = benchmark_index(
        index_flat, xq, args.k, n_runs=args.n_runs
    )
    recall_flat = compute_recall(I_gt, I_gt, args.k)
    print("done.")

    # ED-01: Fixed elimination
    print("Running ED-01 (Fixed)...", end=" ", flush=True)
    ed01 = faiss.IndexNeuroElimination(index_flat, faiss.NEURO_FIXED)
    ed01.cutoff_percentile = args.cutoff
    if args.min_candidates > 0:
        ed01.min_candidates = args.min_candidates
    D_ed01, I_ed01, time_ed01, std_ed01 = benchmark_index(
        ed01, xq, args.k, n_runs=args.n_runs
    )
    recall_ed01_1 = compute_recall(I_ed01, I_gt, 1)
    recall_ed01_10 = compute_recall(I_ed01, I_gt, args.k)
    print("done.")

    # ED-02: Adaptive dispersion
    print("Running ED-02 (Adaptive)...", end=" ", flush=True)
    ed02 = faiss.IndexNeuroElimination(
        index_flat, faiss.NEURO_ADAPTIVE_DISPERSION
    )
    if args.min_candidates > 0:
        ed02.min_candidates = args.min_candidates
    D_ed02, I_ed02, time_ed02, std_ed02 = benchmark_index(
        ed02, xq, args.k, n_runs=args.n_runs
    )
    recall_ed02_1 = compute_recall(I_ed02, I_gt, 1)
    recall_ed02_10 = compute_recall(I_ed02, I_gt, args.k)
    print("done.")

    # Results table
    print()
    print(f"{'Strategy':<25} {'Recall@1':>10} {'Recall@10':>10} "
          f"{'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'IndexFlatL2 (baseline)':<25} {recall_flat:>10.4f} "
          f"{recall_flat:>10.4f} {time_flat:>9.2f}+-{std_flat:.2f} "
          f"{'1.00x':>10}")
    print(f"{'ED-01 Fixed':<25} {recall_ed01_1:>10.4f} "
          f"{recall_ed01_10:>10.4f} {time_ed01:>9.2f}+-{std_ed01:.2f} "
          f"{time_flat/time_ed01:>9.2f}x")
    print(f"{'ED-02 Adaptive':<25} {recall_ed02_1:>10.4f} "
          f"{recall_ed02_10:>10.4f} {time_ed02:>9.2f}+-{std_ed02:.2f} "
          f"{time_flat/time_ed02:>9.2f}x")
    print()

    # Hypothesis validation
    print("Hypothesis Validation:")
    print(f"  ED-01 Recall@10 >= 85%: {'PASS' if recall_ed01_10 >= 0.85 else 'FAIL'} ({recall_ed01_10:.1%})")
    print(f"  ED-02 Recall@10 >= 90%: {'PASS' if recall_ed02_10 >= 0.90 else 'FAIL'} ({recall_ed02_10:.1%})")
    if time_ed01 > 0:
        overhead = (time_ed02 - time_ed01) / time_ed01 * 100
        print(f"  ED-02 overhead vs ED-01 < 10%: {'PASS' if overhead < 10 else 'FAIL'} ({overhead:.1f}%)")


if __name__ == "__main__":
    main()
