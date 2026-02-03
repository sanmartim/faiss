#!/usr/bin/env python3
"""NeuroDistance V4 Benchmark - Multi-Scale Sign and Fast-Slow Strategies"""

import sys
sys.path.insert(0, 'build/faiss/python')

import swigfaiss as faiss
import numpy as np
import time

D, N, NQ, K = 64, 10000, 100, 10
np.random.seed(42)

print("=" * 80)
print("NeuroDistance V4 Benchmark")
print("=" * 80)
print(f"D={D}, N={N:,}, NQ={NQ}, K={K}")

# Generate data
print("\nGenerating data...")
data = np.random.randn(N, D).astype('float32')
queries = np.random.randn(NQ, D).astype('float32')

# Ground truth
print("Computing ground truth...")
flat = faiss.IndexFlatL2(D)
flat.add(N, faiss.swig_ptr(data))
gt_D = np.empty((NQ, K), dtype='float32')
gt_I = np.empty((NQ, K), dtype='int64')
flat.search(NQ, faiss.swig_ptr(queries), K, faiss.swig_ptr(gt_D), faiss.swig_ptr(gt_I))

def recall(gt, pred, k):
    r = []
    for i in range(len(gt)):
        gs, ps = set(gt[i][:k]), set(pred[i][:k])
        r.append(len(gs & ps) / len(gs) if gs else 0)
    return np.mean(r)

def test(name, idx, need_train=False):
    D_out = np.empty((NQ, K), dtype='float32')
    I_out = np.empty((NQ, K), dtype='int64')
    try:
        if need_train and hasattr(idx, 'is_trained') and not idx.is_trained:
            idx.train(N, faiss.swig_ptr(data))
        idx.add(N, faiss.swig_ptr(data))
        start = time.time()
        idx.search(NQ, faiss.swig_ptr(queries), K, faiss.swig_ptr(D_out), faiss.swig_ptr(I_out))
        elapsed = time.time() - start
        r = recall(gt_I, I_out, K)
        qps = NQ / elapsed
        return {'name': name, 'recall': r, 'qps': qps, 'time_ms': elapsed * 1000}
    except Exception as e:
        return {'name': name, 'error': str(e)}

results = []

print("\nRunning benchmarks...")
print("-" * 80)

# Baseline
print("Testing Baseline...")
results.append(test('Baseline: IndexFlatL2', faiss.IndexFlatL2(D)))

# MS Family (Multi-Scale Sign)
print("Testing MS Family...")
results.append(test('V4-MS01: MultiScaleSign', faiss.IndexNeuroMultiScaleSign(D), True))
results.append(test('V4-MS02: AdaptiveScale', faiss.IndexNeuroAdaptiveScale(D, 3), True))
results.append(test('V4-MS03: HierarchicalScale', faiss.IndexNeuroHierarchicalScale(D), True))
results.append(test('V4-MS04: MultiScaleIntersection', faiss.IndexNeuroMultiScaleIntersection(D), True))
results.append(test('V4-MS05: LearnedScale', faiss.IndexNeuroLearnedScale(D, 5), True))

# FS Family (Fast-Slow)
print("Testing FS Family...")
results.append(test('V4-FS01: HammingPrefilter', faiss.IndexNeuroHammingPrefilter(D, 0.10), True))
results.append(test('V4-FS02: CentroidBounds', faiss.IndexNeuroCentroidBounds(D, 100), True))
results.append(test('V4-FS03: ProjectionCascade', faiss.IndexNeuroProjectionCascade(D), True))
results.append(test('V4-FS04: StatisticalPrescreen', faiss.IndexNeuroStatisticalPrescreen(D, 0.20), True))
results.append(test('V4-FS05: EnsembleVoting', faiss.IndexNeuroEnsembleVoting(D), True))
results.append(test('V4-FS06: RecommendedPipeline', faiss.IndexNeuroRecommendedPipeline(D), True))

# Print results
print("\n" + "=" * 80)
print("BENCHMARK RESULTS")
print("=" * 80)
print(f"{'Strategy':<40} {'Recall@10':>10} {'QPS':>12} {'Time(ms)':>10}")
print("-" * 80)

baseline_qps = None
valid = [r for r in results if 'error' not in r]
valid.sort(key=lambda x: x['recall'], reverse=True)

for r in valid:
    if 'Baseline' in r['name']:
        baseline_qps = r['qps']

for r in valid:
    speedup = ""
    if baseline_qps and 'Baseline' not in r['name']:
        ratio = r['qps'] / baseline_qps
        speedup = f" ({ratio:.1f}x)"
    print(f"{r['name']:<40} {r['recall']:>10.4f} {r['qps']:>8,.0f}{speedup:>6} {r['time_ms']:>10.1f}")

# Print errors
errors = [r for r in results if 'error' in r]
if errors:
    print("\nErrors:")
    for r in errors:
        print(f"  {r['name']}: {r['error']}")

# Family summary
print("\n" + "=" * 80)
print("FAMILY SUMMARY")
print("=" * 80)

families = {'Baseline': [], 'MS Family': [], 'FS Family': []}
for r in valid:
    if 'Baseline' in r['name']:
        families['Baseline'].append(r)
    elif '-MS' in r['name']:
        families['MS Family'].append(r)
    elif '-FS' in r['name']:
        families['FS Family'].append(r)

for family, items in families.items():
    if items:
        avg_recall = np.mean([r['recall'] for r in items])
        avg_qps = np.mean([r['qps'] for r in items])
        best = max(items, key=lambda x: x['recall'])
        print(f"{family:<15}: {len(items)} strategies, Avg Recall={avg_recall:.4f}, Avg QPS={avg_qps:,.0f}")
        short_name = best['name'].split(': ')[1] if ': ' in best['name'] else best['name']
        print(f"                 Best: {short_name} (R@10={best['recall']:.4f})")

# Target comparison
print("\n" + "=" * 80)
print("TARGET COMPARISON (from PRD)")
print("=" * 80)
print(f"{'Strategy':<30} {'Target Recall':>15} {'Actual Recall':>15} {'Status':>10}")
print("-" * 80)

targets = {
    'V4-MS01': (0.92, 0.97),
    'V4-MS02': (0.93, 0.97),
    'V4-MS03': (0.90, 0.95),
    'V4-MS04': (0.75, 0.85),
    'V4-MS05': (0.95, 0.98),
    'V4-FS01': (0.97, 0.99),
    'V4-FS02': (0.98, 0.99),
    'V4-FS03': (0.93, 0.97),
    'V4-FS04': (0.95, 0.98),
    'V4-FS05': (0.98, 0.99),
    'V4-FS06': (0.96, 0.99),
}

for r in valid:
    if 'Baseline' in r['name']:
        continue
    prefix = r['name'].split(':')[0]
    if prefix in targets:
        lo, hi = targets[prefix]
        status = "PASS" if lo <= r['recall'] <= 1.0 else "FAIL"
        target_str = f"{lo:.0%}-{hi:.0%}"
        print(f"{r['name'].split(': ')[1]:<30} {target_str:>15} {r['recall']:>14.2%} {status:>10}")

print("\n" + "=" * 80)
print("Benchmark complete!")
