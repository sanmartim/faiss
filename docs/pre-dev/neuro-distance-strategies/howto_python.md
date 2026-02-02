# NeuroDistance Python How-To Guide

Bio-inspired vector search strategies for FAISS. 13 strategies across 10 index classes, all accessible from Python.

---

## Setup

```python
import faiss
import numpy as np

# Sample data
d = 64          # dimension
nb = 10000      # database size
nq = 100        # number of queries
k = 10          # neighbors to retrieve

rng = np.random.RandomState(42)
xb = rng.random((nb, d)).astype('float32')
xq = rng.random((nq, d)).astype('float32')

# All strategies wrap an IndexFlatL2 that holds the data
index = faiss.IndexFlatL2(d)
index.add(xb)

# Ground truth for recall measurement
D_gt, I_gt = index.search(xq, k)
```

---

## 1. Progressive Elimination (ED-01 to ED-04)

Eliminates candidates column-by-column. Fast but recall drops at high dimensions.

### ED-01: Fixed Elimination

```python
ed = faiss.IndexNeuroElimination(index, faiss.NEURO_FIXED)
ed.cutoff_percentile = 0.7   # keep 70% of candidates each round
ed.min_candidates = 50       # never go below 50 candidates
D, I = ed.search(xq, k)
```

### ED-02: Adaptive Dispersion

Adjusts cutoff based on column dispersion (std/mean). High dispersion columns cut more aggressively.

```python
ed = faiss.IndexNeuroElimination(index, faiss.NEURO_ADAPTIVE_DISPERSION)
ed.dispersion_low = 0.3
ed.dispersion_high = 0.7
ed.cutoff_low_dispersion = 0.8   # keep 80% when dispersion is low
ed.cutoff_high_dispersion = 0.3  # keep 30% when dispersion is high
D, I = ed.search(xq, k)
```

### ED-03: Variance Order

Orders columns by data variance (most discriminative first). Requires `train()`.

```python
ed = faiss.IndexNeuroElimination(index, faiss.NEURO_VARIANCE_ORDER)
ed.sample_fraction = 0.05  # sample 5% of data for variance estimation
ed.train(xb)               # computes variance-based column ordering
D, I = ed.search(xq, k)
```

### ED-04: Uncertainty Deferred

Defers elimination when a column provides weak signal. Accumulates evidence before cutting.

```python
ed = faiss.IndexNeuroElimination(index, faiss.NEURO_UNCERTAINTY_DEFERRED)
ed.confidence_threshold = 0.4     # defer if dispersion < 0.4
ed.max_accumulated_columns = 3    # force decision after 3 deferred columns
D, I = ed.search(xq, k)
```

### Per-query parameter override

```python
params = faiss.NeuroEliminationParams()
params.cutoff_percentile = 0.9   # more conservative for this batch
params.min_candidates = 100
params.collect_stats = True

ed = faiss.IndexNeuroElimination(index, faiss.NEURO_FIXED)
D, I = ed.search(xq, k, params)
print(f"Calculations: {ed.last_stats.calculations_performed}")
```

---

## 2. Dropout Ensemble (ED-05)

Creates multiple "views" with different dimension subsets, combines results via voting.

```python
ens = faiss.IndexNeuroDropoutEnsemble(index)
ens.num_views = 7               # number of parallel views
ens.dropout_rate = 0.3          # drop 30% of dimensions per view
ens.top_k_per_view = 30         # candidates per view (0 = auto)

# Dropout modes
ens.dropout_mode = faiss.NEURO_DROPOUT_COMPLEMENTARY  # minimize overlap
# Other options:
#   faiss.NEURO_DROPOUT_RANDOM       - independent random masks
#   faiss.NEURO_DROPOUT_STRUCTURED   - semantic grouping
#   faiss.NEURO_DROPOUT_ADVERSARIAL  - exclude previous top columns

# Integration methods
ens.integration = faiss.NEURO_INTEGRATE_BORDA         # sum of ranks
# Other options:
#   faiss.NEURO_INTEGRATE_VOTING      - count appearances
#   faiss.NEURO_INTEGRATE_MEAN_DIST   - average distances
#   faiss.NEURO_INTEGRATE_FULL_RERANK - exact L2 on union (highest recall)

D, I = ens.search(xq, k)
```

---

## 3. Learned Weights (PA-01)

Per-dimension weights learned from relevance feedback. Dimensions that help correct rankings get higher weight.

```python
pw = faiss.IndexNeuroWeighted(index)
pw.learning_rate = 0.05   # gradient step size
pw.weight_decay = 0.99    # decay toward uniform each round
pw.min_weight = 0.01      # floor to prevent zero weights

# Initialize weights (must call before search)
pw.train(xq)

# Search with uniform weights (before feedback)
D_before, I_before = pw.search(xq, k)

# Provide relevance feedback: (query, positive_match, negative_match)
for i in range(200):
    query = xq[i:i+1]
    positive = xb[I_gt[i, 0]:I_gt[i, 0]+1]    # true nearest neighbor
    negative = xb[rng.randint(0, nb, 1)]         # random non-relevant
    pw.feedback(
        1,
        faiss.swig_ptr(query),
        faiss.swig_ptr(positive),
        faiss.swig_ptr(negative),
    )

# Search with learned weights
D_after, I_after = pw.search(xq, k)
print(f"Feedback rounds: {pw.feedback_count}")

# Save / load weights
pw.save_weights(b"/tmp/neuro_weights.bin")
pw.load_weights(b"/tmp/neuro_weights.bin")
```

### Per-query weight override

```python
params = faiss.NeuroWeightedParams()
custom_w = np.ones(d, dtype='float32')
custom_w[:d//2] = 2.0  # boost first half of dimensions
params.weights.resize(d)
for j in range(d):
    params.weights[j] = float(custom_w[j])
D, I = pw.search(xq, k, params)
```

---

## 4. Contrastive Learning (MR-03)

Margin-based weight updates with hard negative mining. Uses the same `IndexNeuroWeighted` class.

```python
pw = faiss.IndexNeuroWeighted(index)
pw.train(xq)

# Prepare training data
positives = xb[I_gt[:, 0]]                          # nearest neighbors
negatives = xb[rng.randint(0, nb, nq)]              # random negatives

# Single contrastive feedback call (batch)
pw.feedback_contrastive(
    nq,                              # number of queries
    faiss.swig_ptr(xq),             # queries
    faiss.swig_ptr(positives),      # positives (nq * d)
    faiss.swig_ptr(negatives),      # negatives (nq * n_negatives * d)
    1,                               # n_negatives per query
    1.0,                             # margin_scale
)

D, I = pw.search(xq, k)
```

### Multiple negatives (hard negative mining)

```python
n_neg = 5
# negatives shape: (nq * n_neg, d) - flattened
all_negatives = xb[rng.randint(0, nb, (nq, n_neg)).ravel()]

pw.feedback_contrastive(
    nq,
    faiss.swig_ptr(xq),
    faiss.swig_ptr(positives),
    faiss.swig_ptr(all_negatives.astype('float32')),
    n_neg,     # 5 negatives per query, hardest one is selected
    1.5,       # stronger margin scaling
)
```

---

## 5. Contextual Weights (PA-02)

Maintains per-query-type weight vectors. Queries are clustered; each cluster learns its own weights.

```python
cw = faiss.IndexNeuroContextualWeighted(index, 5)  # 5 query clusters
cw.learning_rate = 0.05
cw.weight_decay = 0.99

# Train: clusters the query space via k-means
cw.train(xq)

# Search (auto-classifies each query to nearest cluster)
D, I = cw.search(xq, k)

# Feedback updates per-cluster weights independently
positives = xb[I_gt[:, 0]]
negatives = xb[rng.randint(0, nb, nq)]
for i in range(200):
    cw.feedback(
        1,
        faiss.swig_ptr(xq[i:i+1]),
        faiss.swig_ptr(positives[i:i+1]),
        faiss.swig_ptr(negatives[i:i+1]),
    )

# Check which cluster a query belongs to
cluster_id = cw.classify_query(faiss.swig_ptr(xq[0:1]))
print(f"Query 0 -> cluster {cluster_id}")
```

### Force a specific cluster

```python
params = faiss.NeuroContextualParams()
params.force_cluster = 2  # use cluster 2's weights regardless of query
D, I = cw.search(xq, k, params)
```

---

## 6. Missing Value Aware (PA-03)

Handles NaN in query vectors. Skips missing dimensions and renormalizes distances.

```python
# Inject NaN into queries (simulating missing features)
xq_sparse = xq.copy()
mask = rng.random(xq_sparse.shape) < 0.15  # 15% missing
xq_sparse[mask] = np.nan

# PROPORTIONAL: weight = (1 - missing_rate)
mv = faiss.IndexNeuroMissingValue(index, faiss.NEURO_MISSING_PROPORTIONAL)
D, I = mv.search(xq_sparse, k)

# THRESHOLD: ignore column if missing_rate > threshold
mv2 = faiss.IndexNeuroMissingValue(index, faiss.NEURO_MISSING_THRESHOLD)
mv2.ignore_threshold = 0.5  # drop columns with >50% NaN
D, I = mv2.search(xq_sparse, k)

# HYBRID (default): weight = (1 - missing_rate)^2
mv3 = faiss.IndexNeuroMissingValue(index, faiss.NEURO_MISSING_HYBRID)
D, I = mv3.search(xq_sparse, k)
```

---

## 7. Parallel Voting (PP-01)

Divides dimensions into groups, each group independently selects candidates, results are merged.

```python
pv = faiss.IndexNeuroParallelVoting(index, 4)  # 4 dimension groups
pv.top_k_per_group = 50  # candidates per group (0 = auto k*3)

# Grouping method
pv.grouping = faiss.NEURO_GROUP_CONSECUTIVE   # dims 0-15, 16-31, ...
# or
pv.grouping = faiss.NEURO_GROUP_INTERLEAVED   # dim i -> group (i % 4)

# Integration method (FULL_RERANK gives best recall)
pv.integration = faiss.NEURO_INTEGRATE_FULL_RERANK
D, I = pv.search(xq, k)
```

### All integration methods

```python
for method, name in [
    (faiss.NEURO_INTEGRATE_VOTING, "Voting"),
    (faiss.NEURO_INTEGRATE_BORDA, "Borda"),
    (faiss.NEURO_INTEGRATE_MEAN_DIST, "Mean Distance"),
    (faiss.NEURO_INTEGRATE_FULL_RERANK, "Full Rerank"),
]:
    pv.integration = method
    D, I = pv.search(xq, k)
    # compute recall vs ground truth...
    print(f"{name}: done")
```

---

## 8. Coarse-to-Fine (PP-02)

Multi-resolution progressive refinement. Cheap coarse distances eliminate candidates before expensive full L2.

```python
cf = faiss.IndexNeuroCoarseToFine(index, 3)  # 3 resolution levels

# Cutoffs: fraction of candidates kept at each level
# Level 0 (coarsest): keep 30%
# Level 1 (medium):   keep 50%
# Level 2 (full):     keep 100% (always 1.0)
cf.cutoff_per_level.resize(3)
cf.cutoff_per_level[0] = 0.3
cf.cutoff_per_level[1] = 0.5
cf.cutoff_per_level[2] = 1.0

# Must train to precompute coarse representations
cf.train(xq)

D, I = cf.search(xq, k)
```

### Conservative cutoffs (high recall)

```python
cf.cutoff_per_level[0] = 0.8   # keep 80% at coarse level
cf.cutoff_per_level[1] = 0.9   # keep 90% at medium level
cf.cutoff_per_level[2] = 1.0
cf.train(xq)
D, I = cf.search(xq, k)
# Higher recall, less speedup
```

### Aggressive cutoffs (max speed)

```python
cf.cutoff_per_level[0] = 0.1   # keep only 10% at coarse
cf.cutoff_per_level[1] = 0.3   # keep 30% at medium
cf.cutoff_per_level[2] = 1.0
cf.train(xq)
D, I = cf.search(xq, k)
# Lower recall, more speedup
```

---

## 9. Lateral Inhibition (MR-01)

Promotes diversity by suppressing results that are too similar to each other. Wraps any index.

```python
inh = faiss.IndexNeuroInhibition(index)
inh.similarity_threshold = 0.5  # L2 distance below this = "too similar"
inh.max_per_cluster = 2         # max results per similarity group
inh.k_expansion = 3.0           # request k*3 candidates, then diversify

D, I = inh.search(xq, k)
```

### Wrap another strategy

```python
# Diverse results from learned-weight search
pw = faiss.IndexNeuroWeighted(index)
pw.train(xq)

inh = faiss.IndexNeuroInhibition(pw)
inh.similarity_threshold = 1.0
inh.max_per_cluster = 2
D, I = inh.search(xq, k)
```

---

## 10. Query Cache (MR-02)

Caches search results for repeated or similar queries. Wraps any index.

```python
cache = faiss.IndexNeuroCache(index, 1024, 0.1)
#                              ^       ^     ^
#                              |       |     grid_step (query discretization)
#                              |       cache_size (max LRU entries)
#                              sub_index

# First search: all cache misses (populates cache)
D1, I1 = cache.search(xq, k)

# Second search with same queries: all cache hits (instant)
D2, I2 = cache.search(xq, k)

# Check hit rate
print(f"Hit rate: {cache.hit_rate():.2f}")  # 0.50 (100 miss + 100 hit)

# Clear cache manually
cache.clear_cache()

# Cache auto-invalidates on data changes
cache.add(new_vectors)  # clears cache
cache.reset()           # clears cache
```

### Tuning grid_step

```python
# Smaller grid_step = stricter matching (fewer false hits)
cache_strict = faiss.IndexNeuroCache(index, 1024, 0.01)

# Larger grid_step = fuzzy matching (more cache hits, may return
# results for "similar enough" queries)
cache_fuzzy = faiss.IndexNeuroCache(index, 1024, 1.0)
```

---

## Combinations

Decorators (MR-01 Inhibition, MR-02 Cache) can wrap any strategy.

### Cache + Inhibition (diverse + fast repeated queries)

```python
inh = faiss.IndexNeuroInhibition(index)
inh.similarity_threshold = 1.0
inh.max_per_cluster = 2

cached = faiss.IndexNeuroCache(inh, 512, 0.1)
D, I = cached.search(xq, k)  # diverse + cached
```

### Cache + Learned Weights

```python
pw = faiss.IndexNeuroWeighted(index)
pw.train(xq)
# ... feedback loop ...

cached = faiss.IndexNeuroCache(pw, 2048, 0.1)
D, I = cached.search(xq, k)

# Note: cache is NOT invalidated by feedback() - only by add()/reset().
# If you update weights, call clear_cache() manually:
cached.clear_cache()
```

### Cache + Elimination

```python
ed = faiss.IndexNeuroElimination(index, faiss.NEURO_FIXED)
ed.cutoff_percentile = 0.7

cached = faiss.IndexNeuroCache(ed, 1024, 0.1)
D, I = cached.search(xq, k)
```

---

## Collecting Search Statistics

```python
params = faiss.NeuroSearchParameters()
params.collect_stats = True

ed = faiss.IndexNeuroElimination(index, faiss.NEURO_FIXED)
D, I = ed.search(xq, k, params)

print(f"Calculations: {ed.last_stats.calculations_performed}")
print(f"Columns used: {ed.last_stats.columns_used}")
```

---

## Strategy Selection Guide

| Scenario | Strategy | Why |
|----------|----------|-----|
| General purpose, have feedback | PA-01 Learned | 92-97% recall, 4-5x speedup |
| Multiple query types + feedback | PA-02 Contextual | Per-type adaptation |
| Missing/sparse features | PA-03 Missing | Graceful NaN handling |
| Maximum recall | PP-01 Parallel (FULL_RERANK) | >99% recall |
| High-dimensional (d>=256) | PP-02 CoarseToFine | 6x speedup at d=512 |
| Low-dimensional (d<=32) | ED-01 Fixed | 93% recall, 3x speedup |
| Diverse results needed | MR-01 Inhibition | Wrap any strategy |
| Repeated queries | MR-02 Cache | >1000x on cache hits |
| Fast approximate + speed | ED-04 Uncertainty | 8-25x speedup, lower recall |

---

## Running the Benchmark

```bash
# Quick test (d=32)
python benchs/bench_neurodistance.py

# Multi-dimension sweep
python benchs/bench_neurodistance.py --dims 32,128,512 --nb 10000 --nq 100

# Custom parameters + CSV output
python benchs/bench_neurodistance.py --d 64 --nb 50000 --nq 200 --k 20 \
    --csv results.csv

# All options
python benchs/bench_neurodistance.py --help
```
