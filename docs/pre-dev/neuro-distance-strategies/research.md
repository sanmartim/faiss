# Gate 0: Research - NeuroDistance Strategies

**Feature:** neuro-distance-strategies
**Research Mode:** New Feature (C++ implementation, Python factory for testing)
**Date:** 2026-02-02

---

## 1. FAISS Distance Computation Architecture

### 1.1 MetricType Enum

**File:** `faiss/MetricType.h:24-44`

FAISS uses an enum with intentional gaps for extensibility:

```cpp
enum MetricType {
    METRIC_INNER_PRODUCT = 0,  // similarity metric
    METRIC_L2 = 1,              // squared L2
    METRIC_L1,
    METRIC_Linf,
    METRIC_Lp,
    METRIC_Canberra = 20,       // gap from 5 to 20
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
    METRIC_Jaccard,
    METRIC_NaNEuclidean,
    METRIC_GOWER,               // = 25
};
```

**Key insight:** The enum has gaps (5 to 20), currently ends at ~25. New NeuroDistance metrics can start at a new block (e.g., 100+).

### 1.2 Compile-Time Dispatch

**File:** `faiss/MetricType.h:67-100`

`with_metric_type()` template dispatches runtime metric values to compile-time constants, enabling compiler optimizations per metric.

### 1.3 VectorDistance Template

**File:** `faiss/utils/extra_distances-inl.h:21-227`

Each metric implements a `VectorDistance<MetricType>` specialization:

```cpp
template <MetricType mt>
struct VectorDistance {
    size_t d;
    float metric_arg;
    inline float operator()(const float* x, const float* y) const;
};
```

11 existing specializations (L2, IP, L1, Linf, Lp, Canberra, BrayCurtis, JensenShannon, Jaccard, NaNEuclidean, GOWER).

### 1.4 Dispatch Pattern

**File:** `faiss/utils/extra_distances-inl.h:200-227`

`dispatch_VectorDistance()` uses a Consumer pattern - same dispatch serves pairwise distances, KNN, and DistanceComputer creation.

### 1.5 DistanceComputer Interface

**File:** `faiss/impl/DistanceComputer.h:25-60`

```cpp
struct DistanceComputer {
    virtual void set_query(const float* x) = 0;
    virtual float operator()(idx_t i) = 0;
    virtual void distances_batch_4(...);
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;
};
```

`FlatCodesDistanceComputer` extends this with `distance_to_code()` for flat indexes.

### 1.6 IndexFlat Search Flow

**File:** `faiss/IndexFlat.cpp:29-60`

```cpp
void IndexFlat::search(...) const {
    if (metric_type == METRIC_INNER_PRODUCT) {
        knn_inner_product(...);
    } else if (metric_type == METRIC_L2) {
        knn_L2sqr(...);
    } else {
        knn_extra_metrics(...);  // All other metrics go here
    }
}
```

### 1.7 Extra Metrics KNN Path

**File:** `faiss/utils/extra_distances.cpp:180-195`

```cpp
void knn_extra_metrics(...) {
    Run_knn_extra_metrics run;
    dispatch_VectorDistance(d, mt, metric_arg, run, x, y, nx, ny, k, distances, indexes, sel);
}
```

### 1.8 Index Factory

**File:** `faiss/index_factory.h:17-23`

```cpp
Index* index_factory(int d, const char* description,
                     MetricType metric = METRIC_L2, bool own_invlists = true);
```

Accepts any MetricType value - new metrics work automatically with factory.

---

## 2. C++ Extension Points

### 2.1 Adding New MetricType Values

**Files to modify:**
1. `faiss/MetricType.h` - Add enum values (e.g., starting at 100)
2. `faiss/MetricType.h` - Update `with_metric_type()` dispatch
3. `faiss/utils/extra_distances-inl.h` - Add `VectorDistance<>` specializations
4. `faiss/utils/extra_distances-inl.h` - Add dispatch cases

**What works automatically after this:**
- `IndexFlat::search()` routes to `knn_extra_metrics()`
- `pairwise_extra_distances()` works
- `get_extra_distance_computer()` works
- Python `pairwise_distances()` works
- `index_factory()` accepts the new metric

### 2.2 Custom Index Subclasses for Search Strategies

For strategies that change the search process (progressive elimination, dropout ensemble, etc.):

**Pattern:** Subclass `Index` and override `search()`, wrapping an inner `IndexFlat`.

**Reference:** `faiss/IndexFlat.cpp`, `faiss/IndexRefine.h` (refine pattern), `faiss/IndexPreTransform.h` (transform pattern)

Key files for Index subclass pattern:
- `faiss/Index.h:101-125` - Base Index interface
- `faiss/IndexFlat.h:21-74` - Flat index implementation
- `faiss/IndexPreTransform.h` - Wrapping pattern (transforms vectors before search)

### 2.3 SIMD Optimization Patterns

**File:** `faiss/utils/distances_fused/simdlib_based.h:15-32`

Platform-specific with `#if defined(__AVX2__) || defined(__aarch64__)`.

**File:** `faiss/utils/distances.h:41-65`

Batch-4 functions for SIMD throughput:
```cpp
void fvec_L2sqr_batch_4(const float* x,
    const float* y0, const float* y1, const float* y2, const float* y3,
    const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);
```

---

## 3. Benchmark & Testing Infrastructure

### 3.1 Benchmark Framework

**Directory:** `benchs/bench_fw/`
- `benchmark.py` - Core orchestration with Pareto optimization
- `descriptors.py` - Configuration data structures

### 3.2 Simple Benchmark Pattern

**File:** `benchs/bench_index_flat.py`

```python
nrun = 10; times = []
for run in range(nrun):
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    if run >= nrun // 5:  # skip warmup
        times.append(t1 - t0)
```

### 3.3 Datasets

**File:** `contrib/datasets.py`

- `SyntheticDataset` - Reproducible synthetic data
- `DatasetSIFT1M` - d=128, 1M vectors
- `DatasetBigANN` - d=128, up to 1B vectors
- `DatasetDeep1B` - d=96, up to 1B vectors

### 3.4 Evaluation

**File:** `contrib/evaluation.py`

- `knn_intersection_measure()` - Recall@K computation
- `OperatingPoints` - Pareto frontier management
- `RepeatTimer` - Timing with warmup

### 3.5 Test Patterns

**File:** `tests/test_extra_distances.py` - Compare FAISS vs scipy reference
**File:** `tests/test_distances_simd.cpp` - SIMD vs scalar reference

---

## 4. Strategy-to-Implementation Mapping

### Category A: Distance-Level (VectorDistance specialization)

These compute distance between two vectors with modified logic:

| Strategy | C++ Approach |
|----------|-------------|
| PA-01 LearnedWeights | Weighted L2: `sum(w[i] * (x[i]-y[i])^2)`. Weights stored as `metric_arg` or in a side structure |
| PA-03 MissingValueAdjusted | Modified L2 that skips NaN dimensions and renormalizes |

### Category B: Search-Level (Custom Index subclass)

These modify the search process itself:

| Strategy | C++ Approach |
|----------|-------------|
| ED-01 FixedElimination | Custom Index wrapping IndexFlat. Override `search()` with progressive column elimination |
| ED-02 AdaptiveDispersion | Same as ED-01 but with dispersion-based cutoff |
| ED-03 VarianceOrder | Same pattern, sample-based column ordering |
| ED-04 UncertaintyDeferred | Same pattern, deferred elimination on low confidence |
| ED-05 DropoutEnsemble | Multiple masked searches, integrated by voting/borda |
| PP-01 ParallelVoting | Parallel group searches with vote integration |
| PP-02 CoarseToFine | Multi-resolution search with progressive refinement |
| MR-01 LateralInhibition | Post-processing step on search results |
| MR-02 ContextCache | Cache wrapper around any Index |
| MR-03 ContrastiveLearning | Weight update mechanism (extends PA-01) |
| PA-02 ContextualWeights | Multi-weight-set selection before search |

### Category C: Composable (Modifier/Decorator)

These wrap other strategies:

| Strategy | C++ Approach |
|----------|-------------|
| MR-01 LateralInhibition | `IndexPostProcess` wrapper |
| MR-02 ContextCache | `IndexCache` wrapper |

---

## 5. Proposed C++ Architecture

```
faiss/
├── MetricType.h                        # Add METRIC_NEURO_WEIGHTED_L2, etc.
├── utils/
│   ├── extra_distances-inl.h           # VectorDistance<> for Category A
│   └── neuro_distances.h/cpp           # Helper functions for NeuroDistance
├── impl/
│   └── NeuroSearchStrategy.h           # Base class for Category B strategies
├── IndexNeuroElimination.h/cpp         # ED-01 through ED-04
├── IndexNeuroDropoutEnsemble.h/cpp     # ED-05
├── IndexNeuroParallelVoting.h/cpp      # PP-01
├── IndexNeuroCoarseToFine.h/cpp        # PP-02
├── IndexNeuroLateralInhibition.h/cpp   # MR-01 (decorator)
├── IndexNeuroCache.h/cpp               # MR-02 (decorator)
└── IndexNeuroWeighted.h/cpp            # PA-01, PA-02, PA-03
```

Python factory usage:
```python
import faiss

# Category A: simple metric
index = faiss.index_factory(128, "Flat", faiss.METRIC_NEURO_WEIGHTED_L2)

# Category B: custom index (registered or direct)
index = faiss.IndexNeuroElimination(128, faiss.METRIC_L2, config)

# Decorator pattern
base = faiss.IndexFlat(128)
index = faiss.IndexNeuroLateralInhibition(base, config)
```

---

## 6. Key Files to Modify/Create

### Modify:
| File | Change |
|------|--------|
| `faiss/MetricType.h` | Add enum values + dispatch cases |
| `faiss/utils/extra_distances-inl.h` | VectorDistance specializations for Category A |
| `CMakeLists.txt` | Add new source files |
| `faiss/python/swigfaiss.swig` | Expose new classes to Python |

### Create:
| File | Purpose |
|------|---------|
| `faiss/IndexNeuro*.h/cpp` | One per strategy family |
| `faiss/impl/NeuroSearchStrategy.h` | Common base/utilities |
| `tests/test_neurodistance.cpp` | C++ unit tests |
| `tests/test_neurodistance.py` | Python integration tests |
| `benchs/bench_neurodistance.py` | Benchmark suite |

---

## 7. No Conflicting Implementations Found

- No existing "neural" or "bio-inspired" distance implementations
- No progressive elimination or adaptive search strategies
- No dropout-based ensemble search
- No lateral inhibition post-processing
- The contrib/ module has no similar experimental features

---

## Gate 0 Pass Criteria

- [x] Research mode determined: New Feature (C++ implementation)
- [x] Existing patterns identified: MetricType, VectorDistance, Index subclass, dispatch
- [x] No conflicting implementations found
- [x] Two integration levels identified (distance-level vs search-level)
- [x] C++ architecture proposed with Python factory for testing
