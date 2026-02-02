# TRD: NeuroDistance - Technical Architecture

**Feature:** neuro-distance-strategies
**Version:** 1.0
**Date:** 2026-02-02
**Inputs:** PRD v1.0, Gate 0 Research

---

## 1. Architecture Overview

NeuroDistance extends FAISS at two integration levels:

```
┌────────────────────────────────────────────────────────────┐
│                     User Code (Python/C++)                  │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐   ┌──────────────────────────┐    │
│  │  Level 1: Metrics    │   │  Level 2: Search Index   │    │
│  │  (VectorDistance<>)  │   │  (Index subclass)        │    │
│  │                      │   │                          │    │
│  │  PA-01 WeightedL2    │   │  IndexNeuroElim (ED-01..04)│  │
│  │  PA-03 NaNWeighted   │   │  IndexNeuroDropout (ED-05) │  │
│  │                      │   │  IndexNeuroVoting (PP-01)  │  │
│  └──────────┬───────────┘   │  IndexNeuroCoarse (PP-02)  │  │
│             │               │  IndexNeuroWeighted (PA-01..02)│
│             │               └──────────┬───────────────┘    │
│             │                          │                     │
│  ┌──────────┴──────────────────────────┴──────────────┐     │
│  │          Level 3: Decorators (composable)          │     │
│  │                                                     │     │
│  │  IndexNeuroInhibition (MR-01) - wraps any Index    │     │
│  │  IndexNeuroCache (MR-02) - wraps any Index         │     │
│  │  IndexNeuroContrastive (MR-03) - wraps PA-01/PA-02 │     │
│  └─────────────────────────────────────────────────────┘     │
│                          │                                    │
├──────────────────────────┴───────────────────────────────────┤
│                    FAISS Core (unchanged)                      │
│  MetricType enum │ IndexFlat │ knn_extra_metrics │ SWIG      │
└──────────────────────────────────────────────────────────────┘
```

**Design principle:** FAISS core is modified minimally (only MetricType enum). All strategy logic lives in new files.

---

## 2. Component Design

### 2.1 Component C1: MetricType Extensions

**Purpose:** Register new distance metrics for Level 1 strategies.
**Integration point:** `faiss/MetricType.h`

**New enum values:**

```
METRIC_NEURO_WEIGHTED_L2 = 100    // Weighted L2 with external weight vector
METRIC_NEURO_NAN_WEIGHTED = 101   // Missing-value-aware weighted L2
```

Only 2 new enum values needed. The remaining 11 strategies operate at search level, not distance level.

**Interface contract:**
- Each new metric implements `VectorDistance<METRIC_NEURO_*>::operator()(const float* x, const float* y) -> float`
- Each is registered in `dispatch_VectorDistance()` switch
- Each is registered in `with_metric_type()` dispatch

**Weight storage mechanism:**
- Weights are stored in a side structure (not in `metric_arg`, which is a single float)
- The `VectorDistance<>` specialization accesses weights via a thread-local or instance-level pointer
- Alternative: encode weight index in `metric_arg` and use a global weight registry

### 2.2 Component C2: NeuroSearchParameters

**Purpose:** Extended search parameters for NeuroDistance strategies.
**Pattern follows:** `SearchParameters` base class (faiss/Index.h:88-93), `IndexRefineSearchParameters` (faiss/IndexRefine.h:14-19)

```cpp
struct NeuroSearchParameters : SearchParameters {
    // Common to all neuro strategies
    bool collect_stats = false;        // collect search metadata
};

struct NeuroEliminationParams : NeuroSearchParameters {
    std::vector<int> column_order;     // empty = default order
    float cutoff_percentile = 0.5f;
    int min_candidates = 0;            // 0 = auto (k*2)
};

struct NeuroDropoutParams : NeuroSearchParameters {
    int num_views = 5;
    float dropout_rate = 0.3f;
    std::string dropout_mode = "complementary";
    std::string integration_method = "borda";
    int top_k_per_view = 0;           // 0 = auto (k*2)
};

// ... similar for each strategy family
```

### 2.3 Component C3: NeuroSearchResult

**Purpose:** Extended result metadata beyond standard distances+indices.

```cpp
struct NeuroSearchStats {
    int64_t calculations_performed;
    int columns_used;
    float time_ms;
    std::unordered_map<std::string, float> strategy_metadata;
};
```

**Storage:** Attached to search via `NeuroSearchParameters::collect_stats`. Retrieved post-search via a method on the index.

### 2.4 Component C4: IndexNeuroElimination (ED-01..04)

**Purpose:** Progressive elimination search strategies.
**Pattern follows:** `IndexRefine` (wraps an inner index, overrides `search()`)

```
                    IndexNeuroElimination
                    ├── inner_index: IndexFlat*    // owns the data
                    ├── strategy: EliminationStrategy
                    ├── d: int (inherited)
                    ├── ntotal: idx_t (inherited)
                    │
                    ├── add(n, x)        → delegates to inner_index
                    ├── search(n, x, k)  → progressive elimination algorithm
                    ├── train(n, x)      → optional (for ED-03 variance sampling)
                    └── reset()          → delegates to inner_index
```

**EliminationStrategy** is an enum selecting between:
- `FIXED` (ED-01): Fixed column order, fixed cutoff
- `ADAPTIVE_DISPERSION` (ED-02): Cutoff varies with column dispersion
- `VARIANCE_ORDER` (ED-03): Columns reordered by sampled variance
- `UNCERTAINTY_DEFERRED` (ED-04): Defers elimination when uncertain

**Core algorithm interface:**

```cpp
// Internal: one elimination step
struct EliminationStep {
    std::vector<idx_t> surviving_candidates;
    int column_index;
    float cutoff_used;
    int eliminated_count;
};

// The search override:
void IndexNeuroElimination::search(
    idx_t n, const float* x, idx_t k,
    float* distances, idx_t* labels,
    const SearchParameters* params) const;
```

**Data access pattern:**
- Needs column-wise access to the stored vectors: `inner_index->get_xb()[vector_idx * d + col_idx]`
- For ED-03, needs a variance cache computed during `train()` or lazily on first search

### 2.5 Component C5: IndexNeuroDropoutEnsemble (ED-05)

**Purpose:** Multiple masked searches with result integration.
**Pattern follows:** Standalone Index wrapping IndexFlat

```
                    IndexNeuroDropoutEnsemble
                    ├── inner_index: IndexFlat*
                    ├── num_views: int
                    ├── dropout_rate: float
                    ├── dropout_mode: DropoutMode enum
                    ├── integration_method: IntegrationMethod enum
                    │
                    ├── add(n, x)        → delegates to inner_index
                    ├── search(n, x, k)  → multi-view search + integration
                    └── reset()          → delegates to inner_index
```

**View generation:**

```cpp
enum DropoutMode {
    RANDOM,         // independent random masks per view
    COMPLEMENTARY,  // minimize overlap, guarantee coverage
    STRUCTURED,     // semantic grouping (first half, second half, etc.)
    ADVERSARIAL     // each view excludes previous view's top columns
};
```

**Integration methods:**

```cpp
enum IntegrationMethod {
    VOTING,      // count appearances across views
    BORDA,       // sum of ranks (lower = better)
    MEAN_DIST,   // average distances from views where candidate appeared
    FULL_RERANK  // full distance recomputation on union of candidates
};
```

**Internal flow:**
1. Generate `num_views` column masks based on `dropout_mode`
2. For each view: create masked query + masked database subset, run brute-force search
3. Collect top-k candidates per view
4. Integrate results using `integration_method`
5. Return final top-k

### 2.6 Component C6: IndexNeuroWeighted (PA-01, PA-02)

**Purpose:** Weighted distance search with learnable weights.

```
                    IndexNeuroWeighted
                    ├── inner_index: IndexFlat*
                    ├── weights: std::vector<float>     // per-dimension weights
                    ├── learning_rate: float
                    ├── weight_decay: float
                    ├── min_weight: float
                    │
                    ├── add(n, x)           → delegates to inner_index
                    ├── search(n, x, k)     → weighted distance search
                    ├── feedback(query, ground_truth) → update weights
                    ├── save_weights(path)   → serialize weights
                    ├── load_weights(path)   → deserialize weights
                    └── reset()             → delegates to inner_index
```

**PA-02 extension (ContextualWeights):**

```
                    IndexNeuroContextualWeighted : IndexNeuroWeighted
                    ├── num_clusters: int
                    ├── cluster_weights: vector<vector<float>>  // per-cluster
                    ├── cluster_centroids: vector<vector<float>>
                    │
                    ├── search(n, x, k)     → classify query → select weights → search
                    ├── feedback(...)        → update cluster-specific weights
                    └── train(n, x)          → cluster the query space
```

### 2.7 Component C7: IndexNeuroParallelVoting (PP-01)

**Purpose:** Divide dimensions into groups, search each group, integrate by voting.

```
                    IndexNeuroParallelVoting
                    ├── inner_index: IndexFlat*
                    ├── num_groups: int
                    ├── grouping_method: GroupingMethod enum
                    ├── integration_method: IntegrationMethod enum
                    │
                    ├── search(n, x, k) → group-wise search + vote integration
                    └── ...
```

**GroupingMethod:**
- `CONSECUTIVE` - Groups of consecutive dimensions
- `INTERLEAVED` - Round-robin assignment
- `VARIANCE_BASED` - Group by similar variance (requires train)

### 2.8 Component C8: IndexNeuroCoarseToFine (PP-02)

**Purpose:** Multi-resolution progressive refinement.

```
                    IndexNeuroCoarseToFine
                    ├── inner_index: IndexFlat*
                    ├── n_levels: int
                    ├── reduction_per_level: vector<int>   // e.g., [8, 4, 1]
                    ├── cutoff_per_level: vector<float>    // e.g., [0.3, 0.5, 0.7]
                    ├── coarse_representations: vector<vector<float>>  // precomputed
                    │
                    ├── add(n, x)       → compute multi-resolution representations
                    ├── search(n, x, k) → coarse filter → medium filter → fine ranking
                    └── reset()         → clear all levels
```

**Multi-resolution storage:**
- Level 1 (coarsest): average of every `reduction[0]` consecutive dimensions
- Level 2: average of every `reduction[1]` consecutive dimensions
- Level 3: original dimensions

Precomputed at `add()` time. Stored as separate float arrays per level.

### 2.9 Component C9: IndexNeuroInhibition (MR-01) - Decorator

**Purpose:** Post-process search results to remove near-duplicates.
**Pattern follows:** `NegativeDistanceComputer` decorator pattern, `IndexPreTransform` wrapper pattern

```
                    IndexNeuroInhibition : Index
                    ├── sub_index: Index*         // any index
                    ├── own_fields: bool
                    ├── similarity_threshold: float
                    ├── max_per_cluster: int
                    │
                    ├── add(n, x)       → delegates to sub_index
                    ├── search(n, x, k) → sub_index->search(n, x, k_expanded)
                    │                     → cluster similar results
                    │                     → keep max_per_cluster per cluster
                    │                     → return top-k diverse results
                    └── ...delegates everything else to sub_index
```

**Key:** Requests `k * expansion_factor` from sub_index, then filters down to `k` diverse results.

### 2.10 Component C10: IndexNeuroCache (MR-02) - Decorator

**Purpose:** Cache recent query results and configurations.
**Pattern follows:** Decorator wrapping any Index

```
                    IndexNeuroCache : Index
                    ├── sub_index: Index*
                    ├── own_fields: bool
                    ├── cache_size: int
                    ├── similarity_threshold: float
                    ├── cache: LRUCache<query_hash, CacheEntry>
                    │
                    ├── search(n, x, k) → check cache for similar query
                    │                     → if hit: return cached result
                    │                     → if miss: sub_index->search() + cache result
                    └── ...delegates everything else to sub_index
```

### 2.11 Component C11: ContrastiveLearning (MR-03) - Extension to PA-01

**Purpose:** Faster weight convergence using positive+negative examples.
**Not a separate Index.** It extends `IndexNeuroWeighted` with an alternative `feedback()` method.

```cpp
struct ContrastiveFeedback {
    const float* query;
    std::vector<idx_t> positive_ids;   // known good results
    std::vector<idx_t> negative_ids;   // known bad results
    float positive_weight = 1.0f;
    float negative_weight = 0.5f;
    bool hard_negative_mining = true;
};

// Added to IndexNeuroWeighted:
void feedback_contrastive(const ContrastiveFeedback& fb);
```

---

## 3. Interface Definitions

### 3.1 Common C++ Interface

All NeuroDistance indexes share:

```cpp
// Base for all NeuroDistance indexes
struct IndexNeuro : Index {
    Index* inner_index;     // the underlying flat index (owns the data)
    bool own_inner;         // ownership flag

    // Common constructor
    IndexNeuro(Index* inner_index, bool own_inner = false);

    // Delegate storage operations
    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;

    // Each subclass overrides search()
    virtual void search(...) const override = 0;

    ~IndexNeuro() override;
};
```

### 3.2 Python Interface (via SWIG)

```python
# Level 1: Simple metric (works with index_factory)
index = faiss.IndexFlat(128, faiss.METRIC_NEURO_WEIGHTED_L2)

# Level 2: Search-level strategies
inner = faiss.IndexFlatL2(128)
inner.add(xb)

# Progressive elimination
index = faiss.IndexNeuroElimination(inner, faiss.NEURO_ADAPTIVE_DISPERSION)
params = faiss.NeuroEliminationParams()
params.cutoff_percentile = 0.5
D, I = index.search(xq, 10, params)

# Dropout ensemble
index = faiss.IndexNeuroDropoutEnsemble(inner)
index.num_views = 5
index.dropout_rate = 0.3
D, I = index.search(xq, 10)

# Weighted with learning
index = faiss.IndexNeuroWeighted(inner)
D, I = index.search(xq, 10)
index.feedback(query, ground_truth_ids)
index.save_weights("weights.bin")

# Decorators (composable)
base = faiss.IndexNeuroElimination(inner, faiss.NEURO_FIXED)
diverse = faiss.IndexNeuroInhibition(base)
cached = faiss.IndexNeuroCache(diverse, cache_size=100)
D, I = cached.search(xq, 10)
```

### 3.3 Benchmark Interface

```python
# Uses existing FAISS benchmark patterns
from faiss.contrib.datasets import SyntheticDataset
from faiss.contrib.evaluation import knn_intersection_measure

ds = SyntheticDataset(d=128, nb=100000, nq=1000)

# Ground truth
index_gt = faiss.IndexFlatL2(128)
index_gt.add(ds.get_database())
D_gt, I_gt = index_gt.search(ds.get_queries(), 10)

# NeuroDistance strategy
inner = faiss.IndexFlatL2(128)
inner.add(ds.get_database())
index_neuro = faiss.IndexNeuroElimination(inner, faiss.NEURO_ADAPTIVE_DISPERSION)
D_neuro, I_neuro = index_neuro.search(ds.get_queries(), 10)

# Compare
recall = knn_intersection_measure(I_gt, I_neuro)
print(f"Recall@10: {recall:.4f}")
```

---

## 4. Data Architecture

### 4.1 Storage (Conceptual)

| Data | Storage Pattern | Location |
|------|----------------|----------|
| Vector data | Flat float array (N x d) | `inner_index->codes` (existing IndexFlat) |
| Dimension weights | Float array (d) | `IndexNeuroWeighted::weights` |
| Cluster weights | Float matrix (num_clusters x d) | `IndexNeuroContextualWeighted::cluster_weights` |
| Coarse representations | Float arrays per level | `IndexNeuroCoarseToFine::coarse_representations` |
| Column variance cache | Float array (d) | `IndexNeuroElimination::variance_cache` |
| Query cache | Hash map (query_hash -> result) | `IndexNeuroCache::cache` |

### 4.2 Serialization

All NeuroDistance indexes support save/load following FAISS I/O patterns:
- Binary serialization of weights, caches, and configuration
- The inner index is serialized via standard FAISS `write_index()`/`read_index()`

---

## 5. Component Interaction Diagram

### 5.1 Progressive Elimination Flow (ED-02)

```
search(queries, k=10)
│
├── For each query:
│   ├── candidates = {0, 1, ..., ntotal-1}
│   ├── For each column (in configured order):
│   │   ├── Compute distance on this column for all candidates
│   │   ├── Compute dispersion = std / mean
│   │   ├── Determine cutoff based on dispersion
│   │   │   ├── Low dispersion  → pass 80% (not discriminative)
│   │   │   ├── High dispersion → pass 30% (very discriminative)
│   │   │   └── Medium          → linear interpolation
│   │   ├── Sort candidates by column distance
│   │   ├── candidates = top cutoff% of candidates
│   │   └── If len(candidates) <= min_candidates: break
│   │
│   ├── Compute full distance for surviving candidates
│   └── Return top-k by full distance
│
└── Output: distances[n][k], labels[n][k]
```

### 5.2 Dropout Ensemble Flow (ED-05)

```
search(queries, k=10)
│
├── Generate num_views column masks
│   └── Mode: complementary → divide d columns into views with minimal overlap
│
├── For each view (parallelizable):
│   ├── masked_query = query[mask]
│   ├── masked_db = database[:, mask]  (or compute distances only on masked cols)
│   ├── D_view, I_view = brute_force_search(masked_query, masked_db, top_k_per_view)
│   └── Collect (I_view, ranks)
│
├── Integration (borda count):
│   ├── For each candidate across all views:
│   │   └── score[candidate] = sum(rank_in_each_view)
│   └── Return top-k by lowest score
│
└── Output: distances[n][k], labels[n][k]
```

### 5.3 Decorator Composition Flow

```
IndexNeuroCache → IndexNeuroInhibition → IndexNeuroElimination → IndexFlat
     │                    │                       │                   │
     │  search(q, k)      │                       │                   │
     ├──► cache lookup     │                       │                   │
     │    miss ──────────► │  search(q, k*3)       │                   │
     │                     ├──► get expanded results│                   │
     │                     │    ───────────────────►│  search(q, k*3)   │
     │                     │                        ├──► elimination    │
     │                     │                        │    search         │
     │                     │                        │    ──────────────►│ get_xb()
     │                     │                        │◄─── distances     │
     │                     │◄─── k*3 results        │                   │
     │                     ├──► cluster similar      │                   │
     │                     ├──► keep diverse top-k   │                   │
     │◄─── k diverse results│                       │                   │
     ├──► cache store       │                       │                   │
     └──► return            │                       │                   │
```

---

## 6. Security Architecture

| Concern | Mitigation |
|---------|------------|
| Buffer overflow in column access | Bounds checking on column indices; FAISS_THROW on invalid |
| Weight injection (malicious weights) | Validate weight dimensions match index dimensions on load |
| Cache poisoning | Cache entries are keyed by exact query hash; no user-controlled keys |
| Memory exhaustion (large caches) | LRU eviction with configurable max size |
| Thread safety | NeuroDistance indexes are not thread-safe for mutation (same as FAISS convention); search is const and safe for concurrent reads |

---

## 7. Error Handling

| Error Case | Behavior |
|-----------|----------|
| Strategy requires train() but not called | FAISS_THROW with message |
| Empty inner index (ntotal = 0) | Return empty results (k=0 behavior) |
| Invalid column order (out of bounds) | FAISS_THROW at search time |
| Weight vector size != d | FAISS_THROW at construction or load time |
| Feedback with invalid IDs | FAISS_THROW with message |
| NaN in query vector | Strategy-dependent: PA-03 handles gracefully, others may produce NaN distances |

---

## 8. Testing Architecture

### 8.1 C++ Unit Tests

**File:** `tests/test_neurodistance.cpp`

| Test Category | What is Tested |
|--------------|----------------|
| Metric correctness | VectorDistance<METRIC_NEURO_*> vs reference implementation |
| Elimination correctness | IndexNeuroElimination returns subset of brute-force results |
| Dropout correctness | IndexNeuroDropoutEnsemble recall >= threshold |
| Weight learning | IndexNeuroWeighted recall improves after feedback |
| Decorator composition | Stacked decorators produce valid results |
| Edge cases | Empty index, single vector, k > ntotal, d=1 |
| Serialization | Save/load round-trip preserves weights and config |

### 8.2 Python Integration Tests

**File:** `tests/test_neurodistance.py`

Pattern: Compare against `IndexFlatL2` brute-force ground truth using `knn_intersection_measure()`.

### 8.3 Benchmark Script

**File:** `benchs/bench_neurodistance.py`

All strategies benchmarked on:
- SyntheticDataset (d=32, d=128, d=512)
- SIFT1M (if available)

Output: Table with recall@1, recall@10, time_ms, calculations per strategy.

---

## 9. File Organization

```
faiss/
├── MetricType.h                          # MODIFY: add 2 enum values + dispatch
├── impl/
│   └── NeuroDistance.h                   # NEW: common base, enums, params structs
├── IndexNeuroElimination.h               # NEW: ED-01..04
├── IndexNeuroElimination.cpp             # NEW
├── IndexNeuroDropoutEnsemble.h           # NEW: ED-05
├── IndexNeuroDropoutEnsemble.cpp         # NEW
├── IndexNeuroWeighted.h                  # NEW: PA-01..03
├── IndexNeuroWeighted.cpp                # NEW
├── IndexNeuroParallelVoting.h            # NEW: PP-01
├── IndexNeuroParallelVoting.cpp          # NEW
├── IndexNeuroCoarseToFine.h              # NEW: PP-02
├── IndexNeuroCoarseToFine.cpp            # NEW
├── IndexNeuroInhibition.h                # NEW: MR-01 decorator
├── IndexNeuroInhibition.cpp              # NEW
├── IndexNeuroCache.h                     # NEW: MR-02 decorator
├── IndexNeuroCache.cpp                   # NEW
├── utils/
│   └── extra_distances-inl.h             # MODIFY: add 2 VectorDistance specializations
│
tests/
├── test_neurodistance.cpp                # NEW: C++ tests
├── test_neurodistance.py                 # NEW: Python tests
│
benchs/
└── bench_neurodistance.py                # NEW: Benchmark script
│
CMakeLists.txt                            # MODIFY: add new source files
faiss/python/swigfaiss.swig               # MODIFY: expose new classes
```

**Total new files:** 15 (8 header/source pairs + 3 test/bench)
**Modified files:** 4 (MetricType.h, extra_distances-inl.h, CMakeLists.txt, swigfaiss.swig)

---

## 10. PRD Feature → Component Mapping

| PRD User Story | Components Used | Priority |
|---------------|-----------------|----------|
| US-01: Basic strategy usage | C1, C2, C3, all IndexNeuro* | Phase 1 |
| US-02: Progressive elimination | C4 (IndexNeuroElimination) | Phase 1 |
| US-03: Dropout ensemble | C5 (IndexNeuroDropoutEnsemble) | Phase 2 |
| US-04: Adaptive weight learning | C6 (IndexNeuroWeighted), C11 | Phase 3 |
| US-05: Missing value search | C1 (METRIC_NEURO_NAN_WEIGHTED) or C6 | Phase 2 |
| US-06: Diverse results | C9 (IndexNeuroInhibition) | Phase 2 |
| US-07: Benchmark | Benchmark script + all strategies | All phases |
| US-08: Strategy combinations | C9, C10 decorators | Phase 2+ |

---

## 11. Constraints and Assumptions

### Technology-Agnostic (per TRD guidelines)

| Concept | Description (no product names) |
|---------|-------------------------------|
| Vector storage | Contiguous flat array, row-major |
| Distance computation | Compile-time dispatch via template specialization |
| Search orchestration | Subclass override of virtual search method |
| Composition | Decorator pattern with inner index delegation |
| Serialization | Binary format with version header |
| Bindings | Interface definition language generating target-language wrappers |
| Build system | Build configuration adding new source files to existing library target |
| Testing | Unit test framework with assertion macros |

### Assumptions

1. Inner index is always `IndexFlat` (or compatible flat storage with `get_xb()`)
2. Data fits in memory (no streaming/disk-based strategies)
3. Single-threaded search (parallelism within a strategy, not across queries)
4. Float32 vectors only (no int8/float16 for MVP)

---

## Gate 2 Pass Criteria

- [x] All 13 PRD strategies mapped to components (C1-C11)
- [x] Component boundaries are clear (each has single responsibility)
- [x] Interfaces are technology-agnostic (no specific products named in architecture)
- [x] Two integration levels defined (metric-level vs search-level)
- [x] Decorator pattern enables composability (US-08)
- [x] Data flow diagrams for key algorithms
- [x] File organization defined
- [x] Testing architecture specified
