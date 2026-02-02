# Gate 3: Task Breakdown - NeuroDistance Strategies

**Feature:** neuro-distance-strategies
**Version:** 1.0
**Date:** 2026-02-02
**Inputs:** PRD v1.0, TRD v1.0

---

## Task Overview

| Phase | Tasks | Strategies Delivered | Cumulative |
|-------|-------|---------------------|------------|
| Phase 1: Core MVP | T01-T05 | ED-01, ED-02 + infrastructure | 2 strategies |
| Phase 2: Elimination Complete | T06-T11 | ED-03, ED-04, ED-05, PA-03, MR-01 | 7 strategies |
| Phase 3: Learning | T12-T14 | PA-01, PA-02, MR-03 | 10 strategies |
| Phase 4: Performance | T15-T17 | PP-01, PP-02, MR-02 | 13 strategies |
| Phase 5: Benchmark & Polish & use docs (py) | T18-T19 | Full benchmark suite, docs | Complete |

---

## Phase 1: Core MVP

### T01: Infrastructure - Base Classes, Enums, and Build Setup

**Delivers:** Foundation for all subsequent strategies. No user-facing functionality yet, but build compiles and tests run.

**Components:** C1 (MetricType), C2 (NeuroSearchParameters), C3 (NeuroSearchStats), IndexNeuro base

**What to implement:**

1. **`faiss/impl/NeuroDistance.h`** - Common header with:
   - `EliminationStrategy` enum (`FIXED`, `ADAPTIVE_DISPERSION`, `VARIANCE_ORDER`, `UNCERTAINTY_DEFERRED`)
   - `DropoutMode` enum (`RANDOM`, `COMPLEMENTARY`, `STRUCTURED`, `ADVERSARIAL`)
   - `IntegrationMethod` enum (`VOTING`, `BORDA`, `MEAN_DIST`, `FULL_RERANK`)
   - `NeuroSearchParameters` struct (extends `SearchParameters`)
   - `NeuroEliminationParams` struct
   - `NeuroSearchStats` struct
   - `IndexNeuro` base class:
     ```cpp
     struct IndexNeuro : Index {
         Index* inner_index;
         bool own_inner;
         IndexNeuro(Index* inner_index, bool own_inner = false);
         void add(idx_t n, const float* x) override;
         void reset() override;
         void reconstruct(idx_t key, float* recons) const override;
         ~IndexNeuro() override;
     };
     ```

2. **`faiss/impl/NeuroDistance.cpp`** - Implementation of base class delegation

3. **`faiss/MetricType.h`** - Add:
   ```cpp
   METRIC_NEURO_WEIGHTED_L2 = 100,
   METRIC_NEURO_NAN_WEIGHTED = 101,
   ```
   Update `with_metric_type()` switch with 2 new cases.

4. **`faiss/utils/extra_distances-inl.h`** - Add:
   - `VectorDistance<METRIC_NEURO_WEIGHTED_L2>` specialization (placeholder: same as L2 for now)
   - `VectorDistance<METRIC_NEURO_NAN_WEIGHTED>` specialization (placeholder: same as NaNEuclidean for now)
   - 2 new `DISPATCH_VD()` entries

5. **`faiss/CMakeLists.txt`** - Add `impl/NeuroDistance.cpp` to `FAISS_SRC`, `impl/NeuroDistance.h` to `FAISS_HEADERS`

6. **`faiss/python/swigfaiss.swig`** - Add `#include` and `%include` for `impl/NeuroDistance.h`

7. **`tests/test_neurodistance.cpp`** - Skeleton test file with:
   - Test that `METRIC_NEURO_WEIGHTED_L2` compiles and runs with IndexFlat
   - Test that IndexNeuro base delegates add/reset correctly

8. **`tests/CMakeLists.txt`** - Add `test_neurodistance.cpp` to `FAISS_TEST_SRC`

**Dependencies:** None
**Testing:** Build compiles. C++ test passes. Python `import faiss; faiss.METRIC_NEURO_WEIGHTED_L2` works.

---

### T02: ED-01 FixedElimination

**Delivers:** First working NeuroDistance strategy. Users can search with progressive elimination using fixed column order and fixed cutoff.

**Components:** C4 (IndexNeuroElimination with FIXED strategy)

**What to implement:**

1. **`faiss/IndexNeuroElimination.h`**:
   ```cpp
   struct IndexNeuroElimination : IndexNeuro {
       EliminationStrategy strategy;
       std::vector<int> column_order;     // default: reversed
       float cutoff_percentile;           // default: 0.5
       int min_candidates;                // default: 0 (= k*2)

       IndexNeuroElimination(Index* inner, EliminationStrategy strategy = FIXED);

       void search(idx_t n, const float* x, idx_t k,
                   float* distances, idx_t* labels,
                   const SearchParameters* params = nullptr) const override;
   };
   ```

2. **`faiss/IndexNeuroElimination.cpp`** - ED-01 algorithm:
   - Initialize candidates = {0..ntotal-1}
   - For each column in `column_order`:
     - Compute `|query[col] - candidate[col]|` for all candidates
     - Sort candidates by this single-column distance
     - Keep top `cutoff_percentile` fraction
     - If candidates <= `min_candidates`: stop
   - Compute full L2 distance for survivors
   - Return top-k

3. **Update `faiss/CMakeLists.txt`** - Add source + header

4. **Update `faiss/python/swigfaiss.swig`** - Expose IndexNeuroElimination

5. **Tests in `tests/test_neurodistance.cpp`**:
   - ED-01 returns valid indices (subset of 0..ntotal-1)
   - ED-01 with cutoff=1.0 equals brute force (no elimination)
   - ED-01 recall@10 >= 80% on SyntheticDataset(d=32, nb=10000, nq=100)
   - ED-01 performs fewer calculations than brute force

6. **Python test in `tests/test_neurodistance.py`** (create file):
   - Basic search via Python API
   - Compare recall vs IndexFlatL2

**Dependencies:** T01
**Testing:** Recall@10 >= 85% with 50% fewer calculations on d=32 synthetic data.

---

### T03: ED-02 AdaptiveDispersion

**Delivers:** Improved elimination that adapts cutoff based on column dispersion.

**Components:** C4 (IndexNeuroElimination with ADAPTIVE_DISPERSION strategy)

**What to implement:**

1. **Extend `IndexNeuroElimination.cpp`** with ADAPTIVE_DISPERSION branch:
   - Same column iteration as ED-01
   - At each column, compute `dispersion = std(distances) / mean(distances)`
   - If dispersion < `dispersion_low` (0.3): cutoff = 0.8 (pass most)
   - If dispersion > `dispersion_high` (0.7): cutoff = 0.3 (aggressive)
   - Otherwise: linear interpolation

2. **Add params to `NeuroDistance.h`**:
   ```cpp
   struct NeuroAdaptiveParams : NeuroEliminationParams {
       float dispersion_low = 0.3f;
       float dispersion_high = 0.7f;
       float cutoff_low_dispersion = 0.8f;
       float cutoff_high_dispersion = 0.3f;
   };
   ```

3. **Tests**:
   - ED-02 recall >= ED-01 recall (on same dataset)
   - ED-02 overhead vs ED-01 < 10%
   - ED-02 recall@10 >= 90% on synthetic data

**Dependencies:** T02
**Testing:** Recall@10 >= 90%, overhead vs ED-01 < 10%.

---

### T04: Benchmark Script (Basic)

**Delivers:** Ability to compare strategies against brute-force baseline.

**What to implement:**

1. **`benchs/bench_neurodistance.py`**:
   ```python
   # Uses contrib/datasets.py SyntheticDataset
   # Runs: IndexFlatL2 (baseline), ED-01, ED-02
   # For each: measures recall@1, recall@10, time_ms
   # Outputs table to stdout
   # Configurable: d, nb, nq, k via argparse
   ```

2. Follow existing pattern from `benchs/bench_index_flat.py`:
   - Warmup + multiple runs
   - Ground truth from IndexFlatL2
   - Recall via `knn_intersection_measure()` or manual computation

**Dependencies:** T02, T03
**Testing:** Script runs without errors. Output table is readable.

---

### T05: Phase 1 Validation

**Delivers:** Confidence that MVP works. Not a code task - validation and documentation.

**What to do:**

1. Run benchmark on multiple dataset sizes: d={32, 64, 128}, nb={10k, 100k}
2. Validate ED-01 and ED-02 hypothesis criteria from PRD:
   - ED-01: Recall >= 85% with 50% fewer calculations
   - ED-02: Recall +5% vs ED-01 with < 5% overhead
3. Document results in `docs/pre-dev/neuro-distance-strategies/validation_phase1.md`
4. Identify any architectural issues before Phase 2

**Dependencies:** T04
**Testing:** Hypotheses validated or documented as "needs adjustment".

---

## Phase 2: Elimination Family Complete

### T06: ED-03 VarianceOrder

**Delivers:** Dynamic column ordering based on sampled variance.

**What to implement:**

1. **Extend `IndexNeuroElimination`** with VARIANCE_ORDER branch:
   - Override `train()`: sample `fracao_amostra` (5%) of vectors
   - Compute per-column variance of distances to sample centroid
   - Store sorted column order in `variance_cache`
   - `search()` uses this order instead of user-provided/reversed

2. **Add to params:**
   ```cpp
   float sample_fraction = 0.05f;
   bool cache_order = true;
   ```

3. **Tests:**
   - ED-03 recall > random column order by >= 10%
   - ED-03 train() computes valid column order
   - ED-03 with cache_order=true reuses order across searches

**Dependencies:** T03
**Testing:** +15% recall vs random column order.

---

### T07: ED-04 UncertaintyDeferred

**Delivers:** Elimination that defers decisions when uncertain.

**What to implement:**

1. **Extend `IndexNeuroElimination`** with UNCERTAINTY_DEFERRED branch:
   - At each column, compute dispersion
   - If dispersion < `threshold_confidence` (0.4): accumulate distance, don't eliminate
   - Track `accumulated_columns` counter
   - If `accumulated_columns >= max_accumulated` (3): eliminate based on combined distance
   - Otherwise: normal elimination

2. **Add to params:**
   ```cpp
   float confidence_threshold = 0.4f;
   int max_accumulated_columns = 3;
   std::string accumulation_mode = "sum"; // "sum" or "mean"
   ```

3. **Tests:**
   - ED-04 recall >= ED-02 recall
   - ED-04 is more robust to noise (add gaussian noise to 20% of dimensions, measure recall degradation vs ED-01)

**Dependencies:** T03
**Testing:** Recall@10 >= 93%, robust to noise.

---

### T08: ED-05 DropoutEnsemble

**Delivers:** Multi-view search with dropout masks and vote integration.

**Components:** C5 (IndexNeuroDropoutEnsemble)

**What to implement:**

1. **`faiss/IndexNeuroDropoutEnsemble.h`**:
   ```cpp
   struct IndexNeuroDropoutEnsemble : IndexNeuro {
       int num_views = 5;
       float dropout_rate = 0.3f;
       DropoutMode dropout_mode = COMPLEMENTARY;
       IntegrationMethod integration = BORDA;
       int top_k_per_view = 0;  // 0 = k*2

       void search(...) const override;
   };
   ```

2. **`faiss/IndexNeuroDropoutEnsemble.cpp`**:
   - `generate_masks()`: based on dropout_mode, create column masks
   - For each view: compute distances using only masked columns
   - `integrate_results()`: combine per-view results using chosen method
   - Implement all 4 dropout modes and all 4 integration methods

3. **Update CMakeLists.txt, swigfaiss.swig**

4. **Tests:**
   - ED-05 recall >= 93% on clean data
   - ED-05 recall degradation < 10% when 20% of dimensions have noise
   - All 4 dropout modes produce valid results
   - All 4 integration methods produce valid results
   - Candidates appear in >= 3 views in 80% of cases (for complementary mode)

**Dependencies:** T01
**Testing:** Recall >= 95% clean, < 10% degradation with noise.

---

### T09: PA-03 MissingValueAdjusted

**Delivers:** Search that handles NaN/missing values gracefully.

**What to implement:**

1. **Implement `VectorDistance<METRIC_NEURO_NAN_WEIGHTED>`** properly in `extra_distances-inl.h`:
   - Like NaNEuclidean but with configurable weight reduction: `weight = (1 - missing_rate)^2`
   - Skip dimensions where query or database vector is NaN
   - Renormalize by present dimensions

2. **Tests:**
   - With 0% missing: result identical to L2
   - With 30% missing: degradation < 15% vs baseline (L2 on complete data)
   - With > 80% missing in a column: that column is effectively ignored

3. **Python test:** Create dataset with NaN injected, compare recall

**Dependencies:** T01
**Testing:** < 15% degradation at 30% missing.

---

### T10: MR-01 LateralInhibition (Decorator)

**Delivers:** Post-processing decorator that diversifies results.

**Components:** C9 (IndexNeuroInhibition)

**What to implement:**

1. **`faiss/IndexNeuroInhibition.h`**:
   ```cpp
   struct IndexNeuroInhibition : Index {
       Index* sub_index;
       bool own_fields;
       float similarity_threshold = 0.1f;
       int max_per_cluster = 3;
       float k_expansion = 3.0f;

       IndexNeuroInhibition(Index* sub_index, bool own = false);
       void search(...) const override;
       // delegates add, reset, reconstruct to sub_index
   };
   ```

2. **`faiss/IndexNeuroInhibition.cpp`**:
   - `search()`: call `sub_index->search(n, x, k * k_expansion, ...)`
   - For each query's expanded results: group similar candidates
   - Keep `max_per_cluster` best from each group
   - Return top-k from diverse set

3. **Update CMakeLists.txt, swigfaiss.swig**

4. **Tests:**
   - Diversity metric increases >= 20% vs raw results
   - Recall loss < 5%
   - Works wrapping IndexFlatL2, IndexNeuroElimination, and IndexNeuroDropoutEnsemble

**Dependencies:** T01
**Testing:** +20% diversity, < 5% recall loss.

---

### T11: Phase 2 Validation

**Delivers:** Benchmark of all 7 strategies + validation.

**What to do:**

1. Extend `benchs/bench_neurodistance.py` to include ED-03, ED-04, ED-05, PA-03, MR-01
2. Add noise robustness test (compare strategies on clean vs noisy data)
3. Add missing value test (PA-03 specific)
4. Validate all Phase 2 hypothesis criteria
5. Document results in `docs/pre-dev/neuro-distance-strategies/validation_phase2.md`
6. Test decorator composition: `IndexNeuroInhibition(IndexNeuroElimination(...))`

**Dependencies:** T06-T10
**Testing:** All Phase 2 hypotheses validated.

---

## Phase 3: Learning Strategies

### T12: PA-01 LearnedWeights

**Delivers:** Index that learns per-dimension weights from feedback.

**Components:** C6 (IndexNeuroWeighted)

**What to implement:**

1. **`faiss/IndexNeuroWeighted.h`**:
   ```cpp
   struct IndexNeuroWeighted : IndexNeuro {
       std::vector<float> weights;   // per-dimension, size d
       float learning_rate = 0.05f;
       float weight_decay = 0.99f;
       float min_weight = 0.01f;

       IndexNeuroWeighted(Index* inner);
       void search(...) const override;   // weighted L2 distance
       void feedback(const float* query, const idx_t* ground_truth, idx_t k_gt);
       void save_weights(const char* path) const;
       void load_weights(const char* path);
   };
   ```

2. **`faiss/IndexNeuroWeighted.cpp`**:
   - `search()`: compute `sum(weights[i] * (query[i] - vec[i])^2)` for all vectors
   - `feedback()`: analyze which dimensions contributed to correct/incorrect rankings, adjust weights
   - Weight serialization: binary format with dimension check

3. **Update CMakeLists.txt, swigfaiss.swig**

4. **Tests:**
   - Initial recall (uniform weights) matches L2 baseline
   - After 1000 feedback iterations: recall improves >= 5%
   - Weights converge (variance of weight updates decreases)
   - Save/load round-trip preserves weights exactly

**Dependencies:** T01
**Testing:** +5% recall after 1000 queries.

---

### T13: PA-02 ContextualWeights

**Delivers:** Per-query-type weight selection.

**Components:** C6 extended (IndexNeuroContextualWeighted)

**What to implement:**

1. **`IndexNeuroContextualWeighted`** extending `IndexNeuroWeighted`:
   - `num_clusters` weight sets
   - `train()`: cluster the query space using simple k-means on query features (magnitude, sparsity)
   - `search()`: classify incoming query → select cluster's weights → weighted search
   - `feedback()`: update the correct cluster's weights

2. **Tests:**
   - Recall >= PA-01 on heterogeneous query sets
   - +3% recall on "difficult" queries (identified as those where PA-01 has low recall)

**Dependencies:** T12
**Testing:** +3% recall on difficult queries vs PA-01.

---

### T14: MR-03 ContrastiveLearning

**Delivers:** Faster weight convergence using positive+negative examples.

**Components:** C11 (extension to IndexNeuroWeighted)

**What to implement:**

1. **Add to `IndexNeuroWeighted`**:
   ```cpp
   void feedback_contrastive(
       const float* query,
       const idx_t* positives, idx_t n_pos,
       const idx_t* negatives, idx_t n_neg,
       float pos_weight = 1.0f, float neg_weight = 0.5f);
   ```

2. **Algorithm:**
   - For each dimension: compute margin = mean_negative_dist - mean_positive_dist
   - If margin > 0: dimension helps separate → increase weight
   - If margin < 0: dimension confuses → decrease weight
   - Hard negative mining: focus on negatives closest to query

3. **Tests:**
   - Converges to same recall as PA-01 in 2x fewer iterations
   - Weight stability (less oscillation than PA-01)

**Dependencies:** T12
**Testing:** Same recall in 2x fewer iterations.

---

## Phase 4: Performance & Caching

### T15: PP-01 ParallelVoting

**Delivers:** Group-wise parallel search with vote integration.

**Components:** C7 (IndexNeuroParallelVoting)

**What to implement:**

1. **`faiss/IndexNeuroParallelVoting.h/cpp`**:
   - Divide d dimensions into `num_groups` groups
   - For each group: compute partial distances, find top-k candidates
   - Integrate by voting/borda/mean
   - Grouping methods: consecutive, interleaved

2. **Update CMakeLists.txt, swigfaiss.swig**

3. **Tests:**
   - Recall >= 92%
   - Lower variance than single-pass elimination
   - All grouping and integration methods work

**Dependencies:** T01
**Testing:** Recall >= 92%, lower variance than ED-01.

---

### T16: PP-02 CoarseToFine

**Delivers:** Multi-resolution progressive refinement.

**Components:** C8 (IndexNeuroCoarseToFine)

**What to implement:**

1. **`faiss/IndexNeuroCoarseToFine.h/cpp`**:
   - `add()`: precompute coarse representations (avg of dimension groups) at each level
   - `search()`: Level 1 (coarsest) → eliminate 70% → Level 2 → eliminate 50% → Level 3 (full) → final ranking
   - Configurable levels, reduction factors, cutoff per level

2. **Update CMakeLists.txt, swigfaiss.swig**

3. **Tests:**
   - Recall >= 88%
   - 60%+ calculation reduction
   - Scales well with dimensionality (test d=32, 128, 512)

**Dependencies:** T01
**Testing:** Recall >= 88%, 60%+ calc reduction.

---

### T17: MR-02 ContextCache (Decorator)

**Delivers:** Cache decorator for repeated similar queries.

**Components:** C10 (IndexNeuroCache)

**What to implement:**

1. **`faiss/IndexNeuroCache.h/cpp`**:
   - LRU cache with configurable size
   - Query hashing (discretize to grid for similarity matching)
   - `search()`: hash query → check cache → hit: return cached → miss: search + cache
   - Cache invalidation on `add()` or `reset()`

2. **Update CMakeLists.txt, swigfaiss.swig**

3. **Tests:**
   - Identical queries return cached results (faster)
   - Similar queries (within threshold) return cached results
   - Cache miss falls through to sub_index correctly
   - Cache is invalidated on add()

**Dependencies:** T01
**Testing:** 30%+ speedup on cache hits.

---

## Phase 5: Benchmark & Polish

### T18: Full Benchmark Suite

**Delivers:** Comprehensive comparison of all 13 strategies.

**What to implement:**

1. **Extend `benchs/bench_neurodistance.py`**:
   - All 13 strategies + brute-force baseline
   - Multiple datasets: Synthetic (d=32, 128, 512), SIFT1M if available
   - Noise robustness test (inject noise, measure degradation)
   - Missing value test (inject NaN, measure degradation)
   - Combo tests (recommended combinations from PRD)
   - Output: full results table + Pareto plot data (CSV)

2. **Hypothesis validation table:** For each of the 13 hypotheses from PRD, mark VALIDATED / NOT VALIDATED / PARTIAL with evidence.

**Dependencies:** T05-T17
**Testing:** All 13 strategies benchmarked, all hypotheses evaluated.

---

### T19: Documentation and Cleanup

**Delivers:** User documentation and code cleanup.

**What to do:**

1. Docstrings in all C++ headers (follow FAISS style)
2. Python usage examples (as comments in benchmark or separate README section)
3. Parameter tuning guide (which params matter most per strategy)
4. Recommended combinations with use cases
5. Final validation document: `docs/pre-dev/neuro-distance-strategies/validation_final.md`

**Dependencies:** T18
**Testing:** All docs complete, examples run.

---

## Dependency Graph

```
T01 (Infrastructure)
├── T02 (ED-01) ──► T03 (ED-02) ──► T04 (Benchmark) ──► T05 (Phase 1 Validation)
│   │                                                          │
│   └── T06 (ED-03) ─┐                                        │
│   └── T07 (ED-04) ─┤                                        │
│                     ├──► T11 (Phase 2 Validation)            │
├── T08 (ED-05) ──────┤                                        │
├── T09 (PA-03) ──────┤                                        │
├── T10 (MR-01) ──────┘                                        │
│                                                               │
├── T12 (PA-01) ──► T13 (PA-02)                                │
│              └──► T14 (MR-03)                                │
│                                                               │
├── T15 (PP-01)                                                 │
├── T16 (PP-02)                                                 │
├── T17 (MR-02)                                                 │
│                                                               │
└── T18 (Full Benchmark) ◄── all above ──► T19 (Docs)         │
```

---

## Testing Strategy Summary

| Phase | Test Type | Tool |
|-------|-----------|------|
| All | C++ unit tests | Google Test (`tests/test_neurodistance.cpp`) |
| All | Python integration tests | unittest (`tests/test_neurodistance.py`) |
| Phase 1+ | Benchmark comparison | `benchs/bench_neurodistance.py` |
| Phase 2 | Noise robustness | Synthetic noisy dataset in benchmark |
| Phase 2 | Missing value handling | NaN injection in benchmark |
| Phase 3 | Learning convergence | Feedback loop in Python test |
| Phase 4 | Cache performance | Timing comparison in Python test |
| Phase 5 | Full validation | All above + hypothesis table |

---

## Gate 3 Pass Criteria

- [x] Every task delivers working software (no "infrastructure-only" tasks except T01)
- [x] No task larger than 2 weeks (most are 2-5 days)
- [x] Dependencies are clear (dependency graph provided)
- [x] Testing approach defined per task
- [x] Phased delivery: each phase produces usable strategies
- [x] 19 tasks total across 5 phases
- [x] All 13 PRD strategies assigned to tasks
- [x] Hypothesis validation checkpoints at end of each phase
