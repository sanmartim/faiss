# PRD: NeuroDistance - Bio-Inspired Vector Search Strategies for FAISS

**Feature:** neuro-distance-strategies
**Version:** 1.0
**Date:** 2026-02-02
**Based on:** prd_neurodistance_completo.md + Gate 0 Research

---

## 1. Problem Statement

FAISS provides highly optimized brute-force and approximate nearest neighbor search using standard distance metrics (L2, Inner Product, etc.). However, all dimensions are treated equally during search, which ignores:

- **Temporal structure** - In time series embeddings, recent values carry more signal than old ones
- **Feature importance** - Some dimensions are more discriminative than others for a given dataset
- **Query context** - Different queries may benefit from different dimension weighting
- **Data quality** - Missing or noisy dimensions should contribute less to distance
- **Adaptability** - The system does not learn from usage patterns

These limitations mean FAISS achieves 100% recall with brute force but offers no middle ground between "use all dimensions equally" and "build an approximate index." NeuroDistance fills this gap with bio-inspired search strategies that exploit data structure for better recall/speed tradeoffs.

---

## 2. Users and Use Cases

### Primary Users
- **ML Engineers** building retrieval systems on structured embeddings (time series, sensor data, multi-modal)
- **Data Scientists** exploring which dimensions matter for their specific dataset
- **Research teams** testing hypotheses about distance computation strategies

### Use Cases

| Use Case | Strategy Family | Example |
|----------|----------------|---------|
| Time series similarity search | Elimination (ED-01..04) | Financial data where recent columns matter more |
| Noisy embedding search | Dropout Ensemble (ED-05) | OCR embeddings with variable quality |
| Adaptive retrieval system | Learned Weights (PA-01..02) | Recommendation engine that improves with feedback |
| Incomplete data search | Missing Value Adjusted (PA-03) | Sensor data with gaps |
| High-dimensional search | Coarse-to-Fine (PP-02) | Image embeddings (d=1024+) |
| Diverse result retrieval | Lateral Inhibition (MR-01) | Search results that avoid redundancy |
| Low-latency repeated queries | Context Cache (MR-02) | API serving similar queries |

---

## 3. Strategy Catalogue

### Family 1: Progressive Elimination (ED)

Strategies that reduce candidates column-by-column instead of computing full distance.

| ID | Name | Core Idea | Expected Recall@10 | Calc Reduction |
|----|------|-----------|--------------------:|---------------:|
| ED-01 | FixedElimination | Fixed column order, fixed cutoff % | 85-95% | 50-70% |
| ED-02 | AdaptiveDispersion | Cutoff adapts to column dispersion | 90-97% | 45-65% |
| ED-03 | VarianceOrder | Columns ordered by sampled variance | 88-94% | 40-60% |
| ED-04 | UncertaintyDeferred | Defers elimination when uncertain | 93-98% | 30-50% |
| ED-05 | DropoutEnsemble | Multiple masked searches + vote | 93-98% | N/A (parallel) |

### Family 2: Adaptive Weights (PA)

Strategies that learn or adjust per-dimension importance.

| ID | Name | Core Idea | Key Metric |
|----|------|-----------|------------|
| PA-01 | LearnedWeights | Hebbian weight update from feedback | +5% recall after 1000 queries |
| PA-02 | ContextualWeights | Per-query-cluster weight sets | +3-8% recall vs global weights |
| PA-03 | MissingValueAdjusted | Reduce weight for missing dimensions | <15% degradation at 30% missing |

### Family 3: Parallel Processing (PP)

Strategies that process dimension groups simultaneously.

| ID | Name | Core Idea | Key Metric |
|----|------|-----------|------------|
| PP-01 | ParallelVoting | Group-wise search + vote integration | 92-97% recall, 2-3x speedup |
| PP-02 | CoarseToFine | Multi-resolution progressive refinement | 88-94% recall, 60-80% calc reduction |

### Family 4: Refinement Mechanisms (MR)

Post-processing and meta-strategies.

| ID | Name | Core Idea | Key Metric |
|----|------|-----------|------------|
| MR-01 | LateralInhibition | Remove similar candidates, keep diversity | +20-40% diversity |
| MR-02 | ContextCache | Cache recent query configs for reuse | 30-50% speedup on hits |
| MR-03 | ContrastiveLearning | Learn weights from positive+negative examples | 2-3x faster convergence than PA-01 |

---

## 4. User Stories

### US-01: Basic Strategy Usage
**As a** ML engineer,
**I want to** create an index with a NeuroDistance strategy and search it,
**So that** I can compare recall/speed tradeoffs against standard FAISS brute force.

**Acceptance Criteria:**
- Can create any of the 13 strategies via C++ API
- Can create strategies via Python bindings
- `search()` returns standard FAISS result format (distances + indices)
- Each search result includes metadata (calculations performed, columns used, time)

### US-02: Progressive Elimination Search
**As a** data scientist with time series data,
**I want to** search using progressive elimination (ED-01..04) with configurable column order and cutoff,
**So that** I get faster search by exploiting the known structure of my data.

**Acceptance Criteria:**
- Column order is configurable (reversed, custom, variance-based)
- Cutoff percentage is configurable per strategy
- Recall@10 >= 85% with >= 50% fewer distance calculations than brute force
- Works with IndexFlat as the underlying storage

### US-03: Robust Search with Dropout Ensemble
**As a** ML engineer with noisy embeddings,
**I want to** run multiple masked searches and combine results by voting,
**So that** my search is robust to noise in individual dimensions.

**Acceptance Criteria:**
- Configurable number of views (default 5)
- Configurable dropout rate (default 0.3)
- Multiple dropout modes: random, complementary, structured, adversarial
- Multiple integration methods: voting, borda count, mean, full recalculation
- Recall >= 95% on clean data, degradation < 10% with 20% noisy dimensions

### US-04: Adaptive Weight Learning
**As a** developer building a retrieval system,
**I want to** provide feedback after searches so the system learns which dimensions matter,
**So that** search quality improves over time.

**Acceptance Criteria:**
- `feedback()` or `train()` method accepts query + ground truth
- Weights converge within 500-2000 queries
- Recall improves by >= 5% after training
- Learned weights are serializable (save/load)

### US-05: Search with Missing Values
**As a** data engineer with incomplete sensor data,
**I want to** search vectors with NaN/missing values without them corrupting results,
**So that** I get reasonable results even with incomplete data.

**Acceptance Criteria:**
- NaN values are detected and their dimensions down-weighted
- Configurable missing value threshold (ignore column if > X% missing)
- Degradation < 15% with 30% missing values (vs > 30% without adjustment)

### US-06: Diverse Results via Lateral Inhibition
**As a** search product engineer,
**I want to** post-process search results to remove near-duplicates,
**So that** users see diverse results instead of variations of the same item.

**Acceptance Criteria:**
- Configurable similarity threshold for inhibition
- Configurable max candidates per cluster
- Works as a decorator on any Index
- Diversity improves by >= 20% with <= 5% recall loss

### US-07: Benchmark and Compare Strategies
**As a** researcher,
**I want to** benchmark all strategies on the same dataset and compare them,
**So that** I can validate which hypotheses hold for my data.

**Acceptance Criteria:**
- Benchmark script that runs all strategies on a dataset
- Outputs table: strategy, recall@1, recall@10, time_ms, calculations
- Supports SyntheticDataset and SIFT1M
- Pareto plot (recall vs speed) generation

### US-08: Strategy Combinations
**As a** ML engineer,
**I want to** combine strategies (e.g., learned weights + uncertainty deferred + lateral inhibition),
**So that** I can build composite search pipelines.

**Acceptance Criteria:**
- Decorator/wrapper pattern allows composing strategies
- At least 5 recommended combinations documented and tested
- Combined strategies produce valid FAISS-compatible results

---

## 5. Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Strategies implemented | 13 | Code count |
| Average recall vs brute force | >= 92% | Benchmark on SIFT1M |
| Average speedup vs brute force | >= 2x | Benchmark on SIFT1M |
| Recall with noise robustness (ED-05) | >= 95% clean, < 10% degradation with 20% noise | Synthetic noisy dataset |
| Learning convergence (PA-01) | Recall improvement >= 5% in 1000 queries | Training loop benchmark |
| Missing value robustness (PA-03) | < 15% degradation at 30% missing | Synthetic missing data |
| Result diversity (MR-01) | >= 20% improvement | Diversity metric on clustered data |
| All C++ tests pass | 100% | CI |
| All Python tests pass | 100% | CI |
| Benchmark reproducible | Yes | Script with fixed seeds |

---

## 6. Hypotheses and Validation Criteria

Each strategy is a hypothesis. If a hypothesis fails validation, the strategy is documented as "not validated" rather than removed.

| Strategy | Hypothesis | Validation Criterion |
|----------|-----------|---------------------|
| ED-01 | Fixed elimination works for structured data | Recall >= 85% with 50% fewer calculations |
| ED-02 | Dispersion is a proxy for discriminative power | Recall +5% vs ED-01 with < 5% overhead |
| ED-03 | Variance-ordered columns beat random order | +15% recall vs random order |
| ED-04 | Deferring uncertain decisions improves recall | +3% recall vs ED-02, robust to noise |
| ED-05 | Multiple dropout views + voting beats single full search | Recall >= 95%, degradation < 10% with noise |
| PA-01 | Learned weights outperform uniform weights | +5% recall after 1000 queries |
| PA-02 | Per-context weights outperform global weights | +3% on difficult queries |
| PA-03 | Missing-aware weights reduce degradation | < 50% of baseline degradation |
| PP-01 | Parallel voting is more robust than sequential elimination | +2% recall, lower variance |
| PP-02 | Coarse-to-fine scales better with dimensionality | 60%+ calc reduction at d >= 256 |
| MR-01 | Lateral inhibition increases useful diversity | +20% diversity with < 5% recall loss |
| MR-02 | Context caching speeds up repeated query patterns | 30%+ speedup on cache hits |
| MR-03 | Contrastive learning converges faster than PA-01 | 2x fewer queries to same recall |

---

## 7. Phased Delivery

### Phase 1: Core MVP
- ED-01 (FixedElimination)
- ED-02 (AdaptiveDispersion)
- Common interface (C++ base class + Python bindings)
- Benchmark script (basic)
- **Deliverable:** Working library with 2 strategies, comparable with FAISS brute force

### Phase 2: Elimination Family Complete
- ED-03 (VarianceOrder)
- ED-04 (UncertaintyDeferred)
- ED-05 (DropoutEnsemble)
- PA-03 (MissingValueAdjusted)
- MR-01 (LateralInhibition)
- **Deliverable:** 7 strategies covering common cases

### Phase 3: Learning Strategies
- PA-01 (LearnedWeights)
- PA-02 (ContextualWeights)
- MR-03 (ContrastiveLearning)
- **Deliverable:** System that improves with usage

### Phase 4: Performance & Caching
- PP-01 (ParallelVoting)
- PP-02 (CoarseToFine)
- MR-02 (ContextCache)
- **Deliverable:** Production-optimized strategies

### Phase 5: Automation
- AutoNeuroDistance (automatic strategy selection)
- Strategy combinator
- **Deliverable:** Self-optimizing system

---

## 8. Out of Scope

- **GPU implementations** - C++ CPU only for MVP; GPU can be added later
- **Approximate index integration** (HNSW, IVF) - Strategies work with flat/brute-force first
- **Distributed search** - Single-node only
- **Custom SIMD kernels** - Use scalar implementations first; SIMD optimization is a future enhancement
- **Dashboard/monitoring UI** - Benchmark outputs are text/CSV; no web UI
- **Auto-tuning of strategy hyperparameters** - Phase 5 scope
- **Integration with other vector databases** (Milvus, Pinecone, etc.)

---

## 9. Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No strategy significantly beats FAISS brute force on standard benchmarks | Medium | High | Position for specific use cases (structured data, noisy data), not general replacement |
| Configuration complexity deters users | High | Medium | Good defaults, recommended combinations, AutoNeuroDistance in Phase 5 |
| Learning overhead doesn't justify improvement | Medium | Medium | Keep static strategies as primary option; learning is additive |
| Doesn't scale beyond millions of vectors | High | High | Combine with approximate indexes in future; document limitations |
| C++ complexity slows development | Medium | Medium | Start with simplest strategies (ED-01, ED-02); iterate |

---

## 10. Technical Constraints (from Gate 0 Research)

- New simple distance metrics integrate via `MetricType` enum + `VectorDistance<>` template specialization
- Search-level strategies require custom `Index` subclasses overriding `search()`
- Python bindings via SWIG (existing pattern)
- Benchmark uses existing `contrib/datasets.py` and `contrib/evaluation.py`
- Tests follow `tests/test_extra_distances.py` pattern (compare against reference)

---

## Gate 1 Pass Criteria

- [x] Problem is clearly defined (standard distances ignore data structure)
- [x] User value is measurable (recall, speed, robustness metrics per strategy)
- [x] Acceptance criteria are testable (quantitative targets for each strategy)
- [x] Scope is explicitly bounded (out of scope section)
- [x] 13 strategies catalogued with hypotheses and validation criteria
- [x] Phased delivery plan (5 phases, MVP first)
- [x] 8 user stories with acceptance criteria
