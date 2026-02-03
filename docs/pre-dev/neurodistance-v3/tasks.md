# Task Breakdown: NeuroDistance V3

## Metadata

| Field | Value |
|-------|-------|
| Date | 2026-02-03 |
| Feature | neurodistance-v3 |
| Gate | 4 - Task Breakdown |
| TRD Reference | docs/pre-dev/neurodistance-v3/trd.md |
| PRD Reference | docs/pre-dev/neurodistance-v3/prd.md |
| Status | Draft |

---

## Task Overview

| Phase | Tasks | Focus |
|-------|-------|-------|
| Phase 1 | T01-T06 | Foundation (QT-01, BZ-01, SY-02) |
| Phase 2 | T07-T13 | Scale (QT-02, DK-01, PT-01) |
| Phase 3 | T14-T20 | Optimization (SY-03, SY-04, PT-02) |
| Phase 4 | T21-T27 | Learning (BZ-03, QT-04, PT-04) |
| Phase 5 | T28-T33 | Production (SY-05, DK-02, PT-03) |

---

## Phase 1: Foundation (Week 1-2)

### T01: IndexNeuroScalarQuantization (QT-01)

**Description:** Implement per-dimension scalar quantization with calibration and rerank.

**Deliverables:**
- `faiss/IndexNeuroScalarQuantization.h`
- `faiss/IndexNeuroScalarQuantization.cpp`
- SWIG bindings added to `swigfaiss.swig`
- Unit tests in `tests/test_neurodistance_v3.cpp`

**Acceptance Criteria:**
- [ ] Wraps FAISS ScalarQuantizer
- [ ] Supports int8 and int4 modes
- [ ] Calibration: minmax, percentile (0.1-99.9), optim
- [ ] Configurable rerank_k (default 100)
- [ ] Pluggable NeuroMetric for rerank
- [ ] Recall@10 >= 98% on synthetic data

**Dependencies:** None

**Testing:**
- Unit test: encode/decode roundtrip
- Integration test: search quality vs IndexFlat
- Benchmark: throughput, memory usage

---

### T02: IndexNeuroZonedBinarization (BZ-01)

**Description:** Implement multi-precision zones (float32 + int8 + binary).

**Deliverables:**
- `faiss/IndexNeuroZonedBinarization.h`
- `faiss/IndexNeuroZonedBinarization.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Configurable zone percentages (default 10/20/70)
- [ ] Dimension importance analysis (variance + neighbor consistency)
- [ ] 3-stage search cascade (Hamming → L1 → L2)
- [ ] Recall@10 >= 95%
- [ ] Compression >= 10x vs float32

**Dependencies:** None

**Testing:**
- Unit test: zone assignment correctness
- Integration test: recall at various compressions
- Benchmark: memory usage, search speed

---

### T03: IndexNeuroSIMDDistance (SY-02)

**Description:** Implement SIMD-optimized batch distance computation decorator.

**Deliverables:**
- `faiss/IndexNeuroSIMDDistance.h`
- `faiss/IndexNeuroSIMDDistance.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Wraps any Index as decorator
- [ ] Uses fvec_L2sqr_ny / fvec_inner_products_ny
- [ ] Auto-batches queries for efficiency
- [ ] 2-3x speedup for d >= 256

**Dependencies:** None

**Testing:**
- Unit test: correctness vs non-SIMD
- Benchmark: speedup measurement

---

### T04: CMake and SWIG Infrastructure

**Description:** Update build system for V3 files.

**Deliverables:**
- Updated `faiss/CMakeLists.txt`
- Updated `faiss/python/swigfaiss.swig`

**Acceptance Criteria:**
- [ ] All V3 .cpp files in CMakeLists
- [ ] All V3 headers in SWIG %include
- [ ] `make` builds successfully
- [ ] Python bindings importable

**Dependencies:** T01, T02, T03

**Testing:**
- Build test: cmake && make
- Import test: `import faiss; faiss.IndexNeuroScalarQuantization`

---

### T05: Benchmark Suite V3

**Description:** Create comprehensive benchmark for V3 strategies.

**Deliverables:**
- `benchs/bench_neurodistance_v3.py`

**Acceptance Criteria:**
- [ ] Tests all 22 V3 strategies
- [ ] Measures recall@1, recall@10
- [ ] Measures latency (p50, p99)
- [ ] Measures memory usage
- [ ] Measures throughput (QPS)
- [ ] Outputs CSV for analysis

**Dependencies:** T04

**Testing:**
- Smoke test: runs on small data
- Full test: 1M vectors, d=128

---

### T06: Phase 1 Integration

**Description:** Integrate Phase 1 components, run benchmarks, validate hypotheses.

**Deliverables:**
- Benchmark results for QT-01, BZ-01, SY-02
- Updated validation report

**Acceptance Criteria:**
- [ ] QT-01: >= 98% recall, 4x compression
- [ ] BZ-01: >= 95% recall, 10x compression
- [ ] SY-02: 2-3x speedup

**Dependencies:** T01, T02, T03, T04, T05

---

## Phase 2: Scale (Week 3-4)

### T07: IndexNeuroProductQuantizationTiered (QT-02)

**Description:** Implement cascading PQ (4-bit → 8-bit → float).

**Deliverables:**
- `faiss/IndexNeuroProductQuantizationTiered.h`
- `faiss/IndexNeuroProductQuantizationTiered.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] 2-3 tier configuration
- [ ] Each tier uses ProductQuantizer
- [ ] Cascade filtering logic
- [ ] Recall@10 >= 99%
- [ ] 10x throughput improvement

**Dependencies:** T04

---

### T08: IndexNeuroDiskANN (DK-01)

**Description:** Implement DiskANN-style graph in RAM, vectors on disk.

**Deliverables:**
- `faiss/IndexNeuroDiskANN.h`
- `faiss/IndexNeuroDiskANN.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Vamana graph construction
- [ ] Beam search navigation
- [ ] LRU cache for vectors
- [ ] mmap for disk access
- [ ] p99 latency < 20ms for 1M vectors

**Dependencies:** T04

---

### T09: IndexNeuroOverlappingPartitions (PT-01)

**Description:** Implement partitions with overlap (vectors in 2-3 partitions).

**Deliverables:**
- `faiss/IndexNeuroOverlappingPartitions.h`
- `faiss/IndexNeuroOverlappingPartitions.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Configurable overlap (default 2)
- [ ] Extends IVF semantics
- [ ] +5-10% recall vs standard IVF

**Dependencies:** T04

---

### T10: IndexNeuroResidualQuantization (QT-03)

**Description:** Implement iterative residual encoding.

**Deliverables:**
- `faiss/IndexNeuroResidualQuantization.h`
- `faiss/IndexNeuroResidualQuantization.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Wraps FAISS ResidualQuantizer
- [ ] Configurable number of stages (default 4)
- [ ] 32 bits/vector target
- [ ] Recall >= 95%

**Dependencies:** T04

---

### T11: IndexNeuroMemoryMapped (DK-04)

**Description:** Implement simple mmap wrapper.

**Deliverables:**
- `faiss/IndexNeuroMemoryMapped.h`
- `faiss/IndexNeuroMemoryMapped.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] mmap for vector storage
- [ ] OS-managed caching
- [ ] 80% of RAM performance

**Dependencies:** T04

---

### T12: IndexNeuroHybridQuantization (QT-05)

**Description:** Implement SQ for top dims + PQ for rest.

**Deliverables:**
- `faiss/IndexNeuroHybridQuantization.h`
- `faiss/IndexNeuroHybridQuantization.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Configurable top-K dimensions for SQ
- [ ] Dimension importance analysis
- [ ] +2-3% recall vs PQ alone

**Dependencies:** T04

---

### T13: Phase 2 Integration

**Description:** Integrate Phase 2 components, validate scale hypotheses.

**Deliverables:**
- Benchmark results for QT-02, QT-03, QT-05, DK-01, DK-04, PT-01
- Updated validation report

**Acceptance Criteria:**
- [ ] QT-02: >= 99% recall, 10x throughput
- [ ] DK-01: p99 < 20ms on 1M vectors
- [ ] PT-01: +5% recall vs baseline

**Dependencies:** T07-T12

---

## Phase 3: Optimization (Week 5-6)

### T14: IndexNeuroEarlyTermination (SY-03)

**Description:** Implement confidence-based early stopping.

**Deliverables:**
- `faiss/IndexNeuroEarlyTermination.h`
- `faiss/IndexNeuroEarlyTermination.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Decorator pattern (wraps any Index)
- [ ] Configurable confidence threshold
- [ ] -40% candidates while maintaining 98% recall

**Dependencies:** T04

---

### T15: IndexNeuroBatchedQueries (SY-04)

**Description:** Implement batch query processing with matrix operations.

**Deliverables:**
- `faiss/IndexNeuroBatchedQueries.h`
- `faiss/IndexNeuroBatchedQueries.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Batch size optimization
- [ ] Matrix-based distance computation
- [ ] 5-10x throughput for batch queries

**Dependencies:** T04

---

### T16: IndexNeuroAdaptiveProbe (PT-02)

**Description:** Implement dynamic nprobe based on query difficulty.

**Deliverables:**
- `faiss/IndexNeuroAdaptiveProbe.h`
- `faiss/IndexNeuroAdaptiveProbe.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Gap-based nprobe selection
- [ ] -30% candidates on average
- [ ] Maintains recall

**Dependencies:** T04

---

### T17: IndexNeuroPrefetchOptimized (SY-01)

**Description:** Implement Hilbert curve data layout for cache efficiency.

**Deliverables:**
- `faiss/IndexNeuroPrefetchOptimized.h`
- `faiss/IndexNeuroPrefetchOptimized.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Hilbert curve ordering
- [ ] +30-50% throughput for large datasets

**Dependencies:** T04

---

### T18: IndexNeuroMultiResolutionBinary (BZ-04)

**Description:** Implement cascading bit resolution (1→2→4→8).

**Deliverables:**
- `faiss/IndexNeuroMultiResolutionBinary.h`
- `faiss/IndexNeuroMultiResolutionBinary.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] 4-level cascade
- [ ] Configurable keep ratios
- [ ] >= 97% recall, 50x speedup

**Dependencies:** T04

---

### T19: IndexNeuroAdaptiveZones (BZ-02)

**Description:** Implement region-specific zone configurations.

**Deliverables:**
- `faiss/IndexNeuroAdaptiveZones.h`
- `faiss/IndexNeuroAdaptiveZones.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Density-based region detection
- [ ] Per-region precision config
- [ ] +2-3% recall vs fixed zones

**Dependencies:** T02

---

### T20: Phase 3 Integration

**Description:** Integrate Phase 3 components, validate optimization hypotheses.

**Deliverables:**
- Benchmark results for SY-01, SY-03, SY-04, PT-02, BZ-02, BZ-04
- Updated validation report

**Acceptance Criteria:**
- [ ] SY-03: -40% candidates
- [ ] SY-04: 5-10x throughput
- [ ] BZ-04: 50x speedup

**Dependencies:** T14-T19

---

## Phase 4: Learning (Week 7-8)

### T21: IndexNeuroLearnedBinarization (BZ-03)

**Description:** Implement learned thresholds for binarization.

**Deliverables:**
- `faiss/IndexNeuroLearnedBinarization.h`
- `faiss/IndexNeuroLearnedBinarization.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Threshold optimization from similarity pairs
- [ ] +5-8% recall vs median thresholds

**Dependencies:** T02

---

### T22: IndexNeuroAdaptiveQuantization (QT-04)

**Description:** Implement hot/cold region variable precision.

**Deliverables:**
- `faiss/IndexNeuroAdaptiveQuantization.h`
- `faiss/IndexNeuroAdaptiveQuantization.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Density-based region detection
- [ ] Hot regions: float32
- [ ] Cold regions: 4-bit
- [ ] >= 99% recall on hot queries

**Dependencies:** T01

---

### T23: IndexNeuroDynamicPartitions (PT-04)

**Description:** Implement online partition rebalancing.

**Deliverables:**
- `faiss/IndexNeuroDynamicPartitions.h`
- `faiss/IndexNeuroDynamicPartitions.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Query-driven rebalancing
- [ ] +20% throughput after warm-up

**Dependencies:** T09

---

### T24-T26: Reserved for Phase 4 refinements

---

### T27: Phase 4 Integration

**Description:** Integrate Phase 4 components, validate learning hypotheses.

**Deliverables:**
- Benchmark results for BZ-03, QT-04, PT-04
- Updated validation report

**Acceptance Criteria:**
- [ ] BZ-03: +5-8% recall
- [ ] QT-04: 99% recall on hot
- [ ] PT-04: +20% throughput

**Dependencies:** T21-T26

---

## Phase 5: Production (Week 9-10)

### T28: IndexNeuroQueryCache (SY-05)

**Description:** Implement LRU cache for similar queries.

**Deliverables:**
- `faiss/IndexNeuroQueryCache.h`
- `faiss/IndexNeuroQueryCache.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Reuses IndexNeuroCache LRU pattern
- [ ] Similarity-based cache key
- [ ] 20-30% hit rate on real workloads

**Dependencies:** T04

---

### T29: IndexNeuroHierarchicalDisk (DK-02)

**Description:** Implement multi-tier storage hierarchy.

**Deliverables:**
- `faiss/IndexNeuroHierarchicalDisk.h`
- `faiss/IndexNeuroHierarchicalDisk.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] RAM → SSD → HDD tiers
- [ ] Promotion/demotion policies
- [ ] p99 < 50ms for 10B vectors (theoretical)

**Dependencies:** T08

---

### T30: IndexNeuroCompressedDisk (DK-03)

**Description:** Implement on-disk compression.

**Deliverables:**
- `faiss/IndexNeuroCompressedDisk.h`
- `faiss/IndexNeuroCompressedDisk.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] LZ4/zstd codec support
- [ ] Block-based compression
- [ ] 2-3x disk compression

**Dependencies:** T08

---

### T31: IndexNeuroSemanticSharding (PT-03)

**Description:** Implement cluster-aware sharding.

**Deliverables:**
- `faiss/IndexNeuroSemanticSharding.h`
- `faiss/IndexNeuroSemanticSharding.cpp`
- SWIG bindings
- Unit tests

**Acceptance Criteria:**
- [ ] Multi-shard management
- [ ] 2/10 shards for 95% recall

**Dependencies:** T09

---

### T32: Final Benchmark and Validation

**Description:** Run complete benchmark suite, validate all hypotheses.

**Deliverables:**
- Complete validation report
- Performance summary
- Recommendations

**Acceptance Criteria:**
- [ ] All 22 strategies tested
- [ ] Success metrics validated
- [ ] Documentation updated

**Dependencies:** All previous tasks

---

### T33: Documentation and Release

**Description:** Complete documentation and prepare release.

**Deliverables:**
- API documentation
- Usage examples
- Migration guide from V2

**Acceptance Criteria:**
- [ ] All strategies documented
- [ ] Python examples provided
- [ ] Performance guidelines

**Dependencies:** T32

---

## Dependency Graph

```
Phase 1 (Foundation):
T01 (QT-01) ────┐
T02 (BZ-01) ────┼──► T04 (Build) ──► T05 (Benchmark) ──► T06 (Integration)
T03 (SY-02) ────┘

Phase 2 (Scale):
T04 ──► T07 (QT-02) ──┐
T04 ──► T08 (DK-01) ──┼──► T13 (Integration)
T04 ──► T09 (PT-01) ──┤
T04 ──► T10 (QT-03) ──┤
T04 ──► T11 (DK-04) ──┤
T04 ──► T12 (QT-05) ──┘

Phase 3 (Optimization):
T04 ──► T14 (SY-03) ──┐
T04 ──► T15 (SY-04) ──┤
T04 ──► T16 (PT-02) ──┼──► T20 (Integration)
T04 ──► T17 (SY-01) ──┤
T04 ──► T18 (BZ-04) ──┤
T02 ──► T19 (BZ-02) ──┘

Phase 4 (Learning):
T02 ──► T21 (BZ-03) ──┐
T01 ──► T22 (QT-04) ──┼──► T27 (Integration)
T09 ──► T23 (PT-04) ──┘

Phase 5 (Production):
T04 ──► T28 (SY-05) ──┐
T08 ──► T29 (DK-02) ──┤
T08 ──► T30 (DK-03) ──┼──► T32 (Validation) ──► T33 (Release)
T09 ──► T31 (PT-03) ──┘
```

---

## Gate 4 Validation

### Checklist

- [x] **Every task** delivers working software
- [x] **No task** larger than 2 weeks
- [x] **Dependencies** are clear
- [x] **Testing approach** defined per task
- [x] **Phase integration** tasks validate hypotheses

**Gate 4 Result:** PASS

---

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | 33 |
| Total Duration | 10 weeks |
| Strategies Implemented | 22 |
| New Files | ~50 (.h + .cpp + tests) |
| Estimated LOC | ~8,500 |
