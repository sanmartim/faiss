# Product Requirements Document: NeuroDistance V3

## Metadata

| Field | Value |
|-------|-------|
| Date | 2026-02-03 |
| Feature | neurodistance-v3 |
| Gate | 2 - Product Requirements Document |
| BRD Reference | docs/pre-dev/neurodistance-v3/brd.md |
| Research Reference | docs/pre-dev/neurodistance-v3/research.md |
| Status | Draft |

---

## Problem Statement

ML engineers and researchers with billion-scale vector datasets face a critical barrier: **memory and infrastructure costs make high-accuracy search impractical**. A 1B vector dataset at d=768 requires 3TB of RAM in float32, forcing teams to either:

1. Deploy expensive distributed clusters ($50K+/month)
2. Sacrifice accuracy with aggressive compression (recall drops to 70-80%)
3. Limit dataset size, losing valuable information

NeuroDistance V3 solves this by providing 22 production-scale strategies that achieve **95%+ recall with 10-50x compression** on standard hardware (64GB RAM + NVMe SSD).

---

## User Stories

### US-01: Memory-Constrained ML Engineer

**As a** ML engineer with limited infrastructure budget,
**I want to** index and search my billion-vector dataset on a single machine,
**So that** I can power my recommendation system without cloud infrastructure costs.

**Acceptance Criteria:**
- [ ] Can index 1B vectors (d=768) with <100GB RAM
- [ ] Search returns results in <50ms p99 latency
- [ ] Recall@10 is >=95% compared to brute force

---

### US-02: High-Throughput Search Application

**As a** backend engineer building a real-time search API,
**I want to** serve thousands of queries per second with consistent latency,
**So that** my users get instant results without timeouts.

**Acceptance Criteria:**
- [ ] Throughput >=1000 QPS on single machine
- [ ] p99 latency <50ms under load
- [ ] No query fails due to resource exhaustion

---

### US-03: Precision-Critical Application

**As a** data scientist building a medical image retrieval system,
**I want to** maintain very high accuracy while benefiting from compression,
**So that** diagnostic decisions are based on truly similar cases.

**Acceptance Criteria:**
- [ ] Recall@10 >=98% on 1M vectors
- [ ] Critical dimensions preserved in full precision
- [ ] Gradual accuracy vs memory trade-off options

---

### US-04: Cost-Conscious Team Lead

**As a** engineering team lead responsible for infrastructure costs,
**I want to** reduce our vector search memory footprint by 10x+,
**So that** we can stay within budget while expanding our search capabilities.

**Acceptance Criteria:**
- [ ] Memory reduction >=10x compared to float32 baseline
- [ ] No new infrastructure dependencies required
- [ ] Clear migration path from existing indices

---

### US-05: Adaptive Query Optimizer

**As a** platform engineer optimizing system performance,
**I want to** reduce unnecessary computation on "easy" queries,
**So that** resources are focused on difficult queries that need them.

**Acceptance Criteria:**
- [ ] Average computation reduced by >=30%
- [ ] Recall maintains >=98% after optimization
- [ ] No manual tuning required per query

---

## Feature Requirements

### FR-01: Quantization Family (QT)

**Description:** Strategies that compress vectors while maintaining search quality.

| ID | Strategy | Description | Success Metric |
|----|----------|-------------|----------------|
| QT-01 | ScalarQuantization | Per-dimension int8/int4 with calibration + rerank | >=98% recall, 4x compression |
| QT-02 | ProductQuantizationTiered | Cascading PQ (4-bit → 8-bit → float) | >=99% recall, 10x throughput |
| QT-03 | ResidualQuantization | Iterative residual encoding (4 stages) | >=95% recall, 32 bits/vector |
| QT-04 | AdaptiveQuantization | Hot regions float32, cold regions 4-bit | >=99% recall on hot queries |
| QT-05 | HybridQuantization | SQ-8bit top dims + PQ-4bit rest | +2-3% recall vs PQ alone |

**Acceptance Criteria:**
- [ ] All strategies inherit from IndexNeuro
- [ ] Configurable rerank candidates (default 100)
- [ ] Support pluggable NeuroMetric for reranking
- [ ] Python bindings via SWIG

---

### FR-02: Disk Family (DK)

**Description:** Strategies for datasets larger than RAM.

| ID | Strategy | Description | Success Metric |
|----|----------|-------------|----------------|
| DK-01 | DiskANNStyle | Vamana graph in RAM, vectors on SSD | p99 <20ms for 1B vectors |
| DK-02 | HierarchicalDisk | Multi-tier storage (RAM → SSD → HDD) | p99 <50ms for 10B vectors |
| DK-03 | CompressedDisk | LZ4/zstd compression on disk | 2-3x disk compression |
| DK-04 | MemoryMapped | mmap with OS-managed cache | 80% of RAM performance |

**Acceptance Criteria:**
- [ ] All strategies accept path parameter for disk storage
- [ ] LRU cache for hot vectors configurable
- [ ] Graceful degradation under memory pressure
- [ ] Python bindings via SWIG

---

### FR-03: Partition Family (PT)

**Description:** Strategies for intelligent data partitioning.

| ID | Strategy | Description | Success Metric |
|----|----------|-------------|----------------|
| PT-01 | OverlappingPartitions | Vectors in 2-3 partitions | +5-10% recall vs IVF |
| PT-02 | AdaptiveProbe | Dynamic nprobe based on query difficulty | -30% candidates |
| PT-03 | SemanticSharding | Cluster-based shards for distributed | 2/10 shards for 95% recall |
| PT-04 | DynamicPartitions | Online rebalancing based on query patterns | +20% throughput |

**Acceptance Criteria:**
- [ ] PT-01 configurable overlap count (default 2)
- [ ] PT-02 configurable gap thresholds
- [ ] All extend existing IVF patterns where possible
- [ ] Python bindings via SWIG

---

### FR-04: System Family (SY)

**Description:** Runtime optimizations for better performance.

| ID | Strategy | Description | Success Metric |
|----|----------|-------------|----------------|
| SY-01 | PrefetchOptimized | Hilbert curve data layout | +30-50% throughput |
| SY-02 | SIMDDistance | AVX2/AVX-512 optimized kernels | 2-3x speedup |
| SY-03 | EarlyTermination | Stop when confidence high | -40% candidates |
| SY-04 | BatchedQueries | Process queries together | 5-10x throughput |
| SY-05 | QueryCache | Cache similar query results | 20-30% hit rate |

**Acceptance Criteria:**
- [ ] SY-02 auto-detects best SIMD instruction set
- [ ] SY-03 configurable confidence threshold
- [ ] SY-05 configurable cache size and eviction policy
- [ ] All can be composed with other strategies
- [ ] Python bindings via SWIG

---

### FR-05: Binarization Family (BZ)

**Description:** Multi-precision binarization strategies.

| ID | Strategy | Description | Success Metric |
|----|----------|-------------|----------------|
| BZ-01 | ZonedBinarization | 10% float32 + 20% int8 + 70% binary | >=95% recall, 10x compression |
| BZ-02 | AdaptiveZones | Region-specific precision | +2-3% recall vs fixed zones |
| BZ-03 | LearnedBinarization | Optimize thresholds from data | +5-8% recall vs median |
| BZ-04 | MultiResolutionBinary | 1-bit → 2-bit → 4-bit → 8-bit cascade | >=97% recall, 50x speedup |

**Acceptance Criteria:**
- [ ] BZ-01 configurable zone percentages
- [ ] BZ-03 learns from similarity pairs
- [ ] All support Hamming distance for binary zones
- [ ] Python bindings via SWIG

---

## Non-Functional Requirements

### NFR-01: Performance

| Metric | Requirement |
|--------|-------------|
| Indexing speed | >=10K vectors/second for QT strategies |
| Search latency p50 (1M vectors) | <5ms |
| Search latency p99 (1B vectors) | <50ms |
| Memory overhead | <10% above vector storage |

### NFR-02: Compatibility

| Aspect | Requirement |
|--------|-------------|
| C++ standard | C++17 |
| Python version | >=3.8 |
| FAISS version | Compatible with main branch |
| API stability | Backward compatible with V2 |

### NFR-03: Reliability

| Aspect | Requirement |
|--------|-------------|
| Thread safety | Search operations are thread-safe |
| Error handling | Clear exceptions with messages |
| Graceful degradation | Memory pressure doesn't crash |

---

## Success Metrics

| Metric | Baseline (V2) | Target (V3) | Validation Method |
|--------|---------------|-------------|-------------------|
| Recall@10 (1M vectors) | 95% | 98% | Benchmark suite |
| Recall@10 (1B vectors) | N/A | 95% | Benchmark suite |
| Memory (1B × 768d) | 3 TB | 100 GB | Resource monitoring |
| p99 latency (1B vectors) | N/A | 50 ms | Benchmark suite |
| Throughput (QPS) | 100 | 1000+ | Load testing |
| Compression ratio | 4x (PQ) | 10x+ (BZ-01) | Memory measurement |

---

## Out of Scope

| Item | Rationale |
|------|-----------|
| GPU implementation | Separate initiative; V3 focuses CPU/SSD |
| Distributed cluster coordination | V3 is single-node library |
| Training embedding models | V3 is search/indexing only |
| Real-time streaming updates | V3 is batch indexing |
| Web UI or REST API | V3 is library only |

---

## Dependencies

| Dependency | Type | Notes |
|------------|------|-------|
| FAISS core (Index, Quantizer, etc.) | Internal | Required |
| ScalarQuantizer | Internal | Reuse for QT-01 |
| ProductQuantizer | Internal | Reuse for QT-02, QT-05 |
| ResidualQuantizer | Internal | Reuse for QT-03 |
| SIMD utilities (simdlib.h) | Internal | Reuse for SY-02 |
| LZ4/zstd | External (optional) | For DK-03 compression |

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Disk I/O bottleneck in DK strategies | High | Medium | Use prefetching, NVMe requirement |
| SWIG complexity for new types | Medium | Medium | Follow V2 patterns, test early |
| Low recall in binarization | High | Low | Zoned approach preserves critical dims |
| Cache thrashing in SY-05 | Medium | Low | LRU with configurable size |

---

## Gate 2 Validation

### Checklist

- [x] **Problem** is clearly defined with quantified pain points
- [x] **User stories** align with BRD jobs (FJ-01 → US-01, etc.)
- [x] **User value** is measurable (recall, latency, memory)
- [x] **Acceptance criteria** are testable
- [x] **Scope** is explicitly bounded
- [x] **Dependencies** identified
- [x] **Risks** documented with mitigations

**Gate 2 Result:** PASS

**Next Step:** Gate 3 - TRD Creation
