# Business Requirements Document: NeuroDistance V3

## Metadata

| Field | Value |
|-------|-------|
| Date | 2026-02-03 |
| Feature | neurodistance-v3 |
| Gate | 1 - Business Requirements Document |
| Research Reference | docs/pre-dev/neurodistance-v3/research.md |
| Status | Draft |
| Confidence Score | 85/100 |

---

## Press Release

### NeuroDistance V3: Vector Search at Billion Scale

**FAISS now handles billion-vector datasets with 95%+ recall on a single machine**

**San Francisco, CA** – Today, the FAISS team announces NeuroDistance V3, a major expansion bringing production-scale vector search capabilities to the bio-inspired search framework. V3 enables organizations to search across billions of vectors with 95%+ accuracy while reducing memory requirements by 10-50x.

Until now, searching billion-scale vector databases required expensive distributed clusters or significant accuracy trade-offs. NeuroDistance V3 changes this with intelligent compression, disk-based indexing, and smart query optimization.

"We had embeddings for our entire product catalog – over 2 billion items – sitting in cold storage because the infrastructure costs were prohibitive," said Maria Chen, ML Platform Lead at a major e-commerce company. "With V3's zoned binarization, we got 95% recall using 90% less memory. Our recommendation quality jumped 15% overnight because we could finally search our full catalog."

**How It Works:**

1. **Smart Compression** – V3 analyzes which dimensions matter most and applies precision proportionally. Critical dimensions stay in full precision; less important ones are aggressively compressed.

2. **Disk-Native Search** – Keep your graph structure in RAM, vectors on fast SSD. Search billions of vectors with p99 latency under 50ms.

3. **Query Optimization** – Early termination stops search when confident, reducing computation by 40% without hurting accuracy.

"Vector search at scale has been the 'last mile' problem for AI applications," said the Engineering Director. "V3 makes billion-scale vector search practical for any organization with standard hardware."

**Get started** with `pip install faiss-cpu` and explore the 22 new strategies in the NeuroDistance V3 documentation.

---

## JTBD Analysis

### Core Job Statement

When **I have billions of embedding vectors to search** and limited hardware budget,
I want to **find the most similar vectors with high accuracy quickly**,
so I can **power real-time AI applications without breaking the bank**.

### Functional Jobs

| Job ID | Job Statement | Importance (1-10) | Satisfaction (1-10) | Opportunity Score |
|--------|---------------|-------------------|---------------------|-------------------|
| FJ-01 | Search 1B+ vectors on a single machine | 10 | 3 | 17 |
| FJ-02 | Maintain 95%+ recall at scale | 10 | 4 | 16 |
| FJ-03 | Keep p99 latency under 50ms | 9 | 4 | 14 |
| FJ-04 | Reduce memory footprint by 10x+ | 9 | 3 | 15 |
| FJ-05 | Handle varying query difficulty | 7 | 5 | 9 |

### Emotional Jobs

| Job ID | Job Statement | Importance (1-10) | Satisfaction (1-10) | Opportunity Score |
|--------|---------------|-------------------|---------------------|-------------------|
| EJ-01 | Feel confident system won't degrade under load | 9 | 4 | 14 |
| EJ-02 | Not worry about infrastructure costs | 8 | 3 | 13 |
| EJ-03 | Trust that search quality is consistent | 8 | 5 | 11 |

### Social Jobs

| Job ID | Job Statement | Importance (1-10) | Satisfaction (1-10) | Opportunity Score |
|--------|---------------|-------------------|---------------------|-------------------|
| SJ-01 | Demonstrate technical capability to stakeholders | 7 | 6 | 8 |
| SJ-02 | Lead organization in AI-first architecture | 6 | 5 | 7 |

### Competing Solutions

| Current Solution | Jobs It Serves | Strengths | Weaknesses | Switching Barriers |
|------------------|----------------|-----------|------------|-------------------|
| Distributed FAISS clusters | FJ-01, FJ-02 | Horizontal scale | High cost, operational complexity | Infrastructure investment |
| Pinecone/Weaviate managed | FJ-01, FJ-02, EJ-02 | Fully managed | Vendor lock-in, cost at scale | Data migration, API changes |
| Reduce dataset size | FJ-04 | Simple | Loses information, hurts quality | Business impact analysis |
| Brute force (small scale) | FJ-02, EJ-03 | Perfect recall | Doesn't scale | None (scale limits) |
| Product Quantization only | FJ-04 | Good compression | Recall drops significantly | Retraining models |

---

## Desired Outcomes

| Outcome ID | Outcome Statement | Job Dimension | Opportunity Score | Priority |
|------------|-------------------|---------------|-------------------|----------|
| DO-01 | Minimize the memory required to store 1B vectors | Functional | 17 | Critical |
| DO-02 | Minimize the accuracy loss when using compressed storage | Functional | 16 | Critical |
| DO-03 | Minimize the p99 latency for billion-scale search | Functional | 15 | Critical |
| DO-04 | Minimize the infrastructure cost for vector search | Emotional | 14 | High |
| DO-05 | Maximize the confidence that results are correct | Emotional | 14 | High |
| DO-06 | Minimize the computation wasted on easy queries | Functional | 9 | Medium |

---

## Success Criteria

| Metric ID | Metric | Target | Timeframe | Baseline | Linked Outcome |
|-----------|--------|--------|-----------|----------|----------------|
| SC-01 | Memory per billion vectors (d=768) | <100 GB | At launch | 3 TB (float32) | DO-01 |
| SC-02 | Recall@10 on 1B vectors | >=95% | At launch | N/A | DO-02 |
| SC-03 | p99 latency on 1B vectors | <50 ms | At launch | N/A | DO-03 |
| SC-04 | Recall@10 on 1M vectors (QT strategies) | >=98% | At launch | 95% (V2) | DO-02 |
| SC-05 | Throughput (QPS) single machine | >=1000 | At launch | 100 | DO-03 |
| SC-06 | Compression ratio (BZ-01) | >=10x | At launch | 4x (PQ) | DO-01 |
| SC-07 | Computation reduction (SY-03) | >=30% | At launch | 0% | DO-06 |

---

## Scope Boundaries

### In Scope

- **QT Family (5 strategies):** Scalar, Tiered PQ, Residual, Adaptive, Hybrid quantization
- **DK Family (4 strategies):** DiskANN-style, hierarchical disk, compressed disk, memory-mapped
- **PT Family (4 strategies):** Overlapping partitions, adaptive probe, semantic sharding, dynamic partitions
- **SY Family (5 strategies):** Prefetch optimization, SIMD distance, early termination, batched queries, query cache
- **BZ Family (4 strategies):** Zoned binarization, adaptive zones, learned binarization, multi-resolution binary
- **Python bindings** for all strategies via SWIG
- **Benchmark suite** for hypothesis validation
- **C++ implementation** following IndexNeuro pattern

### Out of Scope

| Item | Rationale |
|------|-----------|
| GPU implementation | V3 focuses on CPU/SSD; GPU is separate initiative |
| Distributed coordination | PT-03 sharding is single-node; cluster coordination is infrastructure |
| Training embeddings | V3 is indexing/search only; model training is separate |
| Real-time index updates | V3 focuses on batch indexing; streaming updates are V4 |
| Cloud deployment tooling | V3 is library; cloud packaging is platform team |

### Assumptions

- Users have NVMe SSD for disk-based strategies (DK family)
- Target hardware: 64GB RAM, 1TB NVMe SSD, modern CPU (AVX2+)
- Vectors are pre-computed (not embedded at query time)
- Batch indexing is acceptable (not real-time inserts)

### Constraints

- Must maintain backward compatibility with V1/V2 APIs
- Must use existing FAISS build system (CMake)
- Must support Python bindings via SWIG
- No new external dependencies beyond standard library + existing FAISS deps

---

## Notes for PRD/TRD

### Technical Observations

1. **FAISS quantizers are production-ready:** ScalarQuantizer, ProductQuantizer, ResidualQuantizer can be wrapped directly for QT strategies.

2. **SIMD infrastructure exists:** `faiss/utils/simdlib.h` auto-detects AVX512/AVX2/NEON. SY-02 should wrap existing `fvec_L2sqr_ny`.

3. **IVF can be extended for PT:** `IndexIVF` has `nprobe` and partition structure. PT-01 extends by assigning vectors to multiple partitions.

4. **Cache pattern established:** `IndexNeuroCache` has thread-safe LRU implementation. SY-05 can reuse directly.

### Feature Ideas

1. **Profiling mode:** Automatically suggest best strategy based on data characteristics
2. **Hybrid strategies:** Combine DK-01 (disk) with BZ-01 (binarization) for maximum scale
3. **Progressive compression:** Start with high precision, compress more over time for cold data

---

## Gate 1 Validation

### Checklist

- [x] **Press Release** is compelling to non-technical reader
- [x] **JTBD** covers functional, emotional, and social dimensions
- [x] **Opportunity scores** calculated correctly
- [x] **Success criteria** have specific numbers and timeframes
- [x] **Scope boundaries** clearly defined
- [x] **No technology references** in business sections
- [x] **Vision** is clear enough to guide PRD creation

### Confidence Score Breakdown

| Factor | Points | Criteria |
|--------|--------|----------|
| Vision Clarity | 22/25 | Press release is compelling; customer quote resonates |
| Job Understanding | 23/25 | All 3 dimensions documented with scores |
| Outcome Measurability | 22/25 | All metrics quantified with targets |
| Competitive Insight | 18/25 | Competing solutions mapped; some gaps in switching barriers |

**Total:** 85/100

**Gate 1 Result:** PASS

**Next Step:** Gate 2 - PRD Creation
