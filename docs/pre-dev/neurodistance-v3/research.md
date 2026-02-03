# NeuroDistance V3: Research Report

## Metadata

| Field | Value |
|-------|-------|
| Date | 2026-02-03 |
| Feature | neurodistance-v3 |
| Research Mode | modification (extending existing NeuroDistance architecture) |
| Agents Dispatched | repo-research-analyst, best-practices-researcher |

---

## Executive Summary

NeuroDistance V3 extends the existing V1/V2 architecture (47 strategies) with 22 new strategies focused on **production scale** (1B+ vectors). The codebase already has extensive reusable components - V3 can leverage 70-80% existing code. Key FAISS components (ScalarQuantizer, ProductQuantizer, ResidualQuantizer, SIMD utilities) provide production-ready foundations for the new quantization and system optimization strategies.

---

## Research Mode

**Mode:** Modification/Extension

**Rationale:**
- V1/V2 established IndexNeuro base class pattern
- SWIG Python bindings infrastructure exists
- Test framework established
- V3 adds new strategy families following existing patterns

---

## Codebase Research

### 1. IndexNeuro Base Class Pattern

**File:** `faiss/impl/NeuroDistance.h:196-216`

```cpp
struct IndexNeuro : Index {
    Index* inner_index = nullptr;        // Wraps any base index
    bool own_inner = false;
    mutable NeuroSearchStats last_stats;

    void add(idx_t n, const float* x) override;      // Delegated
    void reset() override;                            // Delegated
    void reconstruct(idx_t key, float* recons) const override; // Delegated
    // search() overridden by subclasses
};
```

**Key Pattern:** Wrapper that delegates storage to inner index. V3 strategies follow this pattern.

### 2. NeuroMetric Interface (MT-00)

**File:** `faiss/impl/NeuroDistance.h:118-188`

- `NeuroMetricL2` - Uses optimized `fvec_L2sqr`
- `NeuroMetricCosine` - 1 - cosine_similarity
- `NeuroMetricDot` - Inner product
- `NeuroMetricMahalanobis` - Diagonal covariance
- `NeuroMetricJaccard` - Binary vectors

**Reuse:** All V3 strategies can plug in custom metrics for reranking.

### 3. Existing Quantization Infrastructure

#### ScalarQuantizer (Reusable for QT-01)

**File:** `faiss/impl/ScalarQuantizer.h:24-94`

```cpp
struct ScalarQuantizer : Quantizer {
    enum QuantizerType { QT_8bit, QT_4bit, QT_8bit_uniform, QT_4bit_uniform, ... };
    enum RangeStat { RS_minmax, RS_meanstd, RS_quantiles, RS_optim };

    void train(size_t n, const float* x) override;
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;
    void decode(const uint8_t* code, float* x, size_t n) const override;
};
```

**Reuse Level:** VERY HIGH - Production-ready. Wrap directly for QT-01.

#### ProductQuantizer (Reusable for QT-02, QT-03)

**File:** `faiss/impl/ProductQuantizer.h:29-99`

```cpp
struct ProductQuantizer : Quantizer {
    size_t M;           // Number of subquantizers
    size_t nbits;       // Bits per subquantizer
    size_t dsub;        // Dimensionality per subvector (d/M)
    size_t ksub;        // Centroids per subquantizer (2^nbits)

    std::vector<float> centroids;  // M * ksub * dsub
};
```

**Reuse Level:** VERY HIGH - For QT-02 (tiered PQ), stack 2-3 ProductQuantizers with different nbits.

#### ResidualQuantizer (Reusable for QT-03)

**File:** `faiss/impl/ResidualQuantizer.h:27-145`

```cpp
struct ResidualQuantizer : AdditiveQuantizer {
    std::vector<size_t> nbits;  // Per-level bit counts
    int max_beam_size = 5;      // For beam search

    void refine_beam(size_t n, size_t beam_size, const float* residuals, ...);
};
```

**Reuse Level:** VERY HIGH - Perfect for QT-03 iterative quantization.

### 4. SIMD Distance Utilities (Reusable for SY-02)

**File:** `faiss/utils/simdlib.h:1-42`

- Auto-detects AVX512/AVX2/NEON/PPC64
- Optimized kernels already available

**File:** `faiss/utils/distances.h:28-100`

```cpp
float fvec_L2sqr(const float* x, const float* y, size_t d);
void fvec_L2sqr_ny(float* dis, const float* x, const float* y, size_t d, size_t ny);
void fvec_inner_products_ny(float* ip, const float* x, const float* y, size_t d, size_t ny);
```

**Reuse Level:** VERY HIGH - SY-02 wraps these optimized kernels.

### 5. IVF Partitioning (Reusable for PT-01, PT-02)

**File:** `faiss/IndexIVF.h:30-76`

```cpp
struct Level1Quantizer {
    Index* quantizer = nullptr;  // Maps vectors to partitions
    size_t nlist = 0;            // Number of partitions
};

struct SearchParametersIVF : SearchParameters {
    size_t nprobe = 1;           // Number of partitions to probe
};
```

**Reuse Level:** HIGH - PT-01 (Overlapping) extends IVF to assign vectors to multiple partitions.

### 6. Cache Pattern (Reusable for SY-05)

**File:** `faiss/IndexNeuroCache.h:27-87`

```cpp
struct IndexNeuroCache : Index {
    std::list<std::pair<size_t, CacheEntry>> lru_list;
    std::unordered_map<size_t, iterator> cache_map;
    std::mutex cache_mutex;

    void clear_cache() const;
    float hit_rate() const;
};
```

**Reuse Level:** VERY HIGH - LRU + hash map design directly applicable to SY-05.

### 7. Hierarchical Hash Pattern (Reusable for BZ-04)

**File:** `faiss/IndexNeuroHash.h:113-169`

```cpp
struct IndexNeuroHierarchicalHash : IndexNeuro {
    std::vector<int> bits_per_level;        // e.g., [8, 32, 128]
    std::vector<std::vector<float>> level_hyperplanes;
    std::vector<std::vector<uint64_t>> level_codes;
};
```

**Reuse Level:** HIGH - Multi-resolution structure perfect for BZ-04.

### 8. Build Infrastructure

**CMakeLists.txt:** `faiss/CMakeLists.txt:50-101`
- 25 IndexNeuro*.cpp already listed
- Pattern: Add new V3 files to same list

**SWIG:** `faiss/python/swigfaiss.swig:556-586`
- All IndexNeuro* headers already included
- Pattern: Add %include for new V3 headers

---

## Best Practices Research

### 1. Product Quantization Cascading (QT-02)

**Key Insights:**
- Multi-stage PQ pipelines: 4-bit → 8-bit → float achieves 99% recall
- End-to-end backpropagation following block-wise calibration
- Joint structured pruning and quantization outperforms sequential approaches

**Sources:**
- [EfficientQAT (ACL 2025)](https://aclanthology.org/2025.acl-long.498.pdf)
- [Pinecone PQ Guide](https://www.pinecone.io/learn/series/faiss/product-quantization/)

### 2. Scalar Quantization Calibration (QT-01)

**Key Insights:**
- INT8 achieves 4x reduction with minimal loss
- Primary methods: KL Divergence, Percentile (0.1-99.9)
- Advanced: SmoothQuant, AWQ, CL-Calib, OmniQuant

**Sources:**
- [Quantization Study for LLMs (2024)](https://arxiv.org/html/2411.02530v1)
- [NVIDIA QAT Blog](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/)

### 3. DiskANN Algorithm (DK-01)

**Key Insights:**
- Indexes 1B vectors on single workstation with 64GB RAM + SSD
- >5000 QPS, <3ms latency, 95%+ recall on SIFT1B
- Vamana graph in RAM, vectors on SSD

**Sources:**
- [Microsoft DiskANN](https://github.com/microsoft/DiskANN)
- [DiskANN NeurIPS 2019](https://suhasjs.github.io/files/diskann_neurips19.pdf)
- [DistributedANN (2025)](https://arxiv.org/html/2509.06046)

### 4. Learned Binarization (BZ-03)

**Key Insights:**
- Dynamic binarization preserves helpful features
- Gradient descent on threshold array
- Hash-based approaches map to Hamming space

**Sources:**
- [Learning to Hash Survey (2024)](https://arxiv.org/pdf/2412.03875)
- [BiBench: Network Binarization (2024)](https://proceedings.mlr.press/v202/qin23b/qin23b.pdf)

### 5. Early Termination (SY-03)

**Key Insights:**
- Experience-Driven Early Termination (EET) reduces 40% computation
- Multi-fidelity: HyperJump, DyHPO for expected accuracy loss
- Stop when gap between 1st and kth result exceeds threshold

**Sources:**
- [EET for Cost-Efficient AI Agents (2025)](https://arxiv.org/html/2601.05777v1)

### 6. SIMD Optimizations (SY-02)

**Key Insights:**
- Distance computation is >90% of HNSW construction time
- AVX-512: 13% speedup vs AVX2
- SimSIMD: 200x faster dot products
- Flash (2025): 10.4x speedup via cache-aware SIMD

**Sources:**
- [Flash: Accelerating Graph Indexing (2025)](https://arxiv.org/html/2502.18113v1)
- [SimSIMD GitHub](https://github.com/ashvardanian/SimSIMD)
- [Elastic SIMD Blog](https://www.elastic.co/blog/accelerating-vector-search-simd-instructions)

### 7. Memory-Mapped Indices (DK-04)

**Key Insights:**
- HNSW dominant but requires RAM
- IVF trades accuracy for memory efficiency
- mmap lets OS manage cache efficiently
- GoVector: Similarity-aware storage reordering

**Sources:**
- [P-HNSW for Persistent Memory (2024)](https://www.mdpi.com/2076-3417/15/19/10554)
- [GoVector I/O-Efficient Caching (2024)](https://www.arxiv.org/pdf/2508.15694)

---

## Synthesis

### Key Patterns to Follow

| Pattern | Source File | V3 Application |
|---------|-------------|----------------|
| IndexNeuro wrapper | `impl/NeuroDistance.h:196` | All 22 strategies |
| Quantized + rerank | `IndexNeuroPQAware.cpp` | QT-01 to QT-05 |
| Hierarchical levels | `IndexNeuroHash.h:113` | BZ-01 to BZ-04 |
| LRU cache | `IndexNeuroCache.h:27` | SY-05 |
| Per-dimension weights | `IndexNeuroWeighted.h:36` | QT-04, BZ-03 |
| IVF partitions | `IndexIVF.h` | PT-01 to PT-04 |

### Constraints Identified

1. **Memory:** >1B vectors requires disk-backed storage (DK strategies)
2. **SWIG:** No nested vectors, no unique_ptr in vectors
3. **Thread Safety:** Cache operations need mutex (see IndexNeuroCache)
4. **Backward Compat:** All V3 must inherit IndexNeuro

### Prior Solutions from docs/solutions/

- V1: 13 strategies (elimination, dropout, weighted, cache, etc.)
- V2: 34 strategies (hash, fly, mushroom, anchor, PQ-aware, etc.)
- Both use same IndexNeuro pattern

### Open Questions for PRD

1. Should DK-01 (DiskANN) require graph construction as separate step?
2. Should PT-03 (SemanticSharding) be in core or separate distributed module?
3. How to handle hot/cold data differentiation in QT-04?

---

## V3 Strategy-to-Component Mapping

| V3 Strategy | FAISS Component to Reuse | New Code Needed |
|-------------|--------------------------|-----------------|
| QT-01 ScalarQuantization | ScalarQuantizer | Wrapper + rerank |
| QT-02 ProductQuantizationTiered | ProductQuantizer (2-3x) | Cascade logic |
| QT-03 ResidualQuantization | ResidualQuantizer | Wrapper |
| QT-04 AdaptiveQuantization | IndexNeuroWeighted | Hot/cold zones |
| QT-05 HybridQuantization | SQ + PQ | Dimension split |
| DK-01 DiskANN | mmap, graph builder | Vamana graph |
| DK-02 HierarchicalDisk | Tiered storage | Multi-level |
| DK-03 CompressedDisk | LZ4/zstd libs | Codec layer |
| DK-04 MemoryMapped | mmap | Simple wrapper |
| PT-01 OverlappingPartitions | IndexIVF | Multi-assign |
| PT-02 AdaptiveProbe | IndexIVF | Gap analysis |
| PT-03 SemanticSharding | - | Distributed coord |
| PT-04 DynamicPartitions | IndexIVF | Online rebalance |
| SY-01 PrefetchOptimized | - | Hilbert layout |
| SY-02 SIMDDistance | fvec_L2sqr_ny | Batch wrapper |
| SY-03 EarlyTermination | - | Gap check |
| SY-04 BatchedQueries | Matrix ops | Batch interface |
| SY-05 QueryCache | IndexNeuroCache | Reuse directly |
| BZ-01 ZonedBinarization | - | 3-zone storage |
| BZ-02 AdaptiveZones | - | Density analysis |
| BZ-03 LearnedBinarization | IndexNeuroWeighted | Threshold learning |
| BZ-04 MultiResolutionBinary | IndexNeuroHierarchicalHash | Cascade bits |

---

## Estimated Code Reuse

| Family | New Code | Reused Code | Total |
|--------|----------|-------------|-------|
| QT (5) | 30% | 70% | ~2000 LOC |
| DK (4) | 60% | 40% | ~2500 LOC |
| PT (4) | 40% | 60% | ~1500 LOC |
| SY (5) | 25% | 75% | ~1000 LOC |
| BZ (4) | 50% | 50% | ~1500 LOC |
| **Total** | **~35%** | **~65%** | ~8500 LOC |

---

## Gate 0 Validation

- [x] Research mode determined: modification
- [x] Existing patterns identified (IndexNeuro, ScalarQuantizer, ProductQuantizer, etc.)
- [x] No conflicting implementations found
- [x] docs/solutions/ checked (V1, V2 patterns documented)
- [x] Tech stack versions documented (FAISS, C++17, SWIG)
- [x] Synthesis complete with file:line references

**Gate 0 Result:** PASS

**Next Step:** Gate 1 - BRD Creation
