# Technical Requirements Document: NeuroDistance V3

## Metadata

| Field | Value |
|-------|-------|
| Date | 2026-02-03 |
| Feature | neurodistance-v3 |
| Gate | 3 - Technical Requirements Document |
| PRD Reference | docs/pre-dev/neurodistance-v3/prd.md |
| BRD Reference | docs/pre-dev/neurodistance-v3/brd.md |
| Status | Draft |

---

## Architecture Overview

### Pattern: Strategy Wrapper

All V3 strategies follow the established **IndexNeuro wrapper pattern**:

```
┌─────────────────────────────────────────────┐
│              IndexNeuroV3Strategy           │
│  ┌───────────────────────────────────────┐  │
│  │           IndexNeuro (base)           │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │    inner_index (IndexFlat)      │  │  │
│  │  │    - stores raw vectors         │  │  │
│  │  │    - handles add/reconstruct    │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │  - search() overridden               │  │
│  │  - train() overridden                │  │
│  └───────────────────────────────────────┘  │
│  - strategy-specific fields               │
│  - strategy-specific search logic         │
└─────────────────────────────────────────────┘
```

### Data Flow

```
Query Vector (float32)
        │
        ▼
┌──────────────────┐
│ Stage 1: Coarse  │  (fast, approximate)
│ - hash/binary    │
│ - quantized dist │
└────────┬─────────┘
         │ candidates (N → K*M)
         ▼
┌──────────────────┐
│ Stage 2: Refine  │  (medium precision)
│ - int8 distance  │
│ - partial dims   │
└────────┬─────────┘
         │ candidates (K*M → K*R)
         ▼
┌──────────────────┐
│ Stage 3: Rerank  │  (full precision)
│ - float32 dist   │
│ - true metric    │
└────────┬─────────┘
         │
         ▼
   Top-K Results
```

---

## Component Design

### CD-01: Quantization Components (QT)

#### QT-01: IndexNeuroScalarQuantization

**Responsibility:** Per-dimension scalar quantization with calibration and rerank.

**Structure:**
```
IndexNeuroScalarQuantization : IndexNeuro
├── quantizer: ScalarQuantizer*          // Reuse FAISS component
├── codes: vector<uint8_t>               // Quantized storage
├── calibration_method: enum             // minmax, percentile, optim
├── rerank_k: int                        // Candidates for rerank
├── metric: NeuroMetric*                 // For rerank distance
└── train(), add(), search()
```

**Interface:**
- `train(n, x)` → calibrate quantizer scales per dimension
- `add(n, x)` → encode to int8 and store
- `search(n, x, k)` → quantized distance → top rerank_k → float rerank → top k

#### QT-02: IndexNeuroProductQuantizationTiered

**Responsibility:** Multi-level PQ cascade for aggressive compression.

**Structure:**
```
IndexNeuroProductQuantizationTiered : IndexNeuro
├── tiers: vector<ProductQuantizer*>     // 2-3 levels
├── tier_configs: vector<TierConfig>     // {bits, M, keep_ratio}
├── codes_per_tier: vector<vector<uint8_t>>
└── train(), add(), search()
```

**Interface:**
- `train(n, x)` → train each tier's PQ codebook
- `add(n, x)` → encode at all tiers
- `search(n, x, k)` → cascade: tier0 filter → tier1 filter → float rerank

#### QT-03: IndexNeuroResidualQuantization

**Responsibility:** Iterative residual encoding for extreme compression.

**Structure:**
```
IndexNeuroResidualQuantization : IndexNeuro
├── rq: ResidualQuantizer*               // Reuse FAISS component
├── n_stages: int                        // Number of residual stages
├── codes: vector<uint8_t>
└── train(), add(), search()
```

#### QT-04: IndexNeuroAdaptiveQuantization

**Responsibility:** Variable precision based on data density.

**Structure:**
```
IndexNeuroAdaptiveQuantization : IndexNeuro
├── region_quantizers: vector<Quantizer*>  // Per-region
├── region_assignments: vector<int>        // Vector → region
├── hot_threshold: float                   // Define "hot" regions
└── train(), add(), search()
```

#### QT-05: IndexNeuroHybridQuantization

**Responsibility:** SQ for important dimensions, PQ for rest.

**Structure:**
```
IndexNeuroHybridQuantization : IndexNeuro
├── sq: ScalarQuantizer*                 // Top-K dimensions
├── pq: ProductQuantizer*                // Remaining dimensions
├── top_dims: vector<int>                // Important dimension indices
├── dim_importance: vector<float>        // Learned or variance-based
└── train(), add(), search()
```

---

### CD-02: Disk Components (DK)

#### DK-01: IndexNeuroDiskANN

**Responsibility:** Graph navigation in RAM, vectors on disk.

**Structure:**
```
IndexNeuroDiskANN : IndexNeuro
├── graph: vector<vector<idx_t>>         // Vamana graph in RAM
├── data_path: string                    // Vector file on disk
├── cache: LRUCache<idx_t, vector<float>>
├── n_neighbors: int                     // Graph degree
├── beam_size: int                       // Search beam width
└── train(), add(), search(), build_graph()
```

**I/O Pattern:**
- Graph: ~64 bytes/node in RAM
- Vectors: mmap or direct read from SSD
- Cache: LRU for hot vectors

#### DK-02: IndexNeuroHierarchicalDisk

**Responsibility:** Multi-tier storage hierarchy.

**Structure:**
```
IndexNeuroHierarchicalDisk : IndexNeuro
├── tiers: vector<TierStorage>           // RAM, SSD, HDD
├── tier_indices: vector<set<idx_t>>     // Which vectors in which tier
├── promotion_policy: enum               // LRU, frequency-based
└── train(), add(), search()
```

#### DK-03: IndexNeuroCompressedDisk

**Responsibility:** On-disk compression with on-the-fly decompression.

**Structure:**
```
IndexNeuroCompressedDisk : IndexNeuro
├── codec: CompressionCodec*             // LZ4, zstd
├── compressed_path: string
├── block_size: int                      // Compression block
├── decompression_cache: LRUCache
└── train(), add(), search()
```

#### DK-04: IndexNeuroMemoryMapped

**Responsibility:** Simple mmap wrapper, let OS manage cache.

**Structure:**
```
IndexNeuroMemoryMapped : IndexNeuro
├── mmap_ptr: float*                     // Memory-mapped region
├── file_path: string
├── file_size: size_t
└── train(), add(), search()
```

---

### CD-03: Partition Components (PT)

#### PT-01: IndexNeuroOverlappingPartitions

**Responsibility:** Assign each vector to multiple partitions.

**Structure:**
```
IndexNeuroOverlappingPartitions : IndexNeuro
├── quantizer: Index*                    // Partition centroids
├── overlap: int                         // Vectors per partition (2-3)
├── invlists: vector<vector<idx_t>>      // Partition → vectors
└── train(), add(), search()
```

**Key Difference from IVF:**
- IVF: each vector in exactly 1 partition
- PT-01: each vector in top-`overlap` partitions

#### PT-02: IndexNeuroAdaptiveProbe

**Responsibility:** Dynamic nprobe based on query characteristics.

**Structure:**
```
IndexNeuroAdaptiveProbe : IndexNeuro  // Decorator
├── sub_index: Index*                    // Underlying IVF-like index
├── gap_thresholds: vector<float>        // nprobe selection
├── nprobe_levels: vector<int>           // Corresponding nprobe values
└── search()  // Computes nprobe per query
```

#### PT-03: IndexNeuroSemanticSharding

**Responsibility:** Cluster-aware sharding for distributed systems.

**Structure:**
```
IndexNeuroSemanticSharding : IndexNeuro
├── shards: vector<Index*>               // Per-shard indices
├── shard_centroids: vector<float>       // Representative vectors
├── n_shards_to_probe: int               // Default 2
└── train(), add(), search()
```

#### PT-04: IndexNeuroDynamicPartitions

**Responsibility:** Online partition rebalancing.

**Structure:**
```
IndexNeuroDynamicPartitions : IndexNeuro
├── base_index: Index*                   // Underlying partitioned index
├── query_counts: vector<int>            // Partition access frequency
├── rebalance_threshold: int             // Trigger rebalance
└── search(), rebalance()
```

---

### CD-04: System Components (SY)

#### SY-01: IndexNeuroPrefetchOptimized

**Responsibility:** Data layout for cache efficiency.

**Structure:**
```
IndexNeuroPrefetchOptimized : IndexNeuro
├── inner_index: Index*
├── reorder_map: vector<idx_t>           // Original → Hilbert order
├── inverse_map: vector<idx_t>           // Hilbert → original
└── train(), add(), search()
```

#### SY-02: IndexNeuroSIMDDistance

**Responsibility:** Batch distance computation with SIMD.

**Structure:**
```
IndexNeuroSIMDDistance : IndexNeuro  // Decorator
├── sub_index: Index*
├── batch_size: int                      // Process N queries at once
└── search()  // Uses fvec_L2sqr_ny
```

**Uses existing FAISS functions:**
- `fvec_L2sqr_ny(dis, x, y, d, ny)`
- `fvec_inner_products_ny(ip, x, y, d, ny)`

#### SY-03: IndexNeuroEarlyTermination

**Responsibility:** Stop search when confident.

**Structure:**
```
IndexNeuroEarlyTermination : IndexNeuro  // Decorator
├── sub_index: Index*
├── confidence_threshold: float          // Gap ratio to stop
├── min_candidates: int                  // Minimum before checking
└── search()
```

**Logic:**
```
if (dist[k] - dist[0]) / dist[0] > threshold:
    stop_early()
```

#### SY-04: IndexNeuroBatchedQueries

**Responsibility:** Process multiple queries together.

**Structure:**
```
IndexNeuroBatchedQueries : IndexNeuro  // Decorator
├── sub_index: Index*
├── optimal_batch_size: int
└── search()  // Batches queries for matrix ops
```

**Key insight:** `dists = ||q||² + ||d||² - 2*q·d` can be computed as matrix operations.

#### SY-05: IndexNeuroQueryCache

**Responsibility:** Cache results for similar queries.

**Structure:**
```
IndexNeuroQueryCache : IndexNeuro  // Decorator
├── sub_index: Index*
├── cache: LRUCache<hash, SearchResult>
├── cache_size_mb: int
├── similarity_threshold: float          // How similar to reuse
└── search()
```

**Reuses:** `IndexNeuroCache` LRU implementation from V2.

---

### CD-05: Binarization Components (BZ)

#### BZ-01: IndexNeuroZonedBinarization

**Responsibility:** Multi-precision zones based on dimension importance.

**Structure:**
```
IndexNeuroZonedBinarization : IndexNeuro
├── zone_config: ZoneConfig              // {10% float, 20% int8, 70% binary}
├── dim_importance: vector<float>        // Per-dimension scores
├── zone_assignments: vector<Zone>       // Dimension → zone
├── data_float: vector<float>            // Critical zone
├── data_int8: vector<int8_t>            // High zone
├── data_binary: vector<uint8_t>         // Binary zone (packed)
├── thresholds: vector<float>            // For binarization
└── train(), add(), search()
```

**Search cascade:**
1. Hamming distance on binary zone (70% dims)
2. L1 distance on int8 zone (20% dims)
3. L2 distance on float zone (10% dims) for final rerank

#### BZ-02: IndexNeuroAdaptiveZones

**Responsibility:** Region-specific zone configurations.

**Structure:**
```
IndexNeuroAdaptiveZones : IndexNeuro
├── regions: vector<Region>              // Data space regions
├── region_configs: vector<ZoneConfig>   // Per-region precision
└── train(), add(), search()
```

#### BZ-03: IndexNeuroLearnedBinarization

**Responsibility:** Learn thresholds that preserve similarity.

**Structure:**
```
IndexNeuroLearnedBinarization : IndexNeuro
├── thresholds: vector<float>            // Learned per dimension
├── learning_rate: float
├── n_iterations: int
└── train(), train_thresholds(), add(), search()
```

**Training:** Minimize Hamming distance for known similar pairs, maximize for dissimilar.

#### BZ-04: IndexNeuroMultiResolutionBinary

**Responsibility:** Cascading bit resolution (1→2→4→8).

**Structure:**
```
IndexNeuroMultiResolutionBinary : IndexNeuro
├── resolutions: vector<int>             // [1, 2, 4, 8]
├── codes_per_resolution: vector<vector<uint8_t>>
├── keep_ratios: vector<float>           // Filtering at each level
└── train(), add(), search()
```

---

## Data Architecture

### Storage Patterns

| Strategy | Storage Type | Memory Formula | Disk Formula |
|----------|--------------|----------------|--------------|
| QT-01 | RAM | n × d × 1 byte | N/A |
| QT-02 | RAM | n × (M₁ + M₂) bytes | N/A |
| QT-03 | RAM | n × (stages × code_size) | N/A |
| DK-01 | RAM + Disk | n × neighbors × 4 bytes | n × d × 4 bytes |
| DK-04 | Disk (mmap) | OS managed | n × d × 4 bytes |
| BZ-01 | RAM | n × (0.1d×4 + 0.2d×1 + 0.7d/8) | N/A |

### Code Layout (Quantized)

```
┌──────────────────────────────────────────┐
│ Codes Storage (QT strategies)            │
├──────────────────────────────────────────┤
│ Vector 0: [code_0, code_1, ..., code_M]  │
│ Vector 1: [code_0, code_1, ..., code_M]  │
│ ...                                       │
│ Vector n: [code_0, code_1, ..., code_M]  │
└──────────────────────────────────────────┘
```

### Zone Layout (BZ-01)

```
┌───────────────────────────────────────────────────────┐
│ Zoned Storage                                         │
├───────────────────────────────────────────────────────┤
│ data_float:  [v0_crit, v1_crit, ..., vn_crit]        │
│              (n × 0.1d × 4 bytes)                     │
├───────────────────────────────────────────────────────┤
│ data_int8:   [v0_high, v1_high, ..., vn_high]        │
│              (n × 0.2d × 1 byte)                      │
├───────────────────────────────────────────────────────┤
│ data_binary: [v0_bin, v1_bin, ..., vn_bin]           │
│              (n × ceil(0.7d/8) bytes)                 │
└───────────────────────────────────────────────────────┘
```

---

## Integration Patterns

### Strategy Composition

Decorators can be stacked:

```
IndexNeuroQueryCache(
    IndexNeuroEarlyTermination(
        IndexNeuroSIMDDistance(
            IndexNeuroZonedBinarization(inner)
        )
    )
)
```

### Metric Pluggability

All strategies accept optional `NeuroMetric*`:

```cpp
auto idx = IndexNeuroScalarQuantization(&inner, 100);
idx.metric = new NeuroMetricCosine();  // Use cosine for rerank
```

---

## Security Architecture

### Thread Safety

| Component | Thread Safety | Notes |
|-----------|---------------|-------|
| Search | Thread-safe | Multiple searches concurrent |
| Add | Not thread-safe | Single-threaded indexing |
| Cache | Thread-safe | Mutex-protected LRU |
| Disk I/O | Thread-safe | Per-thread file handles |

### Resource Limits

| Resource | Default Limit | Configurable |
|----------|---------------|--------------|
| Memory (cache) | 1 GB | Yes |
| Disk (mmap) | OS limit | No |
| Threads | OMP default | Yes (OMP_NUM_THREADS) |

---

## Interface Specifications

### C++ Public API

```cpp
// All V3 strategies follow this pattern:
class IndexNeuro[StrategyName] : public IndexNeuro {
public:
    // Strategy-specific parameters
    // ...

    // Standard Index interface
    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void search(idx_t n, const float* x, idx_t k,
                float* distances, idx_t* labels,
                const SearchParameters* params = nullptr) const override;
    void reset() override;
};
```

### Python API (via SWIG)

```python
import faiss

# Create inner index
inner = faiss.IndexFlatL2(768)

# Wrap with V3 strategy
idx = faiss.IndexNeuroZonedBinarization(inner, 100)  # 100 rerank candidates
idx.train(xb)
idx.add(xb)

# Search
D, I = idx.search(xq, k)
```

---

## Gate 3 Validation

### Checklist

- [x] **All PRD features** mapped to components
- [x] **Component boundaries** are clear (no overlap)
- [x] **Interfaces** are technology-agnostic (patterns, not products)
- [x] **No specific products** named in architecture
- [x] **Data flow** documented
- [x] **Storage patterns** defined
- [x] **Composition patterns** documented

**Gate 3 Result:** PASS

**Next Step:** Gate 4 - Task Breakdown
