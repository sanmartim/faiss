"""
NeuroDistance index definitions and utilities.

This module provides:
- INDEX_CATEGORIES: Dictionary of all available indexes by category
- RECOMMENDED_INDEXES: Lists of recommended indexes for different use cases
- create_index(): Factory function to create indexes by name
- list_indexes(): Print all available indexes
- get_recommended(): Get recommended indexes for a use case
"""

import numpy as np

# Use bundled faiss from this package
try:
    from . import swigfaiss as faiss
except ImportError:
    try:
        import faiss
    except ImportError:
        faiss = None


# Index categories with descriptions
INDEX_CATEGORIES = {
    'standard': {
        'L2': 'L2 (Euclidean) distance - exact search',
        'IP': 'Inner product - exact search',
        'HNSW': 'Hierarchical NSW - fast approximate',
        'IVF': 'Inverted file index - good balance',
        'PQ': 'Product quantization - compact',
        'LSH': 'Locality sensitive hashing',
        'Q': 'Scalar quantization',
    },
    'v1_original': {
        'NEURO_ELIM': 'Column elimination (pruning low-variance dimensions)',
        'NEURO_WEIGHTED': 'Weighted distance (learned dimension weights)',
        'NEURO_CONTEXTUAL': 'Contextual weighting (query-dependent weights)',
        'NEURO_DROPOUT': 'Dropout ensemble (multiple random subspaces)',
        'NEURO_VOTING': 'Parallel voting (ensemble of sub-indexes)',
        'NEURO_COARSE': 'Coarse-to-fine (hierarchical search)',
    },
    'v2_bio': {
        'NEURO_HASH': 'Locality-sensitive hashing (bio-inspired)',
        'NEURO_FLYHASH': 'Fly-inspired sparse random projection',
        'NEURO_MUSHROOM': 'Mushroom body (sparse expansion + winner-take-all)',
        'NEURO_VALENCE': 'Valence-based similarity (emotional memory)',
        'NEURO_PLACECELL': 'Place cell (spatial memory with receptive fields)',
        'NEURO_GRIDCELL': 'Grid cell (periodic spatial encoding)',
        'NEURO_PATTERN': 'Pattern completion (Hopfield-like autoassociative)',
        'NEURO_REMAP': 'Remapping (context-dependent representations)',
        'NEURO_GRANULE': 'Granule cell (cerebellar expansion coding)',
        'NEURO_TEMPORAL': 'Temporal basis (sequence encoding)',
        'NEURO_ERRORDRIVEN': 'Error-driven learning (cerebellar model)',
        'NEURO_ANCHOR': 'Anchor-based similarity (landmark vectors)',
        'NEURO_ADAPTIVE_METRIC': 'Adaptive metric learning',
    },
    'v3_perf': {
        'NEURO_SCALAR_Q': 'Scalar quantization (8-bit)',
        'NEURO_PQ_TIERED': 'Tiered product quantization',
        'NEURO_ADAPTIVE_Q': 'Adaptive quantization',
        'NEURO_ZONED_BIN': 'Zoned binarization',
        'NEURO_ADAPTIVE_ZONES': 'Adaptive zone boundaries',
        'NEURO_LEARNED_BIN': 'Learned binarization thresholds',
        'NEURO_MULTIREZ_BIN': 'Multi-resolution binary codes',
        'NEURO_PREFETCH': 'Prefetch-optimized traversal',
        'NEURO_CACHE': 'Query result caching',
        'NEURO_DISKANN': 'DiskANN-style graph index',
        'NEURO_HIER_DISK': 'Hierarchical disk-based index',
        'NEURO_COMP_DISK': 'Compressed disk index',
        'NEURO_OVERLAP_PART': 'Overlapping partitions',
        'NEURO_ADAPTIVE_PROBE': 'Adaptive IVF probing',
        'NEURO_SEMANTIC_SHARD': 'Semantic sharding (69x speedup, 98% recall)',
        'NEURO_DYNAMIC_PART': 'Dynamic partitions (27x speedup, 100% recall)',
    },
    'v4_multiscale': {
        'NEURO_MS_SIGN': 'Multi-scale sign (binary, 50% recall limit)',
        'NEURO_ADAPTIVE_SCALE': 'Adaptive scale selection',
        'NEURO_HIER_SCALE': 'Hierarchical scales',
        'NEURO_MS_INTERSECT': 'Multi-scale intersection',
        'NEURO_LEARNED_SCALE': 'Learned optimal scales',
        'NEURO_HAMMING_PRE': 'Hamming prefilter',
        'NEURO_CENTROID_BOUNDS': 'Centroid bounding',
        'NEURO_PROJ_CASCADE': 'Projection cascade',
        'NEURO_STAT_PRESCREEN': 'Statistical prescreening',
        'NEURO_ENSEMBLE': 'Ensemble voting',
        'NEURO_PIPELINE': 'Recommended pipeline',
    },
    'v5_microzones': {
        'NEURO_MICROZONES': 'MicroZones 4 zones (100% recall)',
        'NEURO_MICROZONES_8': 'MicroZones 8 zones',
        'NEURO_MICROZONES_16': 'MicroZones 16 zones',
        'NEURO_MZS_100': 'MultiZoneSign 100 scales (tanh)',
        'NEURO_MZS_300': 'MultiZoneSign 300 scales',
        'NEURO_MZS_1000': 'MultiZoneSign 1000 scales',
        'NEURO_MZS_SIN': 'MultiZoneSign with sin function',
        'NEURO_MZS_SIGMOID': 'MultiZoneSign with sigmoid',
        'NEURO_MZS_ATAN': 'MultiZoneSign with atan',
        'NEURO_MZS_ERF': 'MultiZoneSign with erf',
    },
}

# Recommended indexes for different use cases
RECOMMENDED_INDEXES = {
    'production_100_recall': [
        ('NEURO_DYNAMIC_PART', '100% recall, 27x faster than L2'),
        ('IVF', '100% recall, 25-30x faster than L2'),
        ('HNSW', '99%+ recall, very fast'),
    ],
    'production_max_speed': [
        ('NEURO_SEMANTIC_SHARD', '98% recall, 69x faster than L2'),
        ('NEURO_DYNAMIC_PART', '100% recall, 27x faster than L2'),
        ('PQ', 'Low recall but very fast'),
    ],
    'research': [
        ('NEURO_MICROZONES', 'V5 - proves zone-based quantization'),
        ('NEURO_MZS_100', 'V5 - multi-scale with trig functions'),
        ('NEURO_PLACECELL', 'V2 - bio-inspired spatial memory'),
        ('NEURO_TEMPORAL', 'V2 - sequence encoding'),
    ],
    'memory_constrained': [
        ('PQ', 'Very compact representation'),
        ('NEURO_SCALAR_Q', 'Good balance of size and recall'),
        ('LSH', 'Compact binary codes'),
    ],
}


def list_indexes(category=None):
    """
    Print available indexes.

    Args:
        category: Optional category name to filter (e.g., 'v3_perf')
    """
    if category:
        if category in INDEX_CATEGORIES:
            print(f"\n{category.upper()}:")
            for name, desc in INDEX_CATEGORIES[category].items():
                print(f"  {name:<25} - {desc}")
        else:
            print(f"Unknown category: {category}")
            print(f"Available: {list(INDEX_CATEGORIES.keys())}")
    else:
        print("Available NeuroDistance Indexes:")
        print("=" * 70)
        for cat, indexes in INDEX_CATEGORIES.items():
            print(f"\n{cat.upper()}:")
            for name, desc in indexes.items():
                print(f"  {name:<25} - {desc}")
        print()


def get_recommended(use_case='production_100_recall'):
    """
    Get recommended indexes for a use case.

    Args:
        use_case: One of 'production_100_recall', 'production_max_speed',
                  'research', 'memory_constrained'

    Returns:
        List of (index_name, description) tuples
    """
    if use_case in RECOMMENDED_INDEXES:
        return RECOMMENDED_INDEXES[use_case]
    else:
        print(f"Unknown use case: {use_case}")
        print(f"Available: {list(RECOMMENDED_INDEXES.keys())}")
        return []


def create_index(metric, d, n=None):
    """
    Create a FAISS index by metric name.

    Args:
        metric: Index type name (e.g., 'NEURO_DYNAMIC_PART')
        d: Dimension of vectors
        n: Optional number of vectors (for index sizing)

    Returns:
        FAISS index object

    Raises:
        ImportError: If FAISS is not available
        ValueError: If metric is unknown
    """
    if faiss is None:
        raise ImportError("FAISS not found. Install with: pip install faiss-cpu")

    n = n or 1000
    ls = max(1, int(np.log10(n)))

    # Keep reference to inner index to prevent GC
    inner = None

    # Standard indexes
    if metric == 'L2':
        return faiss.IndexFlatL2(d)
    elif metric == 'IP':
        return faiss.IndexFlatIP(d)
    elif metric == 'HNSW':
        return faiss.IndexHNSWFlat(d, 32)
    elif metric == 'IVF':
        quantizer = faiss.IndexHNSWFlat(d, 32)
        return faiss.IndexIVFFlat(quantizer, d, max(ls, 10))
    elif metric == 'PQ':
        return faiss.IndexPQ(d, min(d, 16), 8)
    elif metric == 'LSH':
        return faiss.IndexLSH(d, d)
    elif metric == 'Q':
        return faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)

    # V1 Original
    elif metric == 'NEURO_ELIM':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroElimination(inner, 0.1), inner
    elif metric == 'NEURO_WEIGHTED':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroWeighted(inner), inner
    elif metric == 'NEURO_CONTEXTUAL':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroContextualWeighted(inner), inner
    elif metric == 'NEURO_DROPOUT':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroDropoutEnsemble(inner, 5, 0.3), inner
    elif metric == 'NEURO_VOTING':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroParallelVoting(inner, 3), inner

    # V2 Bio-Inspired
    elif metric == 'NEURO_HASH':
        return faiss.IndexNeuroHash(d, 64, 10)
    elif metric == 'NEURO_PLACECELL':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroPlaceCell(inner, 100, 0.1), inner
    elif metric == 'NEURO_PATTERN':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroPatternCompletion(inner, 5), inner
    elif metric == 'NEURO_TEMPORAL':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroTemporal(inner, 100), inner
    elif metric == 'NEURO_ADAPTIVE_METRIC':
        inner = faiss.IndexFlatL2(d)
        return faiss.IndexNeuroAdaptiveMetric(inner), inner

    # V3 Performance
    elif metric == 'NEURO_SCALAR_Q':
        return faiss.IndexNeuroScalarQuantization(d, 8)
    elif metric == 'NEURO_ADAPTIVE_Q':
        return faiss.IndexNeuroAdaptiveQuantization(d, 16, 0.2)
    elif metric == 'NEURO_ADAPTIVE_ZONES':
        return faiss.IndexNeuroAdaptiveZones(d, 8)
    elif metric == 'NEURO_DISKANN':
        return faiss.IndexNeuroDiskANN(d, 32)
    elif metric == 'NEURO_SEMANTIC_SHARD':
        return faiss.IndexNeuroSemanticSharding(d, 16)
    elif metric == 'NEURO_DYNAMIC_PART':
        return faiss.IndexNeuroDynamicPartitions(d, 32)

    # V4 Multi-Scale
    elif metric == 'NEURO_MS_SIGN':
        return faiss.IndexNeuroMultiScaleSign(d)
    elif metric == 'NEURO_ADAPTIVE_SCALE':
        return faiss.IndexNeuroAdaptiveScale(d, 3)
    elif metric == 'NEURO_HAMMING_PRE':
        return faiss.IndexNeuroHammingPrefilter(d, 0.15)
    elif metric == 'NEURO_PIPELINE':
        return faiss.IndexNeuroRecommendedPipeline(d)

    # V5 MicroZones
    elif metric == 'NEURO_MICROZONES':
        return faiss.IndexNeuroMicroZones(d, 4)
    elif metric == 'NEURO_MICROZONES_8':
        return faiss.IndexNeuroMicroZones(d, 8)
    elif metric == 'NEURO_MICROZONES_16':
        return faiss.IndexNeuroMicroZones(d, 16)
    elif metric == 'NEURO_MZS_100':
        return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_TANH, 2)
    elif metric == 'NEURO_MZS_300':
        return faiss.IndexNeuroMultiZoneSign(d, 300, faiss.TrigFunction_TANH, 2)
    elif metric == 'NEURO_MZS_1000':
        return faiss.IndexNeuroMultiZoneSign(d, 1000, faiss.TrigFunction_TANH, 2)
    elif metric == 'NEURO_MZS_SIN':
        return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_SIN, 2)
    elif metric == 'NEURO_MZS_SIGMOID':
        return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_SIGMOID, 2)
    elif metric == 'NEURO_MZS_ATAN':
        return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_ATAN, 2)
    elif metric == 'NEURO_MZS_ERF':
        return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_ERF, 2)

    else:
        raise ValueError(f"Unknown metric: {metric}. Use list_indexes() to see available options.")
