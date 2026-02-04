"""
NeuroDistance: Bio-inspired vector search strategies for FAISS.

This package provides 60+ neural-inspired index types for approximate
nearest neighbor search, organized into 5 versions:

- V1 (Original): Elimination, Weighted, Dropout, etc.
- V2 (Bio-Inspired): Hash, PlaceCell, Temporal, etc.
- V3 (Performance): DynamicPartitions, SemanticSharding, etc.
- V4 (Multi-Scale Sign): 50% recall limitation
- V5 (MicroZones): 100% recall with zone-based quantization

Quick Start:
    from neurodistance import Memory

    mem = Memory(n_neighbors=5, metric='NEURO_DYNAMIC_PART')
    mem.fit(X_train, y_train)
    predictions = mem.predict(X_test)

For production use with 100% recall:
    - NEURO_DYNAMIC_PART: 27x faster than L2, 100% recall
    - NEURO_SEMANTIC_SHARD: 69x faster than L2, 98% recall
"""

__version__ = "0.5.0"
__author__ = "NeuroDistance Team"

from .memory import Memory
from .indexes import (
    # Index categories
    INDEX_CATEGORIES,
    RECOMMENDED_INDEXES,

    # Utility functions
    list_indexes,
    get_recommended,
    create_index,
)

__all__ = [
    "Memory",
    "INDEX_CATEGORIES",
    "RECOMMENDED_INDEXES",
    "list_indexes",
    "get_recommended",
    "create_index",
    "__version__",
]
