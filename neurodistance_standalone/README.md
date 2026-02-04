# NeuroDistance Standalone

Bio-inspired vector search strategies with bundled FAISS binaries.

**This is a standalone package** - no separate FAISS installation required!

## Installation

```bash
pip install neurodistance_standalone-0.5.0-py3-none-linux_x86_64.whl
```

## Google Colab

```python
# Upload the wheel file to Colab, then:
!pip install neurodistance_standalone-0.5.0-py3-none-linux_x86_64.whl

from neurodistance import Memory, faiss

# Create index
mem = Memory(n_neighbors=5, metric='NEURO_DYNAMIC_PART')
mem.fit(X_train, y_train)
predictions = mem.predict(X_test)

# Or use faiss directly
index = faiss.IndexFlatL2(128)
```

## Quick Start

```python
from neurodistance import Memory, list_indexes, faiss

# List all 60+ available indexes
list_indexes()

# Create a Memory with bio-inspired index
mem = Memory(n_neighbors=5, metric='NEURO_DYNAMIC_PART')
mem.fit(X_train, y_train)
predictions = mem.predict(X_test)

# Access FAISS directly
index = faiss.IndexNeuroMicroZones(128, 4)
```

## Recommended Indexes for Production

| Index | Recall | Speed |
|-------|--------|-------|
| `NEURO_DYNAMIC_PART` | 100% | 27x faster than L2 |
| `NEURO_SEMANTIC_SHARD` | 98% | 69x faster than L2 |
| `NEURO_MICROZONES` | 100% | Zone-based quantization |

## Platform Support

- Linux x86_64 (Ubuntu 20.04+)
- Google Colab
- Other Linux with glibc 2.31+

## License

MIT License
