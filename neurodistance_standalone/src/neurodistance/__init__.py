"""
NeuroDistance: Bio-inspired vector search strategies for FAISS.

Standalone package with bundled FAISS binaries.
"""

__version__ = "0.5.0"
__author__ = "NeuroDistance Team"

import os
import sys
import ctypes

# Get the directory where this package is installed
_package_dir = os.path.dirname(os.path.abspath(__file__))
_libs_dir = os.path.join(_package_dir, 'libs')

# Preload the shared libraries in the correct order
if os.path.exists(_libs_dir):
    # Load libfaiss.so first (main FAISS library)
    _libfaiss_path = os.path.join(_libs_dir, 'libfaiss.so')
    if os.path.exists(_libfaiss_path):
        ctypes.CDLL(_libfaiss_path, mode=ctypes.RTLD_GLOBAL)

    # Load callbacks library
    _callbacks_path = os.path.join(_libs_dir, 'libfaiss_python_callbacks.so')
    if os.path.exists(_callbacks_path):
        ctypes.CDLL(_callbacks_path, mode=ctypes.RTLD_GLOBAL)

    # Add libs dir to path for _swigfaiss.so
    if _libs_dir not in sys.path:
        sys.path.insert(0, _libs_dir)

# Now import swigfaiss (which loads _swigfaiss.so)
try:
    from . import swigfaiss as faiss
except ImportError as e:
    # Try loading from libs directory directly
    try:
        sys.path.insert(0, _libs_dir)
        import _swigfaiss
        from . import swigfaiss as faiss
    except ImportError:
        raise ImportError(
            f"Failed to load FAISS bindings. Error: {e}\n"
            f"Package dir: {_package_dir}\n"
            f"Libs dir: {_libs_dir}\n"
            f"Libs exist: {os.listdir(_libs_dir) if os.path.exists(_libs_dir) else 'NO'}"
        )

# Patch swigfaiss module to be accessible as 'faiss'
sys.modules['faiss'] = faiss

# Import our modules
from .memory import Memory
from .indexes import (
    INDEX_CATEGORIES,
    RECOMMENDED_INDEXES,
    list_indexes,
    get_recommended,
    create_index,
)

__all__ = [
    "faiss",
    "Memory",
    "INDEX_CATEGORIES",
    "RECOMMENDED_INDEXES",
    "list_indexes",
    "get_recommended",
    "create_index",
    "__version__",
]
