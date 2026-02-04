"""
Memory class for nearest neighbor search using FAISS indexes.
Includes all NeuroDistance strategies (V1-V5) plus standard FAISS indexes.
"""

import numpy as np

try:
    import sys
    sys.path.insert(0, 'build/faiss/python')
    import swigfaiss as faiss
except ImportError:
    import faiss


class Memory:
    """
    Nearest neighbor memory using various FAISS index types.

    Supports:
    - Standard FAISS indexes (L2, IP, HNSW, IVF, PQ, LSH, etc.)
    - NeuroDistance V1: Original bio-inspired (Elimination, Weighted, etc.)
    - NeuroDistance V2: Bio-inspired (Hash, PlaceCell, Temporal, etc.)
    - NeuroDistance V3: Performance (DynamicPartitions, SemanticSharding, etc.)
    - NeuroDistance V4: Multi-Scale Sign (50% recall limitation)
    - NeuroDistance V5: MicroZones, MultiZoneSign (100% recall)
    """

    # Index categories for documentation
    INDEX_CATEGORIES = {
        'standard': ['L2', 'IP', 'HNSW', 'IVF', 'PQ', 'LSH', 'Q'],
        'v1_original': ['NEURO_ELIM', 'NEURO_WEIGHTED', 'NEURO_CONTEXTUAL',
                        'NEURO_DROPOUT', 'NEURO_VOTING', 'NEURO_COARSE'],
        'v2_bio': ['NEURO_HASH', 'NEURO_FLYHASH', 'NEURO_MUSHROOM', 'NEURO_VALENCE',
                   'NEURO_PLACECELL', 'NEURO_GRIDCELL', 'NEURO_PATTERN',
                   'NEURO_REMAP', 'NEURO_GRANULE', 'NEURO_TEMPORAL',
                   'NEURO_ERRORDRIVEN', 'NEURO_ANCHOR', 'NEURO_ADAPTIVE_METRIC'],
        'v3_perf': ['NEURO_SCALAR_Q', 'NEURO_PQ_TIERED', 'NEURO_ADAPTIVE_Q',
                    'NEURO_ZONED_BIN', 'NEURO_ADAPTIVE_ZONES', 'NEURO_LEARNED_BIN',
                    'NEURO_MULTIREZ_BIN', 'NEURO_PREFETCH', 'NEURO_CACHE',
                    'NEURO_DISKANN', 'NEURO_HIER_DISK', 'NEURO_COMP_DISK',
                    'NEURO_OVERLAP_PART', 'NEURO_ADAPTIVE_PROBE',
                    'NEURO_SEMANTIC_SHARD', 'NEURO_DYNAMIC_PART'],
        'v4_multiscale': ['NEURO_MS_SIGN', 'NEURO_ADAPTIVE_SCALE',
                          'NEURO_HIER_SCALE', 'NEURO_MS_INTERSECT',
                          'NEURO_LEARNED_SCALE', 'NEURO_HAMMING_PRE',
                          'NEURO_CENTROID_BOUNDS', 'NEURO_PROJ_CASCADE',
                          'NEURO_STAT_PRESCREEN', 'NEURO_ENSEMBLE',
                          'NEURO_PIPELINE'],
        'v5_microzones': ['NEURO_MICROZONES', 'NEURO_MICROZONES_8',
                          'NEURO_MICROZONES_16', 'NEURO_MZS_100',
                          'NEURO_MZS_300', 'NEURO_MZS_1000',
                          'NEURO_MZS_SIN', 'NEURO_MZS_SIGMOID',
                          'NEURO_MZS_ATAN', 'NEURO_MZS_ERF']
    }

    def __init__(self, n_neighbors=5, metric='HNSW', ybin=False, norm=False, hidden=15):
        """
        Initialize Memory.

        Args:
            n_neighbors: Number of neighbors to retrieve
            metric: Index type (see INDEX_CATEGORIES for options)
            ybin: Whether y values are binary/categorical
            norm: Whether to normalize predictions
            hidden: Hidden layer size for lstsq mode
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.index = None
        self.X_train = None
        self.y_train = None
        self.ybin = ybin
        self.norm = norm
        self.h = hidden
        self.w = None
        self.lstsq = False
        self.al = False
        self._inner_index = None  # Keep reference for wrapped indexes

    def _create_index(self, d, X=None):
        """Create the FAISS index based on metric type."""
        n = X.shape[0] if X is not None else 1000
        ls = int(np.log10(n).round(0)) if n > 1 else 1

        # ============================================================
        # STANDARD FAISS INDEXES
        # ============================================================
        if self.metric == 'IP':
            return faiss.IndexFlatIP(d)
        elif self.metric == 'L2':
            return faiss.IndexFlatL2(d)
        elif self.metric == 'L1':
            return faiss.index_factory(d, 'Flat', faiss.METRIC_L1)
        elif self.metric == 'LSH':
            return faiss.IndexLSH(d, d)
        elif self.metric == 'ILSH':
            return faiss.IndexIDMap(faiss.IndexLSH(d, d))
        elif self.metric == 'PQ':
            return faiss.IndexPQ(d, min(d, 16), 8)
        elif self.metric == 'Q':
            return faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        elif self.metric == 'HNSW':
            return faiss.IndexHNSWFlat(d, 32)
        elif self.metric == 'HNSW2':
            return faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, 16)
        elif self.metric == 'IVF':
            quantizer = faiss.IndexHNSWFlat(d, 32)
            return faiss.IndexIVFFlat(quantizer, d, max(ls, 10))
        elif self.metric == 'IL2':
            quantizer = faiss.IndexFlatL2(d)
            return faiss.IndexIVFPQ(quantizer, d, 20, 5, 8)

        # Special metrics
        elif self.metric == 'JAC':
            return faiss.index_factory(d, 'Flat', faiss.METRIC_Jaccard)
        elif self.metric == 'BC':
            return faiss.index_factory(d, 'Flat', faiss.METRIC_BrayCurtis)
        elif self.metric == 'JEN':
            return faiss.index_factory(d, 'Flat', faiss.METRIC_JensenShannon)
        elif self.metric == 'CAN':
            return faiss.index_factory(d, 'Flat', faiss.METRIC_Canberra)
        elif self.metric == 'AIP':
            return faiss.index_factory(d, 'Flat', faiss.METRIC_ABS_INNER_PRODUCT)

        # Factory-based
        elif self.metric == 'SAN':
            return faiss.index_factory(d, f'PQ{min(d, 16)}x4fs,RFlat')
        elif self.metric == 'SAN2':
            return faiss.index_factory(d, f'IVF{ls},PQ20x4fs')
        elif self.metric == 'SAN3':
            return faiss.index_factory(d, f'IVF{ls},PQ32')

        # ============================================================
        # NEURODISTANCE V1: ORIGINAL
        # ============================================================
        elif self.metric == 'NEURO_ELIM':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroElimination(self._inner_index, 0.1)
        elif self.metric == 'NEURO_WEIGHTED':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroWeighted(self._inner_index)
        elif self.metric == 'NEURO_CONTEXTUAL':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroContextualWeighted(self._inner_index)
        elif self.metric == 'NEURO_DROPOUT':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroDropoutEnsemble(self._inner_index, 5, 0.3)
        elif self.metric == 'NEURO_VOTING':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroParallelVoting(self._inner_index, 3)
        elif self.metric == 'NEURO_COARSE':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroCoarseToFine(self._inner_index, 3)

        # ============================================================
        # NEURODISTANCE V2: BIO-INSPIRED
        # ============================================================
        elif self.metric == 'NEURO_HASH':
            return faiss.IndexNeuroHash(d, 64, 10)
        elif self.metric == 'NEURO_FLYHASH':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroFlyHash(self._inner_index, 20)
        elif self.metric == 'NEURO_MUSHROOM':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroMushroomBody(self._inner_index, 20, 100)
        elif self.metric == 'NEURO_VALENCE':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroValence(self._inner_index)
        elif self.metric == 'NEURO_PLACECELL':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroPlaceCell(self._inner_index, 100, 0.1)
        elif self.metric == 'NEURO_GRIDCELL':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroGridCell(self._inner_index, 10)
        elif self.metric == 'NEURO_PATTERN':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroPatternCompletion(self._inner_index, 5)
        elif self.metric == 'NEURO_REMAP':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroRemapping(self._inner_index)
        elif self.metric == 'NEURO_GRANULE':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroGranule(self._inner_index, 50, 200)
        elif self.metric == 'NEURO_TEMPORAL':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroTemporal(self._inner_index, 100)
        elif self.metric == 'NEURO_ERRORDRIVEN':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroErrorDriven(self._inner_index)
        elif self.metric == 'NEURO_ANCHOR':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroAnchor(self._inner_index, 50)
        elif self.metric == 'NEURO_ADAPTIVE_METRIC':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroAdaptiveMetric(self._inner_index)

        # ============================================================
        # NEURODISTANCE V3: PERFORMANCE
        # ============================================================
        elif self.metric == 'NEURO_SCALAR_Q':
            return faiss.IndexNeuroScalarQuantization(d, 8)
        elif self.metric == 'NEURO_PQ_TIERED':
            return faiss.IndexNeuroProductQuantizationTiered(d, 8, 8)
        elif self.metric == 'NEURO_ADAPTIVE_Q':
            return faiss.IndexNeuroAdaptiveQuantization(d, 16, 0.2)
        elif self.metric == 'NEURO_ZONED_BIN':
            return faiss.IndexNeuroZonedBinarization(d, 4)
        elif self.metric == 'NEURO_ADAPTIVE_ZONES':
            return faiss.IndexNeuroAdaptiveZones(d, 8)
        elif self.metric == 'NEURO_LEARNED_BIN':
            return faiss.IndexNeuroLearnedBinarization(d)
        elif self.metric == 'NEURO_MULTIREZ_BIN':
            return faiss.IndexNeuroMultiResolutionBinary(d)
        elif self.metric == 'NEURO_PREFETCH':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroPrefetchOptimized(self._inner_index)
        elif self.metric == 'NEURO_CACHE':
            self._inner_index = faiss.IndexFlatL2(d)
            return faiss.IndexNeuroQueryCache(self._inner_index, 100, 0.01)
        elif self.metric == 'NEURO_DISKANN':
            return faiss.IndexNeuroDiskANN(d, 32)
        elif self.metric == 'NEURO_HIER_DISK':
            return faiss.IndexNeuroHierarchicalDisk(d, 50)
        elif self.metric == 'NEURO_COMP_DISK':
            return faiss.IndexNeuroCompressedDisk(d, 50)
        elif self.metric == 'NEURO_OVERLAP_PART':
            return faiss.IndexNeuroOverlappingPartitions(d, 32)
        elif self.metric == 'NEURO_ADAPTIVE_PROBE':
            self._inner_index = faiss.IndexFlatL2(d)
            ivf = faiss.IndexIVFFlat(self._inner_index, d, 32)
            return faiss.IndexNeuroAdaptiveProbe(ivf)
        elif self.metric == 'NEURO_SEMANTIC_SHARD':
            return faiss.IndexNeuroSemanticSharding(d, 16)
        elif self.metric == 'NEURO_DYNAMIC_PART':
            return faiss.IndexNeuroDynamicPartitions(d, 32)

        # ============================================================
        # NEURODISTANCE V4: MULTI-SCALE SIGN (50% recall limitation)
        # ============================================================
        elif self.metric == 'NEURO_MS_SIGN':
            return faiss.IndexNeuroMultiScaleSign(d)
        elif self.metric == 'NEURO_ADAPTIVE_SCALE':
            return faiss.IndexNeuroAdaptiveScale(d, 3)
        elif self.metric == 'NEURO_HIER_SCALE':
            return faiss.IndexNeuroHierarchicalScale(d)
        elif self.metric == 'NEURO_MS_INTERSECT':
            return faiss.IndexNeuroMultiScaleIntersection(d)
        elif self.metric == 'NEURO_LEARNED_SCALE':
            return faiss.IndexNeuroLearnedScale(d, 5)
        elif self.metric == 'NEURO_HAMMING_PRE':
            return faiss.IndexNeuroHammingPrefilter(d, 0.15)
        elif self.metric == 'NEURO_CENTROID_BOUNDS':
            return faiss.IndexNeuroCentroidBounds(d, 200)
        elif self.metric == 'NEURO_PROJ_CASCADE':
            return faiss.IndexNeuroProjectionCascade(d)
        elif self.metric == 'NEURO_STAT_PRESCREEN':
            return faiss.IndexNeuroStatisticalPrescreen(d, 0.25)
        elif self.metric == 'NEURO_ENSEMBLE':
            return faiss.IndexNeuroEnsembleVoting(d)
        elif self.metric == 'NEURO_PIPELINE':
            return faiss.IndexNeuroRecommendedPipeline(d)

        # ============================================================
        # NEURODISTANCE V5: MICROZONES (100% recall!)
        # ============================================================
        elif self.metric == 'NEURO_MICROZONES':
            return faiss.IndexNeuroMicroZones(d, 4)
        elif self.metric == 'NEURO_MICROZONES_8':
            return faiss.IndexNeuroMicroZones(d, 8)
        elif self.metric == 'NEURO_MICROZONES_16':
            return faiss.IndexNeuroMicroZones(d, 16)
        elif self.metric == 'NEURO_MZS_100':
            return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_TANH, 2)
        elif self.metric == 'NEURO_MZS_300':
            return faiss.IndexNeuroMultiZoneSign(d, 300, faiss.TrigFunction_TANH, 2)
        elif self.metric == 'NEURO_MZS_1000':
            return faiss.IndexNeuroMultiZoneSign(d, 1000, faiss.TrigFunction_TANH, 2)
        elif self.metric == 'NEURO_MZS_SIN':
            return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_SIN, 2)
        elif self.metric == 'NEURO_MZS_SIGMOID':
            return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_SIGMOID, 2)
        elif self.metric == 'NEURO_MZS_ATAN':
            return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_ATAN, 2)
        elif self.metric == 'NEURO_MZS_ERF':
            return faiss.IndexNeuroMultiZoneSign(d, 100, faiss.TrigFunction_ERF, 2)

        # Default: L2
        else:
            return faiss.IndexFlatL2(d)

    def fit(self, X, y):
        """
        Fit the memory with training data.

        Args:
            X: Training features (N x D)
            y: Training targets
        """
        if hasattr(y, 'values'):
            y = y.values
        if hasattr(X, 'values'):
            X = X.values

        X = np.nan_to_num(X).astype('float32')

        self.X_train = X
        self.y_train = y

        d = X.shape[1]

        # Create the index
        self.index = self._create_index(d, X)

        # Train and add data
        try:
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                self.index.train(X.shape[0], faiss.swig_ptr(X))
            self.index.add(X.shape[0], faiss.swig_ptr(X))
        except Exception:
            try:
                self.index.train(X.shape[0], faiss.swig_ptr(X))
                self.index.add(X.shape[0], faiss.swig_ptr(X))
            except Exception:
                # Fallback for standard FAISS interface
                if hasattr(self.index, 'train'):
                    self.index.train(X)
                self.index.add(X)

    def predict(self, X, n=1, rerank=True):
        """
        Predict using nearest neighbors.

        Args:
            X: Query features
            n: Number of predictions to return (for classification)
            rerank: Whether to rerank results

        Returns:
            Predictions array
        """
        if hasattr(X, 'values'):
            X = X.values

        X = np.nan_to_num(X).astype('float32')

        # Search
        try:
            D_out = np.empty((X.shape[0], self.n_neighbors), dtype='float32')
            I_out = np.empty((X.shape[0], self.n_neighbors), dtype='int64')
            self.index.search(X.shape[0], faiss.swig_ptr(X), self.n_neighbors,
                            faiss.swig_ptr(D_out), faiss.swig_ptr(I_out))
            distances, indices = D_out, I_out
        except Exception:
            distances, indices = self.index.search(X, k=self.n_neighbors)

        predictions = [[]] * X.shape[0]
        metrics = [[]] * X.shape[0]

        for i in range(X.shape[0]):
            index = indices[i]
            # Handle -1 indices (no result)
            valid_mask = index >= 0
            index = index[valid_mask]

            if len(index) == 0:
                predictions[i] = 0 if not self.ybin else ['---'] * n
                continue

            nearest_neighbors = self.y_train[index]
            dist = distances[i][valid_mask]

            if self.lstsq:
                Xt = self.X_train[index]
                yt = self.y_train[index]
                h = np.random.normal(-1, 1, (Xt.shape[1], 200))
                Xt = np.tanh(Xt @ h)
                w = np.linalg.lstsq(Xt, yt, rcond=None)[0]
                Xp = X[i]
                Xp = np.tanh(Xp @ h)
                predictions[i] = np.dot(np.nan_to_num(Xp), np.nan_to_num(w))

            elif self.al:
                from sklearn.linear_model import LinearRegression
                clf = LinearRegression()
                Xt = self.X_train[index]
                yt = self.y_train[index]
                Xp = X[i]
                clf.fit(Xt, yt)
                predictions[i] = clf.predict([Xp])

            else:
                if self.ybin:
                    top_predictions = []
                    top_metrics = []
                    for j, pred in enumerate(nearest_neighbors):
                        if pred not in top_predictions:
                            top_predictions.append(pred)
                            top_metrics.append(dist[j])
                        if len(top_predictions) == n:
                            break
                    if len(top_predictions) < n:
                        top_predictions += ['---'] * (n - len(top_predictions))
                        top_metrics += ['---'] * (n - len(top_metrics))
                    predictions[i] = top_predictions
                    metrics[i] = top_metrics
                else:
                    predictions[i] = np.mean(nearest_neighbors)

        self.metrics = metrics
        return np.array(predictions)

    @classmethod
    def list_indexes(cls):
        """List all available index types by category."""
        print("Available index types:")
        print("=" * 60)
        for category, indexes in cls.INDEX_CATEGORIES.items():
            print(f"\n{category.upper()}:")
            for idx in indexes:
                print(f"  - {idx}")
        print()

    @classmethod
    def get_recommended(cls, recall_priority=True):
        """
        Get recommended indexes based on priority.

        Args:
            recall_priority: If True, prioritize recall; else prioritize speed

        Returns:
            List of recommended index names
        """
        if recall_priority:
            return [
                'NEURO_DYNAMIC_PART',   # 100% recall, 26x speedup
                'NEURO_MICROZONES',     # 100% recall (V5)
                'HNSW',                 # Standard, reliable
                'IVF',                  # Standard, good balance
            ]
        else:
            return [
                'NEURO_SEMANTIC_SHARD', # 98% recall, 69x speedup
                'NEURO_DYNAMIC_PART',   # 100% recall, 26x speedup
                'IVF',                  # Standard, good balance
                'PQ',                   # Fast but low recall
            ]
