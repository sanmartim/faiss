/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace faiss {

/*************************************************************
 * NeuroDistance: Bio-inspired vector search strategies
 *
 * Strategies that exploit data structure through progressive
 * elimination, adaptive weights, and ensemble methods.
 *************************************************************/

/// Elimination strategy for progressive column-based search
enum NeuroEliminationStrategy {
    NEURO_FIXED = 0,              ///< Fixed column order, fixed cutoff
    NEURO_ADAPTIVE_DISPERSION,    ///< Cutoff adapts to column dispersion
    NEURO_VARIANCE_ORDER,         ///< Columns ordered by sampled variance
    NEURO_UNCERTAINTY_DEFERRED,   ///< Defers elimination when uncertain
};

/// Dropout mode for ensemble search
enum NeuroDropoutMode {
    NEURO_DROPOUT_RANDOM = 0,     ///< Independent random masks per view
    NEURO_DROPOUT_COMPLEMENTARY,  ///< Minimize overlap, guarantee coverage
    NEURO_DROPOUT_STRUCTURED,     ///< Semantic grouping (halves, etc.)
    NEURO_DROPOUT_ADVERSARIAL,    ///< Each view excludes previous top columns
};

/// Integration method for combining multi-view results
enum NeuroIntegrationMethod {
    NEURO_INTEGRATE_VOTING = 0,   ///< Count appearances across views
    NEURO_INTEGRATE_BORDA,        ///< Sum of ranks (lower = better)
    NEURO_INTEGRATE_MEAN_DIST,    ///< Average distances from views
    NEURO_INTEGRATE_FULL_RERANK,  ///< Full distance on union of candidates
};

/// Extended search parameters for NeuroDistance strategies
struct NeuroSearchParameters : SearchParameters {
    bool collect_stats = false;
    virtual ~NeuroSearchParameters() = default;
};

/// Parameters for progressive elimination strategies (ED-01..04)
struct NeuroEliminationParams : NeuroSearchParameters {
    std::vector<int> column_order;   ///< empty = default (reversed)
    float cutoff_percentile = 0.5f;  ///< fraction of candidates to keep
    int min_candidates = 0;          ///< 0 = auto (k*2)

    // ED-02 specific
    float dispersion_low = 0.3f;
    float dispersion_high = 0.7f;
    float cutoff_low_dispersion = 0.8f;
    float cutoff_high_dispersion = 0.3f;

    // ED-04 specific
    float confidence_threshold = 0.4f;
    int max_accumulated_columns = 3;

    virtual ~NeuroEliminationParams() = default;
};

/// Parameters for dropout ensemble (ED-05, ED-05v2)
struct NeuroDropoutParams : NeuroSearchParameters {
    int num_views = 7;  ///< V2: increased from 5 to 7 for better coverage
    float dropout_rate = 0.3f;
    NeuroDropoutMode dropout_mode = NEURO_DROPOUT_COMPLEMENTARY;
    NeuroIntegrationMethod integration = NEURO_INTEGRATE_FULL_RERANK;  ///< V2: FULL_RERANK for better recall
    int top_k_per_view = 0; ///< 0 = auto (k*2)

    virtual ~NeuroDropoutParams() = default;
};

/// Strategy for handling missing (NaN) values in distance computation
enum NeuroMissingStrategy {
    NEURO_MISSING_PROPORTIONAL = 0, ///< weight = (1 - missing_rate)
    NEURO_MISSING_THRESHOLD,        ///< ignore column if missing > threshold
    NEURO_MISSING_HYBRID,           ///< weight = (1 - missing_rate)^2
};

/// Parameters for missing value search (PA-03)
struct NeuroMissingValueParams : NeuroSearchParameters {
    NeuroMissingStrategy missing_strategy = NEURO_MISSING_HYBRID;
    float ignore_threshold = 0.8f;

    virtual ~NeuroMissingValueParams() = default;
};

/// Search statistics collected when collect_stats = true
struct NeuroSearchStats {
    int64_t calculations_performed = 0;
    int columns_used = 0;
    float time_ms = 0.0f;
};

/*************************************************************
 * MT-00: Pluggable Metric Interface
 *
 * Abstract distance computation allowing V2 strategies to use
 * different similarity/distance measures.
 *************************************************************/

/** Abstract base class for distance/similarity metrics.
 *
 * All V2 strategies can optionally accept a NeuroMetric* to
 * override the default L2 distance computation.
 */
struct NeuroMetric {
    virtual ~NeuroMetric() = default;

    /** Compute distance between two vectors.
     * @param x1  first vector of dimension d
     * @param x2  second vector of dimension d
     * @param d   dimensionality
     * @return distance value (lower = more similar for most metrics)
     */
    virtual float distance(const float* x1, const float* x2, int d) const = 0;

    /** Compute distances from query to multiple data vectors.
     * @param query  query vector of dimension d
     * @param data   n data vectors, contiguous (n * d floats)
     * @param n      number of data vectors
     * @param d      dimensionality
     * @param out    output array of n distances
     */
    virtual void distance_batch(
            const float* query,
            const float* data,
            idx_t n,
            int d,
            float* out) const;

    /// Whether distance(x, y) == distance(y, x)
    virtual bool is_symmetric() const { return true; }

    /// Whether lower values indicate more similar vectors
    virtual bool lower_is_better() const { return true; }
};

/// L2 (Euclidean squared) distance metric
struct NeuroMetricL2 : NeuroMetric {
    float distance(const float* x1, const float* x2, int d) const override;
    void distance_batch(
            const float* query,
            const float* data,
            idx_t n,
            int d,
            float* out) const override;
};

/// Cosine distance: 1 - cosine_similarity
struct NeuroMetricCosine : NeuroMetric {
    float distance(const float* x1, const float* x2, int d) const override;
};

/// Inner product (dot product) - higher is better
struct NeuroMetricDot : NeuroMetric {
    float distance(const float* x1, const float* x2, int d) const override;
    bool lower_is_better() const override { return false; }
};

/// Mahalanobis distance with diagonal covariance
struct NeuroMetricMahalanobis : NeuroMetric {
    std::vector<float> inv_variances;  ///< 1/variance per dimension

    explicit NeuroMetricMahalanobis(const std::vector<float>& variances);
    NeuroMetricMahalanobis() = default;

    /// Compute inverse variances from data (n samples, d dimensions)
    void fit(const float* data, idx_t n, int d);

    float distance(const float* x1, const float* x2, int d) const override;
};

/// Jaccard distance for sparse binary vectors (stored as floats, 0/1)
struct NeuroMetricJaccard : NeuroMetric {
    float distance(const float* x1, const float* x2, int d) const override;
};

/*************************************************************
 * V4: Multi-Scale Sign (MS) Family Parameters
 *
 * Strategies using sign(tanh(x * scale)) at multiple scales
 * for ultra-fast XOR-based candidate filtering.
 *************************************************************/

/// Parameters for MS-01: MultiScaleSign
struct NeuroMultiScaleSignParams : NeuroSearchParameters {
    std::vector<float> scales = {0.5f, 1.0f, 2.0f};  ///< scale factors
    int max_hamming_distance = 0;  ///< 0 = auto (d/16 per scale)
    bool use_intersection = true;  ///< require match on all scales

    virtual ~NeuroMultiScaleSignParams() = default;
};

/// Parameters for MS-02: AdaptiveScale
struct NeuroAdaptiveScaleParams : NeuroSearchParameters {
    int num_scales = 3;
    float min_entropy = 0.8f;  ///< minimum entropy threshold for thresholds
    int sample_size = 10000;   ///< samples for threshold learning

    virtual ~NeuroAdaptiveScaleParams() = default;
};

/// Parameters for MS-03: HierarchicalScale
struct NeuroHierarchicalScaleParams : NeuroSearchParameters {
    std::vector<float> cascade_scales = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};
    std::vector<float> keep_ratios = {0.5f, 0.5f, 0.5f, 0.5f};
    int target_candidates = 0;  ///< 0 = use all levels

    virtual ~NeuroHierarchicalScaleParams() = default;
};

/// Parameters for MS-04: MultiScaleIntersection
struct NeuroMultiScaleIntersectionParams : NeuroSearchParameters {
    std::vector<float> scales = {0.5f, 1.0f, 2.0f};
    bool strict_intersection = true;  ///< exact match required
    int fallback_k = 100;  ///< fallback candidates if bucket empty

    virtual ~NeuroMultiScaleIntersectionParams() = default;
};

/// Parameters for MS-05: LearnedScale
struct NeuroLearnedScaleParams : NeuroSearchParameters {
    int num_scales = 5;
    int search_iterations = 100;  ///< random search iterations
    int validation_queries = 1000;
    int validation_k = 10;

    virtual ~NeuroLearnedScaleParams() = default;
};

/*************************************************************
 * V4: Fast-Slow (FS) Family Parameters
 *
 * Kahneman-inspired System 1 (cheap) / System 2 (precise) filtering.
 *************************************************************/

/// Parameters for FS-01: HammingPrefilter
struct NeuroHammingPrefilterParams : NeuroSearchParameters {
    float keep_ratio = 0.10f;  ///< keep top 10% by Hamming distance
    int max_hamming_bits = 0;  ///< 0 = auto (d/4)

    virtual ~NeuroHammingPrefilterParams() = default;
};

/// Parameters for FS-02: CentroidBounds
struct NeuroCentroidBoundsParams : NeuroSearchParameters {
    int nlist = 1000;  ///< number of clusters
    int nprobe = 10;   ///< clusters to search
    bool use_triangle_inequality = true;

    virtual ~NeuroCentroidBoundsParams() = default;
};

/// Parameters for FS-03: ProjectionCascade
struct NeuroProjectionCascadeParams : NeuroSearchParameters {
    std::vector<int> projection_dims = {8, 32, 128};
    std::vector<float> keep_ratios = {0.10f, 0.10f, 0.10f};

    virtual ~NeuroProjectionCascadeParams() = default;
};

/// Parameters for FS-04: StatisticalPrescreen
struct NeuroStatisticalPrescreenParams : NeuroSearchParameters {
    float keep_ratio = 0.20f;  ///< keep top 20%
    float norm_weight = 1.0f;
    float mean_weight = 1.0f;

    virtual ~NeuroStatisticalPrescreenParams() = default;
};

/// Parameters for FS-05: EnsembleVoting
struct NeuroEnsembleVotingParams : NeuroSearchParameters {
    int min_votes = 2;
    bool use_hamming = true;
    bool use_stats = true;
    bool use_projection = true;

    virtual ~NeuroEnsembleVotingParams() = default;
};

/// Parameters for FS-06: RecommendedPipeline
struct NeuroRecommendedPipelineParams : NeuroSearchParameters {
    float ms_keep_ratio = 0.15f;  ///< after MS stage
    float fs_keep_ratio = 0.05f;  ///< after FS stage (before L2)

    virtual ~NeuroRecommendedPipelineParams() = default;
};

/** Base class for all NeuroDistance index wrappers.
 *
 * Wraps an inner index (typically IndexFlat) and overrides search()
 * with a bio-inspired strategy. Storage operations (add, reset,
 * reconstruct) are delegated to the inner index.
 */
struct IndexNeuro : Index {
    Index* inner_index = nullptr; ///< the underlying index (owns the data)
    bool own_inner = false;       ///< whether to delete inner_index in dtor

    /// Last search stats (only populated if collect_stats=true)
    mutable NeuroSearchStats last_stats;

    IndexNeuro() = default;

    /** Construct wrapping an existing index.
     * @param inner_index  the index to wrap
     * @param own_inner    if true, inner_index is deleted in destructor
     */
    IndexNeuro(Index* inner_index, bool own_inner = false);

    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;

    ~IndexNeuro() override;
};

} // namespace faiss
