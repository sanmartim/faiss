/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <vector>

namespace faiss {

/// Anchor selection method
enum NeuroAnchorSelection {
    NEURO_ANCHOR_RANDOM = 0,   ///< Random selection
    NEURO_ANCHOR_KMEANS,       ///< K-means clustering
    NEURO_ANCHOR_FARTHEST,     ///< Farthest point sampling
    NEURO_ANCHOR_LEARNED,      ///< MT-02: Gradient-optimized positions
};

/// Parameters for anchor search
struct NeuroAnchorParams : NeuroSearchParameters {
    bool rerank = true;        ///< whether to rerank with true distance
    int candidates_per_anchor = -1;  ///< -1 = use index default

    ~NeuroAnchorParams() override = default;
};

/** MT-01/MT-02/MT-03: Anchor-based Distance Approximation.
 *
 * Uses anchor points (landmarks) to approximate distances:
 *   profile(x) = [d(x, a_1), d(x, a_2), ..., d(x, a_m)]
 *
 * Triangle inequality: |d(x,y) - d(a,y)| <= d(x,a)
 * This bounds the true distance from profile differences.
 *
 * MT-01 (Basic): Random/KMeans/Farthest anchor selection
 * MT-02 (Learned): Gradient-based anchor position optimization
 * MT-03 (Hierarchical): Multi-level anchor cascade for 20x+ speedup
 *
 * References:
 *   - LAESA (Linear Approximating Eliminating Search Algorithm)
 *   - Pivot-based indexing methods
 */
struct IndexNeuroAnchor : IndexNeuro {
    /// Number of anchor points
    int n_anchors = 64;

    /// Anchor selection method
    NeuroAnchorSelection selection = NEURO_ANCHOR_KMEANS;

    /// Whether to rerank candidates with true distance
    bool rerank = true;

    /// Number of candidates to retrieve before reranking
    int n_candidates = 100;

    /// Optional pluggable metric
    NeuroMetric* metric = nullptr;

    // MT-02: Learning parameters
    float learning_rate = 0.01f;
    int n_optimization_steps = 100;

    // MT-03: Hierarchical parameters
    bool hierarchical = false;
    std::vector<int> anchors_per_level;  ///< e.g., {8, 32, 128}
    std::vector<int> candidates_per_level;  ///< candidates to keep at each level

    // Internal structures

    /// Anchor vectors: n_anchors * d (or sum of anchors_per_level * d if hierarchical)
    std::vector<float> anchors;

    /// Precomputed profiles: ntotal * n_anchors
    std::vector<float> profiles;

    /// For hierarchical: profiles per level
    std::vector<std::vector<float>> level_profiles;

    IndexNeuroAnchor() = default;

    /** Construct with inner IndexFlat.
     * @param inner      inner index (must be IndexFlat)
     * @param n_anchors  number of anchor points (default 64)
     * @param own_inner  take ownership
     */
    IndexNeuroAnchor(Index* inner, int n_anchors = 64, bool own_inner = false);

    /** Train: select/optimize anchor positions. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: compute and store profiles. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Set hierarchical anchor configuration
    void set_hierarchical(
            const std::vector<int>& anchors_per_level,
            const std::vector<int>& candidates_per_level);

    /// Compute profile for a vector
    void compute_profile(const float* x, float* profile) const;

    /// Compute hierarchical profile
    void compute_hierarchical_profile(
            const float* x,
            std::vector<std::vector<float>>& profiles) const;

    /// Distance between two profiles (L1 or L2)
    float profile_distance(const float* p1, const float* p2, int n) const;

private:
    /// Select anchors using specified method
    void select_anchors(idx_t n, const float* x);

    /// MT-02: Optimize anchor positions
    void optimize_anchors(idx_t n, const float* x);

    /// Farthest point sampling
    void farthest_point_sampling(idx_t n, const float* x);

    /// K-means anchor selection
    void kmeans_anchors(idx_t n, const float* x);
};

} // namespace faiss
