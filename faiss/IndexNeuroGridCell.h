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

/// Parameters for grid cell search
struct NeuroGridCellParams : NeuroSearchParameters {
    std::vector<float> scale_weights;  ///< empty = use index weights

    ~NeuroGridCellParams() override = default;
};

/** HP-02: Grid Cell Metric.
 *
 * Inspired by entorhinal cortex grid cells that represent space
 * at multiple scales with periodic firing patterns.
 *
 * Projects vectors to multiple scales and combines distances:
 *   - Scale 1: Fine-grained local structure
 *   - Scale 2: Medium-grained structure
 *   - Scale N: Coarse global structure
 *
 * Each scale uses a random projection matrix. Combining scales
 * provides robustness: coarse scales handle global position,
 * fine scales discriminate locally.
 */
struct IndexNeuroGridCell : IndexNeuro {
    /// Number of scales
    int n_scales = 4;

    /// Scale factors (projection dimension = d / factor)
    std::vector<float> scale_factors;

    /// Weights for combining scale distances
    std::vector<float> scale_weights;

    /// Projection matrices per scale
    std::vector<std::vector<float>> scale_projections;

    /// Projected data per scale
    std::vector<std::vector<float>> scale_data;

    /// Output dimensions per scale
    std::vector<int> scale_dims;

    /// Whether to normalize projections
    bool normalize_projections = true;

    /// Optional pluggable metric
    NeuroMetric* metric = nullptr;

    IndexNeuroGridCell() = default;

    /** Construct with inner IndexFlat.
     * @param inner      inner index (must be IndexFlat)
     * @param n_scales   number of scales (default 4)
     * @param own_inner  take ownership
     */
    IndexNeuroGridCell(Index* inner, int n_scales = 4, bool own_inner = false);

    /** Set custom scale factors.
     * @param factors  dimension reduction factor per scale
     */
    void set_scale_factors(const std::vector<float>& factors);

    /** Train: initialize projection matrices. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: project to all scales. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Project a vector to a specific scale
    void project_to_scale(int scale, const float* x, float* out) const;
};

} // namespace faiss
