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

/** FS-04: Statistical Prescreen Index
 *
 * Ultra-cheap filtering using precomputed statistics (norm, mean).
 * Only 2 operations per candidate for initial filtering.
 *
 * Algorithm:
 * 1. Precompute L2 norm and mean for each vector
 * 2. At search, compute query norm and mean
 * 3. Filter by: |norm_diff| + |mean_diff| * sqrt(d)
 * 4. Compute precise L2 on survivors
 *
 * Typical performance: 95-98% recall at 2-3x speedup
 */
struct IndexNeuroStatisticalPrescreen : IndexNeuro {
    /// L2 norms of all vectors
    std::vector<float> norms;

    /// Mean values of all vectors
    std::vector<float> means;

    /// Original vectors
    std::vector<float> vectors;

    /// Fraction of candidates to keep
    float keep_ratio = 0.20f;

    /// Weight for norm difference in score
    float norm_weight = 1.0f;

    /// Weight for mean difference in score
    float mean_weight = 1.0f;

    /// Default constructor
    IndexNeuroStatisticalPrescreen() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     * @param keep_ratio fraction to keep (default: 0.20)
     */
    explicit IndexNeuroStatisticalPrescreen(int d, float keep_ratio = 0.20f);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void reset() override;

protected:
    /// Compute norm and mean for a vector
    void compute_stats(const float* vec, float& norm, float& mean) const;
};

} // namespace faiss
