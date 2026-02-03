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

/** FS-03: Projection Cascade Index
 *
 * Filter through progressively higher-dimension random projections.
 * Each stage keeps top keep_ratio by projected distance.
 *
 * Algorithm:
 * 1. Generate random projection matrices for each level
 * 2. Project all data and queries to each dimension level
 * 3. Filter progressively: d=8 -> d=32 -> d=128 -> full L2
 * 4. Each stage keeps top keep_ratio candidates
 *
 * Typical performance: 93-97% recall at 5-10x speedup
 */
struct IndexNeuroProjectionCascade : IndexNeuro {
    /// Target dimensions for cascade
    std::vector<int> projection_dims = {8, 32, 128};

    /// Keep ratio per level
    std::vector<float> keep_ratios = {0.10f, 0.10f, 0.10f};

    /// Random projection matrices: matrices[level] is d x projection_dims[level]
    std::vector<std::vector<float>> projection_matrices;

    /// Projected data: projected[level] is ntotal x projection_dims[level]
    std::vector<std::vector<float>> projected_data;

    /// Original vectors
    std::vector<float> vectors;

    /// Random seed for reproducibility
    int random_seed = 42;

    /// Default constructor
    IndexNeuroProjectionCascade() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     */
    explicit IndexNeuroProjectionCascade(int d);

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
    /// Initialize random projection matrices
    void init_projection_matrices();

    /// Project a vector to a given level
    void project(const float* vec, int level, float* out) const;
};

} // namespace faiss
