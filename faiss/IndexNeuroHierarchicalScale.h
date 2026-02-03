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

/** MS-03: Hierarchical Scale Index
 *
 * Implements hierarchical cascade filtering from coarse (scale=0.25)
 * to fine (scale=4.0), with each level halving candidates.
 *
 * Algorithm:
 * 1. Start with all candidates
 * 2. At each cascade level, filter by Hamming distance
 * 3. Keep only top keep_ratio candidates for next level
 * 4. Early terminate when target_candidates reached
 * 5. Compute precise L2 on survivors
 *
 * Typical performance: 90-95% recall at 10-30x speedup
 */
struct IndexNeuroHierarchicalScale : IndexNeuro {
    /// Cascade scales from coarse to fine
    std::vector<float> cascade_scales = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};

    /// Keep ratio per level (fraction to keep)
    std::vector<float> keep_ratios = {0.5f, 0.5f, 0.5f, 0.5f};

    /// Target number of candidates (0 = use all levels)
    int target_candidates = 0;

    /// Packed binary signatures per scale level
    std::vector<std::vector<uint64_t>> signatures;

    /// Number of 64-bit words per vector
    int words_per_vec = 0;

    /// Default constructor
    IndexNeuroHierarchicalScale() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     */
    explicit IndexNeuroHierarchicalScale(int d);

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
    /// Compute signature for a single vector at given scale
    void compute_signature(
            const float* vec,
            float scale,
            uint64_t* sig) const;

    /// Compute Hamming distance
    int hamming_distance(const uint64_t* sig1, const uint64_t* sig2) const;
};

} // namespace faiss
