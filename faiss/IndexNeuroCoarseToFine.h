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

/// Parameters for coarse-to-fine search
struct NeuroCoarseToFineParams : NeuroSearchParameters {
    int num_levels = 0;                    ///< 0 = use index default
    std::vector<float> cutoff_per_level;   ///< empty = use index defaults

    ~NeuroCoarseToFineParams() override = default;
};

/** PP-02: Multi-resolution progressive refinement.
 *
 * Precomputes coarse representations by averaging groups of dimensions:
 *   Level 0 (coarsest): groups of d/4 dims averaged to 4 values
 *   Level 1 (medium):   groups of d/8 dims averaged to 8 values
 *   Level 2 (finest):   full d dimensions
 *
 * Search: Level 0 eliminates 70% -> Level 1 eliminates 50% -> Level 2
 * ranks remaining candidates.
 *
 * Requires train() to precompute coarse representations of the database.
 */
struct IndexNeuroCoarseToFine : IndexNeuro {
    int num_levels = 3;  ///< number of resolution levels (including full)

    /// Fraction of candidates to keep at each level (except last = all)
    std::vector<float> cutoff_per_level;

    /// Precomputed coarse representations per level
    /// coarse_data[level] has size ntotal * coarse_dims[level]
    std::vector<std::vector<float>> coarse_data;
    std::vector<int> coarse_dims; ///< number of dims at each level

    IndexNeuroCoarseToFine() = default;

    explicit IndexNeuroCoarseToFine(
            Index* inner,
            int num_levels = 3,
            bool own_inner = false);

    /// Precompute coarse representations from the database
    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

private:
    /// Compute coarse representation of a vector at given level
    void compute_coarse(
            const float* vec,
            int level,
            std::vector<float>& out) const;
};

} // namespace faiss
