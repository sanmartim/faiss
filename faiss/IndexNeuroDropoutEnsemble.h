/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>

namespace faiss {

/** Multi-view Dropout Ensemble Search Index (ED-05)
 *
 * Wraps an IndexFlat and performs search by creating multiple "views"
 * of the data, each with a different subset of dimensions (dropout mask).
 * Results from all views are integrated using a chosen method.
 *
 * Dropout modes:
 *   RANDOM: Independent random masks per view
 *   COMPLEMENTARY: Minimize overlap, guarantee full dimension coverage
 *   STRUCTURED: Semantic grouping (halves, thirds, etc.)
 *   ADVERSARIAL: Each view excludes previous top-contributing columns
 *
 * Integration methods:
 *   VOTING: Count appearances across views
 *   BORDA: Sum of ranks (lower = better)
 *   MEAN_DIST: Average distances from views
 *   FULL_RERANK: Full distance on union of candidates
 */
struct IndexNeuroDropoutEnsemble : IndexNeuro {
    /// Number of parallel views
    int num_views = 5;

    /// Fraction of dimensions to drop per view
    float dropout_rate = 0.3f;

    /// How to generate dropout masks
    NeuroDropoutMode dropout_mode = NEURO_DROPOUT_COMPLEMENTARY;

    /// How to combine results from views
    NeuroIntegrationMethod integration = NEURO_INTEGRATE_BORDA;

    /// Candidates per view; 0 = auto (k * 2)
    int top_k_per_view = 0;

    IndexNeuroDropoutEnsemble() = default;

    /** Construct wrapping an existing flat index.
     * @param inner       the flat index holding the data
     * @param own_inner   if true, deletes inner in destructor
     */
    IndexNeuroDropoutEnsemble(Index* inner, bool own_inner = false);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void train(idx_t n, const float* x) override;
};

} // namespace faiss
