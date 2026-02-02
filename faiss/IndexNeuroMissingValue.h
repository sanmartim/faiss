/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>

namespace faiss {

/** Missing-Value-Adjusted Search Index (PA-03)
 *
 * Wraps an IndexFlat and performs search with NaN-aware distance
 * computation. Dimensions where either query or database vector is
 * NaN are skipped, and distances are renormalized by the number of
 * present dimensions with a configurable weight reduction.
 *
 * Bio-inspiration: "Epistemic confidence" - the brain knows when it
 * doesn't know. Uncertain information weighs less in the decision.
 *
 * Missing strategies:
 *   PROPORTIONAL: weight = (1 - missing_rate)
 *   THRESHOLD: ignore column entirely if missing_rate > threshold
 *   HYBRID: weight = (1 - missing_rate)^2 (default)
 */
struct IndexNeuroMissingValue : IndexNeuro {
    /// How to adjust distance for missing values
    NeuroMissingStrategy missing_strategy = NEURO_MISSING_HYBRID;

    /// Columns with missing rate above this are ignored (THRESHOLD mode)
    float ignore_threshold = 0.8f;

    IndexNeuroMissingValue() = default;

    /** Construct wrapping an existing flat index.
     * @param inner       the flat index holding the data
     * @param strategy    missing value strategy
     * @param own_inner   if true, deletes inner in destructor
     */
    IndexNeuroMissingValue(
            Index* inner,
            NeuroMissingStrategy strategy = NEURO_MISSING_HYBRID,
            bool own_inner = false);

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
