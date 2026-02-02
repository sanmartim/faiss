/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>

namespace faiss {

/** Progressive Elimination Search Index (ED-01 through ED-04)
 *
 * Wraps an IndexFlat and performs search by progressively
 * eliminating candidates one column at a time. Each round
 * computes single-dimension distances and discards a fraction
 * of candidates based on the cutoff.
 *
 * ED-01 (NEURO_FIXED): Fixed column order, fixed cutoff per round.
 * ED-02 (NEURO_ADAPTIVE_DISPERSION): Cutoff adapts based on column
 *        dispersion (std/mean of single-col distances).
 * ED-03 (NEURO_VARIANCE_ORDER): Columns ordered by sampled variance
 *        (highest variance first = most discriminative).
 * ED-04 (NEURO_UNCERTAINTY_DEFERRED): Defers elimination when column
 *        dispersion is low, accumulates before deciding.
 */
struct IndexNeuroElimination : IndexNeuro {
    NeuroEliminationStrategy strategy;

    /// Column visitation order; empty = reversed (d-1, d-2, ..., 0)
    std::vector<int> column_order;

    /// Fraction of candidates to keep each round (ED-01)
    float cutoff_percentile = 0.5f;

    /// Minimum surviving candidates; 0 = auto (k * 2)
    int min_candidates = 0;

    /// ED-02: dispersion thresholds
    float dispersion_low = 0.3f;
    float dispersion_high = 0.7f;
    float cutoff_low_dispersion = 0.8f;
    float cutoff_high_dispersion = 0.3f;

    /// ED-03: fraction of vectors to sample for variance computation
    float sample_fraction = 0.05f;

    /// ED-03: cached column order from train() (sorted by variance desc)
    std::vector<int> variance_column_order;

    /// ED-04: confidence threshold for deferring elimination
    float confidence_threshold = 0.4f;

    /// ED-04: max columns to accumulate before forced elimination
    int max_accumulated_columns = 3;

    IndexNeuroElimination() = default;

    /** Construct wrapping an existing flat index.
     * @param inner       the flat index holding the data
     * @param strategy    elimination strategy to use
     * @param own_inner   if true, deletes inner in destructor
     */
    IndexNeuroElimination(
            Index* inner,
            NeuroEliminationStrategy strategy = NEURO_FIXED,
            bool own_inner = false);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Train: for VARIANCE_ORDER, samples data to compute optimal
     * column ordering by variance (highest variance columns first).
     * For other strategies, delegates to inner index.
     */
    void train(idx_t n, const float* x) override;
};

} // namespace faiss
