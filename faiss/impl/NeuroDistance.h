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

/// Parameters for dropout ensemble (ED-05)
struct NeuroDropoutParams : NeuroSearchParameters {
    int num_views = 5;
    float dropout_rate = 0.3f;
    NeuroDropoutMode dropout_mode = NEURO_DROPOUT_COMPLEMENTARY;
    NeuroIntegrationMethod integration = NEURO_INTEGRATE_BORDA;
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
