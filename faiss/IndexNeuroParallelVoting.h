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

/// How to partition dimensions into groups
enum NeuroGroupingMethod {
    NEURO_GROUP_CONSECUTIVE = 0, ///< dims 0..g-1, g..2g-1, etc.
    NEURO_GROUP_INTERLEAVED,     ///< dim i -> group (i % num_groups)
};

/// Parameters for parallel voting search
struct NeuroParallelVotingParams : NeuroSearchParameters {
    int num_groups = 0;            ///< 0 = use index default
    int top_k_per_group = 0;       ///< 0 = auto (k * 3)
    NeuroGroupingMethod grouping = NEURO_GROUP_CONSECUTIVE;
    NeuroIntegrationMethod integration = NEURO_INTEGRATE_BORDA;

    ~NeuroParallelVotingParams() override = default;
};

/** PP-01: Group-wise parallel search with vote integration.
 *
 * Divides d dimensions into num_groups groups. Each group independently
 * computes partial L2 distances and selects top-k candidates. Results
 * are combined via voting, Borda count, mean distance, or full rerank.
 *
 * This provides dimensionality-parallel search: groups can identify
 * candidates using different subsets of the feature space, then
 * consensus determines the final results.
 */
struct IndexNeuroParallelVoting : IndexNeuro {
    int num_groups = 4;
    int top_k_per_group = 0; ///< 0 = auto (k * 3)
    NeuroGroupingMethod grouping = NEURO_GROUP_CONSECUTIVE;
    NeuroIntegrationMethod integration = NEURO_INTEGRATE_BORDA;

    IndexNeuroParallelVoting() = default;

    explicit IndexNeuroParallelVoting(
            Index* inner,
            int num_groups = 4,
            bool own_inner = false);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss
