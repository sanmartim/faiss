/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

namespace faiss {

/** Lateral Inhibition Decorator (MR-01)
 *
 * Wraps any Index and promotes diversity in search results by
 * suppressing candidates that are too similar to each other.
 * Inspired by lateral inhibition in the retina, where neighboring
 * neurons suppress each other's signals to increase contrast.
 *
 * Algorithm:
 *   1. Request k * k_expansion candidates from sub_index
 *   2. Greedy selection: iterate sorted candidates, group by
 *      pairwise similarity, keep at most max_per_cluster from
 *      each group
 *   3. Return top-k from the diverse set
 *
 * This is a decorator: it wraps ANY Index (IndexFlat,
 * IndexNeuroElimination, IndexNeuroDropoutEnsemble, etc.)
 * and can be composed with other decorators.
 */
struct IndexNeuroInhibition : Index {
    Index* sub_index = nullptr;    ///< the wrapped index
    bool own_fields = false;       ///< whether to delete sub_index in dtor

    /// L2 distance threshold: candidates closer than this are "similar"
    float similarity_threshold = 0.1f;

    /// Maximum candidates to keep from each similarity cluster
    int max_per_cluster = 3;

    /// Expansion factor: request k * k_expansion candidates from sub_index
    float k_expansion = 3.0f;

    IndexNeuroInhibition() = default;

    /** Construct wrapping an existing index.
     * @param sub_index  the index to wrap
     * @param own        if true, sub_index is deleted in destructor
     */
    IndexNeuroInhibition(Index* sub_index, bool own = false);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;

    ~IndexNeuroInhibition() override;
};

} // namespace faiss
