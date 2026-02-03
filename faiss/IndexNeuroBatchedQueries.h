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

/** SY-04: Batched Query Processing with Matrix Operations.
 *
 * Processes multiple queries together using matrix multiplication
 * for distance computation: D = ||Q||^2 + ||X||^2 - 2*Q*X^T
 *
 * Key features:
 *   - 5-10x throughput improvement for batch queries
 *   - Uses BLAS for matrix operations
 *   - Optimal batch size selection
 */
struct IndexNeuroBatchedQueries : IndexNeuro {
    /// Inner index to wrap
    Index* sub_index = nullptr;

    /// Optimal batch size
    int batch_size = 64;

    /// Precomputed database norms (||x||^2)
    std::vector<float> xb_norms;

    /// Whether norms are precomputed
    bool norms_computed = false;

    IndexNeuroBatchedQueries() = default;

    /** Construct wrapping an existing index.
     * @param sub_index   the index to wrap
     * @param batch_size  query batch size
     */
    IndexNeuroBatchedQueries(Index* sub_index, int batch_size = 64);

    ~IndexNeuroBatchedQueries() override = default;

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /// Precompute database norms
    void precompute_norms();

private:
    /// Get raw vectors
    const float* get_xb() const;
};

/// Parameters for batched queries
struct NeuroBatchedParams : NeuroSearchParameters {
    int batch_size = -1;

    ~NeuroBatchedParams() override = default;
};

} // namespace faiss
