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

/** SY-02: SIMD-Optimized Distance Computation Decorator.
 *
 * Wraps any Index and optimizes search by using SIMD-accelerated
 * batch distance computation functions from FAISS.
 *
 * Key features:
 *   - Decorator pattern (wraps any Index)
 *   - Uses fvec_L2sqr_ny / fvec_inner_products_ny for batch distance
 *   - Auto-batches queries for cache efficiency
 *   - 2-3x speedup for high-dimensional vectors (d >= 256)
 *
 * This strategy is most effective when:
 *   - The inner index uses brute-force search (IndexFlat)
 *   - Dimension is high (d >= 256)
 *   - Multiple queries are processed together
 */
struct IndexNeuroSIMDDistance : IndexNeuro {
    /// Inner index to wrap
    Index* sub_index = nullptr;

    /// Batch size for query processing
    int batch_size = 32;

    /// Whether to use inner product (vs L2)
    bool use_inner_product = false;

    /// Pre-computed L2 norms of database vectors (for inner product)
    std::vector<float> db_norms;

    /// Whether db_norms is computed
    bool norms_computed = false;

    IndexNeuroSIMDDistance() = default;

    /** Construct wrapping an existing index.
     * @param sub_index   the index to wrap
     * @param batch_size  query batch size for SIMD optimization
     */
    explicit IndexNeuroSIMDDistance(Index* sub_index, int batch_size = 32);

    ~IndexNeuroSIMDDistance() override = default;

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

    /// Precompute database vector norms for inner product
    void precompute_norms();

private:
    /// SIMD-accelerated brute-force search for a batch of queries
    void search_batch_simd(
            idx_t nq,
            const float* xq,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    /// Get raw vectors from sub_index
    const float* get_xb() const;
};

/// Parameters for SIMD distance search
struct NeuroSIMDParams : NeuroSearchParameters {
    int batch_size = -1;  ///< -1 = use index default

    ~NeuroSIMDParams() override = default;
};

} // namespace faiss
