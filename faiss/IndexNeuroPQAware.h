/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexPQ.h>
#include <faiss/impl/NeuroDistance.h>
#include <faiss/impl/ProductQuantizer.h>
#include <memory>
#include <vector>

namespace faiss {

/** MT-04: PQ-Aware Neuro Strategy.
 *
 * Applies bio-inspired strategies to product-quantized codes.
 * Combines fast PQ distance approximation with neuro-style
 * candidate selection and reranking.
 *
 * Key features:
 *   - Works with existing PQ codes (32x+ compression)
 *   - Asymmetric distance computation on codes
 *   - Neuro-style candidate filtering before rerank
 *   - Optional rerank with reconstructed vectors
 *
 * The strategy applies elimination/weighting at the code level:
 *   1. Compute ADC (asymmetric distance computation) distances
 *   2. Apply neuro-style filtering (dispersion, entropy, etc.)
 *   3. Rerank top candidates with true distance
 */
struct IndexNeuroPQAware : IndexNeuro {
    /// Product quantizer (owned or borrowed)
    ProductQuantizer* pq = nullptr;
    bool own_pq = false;

    /// PQ codes storage: ntotal * pq->code_size
    std::vector<uint8_t> codes;

    /// Original vectors for reranking (optional)
    std::vector<float> orig_vectors;
    bool store_original = false;

    /// Neuro filtering mode
    enum FilterMode {
        PQ_FILTER_NONE = 0,      ///< No filtering, just PQ distance
        PQ_FILTER_DISPERSION,    ///< Skip uniform-distance candidates
        PQ_FILTER_ENTROPY,       ///< Weight by subquantizer entropy
        PQ_FILTER_WEIGHTED,      ///< Learned subquantizer weights
    };
    FilterMode filter_mode = PQ_FILTER_DISPERSION;

    /// Number of candidates before rerank
    int n_candidates = 100;

    /// Whether to rerank with true distance
    bool rerank = true;

    /// Dispersion threshold for filtering
    float dispersion_threshold = 0.1f;

    /// Per-subquantizer weights (for WEIGHTED mode)
    std::vector<float> subq_weights;

    /// Optional pluggable metric for reranking
    NeuroMetric* metric = nullptr;

    IndexNeuroPQAware() = default;

    /** Construct with PQ parameters.
     * @param d          dimension
     * @param M          number of subquantizers
     * @param nbits      bits per subquantizer (typically 8)
     */
    IndexNeuroPQAware(int d, int M, int nbits = 8);

    /** Construct wrapping existing PQ.
     * @param pq_in      product quantizer to use
     * @param own_pq     take ownership of pq
     */
    IndexNeuroPQAware(ProductQuantizer* pq_in, bool own_pq = false);

    ~IndexNeuroPQAware() override;

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

    /// Reconstruct vector from code
    void reconstruct(idx_t key, float* recons) const override;

    /// Learn subquantizer weights from data
    void learn_weights(idx_t n, const float* x);

    /// Compute entropy of a subquantizer's assignments
    float compute_subq_entropy(int subq_idx) const;

    /// Get code for a specific vector
    const uint8_t* get_code(idx_t i) const;

private:
    /// Compute ADC distance between query and code
    float compute_adc_distance(
            const float* query,
            const uint8_t* code,
            const float* dis_table) const;

    /// Build distance table for a query
    void compute_distance_table(const float* x, float* dis_table) const;

    /// Apply neuro filtering to candidates
    void filter_candidates(
            const float* dis_table,
            const std::vector<std::pair<float, idx_t>>& candidates,
            std::vector<std::pair<float, idx_t>>& filtered) const;
};

/// Parameters for PQ-aware search
struct NeuroPQAwareParams : NeuroSearchParameters {
    int candidates = -1;           ///< -1 = use index default
    bool rerank = true;            ///< whether to rerank

    ~NeuroPQAwareParams() override = default;
};

} // namespace faiss
