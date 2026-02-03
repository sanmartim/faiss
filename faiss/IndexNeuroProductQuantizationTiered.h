/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <faiss/impl/ProductQuantizer.h>
#include <vector>

namespace faiss {

/// Tier configuration for QT-02 (moved outside struct for SWIG)
struct NeuroPQTierConfig {
    int M = 8;              ///< Number of subquantizers
    int nbits = 4;          ///< Bits per subquantizer
    float keep_ratio = 0.1f; ///< Fraction of candidates to keep
};

/** QT-02: Tiered Product Quantization with Cascade Filtering.
 *
 * Implements cascading PQ tiers (e.g., 4-bit -> 8-bit -> float) for
 * aggressive compression with high recall through progressive refinement.
 *
 * Key features:
 *   - 2-3 tier cascade with different precision levels
 *   - Each tier filters candidates for the next
 *   - Final reranking with full precision
 *   - >=99% recall with 10x throughput improvement
 */
struct IndexNeuroProductQuantizationTiered : IndexNeuro {
    std::vector<NeuroPQTierConfig> tier_configs;

    /// Product quantizers for each tier
    std::vector<ProductQuantizer*> pqs;

    /// Codes storage per tier
    std::vector<std::vector<uint8_t>> codes_per_tier;

    /// Original vectors for final reranking
    std::vector<float> orig_vectors;

    /// Number of final rerank candidates
    int rerank_k = 100;

    /// Whether to do full-precision reranking
    bool do_rerank = true;

    /// Optional metric for reranking
    NeuroMetric* metric = nullptr;

    IndexNeuroProductQuantizationTiered() = default;

    /** Construct with default 2-tier configuration.
     * @param d         dimension
     * @param M         subquantizers per tier
     * @param rerank_k  final rerank candidates
     */
    IndexNeuroProductQuantizationTiered(int d, int M = 8, int rerank_k = 100);

    /** Construct with custom tier configuration.
     * @param d         dimension
     * @param configs   tier configurations
     * @param rerank_k  final rerank candidates
     */
    IndexNeuroProductQuantizationTiered(
            int d,
            const std::vector<NeuroPQTierConfig>& configs,
            int rerank_k = 100);

    ~IndexNeuroProductQuantizationTiered() override;

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

    /// Get compression ratio
    float get_compression_ratio() const;

    /// Get number of tiers
    int get_num_tiers() const { return static_cast<int>(pqs.size()); }
};

/// Parameters for tiered PQ search
struct NeuroPQTieredParams : NeuroSearchParameters {
    int rerank_k = -1;
    bool do_rerank = true;

    ~NeuroPQTieredParams() override = default;
};

} // namespace faiss
