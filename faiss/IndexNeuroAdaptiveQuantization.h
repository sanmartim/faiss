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

/** QT-04: Adaptive Quantization with Hot/Cold Region Precision.
 *
 * Implements variable precision based on data density:
 * hot regions (frequently accessed) keep float32,
 * cold regions use aggressive quantization.
 *
 * Key features:
 *   - Density-based region detection
 *   - Hot regions: float32, Cold regions: 4-bit
 *   - >= 99% recall on hot queries
 */
struct IndexNeuroAdaptiveQuantization : IndexNeuro {
    /// Number of regions
    int nregions = 16;

    /// Fraction of regions considered "hot"
    float hot_ratio = 0.2f;

    /// Region centroids
    std::vector<float> centroids;

    /// Whether each region is hot
    std::vector<bool> is_hot;

    /// Query counts per region (for adaptive learning)
    mutable std::vector<int64_t> query_counts;

    /// Hot region vectors (full precision)
    std::vector<std::vector<float>> hot_vectors;

    /// Hot region IDs
    std::vector<std::vector<idx_t>> hot_ids;

    /// Cold region codes (4-bit quantized)
    std::vector<std::vector<uint8_t>> cold_codes;

    /// Cold region IDs
    std::vector<std::vector<idx_t>> cold_ids;

    /// Original vectors (for cold region reranking)
    std::vector<float> orig_vectors;

    /// Number of rerank candidates for cold regions
    int rerank_k = 100;

    IndexNeuroAdaptiveQuantization() = default;

    /** Construct with dimension and regions.
     * @param d          dimension
     * @param nregions   number of regions
     * @param hot_ratio  fraction of hot regions
     */
    IndexNeuroAdaptiveQuantization(int d, int nregions = 16, float hot_ratio = 0.2f);

    ~IndexNeuroAdaptiveQuantization() override = default;

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

    /// Update hot/cold assignments based on query patterns
    void update_hot_regions();

private:
    /// Find region for a vector
    int find_region(const float* vec) const;

    /// Encode vector to 4-bit
    void encode_4bit(const float* vec, uint8_t* code) const;

    /// Get code size for 4-bit encoding
    size_t code_size_4bit() const { return (d + 1) / 2; }
};

/// Parameters for adaptive quantization search
struct NeuroAdaptiveQuantParams : NeuroSearchParameters {
    int rerank_k = -1;

    ~NeuroAdaptiveQuantParams() override = default;
};

} // namespace faiss
