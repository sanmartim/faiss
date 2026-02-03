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

/** BZ-03: Learned Binarization with Optimized Thresholds.
 *
 * Learns optimal per-dimension thresholds from similarity pairs
 * to maximize binary code quality.
 *
 * Key features:
 *   - Threshold optimization from training data
 *   - +5-8% recall vs median thresholds
 *   - Fast binary search with Hamming distance
 */
struct IndexNeuroLearnedBinarization : IndexNeuro {
    /// Learned thresholds per dimension
    std::vector<float> thresholds;

    /// Binary codes
    std::vector<uint8_t> codes;

    /// Original vectors for reranking
    std::vector<float> orig_vectors;

    /// Number of rerank candidates
    int rerank_k = 100;

    /// Whether to do full-precision rerank
    bool do_rerank = true;

    /// Learning rate for threshold optimization
    float learning_rate = 0.01f;

    /// Number of optimization iterations
    int n_iter = 100;

    IndexNeuroLearnedBinarization() = default;

    /** Construct with dimension.
     * @param d         dimension
     * @param rerank_k  rerank candidates
     */
    explicit IndexNeuroLearnedBinarization(int d, int rerank_k = 100);

    ~IndexNeuroLearnedBinarization() override = default;

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

private:
    /// Encode a vector using learned thresholds
    void encode(const float* vec, uint8_t* code) const;

    /// Get binary code size in bytes
    size_t code_size() const { return (d + 7) / 8; }

    /// Compute Hamming distance
    int hamming_distance(const uint8_t* a, const uint8_t* b) const;

    /// Learn thresholds from similarity pairs
    void learn_thresholds(idx_t n, const float* x);
};

/// Parameters for learned binarization search
struct NeuroLearnedBinaryParams : NeuroSearchParameters {
    int rerank_k = -1;

    ~NeuroLearnedBinaryParams() override = default;
};

} // namespace faiss
