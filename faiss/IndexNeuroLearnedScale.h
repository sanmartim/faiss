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

/** MS-05: Learned Scale Index
 *
 * Learns optimal scales from validation set via random search to achieve
 * 2-5% better recall than default scales.
 *
 * Algorithm:
 * 1. During training, sample random scale combinations
 * 2. Evaluate recall on validation queries
 * 3. Keep best performing scales
 * 4. Use learned scales for search
 *
 * Typical performance: 95-98% recall at 5-12x speedup
 */
struct IndexNeuroLearnedScale : IndexNeuro {
    /// Learned optimal scales
    std::vector<float> learned_scales;

    /// Number of scales to learn
    int num_scales = 5;

    /// Number of random search iterations
    int search_iterations = 100;

    /// Number of validation queries
    int validation_queries = 1000;

    /// k for validation
    int validation_k = 10;

    /// Packed signatures per scale
    std::vector<std::vector<uint64_t>> signatures;

    /// Number of 64-bit words per vector
    int words_per_vec = 0;

    /// Max Hamming distance per scale
    int max_hamming_distance = 0;

    /// Random seed for reproducibility
    int random_seed = 42;

    /// Default constructor
    IndexNeuroLearnedScale() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     * @param num_scales number of scales to learn (default: 5)
     */
    explicit IndexNeuroLearnedScale(int d, int num_scales = 5);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void reset() override;

protected:
    /// Compute signature at given scale
    void compute_signature(
            const float* vec,
            float scale,
            uint64_t* sig) const;

    /// Compute Hamming distance
    int hamming_distance(const uint64_t* sig1, const uint64_t* sig2) const;

    /// Evaluate recall with given scales on validation data
    float evaluate_recall(
            const float* data,
            idx_t n,
            const float* queries,
            idx_t nq,
            idx_t k,
            const std::vector<float>& scales) const;
};

} // namespace faiss
