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

/** FS-05: Ensemble Voting Index
 *
 * Combines multiple fast filters via voting. Candidates passing
 * multiple filters are prioritized.
 *
 * Algorithm:
 * 1. Run multiple independent fast filters (Hamming, Stats, Projection)
 * 2. Count votes per candidate
 * 3. Keep candidates with >= min_votes
 * 4. Compute precise L2 on survivors
 *
 * Typical performance: 98-99% recall at 2-4x speedup
 */
struct IndexNeuroEnsembleVoting : IndexNeuro {
    /// Minimum votes required
    int min_votes = 2;

    /// Whether to use Hamming filter
    bool use_hamming = true;

    /// Whether to use statistical filter
    bool use_stats = true;

    /// Whether to use projection filter
    bool use_projection = true;

    /// Keep ratio for each filter
    float filter_keep_ratio = 0.15f;

    // Hamming filter data
    std::vector<float> hamming_thresholds;
    std::vector<uint64_t> hamming_codes;
    int hamming_words_per_vec = 0;

    // Statistical filter data
    std::vector<float> stat_norms;
    std::vector<float> stat_means;

    // Projection filter data
    std::vector<float> projection_matrix;  // d x 32
    std::vector<float> projected_data;     // ntotal x 32
    int projection_dim = 32;

    /// Original vectors
    std::vector<float> vectors;

    /// Random seed
    int random_seed = 42;

    /// Default constructor
    IndexNeuroEnsembleVoting() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     */
    explicit IndexNeuroEnsembleVoting(int d);

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
    /// Encode vector to Hamming code
    void encode_hamming(const float* vec, uint64_t* code) const;

    /// Compute Hamming distance
    int hamming_distance(const uint64_t* c1, const uint64_t* c2) const;

    /// Compute statistics for a vector
    void compute_stats(const float* vec, float& norm, float& mean) const;

    /// Project a vector
    void project(const float* vec, float* out) const;
};

} // namespace faiss
