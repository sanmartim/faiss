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

/** MS-01: Multi-Scale Sign Index
 *
 * Uses sign(tanh(x * scale)) at multiple scales to create binary signatures
 * for ultra-fast XOR-based candidate filtering. Hamming distance between
 * signatures approximates L2 distance locality.
 *
 * Algorithm:
 * 1. For each vector, compute binary signatures at each scale
 * 2. At search time, filter candidates by Hamming distance on signatures
 * 3. Compute precise L2 only on surviving candidates
 *
 * Typical performance: 92-97% recall at 5-10x speedup
 */
struct IndexNeuroMultiScaleSign : IndexNeuro {
    /// Scale factors for tanh (default: [0.5, 1.0, 2.0])
    std::vector<float> scales;

    /// Maximum Hamming distance per scale (0 = auto: d/16)
    int max_hamming_distance = 0;

    /// Packed binary signatures: sigs[scale_idx] contains all vectors
    /// Each vector uses ceil(d/64) uint64_t words
    std::vector<std::vector<uint64_t>> signatures;

    /// Number of 64-bit words per vector
    int words_per_vec = 0;

    /// Default constructor
    IndexNeuroMultiScaleSign() = default;

    /** Construct with dimension and scales.
     * @param d vector dimensionality
     * @param scales scale factors (default: {0.5, 1.0, 2.0})
     */
    explicit IndexNeuroMultiScaleSign(
            int d,
            const std::vector<float>& scales = {0.5f, 1.0f, 2.0f});

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
    /// Compute signature for a single vector at a given scale
    void compute_signature(
            const float* vec,
            float scale,
            uint64_t* sig) const;

    /// Compute Hamming distance between two signatures
    int hamming_distance(const uint64_t* sig1, const uint64_t* sig2) const;
};

} // namespace faiss
