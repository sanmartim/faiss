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

/** FS-01: Hamming Prefilter Index
 *
 * Binary prefilter using per-dimension median thresholds.
 * Ultra-fast filtering with ~100 XOR+popcount operations per vector.
 *
 * Algorithm:
 * 1. Learn per-dimension median thresholds during training
 * 2. Binarize vectors: bit[j] = (vec[j] > threshold[j])
 * 3. At search, filter by Hamming distance keeping top keep_ratio
 * 4. Compute precise L2 only on survivors
 *
 * Typical performance: 97-99% recall at 3-5x speedup
 */
struct IndexNeuroHammingPrefilter : IndexNeuro {
    /// Per-dimension threshold (typically median)
    std::vector<float> thresholds;

    /// Packed binary codes for all vectors
    std::vector<uint64_t> binary_codes;

    /// Number of 64-bit words per vector
    int words_per_vec = 0;

    /// Fraction of candidates to keep after Hamming filter
    float keep_ratio = 0.10f;

    /// Default constructor
    IndexNeuroHammingPrefilter() = default;

    /** Construct with dimension and keep ratio.
     * @param d vector dimensionality
     * @param keep_ratio fraction to keep (default: 0.10)
     */
    explicit IndexNeuroHammingPrefilter(int d, float keep_ratio = 0.10f);

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
    /// Encode a vector to binary code
    void encode(const float* vec, uint64_t* code) const;

    /// Compute Hamming distance between two codes
    int hamming_distance(const uint64_t* code1, const uint64_t* code2) const;
};

} // namespace faiss
