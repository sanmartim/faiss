/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <unordered_map>
#include <vector>

namespace faiss {

/** MS-04: Multi-Scale Intersection Index
 *
 * Strict intersection mode requiring exact signature match across
 * all scales for maximum speed (30-100x) when 80% recall is acceptable.
 *
 * Algorithm:
 * 1. Create bucket key by concatenating signatures across all scales
 * 2. At search time, find bucket with matching key
 * 3. If bucket empty, expand Hamming tolerance until candidates found
 * 4. Compute precise L2 on candidates
 *
 * Typical performance: 75-85% recall at 30-100x speedup
 */
struct IndexNeuroMultiScaleIntersection : IndexNeuro {
    /// Scale factors
    std::vector<float> scales = {0.5f, 1.0f, 2.0f};

    /// Number of 64-bit words per vector per scale
    int words_per_vec = 0;

    /// Packed signatures for all vectors at all scales
    std::vector<std::vector<uint64_t>> signatures;

    /// Bucket index: signature hash -> vector IDs
    std::unordered_map<uint64_t, std::vector<idx_t>> buckets;

    /// Fallback number of candidates when bucket is empty
    int fallback_k = 100;

    /// Default constructor
    IndexNeuroMultiScaleIntersection() = default;

    /** Construct with dimension.
     * @param d vector dimensionality
     * @param scales scale factors (default: {0.5, 1.0, 2.0})
     */
    explicit IndexNeuroMultiScaleIntersection(
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
    /// Compute signature for a vector at given scale
    void compute_signature(
            const float* vec,
            float scale,
            uint64_t* sig) const;

    /// Compute bucket key from all signatures (XOR fold)
    uint64_t compute_bucket_key(
            const std::vector<std::vector<uint64_t>>& sigs) const;

    /// Compute Hamming distance
    int hamming_distance(const uint64_t* sig1, const uint64_t* sig2) const;
};

} // namespace faiss
