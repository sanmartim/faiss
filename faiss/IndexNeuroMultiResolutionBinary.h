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

/** BZ-04: Multi-Resolution Binary with Cascading Bit Levels.
 *
 * Implements cascading binary search (1-bit → 2-bit → 4-bit → 8-bit)
 * for progressive filtering with high speedup.
 *
 * Key features:
 *   - 4-level cascade (1, 2, 4, 8 bits)
 *   - Configurable keep ratios per level
 *   - >= 97% recall with 50x speedup
 */
struct IndexNeuroMultiResolutionBinary : IndexNeuro {
    /// Number of resolution levels
    int num_levels = 4;

    /// Bits per level: [1, 2, 4, 8]
    std::vector<int> level_bits;

    /// Keep ratio per level (fraction to pass to next level)
    std::vector<float> keep_ratios;

    /// Encoded vectors per level
    std::vector<std::vector<uint8_t>> level_codes;

    /// Original vectors for final reranking
    std::vector<float> orig_vectors;

    /// Number of final rerank candidates
    int rerank_k = 100;

    /// Whether to do final full-precision rerank
    bool do_rerank = true;

    IndexNeuroMultiResolutionBinary() = default;

    /** Construct with dimension and rerank candidates.
     * @param d         dimension
     * @param rerank_k  final rerank candidates
     */
    explicit IndexNeuroMultiResolutionBinary(int d, int rerank_k = 100);

    ~IndexNeuroMultiResolutionBinary() override = default;

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

    /// Get compression ratio at each level
    std::vector<float> get_compression_ratios() const;

private:
    /// Encode vector at specific bit resolution
    void encode_at_level(const float* vec, int level, uint8_t* code) const;

    /// Compute Hamming distance for codes at a level
    int hamming_distance(const uint8_t* a, const uint8_t* b, int level) const;

    /// Get code size in bytes for a level
    size_t code_size_at_level(int level) const;
};

/// Parameters for multi-resolution binary search
struct NeuroMultiResBinaryParams : NeuroSearchParameters {
    int rerank_k = -1;

    ~NeuroMultiResBinaryParams() override = default;
};

} // namespace faiss
