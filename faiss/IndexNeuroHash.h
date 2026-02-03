/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace faiss {

/// Parameters for hash-based search, overridable per query
struct NeuroHashParams : NeuroSearchParameters {
    int hamming_threshold = -1;  ///< -1 = use index default
    bool rerank = true;          ///< whether to rerank candidates with true distance

    ~NeuroHashParams() override = default;
};

/** HS-01: SimHash locality-sensitive hashing.
 *
 * Projects vectors to binary codes via random hyperplanes:
 *   h_i(x) = sign(r_i · x)
 * where r_i is a random unit vector.
 *
 * Search finds candidates within Hamming threshold, then
 * optionally reranks with true L2 distance.
 *
 * Multiple hash tables increase recall at cost of memory.
 */
struct IndexNeuroHash : IndexNeuro {
    /// Number of bits per hash code
    int n_bits = 64;

    /// Number of independent hash tables
    int n_tables = 4;

    /// Hamming distance threshold for candidate retrieval
    int hamming_threshold = 8;

    /// Whether to rerank candidates with true distance
    bool rerank = true;

    /// Optional pluggable metric for reranking (nullptr = L2)
    NeuroMetric* metric = nullptr;

    /// Random hyperplanes: n_tables * n_bits * d floats
    std::vector<float> hyperplanes;

    /// Hash codes for all vectors: ntotal * n_tables * (n_bits/64) uint64
    std::vector<uint64_t> codes;

    /// Hash tables: bucket -> list of vector indices
    /// One table per hash function
    std::vector<std::unordered_map<uint64_t, std::vector<idx_t>>> tables;

    IndexNeuroHash() = default;

    /** Construct with inner IndexFlat.
     * @param inner      inner index (must be IndexFlat)
     * @param n_bits     bits per hash (default 64)
     * @param n_tables   number of hash tables (default 4)
     * @param own_inner  take ownership of inner
     */
    IndexNeuroHash(
            Index* inner,
            int n_bits = 64,
            int n_tables = 4,
            bool own_inner = false);

    /** Train: generate random hyperplanes.
     * Does not use training data - hyperplanes are random.
     */
    void train(idx_t n, const float* x) override;

    /** Add vectors: compute and store hash codes.
     * Must be called after train().
     */
    void add(idx_t n, const float* x) override;

    /** Reset all stored codes and tables. */
    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Compute hash code for a single vector
    void compute_code(const float* x, uint64_t* code) const;

    /// Get number of uint64 words per code
    int code_words() const { return (n_bits + 63) / 64; }
};

/** HS-04: Hierarchical hash cascade.
 *
 * Multiple hash levels with increasing precision:
 *   Level 0: coarse (8 bits) - fast pruning
 *   Level 1: medium (32 bits) - moderate filtering
 *   Level 2: fine (128 bits) - precise candidates
 *
 * Early termination when candidates drop below threshold.
 */
struct IndexNeuroHierarchicalHash : IndexNeuro {
    /// Bits per level (default: 8, 32, 128)
    std::vector<int> bits_per_level;

    /// Hamming threshold per level (default: 2, 4, 8)
    std::vector<int> thresholds;

    /// Minimum candidates to continue (early termination)
    int min_candidates = 100;

    /// Whether to rerank final candidates
    bool rerank = true;

    /// Optional pluggable metric for reranking
    NeuroMetric* metric = nullptr;

    /// Hyperplanes per level: bits_per_level[l] * d floats each
    std::vector<std::vector<float>> level_hyperplanes;

    /// Codes per level: ntotal * code_words codes each
    std::vector<std::vector<uint64_t>> level_codes;

    IndexNeuroHierarchicalHash() = default;

    /** Construct with default 3 levels (8, 32, 128 bits).
     * @param inner      inner index (must be IndexFlat)
     * @param own_inner  take ownership
     */
    IndexNeuroHierarchicalHash(Index* inner, bool own_inner = false);

    /** Construct with custom levels.
     * @param inner      inner index
     * @param bits       bits per level
     * @param thresholds hamming threshold per level
     * @param own_inner  take ownership
     */
    IndexNeuroHierarchicalHash(
            Index* inner,
            const std::vector<int>& bits,
            const std::vector<int>& thresholds,
            bool own_inner = false);

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

    /// Compute code at a specific level
    void compute_level_code(int level, const float* x, uint64_t* code) const;
};

} // namespace faiss
