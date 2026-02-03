/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <cstdint>
#include <vector>

namespace faiss {

/// Parameters for column hash search, overridable per query
struct NeuroColumnHashParams : NeuroSearchParameters {
    int min_groups_match = -1;  ///< -1 = use index default
    bool rerank = true;         ///< whether to rerank with true distance

    ~NeuroColumnHashParams() override = default;
};

/** HS-05: Column-grouped hashing.
 *
 * Divides dimensions into semantic groups and computes separate
 * hashes per group. Candidates must match on at least min_groups_match
 * groups to be considered.
 *
 * Use cases:
 *   - Semantic groups (e.g., color features, shape features)
 *   - Multi-modal embeddings (text region, image region)
 *   - Hierarchical feature importance
 *
 * Each group can have an associated weight for scoring.
 */
struct IndexNeuroColumnHash : IndexNeuro {
    /// Column groups: groups[g] is the list of column indices for group g
    std::vector<std::vector<int>> groups;

    /// Optional weight per group (empty = uniform)
    std::vector<float> group_weights;

    /// Bits per group hash
    int bits_per_group = 16;

    /// Hamming threshold per group to consider a match
    int group_hamming_threshold = 3;

    /// Minimum number of groups that must match
    int min_groups_match = 2;

    /// Whether to rerank candidates with true distance
    bool rerank = true;

    /// Optional pluggable metric for reranking
    NeuroMetric* metric = nullptr;

    /// Hyperplanes per group: groups.size() * bits_per_group * group_d
    std::vector<std::vector<float>> group_hyperplanes;

    /// Hash codes per group per vector
    /// Layout: codes[g][i] = code for vector i in group g
    std::vector<std::vector<uint64_t>> group_codes;

    IndexNeuroColumnHash() = default;

    /** Construct with inner IndexFlat.
     * @param inner      inner index (must be IndexFlat)
     * @param own_inner  take ownership
     */
    IndexNeuroColumnHash(Index* inner, bool own_inner = false);

    /** Set groups manually.
     * @param groups  list of column index lists
     */
    void set_groups(const std::vector<std::vector<int>>& groups);

    /** Set groups to equal-sized splits.
     * @param n_groups  number of groups to create
     */
    void set_equal_groups(int n_groups);

    /** Train: generate per-group hyperplanes. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: compute and store per-group codes. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Compute hash code for one group
    void compute_group_code(int group, const float* x, uint64_t& code) const;
};

} // namespace faiss
