/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <faiss/IndexFlat.h>
#include <vector>

namespace faiss {

/** PT-03: Semantic Sharding with Meaning-Based Distribution.
 *
 * Implements content-aware sharding where vectors with similar
 * semantics are grouped together, improving cache locality
 * and enabling shard-level pruning.
 *
 * Key features:
 *   - Semantic clustering for shard assignment
 *   - Shard-level early termination
 *   - Load-balanced semantic groups
 */
struct IndexNeuroSemanticSharding : IndexNeuro {
    /// Number of shards
    int nshards = 8;

    /// Shard centroids (semantic representatives)
    std::vector<float> shard_centroids;

    /// Vectors per shard
    std::vector<std::vector<float>> shard_vectors;

    /// IDs per shard
    std::vector<std::vector<idx_t>> shard_ids;

    /// Number of shards to search
    int nprobe = 2;

    /// Early termination threshold (relative distance)
    float early_termination_ratio = 2.0f;

    /// Shard size statistics
    mutable std::vector<size_t> shard_sizes;

    IndexNeuroSemanticSharding() = default;

    /** Construct with dimension and shards.
     * @param d         dimension
     * @param nshards   number of shards
     */
    IndexNeuroSemanticSharding(int d, int nshards = 8);

    ~IndexNeuroSemanticSharding() override = default;

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

    /// Get shard statistics
    std::vector<size_t> get_shard_sizes() const;

    /// Rebalance shards if imbalanced
    void rebalance_shards();

private:
    /// Find shard for a vector
    int find_shard(const float* vec) const;

    /// All vectors storage for reconstruction
    std::vector<float> all_vectors;
};

/// Parameters for semantic sharding search
struct NeuroSemanticShardParams : NeuroSearchParameters {
    int nprobe = -1;
    float early_termination_ratio = -1.0f;

    ~NeuroSemanticShardParams() override = default;
};

} // namespace faiss
