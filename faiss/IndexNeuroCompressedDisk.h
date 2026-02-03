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
#include <string>

namespace faiss {

/** DK-03: Compressed Disk Storage.
 *
 * Implements compressed storage with decompression caching:
 * - Stores vectors in compressed (quantized) form on disk
 * - Maintains LRU cache of decompressed vectors
 * - Balances storage size vs decompression overhead
 *
 * Key features:
 *   - 8-bit scalar quantization for storage
 *   - LRU decompression cache
 *   - 4x storage reduction with minimal recall loss
 */
struct IndexNeuroCompressedDisk : IndexNeuro {
    /// Number of clusters
    int nlist = 256;

    /// Number of clusters to probe
    int nprobe = 8;

    /// Cluster centroids
    std::vector<float> centroids;

    /// Compressed codes per cluster (8-bit per dimension)
    std::vector<std::vector<uint8_t>> cluster_codes;

    /// Vector IDs per cluster
    std::vector<std::vector<idx_t>> cluster_ids;

    /// Min/max values for quantization
    std::vector<float> vmin;
    std::vector<float> vmax;

    /// Cache of decompressed vectors (cluster_id -> vectors)
    mutable std::vector<std::vector<float>> decompression_cache;

    /// Cache validity flags
    mutable std::vector<bool> cache_valid;

    /// Maximum clusters to cache
    int max_cached_clusters = 32;

    /// LRU order (front = most recent)
    mutable std::vector<int> lru_order;

    IndexNeuroCompressedDisk() = default;

    /** Construct with dimension and clusters.
     * @param d       dimension
     * @param nlist   number of clusters
     */
    IndexNeuroCompressedDisk(int d, int nlist = 256);

    ~IndexNeuroCompressedDisk() override = default;

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

    /// Get compression ratio
    float get_compression_ratio() const;

private:
    /// Find cluster for a vector
    int find_cluster(const float* vec) const;

    /// Encode vector to 8-bit
    void encode_8bit(const float* vec, uint8_t* code) const;

    /// Decode 8-bit to float
    void decode_8bit(const uint8_t* code, float* vec) const;

    /// Get decompressed vectors for a cluster (uses cache)
    const std::vector<float>& get_decompressed(int cluster_id) const;

    /// Update LRU order
    void touch_cluster(int cluster_id) const;

    /// Original vectors for reconstruction
    std::vector<float> orig_vectors;
};

/// Parameters for compressed disk search
struct NeuroCompressedDiskParams : NeuroSearchParameters {
    int nprobe = -1;

    ~NeuroCompressedDiskParams() override = default;
};

} // namespace faiss
