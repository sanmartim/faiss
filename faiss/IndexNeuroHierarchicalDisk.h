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
#include <fstream>

namespace faiss {

/** DK-02: Hierarchical Disk Structure.
 *
 * Implements multi-level disk storage with progressive loading:
 * Level 0 (memory): coarse quantizer
 * Level 1 (SSD): cluster centroids
 * Level 2 (disk): full vectors
 *
 * Key features:
 *   - Progressive loading based on search needs
 *   - Memory-mapped cluster access
 *   - 10x capacity at 2x latency
 */
struct IndexNeuroHierarchicalDisk : IndexNeuro {
    /// Number of clusters (Level 1)
    int nlist = 256;

    /// Number of clusters to probe
    int nprobe = 8;

    /// Cluster centroids (in memory)
    std::vector<float> centroids;

    /// Cluster sizes
    std::vector<size_t> cluster_sizes;

    /// File path for disk storage
    std::string disk_path;

    /// Whether disk file is open
    bool is_disk_open = false;

    /// Memory-resident vectors (hot clusters)
    std::vector<std::vector<float>> memory_clusters;

    /// Memory-resident IDs
    std::vector<std::vector<idx_t>> memory_ids;

    /// Cluster query counts (for hot detection)
    mutable std::vector<int64_t> cluster_query_counts;

    /// Fraction of clusters to keep in memory
    float memory_ratio = 0.1f;

    /// Which clusters are in memory
    std::vector<bool> in_memory;

    /// File offsets for each cluster
    std::vector<size_t> cluster_offsets;

    IndexNeuroHierarchicalDisk() = default;

    /** Construct with dimension and clusters.
     * @param d       dimension
     * @param nlist   number of clusters
     * @param disk_path  path to disk storage file
     */
    IndexNeuroHierarchicalDisk(int d, int nlist = 256, const std::string& disk_path = "");

    ~IndexNeuroHierarchicalDisk() override;

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

    /// Save index to disk
    void save_to_disk();

    /// Load hot clusters to memory
    void load_hot_clusters();

    /// Update which clusters are hot based on query patterns
    void update_hot_clusters();

private:
    /// Find cluster for a vector
    int find_cluster(const float* vec) const;

    /// Read cluster from disk
    void read_cluster_from_disk(
            int cluster_id,
            std::vector<float>& vectors,
            std::vector<idx_t>& ids) const;

    /// Temporary storage for building
    std::vector<std::vector<float>> build_clusters;
    std::vector<std::vector<idx_t>> build_ids;

    /// All vectors storage (before save_to_disk)
    std::vector<float> all_vectors;
    std::vector<idx_t> all_ids;
};

/// Parameters for hierarchical disk search
struct NeuroHierarchicalDiskParams : NeuroSearchParameters {
    int nprobe = -1;

    ~NeuroHierarchicalDiskParams() override = default;
};

} // namespace faiss
