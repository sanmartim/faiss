/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <faiss/IndexFlat.h>
#include <string>
#include <vector>
#include <fstream>

namespace faiss {

/** DK-01: Disk-Based Graph Index with SSD-Optimized Access.
 *
 * Implements a graph-based index optimized for SSD access patterns,
 * with in-memory navigation graph and disk-resident vectors.
 *
 * Key features:
 *   - Graph navigation with minimal disk reads
 *   - Prefetching for sequential access patterns
 *   - Support for datasets larger than RAM
 */
struct IndexNeuroDiskANN : IndexNeuro {
    /// Path to disk storage file
    std::string disk_path;

    /// In-memory graph: adjacency list per vector
    std::vector<std::vector<idx_t>> graph;

    /// Maximum graph degree (neighbors per node)
    int max_degree = 32;

    /// Search beam width
    int search_L = 100;

    /// Build graph beam width
    int build_L = 200;

    /// Whether vectors are stored on disk
    bool use_disk = false;

    /// In-memory vectors (fallback when disk not used)
    std::vector<float> mem_vectors;

    /// File handle for disk vectors
    mutable std::fstream disk_file;

    IndexNeuroDiskANN() = default;

    /** Construct with dimension and graph parameters.
     * @param d           dimension
     * @param max_degree  max neighbors per node
     * @param search_L    search beam width
     */
    explicit IndexNeuroDiskANN(int d, int max_degree = 32, int search_L = 100);

    ~IndexNeuroDiskANN() override;

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

    /// Set disk storage path and enable disk mode
    void set_disk_path(const std::string& path);

    /// Flush vectors to disk
    void flush_to_disk();

    /// Load vectors from disk to memory
    void load_to_memory();

private:
    /// Read vector from storage
    void read_vector(idx_t idx, float* vec) const;

    /// Build graph using greedy search
    void build_graph(idx_t n, const float* x);

    /// Greedy graph search
    void greedy_search(
            const float* query,
            idx_t start,
            int beam_width,
            std::vector<std::pair<float, idx_t>>& result,
            int max_results) const;
};

/// Parameters for DiskANN search
struct NeuroDiskANNParams : NeuroSearchParameters {
    int search_L = -1;  ///< -1 = use index default

    ~NeuroDiskANNParams() override = default;
};

} // namespace faiss
