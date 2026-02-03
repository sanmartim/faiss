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

/** FS-02: Centroid Bounds Filter Index
 *
 * Cluster-based pruning using triangle inequality. Skip entire clusters
 * where centroid_dist - radius > best_so_far.
 *
 * Algorithm:
 * 1. Pre-cluster data using k-means
 * 2. Compute radius (max distance to centroid) per cluster
 * 3. At search, compute distances to centroids first
 * 4. Prune clusters using triangle inequality
 * 5. Search only promising clusters
 *
 * Typical performance: 98-99% recall at 10-50x speedup
 */
struct IndexNeuroCentroidBounds : IndexNeuro {
    /// Number of clusters
    int nlist = 100;

    /// Number of clusters to probe
    int nprobe = 10;

    /// Cluster centroids: nlist * d floats
    std::vector<float> centroids;

    /// Max distance to centroid per cluster (radius)
    std::vector<float> radii;

    /// Vector IDs per cluster
    std::vector<std::vector<idx_t>> cluster_ids;

    /// Original vectors (needed for precise L2)
    std::vector<float> vectors;

    /// Whether to use triangle inequality pruning
    bool use_triangle_inequality = true;

    /// Default constructor
    IndexNeuroCentroidBounds() = default;

    /** Construct with dimension and cluster count.
     * @param d vector dimensionality
     * @param nlist number of clusters (default: 100)
     */
    explicit IndexNeuroCentroidBounds(int d, int nlist = 100);

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
    /// Find nearest cluster for a vector
    int find_nearest_cluster(const float* vec) const;
};

} // namespace faiss
