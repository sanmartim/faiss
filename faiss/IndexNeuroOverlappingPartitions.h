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

/** PT-01: Overlapping Partitions with Soft Boundaries.
 *
 * Implements partitioned search where vectors can belong to multiple
 * partitions, avoiding hard boundary issues in clustered indices.
 *
 * Key features:
 *   - Soft partition boundaries with overlap
 *   - Multi-probe search across partitions
 *   - Better recall at cluster boundaries
 */
struct IndexNeuroOverlappingPartitions : IndexNeuro {
    /// Number of partitions
    int npartitions = 16;

    /// Overlap ratio (fraction of partition size)
    float overlap_ratio = 0.2f;

    /// Number of partitions to probe during search
    int nprobe = 4;

    /// Partition centroids
    std::vector<float> centroids;

    /// Vectors per partition (with overlap)
    std::vector<std::vector<idx_t>> partition_lists;

    /// All stored vectors
    std::vector<float> vectors;

    /// Optional metric for distance computation
    NeuroMetric* metric = nullptr;

    IndexNeuroOverlappingPartitions() = default;

    /** Construct with dimension and partition parameters.
     * @param d             dimension
     * @param npartitions   number of partitions
     * @param overlap_ratio overlap as fraction of partition size
     */
    IndexNeuroOverlappingPartitions(
            int d,
            int npartitions = 16,
            float overlap_ratio = 0.2f);

    ~IndexNeuroOverlappingPartitions() override = default;

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

private:
    /// Find nearest partitions for a vector
    void find_partitions(
            const float* vec,
            int n_partitions,
            std::vector<int>& partition_ids) const;

    /// Assign vector to partitions (with overlap)
    void assign_to_partitions(idx_t vec_id, const float* vec);
};

/// Parameters for overlapping partitions search
struct NeuroOverlapParams : NeuroSearchParameters {
    int nprobe = -1;  ///< -1 = use index default

    ~NeuroOverlapParams() override = default;
};

} // namespace faiss
