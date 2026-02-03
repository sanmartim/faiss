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

/** BZ-02: Adaptive Zones with Region-Specific Configurations.
 *
 * Implements density-based region detection with per-region
 * precision configurations for optimal memory/recall tradeoffs.
 *
 * Key features:
 *   - Density-based region clustering
 *   - Per-region precision (high-density = higher precision)
 *   - +2-3% recall vs fixed zones
 */
struct IndexNeuroAdaptiveZones : IndexNeuro {
    /// Number of regions
    int nregions = 8;

    /// Region centroids
    std::vector<float> region_centroids;

    /// Region assignments for each vector
    std::vector<int> region_assignments;

    /// Bits per region (higher for dense regions)
    std::vector<int> region_bits;

    /// Encoded vectors per region
    std::vector<std::vector<uint8_t>> region_codes;

    /// Vectors per region
    std::vector<std::vector<idx_t>> region_vectors;

    /// Original vectors
    std::vector<float> orig_vectors;

    /// Number of rerank candidates
    int rerank_k = 100;

    /// Number of regions to probe
    int nprobe = 3;

    IndexNeuroAdaptiveZones() = default;

    /** Construct with dimension and region count.
     * @param d          dimension
     * @param nregions   number of regions
     */
    IndexNeuroAdaptiveZones(int d, int nregions = 8);

    ~IndexNeuroAdaptiveZones() override = default;

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
    /// Find region for a vector
    int find_region(const float* vec) const;

    /// Encode vector for a region
    void encode_for_region(const float* vec, int region, uint8_t* code) const;

    /// Get code size for a region
    size_t code_size_for_region(int region) const;

    /// Compute approximate distance
    float approx_distance(const float* query, idx_t idx) const;
};

/// Parameters for adaptive zones search
struct NeuroAdaptiveZonesParams : NeuroSearchParameters {
    int nprobe = -1;
    int rerank_k = -1;

    ~NeuroAdaptiveZonesParams() override = default;
};

} // namespace faiss
