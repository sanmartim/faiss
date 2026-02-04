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

/// Parameters for MicroZones search
struct NeuroMicroZonesParams : NeuroSearchParameters {
    /// Maximum zone distance for candidate filtering (0 = auto)
    int max_zone_distance = 0;

    /// Keep ratio for first stage filtering
    float keep_ratio = 0.2f;

    ~NeuroMicroZonesParams() override = default;
};

/** V5-MZ01: Micro Zones Index.
 *
 * Addresses V4's 50% recall limitation by preserving magnitude information.
 *
 * Instead of binary sign(tanh(x * scale)) which loses magnitude:
 * - Uses tanh(x * scale) and quantizes into micro zones
 * - Each dimension is encoded as 2 bits (4 zones)
 * - Zones: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1]
 * - Multiple scales provide different resolutions
 *
 * Zone encoding per dimension:
 *   tanh(x*s) in [-1.0, -0.5) -> zone 0 (0b00)
 *   tanh(x*s) in [-0.5,  0.0) -> zone 1 (0b01)
 *   tanh(x*s) in [ 0.0,  0.5) -> zone 2 (0b10)
 *   tanh(x*s) in [ 0.5,  1.0] -> zone 3 (0b11)
 *
 * Zone distance: abs difference of zone indices (max 3 per dimension)
 * Total zone distance threshold replaces Hamming threshold.
 *
 * This preserves magnitude information while maintaining fast filtering.
 */
struct IndexNeuroMicroZones : IndexNeuro {
    /// Number of zones per dimension (2 bits = 4 zones, 3 bits = 8 zones)
    int n_zones = 4;

    /// Bits per zone (log2 of n_zones)
    int bits_per_zone = 2;

    /// Scales for multi-scale encoding
    std::vector<float> scales = {0.5f, 1.0f, 2.0f};

    /// Maximum zone distance for filtering (0 = auto-compute)
    int max_zone_distance = 0;

    /// Keep ratio for candidates
    float keep_ratio = 0.2f;

    /// Zone boundaries for 4 zones: [-1, -0.5, 0, 0.5, 1]
    std::vector<float> zone_boundaries;

    // Internal structures

    /// Zone codes per scale: scales.size() vectors of (ntotal * bytes_per_vec)
    std::vector<std::vector<uint8_t>> zone_codes;

    /// Bytes per vector (ceil(d * bits_per_zone / 8))
    int bytes_per_vec = 0;

    /// Original vectors for L2 reranking
    std::vector<float> vectors;

    IndexNeuroMicroZones() = default;

    /** Construct MicroZones index.
     * @param d dimension
     * @param n_zones zones per dimension (default 4 = 2 bits)
     */
    explicit IndexNeuroMicroZones(int d, int n_zones = 4);

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

    /// Compute zone code for a vector at given scale
    void compute_zone_code(
            const float* vec,
            float scale,
            uint8_t* code) const;

    /// Compute zone distance between two codes
    int zone_distance(const uint8_t* code1, const uint8_t* code2) const;

    /// Get zone index for a value
    int get_zone(float val) const;
};

} // namespace faiss
