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

/// Zone configuration for zoned binarization
struct NeuroZoneConfig {
    float float_ratio = 0.10f;   ///< 10% critical
    float int8_ratio = 0.20f;    ///< 20% high
    float binary_ratio = 0.70f;  ///< 70% binary
};

/** BZ-01: Zoned Binarization with Multi-Precision Storage.
 *
 * Implements multi-precision zones based on dimension importance:
 *   - Critical zone (10%): float32 for highest-variance dimensions
 *   - High zone (20%): int8 for medium-importance dimensions
 *   - Binary zone (70%): 1-bit for remaining dimensions
 *
 * Key features:
 *   - 10x+ compression vs float32
 *   - >=95% recall with proper zone assignment
 *   - Dimension importance via variance analysis
 *   - 3-stage cascade search (Hamming -> L1 -> L2)
 */
struct IndexNeuroZonedBinarization : IndexNeuro {
    /// Zone configuration
    NeuroZoneConfig zone_config;

    /// Per-dimension importance scores (higher = more important)
    std::vector<float> dim_importance;

    /// Zone assignments for each dimension (0=float, 1=int8, 2=binary)
    std::vector<int> zone_assignments;

    /// Dimension indices sorted by importance
    std::vector<int> float_dims;
    std::vector<int> int8_dims;
    std::vector<int> binary_dims;

    /// Storage for each zone
    std::vector<float> data_float;
    std::vector<int8_t> data_int8;
    std::vector<uint8_t> data_binary;

    /// Binarization thresholds (per binary dimension)
    std::vector<float> thresholds;

    /// int8 scaling parameters (per int8 dimension)
    std::vector<float> int8_scales;
    std::vector<float> int8_mins;

    /// Number of rerank candidates
    int rerank_k = 100;

    /// Zone weights for combined distance
    float weight_binary = 0.3f;
    float weight_int8 = 0.3f;
    float weight_float = 0.4f;

    /// Optional metric for final reranking
    NeuroMetric* metric = nullptr;

    /// Store original vectors for full-precision reranking
    std::vector<float> orig_vectors;

    /// Whether to do full-precision reranking
    bool do_full_rerank = true;

    IndexNeuroZonedBinarization() = default;

    /** Construct with dimension.
     * @param d         dimension
     * @param rerank_k  candidates for reranking
     */
    explicit IndexNeuroZonedBinarization(int d, int rerank_k = 100);

    /** Construct with custom zone configuration.
     * @param d           dimension
     * @param float_pct   percentage for float zone (0-1)
     * @param int8_pct    percentage for int8 zone (0-1)
     * @param rerank_k    candidates for reranking
     */
    IndexNeuroZonedBinarization(
            int d,
            float float_pct,
            float int8_pct,
            int rerank_k = 100);

    /** Construct wrapping existing inner index.
     * @param inner_index  the index to wrap (for original vectors)
     * @param rerank_k     candidates for reranking
     */
    IndexNeuroZonedBinarization(Index* inner_index, int rerank_k = 100);

    ~IndexNeuroZonedBinarization() override = default;

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

    /// Get compression ratio vs float32
    float get_compression_ratio() const;

    /// Get number of dimensions in each zone
    void get_zone_sizes(int& n_float, int& n_int8, int& n_binary) const;

private:
    /// Analyze dimension importance from training data
    void analyze_importance(idx_t n, const float* x);

    /// Assign dimensions to zones based on importance
    void assign_zones();

    /// Encode a vector to zoned storage
    void encode_vector(
            const float* x,
            float* out_float,
            int8_t* out_int8,
            uint8_t* out_binary) const;

    /// Compute Hamming distance for binary zone
    int hamming_distance(const uint8_t* a, const uint8_t* b, size_t nbytes) const;

    /// Compute L1 distance for int8 zone
    float l1_distance_int8(const int8_t* a, const int8_t* b, size_t n) const;

    /// Compute L2 distance for float zone
    float l2_distance_float(const float* a, const float* b, size_t n) const;
};

/// Parameters for zoned binarization search
struct NeuroZonedParams : NeuroSearchParameters {
    int rerank_k = -1;

    ~NeuroZonedParams() override = default;
};

} // namespace faiss
