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

/// Parameters for temporal search
struct NeuroTemporalParams : NeuroSearchParameters {
    int sequence_length = -1;  ///< -1 = use stored length

    ~NeuroTemporalParams() override = default;
};

/** CB-02: Temporal Basis Index.
 *
 * Inspired by cerebellar temporal processing with basis functions
 * that represent sequences at different time scales.
 *
 * Encodes sequences (T time steps x d dimensions) into a fixed
 * temporal representation using basis functions with varying
 * time constants (tau).
 *
 * Temporal basis: b_k(t) = exp(-t / tau_k) * sin(omega_k * t)
 *
 * Each sequence is encoded as:
 *   encoded[k, j] = sum_t x[t, j] * b_k(t)
 *
 * This enables fast approximate sequence matching without DTW.
 */
struct IndexNeuroTemporal : IndexNeuro {
    /// Number of temporal basis functions
    int n_bases = 100;

    /// Time constant range (min, max)
    float tau_min = 0.1f;
    float tau_max = 1.0f;

    /// Maximum sequence length for basis computation
    int max_sequence_length = 100;

    /// Optional pluggable metric
    NeuroMetric* metric = nullptr;

    // Internal structures

    /// Temporal basis functions: n_bases * max_sequence_length
    std::vector<float> basis_functions;

    /// Encoded sequences: ntotal * n_bases * d
    std::vector<float> encoded_data;

    /// Stored sequence lengths for each vector
    std::vector<int> sequence_lengths;

    /// Original vectors for L2 reranking (when using standard add())
    std::vector<float> original_vectors;

    IndexNeuroTemporal() = default;

    /** Construct with inner IndexFlat.
     * @param inner    inner index (must be IndexFlat, stores flattened sequences)
     * @param n_bases  number of temporal basis functions (default 100)
     * @param own_inner take ownership
     */
    IndexNeuroTemporal(Index* inner, int n_bases = 100, bool own_inner = false);

    /** Train: generate temporal basis functions. */
    void train(idx_t n, const float* x) override;

    /** Add a sequence.
     * @param seq_length  length of the sequence (T time steps)
     * @param x           sequence data (T * d floats, row-major)
     */
    void add_sequence(int seq_length, const float* x);

    /** Standard add - treats each vector as single-timestep sequence. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    /** Search with a query sequence.
     * @param seq_length  query sequence length
     * @param query       query sequence (T * d floats)
     * @param k           number of neighbors
     * @param distances   output distances (k floats)
     * @param labels      output labels (k indices)
     */
    void search_sequence(
            int seq_length,
            const float* query,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Encode a sequence using temporal basis
    void encode_sequence(
            int seq_length,
            const float* seq,
            float* encoded) const;

    /// Get the temporal basis function value at time t
    float basis_value(int basis_idx, int t) const;
};

} // namespace faiss
