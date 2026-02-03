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

/// Parameters for granule cell search
struct NeuroGranuleParams : NeuroSearchParameters {
    float threshold = -1.0f;  ///< -1 = use index default
    bool rerank = true;

    ~NeuroGranuleParams() override = default;
};

/** CB-01: Granule Cell Expansion.
 *
 * Inspired by cerebellar granule cells that provide massive expansion
 * of mossy fiber input (~50x) with sparse, non-negative connectivity.
 *
 * Architecture:
 *   Mossy fibers (d dims)
 *       ↓ sparse random expansion (non-negative weights)
 *   Granule cells (n_granule = expansion * d)
 *       ↓ threshold activation (ReLU-like)
 *   Granule_sparse
 *       ↓ Purkinje readout (learned weights)
 *   Output
 *
 * The massive expansion followed by sparse activation creates
 * high-dimensional sparse codes suitable for pattern classification.
 */
struct IndexNeuroGranule : IndexNeuro {
    /// Expansion factor (n_granule = expansion * d)
    int expansion = 50;

    /// Connections per granule cell from mossy fibers
    int connections_per_granule = 4;

    /// Activation threshold (ReLU with threshold)
    float threshold = 0.0f;

    /// Number of Purkinje cells (output dimension)
    int n_purkinje = 1;

    /// Learning rate for Purkinje weight updates
    float learning_rate = 0.01f;

    /// Whether to rerank with true distance
    bool rerank = true;

    /// Optional pluggable metric
    NeuroMetric* metric = nullptr;

    // Internal structures

    /// Number of granule cells
    int n_granule = 0;

    /// Mossy fiber → Granule projection: n_granule * connections_per_granule indices
    std::vector<int> mf_to_granule_indices;

    /// Mossy fiber → Granule weights: n_granule * connections_per_granule (non-negative)
    std::vector<float> mf_to_granule_weights;

    /// Purkinje readout weights: n_purkinje * n_granule
    std::vector<float> purkinje_weights;

    /// Sparse granule codes for all vectors (thresholded activations)
    std::vector<std::vector<std::pair<int, float>>> granule_codes;

    IndexNeuroGranule() = default;

    /** Construct with inner IndexFlat.
     * @param inner      inner index (must be IndexFlat)
     * @param expansion  expansion factor (default 50)
     * @param own_inner  take ownership
     */
    IndexNeuroGranule(Index* inner, int expansion = 50, bool own_inner = false);

    /** Train: initialize projection matrices. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: compute and store granule codes. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Compute sparse granule code for a vector
    void compute_granule_code(
            const float* x,
            std::vector<std::pair<int, float>>& code) const;

    /// Compute Purkinje output from granule code
    void compute_purkinje_output(
            const std::vector<std::pair<int, float>>& code,
            std::vector<float>& output) const;

    /** Supervised training of Purkinje weights.
     *
     * @param n_samples   number of training samples
     * @param x           input vectors (n_samples * d)
     * @param targets     target outputs (n_samples * n_purkinje)
     * @param n_epochs    number of training epochs
     */
    void train_purkinje(
            idx_t n_samples,
            const float* x,
            const float* targets,
            int n_epochs = 10);
};

} // namespace faiss
