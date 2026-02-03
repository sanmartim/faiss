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

/// Parameters for place cell search
struct NeuroPlaceCellParams : NeuroSearchParameters {
    float activation_threshold = -1.0f;  ///< -1 = use index default
    bool rerank = true;

    ~NeuroPlaceCellParams() override = default;
};

/** HP-01: Place Cell Index.
 *
 * Inspired by hippocampal place cells that fire when the animal
 * is at specific locations in an environment.
 *
 * Each place cell has:
 *   - A center in the vector space
 *   - A Gaussian receptive field (field_size)
 *   - An inverted list of vectors that activate it
 *
 * Search:
 *   1. Find cells activated by query (Gaussian activation)
 *   2. Union the inverted lists of active cells
 *   3. Rerank candidates with true distance
 *
 * The receptive field overlap creates smooth coverage of the space.
 */
struct IndexNeuroPlaceCell : IndexNeuro {
    /// Number of place cells
    int n_cells = 1000;

    /// Receptive field size (standard deviation of Gaussian)
    float field_size = 0.1f;

    /// Activation threshold (minimum Gaussian response to activate)
    float activation_threshold = 0.1f;

    /// Maximum cells to activate per query
    int max_active_cells = 50;

    /// Whether to rerank candidates with true distance
    bool rerank = true;

    /// Optional pluggable metric for reranking
    NeuroMetric* metric = nullptr;

    /// Place cell centers: n_cells * d
    std::vector<float> cell_centers;

    /// Inverted lists: cell -> list of vector indices
    std::vector<std::vector<idx_t>> cell_lists;

    /// Precomputed 1 / (2 * field_size^2) for Gaussian
    float inv_2_sigma_sq = 0.0f;

    IndexNeuroPlaceCell() = default;

    /** Construct with inner IndexFlat.
     * @param inner       inner index (must be IndexFlat)
     * @param n_cells     number of place cells
     * @param field_size  receptive field size
     * @param own_inner   take ownership
     */
    IndexNeuroPlaceCell(
            Index* inner,
            int n_cells = 1000,
            float field_size = 0.1f,
            bool own_inner = false);

    /** Train: initialize cell centers via k-means or random sampling. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: assign to overlapping cells. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Compute Gaussian activation for a cell
    float cell_activation(int cell, const float* x) const;

    /// Get cells activated by a vector
    void get_active_cells(
            const float* x,
            float threshold,
            std::vector<int>& cells) const;
};

} // namespace faiss
