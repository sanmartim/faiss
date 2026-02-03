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

/// Parameters for pattern completion search
struct NeuroPatternCompletionParams : NeuroSearchParameters {
    int n_iterations = -1;     ///< -1 = use index default
    std::vector<bool> mask;    ///< which dimensions are known (true = known)

    ~NeuroPatternCompletionParams() override = default;
};

/** HP-03: Pattern Completion Index.
 *
 * Inspired by hippocampal pattern completion where partial cues
 * activate stored memories.
 *
 * Uses a Hopfield-like association matrix learned from data:
 *   W[i,j] = correlation between dimensions i and j
 *
 * Search with partial query (some dimensions unknown):
 *   1. Start with query (unknown dims = 0 or mean)
 *   2. Iterate: x_new = activation(W * x_old)
 *   3. Search with completed query
 *
 * Enables search when query has missing/unknown dimensions.
 */
struct IndexNeuroPatternCompletion : IndexNeuro {
    /// Number of completion iterations
    int n_iterations = 10;

    /// Association matrix: d * d (Hopfield weights)
    std::vector<float> association_matrix;

    /// Per-dimension mean (for initialization of unknown dims)
    std::vector<float> dim_means;

    /// Per-dimension std (for normalization)
    std::vector<float> dim_stds;

    /// Whether to normalize input before completion
    bool normalize_input = true;

    /// Activation function: "tanh", "relu", "linear"
    std::string activation = "tanh";

    /// Optional pluggable metric
    NeuroMetric* metric = nullptr;

    IndexNeuroPatternCompletion() = default;

    /** Construct with inner IndexFlat.
     * @param inner        inner index (must be IndexFlat)
     * @param n_iterations completion iterations (default 10)
     * @param own_inner    take ownership
     */
    IndexNeuroPatternCompletion(
            Index* inner,
            int n_iterations = 10,
            bool own_inner = false);

    /** Train: compute association matrix from data. */
    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Complete a partial pattern.
     *
     * @param x        input pattern (d floats, unknown dims can be any value)
     * @param mask     which dimensions are known (true = known)
     * @param out      completed pattern (d floats)
     * @param n_iter   number of iterations (-1 = use default)
     */
    void complete_pattern(
            const float* x,
            const std::vector<bool>& mask,
            float* out,
            int n_iter = -1) const;

    /// Apply activation function
    float apply_activation(float x) const;
};

} // namespace faiss
