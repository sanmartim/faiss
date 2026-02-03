/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <string>
#include <vector>

namespace faiss {

/// Parameters for learned-weight search (PA-01), overridable per query
struct NeuroWeightedParams : NeuroSearchParameters {
    std::vector<float> weights; ///< empty = use index weights

    ~NeuroWeightedParams() override = default;
};

/** PA-01: Per-dimension learned weights via Hebbian feedback.
 *
 * Wraps an IndexFlat and computes weighted L2 distance:
 *   dist(q, x) = sum_j w[j] * (q[j] - x[j])^2
 *
 * Weights start uniform (1.0) and are updated via feedback():
 *   - Dimensions that help correct rankings get weight increase
 *   - Dimensions that hurt get weight decrease
 *   - Exponential decay toward uniform to prevent divergence
 *
 * MR-03v2: feedback_contrastive() supports multiple iterations with
 * momentum-based updates and hard negative mining for improved convergence.
 */
struct IndexNeuroWeighted : IndexNeuro {
    std::vector<float> weights; ///< per-dimension weights, size = d

    float learning_rate = 0.05f;  ///< step size for weight updates
    float weight_decay = 0.99f;   ///< multiplicative decay per feedback round
    float min_weight = 0.01f;     ///< floor to prevent zero weights

    /// MR-03v2: Number of iterations for contrastive feedback (default 5)
    int contrastive_iterations = 5;

    /// MR-03v2: Ratio of negatives to use for hard mining (0.3 = top 30%)
    float hard_negative_ratio = 0.3f;

    /// MR-03v2: Momentum for gradient accumulation (0.9)
    float contrastive_momentum = 0.9f;

    int feedback_count = 0; ///< number of feedback() calls so far

    IndexNeuroWeighted() = default;

    /** Construct with inner IndexFlat.
     * @param inner  inner index (must be IndexFlat)
     * @param own_inner  take ownership
     */
    explicit IndexNeuroWeighted(Index* inner, bool own_inner = false);

    /// Initialize weights to uniform 1.0
    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Hebbian feedback: update weights from relevance judgments.
     *
     * @param nq        number of queries
     * @param queries   query vectors (nq * d)
     * @param positives positive (relevant) vectors (nq * d)
     * @param negatives negative (non-relevant) vectors (nq * d)
     *
     * For each query, dimensions where |q-pos| < |q-neg| get weight
     * increase; dimensions where |q-pos| > |q-neg| get decrease.
     */
    void feedback(
            idx_t nq,
            const float* queries,
            const float* positives,
            const float* negatives);

    /** MR-03v2: Multi-iteration contrastive feedback with momentum.
     *
     * Runs multiple iterations (contrastive_iterations) of margin-based
     * gradient updates with momentum accumulation. Each iteration:
     *   1. Selects hard negatives (top hard_negative_ratio by distance)
     *   2. Computes margin-scaled gradient
     *   3. Applies momentum: v = momentum * v + gradient
     *   4. Updates weights with momentum velocity
     *
     * Uses per-dimension margin: margin_j = |dp_j - dn_j| to scale
     * the gradient. Large margins get larger updates (more confident
     * signal).
     *
     * @param nq             number of queries
     * @param queries        query vectors (nq * d)
     * @param positives      positive vectors (nq * d)
     * @param negatives      negative vectors (nq * n_negatives * d)
     * @param n_negatives    number of negatives per query (default 1)
     * @param margin_scale   scaling for margin contribution (default 1.0)
     */
    void feedback_contrastive(
            idx_t nq,
            const float* queries,
            const float* positives,
            const float* negatives,
            int n_negatives = 1,
            float margin_scale = 1.0f);

    /// Save weights to a binary file
    void save_weights(const char* fname) const;

    /// Load weights from a binary file
    void load_weights(const char* fname);
};

} // namespace faiss
