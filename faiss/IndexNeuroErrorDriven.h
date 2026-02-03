/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <vector>

namespace faiss {

/** CB-03: Error-Driven Refinement Decorator.
 *
 * Inspired by cerebellar error-driven learning where climbing fibers
 * provide error signals that refine Purkinje cell responses.
 *
 * Wraps any Index and applies learned per-dimension refinement
 * weights that are updated based on search error feedback.
 *
 * The weights emphasize dimensions that help reduce search errors
 * and de-emphasize dimensions that are noisy or misleading.
 *
 * This is a DECORATOR pattern: wraps any Index.
 */
struct IndexNeuroErrorDriven : Index {
    /// The wrapped index
    Index* sub_index = nullptr;

    /// Whether to delete sub_index in destructor
    bool own_fields = false;

    /// Per-dimension refinement weights
    std::vector<float> refinement_weights;

    /// Learning rate for weight updates
    float learning_rate = 0.01f;

    /// Weight decay
    float weight_decay = 0.999f;

    /// Minimum weight (floor)
    float min_weight = 0.01f;

    /// Maximum weight (ceiling)
    float max_weight = 10.0f;

    /// Number of error feedback iterations
    int feedback_count = 0;

    IndexNeuroErrorDriven() = default;

    /** Construct wrapping any Index.
     * @param sub_index   the index to wrap
     * @param own_fields  take ownership of sub_index
     */
    IndexNeuroErrorDriven(Index* sub_index, bool own_fields = false);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Provide error feedback to refine weights.
     *
     * @param n_queries    number of queries
     * @param queries      query vectors (n_queries * d)
     * @param expected     expected (correct) results (n_queries * d)
     * @param actual       actual retrieved results (n_queries * d)
     */
    void feedback(
            idx_t n_queries,
            const float* queries,
            const float* expected,
            const float* actual);

    /** Provide binary feedback (correct/incorrect).
     *
     * @param query     the query vector (d floats)
     * @param result    the returned result (d floats)
     * @param correct   whether this result was correct
     */
    void feedback_binary(
            const float* query,
            const float* result,
            bool correct);

    /// Reset weights to uniform
    void reset_weights();

    /// Get current weights
    const std::vector<float>& get_weights() const { return refinement_weights; }

    ~IndexNeuroErrorDriven() override;

private:
    /// Apply refinement weights to queries
    void apply_weights(
            const float* queries,
            idx_t n,
            std::vector<float>& weighted) const;
};

} // namespace faiss
