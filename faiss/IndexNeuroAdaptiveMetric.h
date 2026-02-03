/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/NeuroDistance.h>
#include <memory>
#include <vector>

namespace faiss {

/** MT-05: Adaptive Metric Selection Decorator.
 *
 * Wraps any Index and analyzes data to select or combine
 * the best distance metric(s).
 *
 * Can operate in two modes:
 *   1. Selection: Choose single best metric from candidates
 *   2. Combination: Weighted combination of multiple metrics
 *
 * Selection is based on:
 *   - Data distribution (normality, sparsity)
 *   - Dimension correlations
 *   - Sample-based validation
 */
struct IndexNeuroAdaptiveMetric : Index {
    /// The wrapped index
    Index* sub_index = nullptr;

    /// Whether to delete sub_index in destructor
    bool own_fields = false;

    /// Available metrics for selection
    std::vector<std::unique_ptr<NeuroMetric>> candidate_metrics;

    /// Metric names for identification
    std::vector<std::string> metric_names;

    /// Combination weights (empty = single best metric)
    std::vector<float> metric_weights;

    /// Index of selected best metric (-1 = combination mode)
    int selected_metric = -1;

    /// Whether adaptation has been performed
    bool adapted = false;

    IndexNeuroAdaptiveMetric() = default;

    /** Construct wrapping any Index.
     * @param sub_index   the index to wrap
     * @param own_fields  take ownership of sub_index
     */
    IndexNeuroAdaptiveMetric(Index* sub_index, bool own_fields = false);

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

    /// Add a candidate metric
    void add_metric(std::unique_ptr<NeuroMetric> metric, const std::string& name);

    /// Add default metrics (L2, Cosine, Dot)
    void add_default_metrics();

    /** Analyze data and select/combine metrics.
     *
     * @param n        number of sample vectors
     * @param x        sample vectors (n * d)
     * @param combine  if true, use weighted combination; if false, select best
     */
    void adapt(idx_t n, const float* x, bool combine = false);

    /// Get the name of the selected/primary metric
    std::string get_selected_metric_name() const;

    /// Compute combined distance
    float combined_distance(const float* x1, const float* x2) const;

    ~IndexNeuroAdaptiveMetric() override;

private:
    /// Evaluate a metric on sample pairs
    float evaluate_metric(
            int metric_idx,
            idx_t n,
            const float* x) const;

    /// Analyze data distribution
    void analyze_distribution(idx_t n, const float* x,
                              float& sparsity, float& normality) const;
};

} // namespace faiss
