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

/// Parameters for contextual-weight search (PA-02), overridable per query
struct NeuroContextualParams : NeuroSearchParameters {
    int force_cluster = -1; ///< -1 = auto-classify, >= 0 = use this cluster

    ~NeuroContextualParams() override = default;
};

/** PA-02: Contextual per-query-type learned weights.
 *
 * Extends PA-01 by maintaining multiple weight vectors, one per query
 * cluster. During train(), the query space is clustered (k-means) and
 * each cluster gets its own weight vector. During search(), incoming
 * queries are assigned to the nearest cluster centroid, and the
 * corresponding weight vector is used for weighted L2 search.
 *
 * Each cluster's weights can be trained independently via feedback().
 */
struct IndexNeuroContextualWeighted : IndexNeuro {
    int n_query_clusters = 4;   ///< number of query-type clusters

    /// Cluster centroids: n_query_clusters * d
    std::vector<float> centroids;

    /// Per-cluster weight vectors: n_query_clusters * d
    std::vector<float> cluster_weights;

    float learning_rate = 0.05f;
    float weight_decay = 0.99f;
    float min_weight = 0.01f;

    std::vector<int> feedback_counts; ///< per-cluster feedback count

    IndexNeuroContextualWeighted() = default;

    /** Construct with inner IndexFlat.
     * @param inner           inner index (must be IndexFlat)
     * @param n_clusters      number of query-type clusters
     * @param own_inner       take ownership
     */
    IndexNeuroContextualWeighted(
            Index* inner,
            int n_clusters = 4,
            bool own_inner = false);

    /** Train: cluster a representative set of queries to define query types.
     * @param n  number of training queries
     * @param x  training query vectors (n * d)
     */
    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Feedback for a specific query cluster.
     * Classifies each query to its cluster, then updates that cluster's
     * weights using Hebbian learning (same as PA-01).
     */
    void feedback(
            idx_t nq,
            const float* queries,
            const float* positives,
            const float* negatives);

    /// Assign a single query to its nearest cluster
    int classify_query(const float* query) const;
};

} // namespace faiss
