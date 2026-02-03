/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/NeuroDistance.h>
#include <cstdint>
#include <vector>

namespace faiss {

/// Parameters for FlyHash search, overridable per query
struct NeuroFlyHashParams : NeuroSearchParameters {
    int k_sparse = -1;       ///< -1 = use index default (d * sparsity)
    bool rerank = true;      ///< whether to rerank with true distance

    ~NeuroFlyHashParams() override = default;
};

/** HS-02: FlyHash - Drosophila mushroom body inspired hashing.
 *
 * Mimics the projection from projection neurons (PN) to Kenyon cells (KC)
 * in the Drosophila mushroom body:
 *
 *   1. Sparse random expansion: d -> m (expansion_factor * d)
 *   2. Winner-take-all (WTA): keep only top k_sparse activations
 *   3. Jaccard similarity on sparse binary codes
 *
 * Key insight: High expansion followed by extreme sparsification
 * creates locality-sensitive hash codes that preserve similarity.
 *
 * Reference: Dasgupta et al. "A neural algorithm for a fundamental
 * computing problem" (Science, 2017)
 */
struct IndexNeuroFlyHash : IndexNeuro {
    /// Expansion factor: m = expansion_factor * d
    int expansion_factor = 20;

    /// Sparsity after WTA: keep top (sparsity * m) neurons
    float sparsity = 0.05f;

    /// Connections per expanded neuron (sparse projection)
    int connections_per_neuron = 6;

    /// Whether to rerank candidates with true distance
    bool rerank = true;

    /// Optional pluggable metric for reranking (nullptr = L2)
    NeuroMetric* metric = nullptr;

    /// Expanded dimension m = expansion_factor * d
    int m = 0;

    /// Number of active neurons after WTA
    int k_sparse = 0;

    /// Sparse projection matrix: m connections, each with (input_idx, weight)
    std::vector<int> projection_indices;
    std::vector<float> projection_weights;

    /// Sparse binary codes for all vectors (flattened)
    std::vector<int> codes_data;
    std::vector<size_t> codes_offsets;

    IndexNeuroFlyHash() = default;

    /** Construct with inner IndexFlat.
     * @param inner            inner index (must be IndexFlat)
     * @param expansion_factor expansion ratio (default 20)
     * @param sparsity         fraction of neurons active after WTA (default 0.05)
     * @param own_inner        take ownership
     */
    IndexNeuroFlyHash(
            Index* inner,
            int expansion_factor = 20,
            float sparsity = 0.05f,
            bool own_inner = false);

    /** Train: generate sparse random projection matrix. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: compute and store sparse codes. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Compute sparse code for a single vector
    void compute_code(const float* x, std::vector<int>& code) const;

    /// Jaccard similarity between two sparse codes
    float jaccard_similarity(
            const std::vector<int>& a,
            const std::vector<int>& b) const;
};

/** HS-03: BioHash - FlyHash with learnable weights.
 *
 * Extends FlyHash with Hebbian learning to adapt the projection
 * weights based on similarity feedback.
 *
 * Training from similarity pairs (a, b, is_similar):
 *   - If similar: increase weights connecting shared active neurons
 *   - If dissimilar: decrease weights connecting shared active neurons
 *
 * After training, the hash codes become more discriminative for
 * the specific similarity structure in the data.
 */
struct IndexNeuroBioHash : IndexNeuroFlyHash {
    /// Learning rate for Hebbian updates
    float learning_rate = 0.01f;

    /// Weight decay factor per update
    float weight_decay = 0.999f;

    /// Number of training iterations performed
    int training_iterations = 0;

    IndexNeuroBioHash() = default;

    /** Construct with inner IndexFlat.
     * @param inner            inner index
     * @param expansion_factor expansion ratio (default 20)
     * @param sparsity         fraction active after WTA (default 0.05)
     * @param own_inner        take ownership
     */
    IndexNeuroBioHash(
            Index* inner,
            int expansion_factor = 20,
            float sparsity = 0.05f,
            bool own_inner = false);

    /** Hebbian training from similarity pairs.
     *
     * @param n_pairs     number of training pairs
     * @param vec_a       first vectors (n_pairs * d)
     * @param vec_b       second vectors (n_pairs * d)
     * @param is_similar  1 if pair is similar, 0 otherwise (n_pairs)
     */
    void train_pairs(
            idx_t n_pairs,
            const float* vec_a,
            const float* vec_b,
            const int* is_similar);

    /** Incremental Hebbian update from a single pair.
     *
     * @param vec_a       first vector (d floats)
     * @param vec_b       second vector (d floats)
     * @param is_similar  true if pair is similar
     */
    void update_pair(
            const float* vec_a,
            const float* vec_b,
            bool is_similar);

    /// Recompute all codes after weight updates
    void recompute_codes();
};

} // namespace faiss
