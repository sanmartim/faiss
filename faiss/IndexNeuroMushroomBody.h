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

/// Parameters for mushroom body search
struct NeuroMushroomBodyParams : NeuroSearchParameters {
    int k_sparse = -1;        ///< -1 = use index default
    int compartment = -1;     ///< -1 = all compartments, else specific one
    bool rerank = true;

    ~NeuroMushroomBodyParams() override = default;
};

/** DR-01/DR-02: Mushroom Body circuit for similarity search.
 *
 * Implements the Drosophila mushroom body circuit:
 *
 *   PN (projection neurons, d dims)
 *       ↓ sparse random projection
 *   KC (Kenyon cells, n_kc dims, ~50x expansion)
 *       ↓ APL inhibition (winner-take-all)
 *   KC_sparse (only top k_sparse active)
 *       ↓ MBON readout weights (per compartment)
 *   Output (per-compartment similarity scores)
 *
 * The circuit implements pattern separation: similar inputs produce
 * decorrelated sparse codes, enabling fine-grained discrimination.
 *
 * DR-02 (PatternSeparation mode): Focuses on decorrelation metrics
 * and provides adaptive threshold tuning based on input statistics.
 */
struct IndexNeuroMushroomBody : IndexNeuro {
    /// Number of Kenyon cells (expansion)
    int n_kc = 2000;

    /// Connections per KC from PNs
    int connections_per_kc = 6;

    /// Sparsity level (fraction of KCs active)
    float kc_sparsity = 0.05f;

    /// Number of MBON compartments
    int n_compartments = 3;

    /// Whether to use pattern separation mode (DR-02)
    bool pattern_separation_mode = false;

    /// Adaptive threshold for pattern separation
    float separation_threshold = 0.0f;

    /// Learning rate for MBON weight updates
    float learning_rate = 0.01f;

    /// Weight decay for MBON updates
    float weight_decay = 0.999f;

    /// Whether to rerank with true distance
    bool rerank = true;

    /// Optional pluggable metric for reranking
    NeuroMetric* metric = nullptr;

    // Internal structures

    /// PN→KC projection: n_kc * connections_per_kc indices
    std::vector<int> pn_to_kc_indices;

    /// PN→KC projection weights: n_kc * connections_per_kc
    std::vector<float> pn_to_kc_weights;

    /// MBON readout weights: n_compartments * n_kc
    std::vector<float> mbon_weights;

    /// Sparse KC codes for all vectors: vector of active KC indices per vector
    std::vector<std::vector<int>> kc_codes;

    /// Number of active KCs after sparsification
    int k_sparse = 0;

    /// Feedback count
    int feedback_count = 0;

    IndexNeuroMushroomBody() = default;

    /** Construct with inner IndexFlat.
     * @param inner        inner index (must be IndexFlat)
     * @param n_kc         number of Kenyon cells (default 2000)
     * @param kc_sparsity  fraction of KCs active (default 0.05)
     * @param own_inner    take ownership
     */
    IndexNeuroMushroomBody(
            Index* inner,
            int n_kc = 2000,
            float kc_sparsity = 0.05f,
            bool own_inner = false);

    /** Train: initialize PN→KC projection and MBON weights. */
    void train(idx_t n, const float* x) override;

    /** Add vectors: compute and store KC codes. */
    void add(idx_t n, const float* x) override;

    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Compute sparse KC code for a single vector
    void compute_kc_code(const float* x, std::vector<int>& code) const;

    /// Compute MBON output for a KC code
    void compute_mbon_output(
            const std::vector<int>& kc_code,
            std::vector<float>& output) const;

    /** Dopaminergic feedback: update MBON weights.
     *
     * @param nq        number of queries
     * @param queries   query vectors (nq * d)
     * @param positives positive (rewarded) vectors (nq * d)
     * @param negatives negative (punished) vectors (nq * d)
     * @param compartment which compartment to update (-1 = all)
     */
    void feedback(
            idx_t nq,
            const float* queries,
            const float* positives,
            const float* negatives,
            int compartment = -1);

    // DR-02: Pattern Separation methods

    /** Compute pattern separation metric between two vectors.
     * Returns the decorrelation achieved by the KC representation.
     */
    float separation_metric(const float* x1, const float* x2) const;

    /** Adapt separation threshold based on data statistics.
     * @param n       number of sample vectors
     * @param x       sample vectors (n * d)
     * @param target_separation  target decorrelation level
     */
    void adapt_threshold(idx_t n, const float* x, float target_separation = 0.1f);
};

} // namespace faiss
