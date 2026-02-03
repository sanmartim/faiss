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

/** MT-06: Cross-Modal Anchor Search.
 *
 * Supports search across different modalities (e.g., text-to-image,
 * audio-to-text) using:
 *   - Per-modality anchor points
 *   - Learned alignment between modality spaces
 *   - Profile-based cross-modal distance
 *
 * Key concepts:
 *   - Each modality has its own anchor set
 *   - Alignment matrix maps profiles between modalities
 *   - Search can query one modality and retrieve from another
 *
 * Training requires paired data from both modalities.
 */
struct IndexNeuroCrossModal : IndexNeuro {
    /// Number of modalities (typically 2)
    int n_modalities = 2;

    /// Number of anchors per modality
    int n_anchors = 64;

    /// Dimension of each modality (can differ)
    std::vector<int> modality_dims;

    /// Anchors per modality: n_modalities arrays of (n_anchors * dim)
    std::vector<std::vector<float>> modality_anchors;

    /// Alignment matrices: n_modalities * n_modalities
    /// alignment[i][j] maps from modality i profile to modality j profile
    std::vector<std::vector<float>> alignment_matrices;

    /// Profiles per modality: vectors in each modality
    std::vector<std::vector<float>> modality_profiles;

    /// Which modality each stored vector belongs to
    std::vector<int> vector_modalities;

    /// Original vectors for reranking
    std::vector<float> stored_vectors;

    /// Current query modality (for search)
    int query_modality = 0;

    /// Target modality (for search, -1 = all)
    int target_modality = -1;

    /// Whether to rerank with true distance
    bool rerank = true;

    /// Number of candidates before rerank
    int n_candidates = 100;

    /// Learning rate for alignment training
    float alignment_lr = 0.01f;

    /// Optional pluggable metric
    NeuroMetric* metric = nullptr;

    IndexNeuroCrossModal() = default;

    /** Construct with modality dimensions.
     * @param dims      dimension of each modality
     * @param n_anchors number of anchors per modality
     */
    IndexNeuroCrossModal(
            const std::vector<int>& dims,
            int n_anchors = 64);

    /** Construct for two modalities.
     * @param d1        dimension of modality 1
     * @param d2        dimension of modality 2
     * @param n_anchors number of anchors per modality
     */
    IndexNeuroCrossModal(int d1, int d2, int n_anchors = 64);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /** Train anchors for a specific modality.
     * @param modality  which modality (0 or 1)
     * @param n         number of vectors
     * @param x         vectors (n * modality_dims[modality])
     */
    void train_modality(int modality, idx_t n, const float* x);

    /** Add vectors for a specific modality.
     * @param modality  which modality
     * @param n         number of vectors
     * @param x         vectors (n * modality_dims[modality])
     */
    void add_modality(int modality, idx_t n, const float* x);

    /** Train alignment from paired data.
     * @param n         number of pairs
     * @param x1        vectors from modality 1 (n * d1)
     * @param x2        vectors from modality 2 (n * d2)
     * @param n_iter    number of training iterations
     */
    void train_alignment(
            idx_t n,
            const float* x1,
            const float* x2,
            int n_iter = 100);

    /** Set the query and target modalities for search.
     * @param query     query modality
     * @param target    target modality (-1 = all)
     */
    void set_search_modalities(int query, int target);

    /// Compute profile for a vector in a specific modality
    void compute_profile(int modality, const float* x, float* profile) const;

    /// Map profile from one modality to another
    void map_profile(
            int from_modality,
            int to_modality,
            const float* src_profile,
            float* dst_profile) const;

    /// Reconstruct vector from stored data
    void reconstruct(idx_t key, float* recons) const override;

private:
    /// Select anchors for a modality using farthest point sampling
    void select_anchors(int modality, idx_t n, const float* x);

    /// Profile distance
    float profile_distance(const float* p1, const float* p2) const;
};

/// Parameters for cross-modal search
struct NeuroCrossModalParams : NeuroSearchParameters {
    int query_modality = 0;      ///< which modality is the query
    int target_modality = -1;    ///< which modality to search (-1 = all)
    int candidates = -1;         ///< -1 = use index default
    bool rerank = true;

    ~NeuroCrossModalParams() override = default;
};

} // namespace faiss
