/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroCrossModal.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace faiss {

IndexNeuroCrossModal::IndexNeuroCrossModal(
        const std::vector<int>& dims,
        int n_anchors)
        : IndexNeuro(nullptr, false),
          n_modalities(dims.size()),
          n_anchors(n_anchors),
          modality_dims(dims) {
    FAISS_THROW_IF_NOT_MSG(!dims.empty(), "need at least one modality");

    // Use dimension of first modality as primary
    d = dims[0];

    // Initialize storage
    modality_anchors.resize(n_modalities);
    modality_profiles.resize(n_modalities);

    // Initialize alignment matrices (identity-ish)
    alignment_matrices.resize(n_modalities * n_modalities);
    for (int i = 0; i < n_modalities; i++) {
        for (int j = 0; j < n_modalities; j++) {
            int idx = i * n_modalities + j;
            alignment_matrices[idx].resize(n_anchors * n_anchors);

            // Initialize as scaled identity
            std::fill(
                    alignment_matrices[idx].begin(),
                    alignment_matrices[idx].end(),
                    0.0f);
            for (int k = 0; k < n_anchors; k++) {
                alignment_matrices[idx][k * n_anchors + k] = 1.0f;
            }
        }
    }
}

IndexNeuroCrossModal::IndexNeuroCrossModal(int d1, int d2, int n_anchors)
        : IndexNeuroCrossModal(std::vector<int>{d1, d2}, n_anchors) {}

void IndexNeuroCrossModal::select_anchors(
        int modality,
        idx_t n,
        const float* x) {
    FAISS_THROW_IF_NOT(modality >= 0 && modality < n_modalities);

    int dim = modality_dims[modality];
    modality_anchors[modality].resize(n_anchors * dim);

    if (n == 0)
        return;

    std::vector<float> min_dists(n, std::numeric_limits<float>::max());
    std::vector<bool> selected(n, false);

    // Start with random point
    std::mt19937 rng(42 + modality);
    std::uniform_int_distribution<idx_t> dist(0, n - 1);
    idx_t first = dist(rng);
    selected[first] = true;
    std::copy(
            x + first * dim,
            x + (first + 1) * dim,
            modality_anchors[modality].data());

    // Update distances to first anchor
    for (idx_t i = 0; i < n; i++) {
        if (!selected[i]) {
            min_dists[i] =
                    fvec_L2sqr(x + i * dim, modality_anchors[modality].data(), dim);
        }
    }

    // Select remaining anchors by farthest point sampling
    for (int a = 1; a < n_anchors; a++) {
        idx_t farthest = 0;
        float max_dist = -1.0f;
        for (idx_t i = 0; i < n; i++) {
            if (!selected[i] && min_dists[i] > max_dist) {
                max_dist = min_dists[i];
                farthest = i;
            }
        }

        selected[farthest] = true;
        std::copy(
                x + farthest * dim,
                x + (farthest + 1) * dim,
                modality_anchors[modality].data() + a * dim);

        // Update distances
        for (idx_t i = 0; i < n; i++) {
            if (!selected[i]) {
                float dist = fvec_L2sqr(
                        x + i * dim,
                        modality_anchors[modality].data() + a * dim,
                        dim);
                min_dists[i] = std::min(min_dists[i], dist);
            }
        }
    }
}

void IndexNeuroCrossModal::train_modality(int modality, idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(modality >= 0 && modality < n_modalities);
    select_anchors(modality, n, x);
}

void IndexNeuroCrossModal::train(idx_t n, const float* x) {
    // Train first modality with provided data
    train_modality(0, n, x);
    is_trained = true;
}

void IndexNeuroCrossModal::compute_profile(
        int modality,
        const float* x,
        float* profile) const {
    FAISS_THROW_IF_NOT(modality >= 0 && modality < n_modalities);
    FAISS_THROW_IF_NOT(!modality_anchors[modality].empty());

    int dim = modality_dims[modality];
    const float* anchors = modality_anchors[modality].data();

    for (int a = 0; a < n_anchors; a++) {
        profile[a] = std::sqrt(fvec_L2sqr(x, anchors + a * dim, dim));
    }
}

void IndexNeuroCrossModal::map_profile(
        int from_modality,
        int to_modality,
        const float* src_profile,
        float* dst_profile) const {
    FAISS_THROW_IF_NOT(from_modality >= 0 && from_modality < n_modalities);
    FAISS_THROW_IF_NOT(to_modality >= 0 && to_modality < n_modalities);

    int idx = from_modality * n_modalities + to_modality;
    const float* A = alignment_matrices[idx].data();

    // dst = A * src (matrix-vector multiply)
    for (int i = 0; i < n_anchors; i++) {
        dst_profile[i] = 0.0f;
        for (int j = 0; j < n_anchors; j++) {
            dst_profile[i] += A[i * n_anchors + j] * src_profile[j];
        }
    }
}

float IndexNeuroCrossModal::profile_distance(
        const float* p1,
        const float* p2) const {
    float dist = 0.0f;
    for (int i = 0; i < n_anchors; i++) {
        float diff = p1[i] - p2[i];
        dist += diff * diff;
    }
    return dist;
}

void IndexNeuroCrossModal::add_modality(int modality, idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(modality >= 0 && modality < n_modalities);
    FAISS_THROW_IF_NOT(!modality_anchors[modality].empty());

    int dim = modality_dims[modality];

    // Store vectors
    size_t old_size = stored_vectors.size();
    size_t old_count = vector_modalities.size();

    // For cross-modal, we use a flat storage with modality tags
    // Pad shorter modalities to max dim for storage
    int max_dim = *std::max_element(modality_dims.begin(), modality_dims.end());

    stored_vectors.resize(old_size + n * max_dim, 0.0f);
    for (idx_t i = 0; i < n; i++) {
        std::copy(
                x + i * dim,
                x + (i + 1) * dim,
                stored_vectors.data() + old_size + i * max_dim);
    }

    // Record modality
    vector_modalities.resize(old_count + n, modality);

    // Compute and store profiles
    size_t old_prof_size = modality_profiles[modality].size();
    modality_profiles[modality].resize(old_prof_size + n * n_anchors);

    for (idx_t i = 0; i < n; i++) {
        compute_profile(
                modality,
                x + i * dim,
                modality_profiles[modality].data() + old_prof_size + i * n_anchors);
    }

    ntotal += n;
}

void IndexNeuroCrossModal::add(idx_t n, const float* x) {
    // Add to primary modality
    add_modality(0, n, x);
}

void IndexNeuroCrossModal::reset() {
    stored_vectors.clear();
    vector_modalities.clear();
    for (auto& profiles : modality_profiles) {
        profiles.clear();
    }
    ntotal = 0;
}

void IndexNeuroCrossModal::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);

    int max_dim = *std::max_element(modality_dims.begin(), modality_dims.end());
    int modality = vector_modalities[key];
    int dim = modality_dims[modality];

    std::copy(
            stored_vectors.data() + key * max_dim,
            stored_vectors.data() + key * max_dim + dim,
            recons);
}

void IndexNeuroCrossModal::train_alignment(
        idx_t n,
        const float* x1,
        const float* x2,
        int n_iter) {
    FAISS_THROW_IF_NOT(n_modalities >= 2);
    FAISS_THROW_IF_NOT(!modality_anchors[0].empty());
    FAISS_THROW_IF_NOT(!modality_anchors[1].empty());

    // Train alignment from modality 0 to modality 1
    int idx_01 = 0 * n_modalities + 1;
    int idx_10 = 1 * n_modalities + 0;

    std::vector<float> prof1(n_anchors);
    std::vector<float> prof2(n_anchors);
    std::vector<float> mapped(n_anchors);

    for (int iter = 0; iter < n_iter; iter++) {
        float total_loss = 0.0f;

        for (idx_t i = 0; i < n; i++) {
            // Compute profiles
            compute_profile(0, x1 + i * modality_dims[0], prof1.data());
            compute_profile(1, x2 + i * modality_dims[1], prof2.data());

            // Map profile 1 to profile 2 space
            map_profile(0, 1, prof1.data(), mapped.data());

            // Compute gradient and update alignment matrix
            // Loss = ||mapped - prof2||^2
            // grad_A[i][j] = 2 * (mapped[i] - prof2[i]) * prof1[j]
            float* A = alignment_matrices[idx_01].data();
            for (int i = 0; i < n_anchors; i++) {
                float error = mapped[i] - prof2[i];
                total_loss += error * error;
                for (int j = 0; j < n_anchors; j++) {
                    A[i * n_anchors + j] -=
                            alignment_lr * 2.0f * error * prof1[j] / n;
                }
            }

            // Also train reverse alignment
            map_profile(1, 0, prof2.data(), mapped.data());
            float* A_rev = alignment_matrices[idx_10].data();
            for (int i = 0; i < n_anchors; i++) {
                float error = mapped[i] - prof1[i];
                for (int j = 0; j < n_anchors; j++) {
                    A_rev[i * n_anchors + j] -=
                            alignment_lr * 2.0f * error * prof2[j] / n;
                }
            }
        }
    }
}

void IndexNeuroCrossModal::set_search_modalities(int query, int target) {
    FAISS_THROW_IF_NOT(query >= 0 && query < n_modalities);
    FAISS_THROW_IF_NOT(target >= -1 && target < n_modalities);
    query_modality = query;
    target_modality = target;
}

void IndexNeuroCrossModal::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Resolve parameters
    int q_mod = query_modality;
    int t_mod = target_modality;
    int n_cand = n_candidates;
    bool do_rerank = rerank;

    auto cmp = dynamic_cast<const NeuroCrossModalParams*>(params);
    if (cmp) {
        q_mod = cmp->query_modality;
        t_mod = cmp->target_modality;
        if (cmp->candidates > 0) {
            n_cand = cmp->candidates;
        }
        do_rerank = cmp->rerank;
    }

    bool collect = false;
    if (params) {
        auto nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;
    int max_dim = *std::max_element(modality_dims.begin(), modality_dims.end());

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * modality_dims[q_mod];

        // Compute query profile
        std::vector<float> query_profile(n_anchors);
        compute_profile(q_mod, query, query_profile.data());

        // Score all vectors
        std::vector<std::pair<float, idx_t>> scored;
        scored.reserve(ntotal);

        idx_t vec_idx = 0;
        for (int mod = 0; mod < n_modalities; mod++) {
            // Skip if not target modality
            if (t_mod >= 0 && mod != t_mod) {
                vec_idx += modality_profiles[mod].size() / n_anchors;
                continue;
            }

            // Map query profile to this modality's space
            std::vector<float> mapped_profile(n_anchors);
            if (mod == q_mod) {
                std::copy(
                        query_profile.begin(),
                        query_profile.end(),
                        mapped_profile.begin());
            } else {
                map_profile(
                        q_mod, mod, query_profile.data(), mapped_profile.data());
            }

            // Score vectors in this modality
            idx_t n_vecs = modality_profiles[mod].size() / n_anchors;
            for (idx_t i = 0; i < n_vecs; i++) {
                const float* vec_profile =
                        modality_profiles[mod].data() + i * n_anchors;
                float dist = profile_distance(mapped_profile.data(), vec_profile);
                scored.emplace_back(dist, vec_idx + i);
            }
            vec_idx += n_vecs;
        }

        if (scored.empty()) {
            for (idx_t i = 0; i < k; i++) {
                distances[q * k + i] = std::numeric_limits<float>::max();
                labels[q * k + i] = -1;
            }
            continue;
        }

        // Get top candidates
        size_t actual_cand =
                std::min(static_cast<size_t>(n_cand), scored.size());
        std::partial_sort(
                scored.begin(), scored.begin() + actual_cand, scored.end());

        // Rerank with true distance if requested
        if (do_rerank) {
            std::vector<float> recons(max_dim);
            for (size_t i = 0; i < actual_cand; i++) {
                idx_t idx = scored[i].second;
                int vec_mod = vector_modalities[idx];
                int vec_dim = modality_dims[vec_mod];

                // Only rerank if same modality or cross-modal distance defined
                if (vec_mod == q_mod) {
                    reconstruct(idx, recons.data());
                    float dist;
                    if (metric) {
                        dist = metric->distance(query, recons.data(), vec_dim);
                    } else {
                        dist = fvec_L2sqr(query, recons.data(), vec_dim);
                    }
                    scored[i].first = dist;
                    total_calcs += vec_dim;
                }
            }

            std::partial_sort(
                    scored.begin(),
                    scored.begin() +
                            std::min(actual_cand, static_cast<size_t>(k)),
                    scored.begin() + actual_cand);
        }

        // Output results
        size_t actual_k = std::min(static_cast<size_t>(k), actual_cand);
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = scored[i].first;
            labels[q * k + i] = scored[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
    }
}

} // namespace faiss
