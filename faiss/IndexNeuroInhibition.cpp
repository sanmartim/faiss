/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroInhibition.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace faiss {

IndexNeuroInhibition::IndexNeuroInhibition(Index* sub_index, bool own)
        : Index(sub_index->d, sub_index->metric_type),
          sub_index(sub_index),
          own_fields(own) {
    ntotal = sub_index->ntotal;
    is_trained = sub_index->is_trained;
}

IndexNeuroInhibition::~IndexNeuroInhibition() {
    if (own_fields) {
        delete sub_index;
    }
}

void IndexNeuroInhibition::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
    is_trained = sub_index->is_trained;
}

void IndexNeuroInhibition::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
}

void IndexNeuroInhibition::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reconstruct(key, recons);
}

namespace {

/// Apply lateral inhibition to one query's expanded results.
/// Greedy selection: iterate candidates sorted by distance to query,
/// group by pairwise L2 similarity, keep at most max_per_cluster
/// from each group.
void inhibit_one_query(
        const float* expanded_dists,
        const idx_t* expanded_labels,
        int n_expanded,
        idx_t k,
        float* out_distances,
        idx_t* out_labels,
        float similarity_threshold,
        int max_per_cluster,
        const Index* sub_index,
        int d) {
    // Each cluster is represented by its first member's reconstructed vector.
    // cluster_counts[i] = how many selected from cluster i so far.
    // cluster_reps[i] = reconstructed vector of cluster representative.
    struct Cluster {
        std::vector<float> rep; // representative vector (d floats)
        int count = 0;
    };

    std::vector<Cluster> clusters;
    std::vector<float> selected_dists;
    std::vector<idx_t> selected_labels;

    // Pre-reconstruct all expanded candidate vectors
    std::vector<std::vector<float>> cand_vecs(n_expanded);
    for (int i = 0; i < n_expanded; i++) {
        if (expanded_labels[i] < 0) continue;
        cand_vecs[i].resize(d);
        sub_index->reconstruct(expanded_labels[i], cand_vecs[i].data());
    }

    // Track which candidates were inhibited (for fallback)
    std::vector<int> inhibited_indices;

    // Greedy selection (candidates already sorted by distance to query)
    for (int i = 0; i < n_expanded; i++) {
        if (expanded_labels[i] < 0) continue;
        if (cand_vecs[i].empty()) continue;

        // Find which cluster this candidate belongs to (if any)
        int assigned_cluster = -1;
        for (size_t c = 0; c < clusters.size(); c++) {
            float dist = fvec_L2sqr(
                    cand_vecs[i].data(), clusters[c].rep.data(), d);
            if (dist < similarity_threshold) {
                assigned_cluster = static_cast<int>(c);
                break;
            }
        }

        if (assigned_cluster >= 0) {
            // Belongs to an existing cluster
            if (clusters[assigned_cluster].count < max_per_cluster) {
                clusters[assigned_cluster].count++;
                selected_dists.push_back(expanded_dists[i]);
                selected_labels.push_back(expanded_labels[i]);
            } else {
                // Cluster full, inhibited — save for fallback
                inhibited_indices.push_back(i);
            }
        } else {
            // New cluster
            Cluster nc;
            nc.rep = cand_vecs[i];
            nc.count = 1;
            clusters.push_back(std::move(nc));
            selected_dists.push_back(expanded_dists[i]);
            selected_labels.push_back(expanded_labels[i]);
        }

        // Early exit if we have enough
        if (static_cast<idx_t>(selected_labels.size()) >= k) {
            break;
        }
    }

    // Fallback: if inhibition was too aggressive, fill remaining
    // slots from inhibited candidates (in original distance order)
    for (size_t j = 0;
         j < inhibited_indices.size() &&
         static_cast<idx_t>(selected_labels.size()) < k;
         j++) {
        int idx = inhibited_indices[j];
        selected_dists.push_back(expanded_dists[idx]);
        selected_labels.push_back(expanded_labels[idx]);
    }

    // Fill output
    idx_t actual_k =
            std::min(k, static_cast<idx_t>(selected_labels.size()));
    for (idx_t i = 0; i < actual_k; i++) {
        out_distances[i] = selected_dists[i];
        out_labels[i] = selected_labels[i];
    }
    for (idx_t i = actual_k; i < k; i++) {
        out_distances[i] = std::numeric_limits<float>::max();
        out_labels[i] = -1;
    }
}

} // anonymous namespace

void IndexNeuroInhibition::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Request expanded candidate set
    idx_t k_expanded = std::min(
            static_cast<idx_t>(std::ceil(k * k_expansion)),
            sub_index->ntotal);
    k_expanded = std::max(k_expanded, k);

    std::vector<float> exp_dists(n * k_expanded);
    std::vector<idx_t> exp_labels(n * k_expanded);
    sub_index->search(
            n, x, k_expanded, exp_dists.data(), exp_labels.data(), params);

    // Apply lateral inhibition per query
#pragma omp parallel for if (n > 1)
    for (idx_t i = 0; i < n; i++) {
        inhibit_one_query(
                exp_dists.data() + i * k_expanded,
                exp_labels.data() + i * k_expanded,
                static_cast<int>(k_expanded),
                k,
                distances + i * k,
                labels + i * k,
                similarity_threshold,
                max_per_cluster,
                sub_index,
                d);
    }
}

} // namespace faiss
