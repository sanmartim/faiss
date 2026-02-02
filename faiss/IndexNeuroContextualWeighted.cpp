/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroContextualWeighted.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace faiss {

IndexNeuroContextualWeighted::IndexNeuroContextualWeighted(
        Index* inner,
        int n_clusters,
        bool own_inner)
        : IndexNeuro(inner, own_inner), n_query_clusters(n_clusters) {}

int IndexNeuroContextualWeighted::classify_query(const float* query) const {
    FAISS_THROW_IF_NOT_MSG(
            !centroids.empty(), "train() must be called first");

    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int c = 0; c < n_query_clusters; c++) {
        float dist = fvec_L2sqr(query, centroids.data() + c * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    return best;
}

void IndexNeuroContextualWeighted::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n > 0, "need at least 1 training vector");
    FAISS_THROW_IF_NOT_MSG(d > 0, "dimension must be positive");

    int nc = std::min((int)n, n_query_clusters);

    // Simple k-means: initialize centroids from first nc samples
    centroids.resize(nc * d);
    for (int c = 0; c < nc; c++) {
        std::copy(x + c * d, x + (c + 1) * d, centroids.data() + c * d);
    }

    // Run k-means iterations
    std::vector<int> assignments(n);
    for (int iter = 0; iter < 20; iter++) {
        // Assign
        for (idx_t i = 0; i < n; i++) {
            int best = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (int c = 0; c < nc; c++) {
                float dist = fvec_L2sqr(x + i * d, centroids.data() + c * d, d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            assignments[i] = best;
        }

        // Update centroids
        std::vector<float> new_centroids(nc * d, 0.0f);
        std::vector<int> counts(nc, 0);
        for (idx_t i = 0; i < n; i++) {
            int c = assignments[i];
            counts[c]++;
            for (int j = 0; j < d; j++) {
                new_centroids[c * d + j] += x[i * d + j];
            }
        }
        for (int c = 0; c < nc; c++) {
            if (counts[c] > 0) {
                for (int j = 0; j < d; j++) {
                    new_centroids[c * d + j] /= counts[c];
                }
            }
        }
        centroids = new_centroids;
    }

    // Update n_query_clusters if we had fewer samples
    n_query_clusters = nc;

    // Initialize per-cluster weights to uniform
    cluster_weights.assign(nc * d, 1.0f);
    feedback_counts.assign(nc, 0);
    is_trained = true;
}

void IndexNeuroContextualWeighted::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT_MSG(inner_index, "inner_index is null");
    FAISS_THROW_IF_NOT_MSG(!centroids.empty(), "train() must be called first");

    auto* flat = dynamic_cast<IndexFlat*>(inner_index);
    FAISS_THROW_IF_NOT_MSG(flat, "inner_index must be IndexFlat");

    const float* xb = flat->get_xb();
    idx_t nb = inner_index->ntotal;

    const NeuroContextualParams* cp = nullptr;
    if (params_in) {
        cp = dynamic_cast<const NeuroContextualParams*>(params_in);
    }

    bool collect = false;
    if (params_in) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params_in);
        if (nsp)
            collect = nsp->collect_stats;
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Classify query to get weight vector
        int cluster;
        if (cp && cp->force_cluster >= 0 &&
            cp->force_cluster < n_query_clusters) {
            cluster = cp->force_cluster;
        } else {
            cluster = classify_query(query);
        }

        const float* w = cluster_weights.data() + cluster * d;

        // Weighted L2 search
        std::vector<std::pair<float, idx_t>> dists(nb);
        for (idx_t i = 0; i < nb; i++) {
            const float* vec = xb + i * d;
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = query[j] - vec[j];
                dist += w[j] * diff * diff;
            }
            dists[i] = {dist, i};
        }
        total_calcs += nb * d;

        std::partial_sort(
                dists.begin(),
                dists.begin() + std::min((idx_t)dists.size(), k),
                dists.end());

        for (idx_t i = 0; i < k && i < nb; i++) {
            distances[q * k + i] = dists[i].first;
            labels[q * k + i] = dists[i].second;
        }
        for (idx_t i = nb; i < k; i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

void IndexNeuroContextualWeighted::feedback(
        idx_t nq,
        const float* queries,
        const float* positives,
        const float* negatives) {
    FAISS_THROW_IF_NOT_MSG(
            !cluster_weights.empty(), "train() must be called first");

    for (idx_t q = 0; q < nq; q++) {
        const float* qvec = queries + q * d;
        const float* pvec = positives + q * d;
        const float* nvec = negatives + q * d;

        int cluster = classify_query(qvec);
        float* w = cluster_weights.data() + cluster * d;

        // Apply decay
        for (int j = 0; j < d; j++) {
            w[j] *= weight_decay;
        }

        // Hebbian update
        for (int j = 0; j < d; j++) {
            float dp = (qvec[j] - pvec[j]) * (qvec[j] - pvec[j]);
            float dn = (qvec[j] - nvec[j]) * (qvec[j] - nvec[j]);

            if (dp < dn) {
                w[j] += learning_rate;
            } else if (dp > dn) {
                w[j] -= learning_rate;
            }
            w[j] = std::max(w[j], min_weight);
        }

        feedback_counts[cluster]++;
    }
}

} // namespace faiss
