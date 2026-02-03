/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroCentroidBounds.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss {

IndexNeuroCentroidBounds::IndexNeuroCentroidBounds(int d, int nlist)
        : IndexNeuro(nullptr, false), nlist(nlist) {
    this->d = d;
    cluster_ids.resize(nlist);
    radii.resize(nlist, 0.0f);
    is_trained = false;
}

int IndexNeuroCentroidBounds::find_nearest_cluster(const float* vec) const {
    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int c = 0; c < nlist; c++) {
        float dist = fvec_L2sqr(vec, centroids.data() + c * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best = c;
        }
    }
    return best;
}

void IndexNeuroCentroidBounds::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT_MSG(n >= nlist, "need at least nlist vectors");

    // Run k-means clustering
    Clustering clus(d, nlist);
    clus.verbose = false;
    clus.niter = 20;

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    // Copy centroids
    centroids.resize(nlist * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + nlist * d,
            centroids.data());

    // Clear cluster data
    for (auto& ids : cluster_ids) {
        ids.clear();
    }
    std::fill(radii.begin(), radii.end(), 0.0f);
    vectors.clear();

    // Assign vectors to clusters and compute radii
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int cluster = find_nearest_cluster(vec);
        cluster_ids[cluster].push_back(i);

        float dist = fvec_L2sqr(vec, centroids.data() + cluster * d, d);
        radii[cluster] = std::max(radii[cluster], dist);
    }

    // Convert radii to actual distances (sqrt)
    for (int c = 0; c < nlist; c++) {
        radii[c] = std::sqrt(radii[c]);
    }

    // Store original vectors
    vectors.resize(n * d);
    std::copy(x, x + n * d, vectors.data());

    ntotal = n;
    is_trained = true;
}

void IndexNeuroCentroidBounds::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained first");

    idx_t old_ntotal = ntotal;

    // Extend vectors storage
    vectors.resize((old_ntotal + n) * d);
    std::copy(x, x + n * d, vectors.data() + old_ntotal * d);

    // Assign to clusters
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int cluster = find_nearest_cluster(vec);
        idx_t vec_id = old_ntotal + i;
        cluster_ids[cluster].push_back(vec_id);

        float dist = std::sqrt(fvec_L2sqr(vec, centroids.data() + cluster * d, d));
        radii[cluster] = std::max(radii[cluster], dist);
    }

    ntotal += n;
}

void IndexNeuroCentroidBounds::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    // Parse parameters
    int np = nprobe;
    bool use_triangle = use_triangle_inequality;

    auto* cbp = dynamic_cast<const NeuroCentroidBoundsParams*>(params);
    if (cbp) {
        np = cbp->nprobe;
        use_triangle = cbp->use_triangle_inequality;
    }
    np = std::min(np, nlist);

    bool collect = false;
    if (params) {
        auto* nsp = dynamic_cast<const NeuroSearchParameters*>(params);
        if (nsp) {
            collect = nsp->collect_stats;
        }
    }

    int64_t total_calcs = 0;

#pragma omp parallel for reduction(+ : total_calcs)
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Compute distances to all centroids
        std::vector<std::pair<float, int>> centroid_dists(nlist);
        for (int c = 0; c < nlist; c++) {
            float dist = std::sqrt(fvec_L2sqr(query, centroids.data() + c * d, d));
            centroid_dists[c] = {dist, c};
        }
        total_calcs += nlist * d;

        // Sort centroids by distance
        std::sort(centroid_dists.begin(), centroid_dists.end());

        // Search with triangle inequality pruning
        std::vector<std::pair<float, idx_t>> candidates;
        float best_so_far = std::numeric_limits<float>::max();
        int clusters_searched = 0;

        for (int i = 0; i < nlist && clusters_searched < np; i++) {
            int c = centroid_dists[i].second;
            float centroid_dist = centroid_dists[i].first;

            // Triangle inequality pruning
            if (use_triangle && centroid_dist - radii[c] > best_so_far) {
                continue;
            }

            clusters_searched++;

            // Search this cluster
            for (idx_t vec_id : cluster_ids[c]) {
                const float* vec = vectors.data() + vec_id * d;
                float dist = fvec_L2sqr(query, vec, d);
                candidates.push_back({dist, vec_id});

                if (dist < best_so_far * best_so_far) {
                    best_so_far = std::sqrt(dist);
                }
            }
            total_calcs += cluster_ids[c].size() * d;
        }

        // Sort and output top-k
        size_t actual_k = std::min(static_cast<size_t>(k), candidates.size());
        std::partial_sort(
                candidates.begin(),
                candidates.begin() + actual_k,
                candidates.end());

        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = candidates[i].first;
            labels[q * k + i] = candidates[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }

    if (collect) {
        last_stats.calculations_performed = total_calcs;
        last_stats.columns_used = d;
    }
}

void IndexNeuroCentroidBounds::reset() {
    for (auto& ids : cluster_ids) {
        ids.clear();
    }
    std::fill(radii.begin(), radii.end(), 0.0f);
    vectors.clear();
    ntotal = 0;
}

} // namespace faiss
