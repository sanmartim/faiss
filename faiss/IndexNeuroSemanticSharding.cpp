/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroSemanticSharding.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroSemanticSharding::IndexNeuroSemanticSharding(int d, int nshards)
        : IndexNeuro(nullptr, false), nshards(nshards) {
    this->d = d;
    shard_vectors.resize(nshards);
    shard_ids.resize(nshards);
    shard_sizes.resize(nshards, 0);
    is_trained = false;
}

int IndexNeuroSemanticSharding::find_shard(const float* vec) const {
    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int s = 0; s < nshards; s++) {
        float dist = fvec_L2sqr(vec, shard_centroids.data() + s * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best = s;
        }
    }
    return best;
}

void IndexNeuroSemanticSharding::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= nshards, "need at least nshards vectors");

    // Use k-means to find semantic clusters
    Clustering clus(d, nshards);
    clus.verbose = false;
    clus.niter = 25;  // More iterations for better semantic grouping

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    shard_centroids.resize(nshards * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + nshards * d,
            shard_centroids.data());

    is_trained = true;
}

void IndexNeuroSemanticSharding::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int shard = find_shard(vec);
        idx_t vec_id = ntotal + i;

        shard_vectors[shard].insert(
                shard_vectors[shard].end(),
                vec,
                vec + d);
        shard_ids[shard].push_back(vec_id);
        shard_sizes[shard]++;
    }

    // Store all vectors
    size_t old_size = all_vectors.size();
    all_vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, all_vectors.data() + old_size);

    ntotal += n;
}

void IndexNeuroSemanticSharding::reset() {
    for (auto& v : shard_vectors) v.clear();
    for (auto& v : shard_ids) v.clear();
    all_vectors.clear();
    std::fill(shard_sizes.begin(), shard_sizes.end(), 0);
    ntotal = 0;
}

void IndexNeuroSemanticSharding::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            all_vectors.data() + key * d,
            all_vectors.data() + (key + 1) * d,
            recons);
}

std::vector<size_t> IndexNeuroSemanticSharding::get_shard_sizes() const {
    return shard_sizes;
}

void IndexNeuroSemanticSharding::rebalance_shards() {
    if (ntotal == 0) {
        return;
    }

    // Retrain centroids on current data
    Clustering clus(d, nshards);
    clus.verbose = false;
    clus.niter = 15;

    IndexFlatL2 quantizer(d);
    clus.train(ntotal, all_vectors.data(), quantizer);

    shard_centroids.resize(nshards * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + nshards * d,
            shard_centroids.data());

    // Reassign vectors to shards
    for (auto& v : shard_vectors) v.clear();
    for (auto& v : shard_ids) v.clear();
    std::fill(shard_sizes.begin(), shard_sizes.end(), 0);

    for (idx_t i = 0; i < ntotal; i++) {
        const float* vec = all_vectors.data() + i * d;
        int shard = find_shard(vec);

        shard_vectors[shard].insert(
                shard_vectors[shard].end(),
                vec,
                vec + d);
        shard_ids[shard].push_back(i);
        shard_sizes[shard]++;
    }
}

void IndexNeuroSemanticSharding::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    int np = nprobe;
    float et_ratio = early_termination_ratio;

    auto sp = dynamic_cast<const NeuroSemanticShardParams*>(params);
    if (sp) {
        if (sp->nprobe > 0) {
            np = sp->nprobe;
        }
        if (sp->early_termination_ratio > 0) {
            et_ratio = sp->early_termination_ratio;
        }
    }
    np = std::min(np, nshards);

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Find nearest shards
        std::vector<std::pair<float, int>> shard_dists(nshards);
        for (int s = 0; s < nshards; s++) {
            float dist = fvec_L2sqr(query, shard_centroids.data() + s * d, d);
            shard_dists[s] = {dist, s};
        }
        std::sort(shard_dists.begin(), shard_dists.end());

        // Search shards with early termination
        std::vector<std::pair<float, idx_t>> candidates;
        float best_dist_so_far = std::numeric_limits<float>::max();

        for (int i = 0; i < np; i++) {
            float shard_dist = shard_dists[i].first;

            // Early termination: if shard centroid is much farther than best result
            if (candidates.size() >= static_cast<size_t>(k) &&
                shard_dist > best_dist_so_far * et_ratio) {
                break;
            }

            int s = shard_dists[i].second;
            size_t nvecs = shard_ids[s].size();

            for (size_t j = 0; j < nvecs; j++) {
                const float* vec = shard_vectors[s].data() + j * d;
                float dist = fvec_L2sqr(query, vec, d);
                candidates.push_back({dist, shard_ids[s][j]});

                if (dist < best_dist_so_far) {
                    best_dist_so_far = dist;
                }
            }
        }

        // Sort and output
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
}

} // namespace faiss
