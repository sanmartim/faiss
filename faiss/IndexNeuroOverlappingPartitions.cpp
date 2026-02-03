/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroOverlappingPartitions.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace faiss {

IndexNeuroOverlappingPartitions::IndexNeuroOverlappingPartitions(
        int d,
        int npartitions,
        float overlap_ratio)
        : IndexNeuro(nullptr, false),
          npartitions(npartitions),
          overlap_ratio(overlap_ratio) {
    this->d = d;
    partition_lists.resize(npartitions);
    is_trained = false;
}

void IndexNeuroOverlappingPartitions::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= npartitions, "need at least npartitions vectors to train");

    // Train centroids using k-means
    Clustering clus(d, npartitions);
    clus.verbose = false;
    clus.niter = 20;

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    // Store centroids
    centroids.resize(npartitions * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + npartitions * d,
            centroids.data());

    is_trained = true;
}

void IndexNeuroOverlappingPartitions::find_partitions(
        const float* vec,
        int n_partitions,
        std::vector<int>& partition_ids) const {
    // Compute distances to all centroids
    std::vector<std::pair<float, int>> scored(npartitions);
    for (int i = 0; i < npartitions; i++) {
        float dist = fvec_L2sqr(vec, centroids.data() + i * d, d);
        scored[i] = {dist, i};
    }

    // Find nearest partitions
    std::partial_sort(
            scored.begin(),
            scored.begin() + n_partitions,
            scored.end());

    partition_ids.clear();
    for (int i = 0; i < n_partitions; i++) {
        partition_ids.push_back(scored[i].second);
    }
}

void IndexNeuroOverlappingPartitions::assign_to_partitions(idx_t vec_id, const float* vec) {
    // Compute distances to all centroids
    std::vector<std::pair<float, int>> scored(npartitions);
    for (int i = 0; i < npartitions; i++) {
        float dist = fvec_L2sqr(vec, centroids.data() + i * d, d);
        scored[i] = {dist, i};
    }
    std::sort(scored.begin(), scored.end());

    // Assign to nearest partition
    int primary = scored[0].second;
    partition_lists[primary].push_back(vec_id);

    // Assign to overlapping partitions based on distance ratio
    float primary_dist = scored[0].first;
    for (int i = 1; i < npartitions; i++) {
        float ratio = (scored[i].first - primary_dist) / (primary_dist + 1e-10f);
        if (ratio < overlap_ratio) {
            partition_lists[scored[i].second].push_back(vec_id);
        } else {
            break;  // Partitions are sorted by distance
        }
    }
}

void IndexNeuroOverlappingPartitions::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained before adding");

    // Store vectors
    size_t old_size = vectors.size();
    vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, vectors.data() + old_size);

    // Assign each vector to partitions
    for (idx_t i = 0; i < n; i++) {
        assign_to_partitions(ntotal + i, x + i * d);
    }

    ntotal += n;
}

void IndexNeuroOverlappingPartitions::reset() {
    vectors.clear();
    for (auto& list : partition_lists) {
        list.clear();
    }
    ntotal = 0;
}

void IndexNeuroOverlappingPartitions::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            vectors.data() + key * d,
            vectors.data() + (key + 1) * d,
            recons);
}

void IndexNeuroOverlappingPartitions::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Resolve parameters
    int np = nprobe;
    auto op = dynamic_cast<const NeuroOverlapParams*>(params);
    if (op && op->nprobe > 0) {
        np = op->nprobe;
    }
    np = std::min(np, npartitions);

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Find nearest partitions
        std::vector<int> partition_ids;
        find_partitions(query, np, partition_ids);

        // Collect candidates from all probed partitions
        std::unordered_set<idx_t> seen;
        std::vector<std::pair<float, idx_t>> candidates;

        for (int pid : partition_ids) {
            for (idx_t vec_id : partition_lists[pid]) {
                if (seen.find(vec_id) == seen.end()) {
                    seen.insert(vec_id);
                    const float* vec = vectors.data() + vec_id * d;
                    float dist;
                    if (metric) {
                        dist = metric->distance(query, vec, d);
                    } else {
                        dist = fvec_L2sqr(query, vec, d);
                    }
                    candidates.push_back({dist, vec_id});
                }
            }
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
}

} // namespace faiss
