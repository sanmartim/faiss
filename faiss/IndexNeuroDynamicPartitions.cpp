/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroDynamicPartitions.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace faiss {

IndexNeuroDynamicPartitions::IndexNeuroDynamicPartitions(int d, int npartitions)
        : IndexNeuro(nullptr, false), npartitions(npartitions) {
    this->d = d;
    query_counts.resize(npartitions, 0);
    partition_vectors.resize(npartitions);
    partition_ids.resize(npartitions);
    is_trained = false;
}

int IndexNeuroDynamicPartitions::find_partition(const float* vec) const {
    int best = 0;
    float best_dist = std::numeric_limits<float>::max();
    for (int p = 0; p < npartitions; p++) {
        float dist = fvec_L2sqr(vec, centroids.data() + p * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best = p;
        }
    }
    return best;
}

void IndexNeuroDynamicPartitions::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= npartitions, "need at least npartitions vectors");

    Clustering clus(d, npartitions);
    clus.verbose = false;
    clus.niter = 20;

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    centroids.resize(npartitions * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + npartitions * d,
            centroids.data());

    is_trained = true;
}

void IndexNeuroDynamicPartitions::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int partition = find_partition(vec);
        idx_t vec_id = ntotal + i;

        partition_vectors[partition].insert(
                partition_vectors[partition].end(),
                vec,
                vec + d);
        partition_ids[partition].push_back(vec_id);
    }

    // Store all vectors
    size_t old_size = all_vectors.size();
    all_vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, all_vectors.data() + old_size);

    ntotal += n;
}

void IndexNeuroDynamicPartitions::reset() {
    for (auto& v : partition_vectors) v.clear();
    for (auto& v : partition_ids) v.clear();
    all_vectors.clear();
    std::fill(query_counts.begin(), query_counts.end(), 0);
    total_queries = 0;
    ntotal = 0;
}

void IndexNeuroDynamicPartitions::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            all_vectors.data() + key * d,
            all_vectors.data() + (key + 1) * d,
            recons);
}

std::vector<int64_t> IndexNeuroDynamicPartitions::get_partition_sizes() const {
    std::vector<int64_t> sizes(npartitions);
    for (int p = 0; p < npartitions; p++) {
        sizes[p] = partition_ids[p].size();
    }
    return sizes;
}

void IndexNeuroDynamicPartitions::rebalance() {
    if (ntotal == 0) return;

    // Recompute centroids based on current vectors with query-weighted k-means
    // For simplicity, we use standard k-means on all vectors
    Clustering clus(d, npartitions);
    clus.verbose = false;
    clus.niter = 10;

    IndexFlatL2 quantizer(d);
    clus.train(ntotal, all_vectors.data(), quantizer);

    centroids.resize(npartitions * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + npartitions * d,
            centroids.data());

    // Reassign all vectors
    for (auto& v : partition_vectors) v.clear();
    for (auto& v : partition_ids) v.clear();

    for (idx_t i = 0; i < ntotal; i++) {
        const float* vec = all_vectors.data() + i * d;
        int partition = find_partition(vec);
        partition_vectors[partition].insert(
                partition_vectors[partition].end(),
                vec,
                vec + d);
        partition_ids[partition].push_back(i);
    }

    // Reset query counts
    std::fill(query_counts.begin(), query_counts.end(), 0);
    total_queries = 0;
}

void IndexNeuroDynamicPartitions::maybe_rebalance() const {
    if (!enable_rebalance) return;
    if (total_queries >= rebalance_interval) {
        // Cast away const for rebalancing (lazy update pattern)
        const_cast<IndexNeuroDynamicPartitions*>(this)->rebalance();
    }
}

void IndexNeuroDynamicPartitions::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    int np = nprobe;
    auto dp = dynamic_cast<const NeuroDynamicPartParams*>(params);
    if (dp && dp->nprobe > 0) {
        np = dp->nprobe;
    }
    np = std::min(np, npartitions);

    // Check if rebalancing is needed
    maybe_rebalance();

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Find nearest partitions
        std::vector<std::pair<float, int>> partition_dists(npartitions);
        for (int p = 0; p < npartitions; p++) {
            float dist = fvec_L2sqr(query, centroids.data() + p * d, d);
            partition_dists[p] = {dist, p};
        }
        std::partial_sort(
                partition_dists.begin(),
                partition_dists.begin() + np,
                partition_dists.end());

        // Update query counts
        for (int i = 0; i < np; i++) {
            query_counts[partition_dists[i].second]++;
        }
        total_queries++;

        // Search probed partitions
        std::vector<std::pair<float, idx_t>> candidates;
        for (int i = 0; i < np; i++) {
            int p = partition_dists[i].second;
            size_t nvecs = partition_ids[p].size();
            for (size_t j = 0; j < nvecs; j++) {
                const float* vec = partition_vectors[p].data() + j * d;
                float dist = fvec_L2sqr(query, vec, d);
                candidates.push_back({dist, partition_ids[p][j]});
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
