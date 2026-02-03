/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroHierarchicalDisk.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroHierarchicalDisk::IndexNeuroHierarchicalDisk(
        int d,
        int nlist,
        const std::string& disk_path)
        : IndexNeuro(nullptr, false), nlist(nlist), disk_path(disk_path) {
    this->d = d;
    cluster_sizes.resize(nlist, 0);
    cluster_query_counts.resize(nlist, 0);
    cluster_offsets.resize(nlist, 0);
    in_memory.resize(nlist, false);
    memory_clusters.resize(nlist);
    memory_ids.resize(nlist);
    build_clusters.resize(nlist);
    build_ids.resize(nlist);
    is_trained = false;
}

IndexNeuroHierarchicalDisk::~IndexNeuroHierarchicalDisk() {
    // Cleanup handled by RAII
}

int IndexNeuroHierarchicalDisk::find_cluster(const float* vec) const {
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

void IndexNeuroHierarchicalDisk::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= nlist, "need at least nlist vectors");

    Clustering clus(d, nlist);
    clus.verbose = false;
    clus.niter = 20;

    IndexFlatL2 quantizer(d);
    clus.train(n, x, quantizer);

    centroids.resize(nlist * d);
    std::copy(
            quantizer.get_xb(),
            quantizer.get_xb() + nlist * d,
            centroids.data());

    is_trained = true;
}

void IndexNeuroHierarchicalDisk::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int cluster = find_cluster(vec);
        idx_t vec_id = ntotal + i;

        build_clusters[cluster].insert(
                build_clusters[cluster].end(),
                vec,
                vec + d);
        build_ids[cluster].push_back(vec_id);
        cluster_sizes[cluster]++;
    }

    // Also store all vectors
    size_t old_size = all_vectors.size();
    all_vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, all_vectors.data() + old_size);

    for (idx_t i = 0; i < n; i++) {
        all_ids.push_back(ntotal + i);
    }

    ntotal += n;
}

void IndexNeuroHierarchicalDisk::reset() {
    for (auto& v : build_clusters) v.clear();
    for (auto& v : build_ids) v.clear();
    for (auto& v : memory_clusters) v.clear();
    for (auto& v : memory_ids) v.clear();
    all_vectors.clear();
    all_ids.clear();
    std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
    std::fill(cluster_query_counts.begin(), cluster_query_counts.end(), 0);
    std::fill(in_memory.begin(), in_memory.end(), false);
    ntotal = 0;
    is_disk_open = false;
}

void IndexNeuroHierarchicalDisk::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);

    // Search in build_clusters first
    for (int c = 0; c < nlist; c++) {
        const auto& ids = build_ids[c];
        for (size_t i = 0; i < ids.size(); i++) {
            if (ids[i] == key) {
                std::copy(
                        build_clusters[c].data() + i * d,
                        build_clusters[c].data() + (i + 1) * d,
                        recons);
                return;
            }
        }
    }

    // Search in all_vectors
    for (size_t i = 0; i < all_ids.size(); i++) {
        if (all_ids[i] == key) {
            std::copy(
                    all_vectors.data() + i * d,
                    all_vectors.data() + (i + 1) * d,
                    recons);
            return;
        }
    }

    FAISS_THROW_MSG("key not found");
}

void IndexNeuroHierarchicalDisk::save_to_disk() {
    if (disk_path.empty()) {
        return;
    }

    std::ofstream out(disk_path, std::ios::binary);
    FAISS_THROW_IF_NOT_MSG(out.is_open(), "cannot open disk file for writing");

    // Write header
    out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    out.write(reinterpret_cast<const char*>(&nlist), sizeof(nlist));
    out.write(reinterpret_cast<const char*>(&ntotal), sizeof(ntotal));

    // Write centroids
    out.write(
            reinterpret_cast<const char*>(centroids.data()),
            centroids.size() * sizeof(float));

    // Write cluster sizes
    out.write(
            reinterpret_cast<const char*>(cluster_sizes.data()),
            cluster_sizes.size() * sizeof(size_t));

    // Calculate and write cluster offsets
    size_t current_offset = out.tellp();
    current_offset += nlist * sizeof(size_t); // Space for offsets

    for (int c = 0; c < nlist; c++) {
        cluster_offsets[c] = current_offset;
        current_offset += cluster_sizes[c] * (d * sizeof(float) + sizeof(idx_t));
    }

    out.write(
            reinterpret_cast<const char*>(cluster_offsets.data()),
            cluster_offsets.size() * sizeof(size_t));

    // Write cluster data
    for (int c = 0; c < nlist; c++) {
        if (!build_clusters[c].empty()) {
            out.write(
                    reinterpret_cast<const char*>(build_clusters[c].data()),
                    build_clusters[c].size() * sizeof(float));
            out.write(
                    reinterpret_cast<const char*>(build_ids[c].data()),
                    build_ids[c].size() * sizeof(idx_t));
        }
    }

    out.close();
    is_disk_open = true;
}

void IndexNeuroHierarchicalDisk::read_cluster_from_disk(
        int cluster_id,
        std::vector<float>& vectors,
        std::vector<idx_t>& ids) const {
    if (!is_disk_open || disk_path.empty()) {
        // Read from build_clusters
        vectors = build_clusters[cluster_id];
        ids = build_ids[cluster_id];
        return;
    }

    std::ifstream in(disk_path, std::ios::binary);
    if (!in.is_open()) {
        vectors = build_clusters[cluster_id];
        ids = build_ids[cluster_id];
        return;
    }

    size_t nvecs = cluster_sizes[cluster_id];
    if (nvecs == 0) {
        vectors.clear();
        ids.clear();
        return;
    }

    vectors.resize(nvecs * d);
    ids.resize(nvecs);

    in.seekg(cluster_offsets[cluster_id]);
    in.read(
            reinterpret_cast<char*>(vectors.data()),
            nvecs * d * sizeof(float));
    in.read(
            reinterpret_cast<char*>(ids.data()),
            nvecs * sizeof(idx_t));
}

void IndexNeuroHierarchicalDisk::load_hot_clusters() {
    // Determine which clusters are hot based on query counts
    std::vector<std::pair<int64_t, int>> sorted_clusters(nlist);
    for (int c = 0; c < nlist; c++) {
        sorted_clusters[c] = {cluster_query_counts[c], c};
    }
    std::sort(sorted_clusters.rbegin(), sorted_clusters.rend());

    // Load top memory_ratio clusters
    int n_hot = static_cast<int>(nlist * memory_ratio);
    std::fill(in_memory.begin(), in_memory.end(), false);

    for (int i = 0; i < n_hot && i < nlist; i++) {
        int c = sorted_clusters[i].second;
        in_memory[c] = true;
        read_cluster_from_disk(c, memory_clusters[c], memory_ids[c]);
    }
}

void IndexNeuroHierarchicalDisk::update_hot_clusters() {
    load_hot_clusters();
}

void IndexNeuroHierarchicalDisk::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    int np = nprobe;
    auto hp = dynamic_cast<const NeuroHierarchicalDiskParams*>(params);
    if (hp && hp->nprobe > 0) {
        np = hp->nprobe;
    }
    np = std::min(np, nlist);

#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Find nearest clusters
        std::vector<std::pair<float, int>> cluster_dists(nlist);
        for (int c = 0; c < nlist; c++) {
            float dist = fvec_L2sqr(query, centroids.data() + c * d, d);
            cluster_dists[c] = {dist, c};
        }
        std::partial_sort(
                cluster_dists.begin(),
                cluster_dists.begin() + np,
                cluster_dists.end());

        // Update query counts
        for (int i = 0; i < np; i++) {
            cluster_query_counts[cluster_dists[i].second]++;
        }

        // Search probed clusters
        std::vector<std::pair<float, idx_t>> candidates;
        for (int i = 0; i < np; i++) {
            int c = cluster_dists[i].second;

            std::vector<float> cluster_vecs;
            std::vector<idx_t> cluster_ids;

            if (in_memory[c]) {
                // Use memory-resident data
                cluster_vecs = memory_clusters[c];
                cluster_ids = memory_ids[c];
            } else {
                // Read from disk/build_clusters
                const_cast<IndexNeuroHierarchicalDisk*>(this)
                        ->read_cluster_from_disk(c, cluster_vecs, cluster_ids);
            }

            size_t nvecs = cluster_ids.size();
            for (size_t j = 0; j < nvecs; j++) {
                const float* vec = cluster_vecs.data() + j * d;
                float dist = fvec_L2sqr(query, vec, d);
                candidates.push_back({dist, cluster_ids[j]});
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
