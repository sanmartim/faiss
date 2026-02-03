/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroCompressedDisk.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroCompressedDisk::IndexNeuroCompressedDisk(int d, int nlist)
        : IndexNeuro(nullptr, false), nlist(nlist) {
    this->d = d;
    cluster_codes.resize(nlist);
    cluster_ids.resize(nlist);
    decompression_cache.resize(nlist);
    cache_valid.resize(nlist, false);
    vmin.resize(d, std::numeric_limits<float>::max());
    vmax.resize(d, std::numeric_limits<float>::lowest());
    is_trained = false;
}

int IndexNeuroCompressedDisk::find_cluster(const float* vec) const {
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

void IndexNeuroCompressedDisk::encode_8bit(const float* vec, uint8_t* code) const {
    for (int i = 0; i < d; i++) {
        float range = vmax[i] - vmin[i];
        if (range < 1e-10f) {
            code[i] = 128;
        } else {
            float normalized = (vec[i] - vmin[i]) / range;
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            code[i] = static_cast<uint8_t>(normalized * 255.0f);
        }
    }
}

void IndexNeuroCompressedDisk::decode_8bit(const uint8_t* code, float* vec) const {
    for (int i = 0; i < d; i++) {
        float normalized = static_cast<float>(code[i]) / 255.0f;
        vec[i] = vmin[i] + normalized * (vmax[i] - vmin[i]);
    }
}

void IndexNeuroCompressedDisk::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(n >= nlist, "need at least nlist vectors");

    // Compute min/max for quantization
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            float val = x[i * d + j];
            vmin[j] = std::min(vmin[j], val);
            vmax[j] = std::max(vmax[j], val);
        }
    }

    // Add small margin to avoid edge cases
    for (int j = 0; j < d; j++) {
        float range = vmax[j] - vmin[j];
        vmin[j] -= range * 0.01f;
        vmax[j] += range * 0.01f;
    }

    // Cluster
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

void IndexNeuroCompressedDisk::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(is_trained, "index must be trained");

    // Update min/max if needed
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            float val = x[i * d + j];
            if (val < vmin[j]) {
                vmin[j] = val - std::abs(val) * 0.01f;
            }
            if (val > vmax[j]) {
                vmax[j] = val + std::abs(val) * 0.01f;
            }
        }
    }

    std::vector<uint8_t> code(d);
    for (idx_t i = 0; i < n; i++) {
        const float* vec = x + i * d;
        int cluster = find_cluster(vec);
        idx_t vec_id = ntotal + i;

        // Encode and store
        encode_8bit(vec, code.data());
        cluster_codes[cluster].insert(
                cluster_codes[cluster].end(),
                code.begin(),
                code.end());
        cluster_ids[cluster].push_back(vec_id);

        // Invalidate cache
        cache_valid[cluster] = false;
    }

    // Store original vectors
    size_t old_size = orig_vectors.size();
    orig_vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, orig_vectors.data() + old_size);

    ntotal += n;
}

void IndexNeuroCompressedDisk::reset() {
    for (auto& v : cluster_codes) v.clear();
    for (auto& v : cluster_ids) v.clear();
    for (auto& v : decompression_cache) v.clear();
    std::fill(cache_valid.begin(), cache_valid.end(), false);
    orig_vectors.clear();
    lru_order.clear();
    ntotal = 0;
}

void IndexNeuroCompressedDisk::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    std::copy(
            orig_vectors.data() + key * d,
            orig_vectors.data() + (key + 1) * d,
            recons);
}

void IndexNeuroCompressedDisk::touch_cluster(int cluster_id) const {
    // Remove from current position
    auto it = std::find(lru_order.begin(), lru_order.end(), cluster_id);
    if (it != lru_order.end()) {
        lru_order.erase(it);
    }

    // Add to front (most recent)
    lru_order.insert(lru_order.begin(), cluster_id);

    // Evict if over capacity
    while (static_cast<int>(lru_order.size()) > max_cached_clusters) {
        int evict_id = lru_order.back();
        lru_order.pop_back();
        cache_valid[evict_id] = false;
        decompression_cache[evict_id].clear();
    }
}

const std::vector<float>& IndexNeuroCompressedDisk::get_decompressed(
        int cluster_id) const {
    if (cache_valid[cluster_id]) {
        touch_cluster(cluster_id);
        return decompression_cache[cluster_id];
    }

    // Decompress
    size_t nvecs = cluster_ids[cluster_id].size();
    decompression_cache[cluster_id].resize(nvecs * d);

    for (size_t i = 0; i < nvecs; i++) {
        decode_8bit(
                cluster_codes[cluster_id].data() + i * d,
                decompression_cache[cluster_id].data() + i * d);
    }

    cache_valid[cluster_id] = true;
    touch_cluster(cluster_id);

    return decompression_cache[cluster_id];
}

float IndexNeuroCompressedDisk::get_compression_ratio() const {
    if (ntotal == 0) {
        return 1.0f;
    }

    size_t uncompressed = ntotal * d * sizeof(float);
    size_t compressed = 0;
    for (const auto& codes : cluster_codes) {
        compressed += codes.size();
    }

    if (compressed == 0) {
        return 1.0f;
    }

    return static_cast<float>(uncompressed) / static_cast<float>(compressed);
}

void IndexNeuroCompressedDisk::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(ntotal > 0);

    int np = nprobe;
    auto cp = dynamic_cast<const NeuroCompressedDiskParams*>(params);
    if (cp && cp->nprobe > 0) {
        np = cp->nprobe;
    }
    np = std::min(np, nlist);

    // Note: Not parallelized due to shared decompression cache
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

        // Search probed clusters
        std::vector<std::pair<float, idx_t>> candidates;
        for (int i = 0; i < np; i++) {
            int c = cluster_dists[i].second;
            const auto& vecs = get_decompressed(c);
            size_t nvecs = cluster_ids[c].size();

            for (size_t j = 0; j < nvecs; j++) {
                const float* vec = vecs.data() + j * d;
                float dist = fvec_L2sqr(query, vec, d);
                candidates.push_back({dist, cluster_ids[c][j]});
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
