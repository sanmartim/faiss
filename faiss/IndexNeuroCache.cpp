/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroCache.h>
#include <faiss/impl/FaissAssert.h>

#include <cmath>
#include <functional>

namespace faiss {

IndexNeuroCache::IndexNeuroCache(
        Index* sub_index,
        int cache_size,
        float grid_step,
        bool own_fields)
        : Index(sub_index->d, sub_index->metric_type),
          sub_index(sub_index),
          own_fields(own_fields),
          cache_size(cache_size),
          grid_step(grid_step) {
    ntotal = sub_index->ntotal;
    is_trained = sub_index->is_trained;
}

IndexNeuroCache::~IndexNeuroCache() {
    if (own_fields) {
        delete sub_index;
    }
}

void IndexNeuroCache::add(idx_t n, const float* x) {
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;
    clear_cache(); // invalidate on data change
}

void IndexNeuroCache::reset() {
    sub_index->reset();
    ntotal = 0;
    clear_cache();
}

void IndexNeuroCache::reconstruct(idx_t key, float* recons) const {
    sub_index->reconstruct(key, recons);
}

size_t IndexNeuroCache::hash_query(const float* x, idx_t k) const {
    // Discretize query to grid, then hash
    size_t h = std::hash<idx_t>()(k);
    float inv_step = 1.0f / grid_step;
    for (int j = 0; j < d; j++) {
        int32_t discretized = (int32_t)std::round(x[j] * inv_step);
        h ^= std::hash<int32_t>()(discretized) + 0x9e3779b9 + (h << 6) +
                (h >> 2);
    }
    return h;
}

void IndexNeuroCache::clear_cache() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    lru_list.clear();
    cache_map.clear();
    cache_hits = 0;
    cache_misses = 0;
}

float IndexNeuroCache::hit_rate() const {
    int64_t total = cache_hits + cache_misses;
    return total > 0 ? (float)cache_hits / total : 0.0f;
}

void IndexNeuroCache::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is null");

    // Process each query: check cache, then search if miss
    // Batch uncached queries for efficiency
    std::vector<idx_t> uncached_indices;
    std::vector<float> uncached_queries;

    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;
        size_t key = hash_query(query, k);

        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache_map.find(key);
        if (it != cache_map.end()) {
            // Cache hit - copy results
            auto& entry = it->second->second;
            idx_t copy_k = std::min(k, (idx_t)entry.labels.size());
            for (idx_t i = 0; i < copy_k; i++) {
                distances[q * k + i] = entry.distances[i];
                labels[q * k + i] = entry.labels[i];
            }
            for (idx_t i = copy_k; i < k; i++) {
                distances[q * k + i] =
                        std::numeric_limits<float>::max();
                labels[q * k + i] = -1;
            }
            // Move to front (most recently used)
            lru_list.splice(lru_list.begin(), lru_list, it->second);
            cache_hits++;
        } else {
            cache_misses++;
            uncached_indices.push_back(q);
            uncached_queries.insert(
                    uncached_queries.end(), query, query + d);
        }
    }

    // Search uncached queries in batch
    if (!uncached_indices.empty()) {
        idx_t n_uncached = uncached_indices.size();
        std::vector<float> uc_distances(n_uncached * k);
        std::vector<idx_t> uc_labels(n_uncached * k);

        sub_index->search(
                n_uncached,
                uncached_queries.data(),
                k,
                uc_distances.data(),
                uc_labels.data(),
                params);

        // Copy results and populate cache
        for (idx_t ui = 0; ui < n_uncached; ui++) {
            idx_t q = uncached_indices[ui];
            for (idx_t i = 0; i < k; i++) {
                distances[q * k + i] = uc_distances[ui * k + i];
                labels[q * k + i] = uc_labels[ui * k + i];
            }

            // Add to cache
            const float* query = x + q * d;
            size_t key = hash_query(query, k);

            std::lock_guard<std::mutex> lock(cache_mutex);

            // Evict LRU if at capacity
            if ((int)cache_map.size() >= cache_size) {
                auto& back = lru_list.back();
                cache_map.erase(back.first);
                lru_list.pop_back();
            }

            CacheEntry entry;
            entry.distances.assign(
                    uc_distances.data() + ui * k,
                    uc_distances.data() + (ui + 1) * k);
            entry.labels.assign(
                    uc_labels.data() + ui * k,
                    uc_labels.data() + (ui + 1) * k);

            lru_list.push_front({key, std::move(entry)});
            cache_map[key] = lru_list.begin();
        }
    }
}

} // namespace faiss
