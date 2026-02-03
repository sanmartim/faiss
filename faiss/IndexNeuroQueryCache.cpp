/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroQueryCache.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <functional>

namespace faiss {

IndexNeuroQueryCache::IndexNeuroQueryCache(
        Index* sub_index,
        int cache_size,
        float similarity_threshold)
        : IndexNeuro(sub_index, false),
          sub_index(sub_index),
          cache_size(cache_size),
          similarity_threshold(similarity_threshold) {
    if (sub_index) {
        this->d = sub_index->d;
        this->ntotal = sub_index->ntotal;
        this->is_trained = sub_index->is_trained;
    }
}

void IndexNeuroQueryCache::train(idx_t n, const float* x) {
    if (sub_index) {
        sub_index->train(n, x);
        is_trained = sub_index->is_trained;
    }
}

void IndexNeuroQueryCache::add(idx_t n, const float* x) {
    if (sub_index) {
        sub_index->add(n, x);
        ntotal = sub_index->ntotal;
    }
    // Invalidate cache when data changes
    clear_cache();
}

void IndexNeuroQueryCache::reset() {
    if (sub_index) {
        sub_index->reset();
        ntotal = 0;
    }
    clear_cache();
}

void IndexNeuroQueryCache::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(sub_index);
    sub_index->reconstruct(key, recons);
}

uint64_t IndexNeuroQueryCache::compute_cache_key(const float* query) const {
    // Simple hash based on quantized query values
    // Use first few dimensions and quantize to reduce collision
    uint64_t hash = 0;
    const uint64_t prime = 31;
    int dims_to_use = std::min(d, 16);

    for (int i = 0; i < dims_to_use; i++) {
        // Quantize to integer bins
        int32_t quantized = static_cast<int32_t>(query[i] * 1000.0f);
        hash = hash * prime + static_cast<uint64_t>(quantized);
    }

    return hash;
}

bool IndexNeuroQueryCache::find_in_cache(
        const float* query,
        int k,
        float* distances,
        idx_t* labels) const {
    std::lock_guard<std::mutex> lock(cache_mutex);

    uint64_t key = compute_cache_key(query);
    auto it = cache_index.find(key);

    if (it == cache_index.end()) {
        return false;
    }

    // Check if cached query is similar enough
    const CacheEntry& entry = *(it->second);

    // Must have same k
    if (entry.k != k) {
        return false;
    }

    // Check L2 distance between query and cached query
    float dist = fvec_L2sqr(query, entry.query.data(), d);
    if (dist > similarity_threshold) {
        return false;
    }

    // Cache hit! Copy results
    std::copy(entry.distances.begin(), entry.distances.end(), distances);
    std::copy(entry.labels.begin(), entry.labels.end(), labels);

    // Move to front (most recently used)
    cache_list.splice(cache_list.begin(), cache_list, it->second);

    return true;
}

void IndexNeuroQueryCache::add_to_cache(
        const float* query,
        int k,
        const float* distances,
        const idx_t* labels) const {
    std::lock_guard<std::mutex> lock(cache_mutex);

    uint64_t key = compute_cache_key(query);

    // If key exists, update it
    auto it = cache_index.find(key);
    if (it != cache_index.end()) {
        // Update existing entry
        CacheEntry& entry = *(it->second);
        entry.query.assign(query, query + d);
        entry.distances.assign(distances, distances + k);
        entry.labels.assign(labels, labels + k);
        entry.k = k;

        // Move to front
        cache_list.splice(cache_list.begin(), cache_list, it->second);
        return;
    }

    // Evict if at capacity
    while (static_cast<int>(cache_list.size()) >= cache_size) {
        // Remove least recently used (back of list)
        auto& lru_entry = cache_list.back();
        uint64_t lru_key = compute_cache_key(lru_entry.query.data());
        cache_index.erase(lru_key);
        cache_list.pop_back();
    }

    // Add new entry at front
    CacheEntry new_entry;
    new_entry.query.assign(query, query + d);
    new_entry.distances.assign(distances, distances + k);
    new_entry.labels.assign(labels, labels + k);
    new_entry.k = k;

    cache_list.push_front(std::move(new_entry));
    cache_index[key] = cache_list.begin();
}

void IndexNeuroQueryCache::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(sub_index);

    float thresh = similarity_threshold;
    auto cp = dynamic_cast<const NeuroQueryCacheParams*>(params);
    if (cp && cp->similarity_threshold >= 0) {
        thresh = cp->similarity_threshold;
    }

    // Process each query
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;
        float* out_distances = distances + q * k;
        idx_t* out_labels = labels + q * k;

        // Try cache first
        if (find_in_cache(query, k, out_distances, out_labels)) {
            cache_hits++;
            continue;
        }

        cache_misses++;

        // Cache miss: search sub_index
        sub_index->search(1, query, k, out_distances, out_labels, params);

        // Add to cache
        add_to_cache(query, k, out_distances, out_labels);
    }
}

float IndexNeuroQueryCache::get_hit_rate() const {
    int64_t total = cache_hits + cache_misses;
    if (total == 0) {
        return 0.0f;
    }
    return static_cast<float>(cache_hits) / static_cast<float>(total);
}

void IndexNeuroQueryCache::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache_list.clear();
    cache_index.clear();
    cache_hits = 0;
    cache_misses = 0;
}

void IndexNeuroQueryCache::get_stats(int64_t& hits, int64_t& misses) const {
    hits = cache_hits;
    misses = cache_misses;
}

} // namespace faiss
