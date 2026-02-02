/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace faiss {

/** MR-02: Cache decorator for repeated similar queries.
 *
 * Wraps any Index and caches search results keyed by discretized
 * query vectors (grid hashing). Cache hits return stored results
 * instantly. Cache is invalidated on add() or reset().
 *
 * Thread-safe via mutex on cache operations.
 */
struct IndexNeuroCache : Index {
    Index* sub_index = nullptr;
    bool own_fields = false;

    int cache_size = 1024;     ///< max number of cached entries (LRU)
    float grid_step = 0.1f;    ///< discretization step for query hashing

    mutable int64_t cache_hits = 0;
    mutable int64_t cache_misses = 0;

    IndexNeuroCache() = default;

    /** Construct wrapping any Index.
     * @param sub_index   the index to wrap
     * @param cache_size  max LRU cache entries
     * @param grid_step   grid discretization step
     * @param own_fields  take ownership
     */
    IndexNeuroCache(
            Index* sub_index,
            int cache_size = 1024,
            float grid_step = 0.1f,
            bool own_fields = false);

    void add(idx_t n, const float* x) override;
    void reset() override;
    void reconstruct(idx_t key, float* recons) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    /// Clear the cache without affecting the underlying index
    void clear_cache() const;

    /// Return cache hit rate (0..1)
    float hit_rate() const;

    ~IndexNeuroCache() override;

private:
    /// Hash a query vector to a cache key
    size_t hash_query(const float* x, idx_t k) const;

    struct CacheEntry {
        std::vector<float> distances;
        std::vector<idx_t> labels;
    };

    /// LRU cache: list front = most recently used
    mutable std::list<std::pair<size_t, CacheEntry>> lru_list;
    mutable std::unordered_map<
            size_t,
            std::list<std::pair<size_t, CacheEntry>>::iterator>
            cache_map;
    mutable std::mutex cache_mutex;
};

} // namespace faiss
