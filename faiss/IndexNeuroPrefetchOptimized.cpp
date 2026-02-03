/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexNeuroPrefetchOptimized.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace faiss {

IndexNeuroPrefetchOptimized::IndexNeuroPrefetchOptimized(
        Index* sub_index,
        int hilbert_bits)
        : IndexNeuro(sub_index, false),
          sub_index(sub_index),
          hilbert_bits(hilbert_bits) {
    if (sub_index) {
        this->d = sub_index->d;
        this->is_trained = sub_index->is_trained;
    }
}

void IndexNeuroPrefetchOptimized::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->train(n, x);
    is_trained = sub_index->is_trained;
}

void IndexNeuroPrefetchOptimized::hilbert_rotate(int n, int* x, int* y, int rx, int ry) const {
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        // Swap x and y
        int t = *x;
        *x = *y;
        *y = t;
    }
}

uint64_t IndexNeuroPrefetchOptimized::xy_to_hilbert(int n, int x, int y) const {
    uint64_t result = 0;
    for (int s = n / 2; s > 0; s /= 2) {
        int rx = (x & s) > 0 ? 1 : 0;
        int ry = (y & s) > 0 ? 1 : 0;
        result += static_cast<uint64_t>(s) * s * ((3 * rx) ^ ry);
        hilbert_rotate(s, &x, &y, rx, ry);
    }
    return result;
}

uint64_t IndexNeuroPrefetchOptimized::compute_hilbert_index(const float* vec) const {
    // Use first two dimensions for Hilbert ordering
    // Normalize to [0, 2^hilbert_bits - 1]
    int n = 1 << hilbert_bits;

    // Find min/max from all stored vectors for normalization
    // For simplicity, assume data is roughly in [-3, 3] (standard normal)
    float min_val = -3.0f;
    float max_val = 3.0f;
    float range = max_val - min_val;

    int x = static_cast<int>((vec[0] - min_val) / range * (n - 1));
    int y = d >= 2 ? static_cast<int>((vec[1] - min_val) / range * (n - 1)) : 0;

    // Clamp to valid range
    x = std::max(0, std::min(n - 1, x));
    y = std::max(0, std::min(n - 1, y));

    return xy_to_hilbert(n, x, y);
}

void IndexNeuroPrefetchOptimized::optimize_layout() {
    if (ntotal == 0 || ordered_vectors.empty()) {
        return;
    }

    // Compute Hilbert index for each vector
    std::vector<std::pair<uint64_t, idx_t>> hilbert_order(ntotal);
    for (idx_t i = 0; i < ntotal; i++) {
        const float* vec = ordered_vectors.data() + i * d;
        hilbert_order[i] = {compute_hilbert_index(vec), i};
    }

    // Sort by Hilbert index
    std::sort(hilbert_order.begin(), hilbert_order.end());

    // Create reordered vectors
    std::vector<float> reordered(ntotal * d);
    new_to_orig.resize(ntotal);
    orig_to_new.resize(ntotal);

    for (idx_t new_pos = 0; new_pos < ntotal; new_pos++) {
        idx_t orig_pos = hilbert_order[new_pos].second;
        new_to_orig[new_pos] = orig_pos;
        orig_to_new[orig_pos] = new_pos;

        std::copy(
                ordered_vectors.data() + orig_pos * d,
                ordered_vectors.data() + (orig_pos + 1) * d,
                reordered.data() + new_pos * d);
    }

    ordered_vectors = std::move(reordered);
    is_reordered = true;
}

void IndexNeuroPrefetchOptimized::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    // Store vectors locally
    size_t old_size = ordered_vectors.size();
    ordered_vectors.resize(old_size + n * d);
    std::copy(x, x + n * d, ordered_vectors.data() + old_size);

    // Add to sub_index
    sub_index->add(n, x);
    ntotal = sub_index->ntotal;

    // Invalidate ordering (would need to re-optimize)
    is_reordered = false;
}

void IndexNeuroPrefetchOptimized::reset() {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");
    sub_index->reset();
    ntotal = 0;
    ordered_vectors.clear();
    new_to_orig.clear();
    orig_to_new.clear();
    is_reordered = false;
}

void IndexNeuroPrefetchOptimized::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT(key >= 0 && key < ntotal);
    if (!ordered_vectors.empty()) {
        std::copy(
                ordered_vectors.data() + key * d,
                ordered_vectors.data() + (key + 1) * d,
                recons);
    } else if (sub_index) {
        sub_index->reconstruct(key, recons);
    }
}

void IndexNeuroPrefetchOptimized::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(sub_index, "sub_index is not set");

    if (!is_reordered || ordered_vectors.empty()) {
        // Fall back to sub_index search
        sub_index->search(n, x, k, distances, labels, params);
        return;
    }

    // Search using optimized layout
#pragma omp parallel for
    for (idx_t q = 0; q < n; q++) {
        const float* query = x + q * d;

        // Brute force search on reordered vectors (cache-friendly)
        std::vector<std::pair<float, idx_t>> scored(ntotal);
        for (idx_t i = 0; i < ntotal; i++) {
            const float* vec = ordered_vectors.data() + i * d;
            float dist = fvec_L2sqr(query, vec, d);
            // Map back to original IDs
            idx_t orig_id = new_to_orig[i];
            scored[i] = {dist, orig_id};
        }

        // Partial sort for top-k
        size_t actual_k = std::min(static_cast<size_t>(k), static_cast<size_t>(ntotal));
        std::partial_sort(
                scored.begin(),
                scored.begin() + actual_k,
                scored.end());

        // Output results
        for (size_t i = 0; i < actual_k; i++) {
            distances[q * k + i] = scored[i].first;
            labels[q * k + i] = scored[i].second;
        }
        for (size_t i = actual_k; i < static_cast<size_t>(k); i++) {
            distances[q * k + i] = std::numeric_limits<float>::max();
            labels[q * k + i] = -1;
        }
    }
}

} // namespace faiss
